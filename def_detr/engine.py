# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable

import torch
from torch.cuda.amp import GradScaler, autocast
import util.misc as utils
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.data import box_utils

import numpy as np
import pickle

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, scaler: GradScaler, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        inputs = []
        targets = []
        for batch_data_i in batch:
            for batch_data_ii in batch_data_i:
                image = batch_data_ii["image"].to(device, non_blocking=True)
                h, w, d = image.shape[-3:]
                labels=batch_data_ii["label"].to(device, non_blocking=True)
                boxes=box_utils.convert_box_mode(batch_data_ii["boxes"].to(device, non_blocking=True), dst_mode="cccwhd")
                boxes = boxes / torch.tensor([w,h,d,w,h,d], dtype=torch.float32, device=device)
                inputs.append(image)
                targets.append(dict(labels=labels, boxes=boxes))

        inputs = torch.stack(inputs)

        with autocast(enabled=scaler._enabled):
            outputs = model(inputs)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # inputs, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Training averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, output_dir, save_preds=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    coco_metric = COCOMetric(classes=["Bifurcation"], iou_list=[0.1, 0.25, 0.5], max_detection=[model.num_queries])
    test_values = []
    out_boxes_all = []
    out_cls_all = []
    out_scores_all = []
    gt_boxes_all = []
    gt_cls_all = []
    val_sizes_all = []
    inputs_all = []
    data_all = []
    for batch in metric_logger.log_every(data_loader, 10, header):

        inputs = []
        targets = []
        sizes = []
        for batch_data_i in batch:
            if save_preds:
                data_all.append(batch_data_i)
            image = batch_data_i["image"].to(device, non_blocking=True)
            h, w, d = image.shape[-3:]
            inputs.append(image)
            tgt_boxes=box_utils.convert_box_mode(batch_data_i["boxes"].to(device, non_blocking=True), dst_mode="cccwhd")
            tgt_boxes = tgt_boxes / torch.tensor([w,h,d,w,h,d], dtype=torch.float32, device=device)
            targets.append(dict(labels=batch_data_i["label"].to(device, non_blocking=True), boxes=tgt_boxes, gt_boxes=batch_data_i["boxes"].to(device, non_blocking=True)))
            sizes.append(torch.tensor([w,h,d,w,h,d], dtype=torch.float32, device=device))

        inputs = torch.stack(inputs)
        sizes = torch.stack(sizes)

        outputs = model(inputs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        out_boxes_all.append((box_utils.convert_box_mode((outputs['pred_boxes'].squeeze(0)), src_mode="cccwhd") * sizes).cpu().detach().numpy())
        out_scores = outputs['pred_logits'].squeeze(0).sigmoid()[:,0].cpu().detach().numpy()
        out_cls = np.zeros_like(out_scores)
        out_scores_all.append(out_scores)
        out_cls_all.append(out_cls)
        gt_boxes_all.append(targets[0]['gt_boxes'].cpu().detach().numpy())
        gt_cls_all.append(targets[0]['labels'].cpu().detach().numpy())
        inputs_all.append(inputs.squeeze(0).cpu().detach().numpy())

    if save_preds:
        np.save(output_dir / "pred_boxes.npy", np.array(out_boxes_all))
        np.save(output_dir / "pred_classes.npy", np.array(out_cls_all))
        np.save(output_dir / "pred_scores.npy", np.array(out_scores_all))

        for i in range(len(gt_boxes_all)):
            np.save(output_dir / f"gt_boxes_{i}.npy", np.array(gt_boxes_all[i]))
        for i in range(len(gt_cls_all)):
            np.save(output_dir / f"gt_cls_all_{i}.npy", np.array(gt_cls_all[i]))
        np.save(output_dir / "inputs.npy", np.array(inputs_all))

        with open(output_dir / "val_data_all.pkl", "wb") as handle:
            pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

    results_metric = matching_batch(
            iou_fn=box_utils.box_iou,
            iou_thresholds=coco_metric.iou_thresholds,
            pred_boxes=out_boxes_all,
            pred_classes=out_cls_all,
            pred_scores=out_scores_all,
            gt_boxes=gt_boxes_all,
            gt_classes=gt_cls_all,
        )
    
    val_batch_metric_dict = coco_metric(results_metric)[0]


    utils.csv_writer(output_dir / "avg_results.csv", val_batch_metric_dict)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Eval averaged stats:", metric_logger)

    return val_batch_metric_dict

