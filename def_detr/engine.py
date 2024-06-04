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

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, scaler: GradScaler, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # prefetcher = data_prefetcher(data_loader, device, prefetch=False)
    # inputs, targets = prefetcher.next()
    # # batch_data = prefetcher.next()


    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        # inputs = [
        #     batch_data_ii["image"].to(device) for batch_data_i in batch_data for batch_data_ii in batch_data_i
        # ]
        # targets = [
        #     dict(
        #         label=batch_data_ii["label"].to(device),
        #         boxes=batch_data_ii["boxes"].to(device),
        #     )
        #     for batch_data_i in batch_data
        #     for batch_data_ii in batch_data_i
        # ]
        # inputs = torch.stack([batch_data_ii["image"].to(device, non_blocking=True) for batch_data_i in batch for batch_data_ii in batch_data_i])
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

        # targets = [
        #     dict(
        #         # labels=(1-batch_data_ii["label"]).to(device, non_blocking=True),
        #         labels=batch_data_ii["label"].to(device, non_blocking=True),
        #         boxes=box_utils.convert_box_mode(batch_data_ii["boxes"], dst_mode="cccwhd").to(device, non_blocking=True),
        #     )
        #     for batch_data_i in batch
        #     for batch_data_ii in batch_data_i
        # ]
        # torch.save(samples, "./samples.pt")
        # torch.save(targets, "./targets.pt")
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
        # losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # inputs, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Training averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Eval:'

    coco_metric = COCOMetric(classes=["Bifurcation"], iou_list=[0.1], max_detection=[model.num_queries])
    test_values = []
    out_boxes_all = []
    out_cls_all = []
    out_scores_all = []
    gt_boxes_all = []
    gt_cls_all = []
    val_sizes_all = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        # samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # inputs = torch.stack([batch_data_i["image"].to(device, non_blocking=True) for batch_data_i in batch])
        # targets = [
        #     dict(
        #         labels=batch_data_i["label"].to(device, non_blocking=True),
        #         boxes=batch_data_i["boxes"].to(device, non_blocking=True),
        #     )
        #     for batch_data_i in batch
        #     # for batch_data_ii in batch_data_i
        # ]
        inputs = []
        targets = []
        sizes = []
        for batch_data_i in batch:
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
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # print("pred_boxes:", outputs['pred_boxes'].squeeze(0).cpu().detach().numpy())
        # print("pred_logits:", outputs['pred_logits'].squeeze(0).sigmoid())
        # print("modified pred_boxes 1:", box_utils.convert_box_mode((outputs['pred_boxes'].squeeze(0) * sizes).cpu().detach().numpy(), src_mode="cccwhd"))
        # print("modified pred_boxes 2:", (box_utils.convert_box_mode((outputs['pred_boxes'].squeeze(0)), src_mode="cccwhd") * sizes).cpu().detach().numpy())
        # print("shapes:", sizes.shape, (outputs['pred_boxes'].squeeze(0).shape))
        # print("gt_boxes:", targets[0]['boxes'].cpu().detach().numpy())
        # print("gt_labels:", targets[0]['labels'].cpu().detach().numpy())
        # print('nb gt:', len(targets[0]['boxes']))

        out_boxes_all.append((box_utils.convert_box_mode((outputs['pred_boxes'].squeeze(0)), src_mode="cccwhd") * sizes).cpu().detach().numpy())
        out_cls_all.append(outputs['pred_logits'].squeeze(0).sigmoid().max(-1)[1].cpu().detach().numpy())
        out_scores_all.append(outputs['pred_logits'].squeeze(0).sigmoid().max(-1)[0].cpu().detach().numpy())
        gt_boxes_all.append(targets[0]['gt_boxes'].cpu().detach().numpy())
        gt_cls_all.append(targets[0]['labels'].cpu().detach().numpy())

        # for data_i in 

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
    # test_values.append(val_batch_metric_dict)

        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # if coco_evaluator is not None:
        #     coco_evaluator.update(res)
    # all_test_values = defaultdict(list)
    # for test_value in test_values:
    #     for key in test_value:
    #         all_test_values[key].append(test_value[key])

    # avg_test_values = {}
    # for key in all_test_values:
    #     avg_test_values[key] = np.mean(all_test_values[key])

    utils.csv_writer(output_dir / "avg_results.csv", val_batch_metric_dict)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Eval averaged stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    # if coco_evaluator is not None:
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if coco_evaluator is not None:
    #     if 'bbox' in postprocessors.keys():
    #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    #     if 'segm' in postprocessors.keys():
    #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # return stats, coco_evaluator
    return val_batch_metric_dict

