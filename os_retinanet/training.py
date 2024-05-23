import argparse
import gc
import logging
import os
import sys
import time
import random
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import monai
from monai.apps.detection.metrics.matching import matching_batch
from monai.data import box_utils
from monai.utils import set_determinism
from monai.networks.nets import resnet, resnet50
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.metrics.coco import COCOMetric
from utils import GradualWarmupScheduler, visualize_one_xy_slice_in_3d_image, make_dirs
from BifDet24DataModule import BifDet2024DataModule
from dotenv import load_dotenv, find_dotenv
import datetime

now = datetime.datetime.now()
load_dotenv(find_dotenv("config.env"))

def main():
    # parser inputs
    args = parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    params_dict = {
        'PATCH_SIZE': (args.patch_size,) * 3,
        'VAL_PATCH_SIZE': (args.val_patch_size,) * 3,
        'BATCH_SIZE': args.batch_size,
        'MAX_CARDINALITY': args.max_cardinality,
        'NUM_WORKERS': args.num_workers,
        'PIXEL_VALUE_MIN': args.pixel_value_min,
        'PIXEL_VALUE_MAX': args.pixel_value_max,
        'PIXEL_NORM_MIN': args.pixel_norm_min,
        'PIXEL_NORM_MAX': args.pixel_norm_max,
        'VOXEL_SIZE': (1,) * 3,
        'NUM_EPOCHS': args.max_epochs,
        'CACHE_DS': args.cache_ds,
        'AVAILABLE_GPUs': torch.cuda.device_count(),
        'DEVICE_NO': int(os.getenv('DEVICE_NO'))
    }

    # output file
    experiment_name = f"bifdet_{now.day}_{now.hour}_{now.minute}_{args.annot_fname.split('.')[0]}"
    exp_path = os.path.join(
        f"{os.getenv('OUTPUT_PATH')}/",
        f"{experiment_name}_b{args.batch_size}_p{args.patch_size}_a{args.detection_per_img}_nms{args.nms_thresh}"
        f"_bs{args.score_thresh_glb}_lr{args.detector_lr}_e{args.max_epochs}_wcls{args.w_cls}_prtr{str(args.pre_trained)}/"
    )
    make_dirs(exp_path)
    # Set device based on GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    monai.config.print_config()
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    print(os.getenv("DATA_SRC"))
    print(args.annot_fname)

    # 1. Data loaders
    data_module = BifDet2024DataModule(
        train_parent_path=os.getenv("DATA_SRC"),
        batch_size=args.batch_size,
        bbox_path=args.annot_fname,
        params=params_dict,
        compute_dtype=args.compute_dtype,
    )
    data_module.prepare_data_monai()
    data_module.setup()


    # 2. build model
    # 1) build anchor generator
    # returned_layers: when target boxes are small, set it smaller
    # base_anchor_shapes: anchor shape for the most high-resolution output,
    #   when target boxes are small, set it smaller
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2 ** l for l in range(len([1, 2]) + 1)],
        base_anchor_shapes=[[28, 19, 37], [23, 16, 24], [20, 16, 20]]
    )

    # 2) build network
    conv1_t_size = [max(7, 2 * s + 1) for s in [2, 2, 2]]
    # print(conv1_t_size)
    backbone = resnet.ResNet(
        block=resnet.ResNetBottleneck,
        layers=[3, 4, 6, 3],
        block_inplanes=resnet.get_inplanes(),
        n_input_channels=1,
        conv1_t_stride=[2, 2, 2],
        conv1_t_size=conv1_t_size,
    )

    # check the model summary
    # input_size = (1, 256, 256, 256)  # Change according to your input size
    # Use torchsummary to print the model summary
    # summary(model=backbone.to(device), input_size=input_size, device="cuda" if torch.cuda.is_available() else "cpu")
    # sys.exit()

    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=3,
        pretrained_backbone=False,
        trainable_backbone_layers=None,
        returned_layers=[1, 2],
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    print(f"The number of anchors is {num_anchors}...")
    size_divisible = [s * 2 * 2 ** max([1, 2]) for s in feature_extractor.body.conv1.stride]
    net = torch.jit.script(
        RetinaNet(
            spatial_dims=3,
            num_classes=len([0]),
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
        )
    )

    if args.pre_trained:
        print("Load the model weights...", end="")
        model_weights = torch.jit.load(
            '/home/infres/akeshavarzi/data/outputs/luna16/pretraining4ATM22/trained_modelsmodel_best.pt')
        net = torch.jit.script(model_weights)
    print("Done!")

    # 3) build detector
    detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=args.debug).to(device)

    # set training components
    detector.set_atss_matcher(num_candidates=args.num_cands, center_in_gt=args.center_in_gt)
    detector.set_hard_negative_sampler(
        batch_size_per_image=args.batch_size_per_image,
        positive_fraction=args.positive_fraction,
        pool_size=args.pool_size,
        min_neg=args.min_neg,
    )
    detector.set_target_keys(box_key=args.box_key, label_key=args.label_key)

    # set validation components
    detector.set_box_selector_parameters(
        score_thresh=args.score_thresh_glb, # no box with scores less than score_thresh will be kept
        topk_candidates_per_level=args.topk_candidates_per_level, # max number of boxes to keep for each level
        nms_thresh=args.nms_thresh, # box overlapping threshold for NMS
        detections_per_img=args.detection_per_img, # max number of boxes to keep for each image
    )
    detector.set_sliding_window_inferer(
        roi_size=[args.patch_size, args.patch_size, args.patch_size],
        overlap=args.sw_inferer_overlap,
        sw_batch_size=args.sw_batch_size,
        mode=args.sw_mode,
        device=device,
    )

    # 4. Initialize training
    # initlize optimizer
    optimizer = torch.optim.SGD(
        params=detector.network.parameters(),
        lr=args.detector_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    after_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=args.a_scheduler_step_size,
        gamma=args.a_scheduler_gamma,
    )
    scheduler_warmup = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=args.wu_scheduler_multiplier,
        total_epoch=args.wu_scheduler_total_epoch,
        after_scheduler=after_scheduler)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    optimizer.zero_grad()
    optimizer.step()

    # initialize tensorboard writer
    tensorboard_writer = SummaryWriter(f"{exp_path}/tfevents/")

    # 5. train
    coco_metric = COCOMetric(classes=["Bifurcation"], iou_list=[0.1], max_detection=[250])

    best_val_epoch_metric = 0.0
    best_val_epoch = -1  # the epoch that gives best validation metrics

    epoch_len = len(data_module.train_set)

    for epoch in range(args.max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.max_epochs}")
        detector.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_box_reg_loss = 0
        step = 0
        start_time = time.time()
        scheduler_warmup.step()
        # Training
        for batch_data in data_module.train_dataloader():
            step += 1
            inputs = [
                batch_data_ii["image"].to(device) for batch_data_i in batch_data for batch_data_ii in batch_data_i
                # batch_data_i["image"].to(device) for batch_data_i in batch_data
            ]
            # print(inputs[0].shape)
            targets = [
                dict(
                    label=batch_data_ii["label"].to(device),
                    boxes=batch_data_ii["boxes"].to(device),
                    # label=batch_data_i["label"].to(device),
                    # boxes=batch_data_i["boxes"].to(device),
                )
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]
            # print(targets[0])
            for param in detector.network.parameters():
                param.grad = None

            if args.amp and (scaler is not None):
                with torch.cuda.amp.autocast():
                    outputs = detector(inputs, targets)
                    loss = args.w_cls * outputs[detector.cls_key] + outputs[detector.box_reg_key]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = detector(inputs, targets)
                loss = args.w_cls * outputs[detector.cls_key] + outputs[detector.box_reg_key]
                loss.backward()
                optimizer.step()

            # save to tensorboard
            epoch_loss += loss.detach().item()
            epoch_cls_loss += outputs[detector.cls_key].detach().item()
            epoch_box_reg_loss += outputs[detector.box_reg_key].detach().item()
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            tensorboard_writer.add_scalar("train_loss", loss.detach().item(), epoch_len * epoch + step)

        end_time = time.time()
        print(f"Training time: {end_time - start_time}s")
        del inputs, batch_data
        torch.cuda.empty_cache()
        gc.collect()

        # save to tensorboard
        epoch_loss /= step
        epoch_cls_loss /= step
        epoch_box_reg_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        tensorboard_writer.add_scalar("avg_train_loss", epoch_loss, epoch + 1)
        tensorboard_writer.add_scalar("avg_train_cls_loss", epoch_cls_loss, epoch + 1)
        tensorboard_writer.add_scalar("avg_train_box_reg_loss", epoch_box_reg_loss, epoch + 1)
        tensorboard_writer.add_scalar("train_lr", optimizer.param_groups[0]["lr"], epoch + 1)

        # save last trained model
        torch.jit.save(detector.network, exp_path + f"/model/" + "_last.pt")
        print("saved last model")
        # ------------- Validation for model selection -------------
        if (epoch + 1) % args.val_interval == 0 and epoch > 30:
            detector.eval()
            val_outputs_all = []
            val_targets_all = []
            start_time = time.time()
            with torch.no_grad():
                for val_data in data_module.train_dataloader():
                    # if all val_data_i["image"] smaller than args.val_patch_size, no need to use inferer
                    # otherwise, need inferer to handle large input images.
                    use_inferer = not all(
                        [val_data_ii["image"][0, ...].numel() < np.prod(args.val_patch_size)
                         for val_data_i in val_data for val_data_ii in val_data_i]
                    )
                    val_inputs = [val_data_ii.pop("image").to(device)
                                  for val_data_i in val_data for val_data_ii in val_data_i]

                    if args.amp:
                        with torch.cuda.amp.autocast():
                            val_outputs = detector(val_inputs, use_inferer=use_inferer)
                    else:
                        val_outputs = detector(val_inputs, use_inferer=use_inferer)
                    # save outputs for evaluation
                    val_outputs_all += val_outputs
                    val_targets_all += val_data
            end_time = time.time()
            print(f"Validation time: {end_time - start_time}s")

            # visualize an inference image and boxes to tensorboard
            idx = random.sample(range(0, len(val_data[0][0]["boxes"])), 2)
            draw_img_0 = visualize_one_xy_slice_in_3d_image(
                gt_boxes=val_data[0][0]["boxes"].cpu().detach().numpy(),
                image=val_inputs[0][0, ...].cpu().detach().numpy(),
                pred_boxes=val_outputs[0][detector.target_box_key].cpu().detach().numpy(),
                gt_box_index=idx[0],
            )
            tensorboard_writer.add_image("val_img_xy_0", draw_img_0.transpose([2, 1, 0]), epoch + 1)
            filename_0 = exp_path + f"/smpl_outputs/val_img_xy_0_{str(epoch+1)}.jpg"
            # print(f"The image shape is {draw_img_0.shape}")
            cv2.imwrite(filename_0, draw_img_0)

            draw_img_1 = visualize_one_xy_slice_in_3d_image(
                gt_boxes=val_data[0][0]["boxes"].cpu().detach().numpy(),
                image=val_inputs[0][0, ...].cpu().detach().numpy(),
                pred_boxes=val_outputs[0][detector.target_box_key].cpu().detach().numpy(),
                gt_box_index=idx[1],
            )
            tensorboard_writer.add_image("val_img_xy_1", draw_img_1.transpose([2, 1, 0]), epoch + 1)
            filename_1 = exp_path + f"/smpl_outputs/val_img_xy_1_{str(epoch+1)}.jpg"
            cv2.imwrite(filename_1, draw_img_1)


            # compute metrics
            del val_inputs
            torch.cuda.empty_cache()
            results_metric = matching_batch(
                iou_fn=box_utils.box_iou,
                iou_thresholds=coco_metric.iou_thresholds,
                pred_boxes=[
                    val_data_i[detector.target_box_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                pred_classes=[
                    val_data_i[detector.target_label_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                pred_scores=[
                    val_data_i[detector.pred_score_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                gt_boxes=[val_data_i[0][detector.target_box_key].cpu().detach().numpy() for val_data_i in val_targets_all],
                gt_classes=[
                    val_data_i[0][detector.target_label_key].cpu().detach().numpy() for val_data_i in val_targets_all
                ],
            )
            val_epoch_metric_dict = coco_metric(results_metric)[0]
            # print(val_epoch_metric_dict)

            # write to tensorboard event
            for k in val_epoch_metric_dict.keys():
                tensorboard_writer.add_scalar("val_" + k, val_epoch_metric_dict[k], epoch + 1)
            val_epoch_metric = val_epoch_metric_dict.values()
            val_epoch_metric = sum(val_epoch_metric) / len(val_epoch_metric)
            tensorboard_writer.add_scalar("val_metric", val_epoch_metric, epoch + 1)

            # save best trained model
            if val_epoch_metric > best_val_epoch_metric:
                best_val_epoch_metric = val_epoch_metric
                best_val_epoch = epoch + 1
                torch.jit.save(detector.network, exp_path + f"model/" + "_best.pt")
                print("saved new best metric model")
            print(
                "current epoch: {} current metric: {:.4f} "
                "best metric: {:.4f} at epoch {}".format(
                    epoch + 1, val_epoch_metric, best_val_epoch_metric, best_val_epoch
                )
            )
    print(f"train completed, best_metric: {best_val_epoch_metric:.4f} " f"at epoch: {best_val_epoch}")
    tensorboard_writer.close()


def parse_args():
    load_dotenv(find_dotenv("config.env"))
    parser = argparse.ArgumentParser(description="The airway bifurcation detection on BifDet24")

    # patch size - for now static
    parser.add_argument("--patch_size", type=int, default=int(os.getenv('PATCH_SIZE')), help="Minimum pixel value")
    parser.add_argument("--val_patch_size", type=int, default=int(os.getenv('PATCH_SIZE')), help="Minimum pixel value")

    # Voxel values - for now static
    parser.add_argument("--pixel_value_min", type=int, default=float(os.getenv('PIXEL_VALUE_MIN')), help="Minimum pixel value")
    parser.add_argument("--pixel_value_max", type=int, default=float(os.getenv('PIXEL_VALUE_MAX')), help="Maximum pixel value")
    parser.add_argument("--pixel_norm_min", type=float, default=float(os.getenv('PIXEL_NORM_MIN')), help="Minimum normalized pixel value")
    parser.add_argument("--pixel_norm_max", type=float, default=float(os.getenv('PIXEL_NORM_MAX')), help="Maximum normalized pixel value")
    parser.add_argument("--voxel_size", type=int, default=float(os.getenv('VOXEL_SIZE')), help="Voxel size")

    # Model Hyperparameters - static for the moment
    parser.add_argument("--model_name", type=str, default=os.getenv('MODEL_NAME'), help="Model name")
    parser.add_argument("--spatial_dims", type=int, default=int(os.getenv('SPATIAL_DIMS')), help="Spatial dimensions")
    parser.add_argument("--in_channels", type=int, default=int(os.getenv('IN_CHANNELS')), help="Input channels")
    parser.add_argument("--out_channels", type=int, default=int(os.getenv('OUT_CHANNELS')), help="Output channels")
    parser.add_argument("--n_layers", type=int, default=int(os.getenv('N_LAYERS')), help="Number of layers")
    parser.add_argument("--channels", type=int, default=int(os.getenv('CHANNELS')), help="Channels")
    parser.add_argument("--strides", type=int, default=int(os.getenv('STRIDES')), help="Strides")
    parser.add_argument("--num_res_units", type=int, default=int(os.getenv('NUM_RES_UNITS')), help="Number of residual units")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate -- not used in this model")
    parser.add_argument("--norm", type=str, default=os.getenv('NORM'), help="Normalization type")

    # Training Hyperparameters
    parser.add_argument("--cls_loss_func", type=str, default=os.getenv('CLS_LOSS_FUNC'), help="The loss function to use")
    parser.add_argument("--reg_loss_func", type=str, default=os.getenv('REG_LOSS_FUNC'), help="The loss function to use")
    parser.add_argument("--scheduler", type=str, default=os.getenv('SCHEDULER_TYPE'), help="Type of scheduler")
    parser.add_argument("--patience", type=int, default=int(os.getenv('PATIENCE')), help="The learning rate patience")
    parser.add_argument("--optimizer", type=str, default=os.getenv('OPTIMIZER'), help="The optimizer to use")
    parser.add_argument("--max_epochs", type=int, default=os.getenv('NUM_EPOCHS'), help="how many epochs?")
    parser.add_argument("--max_cardinality", type=int, default=os.getenv('MAX_CARDINALITY'), help="size of dataset")
    parser.add_argument("--batch_size", type=int, default=int(os.getenv('BATCH_SIZE')), help="size of dataset")

    # File paths
    parser.add_argument("--exp_path", type=str, default="", help="The experiment path")
    parser.add_argument("--model_weights_path", type=str, default="", help="The best model weight path")

    # Other options
    parser.add_argument("--val_interval", type=int, default=10, help="validation running interval")
    parser.add_argument("--amp", action="store_true", default=False, help="For improving the training")
    parser.add_argument("--cache_ds", action="store_true", default=False, help="Caching dataset or not")
    parser.add_argument("--num_workers", type=int, default=1, help="validation running interval")

    # torch options
    parser.add_argument("--annot_fname", type=str, default=os.getenv("ANNOT_FNAME"), help="torch hps")
    parser.add_argument("--debug", action="store_true", help="torch hps")

    # object detection parameters
    parser.add_argument("--detection_per_img", type=int, default=200, help="detection hps")
    parser.add_argument("--nms_thresh", type=float, default=0.22, help="detection hps")
    parser.add_argument("--score_thresh_glb", type=float, default=0.1, help="detection hps")
    parser.add_argument("--detector_lr", type=float, default=1e-2, help="detection hps")
    parser.add_argument("--w_cls", type=float, default=0.5, help="detection hps")
    parser.add_argument("--pre_trained", action="store_true", help="optimizer hps")
    parser.add_argument("--box_key", type=str, default=os.getenv("BOX_KEY"), help="detection hps")
    parser.add_argument("--label_key", type=str, default=os.getenv("LABEL_KEY"), help="detection hps")
    parser.add_argument("--topk_candidates_per_level", type=int, default=1000, help="detection hps")
    parser.add_argument("--sw_inferer_overlap", type=float, default=0.25, help="detection hps")
    parser.add_argument("--sw_batch_size", action="store_true", help="detection hps")
    parser.add_argument("--sw_mode", type=str, default="constant", help="detection hps")

    # ATSS parameters
    parser.add_argument("--num_cands", type=int, default=4, help="atss hps -- num_candidates")
    parser.add_argument("--center_in_gt", action="store_true", help="atss hps -- center_in_gt")
    parser.add_argument("--batch_size_per_image", type=int, default=64, help="atss hps -- batch_size_per_image")
    parser.add_argument("--positive_fraction", type=float, default=0.5, help="atss hps -- positive_fraction")
    parser.add_argument("--pool_size", type=int, default=20, help="atss hps -- pool_size")
    parser.add_argument("--min_neg", type=int, default=16, help="atss hps -- min_neg")

    # optimizer parameters
    parser.add_argument("--momentum", type=float, default=0.9, help="optimizer hps")
    parser.add_argument("--weight_decay", type=float, default=3e-5, help="optimizer hps")
    parser.add_argument("--nesterov", action="store_true", help="optimizer hps")

    # Scheduler parameters
    parser.add_argument("--a_scheduler_step_size", type=int, default=150, help="after scheduler hps")
    parser.add_argument("--a_scheduler_gamma", type=float, default=0.1, help="after scheduler hps")
    parser.add_argument("--wu_scheduler_multiplier", type=int, default=1, help="warmup scheduler hps")
    parser.add_argument("--wu_scheduler_total_epoch", type=int, default=10, help="warmup scheduler hps")

    # Phase of the experiment
    parser.add_argument("--training", action="store_true", help="Run training phase")
    parser.add_argument("--validation", action="store_true", help="Run validation phase")
    parser.add_argument("--testing", action="store_true", help="Run testing phase")

    args = parser.parse_args()
    args.use_pretrained = False # args.model_weights_path != ("" or "/" or None)
    args.compute_dtype = torch.float16 if args.amp else torch.float32
    # args.max_cardinality = 120 if args.dataset == "AIIB23" else 299 # delete

    return args


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
    print("done!")
