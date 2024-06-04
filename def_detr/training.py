# ruff: noqa: F401
import argparse
import datetime
import json
import random
import time
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

import util.misc as utils
# from datasets import build_dataset, get_coco_api_from_dataset
from datasets.BifDet24DataModule import BifDet2024DataModule
import monai

from engine import train_one_epoch, evaluate
from models.transoar.transoarnet import TransoarNet
from models.transoar.build import build_criterion


def get_args_parser():
    load_dotenv(find_dotenv("config.env"))
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=float(os.getenv('LR')), type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=int(os.getenv('BATCH_SIZE')), type=int)
    parser.add_argument('--epochs', default=os.getenv('NUM_EPOCHS'), type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    # parser.add_argument("--optimizer", type=str, default=os.getenv('OPTIMIZER'), help="The optimizer to use")
    parser.add_argument("--momentum", type=float, default=0.9, help="optimizer hps")
    parser.add_argument("--nesterov", action="store_true", help="optimizer hps")
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # Scheduler parameters
    parser.add_argument("--a_scheduler_step_size", type=int, default=150, help="after scheduler hps")
    parser.add_argument("--a_scheduler_gamma", type=float, default=0.1, help="after scheduler hps")
    parser.add_argument("--wu_scheduler_multiplier", type=int, default=1, help="warmup scheduler hps")
    parser.add_argument("--wu_scheduler_total_epoch", type=int, default=10, help="warmup scheduler hps")

    
    # * BifDET parameters
    parser.add_argument("--annot_fname", type=str, default=os.getenv("ANNOT_FNAME"), help="torch hps")
    parser.add_argument("--max_cardinality", type=int, default=os.getenv('MAX_CARDINALITY'), help="size of dataset")

    # patch size - for now static
    parser.add_argument("--patch_size", type=int, default=int(os.getenv('PATCH_SIZE')), help="Minimum pixel value")

    # Voxel values - for now static
    parser.add_argument("--pixel_value_min", type=int, default=float(os.getenv('PIXEL_VALUE_MIN')), help="Minimum pixel value")
    parser.add_argument("--pixel_value_max", type=int, default=float(os.getenv('PIXEL_VALUE_MAX')), help="Maximum pixel value")
    parser.add_argument("--pixel_norm_min", type=float, default=float(os.getenv('PIXEL_NORM_MIN')), help="Minimum normalized pixel value")
    parser.add_argument("--pixel_norm_max", type=float, default=float(os.getenv('PIXEL_NORM_MAX')), help="Maximum normalized pixel value")
    parser.add_argument("--voxel_size", type=int, default=float(os.getenv('VOXEL_SIZE')), help="Voxel size")

    # * Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # * Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--pretrained', default=False, action="store_true")

    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--pos_encoding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=3, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=3, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=6, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Transoar
    parser.add_argument('--use_seg_proxy_loss', action='store_true', default=False)
    parser.add_argument('--fg_bg', action='store_true', default=False)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--start_channels', type=int, default=24)
    parser.add_argument('--input_level', type=str, default='P2')
    parser.add_argument('--use_encoder', action='store_true', default=True)
    parser.add_argument('--out_fmaps', default=["P5", "P4", "P3"])
    parser.add_argument('--fpn_channels', default=384)
    parser.add_argument('--n_points', default=4)
    parser.add_argument('--layers', default=2)
    parser.add_argument('--feature_levels', default=["P2", "P3", "P4", "P5"])
    parser.add_argument('--conv_kernels', default=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]])
    parser.add_argument('--strides', default=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]])
    parser.add_argument('--depths', default=[2, 2, 2, 2])
    parser.add_argument('--num_heads', default=[3, 6, 12, 24])
    parser.add_argument('--window_size', default=[5, 5, 5])
    parser.add_argument('--mlp_ratio', default=4)
    parser.add_argument('--attn_drop_rate', default=0.)
    parser.add_argument('--drop_rate', default=0.)
    parser.add_argument('--drop_path_rate', default=0.2)
    parser.add_argument('--qkv_bias', default=True)
    parser.add_argument('--qk_scale', default='null')
    parser.add_argument('--use_decoder_attn', default=False)
    parser.add_argument('--use_encoder_attn', default=False)
    parser.add_argument('--conv_merging', default=False)

    # * Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # * dataset parameters
    parser.add_argument('--num_classes', default=1)

    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./logs/test/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=int(os.getenv('NUM_WORKERS')), type=int)
    parser.add_argument("--cache_ds", action="store_true", default=False, help="Caching dataset or not")
    parser.add_argument("--amp", action="store_true", default=False, help="For improving the training")

    return parser

def main(args):
    # utils.init_distributed_mode(args)

    params_dict = {
        'PATCH_SIZE': (args.patch_size,) * 3,
        'VAL_PATCH_SIZE': (args.patch_size,) * 3,
        'BATCH_SIZE': args.batch_size,
        'MAX_CARDINALITY': args.max_cardinality,
        'NUM_WORKERS': args.num_workers,
        'PIXEL_VALUE_MIN': args.pixel_value_min,
        'PIXEL_VALUE_MAX': args.pixel_value_max,
        'PIXEL_NORM_MIN': args.pixel_norm_min,
        'PIXEL_NORM_MAX': args.pixel_norm_max,
        'VOXEL_SIZE': (1,) * 3,
        'NUM_EPOCHS': args.epochs,
        'CACHE_DS': args.cache_ds,
        'AVAILABLE_GPUs': torch.cuda.device_count(),
        'DEVICE_NO': int(os.getenv('DEVICE_NO'))
    }

    output_dir = Path(args.output_dir)

    if args.output_dir and utils.is_main_process():
        writer = SummaryWriter(log_dir=output_dir)


    device = torch.device(args.device)
    args.compute_dtype = torch.float16 if args.amp else torch.float32


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    monai.utils.misc.set_determinism(seed=seed)
    print(f"seed fixed to {seed}")

    monai.config.print_config()
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = True
    # torch.set_num_threads(4)

    print(os.getenv("DATA_SRC"))
    print(args.annot_fname)

    # 1. Data loaders
    data_module = BifDet2024DataModule(
        train_parent_path=Path(os.getenv("DATA_SRC")),
        batch_size=args.batch_size,
        bbox_path=args.annot_fname,
        params=params_dict,
        compute_dtype=args.compute_dtype,
    )
    data_module.prepare_data_monai()
    data_module.setup()

    # model = torch.jit.script(TransoarNet(args).to(device=device))
    model = TransoarNet(args).to(device=device)
    criterion = build_criterion(args).to(device=device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

     # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=args.momentum, nesterov=args.nesterov,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.a_scheduler_step_size, gamma=args.a_scheduler_gamma)
    # iter_epochs = len(data_loader_train)
    # if args.lr_drop_epochs is not None and args.scheduler_lr == 'step':
    #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [drop_epoch*iter_epochs for drop_epoch in args.lr_drop_epochs], gamma=0.1)
    # elif args.scheduler_lr == 'step':
    #     lr_scheduler = LinearWarmupStepLR(optimizer, warmup_epochs=args.warmup_lr_epochs*iter_epochs, warmup_start_lr=args.warmup_lr_start, step_size=args.lr_drop*iter_epochs, gamma=0.1)
    # elif args.scheduler_lr == 'cosine':
    #     lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_lr_epochs*iter_epochs, warmup_start_lr=args.warmup_lr_start, max_epochs=args.epochs*iter_epochs)
    scaler = GradScaler(enabled=args.amp)

    if args.resume:
        checkpoint = torch.load(Path(args.resume))
        model.load_state_dict(checkpoint['model'])

    if args.eval:
        test_stats = evaluate(
            model, criterion, data_module.val_dataloader(), device, output_dir
        )
        print(test_stats)
    else:

        # if args.distributed:
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        #     model_without_ddp = model.module

        print("Start training")
        start_time = time.time()

        for epoch in range(args.start_epoch, args.epochs):
            # if args.distributed:
            #     sampler_train.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, data_module.train_dataloader(), optimizer, device, epoch, scaler, max_norm=args.clip_max_norm)
            lr_scheduler.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 5 epochs
                if (epoch + 1) % args.a_scheduler_step_size == 0 or (epoch + 1) % 100 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

            test_stats = evaluate(
                model, criterion, data_module.val_dataloader(), device, output_dir
            )
            print(test_stats)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.output_dir:
                with (output_dir / "log.json").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                for k,v in log_stats.items():
                    if k == "epoch" or k=='n_parameters':
                        continue
                    elif k[:6] == 'train_':
                        writer.add_scalar(f"Training/{k[6:]}", v, log_stats["epoch"])
                    elif k[:5] == 'test_':
                        if isinstance(v, list):
                            writer.add_scalar(f"Testing/{k[5:]}", v[0], log_stats["epoch"])
                        else:
                            writer.add_scalar(f"Testing/{k[5:]}", v, log_stats["epoch"])

            #     # for evaluation logs
            #     if coco_evaluator is not None:
            #         (output_dir / 'eval').mkdir(exist_ok=True)
            #         if "bbox" in coco_evaluator.coco_eval:
            #             filenames = ['latest.pth']
            #             if epoch % 50 == 0:
            #                 filenames.append(f'{epoch:03}.pth')
            #             for name in filenames:
            #                 torch.save(coco_evaluator.coco_eval["bbox"].eval,
            #                            output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    # parser inputs
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)