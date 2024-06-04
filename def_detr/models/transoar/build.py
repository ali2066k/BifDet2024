"""Module containing functionality to build different parts of the model."""

from .matcher import HungarianMatcher
from .criterion import TransoarCriterion
from .attn_fpn.attn_fpn import AttnFPN
from .necks.def_detr_transformer import DeformableTransformer
from .position_encoding import PositionEmbeddingSine3D, PositionEmbeddingLearned3D
from monai.networks.nets import resnet
from monai.networks.blocks import BackboneWithFPN
from monai.networks.blocks.feature_pyramid_network import LastLevelMaxPool

def build_backbone(args):
    model = AttnFPN(
        args
    )
    return model

def build_backbone_monai(args):
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

    # feature_extractor = resnet_fpn_feature_extractor(
    #     backbone=backbone,
    #     spatial_dims=3,
    #     pretrained_backbone=False,
    #     trainable_backbone_layers=None,
    #     returned_layers=[1, 2],
    # )
    return_layers = {f"layer{k}": str(v) for v, k in enumerate([1, 2])}
    in_channels_stage2 = backbone.in_planes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in [1,2]]
    # out_channels = 384
    out_channels = args.hidden_dim

    feature_extractor = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, spatial_dims=3, extra_blocks=LastLevelMaxPool(3))

    return feature_extractor

def build_neck(args):
    model = DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        use_cuda=args.use_cuda,
        use_encoder=args.use_encoder,
        num_feature_levels=args.num_feature_levels
    ) 

    return model

def build_criterion(args):
    matcher = HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou
    )

    weight_dict = {'cls': args.cls_loss_coef, 'bbox': args.bbox_loss_coef, 'giou': args.giou_loss_coef}
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        # aux_weight_dict.update({k + '_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = TransoarCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        seg_proxy=args.use_seg_proxy_loss,
        seg_fg_bg=args.fg_bg,
        weight_dict=weight_dict
    )

    return criterion

def build_pos_enc(args):
    channels = args.hidden_dim
    if args.pos_encoding == 'sine':
        return PositionEmbeddingSine3D(channels=channels)
    elif args.pos_encoding == 'learned':
        return PositionEmbeddingLearned3D(channels=channels)
    else:
        raise ValueError('Please select a implemented pos. encoding.')
