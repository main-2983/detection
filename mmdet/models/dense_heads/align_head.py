import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, bbox_overlaps, build_assigner,
                        build_sampler, images_to_levels, multi_apply,
                        reduce_mean, unmap, distance2bbox)
from mmdet.core.utils import filter_scores_and_topk
from ..builder import HEADS, build_loss
from .atss_head import ATSSHead


@HEADS.register_module()
class AlignHead_v1(ATSSHead):
    """ This is a FCOS Head with simOTA Label Assignment """
    def __init__(self,
                 num_classes,
                 in_channels,
                 anchor_type='anchor_based',
                 **kwargs):
        super(AlignHead_v1, self).__init__(
            num_classes,
            in_channels,
            **kwargs
        )
        assert anchor_type in ['anchor_based', 'anchor_free']
        self.anchor_type = anchor_type
        self.assigner = build_assigner(self.train_cfg.assigner)

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def forward(self, feats):
        return multi_apply(self.forward_single(feats,
                                               self.scales,
                                               self.prior_generator.strides))

    def forward_single(self, x, scale, stride):
        b, c, h, w = x.shape

        level_idx = self.prior_generator.strides.index(stride)
        anchor = self.prior_generator.single_level_grid_priors(
            (h, w), level_idx, device=x.device
        )
        anchor = torch.cat([anchor for _ in range(b)])

        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score = self.atss_cls(cls_feat)

        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        if self.anchor_type == 'anchor_free':
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            bbox_pred = distance2bbox(
                self.anchor_center(anchor) / stride[0], bbox_pred
            ).reshape(b, h, w, 4).permute(0, 3, 1, 2)
        elif self.anchor_type == 'anchor_based':
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred).reshape(
                b, h, w, 4).permute(0, 3, 1, 2) / stride[0]
        else:
            raise NotImplementedError

        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        for idx in range(len(anchor_list)):
            mlvl_anchor_per_image = anchor_list[idx]
            for i in range(len(mlvl_anchor_per_image)):
                one_scale_anchor_per_image = mlvl_anchor_per_image[i]



    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):