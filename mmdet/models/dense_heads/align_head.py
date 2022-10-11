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

    def anchor_center(self, anchors, with_stride=None):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        if with_stride is None:
            return torch.stack([anchors_cx, anchors_cy], dim=-1)
        else:
            stride_w, stride_h = with_stride[0], with_stride[1]
            stride_w = anchors_cx.new_full((anchors_cx.shape[0],),
                                           stride_w)
            stride_h = anchors_cy.new_full((anchors_cy.shape[0],),
                                           stride_h)
            priors = torch.stack([anchors_cx, anchors_cy, stride_w, stride_h],
                                 dim=-1)
            return priors


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


    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])

        # convert anchor to point with stride
        for scale, (single_level_anchor, stride) in enumerate(
            anchor_list, self.prior_generator.strides
        ):
            single_level_prior = self.anchor_center(single_level_anchor,
                                                    stride)
            anchor_list[scale] = single_level_prior



    def _get_target_single(self,
                           flat_priors,
                           cls_preds,
                           decoded_bboxes,
                           lqe_preds,
                           gt_bboxes,
                           gt_labels):
        num_priors = flat_priors.shape[0]
        num_gts = gt_labels.shape[0]
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)

        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [flat_priors[:, :2] + flat_priors[:, 2:] * 0.5, flat_priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * lqe_preds.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)