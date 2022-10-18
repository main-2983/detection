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
from .fcos_head import FCOSHead


@HEADS.register_module()
class AlignHead_v1(FCOSHead):
    """ FCOS Head with simOTA Assigner """
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_lqe=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0
                 ),
                 **kwargs):
        super(AlignHead_v1, self).__init__(
            num_classes,
            in_channels,
            **kwargs
        )
        self.loss_lqe = build_loss(loss_lqe)
        assigner_cfg = dict(type='SimOTAAssigner',
                            center_radius=2.5,
                            candidate_topk=10)
        self.assigner = build_assigner(assigner_cfg)
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)


    def _bbox_decode(self, priors, bbox_preds):
        """ Convert [l, t, r, b] bbox format to [x1, y1, x2, y2] bbox format """
        x1 = priors[..., 0] - bbox_preds[..., 0]
        y1 = priors[..., 1] - bbox_preds[..., 1]
        x2 = priors[..., 0] + bbox_preds[..., 2]
        y2 = priors[..., 1] + bbox_preds[..., 3]
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _bbox_encode(self, priors, bbox_targets):
        """ Convert [x1, y1, x2, y2] bbox format to [l, t, r, b] bbox format """
        left = priors[..., 0] - bbox_targets[..., 0]
        top = priors[..., 1] - bbox_targets[..., 1]
        right = bbox_targets[..., 2] - priors[..., 0]
        bottom = bbox_targets[..., 3] - priors[..., 1]
        return torch.stack([left, top, right, bottom], dim=-1)

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             lqe_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_lqe_preds = [
            lqe_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for lqe_pred in lqe_preds
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_lqe_preds = torch.cat(flatten_lqe_preds, dim=1)
        #flatten_priors =
        flatten_priors = torch.cat(mlvl_priors)
        flatten_priors = flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, lqe_targets, bbox_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_lqe_preds.detach(),
             flatten_priors,
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        lqe_targets = torch.cat(lqe_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)

        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(lqe_targets.sum().detach()), 1e-6)
        loss_bbox = self.loss_bbox(
            flatten_bboxes.view(-1, 4)[pos_masks],
            bbox_targets,
            avg_factor=centerness_denorm)
        loss_lqe = self.loss_lqe(flatten_lqe_preds.view(-1, 1)[pos_masks],
                                 lqe_targets.unsqueeze(-1),
                                 avg_factor=num_total_samples)
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
            cls_targets,
            avg_factor=num_total_samples)

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_lqe=loss_lqe)

        return loss_dict

    @torch.no_grad()
    def _get_target_single(self, cls_preds, lqe_preds, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            lqe_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, lqe_target, bbox_target,
                    l1_target, 0)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * lqe_preds.unsqueeze(1).sigmoid(),
            priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        cls_target = sampling_result.pos_gt_labels
        # cls_target = F.one_hot(sampling_result.pos_gt_labels,
        #                        self.num_classes)
        bbox_target = sampling_result.pos_gt_bboxes
        pos_priors = priors[pos_inds]
        encoded_bbox_target = self._bbox_encode(pos_priors, bbox_target)
        lqe_target = self.centerness_target(encoded_bbox_target)
        foreground_mask = torch.zeros_like(lqe_preds).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, lqe_target, bbox_target, num_pos_per_img)