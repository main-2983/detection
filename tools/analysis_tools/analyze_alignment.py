import copy
import matplotlib.pyplot as plt

import torch
from mmcv import Config

from mmdet.utils import get_device
from mmdet.datasets import build_dataset
from mmdet.core import BboxOverlaps2D
from mmdet.apis.inference import init_detector


def analyze_aligment(config_file,
                     checkpoint_file,
                     category: int=0,
                     img_id: int=0):
    """ Script to visualize alignment for classification and regression
    Args:
        config_file: model config file
        checkpoint_file: model checkpoint
        category: class to visualize
        img_id: image index from dataset
    Example:
        >>> config_file = "configs/tood/tood_r50_voc.py"
        >>> checkpoint_file = "ckpts/tood_r50_fpn_1x_coco.pth"
        >>> analyze_aligment(config_file, checkpoint_file)
    """
    iou_calculator = BboxOverlaps2D()
    cfg = Config.fromfile(config_file)
    device = get_device()

    model = init_detector(config_file, checkpoint_file, device=device)
    model.eval()
    model_head = model.bbox_head

    if cfg.data.train['type'] != 'MultiImageMixDataset':
        val_dataset = copy.deepcopy(cfg.data.train)
    else:
        val_dataset = copy.deepcopy(cfg.data.test)
    dataset = build_dataset(val_dataset)

    img = dataset[img_id]['img'].data.unsqueeze(0).to(device)
    gt_bboxes = dataset[img_id]['gt_bboxes'].data.to(device)

    with torch.no_grad():
        backbone_feat = model.extract_feat(img) # tuple, len = num_level
        cls_scores, bbox_preds = model_head(backbone_feat)
        # cls_score (batch, classes, H, W)
        # bbox_pred (batch, 4, H, W)

        flatten_bbox_preds = []
        for bbox_pred, stride in zip(bbox_preds, model_head.prior_generator.strides):
            flatten_bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(1, -1, 4) * stride[0]
            flatten_bbox_preds.append(flatten_bbox_pred)

        iou_scores = []
        for flatten_bbox_pred, bbox_pred in zip(flatten_bbox_preds, bbox_preds):
            flatten_bbox_pred = flatten_bbox_pred.squeeze() # (N, H*W, 4) -> (H*W, 4)
            overlaps = iou_calculator(flatten_bbox_pred, gt_bboxes)  # (H*W, num_gts)
            orig_shape = bbox_pred.shape[2:]
            overlaps = overlaps.reshape((orig_shape[0], orig_shape[1], overlaps.shape[-1])) # (H, W, num_gts)
            iou_scores.append(overlaps)

        cls_scores = [cls_score.squeeze().permute(1, 2, 0) for cls_score in cls_scores]

    img = img.squeeze().permute(1, 2, 0).cpu().numpy()

    for scale, (cls_score, iou_score) in enumerate(zip(cls_scores, iou_scores)):
        cls_score = cls_score[:, :, category] # (H, W, 1)

        cls_score = cls_score.cpu().numpy()
        iou_score = iou_score.cpu().numpy()
        num_bboxes = iou_score.shape[-1]

        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(f"Scale {scale + 1}")

        ax1 = fig.add_subplot(num_bboxes, 2, 1)
        ax2 = fig.add_subplot(num_bboxes, 2, 2)
        ax1.set_title("Cls Score")
        ax2.set_title("Image")
        im1 = ax1.imshow(cls_score, cmap=plt.cm.jet)
        im2 = ax2.imshow(img)
        cb1 = plt.colorbar(im1, fraction=0.05, ax=ax1)
        cb1.ax.tick_params(labelsize=5)

        for i in range(num_bboxes):
            score = iou_score[:, :, i]
            ax = fig.add_subplot(num_bboxes, 2, i + 3)
            ax.set_title(f"IoU Score {i + 1}")
            im = ax.imshow(score, cmap=plt.cm.jet)
            cb = plt.colorbar(im, fraction=0.05, ax=ax)
            cb.ax.tick_params(labelsize=5)

        plt.show()