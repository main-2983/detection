import copy
import matplotlib.pyplot as plt
import numpy as np

import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool

from mmdet.utils import get_device
from mmdet.datasets import build_dataset, replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.core import BboxOverlaps2D, filter_scores_and_topk
from mmdet.apis.inference import init_detector


def analyze_cls_and_iou_alignment(config_file,
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


def analyze_cls_and_lqe_alignment(config_file,
                                  checkpoint_file,
                                  imgs,
                                  category: int=0):
    cfg = Config.fromfile(config_file)
    device = get_device()

    model = init_detector(config_file, checkpoint_file, device=device)
    model.eval()
    model_head = model.bbox_head

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    with torch.no_grad():
        backbone_feat = model.extract_feat(data['img'][0]) # tuple, len = num_level
        mlvl_predictions = model_head(backbone_feat)
    assert len(mlvl_predictions) == 3, "This function is only for Decode Head with 3 outputs:" \
                                       "'Classification', 'LQE' and 'Bounding Box'"
    mlvl_cls_pred, mlvl_bbox_pred, mlvl_lqe_pred = mlvl_predictions

    for scale, (cls_pred, lqe_pred) in enumerate(zip(mlvl_cls_pred,
                                                            mlvl_lqe_pred)):
        cls_pred, lqe_pred = cls_pred[0], lqe_pred[0]
        cls_pred = cls_pred.permute(1, 2, 0)
        lqe_pred = lqe_pred.permute(1, 2, 0)
        cls_pred = cls_pred.sigmoid()
        lqe_pred = lqe_pred.sigmoid()
        mlvl_cls_pred[scale] = cls_pred
        mlvl_lqe_pred[scale] = lqe_pred

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"Cls and LQE Alignment")
    for scale, (cls_score, lqe_score) in enumerate(zip(mlvl_cls_pred, mlvl_lqe_pred)):
        cls_score = cls_score[:, :, category] # (H, W, 1)

        cls_score = cls_score.cpu().numpy()
        lqe_score = lqe_score.cpu().numpy()

        ax1 = fig.add_subplot(1, len(mlvl_cls_pred), scale + 1)
        ax2 = fig.add_subplot(2, len(mlvl_cls_pred), scale + 1)
        ax1.set_title("Cls Score")
        ax2.set_title("LQE Score")
        im1 = ax1.imshow(cls_score)
        im2 = ax2.imshow(lqe_score)
        cb1 = plt.colorbar(im1, fraction=0.04, ax=ax1)
        cb2 = plt.colorbar(im2, fraction=0.04, ax=ax2)
        cb1.ax.tick_params(labelsize=5)
        cb2.ax.tick_params(labelsize=5)

    plt.show()

def analyze_duplicate(config_file,
                      checkpoint_file,
                      imgs):
    cfg = Config.fromfile(config_file)
    device = get_device()

    model = init_detector(config_file, checkpoint_file, device=device)
    model.eval()
    model_head = model.bbox_head

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    with torch.no_grad():
        backbone_feat = model.extract_feat(data['img'][0]) # tuple, len = num_level
        mlvl_predictions = model_head(backbone_feat)

        if len(mlvl_predictions) == 3:
            mlvl_cls_pred, mlvl_bbox_pred, mlvl_centerness_pred = mlvl_predictions
        elif len(mlvl_predictions) == 2:
            mlvl_cls_pred, mlvl_bbox_pred = mlvl_predictions
            mlvl_centerness_pred = [None] * len(mlvl_cls_pred)
        else:
            raise NotImplementedError

    filtered_mlvl_conf = []
    # cls_pred: (batch, num_priors * num_classes, H, W)
    for scale, (cls_pred, bbox_pred, centerness_pred) in enumerate(zip(mlvl_cls_pred,
                                                                   mlvl_bbox_pred,
                                                                   mlvl_centerness_pred)):
        cls_pred, bbox_pred = cls_pred[0], bbox_pred[0]
        if centerness_pred is not None:
            centerness_pred = centerness_pred[0]
        _, h, w = cls_pred.shape

        cls_pred = cls_pred.permute(1, 2, 0)
        cls_pred = cls_pred.reshape(-1, cls_pred.shape[-1])
        if model_head.loss_cls.activated is not True:
            cls_pred = cls_pred.sigmoid()
        else:
            cls_pred = cls_pred

        if centerness_pred is not None:
            centerness_pred = centerness_pred.permute(1, 2, 0)
            centerness_pred = centerness_pred.reshape(-1,)
            centerness_pred = centerness_pred.sigmoid()

        filtered_scores, labels, keep_idxs, filtered_results = filter_scores_and_topk(
            cls_pred, 0.05, 100, None
        )
        if centerness_pred is not None:
            filtered_centerness = centerness_pred[keep_idxs]

        filtered_conf = torch.zeros((h * w,), device=device)
        filtered_conf[keep_idxs] = filtered_scores * filtered_centerness if centerness_pred is not None else filtered_scores
        filtered_conf = filtered_conf.reshape(h, w)
        filtered_mlvl_conf.append(filtered_conf)

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f"Confidence score")
    for scale, conf in enumerate(filtered_mlvl_conf):
        conf = conf.cpu().numpy()

        ax = fig.add_subplot(1, len(filtered_mlvl_conf), scale + 1)
        ax.set_title(f"{scale + 1}")
        im = ax.imshow(conf)
        cb = plt.colorbar(im, fraction=0.04, ax=ax)
        cb.ax.tick_params(labelsize=5)

    plt.show()
