import numpy as np
import matplotlib.pyplot as plt

from mmcv import Config
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
import torch
import torch.nn.functional as F

from mmdet.utils import get_device
from mmdet.apis import init_detector
from mmdet.apis.inference import custom_inference
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


def vis_featmap(config_file,
                checkpoint_file,
                image_path):
    cfg = Config.fromfile(config_file)
    device = get_device()

    model = init_detector(config_file, checkpoint_file, device=device)
    model.eval()

    if isinstance(image_path, (list, tuple)):
        is_batch = True
    else:
        imgs = [image_path]
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

    bbox_head = model.bbox_head
    with torch.no_grad():
        backbone_feats = model.extract_feat(data['img'][0])
        mlvl_featmap = bbox_head(backbone_feats)
        mlvl_cls_pred, mlvl_bbox_pred, mlvl_centerness = mlvl_featmap
        # mlvl_cls_pred, mlvl_bbox_pred = mlvl_featmap

    debug_featmap = []
    for idx in range(len(mlvl_bbox_pred)):
        featmap = mlvl_bbox_pred[idx][:, :, :, :]
        featmap = torch.mean(featmap, dim=1, keepdim=True)
        featmap = F.upsample(
            input=featmap,
            size=data['img_metas'][0][0]['img_shape'][:2],
            mode='bilinear'
        )
        featmap = featmap.cpu().numpy()
        debug_featmap.append(featmap)

    fig = plt.figure(figsize=(10, 5))
    for scale, featmap in enumerate(debug_featmap):
        ax = fig.add_subplot(1, len(debug_featmap), scale + 1)
        _, c, _, _ = featmap.shape
        print(f"Feature maps shape: {featmap.shape}")
        im = ax.imshow(featmap[0, 0])
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    vis_featmap("../../local_cfg/fcos_test.py",
                "../../work_dirs/fcos_test/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth",
                "../../../Dataset/image (201).jpg")