from typing import Optional

import torch
from mmcv.runner.hooks import HOOKS, Hook
import warnings
from mmcv.runner import EpochBasedRunner

@HOOKS.register_module()
class LabelAssignmentVisHook(Hook):
    def __init__(self,
                 sample_idxs: Optional[int, list]=0,
                 num_images=None):
        self.sample_idxs = [sample_idxs]
        if num_images is not None and isinstance(sample_idxs, int):
            self.sample_idxs = [_ for _ in range(num_images)]
            warnings.warn(f"parameter 'sample_idxs' must be a list when 'num_images' is not None, "
                          f"setting 'sample_idxs' to {self.sample_idxs}")
        self.num_images = num_images
        # parameter to check if this Hook has executed before_train_epoch
        # this will make sure before_train_epoch only executes once
        # we set this explicit in before_train_epoch instead of using before_run hook because
        # before_run doesn't contain dataset or dataloader
        self.sampled = False

    def before_train_epoch(self, runner):
        if self.sampled is False:
            dataset = runner.data_loader.dataset
            self.image_list = []
            self.gt_bboxes_list = []
            self.gt_label_list = []
            self.img_metas_list = []
            for i in self.sample_idxs:
                sample_dict = dataset[i]
                image_tensor = sample_dict['img'].data[None]
                gt_bboxes_tensor = sample_dict['gt_bboxes'].data
                gt_labels_tensor = sample_dict['gt_labels'].data
                img_metas = sample_dict['img_metas'].data
                self.image_list.append(image_tensor)
                self.gt_bboxes_list.append(gt_bboxes_tensor)
                self.gt_label_list.append(gt_labels_tensor)
                self.img_metas_list.append(img_metas)
            self.sampled = True

    def after_train_epoch(self, runner):
        # this will execute label assignment from the start for only the images
        # in self.image_list
        model = runner.model
        # do this to stop model from updating gradient
        model.eval()
        with torch.no_grad():
            for (image, gt_bboxes, gt_labels, img_metas) in zip(
                    self.image_list,
                    self.gt_bboxes_list,
                    self.gt_label_list,
                    self.img_metas_list):
                outs = model(image)
                loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
                # this will calculate loss for one image and also
                # perform label assignment during the process
                losses = model.loss(*loss_inputs, gt_bboxes_ignore=None)

