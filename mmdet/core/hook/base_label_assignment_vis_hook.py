from typing import Optional

from mmcv.runner.hooks import HOOKS, Hook
import warnings

@HOOKS.register_module()
class BaseLabelAssignmentVisHook(Hook):
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
        assign_matrices, strides, priors_per_level = self._get_assign_results(runner)
        self._plot_results(assign_matrices,
                           strides,
                           priors_per_level)

    def _get_assign_results(self, runner):
        """ This will execute label assignment from the start for only the images
        in self.image_list. Since every model has its own implementation of label assignment,
        every specific label assignment strat will inherit from this Base and execute its own
        label assignment
        This function must execute these things in order:
        1. Grab model from runner
        2. Put model in eval mode (avoid model updating gradients)
        3. Perform forward pass, get outputs of bbox_head
        4. Perform label assignment to get `assign_result` and `sampling_result`
        5. Return `assign_matrices`, `strides` and `multi_priors_per_level`
        """
        pass

    def _plot_results(self,
                      assign_matrices,
                      strides,
                      multi_priors_per_level):
        for (image, gt_bboxes, assign_matrix, stride, priors_per_level) in zip(
                self.image_list,
                self.gt_bboxes_list,
                assign_matrices,
                strides,
                multi_priors_per_level
        ):
            pass