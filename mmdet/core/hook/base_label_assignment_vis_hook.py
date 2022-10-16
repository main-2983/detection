from typing import Optional
import cv2

import torch

from mmcv.runner.hooks import HOOKS, Hook
import warnings


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()


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
        assign_matrices, strides, priors_per_level, featmap_sizes =\
            self._get_assign_results(runner)
        self._plot_results(assign_matrices,
                           strides,
                           priors_per_level,
                           featmap_sizes)

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
                      multi_priors_per_level,
                      multi_featmap_sizes):
        for (image, image_metas, gt_bboxes, gt_label, assign_matrix, stride, priors_per_level, featmap_sizes) in \
                zip(
                    self.image_list,
                    self.img_metas_list,
                    self.gt_bboxes_list,
                    self.gt_label_list,
                    assign_matrices,
                    strides,
                    multi_priors_per_level,
                    multi_featmap_sizes):
            assert len(stride) == len(priors_per_level), "Number of level must equal to number of strides"
            results = []
            # loop through each scale level to reshape 1D assign matrix
            # to 2D assign matrix of each scale
            for i in range(len(priors_per_level)):
                stride_level_i = stride[i]
                featmap_size_level_i = featmap_sizes[i]
                num_priors_level_i = priors_per_level[i]
                if i == 0:
                    matrix_level_i = assign_matrix[:num_priors_level_i]
                else:
                    matrix_level_i = \
                        assign_matrix[priors_per_level[i-1]:num_priors_level_i]
                matrix_level_i = matrix_level_i.view((featmap_size_level_i[0],
                                                      featmap_size_level_i[1]))
                # this will return a list of 2D position of where the label is non-zero on the matrix
                pos_location_level_i = torch.nonzero(matrix_level_i)
                if pos_location_level_i.numel() > 0:
                    for location in pos_location_level_i:
                        category_id = matrix_level_i[location[0], location[1]]
                        location = (location + 0.5) * stride_level_i
                        results.append(
                            torch.cat([location.int(), category_id.int(), stride_level_i.int()])
                        )

            np_image = image.clone()[0].cpu().numpy().transpose(1, 2, 0)
            # draw positive anchors as circles
            for result in results:
                coord = result[:2]
                category_id = result[2]
                scale = result[3]
                np_image = cv2.circle(np_image.copy(),
                                      coord[::-1],
                                      int(scale / 2),
                                      colors(category_id),
                                      thickness=-1)
            # draw gt bbox
            for gt_bbox in gt_bboxes:
                np_image = cv2.rectangle(np_image.copy(),
                                         gt_bbox[:2],
                                         gt_bbox[2:],
                                         colors(gt_label.int()))
            cv2.imwrite()