import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale

from ..builder import HEADS
from .atss_head import ATSSHead


@HEADS.register_module()
class DecoupledHead(ATSSHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 **kwargs):
        super(DecoupledHead, self).__init__(
            num_classes,
            in_channels,
            **kwargs
        )

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        pred_pad_size = self.pred_kernel_size // 2

        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            self.pred_kernel_size,
            padding=pred_pad_size)

        self.center_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 2,
            self.pred_kernel_size,
            padding=pred_pad_size
        )

        self.wh_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 2,
            self.pred_kernel_size,
            padding=pred_pad_size)

        self.atss_centerness = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 1,
            self.pred_kernel_size,
            padding=pred_pad_size)

        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred

        center_pred = scale(self.center_reg(cls_feat)).float()
        wh_pred = scale(self.wh_reg(reg_feat)).float()
        bbox_pred = torch.cat([center_pred, wh_pred], dim=1)

        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness
