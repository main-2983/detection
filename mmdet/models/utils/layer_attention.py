import torch.nn as nn

from mmcv.cnn import ConvModule


class LayerAttn(nn.Module):
    def __init__(self,
                 channels,
                 groups,
                 ratio=8,
                 act_cfg=dict(
                     type='Sigmoid'
                 )):
        super(LayerAttn, self).__init__()
        self.channels = channels
        self.groups = groups
        self.layer_attn = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=channels // ratio,
                kernel_size=1
            ),
            ConvModule(
                in_channels=channels // ratio,
                out_channels=groups,
                kernel_size=1,
                act_cfg=act_cfg
            )
        )

    def forward(self, x):
        b, c, h, w = x.shape

        avg_feat = nn.AdaptiveAvgPool2d(1)(x)
        weight = self.layer_attn(avg_feat)

        x = x.view(b, self.groups, c // self.groups, 1, 1)
        weight = weight.view(b, self.groups, 1, 1, 1)
        _x = x.clone()

        for group in range(self.groups):
            _x[:, group] = x[:, group] * weight[:, group]

        _x = _x.view(b, c, h, w)

        return _x


class LayerAttnS(nn.Module):
    def __init__(self,
                 channels,
                 groups,
                 act_cfg=dict(
                     type='Sigmoid'
                 )):
        super(LayerAttnS, self).__init__()
        self.channels = channels
        self.groups = groups
        self.layer_attn = ConvModule(
            in_channels=channels,
            out_channels=groups,
            kernel_size=1,
            act_cfg=act_cfg
        )

    def forward(self, x):
        b, c, h, w = x.shape

        avg_feat = nn.AdaptiveAvgPool2d(1)(x)
        weight = self.layer_attn(avg_feat)

        x = x.view(b, self.groups, c // self.groups, 1, 1)
        weight = weight.view(b, self.groups, 1, 1, 1)
        _x = x.clone()

        for group in range(self.groups):
            _x[:, group] = x[:, group] * weight[:, group]

        _x = _x.view(b, c, h, w)

        return _x
