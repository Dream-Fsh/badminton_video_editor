import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_floor = pad_t // 2
        pad_t_ceil = pad_t - pad_t_floor
        pad_h_floor = pad_h // 2
        pad_h_ceil = pad_h - pad_h_floor
        pad_w_floor = pad_w // 2
        pad_w_ceil = pad_w - pad_w_floor

        x = F.pad(x, (pad_w_floor, pad_w_ceil, pad_h_floor, pad_h_ceil, pad_t_floor, pad_t_ceil))
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1), padding=0,
                 activation_fn=F.relu, use_batch_norm=True, use_bias=False, name='unit_3d'):

        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._name = name

        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=output_channels, kernel_size=kernel_shape,
                                stride=stride, padding=padding, bias=use_bias)

        if self._use_batch_norm:
            # momentum从0.01降到0.001（小数据集batch统计不稳定，需要更平滑的running统计）
            self.bn = nn.BatchNorm3d(output_channels, eps=0.001, momentum=0.001)

    def forward(self, x):
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1],
                         name=name + '/b0')

        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1],
                          name=name + '/b1a')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          padding=1, name=name + '/b1b')

        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1],
                          name=name + '/b2a')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          padding=1, name=name + '/b2b')

        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1],
                          name=name + '/b3b')

        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture."""

    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self._name = name

        self.end_points = {}
        
        self.Conv3d_1a_7x7 = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=[2, 2, 2], padding=[3, 3, 3], name='Conv3d_1a_7x7')
        self.end_points['Conv3d_1a_7x7'] = self.Conv3d_1a_7x7
        if self._final_endpoint == 'Conv3d_1a_7x7': return

        self.MaxPool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=0)
        self.end_points['MaxPool3d_2a_3x3'] = self.MaxPool3d_2a_3x3
        if self._final_endpoint == 'MaxPool3d_2a_3x3': return

        self.Conv3d_2b_1x1 = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_2b_1x1')
        self.end_points['Conv3d_2b_1x1'] = self.Conv3d_2b_1x1
        if self._final_endpoint == 'Conv3d_2b_1x1': return

        self.Conv3d_2c_3x3 = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3],
                                            padding=[1, 1, 1], name='Conv3d_2c_3x3')
        self.end_points['Conv3d_2c_3x3'] = self.Conv3d_2c_3x3
        if self._final_endpoint == 'Conv3d_2c_3x3': return

        self.MaxPool3d_3a_3x3 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=0)
        self.end_points['MaxPool3d_3a_3x3'] = self.MaxPool3d_3a_3x3
        if self._final_endpoint == 'MaxPool3d_3a_3x3': return

        self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32], name='Mixed_3b')
        self.end_points['Mixed_3b'] = self.Mixed_3b
        if self._final_endpoint == 'Mixed_3b': return

        self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64], name='Mixed_3c')
        self.end_points['Mixed_3c'] = self.Mixed_3c
        if self._final_endpoint == 'Mixed_3c': return

        self.MaxPool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=0)
        self.end_points['MaxPool3d_4a_3x3'] = self.MaxPool3d_4a_3x3
        if self._final_endpoint == 'MaxPool3d_4a_3x3': return

        self.Mixed_4b = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name='Mixed_4b')
        self.end_points['Mixed_4b'] = self.Mixed_4b
        if self._final_endpoint == 'Mixed_4b': return

        self.Mixed_4c = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name='Mixed_4c')
        self.end_points['Mixed_4c'] = self.Mixed_4c
        if self._final_endpoint == 'Mixed_4c': return

        self.Mixed_4d = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name='Mixed_4d')
        self.end_points['Mixed_4d'] = self.Mixed_4d
        if self._final_endpoint == 'Mixed_4d': return

        self.Mixed_4e = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name='Mixed_4e')
        self.end_points['Mixed_4e'] = self.Mixed_4e
        if self._final_endpoint == 'Mixed_4e': return

        self.Mixed_4f = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name='Mixed_4f')
        self.end_points['Mixed_4f'] = self.Mixed_4f
        if self._final_endpoint == 'Mixed_4f': return

        self.MaxPool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.end_points['MaxPool3d_5a_2x2'] = self.MaxPool3d_5a_2x2
        if self._final_endpoint == 'MaxPool3d_5a_2x2': return

        self.Mixed_5b = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name='Mixed_5b')
        self.end_points['Mixed_5b'] = self.Mixed_5b
        if self._final_endpoint == 'Mixed_5b': return

        self.Mixed_5c = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name='Mixed_5c')
        self.end_points['Mixed_5c'] = self.Mixed_5c
        if self._final_endpoint == 'Mixed_5c': return

        # 使用自适应平均池化，支持任意输入尺寸
        # 原设计：nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))，只支持224×224输入
        # 改为自适应池化，输出固定维度，支持任意尺寸输入
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 输出 (B, C, 1, 1, 1)
        self.dropout = nn.Dropout(0.7)  # 增加到0.7（小数据集需要更强正则化，原0.6）
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                         kernel_shape=[1, 1, 1],
                         activation_fn=None,
                         use_batch_norm=False,
                         use_bias=True,
                         name='logits')
        self.end_points['Logits'] = self.logits
        if self._final_endpoint == 'Logits': return

        self.softmax = nn.Softmax(dim=1)
        self.end_points['Predictions'] = self.softmax
        
        # 权重初始化（小数据集需要更好的初始化）
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重（Kaiming/He初始化）"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Kaiming初始化（He初始化），适合ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                # BatchNorm初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def freeze_layers(self, num_layers=5):
        """
        冻结前n层参数（小数据集时防止过拟合）
        
        参数:
            num_layers: 冻结的层数（从前面开始）
                     5: 冻结到Mixed_3c（前5个endpoint）
                     8: 冻结到Mixed_4c（前8个endpoint）
        """
        # 需要冻结的层（从前面开始）
        freeze_endpoints = self.VALID_ENDPOINTS[:num_layers]
        
        frozen_params = 0
        total_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            # 检查参数是否属于需要冻结的层
            should_freeze = any(endpoint in name for endpoint in freeze_endpoints)
            
            if should_freeze:
                param.requires_grad = False
                frozen_params += param.numel()
                print(f"Frozen: {name}")
        
        print(f"\n[Frozen] {frozen_params}/{total_params} ({frozen_params/total_params*100:.1f}%) parameters")
        print(f"[Trainable] {total_params-frozen_params}/{total_params} ({(total_params-frozen_params)/total_params*100:.1f}%) parameters")

    def unfreeze_all(self):
        """解冻所有层"""
        for param in self.parameters():
            param.requires_grad = True
        print("[Unfrozen] All layers are now trainable")

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                if end_point == 'Logits':
                    x = self.avg_pool(x)
                    x = self.dropout(x)
                x = self.end_points[end_point](x)
            if end_point == self._final_endpoint:
                break

        if self._final_endpoint == 'Logits' and self._spatial_squeeze:
            if x.dim() > 4:
                x = torch.squeeze(x, 4)
            if x.dim() > 3:
                x = torch.squeeze(x, 3)
        elif self._final_endpoint == 'Predictions':
            x = x.mean(2)

        return x
