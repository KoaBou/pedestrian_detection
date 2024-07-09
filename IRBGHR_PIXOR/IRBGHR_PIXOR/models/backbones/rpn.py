import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...torchplus import Sequential, Empty, change_default_args

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, padding = 0, dilation = 1, groups = 1, pooling_r = 4, norm_layer = Empty):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out


class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 layer_nums=(8, 8),
                 layer_strides=(1, 2),
                 num_filters=(32, 64),
                 upsample_strides=(1, 2),
                 num_upsample_filters=(64, 64),
                 num_input_features=35):

        super(RPN, self).__init__()

        assert len(layer_nums) == 2
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        # upsample_strides = [
        #     np.round(u).astype(np.int64) for u in upsample_strides
        # ]
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(
                layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(
                np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)


        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        block1 = [
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_features, num_filters[0], 3,
                stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        ]
        for i in range(layer_nums[0]):
            block1.append(
                SCConv(num_filters[0], num_filters[0], padding=1, norm_layer=BatchNorm2d))
            #self.block1.add(BatchNorm2d(num_filters[0]))
            block1.append(nn.ReLU())
        deconv1 = [
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        ]
        block2 = [
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        ]
        for i in range(layer_nums[1]):
            block2.append(
                SCConv(num_filters[1], num_filters[1], padding=1, norm_layer=BatchNorm2d))
            block2.append(nn.ReLU())
        deconv2 = [
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        ]

        self.block1 = nn.Sequential(*block1)
        self.deconv1 = nn.Sequential(*deconv1)
        self.block2 = nn.Sequential(*block2)
        self.deconv2 = nn.Sequential(*deconv2)
        

    def forward(self, x):
        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        
        x = torch.cat([up1, up2], dim=1)
        return x