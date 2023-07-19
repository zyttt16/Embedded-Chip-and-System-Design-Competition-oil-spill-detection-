"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""

import torch
import torch.nn.functional as F
from torch import nn
# from network import SEresnext
# from network import Resnet
# from network.wider_resnet import wider_resnet38_a2
# from config import cfg
from torch.autograd import Variable
# from .vit_seg_modeling_resnet_skip import Resnet_BasicBlock
from . import GatedSpatialConv as gsc
# from .vit_seg_modeling import Conv2dReLU


import cv2
import numpy as np


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        # if output_stride == 8:
        #     rates = [2 * r for r in rates]
        # elif output_stride == 16:
        #     pass
        # else:
        #     raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))

        self.out = nn.Conv2d(1536,1024, kernel_size=1, bias=False)

    def forward(self, x, edge):
        # x -->feat 1024 16 16, edge-->GCL output 1 256 256
        x_size = x.size()

        img_features = self.img_pooling(x)    # 2 512 1 1
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:], mode='bilinear', align_corners=True)
        cnn_feats = img_features   # 768 16 16

        edge_features = F.interpolate(edge, x_size[2:], mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)

        out = torch.cat((cnn_feats, edge_features), 1)  # chn 768*2

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)

        out = self.out(out)
        return out


class _fusion(nn.Module):
    def __init__(self, num_classes):
        super(_fusion, self).__init__()
        self.aspp = _AtrousSpatialPyramidPoolingModule(512, 256)

        self.conv_aspp = nn.Conv2d(256*2 + 256*4, 256, kernel_size=1, bias=False)
        self.conv_feats64 = nn.Conv2d(64, 64, kernel_size=3,  padding=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(256 + 64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.final_seg)

    def forward(self, edge_cat, feats):
        # aspp
        aspp_out = self.aspp(feats[3], edge_cat)
        conv_aspp = self.conv_aspp(aspp_out)

        conv_feats64 = self.conv_feats64(feats[0])  # 64 112 112
        conv_aspp_up = F.interpolate(conv_aspp, feats[0].size()[2:], mode='bilinear', align_corners=True)
        cat = torch.cat([conv_feats64, conv_aspp_up], 1)

        # seg head
        seg = self.final_seg(cat)
        seg_out = F.interpolate(seg, scale_factor=2, mode='bilinear')

        return seg_out
        # if self.training:
        #     return self.criterion((seg_out, edge_out), gts)
        # else:
        #     return seg_out, edge_out


def initialize_weights(*models):
       for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()


class GSCNN(nn.Module):
    '''
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7

      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    '''

    def __init__(self, num_classes, trunk=None, criterion=None):
        super(GSCNN, self).__init__()
        # self.criterion = criterion
        self.num_classes = num_classes

        # self.conv_x1 = nn.Conv2d(64, 1, 1)
        self.conv_x3 = nn.Conv2d(256, 1, 1)
        self.conv_x4 = nn.Conv2d(512, 1, 1)

        self.res1 = Resnet_BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = Resnet_BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        # self.res3 = Resnet_BasicBlock(16, 16, stride=1, downsample=None)
        # self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(8*2, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        self.sigmoid = nn.Sigmoid()

    def forward(self, features, inp, gts=None):

        x_size = inp.size()  # 256 256
        # canny edge
        im_arr = inp.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()

        # shape stream
        x1_cnn = features[2]   # 64 128 128
        # x2_cnn = features[2]  # 128 64 64
        x3_cnn = features[1]  # 256 64 64
        x4_cnn = features[0]  # 512 32 32

        x1 = F.interpolate(x1_cnn, x_size[2:], mode='bilinear', align_corners=True)  # 64 256 256
        # x2 = F.interpolate(self.conv_x2(x2_cnn), x_size[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.conv_x3(x3_cnn), x_size[2:], mode='bilinear', align_corners=True)  # 1 256 256
        x4 = F.interpolate(self.conv_x4(x4_cnn), x_size[2:], mode='bilinear', align_corners=True)

        res1 = self.res1(x1)
        res1 = F.interpolate(res1, x_size[2:], mode='bilinear', align_corners=True)
        res1 = self.d1(res1)  # chn 64 --> 32
        gate1 = self.gate1(res1, x3)   # n 32 H w

        res2 = self.res2(gate1)
        res2 = F.interpolate(res2, x_size[2:], mode='bilinear', align_corners=True)
        res2 = self.d2(res2)     # 32 --> 16
        gate2 = self.gate2(res2, x4)

        # res3 = self.res3(gate2)
        # res3 = F.interpolate(res3, x_size[2:], mode='bilinear', align_corners=True)
        # res3 = self.d3(res3)  # 16 --> 8
        # gate3 = self.gate3(res3, x4)

        edge = self.fuse(gate2)      # chn 8-->1
        edge = F.interpolate(edge, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(edge)    # n 1 h w

        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)      # n 1 h w

        return edge_out, acts  # edge_out for loss


class Resnet_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Resnet_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)   # 返回张量方差和均值
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


