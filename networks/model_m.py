# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from .PDFNet12 import PDFNet12
from .PFF import *
from .gscnn import GSCNN,_AtrousSpatialPyramidPoolingModule

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])  # 256/16/16 (1 1)
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)  # 16 16
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])   # (16*16) 256
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16   # 64*16 1024
            self.PDFNet = PDFNet12(num_class=2)

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,  # 1
                                       stride=patch_size)        # 1
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))  # 1 256 768

        self.dropout = Dropout(config.transformer["dropout_rate"])
        # self.GSCNN = GSCNN(config.num_classes)
        # self.aspp = _AtrousSpatialPyramidPoolingModule(1024)

        self.PFF1 = PFF_Block(512, 1024, 3)
        self.PFF2 = PFF_Block(256, 512, 1024)
        self.PFF3 = PFF_Block(64, 256, 512)
        self.PFF4 = PFF_Block(3, 64, 256)

    def forward(self, x):  # 2 3 256 256
        inp = x
        if self.hybrid:
            x, features = self.hybrid_model(inp)  # 1024 16 16 [features] 512 32 / 256 64 / 64 128
            f1 = x
            PDF_out, x2 = self.PDFNet(inp)
            # PDF_out = 0
        else:
            features = None

        # _, edge_cat = self.GSCNN(features, inp)  # n 1 h w
        # aspp_o = self.aspp(f1, edge_cat)
        #
        # x = aspp_o + x

        # x = x + x2
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2)) 768 16 16
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)  256 768

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)   # 256 768

        # Attn
        features_attn = []
        f2, f3, f4 = features[0], features[1], features[2]
        # 512 256 64
        PFF2 = self.PFF2.PFF_1_2(f3, f2, f1)  # 512 32 32
        features_attn.append(PFF2)
        PFF3 = self.PFF3.PFF_1_2(f4, f3, f2)  # 256 64 64
        features_attn.append(PFF3)
        PFF4 = self.PFF4.PFF_0(f4, f3)  # 64 128 128
        features_attn.append(PFF4)

        # features_attn.insert(0,f1)

        return embeddings, features_attn, PDF_out


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features, PDF_out = self.embeddings(input_ids)  # 256 768 features 512 256 64
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)  attn_wei []
        return encoded, attn_weights, features, PDF_out


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x
        # 512 32 -- 256 32 32 -- 128 64 64 -- 64 128 -- 16 256 256
        #


class Decoder_(nn.Module):
    def __init__(self, config):
        super(Decoder_, self).__init__()
        # self.channels = [512, 256, 128, 64]
        self.channels = [1024, 512, 256, 64]
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            self.channels[0],
            kernel_size=1,
            padding=0,
            use_batchnorm=True)

        self.conv_more1 = Conv2dReLU(
            self.channels[0],
            self.channels[0],
            kernel_size=3,
            padding=1,
            use_batchnorm=True)

        self.up_conv1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.channels[0], self.channels[1], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )

        self.up_conv2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.channels[1], self.channels[2], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )

        self.up_conv3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.channels[2], self.channels[3], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )

        self.conv_block4 = nn.Sequential(
            Conv2dReLU(self.channels[0], self.channels[0] // 2, kernel_size=1, use_batchnorm=True),
            Conv2dReLU(self.channels[0] // 2, self.channels[0], kernel_size=1, use_batchnorm=True)
        )

        self.conv_block3 = nn.Sequential(
            Conv2dReLU(self.channels[1], self.channels[1] // 2, kernel_size=1, use_batchnorm=True),
            Conv2dReLU(self.channels[1] // 2, self.channels[1], kernel_size=1, use_batchnorm=True)
        )

        self.conv_block2 = nn.Sequential(
            Conv2dReLU(self.channels[2], self.channels[2] // 2, kernel_size=1, use_batchnorm=True),
            Conv2dReLU(self.channels[2] // 2, self.channels[2], kernel_size=1, use_batchnorm=True)
        )

        self.conv_block1 = nn.Sequential(
            Conv2dReLU(self.channels[3], self.channels[3] // 2, kernel_size=1, use_batchnorm=True),
            Conv2dReLU(self.channels[3] // 2, self.channels[3], kernel_size=1, use_batchnorm=True)
        )

        self.up_f4_32 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.channels[0], self.channels[1], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.up_f4_64 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(self.channels[0], self.channels[2], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.up_f4_128 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=8),
            nn.Conv2d(self.channels[0], self.channels[3], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )

        self.up_f3_64 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.channels[1], self.channels[2], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.up_f3_128 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(self.channels[1], self.channels[3], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )

        self.up_f2_128 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.channels[2], self.channels[3], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )

    def forward(self, hidden_states, features):
        b, n_patch, hidden = hidden_states.size()  # 2 256 768 reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(b, hidden, h, w)  # 2 768 16 16
        x = self.conv_more(x)  # 2 512 16 16

        x = self.conv_more1(x)  # 1024 16 16
        add4 = x + features[0]  # 2 1024 16 16
        de4 = self.conv_block4(add4)  # 2 1024 16 16

        up_cv3 = self.up_conv1(de4)  # 2 512->256 16->32  2 256 32 32
        add3 = up_cv3 + features[1] + self.up_f4_32(features[0])  # 2 256 32 32
        de3 = self.conv_block3(add3)  # 2 256 32 32

        up_cv2 = self.up_conv2(de3)  # 2 128 64 64
        add2 = up_cv2 + features[2] + self.up_f4_64(features[0]) + self.up_f3_64(features[1])
        de2 = self.conv_block2(add2)  # 2 128 64 64

        up_cv1 = self.up_conv3(de2)  # 2 64 128 128
        add1 = up_cv1 + features[3] + self.up_f4_128(features[0]) + self.up_f3_128(features[1]) + self.up_f2_128(features[2])
        de1 = self.conv_block1(add1)  # 2 64 128 128
        de1 = F.interpolate(de1,scale_factor=2,mode='bilinear')

        return de1


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=256, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        # self.decoder = Decoder_(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            # in_channels=64,
            out_channels=config['n_classes'],
            kernel_size=3,
            # upsampling=2
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features, PDF_out = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)   # 256 768  --> 16 256 256
        logits = self.segmentation_head(x)   # 2 256 256
        return logits, PDF_out


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


