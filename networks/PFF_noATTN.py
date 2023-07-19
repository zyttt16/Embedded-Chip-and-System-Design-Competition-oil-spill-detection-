import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.nn import Conv2d
from .CBAM import *
from .PAM import *
"""
No ATTN
"""

class SeparableConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, bias=False):
		super(SeparableConv2d, self).__init__()

		self.depth_wise = Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, groups=in_channels, bias=bias)

		self.point_wise = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

		self.depth_wise.cuda()
		self.point_wise.cuda()

	def forward(self, x):
		x = x.cuda()
		x = self.depth_wise(x)
		x = self.point_wise(x)
		return x


class DepthwiseConv(nn.Module):
	def __init__(self, inp, oup):
		super(DepthwiseConv, self).__init__()
		self.depth_conv = nn.Sequential(
			# dw
			nn.Conv2d(inp, inp, kernel_size=2, stride=2, padding=0, groups=inp, bias=False),
			nn.BatchNorm2d(inp),
			nn.ReLU6(inplace=True),
			# pw
			nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(oup),
			nn.ReLU6(inplace=True)
		)

	def forward(self, x):
		return self.depth_conv(x)


class PFF_Block(nn.Module):
	def __init__(self, c_pre, c_cur, c_nex):
		super(PFF_Block, self).__init__()

		self.DWC = SeparableConv2d(c_pre, c_cur)
		self.conv2d_DW = nn.Conv2d(c_cur, c_cur, 1)

		self.conv2d_UP = nn.Conv2d(c_nex, c_cur, kernel_size=1)
		self.up = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv2d_concat = nn.Conv2d(c_cur * 2, c_cur, kernel_size=1)

		self.attn = attn(chn=c_cur, ks=3)
		self.conv2d_out = nn.Conv2d(c_cur, c_cur, kernel_size=1)
		self.relu = nn.LeakyReLU(0.2)

	def PFF_3(self, pre, cur):
		# pre f2 2 256 28 28, cur f3 2 512 14 14  trm output 2 256 768

		# process pre layer(f2)
		DWC = self.DWC(pre)
		conv1x1 = self.conv2d_DW(DWC)

		# concat f2 f3
		mul = torch.mul(conv1x1, cur)
		concat = torch.cat([cur, mul], 1)
		concat = self.conv2d_concat(concat)  # 2 512 16 16

		# # attn
		att = self.attn(concat, cur)
		out = self.conv2d_out(att)
		out = self.relu(out)

		return out

	def PFF_1_2(self, pre, cur, nex):  # 256 512 1024
		# pre f1 2 128 56 56, cur f2 2 256 28 28, nex f3 2 512 14 14
		# f0 f1 f2
		# process pre layer
		low_DWC = self.DWC(pre)
		low_conv1x1 = self.conv2d_DW(low_DWC)

		# process nex layer
		high_conv1x1 = self.conv2d_UP(nex)
		high_up = self.up(high_conv1x1)

		# mul and concat
		mul_high = torch.mul(low_conv1x1, cur)
		mul_low = torch.mul(high_up, cur)
		concat = torch.cat([mul_high, mul_low], 1)
		concat = self.conv2d_concat(concat)

		# # attn
		# att = self.attn(concat, cur)
		# out = self.conv2d_out(att)
		# out = self.relu(out)
		out = self.relu(concat)
		return out

	def PFF_0(self, cur, nex):
		# cur f0 2 64 112 112, nex f1 2 128 56 56
		# process nex layer(f1)
		conv1x1 = self.conv2d_UP(nex)
		up = self.up(conv1x1)

		# concat f0 f1
		mul = torch.mul(up, cur)
		concat = torch.cat([cur, mul], 1)
		concat = self.conv2d_concat(concat)

		# # attn
		# att = self.attn(concat, cur)
		# out = self.conv2d_out(att)
		# out = self.relu(out)
		out = self.relu(concat)
		return out


class attn(nn.Module):
	def __init__(self, chn, ks):
		super(attn, self).__init__()
		self.CBAM = CBAMBlock(channel=chn, reduction=16, kernel_size=ks)
		# self.PA = PAMBlock(d_model=chn)
		self.ARL = ARL_Block(chn)  # PAM.py
		self.PA = nn.Sequential(
			# nn.Conv2d(chn, chn, kernel_size=1, stride=1, padding=0),
			# nn.Sigmoid()
			nn.Conv2d(chn, chn // 4, kernel_size=1),
			nn.BatchNorm2d(chn // 4),
			nn.ReLU(True),

			nn.Conv2d(chn // 4, chn, kernel_size=1),
			nn.BatchNorm2d(chn),
			nn.Sigmoid(),
		)
		# self.attn_self = self_attn(chn)

	def forward(self, concat, cur):
		res = cur
		# Attn_self = self.attn_self(concat)

		# CBAM (CA + SA)
		CBAM = self.CBAM(concat)

		# PA
		PA = self.PA(concat)

		# ARL
		ARL = self.ARL(concat)

		out = (CBAM + PA + ARL) * res + res
		return out
