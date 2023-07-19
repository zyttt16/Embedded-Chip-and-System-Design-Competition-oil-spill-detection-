import numpy as np
import torch
from torch import nn
from torch.nn import init


class PAMBlock(nn.Module):

	def __init__(self, d_model, kernel_size=3):
		super().__init__()
		self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
		self.pa = ScaledDotProductAttention(d_model, d_k=d_model, d_v=d_model, h=1)

	def forward(self, x):
		bs, c, h, w = x.shape
		y = self.cnn(x)
		y = y.view(bs, c, -1).permute(0, 2, 1)  # bs,h*w,c
		y = self.pa(y, y, y)  # bs,h*w,c
		out = y.permute(0, 2, 1).view(bs, c, h, w)
		return out


class ScaledDotProductAttention(nn.Module):
	"""
	Scaled dot-product attention
	"""
	def __init__(self, d_model, d_k, d_v, h, dropout=.1):
		'''
	    :param d_model: Output dimensionality of the model
	    :param d_k: Dimensionality of queries and keys
	    :param d_v: Dimensionality of values
	    :param h: Number of heads
	    '''
		super(ScaledDotProductAttention, self).__init__()
		self.fc_q = nn.Linear(d_model, h * d_k)
		self.fc_k = nn.Linear(d_model, h * d_k)
		self.fc_v = nn.Linear(d_model, h * d_v)
		self.fc_o = nn.Linear(h * d_v, d_model)
		self.dropout = nn.Dropout(dropout)

		self.d_model = d_model
		self.d_k = d_k
		self.d_v = d_v
		self.h = h

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				init.constant_(m.weight, 1)
				init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight, std=0.001)
				if m.bias is not None:
					init.constant_(m.bias, 0)

	def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
		'''
	    Computes
	    :param queries: Queries (b_s, nq, d_model)
	    :param keys: Keys (b_s, nk, d_model)
	    :param values: Values (b_s, nk, d_model)
	    :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
	    :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
	    :return:
	    '''
		b_s, nq = queries.shape[:2]
		nk = keys.shape[1]

		q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
		k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
		v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

		att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
		if attention_weights is not None:
			att = att * attention_weights
		if attention_mask is not None:
			att = att.masked_fill(attention_mask, -np.inf)
		att = torch.softmax(att, -1)
		att=self.dropout(att)

		out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
		out = self.fc_o(out)  # (b_s, nq, d_model)
		return out


class ARL_Block(nn.Module):
	def __init__(self, inplanes):
		super(ARL_Block, self).__init__()
		planes = inplanes // 2

		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
		self.bn1 = nn.BatchNorm2d(planes)

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(planes)

		self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1)
		self.bn3 = nn.BatchNorm2d(inplanes)
		self.relu = nn.ReLU(inplace=True)

		self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		attn = torch.sigmoid(out) * residual * self.alpha

		out += residual + attn

		out = self.relu(out)

		return out


class self_attn(nn.Module):
	"""
	ori
	"""
	def __init__(self, in_channels, mode='hw'):
		super(self_attn, self).__init__()

		self.mode = mode

		self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
		self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
		self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

		self.gamma = nn.Parameter(torch.zeros(1))
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		batch_size, channel, height, width = x.size()

		axis = 1
		if 'h' in self.mode:
			axis *= height
		if 'w' in self.mode:
			axis *= width

		view = (batch_size, -1, axis)
		projected_query = self.query_conv(x).reshape(*view).permute(0, 2, 1)
		projected_key = self.key_conv(x).reshape(*view)

		attention_map = torch.bmm(projected_query, projected_key)
		attention = self.sigmoid(attention_map)
		projected_value = self.value_conv(x).reshape(*view)

		out = torch.bmm(projected_value, attention.permute(0, 2, 1))
		out = out.reshape(batch_size, channel, height, width)

		out = self.gamma * out + x
		return out