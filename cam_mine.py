import cv2,os
import torch
import numpy as np
import torch.nn as nn
import inspect, re


from pytorch_grad_cam.grad_cam import GradCAM
	# HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import *
from pytorch_grad_cam.utils.image import *
from torchvision.models import resnet50
from torchvision.models.segmentation import deeplabv3_resnet50
from train import *

def varname(p):
	"""
	将变量名转换成字符串
	"""
	for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
		m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
		# re.search会匹配整个字符串，并返回第一个成功的匹配。
		# span(x,y) 这个应该是对应的下标,所以我们想获取匹配的下标可以使用span
		# match 是匹配的内容, 内容为 varname(a)
	if m:
		ret = m.group(1)   # 获取匹配的内容 ,使用group(匹配的整个表达式的字符串) 返回第1组括号匹配的字符
		return ret

# opt = varname(bfs)
# print(opt)


use_cuda = torch.cuda.is_available()

image_path = '/home/user/zjm/DR/01-datasets/seg/test/images/IDRiD_74.jpg'
mask_path = '/home/user/zjm/DR/01-datasets/seg/test/labels/IDRiD_74.tif'
latest_model_path = '/home/user/zjm/DR/03-results/test256_epo1100_lr0.0001bs2/best_checkpoint.pt'
cam_save_path = '/home/user/zjm/DR/03-results/test256_epo1100_lr0.0001bs2/test/cam_results'
img_name = os.path.split(image_path)[1]

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (256, 256))
mask = np.array(mask)
mask[mask > 0] = 1  # 这里我把255转到了1
# mask = Image.fromarray(np.uint8(mask))

rgb_img = cv2.imread(image_path, 1)     # flag = 1 读入一副BGR彩色图片，忽略alpha通道
rgb_img = cv2.resize(rgb_img,(256, 256))
rgb_img = rgb_img[:, :, ::-1]            # 转换成RGB
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# model = deeplabv3_resnet50(pretrained=True)
# model.aux_classifier[-1] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
# a = model.classifier[0].convs[-1]

model = ViT_seg(CONFIGS_ViT_seg[args.vit_name], img_size=256, num_classes=2)
model_state = torch.load(latest_model_path)
model.load_state_dict(model_state, strict=False)
p1 = model.transformer.embeddings.PFF4.relu   # 64  128
p2 = model.transformer.embeddings.PFF3.relu   # 256  64
p3 = model.transformer.embeddings.PFF2.relu    # 512  32

r1 = model.transformer.embeddings.hybrid_model.root     # 64  128
r2 = model.transformer.embeddings.hybrid_model.body[0]  # 256  64
r3 = model.transformer.embeddings.hybrid_model.body[1]   # 512  32

d1 = model.decoder.blocks[1].up   # 128 64
d2 = model.decoder.blocks[2].up   # 64 128
d3 = model.decoder.blocks[3].up   # 16 256

# Before_attn  -->  model.transformer.embeddings.PFF4.conv2d_concat
# CBAM  -->  model.transformer.embeddings.PFF4.attn.CBAM
# ARL  -->  model.transformer.embeddings.PFF4.attn.ARL
# PA  -->  model.transformer.embeddings.PFF4.attn.PA
a1 = model.transformer.embeddings.PFF4.attn.PA  # 64  128
a2 = model.transformer.embeddings.PFF3.attn.PA  # 256  64
a3 = model.transformer.embeddings.PFF2.attn.PA  # 512  32

target_layers1 = [p1]
target_layers2 = [r1]
target_layers3 = [d1]
target_layers4 = [a1]

tl1_name = varname(p1)
tl2_name = varname(r1)
tl3_name = varname(d1)
tl4_name = varname(a1)

# input_tensor = torch.Tensor(0) # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam1 = GradCAM(model=model, target_layers=target_layers1, use_cuda=use_cuda)
cam2 = GradCAM(model=model, target_layers=target_layers2, use_cuda=use_cuda)
cam3 = GradCAM(model=model, target_layers=target_layers3, use_cuda=use_cuda)
cam4 = GradCAM(model=model, target_layers=target_layers4, use_cuda=use_cuda)
# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

# targets = [ClassifierOutputTarget(281)]
targets = [SemanticSegmentationTarget(1, mask)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam1 = cam1(input_tensor=input_tensor, targets=targets)
grayscale_cam2 = cam2(input_tensor=input_tensor, targets=targets)
grayscale_cam3 = cam3(input_tensor=input_tensor, targets=targets)
grayscale_cam4 = cam4(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam1 = grayscale_cam1[0, :]
visualization1 = show_cam_on_image(rgb_img, grayscale_cam1, use_rgb=True)
# visualization1  is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
visualization1 = cv2.cvtColor(visualization1, cv2.COLOR_RGB2BGR)

grayscale_cam2 = grayscale_cam2[0, :]
visualization2 = show_cam_on_image(rgb_img, grayscale_cam2, use_rgb=True)
visualization2 = cv2.cvtColor(visualization2, cv2.COLOR_RGB2BGR)

grayscale_cam3 = grayscale_cam3[0, :]
visualization3 = show_cam_on_image(rgb_img, grayscale_cam3, use_rgb=True)
visualization3 = cv2.cvtColor(visualization3, cv2.COLOR_RGB2BGR)

grayscale_cam4 = grayscale_cam4[0, :]
visualization4 = show_cam_on_image(rgb_img, grayscale_cam4, use_rgb=True)
visualization4 = cv2.cvtColor(visualization4, cv2.COLOR_RGB2BGR)

cv2.imwrite(os.path.join(cam_save_path, 'PFF', tl1_name + '_' +img_name), visualization1)
cv2.imwrite(os.path.join(cam_save_path, 'Res', tl2_name + '_' + img_name), visualization2)
cv2.imwrite(os.path.join(cam_save_path, 'Dec', tl3_name + '_' + img_name), visualization3)
cv2.imwrite(os.path.join(cam_save_path, 'PA', tl4_name + '_' + img_name), visualization4)
print('over')


