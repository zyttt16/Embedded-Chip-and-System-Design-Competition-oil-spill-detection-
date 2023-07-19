import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.model_m import VisionTransformer as ViT_seg
from networks.model_m import CONFIGS as CONFIGS_ViT_seg
from trainer_NO_GAN_NO_PDF import trainer_
from models import unet,unetpp,deeplab,TransUNET,hednet,PDFNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/s802/zyt/data/bijie', help='root dir for data')

parser.add_argument('--dataset', type=str,
                    default='DR', help=' test/DR experiment_name')
parser.add_argument('--save_filename', type=str,
                    default='T20-IDRID-trm3', help='save file name')

parser.add_argument('--list_dir', type=str,
                    default='', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=600, help='maximum epoch number to train')
parser.add_argument('--save_interval', type=int,
                    default=50, help='interval to save the model')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-4,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--base_size', type=int,
                    default=286, help='resize input imgs')

parser.add_argument('--seed', type=int,
                    default=0, help='random seed')

"""
seed() 用于指定随机数生成时所用算法开始的整数值。
1.如果使用相同的seed()值，则每次生成的随即数都相同；
2.如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
3.设置的seed()值一直有效
"""

parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--resume_path', type=str,
                    # default='/home/user/sjj/zjm/DR/01-results/T9_Patch128/DR256_epo1200_lr0.0001bs2/best_checkpoint.pt',
                    default=None,
                    help='continue training: load the latest model')
# '/media/l/Data/zjm/DR/13-TransUnet/T17_PDG+GAN_new/test256_epo1100_lr0.0001bs2/best_checkpoint.pt'

args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    # CUDA_VISIBLE_DEVICES = 0,1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'test': {
            'root_path': '/home/s802/zyt/data',
            'list_dir': '',
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
        'DR': {
            'root_path': '/home/s802/zyt/data/oil-spill',   #里面有三个文件夹
            'list_dir': '',
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']

    args.exp = dataset_name + str(args.img_size)
    save_path = "/home/s802/zyt/Result/oil-spill/nets/PDFnet/{}/{}".format(args.save_filename, args.exp)
    # save_path = '/media/l/Data/zjm/DR/other-models/{}/{}'.format(args.save_filename, args.exp)
    # save_path = save_path + '_pretrain' if args.is_pretrain else save_path
    save_path = save_path + '_epo' + str(args.max_epochs) + '_lr' + str(args.base_lr) + 'bs' + str(args.batch_size)

    # args.is_pretrain = True
    # args.exp = 'TU_' + dataset_name + str(args.img_size)
    # snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    # snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    # snapshot_path += '_' + args.vit_name
    # snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    # snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    # snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    # snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    # snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    # snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    # snapshot_path = snapshot_path + '_'+str(args.img_size)
    # snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.save_path = save_path
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.img_size = args.img_size
    config_vit.n_skip = args.n_skip

    # if args.vit_name.find('R50') != -1:
    #     config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # base = unet.Unet(in_channels=3, classes=args.num_classes).cuda()
    # net3 = unetpp.ResNet34UnetPlus(num_channels=3, num_class=2).cuda()
    # net4 = deeplab.DeepLabV3(in_class=3, class_num=2).cuda()
    # net7 = TransUNET.get_transNet(2).cuda()   # TransUnet

    net8 = PDFNet.PDFNet().cuda()
    # net9 = hednet.HNNNet()

    trainer = {'test': trainer_,'DR': trainer_,}
    trainer[dataset_name](args, net8, save_path)