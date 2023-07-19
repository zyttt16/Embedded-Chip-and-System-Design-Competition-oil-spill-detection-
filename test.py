import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.model_m import VisionTransformer as ViT_seg
from networks.model_m import CONFIGS as CONFIGS_ViT_seg
from datasets.own_data import ImageFolder
from mine_test import mine_eval
from models import unet,unetpp,deeplab,TransUNET,hednet,PDFNet
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='DR', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='', help='list dir')
parser.add_argument('--save_filename', type=str, default='T20-IDRID-trm3', help='save file name')
parser.add_argument('--test_epoch', type=str, default='600', help='choose which epoch model for test')

parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=600, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--is_savenii', type=bool, default=True, help='whether to save results during inference')
parser.add_argument('--is_pretrain', type=bool, default=False, help='whether pretrain')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

# parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-4, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    # db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    db_test = ImageFolder(args, split='test')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)

    logging.info("The length of val set is: {}".format(len(db_test)))
    # logging.info("{} test iterations per epoch".format(len(testloader)))

    model.eval()
    dice_ = 0.0
    hd95_ = 0.0
    mpa_ = 0.0
    miou_ = 0.0
    FWiou_ = 0.0
    se_ = 0.0
    ppv_ = 0.0
    acc_score_ = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image, label, img_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        image,label = image.cuda(), label.cuda()
        metrics = mine_eval(image, label, model, classes=args.num_classes,
                            test_save_path=test_save_path, case=img_name,)

        dice_ += metrics[0]
        hd95_ += metrics[1]
        mpa_ += metrics[3]
        miou_ += metrics[5]
        FWiou_ += metrics[6]
        se_ += metrics[7]
        ppv_ += metrics[8]
        acc_score_ += metrics[9]

        logging.info('The %dth image: %s, dice: %f, hd95: %f, se: %f,ppv: %f, miou: %f, acc: %f' %
                     (i_batch + 1, img_name, metrics[0], metrics[1], metrics[7], metrics[8], metrics[5], metrics[9]))

    dice = dice_ / len(db_test)
    hd95 = hd95_ / len(db_test)
    mpa = mpa_ / len(db_test)
    miou = miou_ / len(db_test)
    FWiou = FWiou_ / len(db_test)
    se = se_ / len(db_test)
    ppv = ppv_ / len(db_test)
    acc = acc_score_/ len(db_test)

    for i in range(1, args.num_classes):
        logging.info('Mean class %d: dice: %f, hd95: %f, mpa: %f, miou: %f, FWiou: %fc, se: %f, ppv: %f, acc: %f' %
                     (i, dice, hd95, mpa, miou, FWiou, se, ppv, acc))

    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'test': {
            'root_path': '/home/s802/zyt/data',
            'list_dir': '',
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
        'DR': {
            'root_path': '/home/s802/zyt/data/oil-spill',
            'list_dir': '',
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }

    dataset_name = args.dataset
    args.root_path = dataset_config[dataset_name]['root_path']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.volume_path = dataset_config[dataset_name]['volume_path']
    # args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']

    args.exp = dataset_name + str(args.img_size)
    save_path = "/home/s802/zyt/Result/oil-spill/nets/unet/{}/{}".format(args.save_filename, args.exp)
    # save_path = '/media/l/Data/zjm/DR/other-models/{}/{}'.format(args.save_filename, args.exp)
    # save_path = save_path + '_pretrain' if args.is_pretrain else save_path
    save_path = save_path + '_epo' + str(args.max_epochs) + '_lr' + str(args.base_lr) + 'bs' + str(args.batch_size)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)

    # if args.vit_name.find('R50') !=-1:
    #     config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    base = unet.Unet(in_channels=3, classes=args.num_classes).cuda()
    # net3 = unetpp.ResNet34UnetPlus(num_channels=3, num_class=2).cuda()
    # net4 = deeplab.DeepLabV3(in_class=3, class_num=2).cuda()
    # net7 = TransUNET.get_transNet(2).cuda()   # TransUnet

    # net8 = PDFNet.PDFNet().cuda()
    # net9 = hednet.HNNNet().cuda()

    from collections import OrderedDict

    latest_model = save_path + '/epoch_' + args.test_epoch + '.pth'
    model = torch.load(latest_model)
    new_state_dict = OrderedDict()
    for k, v in model['net'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    base.load_state_dict(new_state_dict)

    # latest_model = save_path + '/best_checkpoint.pt'
    # model = torch.load(latest_model)
    # new_state_dict = OrderedDict()
    # for k, v in model.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # net.load_state_dict(new_state_dict)

    model_name = latest_model.split('/')[-1]

    log_folder = save_path + '/test'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'test_log_s' + str(args.img_size) + '_' + model_name + ".txt",
                        level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(model_name)

    if args.is_savenii:
        test_save_path = os.path.join(log_folder, 'test_results', model_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, base, test_save_path)


