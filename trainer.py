import argparse
import logging
import os, cv2
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, WeightedFocalLoss,image_to_patch,EarlyStopping

from datasets.own_data import ImageFolder
from thop import profile
from networks.dnet import DNet
from mine_test import mine_eval

"""
GAN+PDF Trainer
"""

def trainer_(args, model, save_path):

    logging.basicConfig(filename=save_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    early_stopping = EarlyStopping(patience=4, path=save_path, verbose=True)  # 早停

    db_train = ImageFolder(args, split='train')
    logging.info("The length of train set is: {}".format(len(db_train)))

    db_val = ImageFolder(args, split='val')
    logging.info("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                           worker_init_fn=worker_init_fn)

    dnet = DNet(input_dim=3, output_dim=1, input_size=64).cuda()
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        dnet = nn.DataParallel(dnet)

    # model.train()
    ce_loss = CrossEntropyLoss()
    focal_loss = WeightedFocalLoss()
    dice_loss = DiceLoss(num_classes)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    g_optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    d_optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)


    writer = SummaryWriter(save_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    if args.resume_path is not None:
        if os.path.isfile(args.resume_path):
            logging.info("============= Continue train, loading checkpoint ============= ")

            checkpoint = torch.load(args.resume_path)
            start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['net'])
            dnet.load_state_dict(checkpoint['dnet'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer'])

            logging.info('============ Loading Sucessfully ============= ')

        else:
            print('not find checkpoint')
    else:
        start_epoch = 0

    g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=200, gamma=0.9)
    d_scheduler = lr_scheduler.StepLR(d_optimizer, step_size=100, gamma=0.9)

    for epoch_num in range(start_epoch+1, start_epoch + args.max_epochs + 1):
        logging.info('\nStarting epoch {}/{}.'.format(epoch_num, start_epoch + max_epoch))
        model.train()
        dnet.train()

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # if epoch_num == 1 and iter_num == 0:
            #     flops, paras = profile(model, (image_batch,))
            #     logging.info('Network total number of parameters :  %.3f M, flops: %3.f G' % (paras / 1e6, flops / 1e9))

            outputs, PDF_out = model(image_batch)
            loss_ce1 = ce_loss(outputs, label_batch[:].long())
            loss_focal1 = focal_loss(outputs, label_batch)
            loss_dice1 = dice_loss(outputs, label_batch, softmax=True)
            loss1 = 0.1 * loss_ce1 + 0.5 * loss_dice1 + 0.4 * loss_focal1
            # loss = 0.5 * loss_ce + 0.5 * loss_dice + 0.5 * loss_focal

            loss_ce2 = ce_loss(PDF_out, label_batch[:].long())
            loss_focal2 = focal_loss(PDF_out, label_batch)
            loss_dice2 = dice_loss(PDF_out, label_batch, softmax=True)
            loss2 = 0.1 * loss_ce2 + 0.5 * loss_dice2 + 0.4 * loss_focal2

            loss_ce = loss_ce1 + loss_ce2
            loss_dice = loss_dice1 + loss_dice2
            loss_focal = loss_focal1 + loss_focal2
            loss = loss1 + loss2

            # discriminator
            dnet_path_size = 128
            input_real = torch.matmul(image_batch, label_batch.unsqueeze(1))  # 2 3 256 256
            input_real = image_to_patch(input_real, dnet_path_size)  # 16 3 64 64
            input_fake = torch.matmul(image_batch, torch.sigmoid(outputs)[:, 1:, :, :])  # 2 3 256 256
            input_fake = image_to_patch(input_fake, dnet_path_size)  # 16 3 64 64

            # input_real = torch.cat([image_batch, label_batch.unsqueeze(1)], 1)  # 2 3 256 256
            # input_real = image_to_patch(input_real, dnet_path_size)  # 32 3 64 64
            # input_fake = torch.cat([image_batch,  torch.sigmoid(outputs)[:, 1:, :, :]], 1)  # 2 3 256 256
            # input_fake = image_to_patch(input_fake, dnet_path_size)  # 32 3 64 64

            d_real = dnet(input_real)  # 32 1           16 256
            d_fake = dnet(input_fake.detach())  # 32 1      16 256
            d_real_loss = torch.mean(1 - d_real)
            d_fake_loss = torch.mean(d_fake)

            # update D loss
            loss_d = d_real_loss + d_fake_loss
            d_optimizer.zero_grad()
            loss_d.backward()
            # d_optimizer.step()

            # update G loss
            d_fake = dnet(input_fake)  # do backward to generator    16 256
            loss_gan = torch.mean(1 - d_fake)
            loss += (loss_gan * 0.01).item()

            g_optimizer.zero_grad()
            loss.backward()
            g_optimizer.step()
            d_optimizer.step()

            torch.cuda.empty_cache()
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_
            lr_ = g_optimizer.state_dict()['param_groups'][0]['lr']
            d_lr = d_optimizer.state_dict()['param_groups'][0]['lr']

            iter_num = iter_num + 1
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/lr_d', d_lr, iter_num)
            writer.add_scalar('train/total_loss', loss, iter_num)
            writer.add_scalar('train/loss_dice', loss_dice, iter_num)
            writer.add_scalar('train/loss_focal', loss_focal, iter_num)
            writer.add_scalar('train/loss_ce', loss_ce, iter_num)

            logging.info('Epoch: %d iter %d: lr: %f, lr_d: %f, loss_total : %f, loss_dice : %f, loss_focal : %f, loss_ce: %f' %
                         (epoch_num, iter_num, lr_, d_lr, loss.item(), loss_dice.item(), loss_focal.item(), loss_ce.item()))

            # ################ save to file
            if iter_num % 100 == 0:
                if not os.path.exists(save_path + '/train_pre'):
                    os.makedirs(save_path + '/train_pre')

                logits = torch.argmax(torch.sigmoid(outputs), dim=1, keepdim=True)
                logits = logits[0, ...]
                img = logits.mul(255).byte()
                img = img.cpu().numpy().transpose((1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_path + '/train_pre/' + 'epo_' + str(epoch_num) + '_iter_' + str(iter_num) + '_pre.png',
                            img)

                label = label_batch[0].unsqueeze(0)
                label = label.mul(255).byte()
                label = label.cpu().numpy().transpose((1, 2, 0))
                label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_path + '/train_pre/' + 'epo_' + str(epoch_num) + '_iter_' + str(iter_num) + '_label.png',
                            label)

            # ################ save to log web
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.sigmoid(outputs), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        if epoch_num % 20 == 0:
            logging.info('\n ===================== Eval Model ===================== \n')

            dice_ = 0.0
            hd95_ = 0.0
            mpa_ = 0.0
            miou_ = 0.0
            FWiou_ = 0.0
            with torch.no_grad():
                model.eval()
                for i_batch_val, sampled_batch_val in enumerate(valloader):
                    image_val, label_val, img_name_val = sampled_batch_val["image"], sampled_batch_val["label"], \
                                                         sampled_batch_val['case_name'][0]
                    image_val, label_val = image_val.cuda(), label_val.cuda()

                    pre_val, _ = model(image_val)

                    val_loss_ce = ce_loss(pre_val, label_val[:].long())
                    val_loss_focal = focal_loss(pre_val, label_val)
                    val_loss_dice = dice_loss(pre_val, label_val, softmax=True)
                    val_loss = 0.1 * val_loss_ce + 0.5 * val_loss_dice + 0.4 * val_loss_focal

                    writer.add_scalar('val/loss', val_loss, epoch_num)
                    val_save_path = save_path + '/val_pre'
                    if not os.path.exists(val_save_path):
                        os.makedirs(val_save_path)

                    metrics = mine_eval(image_val, label_val, model, classes=args.num_classes,
                                        test_save_path=val_save_path, case=img_name_val, )

                    dice_ += metrics[0]
                    hd95_ += metrics[1]
                    mpa_ += metrics[3]
                    miou_ += metrics[5]
                    FWiou_ += metrics[6]

                    logging.info('The %dth image: %s, dice: %f, hd95: %f, mpa: %f, miou: %f, FWiou: %f' %
                                 (i_batch_val + 1, img_name_val, metrics[0], metrics[1], metrics[3], metrics[5],
                                  metrics[6]))

                dice = dice_ / len(db_val)
                hd95 = hd95_ / len(db_val)
                mpa = mpa_ / len(db_val)
                miou = miou_ / len(db_val)
                FWiou = FWiou_ / len(db_val)
                for i in range(1, args.num_classes):
                    logging.info('Mean class %d: dice: %f, hd95: %f, mpa: %f, miou: %f, FWiou: %f' %
                                 (i, dice, hd95, mpa, miou, FWiou))


                checkpoint = {
                    'net': model.state_dict(),
                    'dnet': dnet.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'epoch': epoch_num,
                }
                early_stopping(val_loss, model, checkpoint)
                if early_stopping.early_stop:
                    logging.info('==================== 此时早停！==================== ')
                #     break

        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % args.save_interval == 0:
        if epoch_num % args.save_interval == 0:
            save_mode_path = os.path.join(save_path, 'epoch_' + str(epoch_num) + '.pth')

            checkpoint = {
                'net': model.state_dict(),
                'dnet': dnet.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch_num,
            }

            torch.save(checkpoint, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # if epoch_num >= max_epoch - 1:
        if epoch_num >= max_epoch:
            save_mode_path = os.path.join(save_path, 'epoch_' + str(epoch_num) + '.pth')

            checkpoint = {
                'net': model.state_dict(),
                'dnet': dnet.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch_num,
            }
            torch.save(checkpoint, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            save_mode_path2 = os.path.join(save_path, 'epoch_latest.pth')  # save model+paras
            torch.save(checkpoint, save_mode_path2)

            # shutil.copy2(inspect.getfile(Model), save_path)

            break

        g_scheduler.step()
        d_scheduler.step()

    writer.close()
    return "Training Finished!"