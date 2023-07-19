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
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, WeightedFocalLoss,EarlyStopping
from torchvision import transforms
from datasets.own_data import ImageFolder
from thop import profile
from mine_test import mine_eval
from torch.optim import lr_scheduler

"""
No GAN NO pdf Trainer / for other models
"""


def trainer_(args, model, save_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=save_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

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



    # num_para1 = 0.0
    # for para1 in model.parameters():
    #     num_para1 += para1.numel()
    # print('Gen Network total number of parameters :  %.3f M' % (num_para1 / 1e6))

    # model.train()
    ce_loss = CrossEntropyLoss()
    focal_loss = WeightedFocalLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    writer = SummaryWriter(save_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    early_stopping = EarlyStopping(patience=4, path=save_path, verbose=True)  # 早停

    if args.resume_path is not None:
        if os.path.isfile(args.resume_path):
            logging.info("============= Continue train, loading checkpoint ============= ")

            checkpoint = torch.load(args.resume_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # dnet.load_state_dict(checkpoint['dnet'])
            # g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            # d_optimizer.load_state_dict(checkpoint['d_optimizer'])

            logging.info('============ Loading Sucessfully ============= ')

        else:
            print('not find checkpoint')
    else:
        start_epoch = 0

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    scheduler_lr = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    for epoch_num in range(start_epoch+1, start_epoch + args.max_epochs+1):
        model.train()
        logging.info('\nStarting epoch {}/{}.'.format(epoch_num, start_epoch + max_epoch))

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # if epoch_num == 1 and iter_num == 0:
            #     flops, paras = profile(model, (image_batch,))
            #     logging.info('Network total number of parameters :  %.3f M, flops: %3.f G' % (paras / 1e6, flops / 1e9))

            outputs = model(image_batch)
            loss_ce1 = ce_loss(outputs, label_batch[:].long())
            loss_focal1 = focal_loss(outputs, label_batch)
            loss_dice1 = dice_loss(outputs, label_batch, softmax=True)
            loss1 = 0.1 * loss_ce1 + 0.5 * loss_dice1 + 0.4 * loss_focal1
            # loss = 0.5 * loss_ce + 0.5 * loss_dice + 0.5 * loss_focal

            # loss_ce2 = ce_loss(PDF_out, label_batch[:].long())
            # loss_focal2 = focal_loss(PDF_out, label_batch)
            # loss_dice2 = dice_loss(PDF_out, label_batch, softmax=True)
            # loss2 = 0.1 * loss_ce2 + 0.5 * loss_dice2 + 0.4 * loss_focal2

            loss_ce = loss_ce1
            loss_dice = loss_dice1
            loss_focal = loss_focal1
            loss = loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                lr_ = param_group['lr']

            iter_num = iter_num + 1
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/total_loss', loss, iter_num)
            writer.add_scalar('train/loss_dice', loss_dice, iter_num)
            writer.add_scalar('train/loss_focal', loss_focal, iter_num)
            writer.add_scalar('train/loss_ce', loss_ce, iter_num)

            logging.info('Epoch: %d iter %d: lr: %f, loss_total : %f, loss_dice : %f, loss_focal : %f, loss_ce: %f' %
                         (epoch_num, iter_num, lr_, loss.item(), loss_dice.item(), loss_focal.item(), loss_ce.item()))

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

                    pre_val = model(image_val)

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
                    'optimizer': optimizer.state_dict(),
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
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_num,
            }

            torch.save(checkpoint, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # if epoch_num >= max_epoch - 1:
        if epoch_num >= max_epoch:
            save_mode_path = os.path.join(save_path, 'epoch_' + str(epoch_num) + '.pth')

            checkpoint = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_num,
            }
            torch.save(checkpoint, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            save_mode_path2 = os.path.join(save_path, 'epoch_latest.pth')  # save model+paras
            torch.save(model, save_mode_path2)

            # shutil.copy2(inspect.getfile(Model), save_path)

            break

        scheduler_lr.step()

    writer.close()
    return "Training Finished!"