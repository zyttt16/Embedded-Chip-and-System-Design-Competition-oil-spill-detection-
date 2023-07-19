# import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class look0(nn.Module):
  def __init__(self, in_filters, dilation_rate):
    super(look0,self).__init__()
    self.conv_main = nn.Sequential(nn.Conv2d(in_filters, 32, kernel_size=3, stride =1,padding=dilation_rate, dilation=dilation_rate, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
    self.conv_side =  nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=dilation_rate, dilation=dilation_rate,groups=32, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=dilation_rate, dilation=dilation_rate,groups=32, bias=False),
                                    nn.BatchNorm2d(32))
    self.AB = nn.Sequential(nn.ReLU())
                                

  def forward(self,x):
    residual = self.conv_main(x)
    x = self.conv_side(residual)
    x = x + residual
    x = self.AB(x)

    return x


class look(nn.Module):
  def __init__(self, in_filters, dilation_rate):
    super(look,self).__init__()
    self.conv_main = nn.Sequential(nn.Conv2d(in_filters, in_filters, kernel_size=3, stride =1,padding=dilation_rate, dilation=dilation_rate,groups=in_filters, bias=False),
                                   nn.BatchNorm2d(in_filters),
                                   nn.ReLU(),
                                   nn.Conv2d(in_filters, 32, kernel_size=1, stride =1,bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU()
                                   )
    self.conv_side =  nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=dilation_rate, dilation=dilation_rate,groups=32, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, kernel_size=3, stride =1,padding=dilation_rate, dilation=dilation_rate,groups=32, bias=False),
                                    nn.BatchNorm2d(32))
    self.AB = nn.Sequential(nn.ReLU())
                                

  def forward(self,x):
    residual = self.conv_main(x)
    x = self.conv_side(residual)
    x = x + residual
    x = self.AB(x)

    return x


class PDFNet12(nn.Module):
  def __init__(self, num_class=2):
    super(PDFNet12, self).__init__()

    self.glance11 = look0(in_filters=3, dilation_rate=1)
    self.glance12 = look(in_filters=32, dilation_rate=1)

    self.s1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, stride =1,bias=False),
                            nn.BatchNorm2d(32),
                            nn.ReLU())

    self.pool = nn.AvgPool2d(kernel_size=2)

    self.glance21 = look(in_filters=64, dilation_rate=1)
    self.glance22 = look(in_filters=96, dilation_rate=2)
    self.glance23 = look(in_filters=128, dilation_rate=3)
    
    
    self.s2 = nn.Sequential(nn.Conv2d(160, 64, kernel_size=1, stride =1,bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
    
    self.glance31 = look(in_filters=160, dilation_rate=1)
    self.glance32 = look(in_filters=192, dilation_rate=2)
    self.glance33 = look(in_filters=224, dilation_rate=3)
    self.glance34 = look(in_filters=256, dilation_rate=1)
    self.glance35 = look(in_filters=288, dilation_rate=2)
    self.glance36 = look(in_filters=320, dilation_rate=3)
    self.glance37 = look(in_filters=352, dilation_rate=1)
    self.glance38 = look(in_filters=384, dilation_rate=2)
    self.glance39 = look(in_filters=416, dilation_rate=3)
    self.glance310 = look(in_filters=448, dilation_rate=1)
    self.glance311 = look(in_filters=480, dilation_rate=2)
    self.glance312 = look(in_filters=512, dilation_rate=3)

    self.s3 = nn.Sequential(nn.Conv2d(544, 128, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU())
    
    self.glance41 = look(in_filters=544, dilation_rate=1)
    self.glance42 = look(in_filters=576, dilation_rate=2)
    self.glance43 = look(in_filters=608, dilation_rate=3)
    self.glance44 = look(in_filters=640, dilation_rate=1)
    self.glance45 = look(in_filters=672, dilation_rate=2)
    self.glance46 = look(in_filters=704, dilation_rate=3)
    self.glance47 = look(in_filters=736, dilation_rate=1)
    self.glance48 = look(in_filters=768, dilation_rate=2)
    self.glance49 = look(in_filters=800, dilation_rate=3)
    self.glance410 = look(in_filters=832, dilation_rate=1)
    self.glance411 = look(in_filters=864, dilation_rate=2)
    self.glance412 = look(in_filters=896, dilation_rate=3)

    self.s4 = nn.Sequential(nn.Conv2d(928, 256, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU())

    self.glance51 = look(in_filters=928, dilation_rate=1)
    self.glance52 = look(in_filters=960, dilation_rate=2)
    self.glance53 = look(in_filters=992, dilation_rate=3)
    self.glance54 = look(in_filters=1024, dilation_rate=1)
    self.glance55 = look(in_filters=1056, dilation_rate=2)
    self.glance56 = look(in_filters=1088, dilation_rate=3)
    self.glance57 = look(in_filters=1120, dilation_rate=1)
    self.glance58 = look(in_filters=1152, dilation_rate=2)
    self.glance59 = look(in_filters=1184, dilation_rate=3)
    self.glance510 = look(in_filters=1216, dilation_rate=1)
    self.glance511 = look(in_filters=1248, dilation_rate=2)
    self.glance512 = look(in_filters=1280, dilation_rate=3)



    self.s5 = nn.Sequential(nn.Conv2d(1312, 512, kernel_size=1, stride =1,bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU())

    self.out = nn.Sequential(nn.Conv2d(32+64+128+256+512, num_class, kernel_size=1, stride =1, bias=False))
    self.cv = nn.Sequential(
      nn.Conv2d(1312, 1024, kernel_size=1, stride=1, bias=False),
      nn.BatchNorm2d(1024),
      nn.ReLU()
    )

  def forward(self, x):  # bs 3 256 256

    # Stem module

    c = self.glance11(x)               # 32 256 256
    
    R = self.glance12(c)               # 32 256 256
    c = torch.cat((c, R), dim=1)       # 64 256 256

    s1 = self.s1(c)       # 32 256 256

    ###################################
    c = self.pool(c)     # 64 128 128

    ############### feature 1
    R = self.glance21(c)   # 32 128 128
    c = torch.cat((c, R), dim=1)
    
    R = self.glance22(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance23(c)
    c = torch.cat((c, R), dim=1)   # 160 128 128

    s2 = self.s2(c)  # 64 128 128

    ###################################

    s2 = torch.nn.functional.interpolate(s2,  (x.size()[2:]), mode='bilinear', align_corners=True)

    c = self.pool(c)

    ######################### feature 2
    R = self.glance31(c)       # 32 64 64
    c = torch.cat((c, R), dim=1)
    
    R = self.glance32(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance33(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance34(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance35(c)
    c = torch.cat((c, R), dim=1)
  
    R = self.glance36(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance37(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance38(c)
    c = torch.cat((c, R), dim=1)
  
    R = self.glance39(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance310(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance311(c)
    c = torch.cat((c, R), dim=1)
  
    R = self.glance312(c)
    c = torch.cat((c, R), dim=1)

    s3 = self.s3(c)     # 128 64 64

    ###################################

    s3 = torch.nn.functional.interpolate(s3,  (x.size()[2:]), mode='bilinear', align_corners=True)
  
    c = self.pool(c)

    #################### feature 3
    R = self.glance41(c)
    c = torch.cat((c, R), dim=1)
    
    R = self.glance42(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance43(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance44(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance45(c)
    c = torch.cat((c, R), dim=1)
  
    R = self.glance46(c)
    c = torch.cat((c, R), dim=1)
  
    R = self.glance47(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance48(c)
    c = torch.cat((c, R), dim=1)
  
    R = self.glance49(c)
    c = torch.cat((c, R), dim=1)
  
    R = self.glance410(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance411(c)
    c = torch.cat((c, R), dim=1)
  
    R = self.glance412(c)
    c = torch.cat((c, R), dim=1)

    s4 = self.s4(c)

    ###################################

    s4 = torch.nn.functional.interpolate(s4,  (x.size()[2:]), mode='bilinear',align_corners=True)

    c = self.pool(c)

    # #########################feature 4
    R = self.glance51(c)
    c = torch.cat((c, R), dim=1)
    
    R = self.glance52(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance53(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance54(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance55(c)
    c = torch.cat((c, R), dim=1)
  
    R = self.glance56(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance57(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance58(c)
    c = torch.cat((c, R), dim=1)
  
    R = self.glance59(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance510(c)
    c = torch.cat((c, R), dim=1)

    R = self.glance511(c)
    c = torch.cat((c, R), dim=1)
  
    R = self.glance512(c)
    c = torch.cat((c, R), dim=1)   # 1312 16 16

    s5 = self.s5(c)   # 512 16 16

    ###################################

    s5 = torch.nn.functional.interpolate(s5,  (x.size()[2:]), mode='bilinear', align_corners=True)
    
    R = torch.cat((s1, s2, s3, s4, s5), dim=1)
    PDF_OUT = self.out(R)

    pdf_trm = self.cv(c)

    return PDF_OUT, pdf_trm












# model = PDFNet()

#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model=model.to(device)
#
# criterion = nn.CrossEntropyLoss(reduction="sum")
# from torch import optim
# opt = optim.SGD(model.parameters(), lr=1e-6, momentum=0.7,nesterov=True)
#
# def loss_batch(loss_func, output, target, opt=None):
#     loss = loss_func(output, target)
#
#     if opt is not None:
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#
#     return loss.item(), None
#
#
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)
#
# def get_lr(opt):
#     for param_group in opt.param_groups:
#         return param_group['lr']
#
# current_lr=get_lr(opt)
# print('current lr={}'.format(current_lr))
#
# loss_history={"train": [],"val": []}
#
# import shutil
# def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
#     f_path = checkpoint_dir +'checkpoint.pt'
#     torch.save(state, f_path)
#     if is_best:
#         best_fpath = best_model_dir +'best_model.pt'
#         shutil.copyfile(f_path, best_fpath)
#
# checkpoint_dir = "./PDFNet12-2/"
# best_model_dir = "./PDFNet12-2/"
#
# def load_ckp(checkpoint_fpath, model, opt,lr_scheduler):
#     checkpoint = torch.load(checkpoint_fpath)
#     model.load_state_dict(checkpoint['state_dict'])
#     opt.load_state_dict(checkpoint['optimizer'])
#     lo = (checkpoint['loss'])
#     lr_scheduler.load_state_dict(checkpoint['lr'])
#
#     return model, opt, checkpoint['epoch'],lo,lr_scheduler
#
# ckp_path = "./PDFNet12-2/checkpoint.pt"
#
# model, opt, start_epoch, loss_history, lr_scheduler = load_ckp(ckp_path, model, opt,lr_scheduler)
#
#
#
# def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
#     running_loss=0.0
#     len_data=len(dataset_dl.dataset)
#     for xb, yb in dataset_dl:
#         xb=xb.to(device)
#         yb=yb.to(device)
#         output=model(xb)
#         loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
#         running_loss += loss_b
#         if sanity_check is True:
#             break
#     loss=running_loss/float(len_data)
#     return loss, None
#
# import copy
# def train_val(model, params):
#     num_epochs=params["num_epochs"]
#     loss_func=params["loss_func"]
#     opt=params["optimizer"]
#     train_dl=params["train_dl"]
#     val_dl=params["val_dl"]
#     sanity_check=params["sanity_check"]
#     lr_scheduler=params["lr_scheduler"]
#     path2weights=params["path2weights"]
#
#     #loss_history={"train": [],"val": []}
#
#     #best_model_wts = copy.deepcopy(model.state_dict())
#     best_model_wts = copy.deepcopy(torch.load("./PDFNet12-2/PDFNet12-2_weights.pt"))
#
#     best_loss= min(loss_history["val"])
#     #best_loss=float('inf')
#
#     for epoch in range(start_epoch,num_epochs):
#     #for epoch in range(num_epochs):
#         current_lr=get_lr(opt)
#         o = open('./PDFNet12-2/PDFNet12-2.txt','a')
#
#         print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr), file=o)
#
#
#         print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
#
#         model.train()
#         train_loss, _ = loss_epoch(model,loss_func,train_dl,sanity_check,opt)
#         loss_history["train"].append(train_loss)
#
#         model.eval()
#         with torch.no_grad():
#             val_loss, _ = loss_epoch(model,loss_func,val_dl,sanity_check)
#         loss_history["val"].append(val_loss)
#
#         if val_loss < best_loss:
#             best_loss = val_loss
#             best_model_wts = copy.deepcopy(model.state_dict())
#             torch.save(model.state_dict(), path2weights)
#             print("Copied best model weights!")
#             print("Copied best model weights!",file=o)
#             is_best = True
#         else:
#             is_best = False
#
#         lr_scheduler.step(val_loss)
#         if current_lr != get_lr(opt):
#             print("Loading best model weights!")
#             print("Loading best model weights!",file=o)
#             model.load_state_dict(best_model_wts)
#
#         print("train loss: %.6f" %(train_loss))
#         print("train loss: %.6f" %(train_loss),file=o)
#         print("val loss: %.6f" %(val_loss))
#         print("val loss: %.6f" %(val_loss),file=o)
#         print("-"*10)
#         print("-"*10,file=o)
#         o.close()
#         checkpoint = {'epoch': epoch + 1,
#                       'state_dict': model.state_dict(),
#                       'optimizer': opt.state_dict(),
#                       'loss' : loss_history,
#                       'lr': lr_scheduler.state_dict()
#                      }
#         save_ckp(checkpoint, is_best, checkpoint_dir,best_model_dir)
#         print("true")
#
#
#     model.load_state_dict(best_model_wts)
#
#     return model, loss_history
#
#
#
# start = time.time()
#
# import os
# path2models= "./PDFNet12-2/PDFNet12-2_"
# if not os.path.exists(path2models):
#         os.mkdir(path2models)
# params_train={
#     "num_epochs": 180,
#     "optimizer": opt,
#     "loss_func": criterion,
#     "train_dl": train_dl,
#     "val_dl": val_dl,
#     "sanity_check": False,
#     "lr_scheduler": lr_scheduler,
#     "path2weights": path2models+"weights.pt",}
# model,loss_hist=train_val(model,params_train)
#
# end = time.time()
# o = open('./PDFNet12-2/PDFNet12-2-time.txt','a')
#
# print("TIME TOOK {:3.2f}MIN".format((end - start )/60), file=o)
#
# o.close()
#
# print("TIME TOOK {:3.2f}MIN".format((end - start )/60))
#
#
# num_epochs=params_train["num_epochs"]
# plt.figure(figsize=(30,30))
# plt.title("Train-Val Loss")
# plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
# plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
# plt.ylabel("Loss")
# plt.xlabel("Training Epochs")
# plt.legend()
# plt.savefig('./PDFNet12-2/PDFNet12-2.png', dpi = 300)
#
# a = loss_hist["train"]
# A = [int(x) for x in a]
# b = loss_hist["val"]
# B = [int(x) for x in b]
#
# import csv
#
# with open('./PDFNet12-2/PDFNet12-2.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(zip(A,B))
#
#
#
