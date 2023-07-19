import numpy as np
import torch.nn as nn
from medpy import metric as mt

"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import torch
import cv2
from sklearn.metrics import accuracy_score, roc_auc_score,recall_score,f1_score

# __all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc1 = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def sensitivity(self):
        # TPR：true positive rate，描述识别出的所有正例占所有正例的比例
        # se = TP / (TP+ FN)

        se = TP / (TP + FN)  # axis 0 列和
        # se2 = self.confusionMatrix[0] / self.confusionMatrix.sum(axis=0)
        # seoup = np.nanmean(se2)

        each_sensitivity = (np.nan_to_num(np.diag(self.confusionMatrix).astype('float32') / self.confusionMatrix.sum(axis=0).astype('float32')))
        sensitivity = np.nanmean(each_sensitivity)
        return sensitivity

    # def specifity(self):
    #     #  TNR：true negative rate，描述识别出的负例占所有负例的比例
    #     # 计算公式为：sp = TN / (FP + TN)
    #     sp = TN / (FP + TN)
    #     sp2 = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)  # 列和
    #     spec = np.nanmean(sp2)
    #
    #     # each_sp = (np.nan_to_num(np.diag(self.confusionMatrix).astype('float32') / self.confusionMatrix.sum(axis=0).astype('float32')))
    #     # spec = np.nanmean(each_sp)
    #     return spec

    def positive_predictive_value(self):
        # PPV：Positive predictive value
        # 计算公式为：PPV= TP / (TP + FP)
        ppv1 = TP / (TP + FP)

        # each_ppv = (np.nan_to_num(np.diag(self.confusionMatrix).astype('float32') / self.confusionMatrix.sum(axis=1).astype('float32')))
        # ppv = np.nanmean(each_ppv)

        ppv2 = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        ppv = np.nanmean(ppv2)
        return ppv

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)  # axis 1 行和
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        """
        confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
        P\L     P    N
        P      TP    FP
        N      FN    TN
        """

        global TP, FP, FN, TN
        TP = confusionMatrix[0, 0]
        FP = confusionMatrix[0, 1]
        FN = confusionMatrix[1, 0]
        TN = confusionMatrix[1, 1]

        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def mine_eval(image, label, net, classes, test_save_path=None, case=None):  # 1 3 256 256

    net.eval()
    with torch.no_grad():
        outputs = net(image.cuda())  # 1 2 256 256
        # outputs = net(image)
    out = torch.argmax(torch.sigmoid(outputs),dim=1).squeeze(0)  # 256 256
    # torch.argmax 返回指定维度最大值的序号
    out = out.cpu().detach().numpy().astype(np.uint8)

    # label2 = label.unsqueeze(0).repeat(1,2,1,1)
    # dice2 = dice_coeff(outputs.cpu().detach(), label2)

    label = label.squeeze(0).cpu().detach().numpy().astype(np.uint8)

    if out.sum() > 0 and label.sum() > 0:
        dice = mt.binary.dc(out, label)
        hd95 = mt.binary.hd95(out, label)
    elif out.sum() > 0 and label.sum() == 0:
        dice, hd95 = 1, 0
    else:
        dice, hd95 = 0, 0

    metric = SegmentationMetric(classes)  # 2表示有2个分类，有几个分类就填几
    hist = metric.addBatch(out, label)  # 混淆矩阵
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()

    se = metric.sensitivity()
    ppv = metric.positive_predictive_value()
    acc_score = accuracy_score(label, out)

    # print('image:', case)
    # print('Dice is : ', dice)
    # print('HD95 is : ', hd95)
    # print('hist is :\n', hist)
    # print('PA is : %f' % pa)
    # print('cPA is :', cpa)  # 列表
    # print('mPA is : %f' % mpa)
    # print('IoU is : ', IoU)
    # print('mIoU is : ', mIoU)
    # print('FWIoU is : ', FWIoU)

    if test_save_path is not None:
        cv2.imwrite(test_save_path + '/' + case + '.png', out * 255)
        cv2.imwrite(test_save_path + '/' + case + '_label.png', label * 255)

        img = image.squeeze(0).permute(1,2,0).cpu().detach().numpy().astype(np.uint8)

        # image = image.squeeze(0)[0:1, :, :]
        # image = (image - image.min()) / (image.max() - image.min())
        # img = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
        cv2.imwrite(test_save_path + '/' + case + '_ori.png', img)

    return dice, hd95, cpa, mpa, IoU, mIoU, FWIoU, se, ppv, acc_score


def dice_coeff(pred, target):
    smooth = 1.
    # num = pred.size(0)
    # m1 = pred.view(num, -1)  # Flatten
    # m2 = target.view(num, -1)  # Flatten
    pred = torch.Tensor(pred)
    target = torch.Tensor(target)
    m1 = pred.view(1, -1)  # Flatten
    m2 = target.view(1, -1)  # Flatten

    intersection = (m1 * m2).sum()
    dice_c = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    return dice_c


# 测试内容
if __name__ == '__main__':
    imgPredict = cv2.imread(r'D:\01_Study\03_Task\01-DR\03-results\01-test_metric\epo_169_iter_65300_pre2.png')
    imgLabel = cv2.imread(r'D:\01_Study\03_Task\01-DR\03-results\01-test_metric\epo_169_iter_65300_label.png')
    imgPredict = np.array(cv2.cvtColor(imgPredict, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)   # 256 256
    imgLabel = np.array(cv2.cvtColor(imgLabel, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
    # imgPredict = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成预测图片
    # imgLabel = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成标注图片
    dice = dice_coeff(imgPredict, imgLabel)
    dice2 = mt.binary.dc(imgPredict, imgLabel)

    metric = SegmentationMetric(2)  # 2表示有2个分类，有几个分类就填几
    hist = metric.addBatch(imgPredict, imgLabel)   # 混淆矩阵

    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()

    ##########################
    acc_score = accuracy_score(imgLabel, imgPredict)
    f1 = f1_score(imgLabel, imgPredict, average='micro')
    # aurc = roc_auc_score(imgLabel, imgPredict)

    se = metric.sensitivity()  #
    sp = metric.specifity()
    ppv = metric.positive_predictive_value()
    mIoU = metric.meanIntersectionOverUnion()
    #############################

    IoU = metric.IntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    print('hist is :\n', hist)
    print('PA is : %f' % pa)
    print('cPA is : ', cpa)  # 列表
    print('mPA is : %f' % mpa)

    print('acc is : %f' % acc_score)
    print('se is : %f' % se)
    print('sp is : %f' % sp)
    print('ppv is : %f' % ppv)
    print('f1 score is : %f' % f1)
    print('mIoU is : ', mIoU)

    print('IoU is : ', IoU)

    print('FWIoU is : ', FWIoU)



