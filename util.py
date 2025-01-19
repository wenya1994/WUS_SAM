import sys
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import os
from hausdorff import hausdorff_distance


def cross_entropy_loss_RCF(prediction, labelef, std, ada):
    label = labelef.long()

    # 确保新的张量在与 label 相同的设备上
    label = torch.where(label > 200, torch.tensor(1, dtype=torch.long, device=label.device),
                        torch.tensor(0, dtype=torch.long, device=label.device))

    mask = label.float()
    num_positive = torch.sum(mask == 1).float()
    num_negative = torch.sum(mask == 0).float()
    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)

    new_mask = mask * torch.exp(std * ada)
    cost = F.binary_cross_entropy(
        prediction, label.float(), weight=new_mask.detach(), reduction='sum')

    return cost, mask


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def compute_hausdorff(pred, gt):
    """
    计算预测结果和真实标签之间的Hausdorff距离
    """
    pred = pred.squeeze()
    gt = gt.squeeze()
    pred_points = np.argwhere(pred >= 1)
    gt_points = np.argwhere(gt == 1)
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return 0  # 在计算Hausdorff距离时，避免空预测或真值
    hausdorff_dist = hausdorff_distance(pred_points, gt_points)
    return hausdorff_dist


def get_metrics(pred, gt, all=True, threshold=0.5, smooth=1e-5):
    N = gt.shape[0]
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    SE = (TP + smooth) / (TP + FN + smooth)
    SP = (TN + smooth) / (FP + TN + smooth)
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    Acc = (TP + TN + smooth) / (TP + FP + FN + TN + smooth)
    Precision = (TP + smooth) / (TP + FP + smooth)
    Dice = 2 * TP / (FP + FN + 2 * TP + smooth)
    Recall = (TP + smooth) / (TP + FN + smooth)
    F1 = 2 * Precision * Recall / (Recall + Precision + smooth)
    HD = compute_hausdorff(pred, gt)
    if all:
        return Dice.sum() / N, HD, SE.sum() / N, SP.sum() / N, IOU.sum() / N, Acc.sum() / N, F1.sum() / N
    else:
        return IOU.sum() / N, HD, Acc.sum() / N, SE.sum() / N, SP.sum() / N


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
