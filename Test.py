import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
# from scipy import misc
import cv2
from lib.Network import *
from utils.data_val import test_dataset
from util import get_metrics
from tensorboardX import SummaryWriter
import time
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--pth_path', type=str,
                    default='./snapshot/123/Net_epoch_10.pth')
parser.add_argument('--dataname', type=str, default='TN3k', help='train dataset')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
opt = parser.parse_args()

for _data_name in ['TN3k']: # ['TN3k', 'BUSI']
    data_path = 'D:/ywy/3.Dada_Code/Dataset/Ultrasound/{}'.format(_data_name)
    save_path = './res/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = Network_interFA_noSpade_noEdge_ODE_slot_channel4(channels=128)  # can be different under diverse backbone
    # model = Network(channels=96)
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(opt.pth_path, map_location='cuda:0').items()})
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/test-image/'.format(data_path)
    gt_root = '{}/test-mask/'.format(data_path)
    # edge_root = '{}/Edge/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    """
        validation function
        """
    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0
    sum_time = 0
    eval_number = 0
    model.eval()
    with torch.no_grad():
        metrics = {'Dice': [], 'HD': [], 'SE': [], 'SP': [], 'IOU': [], 'Acc': [], 'F1': []}
        mae_sum = 0
        # mae_sum_edge = 0
        for i in range(test_loader.size):
            image, gt_tensor, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt_tensor.squeeze(), np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            start_time = time.time()
            result = model(image)
            res = F.upsample(result[4], size=gt.shape, mode='bilinear', align_corners=False)
            output = res.sigmoid().data.cpu().numpy()
            eval_number = eval_number + 1
            sum_time = sum_time + (time.time() - start_time)
            res = output.squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            print('./snapshot/' + opt.dataname + '/' + name)
            cv2.imwrite('./snapshot/' + opt.dataname + '/' + name, res * 255)

            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

            # 指标计算
            label = gt_tensor.cpu().detach().numpy()
            Dice, HD, SE, SP, IOU, Acc, F1 = get_metrics(output, label)
            metrics['Dice'].append(Dice)
            metrics['HD'].append(HD)
            metrics['SE'].append(SE)
            metrics['SP'].append(SP)
            metrics['IOU'].append(IOU)
            metrics['Acc'].append(Acc)
            metrics['F1'].append(F1)
            # 计算每个指标的均值和标准差
        print("test speed (FPS):", eval_number / sum_time)
        metrics_mean = {key: np.mean(value) for key, value in metrics.items()}
        metrics_std = {key: np.std(value) for key, value in metrics.items()}
        # Print the log info
        print(
            '[testing] Dice: %.4f ± %.4f, HD: %.4f ± %.4f, IOU: %.4f ± %.4f, Acc: %.4f ± %.4f, SE: %.4f ± %.4f, SP: %.4f ± %.4f, F1: %.4f ± %.4f' %
            (metrics_mean['Dice'], metrics_std['Dice'], metrics_mean['HD'], metrics_std['HD'],
             metrics_mean['IOU'], metrics_std['IOU'], metrics_mean['Acc'], metrics_std['Acc'],
             metrics_mean['SE'], metrics_std['SE'], metrics_mean['SP'],
             metrics_std['SP'], metrics_mean['F1'], metrics_std['F1']))
        # 保存结果到CSV
        with open(os.path.join(save_path, 'result.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(
                ["TN3K", opt.lr, opt.pth_path.split('/')[-1],
                 'Dice: %.4f ± %.4f' % (metrics_mean['Dice'], metrics_std['Dice']),
                 'HD: %.4f ± %.4f' % (metrics_mean['HD'], metrics_std['HD']),
                 'IOU: %.4f ± %.4f' % (metrics_mean['IOU'], metrics_std['IOU']),
                 'Acc: %.4f ± %.4f' % (metrics_mean['Acc'], metrics_std['Acc']),
                 'SE: %.4f ± %.4f' % (metrics_mean['SE'], metrics_std['SE']),
                 'SP: %.4f ± %.4f' % (metrics_mean['SP'], metrics_std['SP']),
                 'F1: %.4f ± %.4f' % (metrics_mean['F1'], metrics_std['F1'])])
        f.close()
    writer.close()