import os
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.Network import *
from utils.data_val import test_dataset, get_loader_scribble_noEdge_3326
from utils.utils import clip_gradient, get_coef, cal_ual
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch import optim
from losses.cross_entropy_loss import partial_cross_entropy
from util import get_metrics
import csv
import cv2


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def weighted_structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    # mask2 means whether to select this point.
    mask1 = mask.cpu()
    # mask_mask = np.where((mask1[...,:]>=0.34) & (mask1[...,:]<=0.67),0,1)
    mask2 = np.where((mask1[:] > 0.3) & (mask1[:] < 0.7), 0, 1)  # 介于 0.3 和 0.7 之间的值被设置为 0，其余为 1
    # mask1[:, :, mask2] = 0
    mask2 = torch.tensor(mask2).cuda()

    wbce = F.binary_cross_entropy_with_logits(pred, mask, mask2, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit * mask2).sum(dim=(2, 3))
    union = ((pred + mask) * weit * mask2).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, model, optimizer, epoch, save_path, writer, dataname):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0

    for i, (images, gts, scribble) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images = images.cuda()
        gts = gts.cuda()
        scribble = scribble.cuda()
        # edges = edges.cuda()

        preds = model(images)

        # ual loss
        ual_coef = get_coef(iter_percentage=i / total_step, method='cos')
        ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts)
        ual_loss *= ual_coef

        # structure loss w/o weighted
        if opt.lossmanner == 'NoWeighted':
            loss_init = structure_loss(preds[0], gts) * 0.0625

            loss_body = structure_loss(preds[1], gts) * 0.125 + structure_loss(preds[2], gts) * 0.25 + \
                        structure_loss(preds[3], gts) * 0.5

            loss_final = structure_loss(preds[4], gts)

        if opt.lossmanner == 'Weighted':
            # weighted sturcture loss
            loss_init = weighted_structure_loss(preds[0], gts) * 0.0625

            loss_body = weighted_structure_loss(preds[1], gts) * 0.125 + weighted_structure_loss(preds[2],
                                                                                                 gts) * 0.25 + \
                        weighted_structure_loss(preds[3], gts) * 0.5

            loss_final = weighted_structure_loss(preds[4], gts)

        if not (bool(opt.lossmanner == 'NoWeighted') | bool(opt.lossmanner == 'Weighted')):
            print("Please provide a valid key words for --lossmanner, i.e., 'NoWeighted' or 'Weighted'.")

        # PCE loss
        loss_init1 = partial_cross_entropy(preds[0], scribble.unsqueeze(1)) * 0.0625
        loss_body1 = partial_cross_entropy(preds[1], scribble.unsqueeze(1)) * 0.125 + partial_cross_entropy(
            preds[2], scribble.unsqueeze(1)) * 0.25 + \
                     partial_cross_entropy(preds[3], scribble.unsqueeze(1)) * 0.5
        loss_final1 = partial_cross_entropy(preds[4], scribble.unsqueeze(1))

        loss1 = loss_init + loss_body + loss_final
        loss2 = loss_init1 + loss_body1 + loss_final1
        loss3 = 2 * ual_loss
        # loss = loss1 + 3 * loss2 + loss3
        loss = loss1 + loss3
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        step += 1
        epoch_step += 1
        loss_all += loss.data

        # 20 iters 1 record
        if i % 20 == 0 or i == total_step or i == 1:
            # print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} Loss3:{:0.4f}'.
            #   format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data, loss_edge.data))
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss1.data, loss2.data,
                           loss3.data))
            logging.info(
                # '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} Loss3:{:0.4f}'.
                # format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data, loss_edge.data))
                '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss1.data, loss2.data,
                           loss3.data))  # 弱监督，没有edge
            # TensorboardX-Loss
            writer.add_scalars('Loss_Statistics',
                               {'Loss_pseudo': loss1.data, 'Loss_scribble': loss2.data, 'Loss_total': loss.data},
                               global_step=step)
            # TensorboardX-Training Data
            grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('RGB', grid_image, step)
            grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('GT', grid_image, step)
            grid_image = make_grid(scribble[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('scribble', grid_image, step)
            # grid_image = make_grid(edges[0].clone().cpu().data, 1, normalize=True)
            # writer.add_image('Edge', grid_image, step)

    loss_all /= epoch_step  # 求average loss
    logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)


def val(test_loader, model, epoch, save_path, writer, dataname):
    """
    validation function
    """
    global best_mae, best_epoch
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

            result = model(image)

            res = F.upsample(result[4], size=gt.shape, mode='bilinear', align_corners=False)
            output = res.sigmoid().data.cpu().numpy()
            res = output.squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            print('./result/' + opt.dataname + '/' + name)
            cv2.imwrite('./result/' + opt.dataname + '/' + name, res * 255)

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
        metrics_mean = {key: np.mean(value) for key, value in metrics.items()}
        metrics_std = {key: np.std(value) for key, value in metrics.items()}
        if metrics_mean['Dice'] > best_mae:
            best_mae = metrics_mean['Dice']
            torch.save(model.state_dict(),
                       save_path + opt.dataname + '_{}.pth'.format(epoch) + "_" + str(best_mae))
        # Print the log info
        print(
            '[testing] Dice: %.4f ± %.4f, HD: %.4f ± %.4f, IOU: %.4f ± %.4f, Acc: %.4f ± %.4f, SE: %.4f ± %.4f, SP: %.4f ± %.4f, F1: %.4f ± %.4f' %
            (metrics_mean['Dice'], metrics_std['Dice'], metrics_mean['HD'], metrics_std['HD'],
             metrics_mean['IOU'], metrics_std['IOU'], metrics_mean['Acc'], metrics_std['Acc'],
             metrics_mean['SE'], metrics_std['SE'], metrics_mean['SP'],
             metrics_std['SP'], metrics_mean['F1'], metrics_std['F1']))
        # 保存结果到CSV
        with open(os.path.join(save_path, opt.dataname, 'result.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(
                ["TN3K", opt.lr, epoch,
                 'Dice: %.4f ± %.4f' % (metrics_mean['Dice'], metrics_std['Dice']),
                 'HD: %.4f ± %.4f' % (metrics_mean['HD'], metrics_std['HD']),
                 'IOU: %.4f ± %.4f' % (metrics_mean['IOU'], metrics_std['IOU']),
                 'Acc: %.4f ± %.4f' % (metrics_mean['Acc'], metrics_std['Acc']),
                 'SE: %.4f ± %.4f' % (metrics_mean['SE'], metrics_std['SE']),
                 'SP: %.4f ± %.4f' % (metrics_mean['SP'], metrics_std['SP']),
                 'F1: %.4f ± %.4f' % (metrics_mean['F1'], metrics_std['F1'])])
        f.close()
    writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--lossmanner', type=str, default='NoWeighted')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--dataname', type=str, default='BUSI', help='train dataset')  # TN3k or BUSI
    parser.add_argument('--train_root', type=str, default='D:/ywy/3.Dada_Code/Dataset/Ultrasound/BUSI/', # TN3k or BUSI
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='D:/ywy/3.Dada_Code/Dataset/Ultrasound/BUSI/',  # TN3k or BUSI
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./snapshot', help='the path to save model and log')
    opt = parser.parse_args()

    # # build the model
    # model = Network(channels=32).cuda()

    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU  0')
    cudnn.benchmark = True

    # build the model
    # model = torch.nn.DataParallel(Network_interFA_noSpade_noEdge_ODE_slot_channel4(channels=128), device_ids=device_ids)   # 多线程代码：指定在多个GPU上并行
    model = Network_interFA_noSpade_noEdge_ODE_slot_channel4(channels=128)
    model = model.cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    '''
    X_COD.txt has 3326 selected images;
    X_SOD.txt has 9162 selected  images;
    COD_1P_0.02_2195.txt has 2195 images;
    COD_1P_0.05.txt has 2916 images.
    '''

    # .................TN3k................
    # image selection
    # train_loader = get_loader_scribble_noEdge_3326(image_root=opt.train_root + 'trainval-image/',
    #                                                gt_root='D:/ywy/3.Dada_Code/Dataset/Ultrasound/TN3k/trainval-psudo-mask/',
    #                                                # 可以是GT，弱监督的情况为SAM得到的伪标签 这里修改路径为SAM的分割结果
    #                                                scribble_root=opt.train_root + 'scribble-binary/',
    #                                                imgname_root='D:/ywy/3.Dada_Code/Dataset/Ultrasound/ImageSets/TN3k_train_name.lst',
    #                                                batchsize=opt.batchsize,
    #                                                trainsize=opt.trainsize,
    #                                                num_workers=0)
    # # val_loader = test_dataset(image_root=opt.val_root + 'test-image/',
    # #                           gt_root=opt.val_root + 'test-mask/',
    # #                           testsize=opt.trainsize)
    # >>>>>>>>>>>>>>>>>>>>BUSI>>>>>>>>>>>>>>>>>>>>>>>>
    train_loader = get_loader_scribble_noEdge_3326(image_root=opt.train_root + 'train/image/',
                                                   gt_root='D:/ywy/3.Dada_Code/Dataset/Ultrasound/BUSI/train/mask/',
                                                   # 可以是GT，弱监督的情况为SAM得到的伪标签 这里修改路径为SAM的分割结果
                                                   scribble_root=opt.train_root + 'train/scribble-binary/',
                                                   imgname_root='D:/ywy/3.Dada_Code/Dataset/Ultrasound/ImageSets/BUSI_train_name.lst',
                                                   batchsize=opt.batchsize,
                                                   trainsize=opt.trainsize,
                                                   num_workers=0)
    val_loader = test_dataset(image_root=opt.val_root + 'test/image/',
                              gt_root=opt.val_root + 'test/mask/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=40, eta_min=1e-6)
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))

        train(train_loader, model, optimizer, epoch, save_path, writer, opt.dataname)
        val(val_loader, model, epoch, save_path, writer, opt.dataname)
