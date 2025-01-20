import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from copy import deepcopy

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from networks.net_factory import BCP_net, net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='BCP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')

args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()
    

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            #-- original
            net_input_a = img_a * img_mask + img_b * (1 - img_mask)
            net_input_b = img_b * img_mask + img_a * (1 - img_mask)
            out_mixl_a = model(net_input_a)
            out_mixl_b = model(net_input_b)
            loss_dice_a, loss_ce_a = mix_loss(out_mixl_a, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss_dice_b, loss_ce_b = mix_loss(out_mixl_b, lab_b, lab_a, loss_mask, u_weight=1.0, unlab=True)
            
            loss_dice = loss_dice_a + loss_dice_b
            loss_ce = loss_ce_a + loss_ce_b

            loss = (loss_dice + loss_ce) / 2            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    
def compute_cosine_similarity(model1, model2):
    similarity = 0.0
    total_count = 0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        p1, p2 = param1.flatten(), param2.flatten()
        sim = F.cosine_similarity(p1, p2, dim=0)
        similarity += (sim.item() + 1) / 2
        total_count += 1
    return similarity / total_count

def get_alpha(loss, iter_num):
    global_w = max(1 / (1 + iter_num), 0.01)

    decay = 1 / torch.exp(loss * 0.3)
    
    weight = global_w * decay
    
    logging.info('iteration %d, global: %.6f: local: %.6f, weight: %.6f, alpha: %.6f'%(iter_num, global_w, decay, weight, 1 - weight))
    
    return 1 - weight

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax - target_logits) ** 2
    return mse_loss


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot

def mix_mse_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False, diff_mask=None):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1 - mask
    img_l_onehot = to_one_hot(img_l.unsqueeze(1), 4)
    patch_l_onehot = to_one_hot(patch_l.unsqueeze(1), 4)

    mse_loss = torch.mean(softmax_mse_loss(net3_output, img_l_onehot), dim=1) * mask * image_weight
    mse_loss += torch.mean(softmax_mse_loss(net3_output, patch_l_onehot), dim=1) * patch_mask * patch_weight


    loss = torch.sum(diff_mask * mse_loss) / (torch.sum(diff_mask) + 1e-16)
    return loss


voxel_kl_loss = nn.KLDivLoss(reduction="none")


def mix_max_kl_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False, diff_mask=None):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1 - mask

    with torch.no_grad():
        s1 = torch.softmax(net3_output, dim=1)
        l1 = torch.argmax(s1, dim=1)
        img_diff_mask = (l1 != img_l)
        patch_diff_mask = (l1 != patch_l)

        uniform_distri = torch.ones(net3_output.shape)
        uniform_distri = uniform_distri.cuda()

    kl_loss = torch.mean(voxel_kl_loss(F.log_softmax(net3_output, dim=1), uniform_distri),
                         dim=1) * mask * img_diff_mask * image_weight
    kl_loss += torch.mean(voxel_kl_loss(F.log_softmax(net3_output, dim=1), uniform_distri),
                          dim=1) * patch_mask * patch_diff_mask * patch_weight

    sum_diff = torch.sum(mask * img_diff_mask * diff_mask) + torch.sum(patch_mask * patch_diff_mask * diff_mask)
    loss = torch.sum(diff_mask * kl_loss) / (sum_diff + 1e-16)
    return loss

def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)
    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)

    diff_mask = (l1 != l2)
    return diff_mask

def calculate_cross_entropy(probs, labels, mask):
    labels = labels.type(torch.int64)
    CE = nn.CrossEntropyLoss(reduction='none')
    ce = (CE(probs, labels) * mask).sum() / (mask.sum() + 1e-16) 
    return ce.item()

def self_train(args ,pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'unet_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model = BCP_net(in_chns=1, class_num=num_classes)
    model2 = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
    
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    load_net_opt(model2, optimizer2, pre_trained_model)

    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_dice = 0
    best_dice_1 = 0
    best_dice_2 = 0
    best_dice_ema = 0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 0
    
    isFirst = True
    correct_ps_ratio = 0
    
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            lab_u = label_batch[args.labeled_bs:]
            
            img = volume_batch[:args.labeled_bs]
            lab = label_batch[:args.labeled_bs]
            lab = lab.type(torch.int64)
            
            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                img_mask, loss_mask = generate_mask(img_a)
                unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
                
                plab = torch.concat([plab_a, plab_b])
                lab_u = lab_u.flatten(start_dim=1)
                plab = plab.flatten(start_dim=1)
                correct_pixel = (lab_u == plab).sum(dim=1)
                correct_ps_ratio = (correct_pixel / lab_u.shape[1]).mean()
                logging.info('iteration %d, correct ratio: %03f, shape: %s, correct: %d'%(iter_num + 1, 
                                                                                        correct_ps_ratio, 
                                                                                        lab_u.shape, 
                                                                                        correct_pixel.sum()))
                
            consistency_weight = get_current_consistency_weight(iter_num//150)

            net_input_l = img_a * img_mask + uimg_a * (1 - img_mask)
            net_input_unl = uimg_b * img_mask + img_b * (1 - img_mask)
            
            outputs_1_l = model(net_input_l)
            outputs_1_unl = model(net_input_unl)
            
            outputs_2_l = model2(net_input_l)
            outputs_2_unl = model2(net_input_unl)

            loss_1_dice_l, loss_1_ce_l = mix_loss(outputs_1_l, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_1_dice_unl, loss_1_ce_unl = mix_loss(outputs_1_unl, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            
            loss_2_dice_l, loss_2_ce_l = mix_loss(outputs_2_l, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_2_dice_unl, loss_2_ce_unl = mix_loss(outputs_2_unl, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)

            with torch.no_grad():
                diff_mask1 = get_XOR_region(outputs_1_l, outputs_2_l)
                diff_mask2 = get_XOR_region(outputs_1_unl, outputs_2_unl)

            net1_mse_loss_lab = mix_mse_loss(outputs_1_l, lab_a, plab_a.long(), loss_mask, diff_mask=diff_mask1)

            net1_mse_loss_unlab = mix_mse_loss(outputs_1_unl, plab_b.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask2)

            net2_mse_loss_lab = mix_mse_loss(outputs_2_l, lab_a, plab_a.long(), loss_mask, diff_mask=diff_mask1)

            net2_mse_loss_unlab = mix_mse_loss(outputs_2_unl, plab_b.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask2)
            
            loss_ce_1 = loss_1_ce_l + loss_1_ce_unl
            loss_dice_1 = loss_1_dice_l + loss_1_dice_unl

            loss_ce_2 = loss_2_ce_l + loss_2_ce_unl
            loss_dice_2 = loss_2_dice_l + loss_2_dice_unl

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + 0.5 * (net1_mse_loss_lab + net1_mse_loss_unlab)

            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + 0.5 * (net2_mse_loss_lab + net2_mse_loss_unlab)

            optimizer.zero_grad()
            loss_1.backward()
            optimizer.step()
            
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()
            
            with torch.no_grad():
                uimg = volume_batch[args.labeled_bs:]
                
                model.eval()
                model2.eval()
                outputs_unl_stu_1 = model(uimg)
                outputs_unl_stu_2 = model2(uimg)
                model.train()
                model2.train()
                
                outputs_unl_stu_1_mask = get_ACDC_masks(outputs_unl_stu_1, nms=1)
                outputs_unl_stu_2_mask = get_ACDC_masks(outputs_unl_stu_2, nms=1)
                
                mask = (outputs_unl_stu_1_mask == outputs_unl_stu_2_mask).float()
                
                ce_model_1 = calculate_cross_entropy(outputs_unl_stu_1, outputs_unl_stu_1_mask, mask)
                
                ce_model_2 = calculate_cross_entropy(outputs_unl_stu_2, outputs_unl_stu_2_mask, mask)
                
                logging.info('iteration %d, ce model 1: %04f, ce model 2: %04f'%(iter_num + 1, ce_model_1, ce_model_2))
                
                if ce_model_1 > ce_model_2:
                    isModel1 = False
                else:
                    isModel1 = True

            iter_num += 1
            
            if isModel1 == False:
                alpha = get_alpha(loss_2, iter_num)
                update_model_ema(model2, ema_model, alpha)
            else:
                alpha = get_alpha(loss_1, iter_num)
                update_model_ema(model, ema_model, alpha)

            logging.info('iteration %d, alpha: %.4f: loss 1: %03f, loss 2: %03f'%(iter_num, alpha, loss_1, loss_2))
            
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                dice_sample_1 = np.mean(metric_list, axis=0)[0]
                model.train()
                
                model2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                dice_sample_2 = np.mean(metric_list, axis=0)[0]
                model2.train()
                
                ema_model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], ema_model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                dice_sample_ema = np.mean(metric_list, axis=0)[0]
                ema_model.train()
                
                if dice_sample_1 > best_dice_1:
                    best_dice_1 = round(dice_sample_1, 4)
                    save_mode_path = os.path.join(self_snapshot_path,  'iter_{}_dice_{}_stu1.pth'.format(iter_num, best_dice_1))
                    torch.save(model.state_dict(), save_mode_path)
                
                if dice_sample_2 > best_dice_2:
                    best_dice_2 = round(dice_sample_2, 4)
                    save_mode_path = os.path.join(self_snapshot_path,  'iter_{}_dice_{}_stu2.pth'.format(iter_num, best_dice_2))
                    torch.save(model2.state_dict(), save_mode_path)
                    
                if dice_sample_ema > best_dice_ema:
                    best_dice_ema = round(dice_sample_ema, 4)
                    save_mode_path = os.path.join(self_snapshot_path,  'iter_{}_dice_{}_ema.pth'.format(iter_num, best_dice_ema))
                    torch.save(ema_model.state_dict(), save_mode_path)

                if dice_sample_1 >= dice_sample_2 and dice_sample_1 > best_dice:
                    best_dice = round(dice_sample_1, 4)
                    save_best_path = os.path.join(self_snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best_path)
                elif dice_sample_2 > dice_sample_1 and dice_sample_2 > best_dice:
                    best_dice = round(dice_sample_2, 4)
                    save_best_path = os.path.join(self_snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_best_path)

                logging.info("Evaluation: Student 1: %.4f, Student 2: %.4f, EMA Model: %.4f"%(dice_sample_1, dice_sample_2, dice_sample_ema))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "../model/BCP/ACDC_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "../model/BCP/ACDC_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('../code/ACDC_train.py', self_snapshot_path)

    #Pre_train
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    #Self_train
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    


