from asyncore import write
import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb
from copy import deepcopy

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='BCP', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2

def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs, _ = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f'%(iter_num, loss, loss_dice, loss_ce))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    # torch.save(model.state_dict(), save_mode_path)
                    # torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()

def compute_cosine_similarity(model1, model2):
    similarity = 0.0
    p1, p2 = None, None
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if p1 == None:
            p1, p2 = param1.flatten(), param2.flatten()
        else:
            p1 = torch.cat([p1, param1.flatten()], dim=0)
            p2 = torch.cat([p2, param2.flatten()], dim=0)
    similarity = F.cosine_similarity(p1, p2, dim=0)
    return similarity

def compute_mse_score(model1, model2):
    p1, p2 = None, None
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if p1 == None:
            p1, p2 = param1.flatten(), param2.flatten()
        else:
            p1 = torch.cat([p1, param1.flatten()], dim=0)
            p2 = torch.cat([p2, param2.flatten()], dim=0)
    score = F.mse_loss(p1, p2, reduction='mean').item()
    
    return score

def compute_cosine(model1, model2, middle_model):
    similarity = 0.0
    p1, p2 = None, None
    for param1, param2, middle_param in zip(model1.parameters(), model2.parameters(), middle_model.parameters()):
        if p1 == None:
            p1, p2 = param1.flatten() - middle_param.flatten(), param2.flatten() - middle_param.flatten()
        else:
            p1 = torch.cat([p1, param1.flatten() - middle_param.flatten()], dim=0)
            p2 = torch.cat([p2, param2.flatten() - middle_param.flatten()], dim=0)
    similarity = F.cosine_similarity(p1, p2, dim=0)
    return similarity

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
    img_l_onehot = to_one_hot(img_l.unsqueeze(1), 2)
    patch_l_onehot = to_one_hot(patch_l.unsqueeze(1), 2)

    mse_loss = torch.mean(softmax_mse_loss(net3_output, img_l_onehot), dim=1) * mask * image_weight
    mse_loss += torch.mean(softmax_mse_loss(net3_output, patch_l_onehot), dim=1) * patch_mask * patch_weight


    loss = torch.sum(diff_mask * mse_loss) / (torch.sum(diff_mask) + 1e-16)
    return loss

def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)
    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)

    diff_mask = (l1 != l2)
    return diff_mask

def get_alpha(loss, iter_num):
    weight = max(1 / (1 + iter_num), 0.01)

    decay = 1 / torch.exp(loss * 0.3)
    
    weight *= decay
    
    return 1 - weight

def calculate_cross_entropy(probs, labels, mask):
    labels = labels.type(torch.int64)
    CE = nn.CrossEntropyLoss(reduction='none')
    ce = (CE(probs, labels) * mask).sum() / (mask.sum() + 1e-16) 
    return ce.item()

def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    model2 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    # for param in ema_model.parameters():
    #         param.detach_()   # ema_model set
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(model, pretrained_model)
    load_net(model2, pretrained_model)
    load_net(ema_model, pretrained_model)
    
    model.train()
    model2.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    best_dice_1 = 0
    best_dice_2 = 0
    best_dice_ema = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 0
    
    isFirst = True
    
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs+sub_bs], volume_batch[args.labeled_bs+sub_bs:]
            with torch.no_grad():
                unoutput_a, _ = ema_model(unimg_a)
                unoutput_b, _ = ema_model(unimg_b)
                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
            
            outputs_l_1, _ = model(mixl_img)
            outputs_unl_1, _ = model(mixu_img)
            
            outputs_l_2, _ = model2(mixl_img)
            outputs_unl_2, _ = model2(mixu_img)
            
            loss_l_1, _, _ = mix_loss(outputs_l_1, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_unl_1, _, _ = mix_loss(outputs_unl_1, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            
            loss_l_2, _, _ = mix_loss(outputs_l_2, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_unl_2, _, _ = mix_loss(outputs_unl_2, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            
            with torch.no_grad():
                diff_mask1 = get_XOR_region(outputs_l_1, outputs_l_2)
                diff_mask2 = get_XOR_region(outputs_unl_1, outputs_unl_2)

            net1_mse_loss_lab = mix_mse_loss(outputs_l_1, lab_a, plab_a.long(), loss_mask, diff_mask=diff_mask1)

            net1_mse_loss_unlab = mix_mse_loss(outputs_unl_1, plab_b.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask2)

            net2_mse_loss_lab = mix_mse_loss(outputs_l_2, lab_a, plab_a.long(), loss_mask, diff_mask=diff_mask1)

            net2_mse_loss_unlab = mix_mse_loss(outputs_unl_2, plab_b.long(), lab_b, loss_mask, unlab=True, diff_mask=diff_mask2)
            
            loss_1 = loss_l_1 + loss_unl_1 + 0.5 * (net1_mse_loss_lab + net1_mse_loss_unlab)
            loss_2 = loss_l_2 + loss_unl_2 + 0.5 * (net2_mse_loss_lab + net2_mse_loss_unlab)

            iter_num += 1

            optimizer.zero_grad()           
            loss_1.backward()
            optimizer.step()
            
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()
            
            isModel1 = False
            
            with torch.no_grad():
                
                unimg = volume_batch[args.labeled_bs:]
                
                model.eval()
                model2.eval()
                outputs_unl_stu_1, _ = model(unimg)
                outputs_unl_stu_2, _ = model2(unimg)
                model.train()
                model2.train()
                
                outputs_unl_stu_1_mask = get_cut_mask(outputs_unl_stu_1, nms=1)
                outputs_unl_stu_2_mask = get_cut_mask(outputs_unl_stu_2, nms=1)
                
                mask = (outputs_unl_stu_1_mask == outputs_unl_stu_2_mask).float()
                
                ce_model_1 = calculate_cross_entropy(outputs_unl_stu_1, outputs_unl_stu_1_mask, mask)
                
                ce_model_2 = calculate_cross_entropy(outputs_unl_stu_2, outputs_unl_stu_2_mask, mask)
                
                logging.info('iteration %d, ce model 1: %04f, ce model 2: %04f'%(iter_num, ce_model_1, ce_model_2))
                
                if ce_model_1 > ce_model_2:
                    isModel1 = False
                else:
                    isModel1 = True
                
            if isModel1 == False:
                alpha = get_alpha(loss_2, iter_num)
                update_ema_variables(model2, ema_model, alpha)
            else:
                alpha = get_alpha(loss_1, iter_num)
                update_ema_variables(model, ema_model, alpha)
            
            logging.info('iteration %d, alpha: %.4f: loss 1: %03f, loss 2: %03f, EMA from model 1: %d'%(iter_num, alpha, loss_1, loss_2, isModel1))

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                dice_sample_1 = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                model.train()
                
                model2.eval()
                dice_sample_2 = test_3d_patch.var_all_case_LA(model2, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                model2.train()
                
                ema_model.eval()
                dice_sample_ema = test_3d_patch.var_all_case_LA(ema_model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
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

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "../model/BCP/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "../model/BCP/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    print("Starting BCP training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('../code/LA_train.py', self_snapshot_path)
    # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    
