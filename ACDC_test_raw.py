import argparse
import os
import shutil
import re
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom, median_filter
from scipy.ndimage.interpolation import zoom
from skimage import exposure
from tqdm import tqdm   
import glob
from networks.net_factory import net_factory
import csv
from torchvision import models
from networks.pretrain_model import Model
from torchvision import transforms
from networks.unet_attention import UNet_2d_Attention

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default=r'c:\Users\ngocd\OneDrive\Documents\Khiet\bourgogne\project\Resources\database\training', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='BCP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--stage_name', type=str, default='self_train', help='self or pre')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume_raw(case, net, test_save_path, FLAGS, type_evaluate):
    with open(os.path.join(case, 'Info.cfg'), 'r') as f:
        for info in f:
            if type_evaluate in info:
                break
    frame = int(re.findall(r'\d+', info)[0])
    frame = f"{frame:02d}"
    
    case_frame = case + '/' + os.path.basename(case) + '_frame' + frame
    
    image = nib.load(case_frame + '.nii.gz').get_fdata()
    image = (image - image.min()) / (image.max() - image.min())
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    
    label = nib.load(case_frame + '_gt.nii.gz').get_fdata().astype(np.int8)
    label = np.transpose(label, (2, 0, 1))
    
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        slice = (slice - slice.min()) / (slice.max() - slice.min())
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        # slice = np.repeat(np.expand_dims(slice, axis=0), repeats=3, axis=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        # input = transforms.Normalize([0.4786, 0.4728, 0.4528], [0.2425, 0.2327, 0.2564])(input)
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            if len(out_main)>1:
                out_main=out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            out = median_filter(out, size=5)
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open('data/ACDC/test_raw.list') as f:
        image_list = f.read().split('\n')
    test_save_path = "../model/supervised/ACDC_{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    # net = DeepLab(FLAGS.num_classes).cuda()
    # net = Model(FLAGS.num_classes, model_type='fcn').cuda()
    # net = UNet_2d_Attention(in_chns=1, class_num=FLAGS.num_classes).cuda()
    # save_model_path = r'c:\Users\ngocd\OneDrive\Documents\Khiet\bourgogne\project\code\model\supervised\ACDC_BCP_70_labeled\self_train\unet_best_model.pth'
    save_model_path = r'c:\Users\ngocd\OneDrive\Documents\Khiet\bourgogne\project\code\model\supervised\ACDC_BCP_7_labeled\self_train\translation_loss1208.pth'
    net.load_state_dict(torch.load(save_model_path))

    print("init weight from {}".format(save_model_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        path_case = os.path.join(FLAGS.root_path, case)
        first_metric, second_metric, third_metric = test_single_volume_raw(path_case, net, test_save_path, FLAGS, type_evaluate='ED')
        # print(case, (first_metric[0] + second_metric[0] + third_metric[0]) / 3, (first_metric[-1] + second_metric[-1] + third_metric[-1]) / 3)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric_ed = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        path_case = os.path.join(FLAGS.root_path, case)
        first_metric, second_metric, third_metric = test_single_volume_raw(path_case, net, test_save_path, FLAGS, type_evaluate='ES')
        # print(case, (first_metric[0] + second_metric[0] + third_metric[0]) / 3, (first_metric[-1] + second_metric[-1] + third_metric[-1]) / 3)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric_es = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    return avg_metric_ed, avg_metric_es, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    metric_ED, metric_ES, test_save_path = Inference(FLAGS)
    csv_file = os.path.join(test_save_path, 'result.csv')
    list_class = ['RV', 'Myo', 'LV']
    average_ = np.zeros(4)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['End_type', 'Class', 'Dice', 'Jaccard', 'HD95', 'ASD'])
        for end_type in ['ED', 'ES']:
            metric_value = locals()[f'metric_{end_type}']
            for i in range(0, 3):
                writer.writerow([end_type, list_class[i]] + list(metric_value[i]))
            average_value = (metric_value[0] + metric_value[1] + metric_value[2]) / 3
            writer.writerow([end_type, 'Average'] + list(average_value))
            average_ = average_ + average_value
        writer.writerow(['Average', ''] + list(average_ / 2))
