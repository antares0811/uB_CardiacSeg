import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import pdb
from skimage import exposure
from torchvision import transforms

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        slice = (slice - slice.min()) / (slice.max() - slice.min())
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        # slice = np.repeat(np.expand_dims(slice, axis=0), repeats=3, axis=0)
        # slice = exposure.equalize_adapthist(slice)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        # input = transforms.Normalize([0.4786, 0.4728, 0.4528], [0.2425, 0.2327, 0.2564])(input)
        model.eval()
        with torch.no_grad():
            # output = model(input)['out']
            output = model(input)
            if len(output)>1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            # out = torch.argmax(output, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def test_single_volume_25d(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        if ind == 0:
            slice = image[ind:ind+2, :, :]
            slice = np.concatenate((slice[0][None, :, :], slice), axis=0)
        elif ind == image.shape[0]-1:
            slice = image[ind-1:, :, :]
            slice = np.concatenate((slice, slice[-1][None, :, :]), axis=0)
        else:
            slice = image[ind-1:ind+2, :, :]

        x, y = slice.shape[1], slice.shape[2]
        slice = zoom(slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            output = model(input)
            if len(output)>1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_single_volume_cross(image, label, model_l, model_r, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model_r.eval()
        model_l.eval()
        with torch.no_grad():
            output_l = model_l(input)
            output_r = model_r(input)
            output = (output_l + output_r) / 2
            if len(output)>1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
