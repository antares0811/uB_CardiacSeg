import h5py
import matplotlib.pyplot as plt
import numpy as np
from networks.net_factory import net_factory
import torch
from scipy.ndimage.interpolation import zoom
from skimage.exposure import match_histograms
import copy
from skimage.segmentation import chan_vese
from cv2 import equalizeHist
from scipy.ndimage import laplace
from dataloaders.dataset import random_rot_flip, random_rotate, random_translate
from medpy import metric
if __name__ == '__main__':
    fontsize=20
    y_text=10
    va='top'
    net = net_factory(net_type='unet', in_chns=1, class_num=4)
    save_model_path = r'c:\Users\ngocd\OneDrive\Documents\Khiet\bourgogne\project\code\model\supervised\ACDC_BCP_7_labeled\self_train\translation_loss1208.pth'
    net.load_state_dict(torch.load(save_model_path))
    net.eval()

    h5f = h5py.File('data/ACDC/data/patient059_frame01.h5', 'r')
    slice = 5
    image = h5f['image']
    image = image[slice, :, :]
    x, y = image.shape
    image = zoom(image, (256 / x, 256 / y), order=0)
    label = zoom(h5f['label'][slice, :, :], (256 / x, 256 / y), order=0)
    input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    out = net(input)[0]
    out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
    out = np.array(out.cpu())
    
    d = metric.binary.dc(out, label)
    dh = metric.binary.hd95(out, label)
    plt.figure(figsize=(15, 16))
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.text(5, y_text, '(a)', color='white', weight='bold', fontsize=fontsize, va=va)
    plt.title('Original images', fontsize=fontsize)
    plt.axis('off')
    plt.subplot(3, 3, 2)
    plt.title('Ground truths', fontsize=fontsize)
    plt.imshow(label)
    plt.axis('off')
    plt.text(5, y_text, '(b)', color='white', weight='bold', fontsize=fontsize, va=va)
    plt.subplot(3, 3, 3)
    plt.imshow(out)
    plt.title('Predictions', fontsize=fontsize)
    plt.text(5, y_text, f'(f) D={d:.3f}\n     $d_H$={dh:.3f}', color='white', weight='bold', fontsize=fontsize, va=va)
    plt.axis('off')
    
    h5f = h5py.File('data/ACDC/data/patient052_frame02.h5', 'r')
    slice = 5
    image = h5f['image']
    image = image[slice, :, :]
    x, y = image.shape
    image = zoom(image, (256 / x, 256 / y), order=0)
    label = zoom(h5f['label'][slice, :, :], (256 / x, 256 / y), order=0)
    input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    out = net(input)[0]
    out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
    out = np.array(out.cpu())
    
    d = metric.binary.dc(out, label)
    dh = metric.binary.hd95(out, label)
    plt.subplot(3, 3, 4)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.text(5, y_text, '(d)', color='white', weight='bold', fontsize=fontsize, va=va)
    plt.subplot(3, 3, 5)
    plt.imshow(label)
    plt.axis('off')
    plt.text(5, y_text, '(e)', color='white', weight='bold', fontsize=fontsize, va=va)
    plt.subplot(3, 3, 6)
    plt.imshow(out)
    plt.text(5, y_text, f'(f) D={d:.3f}\n     $d_H$={dh:.3f}', color='white', weight='bold', fontsize=fontsize, va=va)
    plt.axis('off')
    
    h5f = h5py.File('data/ACDC/data/patient093_frame01.h5', 'r')
    slice = 0
    image = h5f['image']
    image = image[slice, :, :]
    x, y = image.shape
    image = zoom(image, (256 / x, 256 / y), order=0)
    label = zoom(h5f['label'][slice, :, :], (256 / x, 256 / y), order=0)
    input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    out = net(input)[0]
    out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
    out = np.array(out.cpu())
    
    d = metric.binary.dc(out, label)
    dh = metric.binary.hd95(out, label)
    plt.subplot(3, 3, 7)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.text(5, y_text, '(g)', color='white', fontsize=fontsize, va=va, weight='bold')
    plt.subplot(3, 3, 8)
    plt.imshow(label)
    plt.axis('off')
    plt.text(5, y_text, '(h)', color='white', fontsize=fontsize, va=va, weight='bold')
    plt.subplot(3, 3, 9)
    plt.imshow(out)
    plt.text(5, y_text, f'(f) D={d:.3f}\n     $d_H$={dh:.3f}', color='white', weight='bold', fontsize=fontsize, va=va)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(r'c:\Users\ngocd\OneDrive\Documents\Khiet\bourgogne\project\visualize.png', dpi=300)
    # plt.show()