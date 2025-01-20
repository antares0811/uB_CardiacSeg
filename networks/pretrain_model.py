import torch.nn as nn
from torchvision import models
import torch
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, num_classes, model_type):
        super(Model, self).__init__()
        
        self.conv_first = nn.Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))
        
        if model_type == 'deeplab':
            model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        elif model_type == 'fcn':
            model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)
        elif model_type == 'lraspp':
            model = models.segmentation.lraspp_mobilenet_v3_large(weights=models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone = model.backbone
        self.aspp = model.classifier
        
        self.conv_last = nn.Conv2d(21, num_classes, kernel_size=(1, 1), stride=(1, 1))
            
        self.model_type = model_type
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.conv_first(x)
        if self.model_type in ['deeplab', 'fcn']:
            x = self.backbone(x)['out']
        else:
            x = self.backbone(x)
        x = self.aspp(x)
        x = self.conv_last(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x
    
class DilatedModel(nn.Module):
    def __init__(self, num_classes):
        super(DilatedModel, self).__init__()
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=9, stride=1, dilation=1, padding=4)
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 64, kernel_size=7, stride=1, dilation=2, padding=6)
        self.batchnorm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, dilation=4, padding=8)
        self.batchnorm3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=6, padding=6)
        self.batchnorm4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=8, padding=8)
        self.batchnorm5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, dilation=1)
        
    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.relu(self.batchnorm5(self.conv5(x)))
        x = self.conv6(x)
        return x
    
if __name__ == '__main__':
    model = Model(4, 'lraspp')
    x = torch.zeros((24, 1, 256, 256))
    y = model(x)
    print(y.shape)
    print(model)