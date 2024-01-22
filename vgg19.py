import torch
from torch import nn
import torchvision
from torchvision import models


class Vgg19(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2' 'conv5_1']
        
        self.basemodel = models.vgg19(weights='DEFAULT')
        # turn off training for these layers vv
        for x in self.basemodel.features:
            x.requires_grad = False
        self.baselayers = self.basemodel.features
        
        self.until_conv1_1 = nn.Sequential()
        self.until_conv2_1 = nn.Sequential()
        self.until_conv3_1 = nn.Sequential()
        self.until_conv4_1 = nn.Sequential()
        self.until_conv4_2 = nn.Sequential()
        self.until_conv5_1 = nn.Sequential()
        
        self.until_conv1_1.append(self.baselayers[0])
        for n in range(1, 6):
            self.until_conv2_1.append(self.baselayers[n])
        for n in range(6, 11):
            self.until_conv3_1.append(self.baselayers[n])
        for n in range(11, 20):
            self.until_conv4_1.append(self.baselayers[n])
        for n in range(20, 22):
            self.until_conv4_2.append(self.baselayers[n])
        for n in range(22, 29):
            self.until_conv5_1.append(self.baselayers[n])        
    
    def forward(self, x):
        style_1 = self.until_conv1_1(x)
        style_2 = self.until_conv2_1(style_1)
        style_3 = self.until_conv3_1(style_2)
        style_4 = self.until_conv4_1(style_3)
        content_rep = self.until_conv4_1(style_4)
        style_5 = self.until_conv5_1(content_rep)
        
        style_reps = (style_1, style_2, style_3, style_4, style_5)
        
        return (style_reps, content_rep)
