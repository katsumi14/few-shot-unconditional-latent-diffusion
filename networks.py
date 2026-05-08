import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from mylib.basic_layers import *

# Down sampling layer for Discriminator
# output half scale input features by applying Residual Block and average pooling 
class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()
        self.rb = ResBlockPA(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, normalization='spectral', activation='relu')
        self.down = nn.AvgPool2d(kernel_size=2)
    def forward(self, x):
        h = self.rb(x)
        return self.down(h)



# Discriminator network
class Discriminator(nn.Module):

    # C: the channels of input image
    # H: the height of input images
    # W: the width of input images
    def __init__(self, C, H, W):
        super(Discriminator, self).__init__()

        
        L1_C = 16
        L2_C = 32
        L3_C = 64
        L4_C = 128
        L5_C = 256
        L5_N = L5_C * (H//32) * (W//32)
        L6_N = 256

    
        self.preprocess = DiscriminatorAugmentation(H, W, p_hflip=0.5, p_vflip=0.4, p_rot=0.4) # with a 50% probability flipped images, 40% probability up side down images，40% rotate images

        

         
        # kernel size:4，stride:2，padding:1
        self.down1 = DownsamplingBlock(in_channels=C, out_channels=L1_C)
        self.down2 = DownsamplingBlock(in_channels=L1_C, out_channels=L2_C)
        self.down3 = DownsamplingBlock(in_channels=L2_C, out_channels=L3_C)
        self.down4 = DownsamplingBlock(in_channels=L3_C, out_channels=L4_C)
        self.down5 = DownsamplingBlock(in_channels=L4_C, out_channels=L5_C)

        # Fully connected layer
        self.fc = nn.Sequential(
            Flatten(),
            FC(in_features=L5_N, out_features=L6_N, normalization='spectral', activation='relu'), 
            FC(in_features=L6_N, out_features=1, normalization='none', activation='none'),
        )
        # the layer for patch discrimator
        self.patch_fc = nn.Sequential(
            Conv(in_channels=L5_C, out_channels=L5_C, kernel_size=3, stride=1, padding=0, normalization='spectral', activation='relu'),
            Conv(in_channels=L5_C, out_channels=1, kernel_size=1, stride=1, padding=0, normalization='none', activation='none'),
        )

    def forward(self, x):
        h = self.preprocess(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.down5(h)
        y = self.fc(h)
        p = self.patch_fc(h)
        return y, p




