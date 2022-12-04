import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Sequential

from modouls.base_network import BaseNetwork




class VoxResNetModule(nn.Module):
    def __init__(self, num_channels=64, reduction_ratio=2):
        super(VoxResNetModule, self).__init__()

        # Define Convolution Layers
        self.conv_1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv_2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=1, padding=1)

        #Define Bach Normalization Layres
        self.bach_norm_1 = nn.BatchNorm3d(num_features=64)
        self.bach_norm_2 = nn.BatchNorm3d(num_features=64)

        self.SE = ChannelSpatialSELayer3D(num_channels, reduction_ratio)

    def forward(self, input):
        out = F.relu(self.bach_norm_1(input))
        out = self.conv_1(out)
        out = F.relu(self.bach_norm_2(out))
        out = self.conv_2(out)
        out = self.SE(out)
        out += input
        return out





class MultiLayerClassiifire(nn.Module):
    def __init__(self, n_classes=2):
        super(MultiLayerClassiifire, self).__init__()

        self.classifier = Sequential(
            nn.Conv3d(in_channels=224, out_channels=120, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=120, out_channels=n_classes, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, input):
        return self.classifier(input)


class VoxResNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=2):
        super(VoxResNet, self).__init__()

        # Define Convolution Layers
        self.conv_1a = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv_1b = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv_1c = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,3), stride=2, padding=1)
        self.conv_4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=2, padding=1)
        self.conv_7 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=2, padding=1)

        # Define Bach Normalization Layers
        self.bach_norm_1 = nn.BatchNorm3d(num_features=32)
        self.bach_norm_2 = nn.BatchNorm3d(num_features=32)
        self.bach_norm_3 = nn.BatchNorm3d(num_features=64)
        self.bach_norm_4 = nn.BatchNorm3d(num_features=64)

        # Define VoxResNet Layers
        self.voxresnet_2 = VoxResNetModule()
        self.voxresnet_3 = VoxResNetModule()

        self.voxresnet_5 = VoxResNetModule()
        self.voxresnet_6 = VoxResNetModule()

        self.voxresnet_8 = VoxResNetModule()
        self.voxresnet_9 = VoxResNetModule()

        #Define DeConvolution Layers

        self.deconv_1 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(3,3,3), stride=1, padding=1)
        self.deconv_2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4,4,4), stride=2, padding=1)
        self.deconv_3 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(6,6,6), stride=4, padding=1)
        self.deconv_4 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(10,10,10), stride=8, padding=1)

        #Define Classifier Modules

        self.classifier = nn.Conv3d(in_channels=224, out_channels=n_classes, kernel_size=(3,3,3), stride=1, padding=1)
        # self.classifier = MultiLayerClassiifire(n_classes=n_classes)

        


    def forward(self, input, train= False):
        h = F.relu(self.bach_norm_1(self.conv_1a(input)))    
        h = self.conv_1b(h)

        h_1 = self.deconv_1(h)
        h_1 = F.relu(h_1)


        h = F.relu(self.bach_norm_2(h))
        h = self.conv_1c(h)
        h = self.voxresnet_2(h)
        h = self.voxresnet_3(h)

        h_2 = self.deconv_2(h)
        h_2 = F.relu(h_2)

        h = F.relu(self.bach_norm_3(h))
        h = self.conv_4(h)
        h = self.voxresnet_5(h)
        h = self.voxresnet_6(h)

        h_3 = self.deconv_3(h)
        h_3 = F.relu(h_3)

        h = F.relu(self.bach_norm_4(h))
        h = self.conv_7(h)
        h = self.voxresnet_8(h)
        h = self.voxresnet_9(h)

        h_4 = self.deconv_4(h)
        h_4 = F.relu(h_4)

        features = torch.cat((h_1,h_2,h_3,h_4), axis=1)
        logit = self.classifier(features)

        
        if train:
            return [logit, logit]
        else:
            return F.softmax(logit)
        

