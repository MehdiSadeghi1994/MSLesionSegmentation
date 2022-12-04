import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Sequential
from my_models.modules.se_voxresnet import SE_VoxResNetModule

class BaseNetwork(nn.Module):

    def __init__(self, in_channels=1, n_classes=2):
        super(BaseNetwork, self).__init__()

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

        # Define SE_VoxResNet Layers
        self.voxresnet_2 = SE_VoxResNetModule()
        self.voxresnet_3 = SE_VoxResNetModule()

        self.voxresnet_5 = SE_VoxResNetModule()
        self.voxresnet_6 = SE_VoxResNetModule()

        self.voxresnet_8 = SE_VoxResNetModule()
        self.voxresnet_9 = SE_VoxResNetModule()

        #Define DeConvolution Layers

        self.deconv_1 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(3,3,3), stride=1, padding=1)
        self.deconv_2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4,4,4), stride=2, padding=1)
        self.deconv_3 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(6,6,6), stride=4, padding=1)
        self.deconv_4 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(10,10,10), stride=8, padding=1)

        #Define Classifier Modules

        self.classifier = nn.Conv3d(in_channels=224, out_channels=n_classes, kernel_size=(3,3,3), stride=1, padding=1)
        # self.classifier = MultiLayerClassiifire(n_classes=n_classes)

        

    def forward(self, input):
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
        del h_1, h_2, h_3, h_4, h
        
        return features
        
