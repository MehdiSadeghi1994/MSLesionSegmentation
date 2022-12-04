import torch.nn as nn
from my_models.modules.se import ChannelSpatialSELayer3D
from torch.nn import functional as F

class SE_VoxResNetModule(nn.Module):
    def __init__(self, num_channels=64, reduction_ratio=2):
        super(SE_VoxResNetModule, self).__init__()

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

