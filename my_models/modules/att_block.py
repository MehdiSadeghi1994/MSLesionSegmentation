import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding, ReLU
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(Module):

    def __init__(self, in_dim, d_x):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.d_x = d_x

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=d_x, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, self.d_x, height, width)

        out = self.gamma*out
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module, self).__init__()


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



class Spatial_ATT(Module):
    def __init__(self, d_int, d_in):
        super(Spatial_ATT, self).__init__()
        self.d_int = d_int
        self.PAM_Module = PAM_Module(d_int, d_in)
        self.conv_1 = Conv2d(in_channels=d_in, out_channels=d_int, kernel_size=1)
        self.conv_2 = Conv2d(in_channels=d_in, out_channels=d_int, kernel_size=1)
        self.relu = ReLU()

    def forward(self, x, g):
        data = self.relu(self.conv_1(x) + self.conv_2(g))
        out = self.PAM_Module(data) + x

        return out



class Channel_ATT(Module):
    def __init__(self):
        super(Channel_ATT, self).__init__()
        self.cam = CAM_Module()

    def forward(self, x):
        return self.cam(x)
