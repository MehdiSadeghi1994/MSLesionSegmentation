import torch
import torch.nn as nn
from torch.nn import functional as F
from my_models.modules.base_network import BaseNetwork



class MultiTaskModel(nn.Module):
    def __init__(self, in_channels, n_classes=2):
        super(MultiTaskModel, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        # self.multi_task = multi_task

        self.base_network = BaseNetwork(in_channels)
        self.classifire = nn.Conv3d(in_channels=224, out_channels=n_classes, kernel_size=(3,3,3), stride=1, padding=1)




    def forward(self, x):


        features = self.base_network(x)

        logit = self.classifire(features)

        del features

        return logit


        


