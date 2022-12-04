import torch
import torch.nn as nn
from torch.nn import functional as F
from my_models.modules.base_network import BaseNetwork



class TwoPathModel(nn.Module):
    def __init__(self, in_channels, num_modality, n_classes=2):
        super(TwoPathModel, self).__init__()
        self.num_modality = num_modality
        self.in_channels = in_channels
        self.n_classes = n_classes
        # self.multi_task = multi_task

        self.base_network_1 = BaseNetwork(in_channels)
        if num_modality == 2:
            print('The model is multi modal')
            self.base_network_2 = BaseNetwork(in_channels)
            self.classifire = nn.Conv3d(in_channels=448, out_channels=n_classes, kernel_size=(3,3,3), stride=1, padding=1)
        else:
            self.classifire = nn.Conv3d(in_channels=224, out_channels=n_classes, kernel_size=(3,3,3), stride=1, padding=1)
            print('The model is single modal')
        # if multi_task:
        #     self.regresion = nn.Leanier()



    def forward(self, x):

        if self.num_modality == 2:
            flair_features = self.base_network_1(x[:,:,:,:,:,0])
            t1_features = self.base_network_2(x[:,:,:,:,:,1])
            features = torch.cat([flair_features, t1_features], dim=1)
            del flair_features, t1_features
        else:
            features = self.base_network_1(x)
            
        logit = self.classifire(features)

        del features

        return logit


        


