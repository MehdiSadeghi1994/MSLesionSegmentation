import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F


def to_one_hot(tensor,nClasses):
  n,c,h,w = tensor.size()
  one_hot = torch.zeros(n,nClasses,h,w).cuda().scatter_(1,tensor.view(n,1,h,w),1)
  return one_hot
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.BCE = nn.CrossEntropyLoss()


    def forward(self, inputs, targets):

        BCE_loss = self.BCE(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return F_loss



class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1, alpha=0.5, beta=0.5, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer

        inputs = F.sigmoid(inputs)    

        p_class0 = inputs[:,0,:,:,:]
        p_class1 = inputs[:,1,:,:,:]

        #flatten label and prediction tensors
        p_class0 = torch.flatten(p_class0)
        p_class1 = torch.flatten(p_class1)

        targets = torch.flatten(targets)
    
        #True Positives, False Positives & False Negatives
        TP = (p_class1 * targets).sum()    
        FP = ((1-targets) * p_class1).sum()
        FN = (targets * p_class0).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky



class FocalTverskyLoss2(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1, alpha=0.5, beta=0.5, gamma=0.75, weights=[1,1]):
        super(FocalTverskyLoss2, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weights = weights

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        label_1 = targets[:,:,:,:, 0]
        label_2 = targets[:,:,:,:, 1]

        xor_labels = torch.logical_xor(label_1, label_2)
        xnor_labels = torch.logical_not(xor_labels)

        coef = xor_labels * self.weights[0] + xnor_labels * self.weights[1]
        targets = torch.logical_or(label_1, label_2).long()

        inputs = F.sigmoid(inputs)    

        p_class0 = inputs[:,0,:,:,:]
        p_class1 = inputs[:,1,:,:,:]

        #flatten label and prediction tensors
        p_class0 = torch.flatten(p_class0)
        p_class1 = torch.flatten(p_class1)

        targets = torch.flatten(targets)
        coef = torch.flatten(coef)
        
        #True Positives, False Positives & False Negatives
        TP = (p_class1 * targets * coef).sum()    
        FP = ((1-targets) * p_class1 * coef).sum()
        FN = (targets * p_class0 * coef).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky

















class TwoLabelLoss(nn.Module):
    def __init__(self, eta = 0.5):
        super(TwoLabelLoss, self).__init__()
        self.eta = eta
        self.tevloss = FocalTverskyLoss()

    
    def forward(self, prediction, targets):

        label_1 = targets[:,:,:,:, 0]
        label_2 = targets[:,:,:,:, 1]

        xnor_labels = torch.logical_xnor(label_1, label_2)
        xor_labels = torch.logical_xor(label_1, label_2)
        

        # and_predicts =  inputs * 
        # loss_1 = self.tevloss(inputs, label_1)
        # loss_2 = self.tevloss(inputs, label_2)

        # twolabelsloss = self.eta * loss_1 + (1-self.eta) * loss_2
        return twolabelsloss


        


            