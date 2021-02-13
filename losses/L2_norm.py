
import torch
import torch.nn as nn


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()

        return lossvalue