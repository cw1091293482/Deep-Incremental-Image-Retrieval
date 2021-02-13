
import torch
import torch.nn as nn
import torch.nn.functional as F

class Similarity_preserving(nn.Module):
    def __init__(self, num_class=0):
        super(Similarity_preserving, self).__init__()
        self.use_gpu = True
        self.T = 4

    def forward(self, FeaT, FeaS):

        batch_size = FeaT.size()[0]

        # calculate the similar matrix
        if self.use_gpu:
            Sim_T = torch.mm(FeaT, FeaT.t()).type(torch.cuda.FloatTensor) # label-free
            Sim_S = torch.mm(FeaS, FeaS.t()).type(torch.cuda.FloatTensor)
        else:
            Sim_T = torch.mm(FeaT, FeaT.t()).type(torch.FloatTensor)
            Sim_S = torch.mm(FeaS, FeaS.t()).type(torch.FloatTensor)


        # kl divergence
        p_s = F.log_softmax(Sim_S / self.T, dim=1)
        p_t = F.softmax(Sim_T / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / Sim_S.shape[0]

        return loss
