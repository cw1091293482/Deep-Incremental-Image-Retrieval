from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


device = torch.device("cuda:{}".format(0)
                      if torch.cuda.is_available() else "cpu")

def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim

def NCE_regularize(pos_pair, neg_pair):
    '''
     paper: Continual learning of objects instances
    :param pos_pair: the inner product of the positive pair, just one pair
    :param neg_pair: the inner product of the negative pair, nine pairs
    :return: the normalized cross-entropy loss
    '''

    combined = torch.cat((pos_pair, neg_pair), dim=0)

    neg = torch.sum(torch.exp(neg_pair))
    pos = [-torch.log2(torch.exp(combined[i])/neg) for i in range(combined.size(0))]

    return sum(pos)/combined.size(0)


class TripletCEAdapt(nn.Module):
    """
    From: https://github.com/ProsusAI/continual-object-instances/blob/master/src/loss_utils.py
    Triplet loss adaptor for Cross-Entropy
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    Returns predictions and targets (positive or negative indicator)
    """
    def __init__(self, device, neg_samples):
        super(TripletCEAdapt, self).__init__()
        self.device = device
        self.neg_samples = neg_samples

    def forward(self, anchor_pos, anchor_neg):

        predictions = torch.cat((anchor_pos, anchor_neg), dim=0)

        # shuffle predictions
        random_pos_idx = torch.randperm(self.neg_samples+1)
        random_pos_idx = random_pos_idx.to(device)

        shuffled_preds = torch.gather(predictions, 0, random_pos_idx)

        # print('predictions', predictions)
        # print('shuffled_preds', shuffled_preds)
        # print('(random_pos_idx == 0), ', (random_pos_idx == 0))
        # print(random_pos_idx)

        # Generate Targets
        targets = (random_pos_idx == 0).nonzero()[0]  # 0 because we concatenate with pos in id = 0, encoding the one hot encoding to label
        targets = targets.to(device)

        return shuffled_preds.reshape(1, -1), targets

class HardMiningLoss(nn.Module):
    def __init__(self, beta=None, margin=0, **kwargs):
        super(HardMiningLoss, self).__init__()
        self.beta = beta
        self.margin = 0.1

    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets

        base = 0.5
        loss = list()
        nce_loss_ = list()
        neg_samples = 9
        temperature = 1
        ce_criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        c = 0

        for i in range(n):

            pos_pair_ = torch.masked_select(sim_mat[i], targets==targets[i])

            ## the anchor itself
            anchor_ = torch.masked_select(pos_pair_, pos_pair_ == 1)

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets!=targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_nce = torch.sort(neg_pair_, descending=True)[0] ## for NCE loss
            neg_pair_ = torch.sort(neg_pair_)[0]

            triplet_criterion = TripletCEAdapt(device, neg_samples)
            # print(pos_pair_)
            # print('pos_pair_[1:2]', pos_pair_[0:1])
            # print(neg_pair_nce)
            # print('neg_pair_nce[0:9]', neg_pair_nce[0:9])

            shuffled_preds, gene_label = triplet_criterion(pos_pair_[0:1], neg_pair_nce[0:neg_samples])
            losses = ce_criterion(shuffled_preds.data / temperature, gene_label.data)

            nce_loss = NCE_regularize(pos_pair_[0:1], neg_pair_[0:9])

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > pos_pair_[0] - self.margin)
            pos_pair = torch.masked_select(pos_pair_, pos_pair_ < neg_pair_[-1] + self.margin)

            # pos_pair = pos_pair[1:]
            if len(neg_pair) < 1:
                c += 1
                continue

            pos_loss = torch.mean(1 - pos_pair)
            neg_loss = torch.mean(neg_pair)
            # pos_loss = torch.mean(torch.log(1 + torch.exp(-2*(pos_pair - self.margin))))
            # neg_loss = 0.04*torch.mean(torch.log(1 + torch.exp(50*(neg_pair - self.margin))))
            loss.append(pos_loss + neg_loss)
            nce_loss_.append(losses)

        loss = sum(loss)/n
        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()

        nce_loss_ = sum(nce_loss_)/n
        # print('nce_loss_', nce_loss_)

        return loss, prec, mean_pos_sim, mean_neg_sim


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(HardMiningLoss()(inputs, targets))


if __name__ == '__main__':
    main()


