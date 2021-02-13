# coding=utf-8
from __future__ import print_function, absolute_import
import time
from utils import AverageMeter, orth_reg
import  torch
from torch.autograd import Variable
from torch.backends import cudnn
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from utils.serialization import cross_entropy_distillation, cross_entropy_new_class

from TSNE import *

cudnn.benchmark = True

def train(epoch, model, criterion, optimizer, train_loader, args):

    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()

    ce_loss = 0.0
    pair_loss, inter_, dist_ap, dist_an = 0.0, 0.0, 0.0, 0.0

    l2_loss_flag = False
    correlation_pre_flag = True

    freq = min(args.print_freq, len(train_loader))
    Incremental_flag = args.Incremental_flag

    for i, (inputs, labels, name) in enumerate(train_loader, 0):

        # wrap them in Variable
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()

        embed_feat, logits, conv_fea = model[0](inputs) # shape  (batch_size, --dim): (batch_size, 512)
        embed_feat_frozen, logits_frozen, conv_fea_forzen = model[1](inputs)  # shape  (batch_size, --dim): (batch_size, 512)

        # the task is not incremental learning
        if Incremental_flag == False:

            taskindex = 0  # the first tuple in self.task_classifier) is used for computing cross-entropy for label classification

            pair_loss, inter_, dist_ap, dist_an = criterion[0](embed_feat, labels) # set loss function1,  labels are for old task dataset
            # ce_loss = criterion[1](logits[taskindex], labels)  # set cross entropy loss function1

        if Incremental_flag == True:

            pair_loss, inter_, dist_ap, dist_an = criterion[0](embed_feat, labels)  # set loss function1, labels are for new task dataset

            if l2_loss_flag:
                l2_loss = criterion[2](embed_feat, embed_feat_frozen)

            if correlation_pre_flag:

                sequential_inteplatation = False ## for multiple-task scenario
                if sequential_inteplatation:

                    scaling_factor = 1

                    ### stage 1 incremental e.g. (111-120) for cub
                    random_samp = np.random.uniform(low=-0.0442811, high=0.2230079, size=(embed_feat_frozen.shape[0], embed_feat_frozen.shape[1]))
                    random_sample = Variable(torch.FloatTensor(random_samp)).cuda()

                    interpolated_feat_step1 = embed_feat_frozen + scaling_factor * (random_sample * embed_feat_frozen) ###0.5 *random_sample

                    #####for stage 1 incremental, e.g.(111-120) for cub
                    correlation_loss_step1 = criterion[4](embed_feat, interpolated_feat_step1)

                    ##### loss combination for stage 1 incremental, e.g. (111-120) for cub
                    correlation_simi_loss = 10 * correlation_loss_step1

                similarity_loss = criterion[3](embed_feat, embed_feat_frozen)

                loss_correlation = 10 * similarity_loss

        if Incremental_flag == True:
            loss = 1 * (pair_loss) + loss_correlation
        else:
            loss = 1 * (pair_loss)


        loss.backward()
        optimizer.step()

        losses.update(loss.item())

        if (i + 1) % freq == 0 or (i+1) == len(train_loader):
            print('Epoch: [{0:03d}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f} \t'.format
                  (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                   loss=losses))

    return accuracy.avg
