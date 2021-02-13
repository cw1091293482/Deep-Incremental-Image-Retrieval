
'''
Thanks for the code release from WangXun from: https://github.com/bnu-wangxun/Deep_Metric
if use this code, please consider the paper:

@inproceedings{wang2019multi,
title={Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning},
author={Wang, Xun and Han, Xintong and Huang, Weilin and Dong, Dengke and Scott, Matthew R},
booktitle={CVPR},
year={2019}
}

'''

# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
import models
import losses
from utils import FastRandomIdentitySampler, mkdir_if_missing, logging, display
from utils.serialization import save_checkpoint
from trainer import train
import torch.nn as nn

import DataSet
import os.path as osp
from losses.L2_norm import L2Norm
from losses.Similarity_preserving_loss import *

cudnn.benchmark = True

select_gpu = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = select_gpu
use_gpu = True

# Batch Norm Freezer : bring 2% improvement on CUB 
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def main(args):

    # s_ = time.time()
    save_dir = args.save_dir
    mkdir_if_missing(save_dir)

    sys.stdout = logging.Logger(os.path.join(save_dir, 'log.txt'))
    display(args)
    start = 0

    model = models.create(args.net, pretrained=True, dim=args.dim) #
    model_frozen = models.create(args.net, pretrained=True, dim=args.dim)  #

    # for vgg and densenet
    if args.resume is None:
        model_dict = model.state_dict()
    else:
        # resume model
        print('load model from {}'.format(args.resume))

        model_dict = model.state_dict()
        model_dict_frozen = model_frozen.state_dict()
        chk_pt = torch.load(args.resume)
        weight = chk_pt['state_dict']
        start = chk_pt['epoch']
        pretrained_dict = {k: v for k, v in weight.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        pretrained_dict_frozen = {k: v for k, v in weight.items() if k in model_dict_frozen}
        model_dict_frozen.update(pretrained_dict_frozen)
        model_frozen.load_state_dict(model_dict_frozen)
        model_frozen.eval()

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    model_frozen = torch.nn.DataParallel(model_frozen)
    model_frozen = model_frozen.cuda()

    # freeze BN
    if args.freeze_BN is True:
        print(40 * '#', '\n BatchNorm frozen')
        model.apply(set_bn_eval)
    else:
        print(40*'#', 'BatchNorm NOT frozen')
        
    # Fine-tune the model: the learning rate for pre-trained parameter is 1/10
    new_param_ids_fc_layer = set(map(id, model.module.fc_layer.parameters()))

    new_param_ids = new_param_ids_fc_layer

    new_params_fc = [p for p in model.module.parameters() if id(p) in new_param_ids_fc_layer]
    base_params = [p for p in model.module.parameters() if id(p) not in new_param_ids]

    frozen_params = [p for p in model_frozen.module.parameters()] # frozen the model, but with learning_rate = 0.0
    for p in frozen_params:
        p.requires_grad = False
            
    # if fine-tune basenetwork, then lr_mult: 0.1. if lr_mult=0.0, then the basenetwork is not updated
    param_groups = [
                {'params': base_params, 'lr_mult': 0.1},
                {'params': new_params_fc, 'lr_mult': 1.0}]

    print('initial model is save at %s' % save_dir)

    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    criterion_loss = losses.create(args.loss, margin=args.margin, alpha=args.alpha, base=args.loss_base).cuda()
    CE_loss = nn.CrossEntropyLoss().cuda()
    l2_loss = L2Norm().cuda()
    similarity_loss = Similarity_preserving().cuda()

    criterion = [criterion_loss, CE_loss, l2_loss, similarity_loss]

    # Decor_loss = losses.create('decor').cuda()
    data = DataSet.create(args.data, ratio=args.ratio, width=args.width, origin_width=args.origin_width, root=args.data_root)

    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.batch_size,
        sampler=FastRandomIdentitySampler(data.train, num_instances=args.num_instances),
        drop_last=True, pin_memory=True, num_workers=args.nThreads)

    # save the train information
    best_accuracy = 0

    model_list = [model, model_frozen]

    if args.Incremental_flag == False:
        print("######################This is non-incremental learning! ########################")
    if args.Incremental_flag == True:
        print("#########################This is incremental learning! #########################")

    else:
        NotImplementedError()

    for epoch in range(start, args.epochs):

        accuracy = train(epoch=epoch, model=model_list, criterion=criterion, optimizer=optimizer, train_loader=train_loader, args=args)

        if (epoch+1) % args.save_step == 0 or epoch==0:
            if use_gpu:
                state_dict = model.module.state_dict() # save the parameters from updated model
            else:
                state_dict = model.state_dict()

            is_best = accuracy > best_accuracy
            best_accuracy = max(accuracy, best_accuracy)

            save_checkpoint({
                'state_dict': state_dict,
                'epoch': (epoch+1),
            }, is_best, fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))


if __name__ == '__main__':

    Dir_path = 'Path of Project/Deep_Incremental_Retrieval/'
    parser = argparse.ArgumentParser(description='Incremental Fine-grained Image Retrieval')

    # hype-parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of new parameters")
    parser.add_argument('--batch_size', '-b', default=80, type=int, metavar='N', help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('--num_instances', default=5, type=int, metavar='n', help=' number of samples from one class in mini-batch')
    parser.add_argument('--dim', default=512, type=int, metavar='n',  help='dimension of embedding space')
    parser.add_argument('--num', default=59, type=int, metavar='n', help='The number of class for validation [other datasets]')
    parser.add_argument('--width', default=224, type=int, help='width of input image')
    parser.add_argument('--origin_width', default=256, type=int, help='size of origin image')
    parser.add_argument('--ratio', default=0.16, type=float, help='random crop ratio for train data')

    parser.add_argument('--alpha', default=30, type=int, metavar='n', help='hyper parameter in NCA and its variants')
    parser.add_argument('--beta', default=0.1, type=float, metavar='n', help='hyper parameter in some deep metric loss functions')
    parser.add_argument('--orth_reg', default=1, type=float, help='hyper parameter coefficient for orth-reg loss')
    parser.add_argument('-k', default=16, type=int, metavar='n', help='number of neighbour points in KNN')
    parser.add_argument('--margin', default=0.5, type=float, help='margin in loss function')
    parser.add_argument('--init', default='random', help='the initialization way of FC layer')

    # network if Incremental_flag= False, the network can be used for training in a single dataset, but keep the first scalar in [(0, 100), (1, 98)] be the class number of this single dataset
    parser.add_argument('--Incremental_flag', default=False, type=bool, help='incremental learning or not')
    parser.add_argument('--data', default='cub', help='name of Data Set') ### dataset
    parser.add_argument('--validatedata', default='cub', help='name of validation set')  ### dataset
    parser.add_argument('--freeze_BN', default=True, type=bool, required=False, metavar='N', help='Freeze BN if True')
    parser.add_argument('--data_root', type=str, default='data', help='path to Data Set')

    parser.add_argument('--net', default='BN_Inception') #
    parser.add_argument('--loss', default='HardMining', help='loss for training network')
    parser.add_argument('--epochs', default=2300, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('--save_step', default=50, type=int, metavar='N', help='number of epochs to save model')

    # Resume from checkpoint
    ## Dir_path + 'ckps/HardMining/cub/BN_Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80/ckp_ep200.pth.tar'
    parser.add_argument('--resume','-r', default=None) # None  or  'ckp_ep150.pth.tar'
    parser.add_argument('--resume_pre_step_2', default=Dir_path + 'ckps/HardMining/cub/BN_Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80/ckp_ep1500.pth.tar')  # None  or  'ckp_ep1500.pth.tar'

    # train
    parser.add_argument('--print_freq', default=6, type=int, help='display frequency of training')

    # basic parameter
    parser.add_argument('--save_dir', default='ckps/HardMining/cub/BN_Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80')
    parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N', help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    parser.add_argument('--loss_base', type=float, default=0.75)

    print('parser:', parser.parse_args())

    main(parser.parse_args())




