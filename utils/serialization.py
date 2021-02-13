from __future__ import print_function, absolute_import
import json
import os.path as osp
import shutil
from tqdm import tqdm
import numpy as np

import torch
from torch.nn import Parameter
import torch.nn.functional as F

from .osutils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def cross_entropy_new_class(outputs, targets, label_number=1, size_average=True, eps=1e-5):
    out=F.softmax(outputs, dim=1)
    tar = one_hot_embedding(targets, label_number)
    tar = tar.type(torch.cuda.FloatTensor)

    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)

    ce=-(tar*out.log()).sum(1)

    if size_average:
        ce=ce.mean()

    return ce

def cross_entropy_distillationbackup(outputs_ori, targets, exp=1, size_average=True, eps=1e-5):

    outputs = outputs_ori[:, 0:100]

    out=F.softmax(outputs, dim=1)
    tar=F.softmax(targets, dim=1)

    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)

    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()

    confusion_input = F.softmax(outputs, dim=1)

    confusion_input=confusion_input+eps/confusion_input.size(1)
    confusion_input=confusion_input/confusion_input.sum(1).view(-1,1).expand_as(confusion_input)
    balance_loss = -(confusion_input * (confusion_input.log())).sum(1)

    confusion_input_200 = F.softmax(outputs_ori[:, 100:200], dim=1)

    confusion_input_200=confusion_input_200+eps/confusion_input_200.size(1)
    confusion_input_200=confusion_input_200/confusion_input_200.sum(1).view(-1,1).expand_as(confusion_input_200)
    balance_loss_200 = -(confusion_input_200 * (confusion_input_200.log())).sum(1)

    balance_loss1 = F.relu(balance_loss - balance_loss_200 + 0.5)

    balance_loss1 = balance_loss1.mean()

    return ce, balance_loss1

def cross_entropy_distillation(outputs, targets, exp=1, size_average=True, eps=1e-5):

    out=F.softmax(outputs, dim=1)
    tar=F.softmax(targets, dim=1)

    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)

    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()

    confusion_input = F.softmax(outputs, dim=1)

    confusion_input=confusion_input+eps/confusion_input.size(1)
    confusion_input=confusion_input/confusion_input.sum(1).view(-1,1).expand_as(confusion_input)
    balance_loss = -(confusion_input * (confusion_input.log())).sum(1)

    balance_loss1 = balance_loss.mean()

    return ce, balance_loss1

def fisher_matrix_diag(t, x, y, model, criterion, sbatch=20):
    # https://github.com/joansj/hat/blob/master/src/utils.py

    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()

    # b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()

    images=torch.autograd.Variable(x, volatile=False)
    target=torch.autograd.Variable(y, volatile=False)
    # Forward and backward
    model.zero_grad()
    outputs=model.forward(images)
    loss=criterion(t,outputs[t],target)
    loss.backward()
    # Get gradients
    for n,p in model.named_parameters():
        if p.grad is not None:
            fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/x.size(0)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)

    return fisher