# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from torch.backends import cudnn
from evaluations import extract_features
import models
import DataSet
from utils.serialization import load_checkpoint
cudnn.benchmark = True


def Model2Feature(data,net,checkpoint,dim=512, width=224, root=None, Retrieval_visualization = False, nThreads=16, batch_size=100, pool_feature=False, **kargs):
    dataset_name = data
    model = models.create(net, dim=dim, pretrained=False)
    # resume = load_checkpoint(ckp_path)

    resume = checkpoint
    # model.load_state_dict(resume['state_dict'])

    net_dict = model.state_dict()
    weights = resume['state_dict']
    pretrained_dict = {k: v for k, v in weights.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)


    model = torch.nn.DataParallel(model).cuda()
    data = DataSet.create(data, width=width, root=root)
    
    if dataset_name in ['shop', 'jd_test']:
        gallery_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=batch_size, shuffle=False,
            drop_last=False, pin_memory=True, num_workers=nThreads)

        query_loader = torch.utils.data.DataLoader(
            data.query, batch_size=batch_size,
            shuffle=False, drop_last=False,
            pin_memory=True, num_workers=nThreads)

        gallery_feature, gallery_labels, img_name = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        query_feature, query_labels, img_name = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)

    else:
        data_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=batch_size,
        shuffle=False, drop_last=False, pin_memory=True,
            num_workers=nThreads)

        ## if use the retrieval visualization, the dataset should be shuffled
        if Retrieval_visualization:
            data_loader_shuffled = torch.utils.data.DataLoader(
                data.gallery, batch_size=batch_size,
            shuffle=True, drop_last=False, pin_memory=True,
                num_workers=nThreads)

        else:
            data_loader_shuffled = torch.utils.data.DataLoader(
                data.gallery, batch_size=batch_size,
                shuffle=False, drop_last=False, pin_memory=True,
                num_workers=nThreads)

        features, labels, img_name = extract_features(model, data_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        features_shuffled, labels_shuffled, img_name_shuffled = extract_features(model, data_loader_shuffled, print_freq=1e5, metric=None, pool_feature=pool_feature)

        gallery_feature, gallery_labels = query_feature, query_labels = features, labels
        gallery_feature, gallery_labels = features_shuffled, labels_shuffled

    return gallery_feature, gallery_labels, query_feature, query_labels, img_name, img_name_shuffled

