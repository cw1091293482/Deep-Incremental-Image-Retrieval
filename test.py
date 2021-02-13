# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
from Model2Feature import Model2Feature
from evaluations import pairwise_similarity, mean_average_precision, topK_visual
from utils.serialization import load_checkpoint
import torch 
import ast
import numpy as np
from TSNE import *
import os

# select_gpu = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = select_gpu
use_gpu = True

Dir_path = 'Path to project/Deep_Incremental_Retrieval/'

parser = argparse.ArgumentParser(description='Deep Metric Learning Testing')

parser.add_argument('--data', type=str, default='cub')
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--gallery_eq_query', '-g_eq_q', type=ast.literal_eval, default=True, help='Is gallery identical with query')
parser.add_argument('--net', type=str, default='BN_Inception')
parser.add_argument('--resume', '-r', type=str, default=Dir_path + 'ckps/HardMining/cub/BN_Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80/ckp_ep4700.pth.tar', metavar='PATH')
parser.add_argument('--dim', '-d', type=int, default=512, help='Dimension of Embedding Feather')
parser.add_argument('--width', type=int, default=224, help='width of input image')

parser.add_argument('--Incremental_flag',  default=False, type=bool, help='incremental learning or not')
parser.add_argument('--Retrieval_visualization', default=False, type=bool, help='Visualize the retrieved image for a given img')
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N', help='number of data loading threads (default: 2)')
parser.add_argument('--pool_feature', type=ast.literal_eval, default=False, required=False, help='if True extract feature from the last pool layer')

args = parser.parse_args()

checkpoint = load_checkpoint(args.resume)
print(args.pool_feature)

epoch = checkpoint['epoch']
print('Training Epoch:', epoch)

gallery_feature, gallery_labels, query_feature, query_labels, img_name, img_name_shuffled = \
    Model2Feature(data=args.data, root=args.data_root, width=args.width, net=args.net, checkpoint=checkpoint, Retrieval_visualization =args.Retrieval_visualization,
                   dim=args.dim, batch_size=args.batch_size, nThreads=args.nThreads, pool_feature=args.pool_feature)

if args.Retrieval_visualization:
    specific_query = 599  # the index of a query image
    sim_mat = pairwise_similarity(query_feature[specific_query:specific_query+1, :], gallery_feature) #query * gallery
else:
    sim_mat = pairwise_similarity(query_feature, gallery_feature) #query * gallery

if args.gallery_eq_query is True:
    sim_mat = sim_mat - torch.eye(sim_mat.size(0))

### for retrieval visual. the query image is given and fixed, args.Retrieval_visualization = True
if args.Retrieval_visualization:
    topK_visual(sim_mat, img_name, img_name_shuffled, specific_query, query_ids=query_labels[specific_query:specific_query+1], gallery_ids=gallery_labels, data=args.data)

recall_ks = Recall_at_ks(sim_mat, img_name, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)

result = '  '.join(['%.4f' % k for k in recall_ks])
print('Epoch-%d' % epoch, result)

topk = 800
map = mean_average_precision(sim_mat, img_name, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)
print('mAP={0:.4f}'.format(map))
