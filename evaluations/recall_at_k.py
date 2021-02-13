# coding : utf-8
from __future__ import division
from __future__ import absolute_import
import numpy as np
import torch
from utils import to_numpy
import time
import random
import heapq


def topK_visual(sim_mat, img_name, img_name_shuffled, specific_query, data='cub', query_ids=None, gallery_ids=None):
    """
    Visualize the k retrieved images, the only one query image is given and fixed,
    target images are selected from the gallery images, and return the names, IDsof the target images
    """

    k = 10

    show_retrieval_results = True

    rank = 1 - sim_mat # because we will sort its distance according to similarity in sim_mat
    rank = np.argsort(rank, axis=1)
    gallery_ids = np.asarray(gallery_ids)
    query_ids = np.asarray(query_ids)

    rank = to_numpy(rank)
    n_probe, n_gallery = rank.shape ### n_probe is 1 since only one query image is given

    for i in range(n_probe):
        if show_retrieval_results:
            retrieved_img = [img_name_shuffled[rank[i, :k][j]] for j in range(k) if j > 0]

            print('ID of query image is:', query_ids[i])
            print()
            print('The indices of retrieved {} images are {}.'.format(k, rank[i, :k]))
            print()
            print('The ID of retrieved {} images are {}.'.format(k, gallery_ids[rank[i, :k]]))
            print()
            print('The name of query image is {}'.format(img_name[specific_query]))
            print()
            print('The name of retrieved image are: {}'.format(retrieved_img))
            exit()

def mean_average_precision(sim_mat, img_name, data='cub', query_ids=None, gallery_ids=None):
    """Compute the Mean Average Precision.
    ---------------------------------------------------
    Inputs:
    distmat : numpy.ndarray
        The distance matrix. ``distmat[i, j]`` is the distance between i-th
        probe sample and j-th gallery sample.
    glabels : numpy.ndarray or None, optional
    plabels : numpy.ndarray or None, optional
    ---------------------------------------------------
    Outputs:
    out : numpy.ndarray, The MAP result
    ---------------------------------------------------
    """
    rank = 1 - sim_mat  # because we will sort its distance according to similarity in sim_mat

    show_retrieval_results = True

    rank = np.argsort(rank, axis=1)
    rank = to_numpy(rank)
    gallery_ids = np.asarray(gallery_ids)
    query_ids = np.asarray(query_ids)

    n_probe, n_gallery = rank.shape
    average_precision = 1.0 * np.zeros_like(query_ids)

    for i in range(n_probe):
        relevant_size = np.sum(gallery_ids == query_ids[i])
        hit_index = np.where(gallery_ids[rank[i, :]] == query_ids[i])

        # if show_retrieval_results:
        #     print('Query image is', img_name[query_ids[i]])
        #     relevant_size = hit_index[0].shape[0]
        #     retrieved_img = [img_name[hit_index[0][k]] for k in range(relevant_size)]
        #     print('retrieved image are: ', retrieved_img)
        #     exit()


        precision = 1.0 * np.zeros_like(hit_index[0])
        assert relevant_size == hit_index[0].shape[0]
        for j in range(relevant_size):
            hitid = np.max(hit_index[0][j])
            precision[j] = np.sum(gallery_ids[rank[i, :hitid]] == query_ids[i]) * 1.0 / (hit_index[0][j] + 1)
        average_precision[i] = np.sum(precision) * 1.0 / relevant_size

    score = np.mean(average_precision)

    return score

def Recall_at_ks(sim_mat, img_name, data='cub', query_ids=None, gallery_ids=None):
    # start_time = time.time()
    # print(start_time)
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids
    :param data

    Compute  [R@1, R@2, R@4, R@8]
    """

    ks_dict = dict()
    ks_dict['cub'] = [1, 2, 4, 8, 16, 32]
    ks_dict['car'] = [1, 2, 4, 8, 16, 32]
    ks_dict['flw'] = [1, 2, 4, 8, 16, 32]
    ks_dict['craft'] = [1, 2, 4, 8, 16, 32]
    ks_dict['mnist'] = [1, 2, 4, 8, 16, 32]
    ks_dict['dog'] = [1, 2, 4, 8, 16, 32]
    ks_dict['scene'] = [1, 2, 4, 8, 16, 32]
    ks_dict['oct'] = [1, 2, 4, 8, 16, 32]
    ks_dict['jd'] = [1, 2, 4, 8]
    ks_dict['product'] = [1, 10, 100, 1000]
    ks_dict['shop'] = [1, 10, 20, 30, 40, 50]

    if data is None:
        data = 'cub'
    k_s = ks_dict[data]

    sim_mat = to_numpy(sim_mat)
    m, n = sim_mat.shape # m and n are related with the size of test set, sim_mat is computed from the extracted features
    gallery_ids = np.asarray(gallery_ids) #gallery labels
    if query_ids is None:
        query_ids = gallery_ids
    else:
        query_ids = np.asarray(query_ids)

    num_max = int(1e6)

    if m > num_max:
        samples = list(range(m))
        random.shuffle(samples)
        samples = samples[:num_max]
        sim_mat = sim_mat[samples, :]
        query_ids = [query_ids[k] for k in samples]
        m = num_max

    num_valid = np.zeros(len(k_s)) # len(k_s)) = 6
    neg_nums = np.zeros(m)
    for i in range(m):
        x = sim_mat[i]

        pos_max = np.max(x[gallery_ids == query_ids[i]])#x[gallery_ids == query_ids[i]]:only when gallery_ids==query_ids[i] is true,value in x is got
        neg_num = np.sum(x > pos_max)# the number of negative samples
        neg_nums[i] = neg_num # if neg_num=0, indicates that all negtative samples are regarded as positive samples

    for i, k in enumerate(k_s):
        if i == 0:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp
        else:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp - num_valid[i-1]
    # t = time.time() - start_time
    # print(t)

    return num_valid / float(m)
