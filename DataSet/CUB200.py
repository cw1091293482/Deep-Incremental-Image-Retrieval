from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
from PIL import Image

import os
import sys
from DataSet import transforms 
from collections import defaultdict


def default_loader(path):
    return Image.open(path).convert('RGB')

def Generate_transform_Dict(origin_width=256, width=227, ratio=0.16):
    
    std_value = 1.0 / 255.0
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std= [1.0/255, 1.0/255, 1.0/255])

    transform_dict = {}

    transform_dict['rand-crop'] = \
    transforms.Compose([
                transforms.CovertBGR(),
                transforms.Resize((origin_width)),
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
               ])

    transform_dict['center-crop'] = \
    transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.Resize((origin_width)),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])
    
    transform_dict['resize'] = \
    transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.Resize((width)),
                    transforms.ToTensor(),
                    normalize,
                ])
    return transform_dict


class MyData(data.Dataset):
    def __init__(self, root=None, label_txt=None,
                 transform=None, loader=default_loader):

        # Initialization data path and train(gallery or query) txt path
        if root is None:
            self.root = "data/cub/"
        self.root = root
        
        if label_txt is None:
            label_txt = os.path.join(root, 'train.txt')

        if transform is None:
            transform_dict = Generate_transform_Dict()['rand-crop']

        file = open(label_txt)
        images_anon = file.readlines()

        images = []
        labels = []

        for img_anon in images_anon:
            [img, label] = img_anon.split(' ') ## img is the name of images
            images.append(img)
            labels.append(int(label))

        ######### cub: 200 (0~199)(100-100), dog: 120 (0~119) (60-60)
        ##### comment these two rows if training on the first stage, but uncomment them on the incremental stage (second stage)
        # labels_select = [labels[i] for i in range(len(labels)) if labels[i] <= 99] # 61, 71, 81, 91, 101 for flower-102
        # images = [images[i] for i in range(len(labels)) if labels[i] <= 99]

        ##### uncomment these two rows if training on the first stage, but comment them on the incremental stage (second stage)
        # ##### the new classes, added at once or sequentially    # 99, 123, 147, 171, 195, for car-196
        labels_select = [labels[i] for i in range(len(labels)) if labels[i] > 99 and labels[i] <= 199] # 59,74,89,104,119 dog
        images = [images[i] for i in range(len(labels)) if labels[i] > 99 and labels[i] <= 199] # 99, 124, 149, 174, 199 cub
        # 
        classes = list(set(labels_select))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels_select):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.images = images
        self.labels = labels_select
        self.classes = classes
        self.transform = transform
        self.Index = Index
        self.loader = loader


    def __getitem__(self, index):
        fn, label = self.images[index], self.labels[index]
        fn = os.path.join(self.root, fn)
        img_name = fn
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_name

    def __len__(self):
        return len(self.images)

class CUB_200_2011:
    def __init__(self, width=227, origin_width=256, ratio=0.16, root=None, transform=None, new_label_start_point=0):
        print('width: \t {}'.format(width))
        transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = "data/CUB_200_2011/"

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')

        self.train = MyData(root, label_txt=train_txt, transform=transform_Dict['rand-crop'])
        self.gallery = MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop'])



