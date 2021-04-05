
# Deep-Incremental-Image-Retrieval
A Pytorch source code for ***Feature Estimations based Correlation Distillation for Incremental Image Retrieval*** published on
*IEEE Transactions on Multimedia*.
# Dependency:
Pytorch=1.4.0  
Before creating a new environment via the file environment.yml, just use your own pytorch environment to check if this code can be run.

# About datasets

In folder *data*, two datasets are named as *CUB_200_2011* and *Stanforddog120*, respectively.  
Under each folder for each dataset, there are train.txt, test.txt, and two sub-folders to include training images and test images,  
When you have download [UCSD Birds-200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [Stanford-dogs-120](http://vision.stanford.edu/aditya86/ImageNetDogs/),  
just put ***all categories and images*** into the *./train/* folder and *./test/* folder. The training set and testing set have been split via *./train.txt* and *./test.txt*.  
See the .PNG file in each folder.

# Backbone CNN
Please specify the CNN *model_path* in .*./models/BN_Inception.py*:

model_path = 'Path to Model/models/bn_inception-52deb4733.pth'

Download the model from [bn_inception-52deb4733](https://drive.google.com/file/d/1qDBfquYrfM9Msl2q57jxzl9w0y7qwnn0/view?usp=sharing), then put it into the folder *./models*.

# Specify path

Specify the Project path in *train.py*:

Dir_path = 'Path of Project/Deep_Incremental_Retrieval/'



# Intial training in the first stage

(1), in script *train.py,* set --_Incremental_flag_ to *False*  
(2), set --*resume* to *None*.
(3), in script ./Dataset/CUB200.py
 Comment and uncomment:

        labels_select = [labels[i] for i in range(len(labels)) if labels[i] <= 99] # 61, 71, 81, 91, 101 for flower-102
        images = [images[i] for i in range(len(labels)) if labels[i] <= 99]
        
        #labels_select = [labels[i] for i in range(len(labels)) if labels[i] > 99 and labels[i] <= 199] # 59,74,89,104,119 dog
        #images = [images[i] for i in range(len(labels)) if labels[i] > 99 and labels[i] <= 199] # 99, 124, 149, 174, 199 cub 

Note that *99* is for the first 100 classes on UCSD Birds-200, and 59 is for the first 60 classes on Stanford-Dogs-120  
For *Standford_dog.py*, there is no any changes.  
In this stage, an initial model will be trained and saved, according the number of training epochs.  
For example, when training on the first *100* classes on the Caltech-UCSD Birds-200 dataset

under the directory, such as
[ckp_ep1500.pth.tar](ckps/HardMining/cub/BN_Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80/ckp_ep1500.pth.tar)

This saved model will be used as the *teacher* model, and have its parameters fixed.
# Incremental training in the second stage

In this stage, the teacher model is loaded via --*resume* in script *train.py*
(Keep consistent the model directory)
Also:  
set --*Incremental_flag* to *True*

        #labels_select = [labels[i] for i in range(len(labels)) if labels[i] <= 99] # 61, 71, 81, 91, 101 for flower-102
        #images = [images[i] for i in range(len(labels)) if labels[i] <= 99]
        
        labels_select = [labels[i] for i in range(len(labels)) if labels[i] > 99 and labels[i] <= 199] # 59,74,89,104,119 dog
        images = [images[i] for i in range(len(labels)) if labels[i] > 99 and labels[i] <= 199] # 99, 124, 149, 174, 199 cub 

The training epochs (e.g. 2500) in second stage should be greater than that of the first stage (e.g. 1500)  
so the saved student model is named starting from the training epochs in the first stage (e.g. 1500)

If the remaining classes on a considered dataset are added at once, there are only two stages. Because all new classes are added
at once, there is no feature estimation involved, so in the *train.py*: *sequential_inteplatation = False*.

If added in different groups successively, the second stage is *repeated* until all remaining classes
are added. Unfortunately, in this case, I didn't make it train in an end-to-end way.
Therefore, when you are using the feature estimation method, the hyper-parameter *accuracy change* needs to be calculated and added *manually* to
the following code:

random_samp = np.random.uniform(***low=-0.0442811***, ***high=0.2230079***, size=(embed_feat_frozen.shape[0], embed_feat_frozen.shape[1]))

in *trainer.py.*

At the same time, set

*sequential_inteplatation = True*

in *trainer.py.*



# Acknowledgements
Appreciate the codes release from WangXun from: https://github.com/bnu-wangxun/Deep_Metric  
if use this code, please consider citing the papers:

@article{chen2021feature,  
  title={Feature Estimations based Correlation Distillation for Incremental Image Retrieval},  
  author={Wei Chen and Yu Liu and Nan Pu and Weiping Wang and Li Liu and Lew Michael S},  
  journal={IEEE Transactions on Multimedia},  
  year={2021},  
}

@inproceedings{wang2019multi,  
title={Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning},  
author={Wang Xun and Han Xintong and Huang Weilin and Dong Dengke and Scott Matthew R},  
booktitle={CVPR},  
year={2019}  
}
