
import numpy as np
from tsne_official import *
import matplotlib.pyplot as plt
import time
# def main_plot(Fea_ref, Fea_w_MMD, Fea_wo_MMD):
#
#     import pylab
#
#     seleted_num = 108  # selected sample number to display
#     num_per_class = 18
#     concat_fea = np.concatenate((Fea_ref, Fea_w_MMD, Fea_wo_MMD), axis=0)
#
#     color_map = ['b', 'g', 'm', 'r', 'k', 'y']
#
#     color_list = [color_map[i/num_per_class] for i in range(seleted_num) if i < seleted_num]
#     Y = tsne(concat_fea, 2, 50, 20.0)  # no_dims=2, initial_dims=50, perplexity=30.0   Y[seleted_num+seleted_num : 2]
#
#     pylab.scatter(Y[:seleted_num, 0], Y[:seleted_num, 1], 50, c=color_list, marker="o", cmap='jet')  # , label='Reference'
#     pylab.hold(True)
#     pylab.scatter(Y[1 * seleted_num : 2 * seleted_num, 0], Y[1 * seleted_num:2 * seleted_num, 1], 50, c=color_list, marker="v", cmap='jet')  # label='w MMD'
#     pylab.title('Visualization of the original classes on the CUB-Birds dataset')
#
#     pylab.savefig('t-SNE1.pdf')
#     pylab.show()
#
#     pylab.scatter(Y[:seleted_num, 0], Y[:seleted_num, 1], 50, c=color_list, marker="o", cmap='jet')  # label='Reference'
#     pylab.hold(True)
#     pylab.scatter(Y[2 * seleted_num:, 0], Y[2 * seleted_num:, 1], 50, c=color_list, marker="v", cmap='jet')  #label='w/o MMD'
#     pylab.title('Visualization of the original classes on the CUB-Birds dataset')
#
#     pylab.savefig('t-SNE2.pdf')
#     pylab.show()

def main_plot(Fea_step2, Fea_interpolated, embed_feat_frozen_ori, epoch):
    import pylab

    batch_size = Fea_step2.shape[0]

    Fea_step2 = Fea_step2.detach().cpu().numpy()
    Fea_interpolated = Fea_interpolated.detach().cpu().numpy()
    embed_feat_frozen_ori = embed_feat_frozen_ori.detach().cpu().numpy()


    seleted_num = 50  # selected sample number to display
    num_per_class = 5
    concat_fea = np.concatenate((Fea_step2, Fea_interpolated, embed_feat_frozen_ori), axis=0)

    color_map = ['b', 'g', 'm', 'r', 'k', 'y', 'c', 'brown', 'pink', 'gray'] # 10 color, 10 classes

    color_list = [color_map[int(i / num_per_class)] for i in range(seleted_num) if i < seleted_num]
    Y = tsne(concat_fea, 2, 50, 30.0)  # no_dims=2, initial_dims=50, perplexity=30.0   Y[seleted_num+seleted_num : 2]

    pylab.scatter(Y[:seleted_num, 0], Y[:seleted_num, 1], 50, c=color_list, marker="o", cmap='jet')  # feature distribution of step2

    pylab.scatter(Y[batch_size : batch_size + seleted_num, 0], Y[batch_size : batch_size + seleted_num, 1], 50, c=color_list, marker="v", cmap='jet')  # label='w MMD'

    pylab.scatter(Y[2*batch_size: 2*batch_size + seleted_num, 0], Y[2*batch_size: 2*batch_size + seleted_num, 1], 50, c=color_list, marker="s", cmap='jet')  # label='w MMD'

    pylab.title('Distribution visualization of the real feature and the interpolated feature')


    # pylab.savefig('./Tsne_fig/t-SNE-epoch-' + str(epoch) + '.pdf')
    # pylab.clf()
    pylab.show()
    pylab.close()

