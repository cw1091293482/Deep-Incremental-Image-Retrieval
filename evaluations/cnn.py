# from collections import OrderedDict

# from torch.autograd import Variable
from utils import to_torch
import torch
from torch.autograd import Variable



def extract_cnn_feature(model, inputs, pool_feature=False):

    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs)
        inputs = Variable(inputs).cuda()
        if pool_feature is False:
            outputs, _, _ = model(inputs)
            return outputs
        else:
            # Register forward hook for each module
            outputs = {}

        def func(m, i, o): outputs['pool_feature'] = o.data.view(n, -1)
        hook = model.module._modules.get('features').register_forward_hook(func)
        model(inputs)
        hook.remove()
        # print(outputs['pool_feature'].shape)
        return outputs['pool_feature']

    
