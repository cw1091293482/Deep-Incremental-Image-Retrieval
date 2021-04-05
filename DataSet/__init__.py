from .CUB200 import CUB_200_2011
from .Car196 import Cars196
from .Products import Products
from .In_shop_clothes import InShopClothes
from .flowers102 import flowers102
from .Mnist import mnist
from .Scene_15 import Scene15
from .OCT2017 import OCT2017
from .FGVC_aircraft import aircraft100
from .Stanford_dog import Stanforddog120

# from .transforms import *
import os 

__factory = {
    'cub': CUB_200_2011,
    'car': Cars196,
    'flw': flowers102,
    'craft': aircraft100,
    'dog': Stanforddog120,
    'mnist': mnist,
    'scene': Scene15,
    'oct': OCT2017,
    'product': Products,
    'shop': InShopClothes,
}


def names():
    return sorted(__factory.keys())

def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__

def create(name, root=None, *args, **kwargs):
    """
    Create a dataset instance.
    """
    if root is not None:
        root = os.path.join(root, get_full_name(name))

    if name not in __factory:

        raise KeyError("Unknown dataset:", name)
    return __factory[name](root=root, *args, **kwargs)
