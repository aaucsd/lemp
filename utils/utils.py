import numpy as np
from PIL import Image
import torch
import random


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def create_dot_dict(**kwargs):
    return DotDict(dict(**kwargs))    

    
def save_gif(imgs, gif_name, duration=50):
    # Setup the 4 dimensional array
    a_frames = []
    for img in imgs:
        a_frames.append(np.asarray(img))
    a = np.stack(a_frames)

    ims = [Image.fromarray(a_frame) for a_frame in a]
    ims[0].save(gif_name, save_all=True, append_images=ims[1:], loop=0, duration=duration)
    
    
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def to_np(tensor):
    return tensor.data.cpu().numpy()