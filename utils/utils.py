import numpy as np
from PIL import Image

def save_gif(imgs, gif_name):
    # Setup the 4 dimensional array
    a_frames = []
    for img in imgs:
        a_frames.append(np.asarray(img))
    a = np.stack(a_frames)

    ims = [Image.fromarray(a_frame) for a_frame in a]
    ims[0].save(gif_name, save_all=True, append_images=ims[1:], loop=0, duration=50)    