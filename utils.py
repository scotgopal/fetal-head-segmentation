import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

def show_img_mask(img, mask):
    img_mask = mark_boundaries(
        np.array(img),
        np.array(mask),
        outline_color=(0,1,0)
        )
    plt.imshow(img_mask)