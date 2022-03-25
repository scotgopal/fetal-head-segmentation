import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def show_img_mask(img, mask):
    img_mask = mark_boundaries(np.array(img), np.array(mask), outline_color=(0, 1, 0))
    plt.imshow(img_mask)


def get_optim(model):
    """Return Adam optimizer"""
    return optim.Adam(model.parameters(), lr=3e-4)


def get_lr_scheduler(optimizer: optim.Optimizer):
    """Return ReduceLROnPlateau scheduler"""
    return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, verbose=1)


def get_lr(optimizer: optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def compare_state_dicts(state_dict1, state_dict2):
    """Check the similarity between two Torch state_dicts"""
    state_dicts_differ = 0
    for key_item_1, key_item_2 in zip(state_dict1.items(), state_dict2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            state_dicts_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print(f"Mismatch found at {key_item_1[0]}")
            else:
                raise Exception

    if state_dicts_differ == 0:
        print("State_dicts match perfectly")
