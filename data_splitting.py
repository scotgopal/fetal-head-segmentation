from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Dataset


def split_train_test(dataset: Dataset):
    sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    indices = range(len(dataset))
    train_indices, test_indices = next(sss.split(indices))

    return train_indices, test_indices


if __name__ == "__main__":
    import torch
    from torch.utils.data import Subset

    from pathlib import Path
    from transformers import transform_train, transform_val
    import matplotlib.pyplot as plt

    from custom_dataset import fetal_dataset
    from utils import show_img_mask

    data_dir = Path("./data/training_set").resolve()
    fetal_train = fetal_dataset(data_dir, transform_train)
    fetal_val = fetal_dataset(data_dir, transform_val)

    train_indices, val_indices = split_train_test(fetal_train)

    train_ds = Subset(fetal_train, train_indices)
    print(len(train_ds))

    val_ds = Subset(fetal_val, val_indices)
    print(len(val_ds))

    # Visualize a sample image from train_ds
    plt.figure(figsize=(10, 5))
    for img, mask in train_ds:
        img = torch.squeeze(
            img
        )  # change size [1,128,192] to [128,192] and to the correct type
        mask = torch.squeeze(
            mask
        )  # uint8 is unsigned integer; only positive integers allowed in this type
        show_img_mask(img, mask)
        plt.title("Sample image from train_ds")
        plt.show()
        break

    for img, mask in val_ds:
        img = torch.squeeze(
            img
        )  # change size [1,128,192] to [128,192] and to the correct type
        mask = torch.squeeze(
            mask
        )  # uint8 is unsigned integer; only positive integers allowed in this type
        show_img_mask(img, mask)
        plt.title("Sample image from val_ds")
        plt.show()
        break
