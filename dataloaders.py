from torch.utils.data import Dataset, DataLoader

def get_dl(ds:Dataset, batch_size, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

if __name__=="__main__":
    from torch.utils.data import Subset

    from pathlib import Path
    from transformers import transform_train, transform_val

    from custom_dataset import fetal_dataset
    from utils import show_img_mask
    from data_splitting import split_train_test

    data_dir = Path("./data/training_set").resolve()
    fetal_train = fetal_dataset(data_dir, transform_train)
    fetal_val = fetal_dataset(data_dir, transform_val)

    train_indices, val_indices = split_train_test(fetal_train)
    
    train_ds=Subset(fetal_train, train_indices)
    print(len(train_ds))

    val_ds = Subset(fetal_val, val_indices)
    print(len(val_ds))

    train_dl = get_dl(train_ds, 8, True)

    for img, mask in train_dl:
        print(img.shape, img.dtype)
        print(mask.shape, mask.dtype)
        break
    
    val_dl = get_dl(val_ds, 16, False)

    for img, mask in val_dl:
        print(img.shape, img.dtype)
        print(mask.shape, mask.dtype)
        break