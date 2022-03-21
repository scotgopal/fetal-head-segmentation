import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from scipy import ndimage as ndi
import numpy as np

class fetal_dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        all_paths = list(data_dir.glob("*"))
        self.image_paths = []
        self.annotation_paths = []
        for im_path in all_paths:
            if "Annotation" not in str(im_path): self.image_paths.append(im_path)
            else: self.annotation_paths.append(im_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = np.array(Image.open(image_path))

        annotation_path = self.annotation_paths[index]
        annotation_image = Image.open(annotation_path)
        
        mask = ndi.binary_fill_holes(annotation_image)
        mask = np.array(mask, dtype="uint8") # change type 'bool' to 'uint8'
         
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        image = to_tensor(image)*255
        mask = to_tensor(mask)*255

        return(image, mask)

if __name__ == "__main__":
    from pathlib import Path
    from transformers import transform_train, transform_val
    import matplotlib.pyplot as plt

    from utils import show_img_mask

    data_dir = Path("./data/training_set").resolve()
    fetal_train = fetal_dataset(data_dir, transform_train)
    fetal_val = fetal_dataset(data_dir, transform_val)

    img, mask = fetal_train[0]
    print(img.shape, img.type(), torch.max(img))
    print(mask.shape, mask.type(), torch.max(mask))
     
    img = torch.squeeze(img).type(torch.uint8) # change size [1,128,192] to [128,192] and to the correct type
    mask = torch.squeeze(mask).type(torch.uint8) # uint8 is unsigned integer; only positive integers allowed in this type
    show_img_mask(img, mask)
    plt.show()

