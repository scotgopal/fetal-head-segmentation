from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

from utils import show_img_mask

if __name__ == "__main__":

    path2train = Path("data/training_set").resolve()
    imgsList = []
    anntsList = []
    for pp in list(path2train.glob("*")):
        if ".png" in str(pp):
            if "Annotation" in str(pp): anntsList.append(pp)
            else: imgsList.append(pp)
        else:
            print(pp)

    print(f"total images: {len(imgsList)}")
    print(f"total annotations: {len(anntsList)}")

    np.random.seed(0)
    random_image_paths = np.random.choice(imgsList,4)
    random_image_paths

    for image_path in random_image_paths:
        annt_path = str(image_path).replace(".png", "_Annotation.png")
        img = Image.open(image_path)
        annt_edges = Image.open(annt_path)
        mask = ndimage.binary_fill_holes(annt_edges)

        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.title(image_path.stem)
        plt.imshow(
            img, 
            cmap="gray"
            )

        plt.subplot(1,3,2)
        plt.title(str(image_path.stem)+"_mask")
        plt.imshow(
            mask, 
            cmap="gray"
            )

        plt.subplot(1,3,3)
        plt.title(image_path.stem)
        show_img_mask(img,mask)

    plt.show()