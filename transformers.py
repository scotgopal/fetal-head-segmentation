from albumentations import HorizontalFlip, VerticalFlip, Compose, Resize

height, width = 128, 192
transform_train = Compose(
    [
        Resize(height=height, width=width),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5)
        ]
    )
transform_val = Resize(height=height, width=width)

if __name__ == "__main__":
    # Visualize the transforms
    from pathlib import Path
    import  matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    from scipy import ndimage
    
    from utils import show_img_mask

    data_dir = Path("data/training_set").resolve()
    training_imgs_list = list(data_dir.glob("*"))
    img_path = training_imgs_list[0]
    annotation_path = str(img_path).replace(".png","_Annotation.png")

    image_pil = Image.open(img_path)
    annotation_pil = Image.open(annotation_path)
    mask = ndimage.binary_fill_holes(annotation_pil)
    mask = np.array(mask, dtype=np.uint8) # change type 'bool' to 'uint8'

    img_mask_transformed = transform_train(image=np.array(image_pil), mask=mask)
    img_transformed = img_mask_transformed['image']
    mask_transformed = img_mask_transformed['mask']

    plt.figure(figsize=(10,5))

    plt.subplot(3,2,1)
    plt.imshow(image_pil)
    plt.title("before transform image")

    plt.subplot(3,2,2)
    plt.imshow(img_transformed)
    plt.title("after transform image")

    plt.subplot(3,2,3)
    plt.imshow(mask)
    plt.title("before transform mask (hole filled)")
    
    plt.subplot(3,2,4)
    plt.imshow(mask_transformed)
    plt.title("after transform mask")

    plt.subplot(3,2,5)
    show_img_mask(image_pil, mask)
    plt.title("Image+mask before transform")
    
    plt.subplot(3,2,6)
    show_img_mask(img_transformed, mask_transformed)
    plt.title("Image+mask after transform")

    plt.show()