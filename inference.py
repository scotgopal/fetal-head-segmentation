from random import random
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from model import SegNet
from utils import show_img_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# UNCOMMENT THE FOLLOWING TO ENFORCE CPU
# device = torch.device("cpu")

# Instantiating model
h, w = 128, 192
params_model = {"input_shape": (1, h, w), "initial_filters": 16, "num_outputs": 1}
path2weights = Path("models/100epochs_weights.pt").resolve()

model = SegNet(params_model)
model.load_state_dict(torch.load(path2weights))
model.eval()  # Set to eval mode
model.to(device=device)

# Prepare test set
path2test = Path("data/test_set").resolve()
imgsList = []
anntsList = []
for pp in list(path2test.glob("*")):
    if ".png" in str(pp):
        if "Annotation" in str(pp):
            anntsList.append(pp)
        else:
            imgsList.append(pp)
    else:
        print(pp)

print(f"total images: {len(imgsList)}")
print(f"total annotations: {len(anntsList)}")

np.random.seed(0)
random_image_paths = np.random.choice(imgsList, 4)  # Randomly get 4 image paths

# Model inferencing
for image_path in random_image_paths:
    image_pil = Image.open(image_path).resize((w, h))
    image_tensor = (to_tensor(image_pil) * 255).unsqueeze(0).to(device=device)
    with torch.no_grad():
        pred = model(image_tensor)
    pred = torch.sigmoid(pred)[0].cpu()
    mask_pred = pred[0] >= 0.5

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_pil, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_pred, cmap="gray")

    plt.subplot(1, 3, 3)
    show_img_mask(image_pil, mask_pred)
    plt.show()
