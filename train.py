import torch
import copy

from utils import get_lr
from loss import loss_epoch


def train_val(model, params):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    device = params["device"]

    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print(f"Epoch {epoch+1}/{num_epochs}, current lr={current_lr}")

        model.train()
        train_loss, train_metric = loss_epoch(
            model, loss_func, train_dl, sanity_check, opt, device=device
        )
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(
                model, loss_func, val_dl, sanity_check, device=device
            )

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights")

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)
        print(f"train_loss: {train_loss:.6f} dice: {100*train_metric:.2f}")
        print(f"val_loss: {val_loss:.6f} dice: {100*val_metric:.2f}")
        print("-" * 10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


if __name__ == "__main__":
    from torch.utils.data import Subset
    from torchsummary import summary
    import matplotlib.pyplot as plt

    import os
    from pathlib import Path

    from transformers import transform_train, transform_val
    from custom_dataset import fetal_dataset
    from data_splitting import split_train_test
    from dataloaders import get_dl
    from model import SegNet
    from loss import loss_func
    from utils import get_optim, get_lr_scheduler

    # Creating data loaders
    data_dir = Path("./data/training_set").resolve()
    fetal_train = fetal_dataset(data_dir, transform_train)
    fetal_val = fetal_dataset(data_dir, transform_val)

    train_indices, val_indices = split_train_test(fetal_train)
    train_ds = Subset(fetal_train, train_indices)
    val_ds = Subset(fetal_val, val_indices)

    train_dl = get_dl(train_ds, 8, True)
    val_dl = get_dl(val_ds, 16, False)

    # Instantiating model
    h, w = 128, 192
    params_model = {"input_shape": (1, h, w), "initial_filters": 16, "num_outputs": 1}

    model = SegNet(params_model)

    ## move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = model.to(device)
    summary(model, input_size=params_model["input_shape"], device=device.type)

    # Initialize Optimizer and LR Scheduler
    opt = get_optim(model)
    lr_scheduler = get_lr_scheduler(opt)

    # Train model
    path2models = "./models/"
    if not os.path.exists(path2models):
        os.mkdir(path2models)
    params_train = {
        "num_epochs": 15,
        "optimizer": opt,
        "loss_func": loss_func,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "sanity_check": False,
        "lr_scheduler": lr_scheduler,
        "path2weights": path2models + "weights.pt",
        "device": device,
    }
    # torch.cuda.empty_cache()
    model, loss_hist, metric_hist = train_val(model, params_train)

    # Plot historical values of stored during training
    num_epochs = params_train["num_epochs"]
    plt.title("Train-Val Loss")
    plt.plot(range(1, num_epochs + 1), loss_hist["train"], label="train")
    plt.plot(range(1, num_epochs + 1), loss_hist["val"], label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

    plt.title("Train-Val Accuracy")
    plt.plot(range(1, num_epochs + 1), metric_hist["train"], label="train")
    plt.plot(range(1, num_epochs + 1), metric_hist["val"], label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()
