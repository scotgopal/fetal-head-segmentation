import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader


def dice_loss(pred, target, smooth=1e-5):
    """Function to calculate the dice loss per data batch"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    return loss.sum(), dice.sum()


def metrics_batch(pred, target):
    """Return dice.sum() as the validation metric since this is a segmentation task"""
    _, dice_sum = dice_loss(pred, target)
    return dice_sum


def loss_func(pred, target):
    """Calculate the combined loss (dice and BCE)"""
    bce_loss_value = F.binary_cross_entropy_with_logits(pred, target, reduction="sum")
    dice_loss_value, _ = dice_loss(pred, target)
    combined_loss = bce_loss_value + dice_loss_value
    return combined_loss


def loss_batch(loss_func, preds_batch, targets_batch, opt=None):
    """Return the total loss and validation metric of the most recent batch of data"""
    loss = loss_func(preds_batch, targets_batch)
    metric_b = metrics_batch(preds_batch, targets_batch)
    if opt is not None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return loss, metric_b


def loss_epoch(
    model,
    loss_func,
    dataset_dl: DataLoader,
    sanity_check: bool = False,
    opt: torch.optim.Optimizer = None,
    device=torch.device("cpu"),
):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for images_batch, targets_batch in dataset_dl:
        images_batch = images_batch.type(torch.float32).to(device)
        targets_batch = targets_batch.type(torch.float32).to(device)
        preds_batch = model(images_batch)
        loss_b, metric_b = loss_batch(loss_func, preds_batch, targets_batch, opt=opt)
        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b
        if sanity_check is True:
            break
    loss = running_loss.item() / float(len_data)
    metric = running_metric.item() / float(len_data)
    return loss, metric
