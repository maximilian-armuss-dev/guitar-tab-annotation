from torch.utils.data import random_split, DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn as nn


def split_dataset(ds, train_ratio=0.9):
    dataset_size = len(ds)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    split = [train_size, val_size]
    train_dataset, val_dataset = random_split(ds, split)
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    return train_dataset, val_dataset


def show_loss_info(train_loss_values, val_loss_values, title="Training and Validation Loss"):
    assert len(train_loss_values) == len(val_loss_values), \
        f"Length of train and validation must be equal, but were {len(train_loss_values)} and {len(val_loss_values)} respectively"
    epochs = range(1, len(train_loss_values)+1)
    plt.plot(epochs, train_loss_values, label="Training Loss")
    plt.plot(epochs, val_loss_values, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(False)
    plt.show()
    print(f"Training loss: {train_loss_values[-1]:.7f}")
    print(f"Validation loss: {val_loss_values[-1]:.7f}")


def model_montage(dataloader: DataLoader, model: nn.Module, show_num: int, scale: float = 3):
    """
    Display images from a dataset in a horizontal montage, captioned with labels predicted by a model.

    Args:
        dataloader: Dataloader object, returns a batch of images each iteration
        model: Model predicting the label given images from the dataset
        show_num: Number of images to display
        scale: Scaling factor for image size
    """
    assert show_num <= dataloader.batch_size
    for ims in dataloader:
        ims = ims.to(model.device)
        ims = ims[:show_num]
        pred_labels = model.predict(ims)
        montage(zip(ims, pred_labels), show_num=show_num, scale=scale)
    plt.show()


def montage_from_ds(ds: Dataset, show_num: int, scale: float = 3):
    """
    Display images and labels from a dataset in a horizontal montage.

    Args:
        ds: Dataset object, where ds[i] returns (image, label)
        show_num: Number of images to display
        scale: Scaling factor for image size
    """
    fig, axes = plt.subplots(1, show_num, figsize=(scale * 5, scale * show_num))
    for i in range(show_num):
        im, label = next(iter(ds))
        axes[i].imshow(im[0].cpu(), cmap='gray')
        axes[i].set_title(f"Label: {label.tolist()}")
        axes[i].axis('off')
    plt.show()


def montage(dataloader, show_num: int, scale: float = 3):
    fig, axes = plt.subplots(1, show_num, figsize=(scale * 5, scale * show_num))
    for i, data in enumerate(dataloader):
        if i >= show_num:
            break
        im, label = data
        axes[i].imshow(im[0].cpu(), cmap='gray')
        axes[i].set_title(label.tolist())
        axes[i].axis('off')
