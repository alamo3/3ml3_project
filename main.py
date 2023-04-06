import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2

from kitti_loader.dataset_kitti import KittiDataset
from unet import UNET
from tqdm import tqdm
from utils.util import *

from params import *

import matplotlib.pyplot as plt

model_save_number = 0


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    model.train()
    loss_final = 1000
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.squeeze(1).long().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            loss_final = loss

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

    return loss_final


def get_transforms():
    train_transform = A.Compose(
        [
            A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            ToTensorV2()
        ],
        additional_targets={'image1': 'image'}
    )

    val_transform = A.Compose(
        [
            A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
            ToTensorV2()
        ],
        additional_targets={'image1': 'image'}
    )

    return train_transform, val_transform


def main():
    torch.cuda.empty_cache()

    train_transform, val_transform = get_transforms()

    model = UNET(in_channels=3, out_channels=45).to(device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = get_training_loader(kitti_dataset_dir=KITTI_DATASET_PATH,
                                                   batch_size=BATCH_SIZE, train_transform=train_transform,
                                                   val_transform=val_transform, num_workers=NUM_WORKERS,
                                                   pin_memory=True)

    loss_min = 3000000
    for epoch in range(NUM_EPOCHS):
        loss_after_epoch = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        save_checkpoint(checkpoint, filename='monkey_2.pth.tar')


def test_checkpoint():
    torch.cuda.empty_cache()
    file_name = 'monkey.pth.tar'
    checkpoint = torch.load(file_name, map_location=DEVICE)

    model = UNET(in_channels=3, out_channels=45)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint['optimizer'])

    with torch.no_grad():
        model.eval()

        train_transform, val_transform = get_transforms()

        data_loader = KittiDataset(KITTI_DATASET_PATH)

        train_loader, val_loader = get_training_loader(kitti_dataset_dir=KITTI_DATASET_PATH,
                                                       batch_size=BATCH_SIZE, train_transform=train_transform,
                                                       val_transform=val_transform, num_workers=NUM_WORKERS,
                                                       pin_memory=True)

        accuracy_calculator(model=model, loader=val_loader, num_classes=45, use_important_labels_only=True)

        img, label = data_loader[6332]
        img_tensor = val_transform(image=img)['image'].float().unsqueeze(0).to(device=DEVICE)

        predictions = model(img_tensor)

        out_classes = get_output_classes(predictions).cpu()

        fig, axs = plt.subplots(2)
        axs[0].imshow(out_classes.permute(1, 2, 0))
        axs[1].imshow(img)
        plt.show()


if __name__ == "__main__":
    test_checkpoint()
