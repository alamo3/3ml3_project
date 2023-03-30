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

import torch.nn.functional as F

import matplotlib.pyplot as plt


# hyper parameters

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 10
NUM_WORKERS = 5
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
PIN_MEMORY = True
LOAD_MODEL = False

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.squeeze(1).long().to(device=DEVICE)

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main():
    torch.cuda.empty_cache()

    train_transform = A.Compose(
        [
            A.Resize(width=704, height=188),
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
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    model = UNET(in_channels=3, out_channels=45).to(device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    train_loader = get_training_loader(kitti_dataset_dir='G:/kitti/KITTI-360/', batch_size=BATCH_SIZE, train_transform=train_transform, val_transform=None, num_workers=NUM_WORKERS,pin_memory=True)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        save_checkpoint(checkpoint, filename='monkey.pth.tar')


def test_checkpoint():
    file_name = 'checkpoint_model.pth.tar'
    checkpoint = torch.load(file_name)

    model = UNET(in_channels=3, out_channels=45)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()

    data_loader = KittiDataset('G:/kitti/KITTI-360/')
    img, label = data_loader[0]

    val_transform = A.Compose(
        [
            A.Resize(width=704, height=188),
            ToTensorV2()
        ]
    )

    img_tensor = val_transform(image=img)['image'].float().unsqueeze(0)

    predictions = model(img_tensor)

    soft_max = nn.Softmax2d()
    out_classes_prob = soft_max(predictions)

    out_classes = torch.argmax(out_classes_prob, dim=1)

    plt.imshow(out_classes.permute(1, 2, 0))
    plt.show()
    print('break')

if __name__ == "__main__":

    test_checkpoint()

