import segmentation_models_pytorch as smp
import albumentations as A
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2

from unet import UNET
from tqdm import tqdm

# hyper parameters

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 5
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
PIN_MEMORY = True
LOAD_MODEL = False

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

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
    train_transform = A.Compose(
        [
            A.resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
            A.rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0 ,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        [
            A.resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
            A.normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    model = UNET(in_channels=3, out_channels=19).to(device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        


if __name__ == "__main__":

