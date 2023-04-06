import torch

KITTI_DATASET_PATH = 'G:/kitti/KITTI-360/'

# hyper parameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 5
NUM_WORKERS = 5
IMAGE_WIDTH = 704
IMAGE_HEIGHT = 188
PIN_MEMORY = True