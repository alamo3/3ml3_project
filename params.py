import torch

KITTI_DATASET_PATH = 'G:/kitti/KITTI-360/'

parallel_training = False
# hyper parameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 if parallel_training else 2
NUM_EPOCHS = 50 if parallel_training else 20
NUM_WORKERS = 12
IMAGE_WIDTH = 704
IMAGE_HEIGHT = 188
PIN_MEMORY = True
LOAD_MODEL = False
output_folder= 'train_output/'
MODEL_FILE_NAME = 'final_2.pth.tar'
