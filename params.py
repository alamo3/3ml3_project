import torch

KITTI_DATASET_PATH = 'G:/kitti/KITTI-360/' # Define your KITTI dataset path here, this should contained the following folders:
                                            # data_2d_semantics
                                            # data_2d_raw

parallel_training = True # Set to True if you want to train on multiple GPUs. Also used to load models trained on multiple GPUs.
# hyper parameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 if parallel_training else 2
NUM_EPOCHS = 100 if parallel_training else 20
NUM_WORKERS = 12
IMAGE_WIDTH = 704
IMAGE_HEIGHT = 188
PIN_MEMORY = True
LOAD_MODEL = False
output_folder= 'train_output/'
MODEL_FILE_NAME = 'final_2.pth.tar'
CALCULATE_METRICS = False # Set to True if you want to calculate metrics on validation set