import torch
import torchvision
from kitti_loader import dataset_kitti
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="checkpoint_model.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])

ree
