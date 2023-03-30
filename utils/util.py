import torch
import torchvision
from kitti_loader import dataset_kitti
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

def save_checkpoint(state, filename="checkpoint_model.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])

def get_training_loader(kitti_dataset_dir, batch_size, train_transform, val_transform, num_workers, pin_memory=True):

    dataset = dataset_kitti.KittiDataset(kitti_dataset_dir,train_transform)

    ds_size = len(dataset)
    indices = list(range(ds_size))
    np.random.shuffle(indices)

    val_split_index = int(np.floor(0.2 * ds_size))
    test_split_index = int(np.floor(0.1 * ds_size))
    train_idx, val_idx, test_idx = indices[val_split_index:], indices[test_split_index:val_split_index], indices[:test_split_index]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader




