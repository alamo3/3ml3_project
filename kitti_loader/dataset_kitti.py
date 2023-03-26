import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from kitti_loader.drive import Kitti_Drive


class KittiDataset(Dataset):

    def __init__(self, kitti_dir, transform=None, target_transform=None, use_left_and_right=False):
        self.kitti_dir = kitti_dir
        self.use_left_and_right = use_left_and_right

        self.drives = []

        self.id_offset = 250

        self.transform = transform
        self.target_transform = target_transform

        self.get_list_drives()

        print('Loaded Drives: ', [drive.drive_id for drive in self.drives])

        self.total_usable_images = 0
        for drive in self.drives:
            self.total_usable_images += drive.num_labelled_images

    def __len__(self):
        return self.total_usable_images

    def __getitem__(self, idx):

        idx_drive_id = 0
        accum_drive_id = 0

        for i in range(len(self.drives)):
            drive: Kitti_Drive = self.drives[i]
            idx += 250
            accum_drive_id += drive.num_raw_images
            if accum_drive_id > idx:
                actual_drive_id = drive.num_raw_images - accum_drive_id + 250
                return self.drives[idx_drive_id].get_raw_image_by_id(actual_drive_id), self.drives[idx_drive_id].get_semantic_rgb_image_by_id(actual_drive_id)

            idx_drive_id += 1

        raise IndexError("Does not exist, or our implementation is just wrong")

    def get_list_drives(self):
        raw_data_dir = self.kitti_dir + 'data_2d_raw/'
        root, dirs, files = next(os.walk(raw_data_dir))

        dirs.sort()

        print('Found drives: ', dirs)

        for drive in dirs:
            self.drives.append(Kitti_Drive(self.kitti_dir, drive))

        return dirs


if __name__ == "__main__":
    data_loader = KittiDataset('/home/ecocar4/Desktop/kitti/kitti_360/KITTI-360/')
    img, label = data_loader[0]
