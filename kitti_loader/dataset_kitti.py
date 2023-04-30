import os

import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
from kitti_loader.drive import Kitti_Drive
import matplotlib.pyplot as plt
import albumentations as A
import torchvision.transforms as T
import cv2


class KittiDataset(Dataset):

    def __init__(self, kitti_dir, transform=None, use_left_and_right=False):
        self.kitti_dir = kitti_dir
        self.use_left_and_right = use_left_and_right

        self.drives = []

        self.id_offset = 250

        self.transform = transform

        self.get_list_drives()

        print('Loaded Drives: ', [drive.drive_id for drive in self.drives])

        self.total_usable_images = 0
        for drive in self.drives:
            self.total_usable_images += drive.num_labelled_images

    def __len__(self):
        return self.drives[0].num_labelled_images

    def get_img_no_transform(self, idx,size=None):
        img = self.drives[0].get_raw_image_by_id(idx)
        labelled_img = self.drives[0].get_semantic_rgb_image_by_id(idx)

        if size is not None:
            img = cv2.resize(img, size)
            labelled_img = cv2.resize(labelled_img, size)

        return img, labelled_img

    def __getitem__(self, idx):


        img = self.drives[0].get_raw_image_by_id(idx)
        labelled_img = self.drives[0].get_semantic_image_by_id(idx)

        if self.transform is not None:
            augmented_images = self.transform(image=img, image1=labelled_img)
            img = augmented_images["image"]
            labelled_img = augmented_images["image1"]

        return img, labelled_img

        # for i in range(len(self.drives)):
        #     drive: Kitti_Drive = self.drives[i]
        #     idx += 250
        #     accum_drive_id += drive.num_raw_images
        #     if accum_drive_id > idx:
        #         actual_drive_id = drive.num_raw_images - accum_drive_id + 250
        #         return self.drives[idx_drive_id].get_raw_image_by_id(actual_drive_id), self.drives[idx_drive_id].get_semantic_rgb_image_by_id(actual_drive_id)
        #
        #     idx_drive_id += 1

        # raise IndexError("Does not exist, or our implementation is just wrong")

    def get_list_drives(self):
        raw_data_dir = self.kitti_dir + 'data_2d_raw/'
        root, dirs, files = next(os.walk(raw_data_dir))

        dirs.sort()

        print('Found drives: ', dirs)

        for drive in dirs:
            self.drives.append(Kitti_Drive(self.kitti_dir, drive))

        return dirs


if __name__ == "__main__":
    train_transform = A.Compose(
        [
            A.Resize(width=704, height=188),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            ToTensorV2()
        ],
        additional_targets={'image1' : 'image'}
    )
    data_loader = KittiDataset('C:/EcoCAR Projects/kitti/kitti-360/', transform=train_transform)
    img, label = data_loader[0]

    f,axarr = plt.subplots(2)
    axarr[0].imshow(img.permute(1,2,0))
    axarr[1].imshow(label.permute(1, 2, 0))
    plt.show()