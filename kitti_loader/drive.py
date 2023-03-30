import os
from torchvision.io import read_image
import cv2


class Kitti_Drive:

    def __init__(self, dataset_path, drive_id, left_and_right=False):
        self.drive_id = drive_id
        self.dataset_path = dataset_path
        self.drive_raw_data_path = dataset_path + 'data_2d_raw/' + drive_id
        self.drive_labels_path = dataset_path + 'data_2d_semantics/train/' + drive_id
        self.use_left_and_right = left_and_right

        self.num_raw_images = self.count_num_raw_images()
        self.num_labelled_images = 0
        self.labelled_ids = []

        self.count_num_labelled_images()

    def count_num_raw_images(self):
        search_path = self.drive_raw_data_path + '/image_00/data_rect/'

        root, dirs, files = next(os.walk(search_path))

        num_images = len(files)

        if self.use_left_and_right:
            search_path = self.drive_raw_data_path + '/image_01/data_rect/'
            root, dirs, files = next(os.walk(search_path))
            num_images += len(files)

        return num_images

    def count_num_labelled_images(self):
        frames_file_path = self.dataset_path+'data_2d_semantics/train/2013_05_28_drive_train_frames.txt'
        frames_file = open(frames_file_path, 'r')

        lines = frames_file.readlines()

        for i in range(len(lines)):
            line = lines[0]
            if line.__contains__(self.drive_id):
                labelled_id = line[58:68]
                self.labelled_ids.append(labelled_id)
                self.num_labelled_images += 1


    def extend_image_id(self, id):
        str_id = str(id)
        zeros_to_add = 10 - len(str_id)
        extended_id = '0' * zeros_to_add + str_id

        return extended_id

    def get_raw_image_by_id(self, image_id, right_image=False, show_img=False):
        image_dir = 'image_01' if right_image else 'image_00'
        image_path = self.drive_raw_data_path + '/' + image_dir + '/data_rect/' + self.labelled_ids[image_id] + '.png'

        img = cv2.imread(image_path)

        if show_img:
            cv2.imshow('test', img)
            cv2.waitKey(0)

        return img

    def get_semantic_image_by_id(self, image_id, show_img=False):
        return self.get_labelled_image_by_id(image_id=image_id, label_type='semantic', show_img=show_img)

    def get_semantic_rgb_image_by_id(self, image_id, show_img=False):
        return self.get_labelled_image_by_id(image_id=image_id, label_type='semantic_rgb', show_img=show_img)

    def get_instance_image_by_id(self, image_id, show_img=False):
        return self.get_labelled_image_by_id(image_id=image_id, label_type='instance', show_img=show_img)

    def get_labelled_image_by_id(self, image_id, label_type, show_img=False):
        image_dir = 'image_00'
        image_path = self.drive_labels_path + '/' + image_dir + '/' + label_type + '/' + self.labelled_ids[image_id] + '.png'

        img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

        if show_img:
            cv2.imshow('test', img)
            cv2.waitKey(0)

        return img


if __name__ == "__main__":
    drive = Kitti_Drive('G:/kitti/KITTI-360/', '2013_05_28_drive_0000_sync')
    print(drive.extend_image_id(10))
    print(len(drive.extend_image_id(10)))

    test_img_id = 0
    drive.get_raw_image_by_id(image_id=test_img_id, show_img=True)
    drive.get_semantic_image_by_id(image_id=test_img_id, show_img=True)
    drive.get_semantic_rgb_image_by_id(image_id=test_img_id, show_img=True)
    #drive.get_instance_image_by_id(image_id=test_img_id, show_img=True)
