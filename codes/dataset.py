import os
import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image, ImageFilter

from codes import utils

FOLDER_NAMES = ['AFW', 'AFW_Flip', 'HELEN', 'HELEN_Flip', 'IBUG', 'IBUG_Flip', 'LFPW', 'LFPW_Flip']

def get_list_from_filenames(data_dir, folder_names, test_dict):
    """
    """
    filenames_dict=[]

    for folder in folder_names:
        folder_path = os.path.join(data_dir, folder)
        all_files = os.listdir(folder_path)
        filenames_dict += list(set([os.path.join(folder, os.path.splitext(i)[0]) for i in sorted(all_files) if os.path.join(folder, os.path.splitext(i)[0]) not in test_dict]))

    return filenames_dict


class Pose300WLP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self,
                data_dir,
                transform=None,
                img_ext='.jpg',
                annot_ext='.mat',
                image_mode='RGB',
                test=False):
        """Dataset class for the 300W_LP dataset, head pose estimation
        Inputs:
            - data_dir:
            - transform:
            - img_ext:
            - annot_ext:
            - image_mode:
            - test:
        """
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        TEST_SET = [os.path.join('AFW', f'AFW_9545523490_1_{i}') for i in range(18)] + [os.path.join('AFW', f'AFW_9545523490_2_{i}') for i in range(11)]\
                    +[os.path.join('HELEN', f'HELEN_232194_1_{i}') for i in range(17)] + [os.path.join('HELEN', f'HELEN_1629243_1_{i}') for i in range(17)]\
                    +[os.path.join('IBUG', f'IBUG_image_003_1_{i}') for i in range(11)] + [os.path.join('IBUG', f'IBUG_image_004_1_{i}') for i in range(12)]\
                    +[os.path.join('LFPW', f'LFPW_image_test_0001_{i}') for i in range(16)] + [os.path.join('LFPW', f'LFPW_image_test_0002_{i}') for i in range(17)]\
                    +[os.path.join('AFW_Flip', f'AFW_9545523490_1_{i}') for i in range(18)] + [os.path.join('AFW', f'AFW_9545523490_2_{i}') for i in range(11)]\
                    +[os.path.join('HELEN_Flip', f'HELEN_232194_1_{i}') for i in range(17)] + [os.path.join('HELEN', f'HELEN_1629243_1_{i}') for i in range(17)]\
                    +[os.path.join('IBUG_Flip', f'IBUG_image_003_1_{i}') for i in range(11)] + [os.path.join('IBUG', f'IBUG_image_004_1_{i}') for i in range(12)]\
                    +[os.path.join('LFPW_Flip', f'LFPW_image_test_0001_{i}') for i in range(16)] + [os.path.join('LFPW', f'LFPW_image_test_0002_{i}') for i in range(17)]
        
        # Train-Test split
        self.image_mode = image_mode
        if not test:
            self.data = get_list_from_filenames(self.data_dir, FOLDER_NAMES, TEST_SET)
            self.length = len(self.data)
        else:
            self.data = TEST_SET
            self.length = len(self.data)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.data[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.data[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_pose_params_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        tx = pose[3]
        ty = pose[4]
        tz = pose[5]
        scale = pose[6]

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)
        # Make img tensor
        PILtoTensor = transforms.ToTensor()
        img = PILtoTensor(img)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll, tx, ty, tz, scale])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.data[index]

    def __len__(self):
        # 122,450
        return self.length
    
    def _return_dataloader(self, batch_size, shuffle, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
