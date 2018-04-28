import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data

num_classes = 2
ignore_label = 255
mask_root_train = '/media/teamd/New Volume/VLR_Project/top_image_dilation_2/train'
img_root_train = '/media/teamd/New Volume/VLR_Project/top_image_raw/train'
mask_root_val = '/media/teamd/New Volume/VLR_Project/top_image_dilation_2/val'
img_root_val = '/media/teamd/New Volume/VLR_Project/top_image_raw/val'

def make_dataset(mode):

    if mode == 'train':
        mask_root = mask_root_train
        img_root = img_root_train
    else:
        mask_root = mask_root_val
        img_root = img_root_val

    mask_categories = os.listdir(mask_root)
    img_categories = os.listdir(img_root)
    assert img_categories == mask_categories
    items = []
    for dir in img_categories:
        img_dir  = os.path.join(img_root,dir)
        mask_dir = os.path.join(mask_root,dir)
        for file_name in sorted(os.listdir(img_dir)):
            img_file  = os.path.join(img_dir ,file_name)
            mask_file = os.path.join(mask_dir,file_name)
            items.append((img_file,mask_file))
    return items


class Kitti(data.Dataset):
    def __init__(self, quality, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        # self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
        #                       3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
        #                       7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
        #                       14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
        #                       18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        #                       28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        # for k, v in self.id_to_trainid.items():
        #     mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)
