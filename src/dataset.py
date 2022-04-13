import torch
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from src.utils import set_seed
import albumentations as A

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class CustomImageDataset(Dataset):
    def __init__(self, image, label, data_dir,target_transform=None,mode=None,image_size=256,rescale=None,use_meta=False):
        list_agu = [
            A.Transpose(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2, p=0.75),
                    A.OneOf([
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=(3,5)),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ], p=0.7),

                    A.OneOf([
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.),
                        A.ElasticTransform(alpha=3),
                    ], p=0.7),

                    A.CLAHE(clip_limit=4.0, p=0.7),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                    A.CoarseDropout(max_holes=1, max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), min_holes=1, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.7),
                    A.Normalize()
        ]

        if mode == "train":
            image_transformation = A.Compose(list_agu)
        elif mode == "val" or mode == "test":
            image_transformation = A.Compose([A.Normalize()])
        else:
            ValueError("Wrong mode in Set-up-Dataset")

        self.transform = image_transformation
        self.data_dir = data_dir
        self.image_size = image_size
        self.mode = mode
        self.img_data = image
        self.label_data = label
        self.target_transform = target_transform
        self.rescale = rescale
        self.use_meta = use_meta

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        if self.use_meta:
            row = self.img_data[idx]
            label = self.label_data[idx]
            image_path = os.path.join(self.data_dir,row[0])
        else:
            label = self.label_data[idx]
            image_path = os.path.join(self.data_dir,self.img_data[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rescale:
            image = cv2.resize(image,(self.rescale,self.rescale))

        transformed = self.transform(image=image)
        image = transformed["image"]

        if self.target_transform:
            label = self.target_transform(label)

        image = image.transpose(2, 0, 1)

        if self.use_meta:
            data = (torch.tensor(image).float(), row[1])
            return data, label
        else:
            return torch.tensor(image).float(), label
    
    def get_labels(self): return self.label_data