import os
import numpy as np
import pandas as pd
import cv2
import einops
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, 
    CenterCrop, Resize, RandomCrop, GaussianBlur, JpegCompression, Downscale, ElasticTransform, Affine, ToFloat
)


def read_img(img_path:str):

    # img = Image.open(img_path).convert("L")
    # img = np.asarray(img)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


    img = np.expand_dims(img, axis=-1).astype("float32")
    if np.max(img) > 1.0:
        img /= 255.0
    
    return img


# /home/ray/Sda1/rawdata/oai/cleanData/00m/9508335/xr/right/001.png
    
class NewDataset(Dataset):
    """
    Class to store a given dataset.
    Parameters:
    - data (dataframe): data.pt file.
                    Dircetory of tensors {"images": images, "labels": labels}
    - transofrm: data transforms for auugmentation
    """

    def __init__(self, data: pd.DataFrame, transform=None, is_test=False, is_train=False):
        self.df = data
        self.is_test = is_test
        self.is_train = is_train

        if self.is_train:
            self.landmark = pd.DataFrame.from_dict(np.load("Data/csv_data/landmarks/xrTransLandmarks00m.npy", allow_pickle=True).item())
            self.df = pd.merge(self.df, self.landmark, how="left", on="ID")

        # self.left_img_path = self.df.left_xr_path.values
        # self.right_img_path = self.df.right_xr_path.values

        # self.left_labels = self.df.left_labels.values
        # self.right_labels = self.df.right_labels.values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def img_fliiper(self, img, random_val, random_vertical):
        # --------------------------------
        # augmentation - flip, rotate
        # --------------------------------
        # randomly horizontal flip the image with probability of 0.5
        if (random_val > 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # randomly vertically flip the image with probability 0.5
        if random_vertical > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img


    def __getitem__(self, idx):
        # Bilateral side
        # A.transform need image shape (H, W, C)
        # print(idx) #debug
        left_path = self.df["left_xr_path"][idx]
        right_path = self.df["right_xr_path"][idx]

        left_img = read_img(left_path)  # shape(700, 700, 1)
        right_img = read_img(right_path)  # shape(700, 700, 1)


        left_label = self.df["left_labels"][idx]
        right_label = self.df["right_labels"][idx]
        labels = np.array([left_label, right_label]).astype(np.float32)


        left_grade  = self.df[self.df.columns[2]][idx].astype(np.float32) # baseline KL grade of left knee
        right_grade = self.df[self.df.columns[1]][idx].astype(np.float32) # baseline KL grade of right knee
        grades = np.array([left_grade, right_grade ]).astype(np.float32)



        # left_img = Image.fromarray(left_img)
        # right_img = Image.fromarray(right_img)
        #
        # random_val = np.random.uniform(0, 1)
        # random_vertical = np.random.uniform(0, 1)
        # left_img = self.img_fliiper(left_img, random_val, random_vertical)
        # right_img = self.img_fliiper(right_img, random_val, random_vertical)


        if self.is_train:
            left_landmark = self.df["left_xr_landmarks"][idx]
            right_landmark = self.df["right_xr_landmarks"][idx]



        if self.transform:
            if self.is_train:

                trans = self.transform(image=left_img, keypoints=left_landmark)
                left_img, left_landmark = trans["image"], np.array(trans['keypoints']).astype(np.float32)
                trans = self.transform(image=right_img, keypoints=right_landmark)
                right_img, right_landmark = trans["image"], np.array(trans['keypoints']).astype(np.float32)

                # print("landmarks:", left_landmark.shape, right_landmark.shape)  # 12, 16...
                #
                # landmarks = np.array([left_landmark, right_landmark]).astype(np.float32)

            else:
                left_img = self.transform(image=left_img)["image"]
                right_img = self.transform(image=right_img)["image"]

        # if len(left_img.shape) < 4:
        #     left_img = np.expand_dims(left_img, 0)
        #     right_img = np.expand_dims(right_img, 0)

        if self.is_train:
            # Add Noise
            max_sigma = 15
            if np.random.random() < 0.5:
                sigma = np.random.random() * max_sigma / 255
                # print("left_img.shape:", left_img.shape)
                left_img = left_img + torch.FloatTensor(left_img.shape).normal_(mean=0, std=sigma).numpy()
                right_img = right_img + torch.FloatTensor(right_img.shape).normal_(mean=0, std=sigma).numpy()
                left_img = np.clip(left_img, 0, 1)
                right_img = np.clip(right_img, 0, 1)


        # left_img = torch.cat([left_img, left_img, left_img], dim=0)
        # right_img = torch.cat([right_img, right_img, right_img], dim=0)
        # return (left_img, right_img), labels, grades, (left_landmark, right_landmark)

        if self.is_train:
            return {"images": [left_img, right_img], 
                    "labels": labels, 
                    "grades": grades, 
                    "lmk": [left_landmark, right_landmark], 
                    "paths": [left_path, right_path]}
        else:
            return {"images": [left_img, right_img], 
                    "labels": labels, 
                    "grades": grades, 
                    "paths": [left_path, right_path]}


def split_data(data, size=0.2, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=random_state)
    images, labels = data["images"], data["labels"]
    train_index, val_index = next(sss.split(images, labels))

    train_images, train_labels = images[train_index], labels[train_index]
    val_images, val_labels = images[val_index], labels[val_index]

    train_data = {"images": train_images, "labels": train_labels}
    val_data = {"images": val_images, "labels": val_labels}
    
    return train_data, val_data


def new_split_data(data:pd.DataFrame, size=0.2, random_state=42):
    train_data, val_data = train_test_split(data, test_size=size, random_state=random_state)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    
    return train_data, val_data
 

# Using this one
def NewLoader(data:pd.DataFrame, val_data=None, batch_size=32, num_workers=0, val_size=0.1, 
                 is_sampler=True, 
                 is_test=False):
    print("----Loading dataset----")

    TRAIN_TRANSFORM_IMG = A.Compose([
        A.Resize(310, 310, always_apply=True), 
        A.RandomCrop(224, 224, always_apply=True),
        # A.CenterCrop(224, 224, always_apply=True), # Worse than random Crop
        
        # A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),     # Not Flip
        A.VerticalFlip(p=0.5),     # if not, test48m much wo
        # rse
        
        ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, p=0.5),  # 0.3

        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                 contrast_limit=(-0.1, 0.1), p=0.5),
        # JpegCompression(quality_lower=80, quality_upper=100, p=0.3),  # 
        Affine(p=0.3),  # Affine
        # A.Affine(rotate=5, translate_percent=0.1, scale=[0.9, 1.5], shear=0, p=0.5),

        ToTensorV2(), 
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    VAL_TRANSFORM_IMG = A.Compose([
        A.Resize(310, 310, always_apply=True), 
        A.CenterCrop(224, 224, always_apply=True), 
        # A.Normalize(mean=[0.543, ], std=[0.296, ]), 
        ToTensorV2(),
    ])
    
    TEST_TRANSFORM_IMG = VAL_TRANSFORM_IMG


    if is_test:
        test_dataset = NewDataset(data=data, transform=TEST_TRANSFORM_IMG, is_test=is_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=num_workers)
        print("Test data")
        print("#Test patinets", len(test_dataset))
        print("-------------------------")
        
        return test_loader
    
    else:
        if val_data:
            if is_sampler:
                labels= data[["left_labels", "right_labels"]].to_numpy()
                pos_num = int(labels.sum(axis=1).sum())
                neg_num = labels.shape[0] - pos_num
                
                pos_num = int(labels.sum(axis=1).sum())
                neg_num = labels.shape[0] - pos_num
                
                pos_weight = 1.0 / pos_num
                neg_weight = 1.0 / neg_num
                
                weights = [neg_weight if labels.sum(axis=1)[i] == 0 else pos_weight 
                        for i in range(labels.shape[0])]
                sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
                
                # train_dataset = MyDataset(data=data, transform=TRAIN_TRANSFORM_IMG)
                # val_dataset = MyDataset(data=val_data, transform=VAL_TRANSFORM_IMG)
                train_dataset = NewDataset(data=data, transform=TRAIN_TRANSFORM_IMG, is_test=False, is_train=True)
                val_dataset = NewDataset(data=val_data, transform=VAL_TRANSFORM_IMG, is_test=True)
        
                train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers,
                                        sampler=sampler)

                val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                        shuffle=False, num_workers=num_workers)
            
            else:
                train_dataset = NewDataset(data=data, transform=TRAIN_TRANSFORM_IMG, is_train=True)
                val_dataset = NewDataset(data=val_data, transform=VAL_TRANSFORM_IMG)
        
                train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers, 
                                        drop_last=True)

                val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                        shuffle=False, num_workers=num_workers)
            
            print('Train and val data', )
            print("#Train patinets: ", len(train_dataset))
            print("#Val patinets: ", len(val_dataset))
            print("-------------------------")
            
            return train_loader, val_loader
            
        else:
            train_data, val_data = new_split_data(data=data, size=val_size)  # 
            
            if is_sampler:
                labels= train_data[["left_labels", "right_labels"]].to_numpy()
                pos_num = int(labels.sum(axis=1).sum())
                neg_num = labels.shape[0] - pos_num
                
                pos_num = int(labels.sum(axis=1).sum())
                neg_num = labels.shape[0] - pos_num
                
                pos_weight = 1.0 / pos_num
                neg_weight = 1.0 / neg_num
                
                weights = [neg_weight if labels.sum(axis=1)[i] == 0 else pos_weight 
                        for i in range(labels.shape[0])]
                sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
                
                train_dataset = NewDataset(data=train_data, transform=TRAIN_TRANSFORM_IMG, is_train=True)
                val_dataset = NewDataset(data=val_data, transform=VAL_TRANSFORM_IMG)
        
                train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                        shuffle=False, num_workers=num_workers, 
                                        sampler=sampler, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                        shuffle=False, num_workers=num_workers)
            
            else: 
                train_dataset = NewDataset(train_data, transform=TRAIN_TRANSFORM_IMG, is_train=True)
                val_dataset = NewDataset(val_data, transform=VAL_TRANSFORM_IMG)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers, 
                                        drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                        shuffle=False, num_workers=num_workers)
            
            print('Dev data', )
            print("#Train patinets: ", len(train_dataset))
            print("#Val patinets: ", len(val_dataset))
            print("-------------------------")

            return train_loader, val_loader


if __name__ == "__main__":

    lmk = np.load("Data/csv_data/landmarks/xrTransLandmarks00m.npz").item()
    print(lmk)