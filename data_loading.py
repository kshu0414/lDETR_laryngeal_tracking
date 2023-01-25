import torch
from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd
from skimage import io
import numpy as np

'''
Data pipeline
'''

class topTrachealDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None,target_transform=None,heat_map=False,patch_transform=None,nfold=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        if nfold:
            self.landmarks_frame = self.landmarks_frame[self.landmarks_frame['fold_index'].isin(nfold)]
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.heat_map = heat_map
        self.patch_transform = patch_transform
        #self.clahe = cv2.createCLAHE(clipLimit=5.0,tileGridSize=(10,10))

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 4])
        image = io.imread(img_name)
        
        landmarks = self.landmarks_frame.iloc[idx, :4]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float32').reshape( 4)
        sample = {'image': image, 'landmarks': landmarks,'img_name':self.landmarks_frame.iloc[idx, 4]}

        if self.transform:
            sample = self.transform(sample)

        if self.heat_map:
            sample['heatmaps'] = self.target_transform(sample['landmarks'])

        if self.patch_transform:
            sample['image'],sample['heatmaps'],sample['landmarks'] = self.patch_transform(sample['image'],sample['heatmaps'],sample['landmarks'])
        return sample

import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import math
class RandomAugment(object):
    def __init__(self,size,uplim=0.6,dwlim=0.4,scale=(0.7,1.0),ratio=(0.8,1.2),horizontal_flip=0.2,rotation=(-45,45),colorjitter=True,rotation_ratio=0.5,colorjitter_ratio=0.5):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.horizontal_flip = horizontal_flip
        self.rotation = rotation
        self.colorjitter = colorjitter
        self.contrast = [0.5,1.5]
        self.brightness = [0.5,1.5]
        self.rotation_ratio = rotation_ratio
        self.colorjitter_ratio = colorjitter_ratio
        self.uplim = uplim
        self.dwlim = dwlim

    
    def get_params(self,sample):
        width,height = TF.get_image_size(sample['image'])
        area = height*width

        log_ratio = torch.log(torch.tensor(self.ratio))
        for _ in range(10):
            target_area = area*torch.empty(1).uniform_(self.scale[0],self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0],log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area*aspect_ratio)))
            h = int(round(math.sqrt(target_area/aspect_ratio)))

            if 0<w<=width and 0<h<=height:
                top = max(0,int(torch.round(sample['landmarks'][3]-self.uplim*h)))
                buttom = min(height-h+1,int(torch.round(sample['landmarks'][1]-self.dwlim*h)))
                left = max(0,int(torch.round(sample['landmarks'][2]-self.uplim*w)))
                right = min(width-w+1,int(torch.round(sample['landmarks'][0]-self.dwlim*w)))

                if top<buttom and left<right:
                    i = torch.randint(top, buttom, size=(1,)).item()
                    j = torch.randint(left, right, size=(1,)).item()
                    return i,j,h,w

        if width > 1008:
            return 116,244,792,792
        return 108,0,792,792

    def __call__(self,sample):
        i,j,h,w = self.get_params(sample)
        image = TF.resized_crop(sample['image'],i,j,h,w,[self.size,self.size])
        landmarks = torch.mul(sample['landmarks'] - torch.tensor([j,i,j,i]),torch.tensor([1/w,1/h,1/w,1/h]))

        if random.random()<self.rotation_ratio:
            angle = float(torch.empty(1).uniform_(self.rotation[0],self.rotation[1]).item())
            image = TF.rotate(image,angle)

            cosa = math.cos(angle*math.pi/180)
            sina = math.sin(angle*math.pi/180)

            landmarks -= float(1/2)
            landmarks = torch.matmul(landmarks,torch.tensor([[cosa,-sina,0,0],[sina,cosa,0,0],[0,0,cosa,-sina],[0,0,sina,cosa]]))
            landmarks += float(1/2)


        if random.random()<self.horizontal_flip:
            image = TF.hflip(image)
            landmarks = torch.mul(landmarks,torch.tensor([-1,1,-1,1])) + torch.tensor([1,0,1,0])

        if self.colorjitter and random.random()<self.colorjitter_ratio:
            b = float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
            c = float(torch.empty(1).uniform_(self.contrast[0], self.contrast[1]))

            image = TF.adjust_brightness(image, b)
            image = TF.adjust_contrast(image,c)

        sample['landmarks'] = landmarks
        sample['image'] = image
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample['image'] = TF.to_tensor(sample['image'])
        sample['landmarks'] = torch.from_numpy(sample['landmarks'])
        return sample

class ValidTransform(object):
    def __init__(self,size):
        self.size = size
    def __call__(self,sample):
        i,j,h,w = 108,0,792,792
        sample['image'] = TF.resized_crop(sample['image'],i,j,h,w,[self.size,self.size])
        sample['landmarks'] = torch.mul(sample['landmarks'] - torch.tensor([j,i,j,i]),torch.tensor([1/w,1/h,1/w,1/h]))
        return sample


class TestTransform(object):
    def __init__(self,size):
        self.size = size
    def __call__(self,sample):
        i,j,h,w = 116,244,792,792
        sample['image'] = TF.resized_crop(sample['image'],i,j,h,w,[self.size,self.size])
        sample['landmarks'] = torch.mul(sample['landmarks'] - torch.tensor([j,i,j,i]),torch.tensor([1/w,1/h,1/w,1/h]))
        return sample