import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import inception_v3, resnet50

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from skimage import io, transform
import numpy as np
import scipy.io
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from utils import show_landmarks,mean_distance_error
from data_loading import TestTransform, ToTensor, RandomAugment, ValidTransform, topTrachealDataset, TestTransform

label_path = "./" #"../Top_tracheal_labels" #
image_path = "../Top_tracheal_images" #"../Top_tracheal_images"
test_image_path = "../Top_tracheal_healthy_images"

new_size = 512
EPOCHS = 400
BATCH_SIZE = 16
ex_gamma = 0
device = 0
if device == 0:
    num_workers, prefetch_factor= 6,4
else:
    num_workers, prefetch_factor= 0,2

nfold = 1
initial_epoch_list = [520,280,440,440,520]
initial_epoch = initial_epoch_list[nfold-1]
EPOCHS = 400 if initial_epoch<=400 else 600
PATH = './resnet50_transformer_DA_ep%d_fold%d_t2'%(EPOCHS,nfold)
train_fold = set(list(np.arange(1,6)))-set([nfold])
result_path = PATH #'./resnet50_transformer_DA_ep%d'%initial_epoch
'''
Data pipeline
'''

image_transform = transforms.Compose(
    [
    ToTensor(),
    RandomAugment(new_size),
    ]
)
test_transform = transforms.Compose(
    [ToTensor(),
    TestTransform(new_size),
    ]
)
valid_transform = transforms.Compose(
    [ToTensor(),
    ValidTransform(new_size),
    ]
)
trainset = topTrachealDataset(csv_file=os.path.join(label_path, 'train_labels_folds.csv'),
                                    nfold=list(train_fold),
                                    transform=image_transform,
                                    root_dir=image_path)
trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=num_workers,prefetch_factor=prefetch_factor,)

validset = topTrachealDataset(csv_file=os.path.join(label_path, 'train_labels_folds.csv'),
                                    nfold=[nfold],
                                    transform=valid_transform,
                                    root_dir=image_path)

validloader = DataLoader(validset,batch_size=BATCH_SIZE,shuffle=False,num_workers=num_workers,prefetch_factor=prefetch_factor,)


testset = topTrachealDataset(csv_file=os.path.join(label_path, 'healthy_labels.csv'),
                                    transform=test_transform,
                                    root_dir=test_image_path)
testloader = DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False,num_workers=num_workers,prefetch_factor=prefetch_factor,)


print('Dataset preparation completed\n')

def show_landmarks(image, predictions):
    """Show image with landmarks"""
    plt.imshow(image.permute(1, 2, 0)  )
    #plt.scatter(landmarks[0], landmarks[1], marker='o',facecolors='none', edgecolors='red')
    #plt.scatter(landmarks[2], landmarks[3], marker='s',facecolors='none', edgecolors='red')
    plt.scatter(predictions[0],predictions[1],marker='o',facecolors='none', edgecolors='cyan',alpha=0.8,linewidths=1.8)
    plt.scatter(predictions[2],predictions[3],marker='s',facecolors='none', edgecolors='cyan',alpha=0.8,linewidths=1.8)

'''
Neural networks
'''

import torch.nn as nn

class lDETR(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes=4, hidden_dim=256, nheads=4,
                 num_encoder_layers=5, num_decoder_layers=5):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50(pretrained=True)
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = Transformer(
            d_model=hidden_dim, nhead=nheads, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=2048, dropout=0.1,
            activation="relu", normalize_before=False,
            return_intermediate_dec=False)

        # prediction heads
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_pred = nn.Linear(hidden_dim, num_classes)

        # output positional encodings (object queries)
        self.query_pos = nn.Embedding(1, hidden_dim)

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Embedding(50, hidden_dim // 2)
        self.col_embed = nn.Embedding(50, hidden_dim // 2)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        i = torch.arange(W, device=x.device)
        j = torch.arange(H, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(H, 1, 1),
            y_emb.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        
        #print(pos.shape,h.shape,self.query_pos.unsqueeze(1).repeat(1, BATCH_SIZE, 1).shape)

        # propagate through the transformer
        h = self.transformer(src=h, mask=None, query_embed=self.query_pos.weight, pos_embed=pos)[0]
        
        # finally project transformer outputs to class labels and bounding boxes

        h = self.linear_pred(h).sigmoid()

        return h

model = DETRsimple()
model.load_state_dict(torch.load(os.path.join(PATH,'model_ep%d.pth'%initial_epoch),map_location=torch.device('cpu')))

pp=0
for p in list(model.parameters()):
    nparam=1
    for s in list(p.size()):
        nparam = nparam*s
    pp += nparam
print(pp)
