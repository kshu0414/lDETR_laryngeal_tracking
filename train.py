import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import os

import numpy as np
import pandas as pd

from data_loading import *
from utils import *
from model import *
import scipy.io

label_path = "./" 
image_path = "../Top_tracheal_images" 
test_image_path = "../Top_tracheal_healthy_images"

new_size = 512
initial_epoch = 0

nfold = 1
train_fold = set(list(np.arange(1,6)))-set([nfold])

EPOCHS = 400
BATCH_SIZE = 16
PATH = './resnet50_transformer_DA_ep%d_fold%d'%(EPOCHS,nfold)

device = 0
if device == 0:
    num_workers, prefetch_factor= 6,4
else:
    num_workers, prefetch_factor= 0,2

image_transform = transforms.Compose(
    [
    ToTensor(),
    RandomAugment(new_size),
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

model = lDETR()

params_to_update = []
for param in model.parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
model.to(device)

import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.Adam(params_to_update,lr=0.0001)

train_loss  = []
test_loss = []
if not os.path.exists(PATH):
        os.mkdir(PATH)

print('Ready to train\n')

model.train()
for epoch in range(1+initial_epoch,EPOCHS+1):

    ep_train_loss = 0.0
    ep_test_loss = 0.0
    print('.........training %d epoch...........'%epoch)

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['image'].to(device)
        labels = data['landmarks'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        ep_train_loss += loss.item()
    
    
    train_loss.append(ep_train_loss / len(trainloader))

    model.eval()
    with torch.no_grad():
        for data in validloader:
            inputs = data['image'].to(device)
            labels = data['landmarks'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            ep_test_loss += loss.item()
    test_loss.append(ep_test_loss/len(validloader))
    model.train()

    if epoch%1 == 0:
        print('[%d] loss: %4f' %(epoch,ep_train_loss / len(trainloader)))

    if epoch %40 == 0:
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(train_loss,'r',label='train')
        ax.plot(test_loss,'b',label='test')
        plt.yscale("log")
        plt.legend()
        plt.savefig(os.path.join(PATH,'losses.png'))
        plt.close()

        scipy.io.savemat(os.path.join(PATH,'losses.mat'),{'train_loss':train_loss,'test_loss':test_loss})

        model.eval()
        with torch.no_grad():
            for data in validloader:
                inputs = data['image'].to(device)
                labels = data['landmarks'].to(device)
                outputs = model(inputs)
                
                plt.figure(figsize=(12,9)) 
                for i in range(4):
                    error = new_size*mean_distance_error(labels[i,:],outputs[i,:])
                    
                    ax = plt.subplot(1, 4, i + 1)
                    plt.tight_layout()
                    ax.set_title('MDE %.4f'%(error))
                    show_landmarks(inputs[i,:].detach().to('cpu'),new_size*labels[i,:].detach().to('cpu'),new_size*outputs[i,:].detach().to('cpu'))
                
                plt.savefig(os.path.join(PATH,'test_ep%d.png'%epoch))
                plt.close()
                break
        model.train()

    if epoch%40 == 0:
        torch.save(model.state_dict(), os.path.join(PATH,'model_ep%d.pth'%epoch))
        torch.save(optimizer.state_dict(), os.path.join(PATH,'optimizer_ep%d'%epoch))
        model.eval()
        ep_train_loss = 0.0
        ep_test_loss = 0.0
        with torch.no_grad():
            for data in trainloader:
                inputs = data['image'].to(device)
                labels = data['landmarks'].to(device)
                outputs = model(inputs)
                ep_train_loss += mean_distance_error(labels,outputs)
            for data in validloader:
                inputs = data['image'].to(device)
                labels = data['landmarks'].to(device)
                outputs = model(inputs)
                ep_test_loss += mean_distance_error(labels,outputs)

        with open(os.path.join(PATH,'metrics.txt'), 'a') as f:
            f.write("Epoch :%d"%epoch)
            f.write("Training set distance: %.4f \n"%(ep_train_loss/len(trainloader)*new_size))
            f.write("Testing set distance: %.4f \n\n"%(ep_test_loss/len(validloader)*new_size))

        model.train()
print('Finished Training \n')