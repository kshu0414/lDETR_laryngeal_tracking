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

pretrained_file = './pretrained_model.pth'
nfold = 1
new_size = 512
device = 'cpu'
if device == 0:
    num_workers, prefetch_factor= 6,4
else:
    num_workers, prefetch_factor= 0,2

PATH = './Results'
BATCH_SIZE = 16

valid_transform = transforms.Compose(
    [ToTensor(),
    ValidTransform(new_size),
    ]
)
test_transform = transforms.Compose(
    [ToTensor(),
    TestTransform(new_size),
    ]
)

validset = topTrachealDataset(csv_file=os.path.join(label_path, 'train_labels_folds.csv'),
                                    nfold=[nfold],
                                    transform=valid_transform,
                                    root_dir=image_path)

validloader = DataLoader(validset,batch_size=BATCH_SIZE,shuffle=True,num_workers=num_workers,prefetch_factor=prefetch_factor,)
testset = topTrachealDataset(csv_file=os.path.join(label_path, 'healthy_labels.csv'),
                                    transform=test_transform,
                                    root_dir=test_image_path)
testloader = DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False,num_workers=num_workers,prefetch_factor=prefetch_factor,)

model = lDETR(num_classes=4, hidden_dim=256, nheads=2, num_encoder_layers=2, num_decoder_layers=2)
model.load_state_dict(torch.load((pretrained_file),map_location=torch.device('cpu')))

model.to(device)
model.eval()

'''
Get prediction statistics on validation and testing data
'''

plt.figure(figsize=(8,8))
ep_valid_loss = []
ep_test_loss = []
valid_name = []
test_name = []
ax = []
ay = []
px = []
py = []  
time_list = []

import time
start_time = time.time()

with torch.no_grad():
    for data in validloader:
        inputs = data['image'].to(device)
        labels = data['landmarks'].to(device)
        outputs = model(inputs)
        ep_valid_loss += distance_error(labels*new_size,outputs*new_size).tolist()
        valid_name += data['img_name']
        ax += outputs[:,0].tolist()
        ay += outputs[:,1].tolist()
        px += outputs[:,2].tolist()
        py += outputs[:,3].tolist()
        time_list += [time.time()-start_time]*len(data['img_name'])


df_valid = {'name':valid_name,'distance_error':ep_valid_loss,'ax':ax,'ay':ay,'px':px,'py':py,'time':time_list}

df_valid = pd.DataFrame(df_valid)
df_valid.to_csv(os.path.join(PATH,'Validation_results.csv'),index=False)

ax = []
ay = []
px = []
py = []  
time_list = []
with torch.no_grad():
    for data in testloader:
        inputs = data['image'].to(device)
        labels = data['landmarks'].to(device)
        outputs = model(inputs)
        ep_test_loss += distance_error(labels*new_size,outputs*new_size).tolist()
        test_name += data['img_name']
        ax += outputs[:,0].tolist()
        ay += outputs[:,1].tolist()
        px += outputs[:,2].tolist()
        py += outputs[:,3].tolist()
        time_list += [time.time()-start_time]*len(data['img_name'])

df_test = {'name':test_name,'distance_error':ep_test_loss,'ax':ax,'ay':ay,'px':px,'py':py,'time':time_list}

df_test = pd.DataFrame(df_test)
df_test.to_csv(os.path.join(PATH,'Testing_results.csv'),index=False)

# Get error distributions
ep_valid_loss = np.array(ep_valid_loss).squeeze()
valid_mean = np.mean(ep_valid_loss)
valid_std = np.std(ep_valid_loss)
valid_median = np.median(ep_valid_loss)
print("Valid on patient's data: Mean %.4f;Std %.4f; Median %.4f"%(valid_mean,valid_std,valid_median))

ep_test_loss = np.array(ep_test_loss).squeeze()
test_mean = np.mean(ep_test_loss)
test_std = np.std(ep_test_loss)
test_median = np.median(ep_test_loss)
print("Test on healthy data: Mean %.4f;Std %.4f; Median %.4f"%(test_mean,test_std,test_median))

plt.hist(ep_valid_loss,bins=np.arange(20))
plt.title("Valid on patient's data: Mean %.4f;Std %.4f; Median %.4f"%(valid_mean,valid_std,valid_median))
plt.title("Test on healthy data: Mean %.4f;Std %.4f; Median %.4f"%(test_mean,test_std,test_median))
plt.savefig(os.path.join(PATH,'distance_error_distribution.png'))

with open(os.path.join(PATH,'log.txt'), 'a') as f:
    f.write("Localization error: ")
    f.write("    Validation mean error: %.4f"%valid_mean)
    f.write("    Validation std: %.4f"%valid_std)
    f.write("    Validation median error: %.4f"%valid_median)
    f.write("    Testing mean error: %.4f"%test_mean)
    f.write("    Testing std: %.4f"%test_std)
    f.write("    Testing median error: %.4f\n"%test_median)

'''
Model visualization
original code available from https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb#scrollTo=0j8yy6vHf0Q-
'''
conv_features, enc_attn_weights, dec_attn_weights = [], [], []

hooks = [
    model.backbone.layer4.register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    ),
    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
]

num = 5

for sample in validloader:
    inputs = sample['image'].to(device)
    outputs = model(inputs)
    break

for hook in hooks:
    hook.remove()

conv_features = conv_features[0]
enc_attn_weights = enc_attn_weights[0]
dec_attn_weights = dec_attn_weights[0]

h, w = conv_features.shape[-2:]

fig, axs = plt.subplots(ncols=num, nrows=2, figsize=(20, 7))

for i in range(num):
    ax = axs[0,i]
    weights = dec_attn_weights[i,0].view(h, w).detach().numpy()
    ax.imshow(weights)
    ax.axis('off')
    ax = axs[1,i]
    ax.imshow(inputs[i].permute(1, 2, 0))
    pred = outputs.detach().numpy()
    ax.scatter(pred[i,0]*512,pred[i,1]*512,marker='o',color='m',facecolors='b',edgecolors='w',alpha=1,s=60,linewidths=1)
    ax.scatter(pred[i,2]*512,pred[i,3]*512,marker='s',color='m',facecolors='b',edgecolors='w',alpha=1,s=60,linewidths=1)
    ax.axis('off')
    
fig.tight_layout()
fig.savefig(os.path.join(PATH,'decoder_weights_examples.png'))

sattn = enc_attn_weights[0].reshape((h,w) + (h,w))
fact = 32

# Select 4 reference points for encoder weights visualization
def on_click(event):

    global ix,iy
    ix,iy= event.xdata,event.ydata

    global count
    global coords
    global num
    
    x = ((ix // fact) + 0.5) * fact
    y = ((iy // fact) + 0.5) * fact
    coords.append(fcenter_ax.add_patch(plt.Circle((x, y), 8, facecolor='m',edgecolor='w',linewidth=1)))

    if count >= num:
        count -= num

    if len(coords) > num:
        coords.pop(0).remove()
    
    ax = axs[count]

    idx = (int(ix // fact), int(iy // fact))
    ax.imshow(sattn[..., idx[0], idx[1]].detach().numpy(), cmap='magma', interpolation='nearest')
    ax.axis('off')
    ax.set_title('(%d,%d)'%(x,y))
    fig.canvas.draw()

    count += 1
    return 

count = 0
coords = []

fig = plt.figure(constrained_layout=True, figsize=(18 * 0.7, 12 * 0.7))
num = 4
gs = fig.add_gridspec(1, num+1)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 3]),
    fig.add_subplot(gs[0, 4]),
]

fcenter_ax = fig.add_subplot(gs[0,2])
fcenter_ax.imshow(inputs[0].permute(1, 2, 0))
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
import time
time.sleep(0.1)
fig.canvas.mpl_disconnect(cid)
   

fig.savefig(os.path.join(PATH,'encoder_weights_examples.png'))