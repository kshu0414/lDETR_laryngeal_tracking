import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def show_landmarks(image, landmarks,predictions):
    """Show image with landmarks"""
    plt.imshow(image.permute(1, 2, 0)  )
    plt.scatter(landmarks[0], landmarks[1], marker='o',facecolors='none', edgecolors='red')
    plt.scatter(landmarks[2], landmarks[3], marker='s',facecolors='none', edgecolors='red')
    plt.scatter(predictions[0],predictions[1],marker='o',facecolors='none', edgecolors='blue')
    plt.scatter(predictions[2],predictions[3],marker='s',facecolors='none', edgecolors='blue')

def distance_error(targ,pred):
    targ = torch.reshape(targ,[-1,2])
    pred = torch.reshape(pred,[-1,2])

    dist = F.pairwise_distance(pred,targ,p=2)
    dist = dist.view(-1,2)
    temp = torch.mean(dist,dim=1)
    return torch.mean(dist,dim=1)

def mean_distance_error(targ,pred):
    targ = torch.reshape(targ,[-1,2])
    pred = torch.reshape(pred,[-1,2])

    return torch.mean(F.pairwise_distance(pred,targ,p=2))