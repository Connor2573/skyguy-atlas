import torch, torchvision
import models as my_models
from torchvision import datasets, models, transforms

dataroot = "./datasets/dataset3"

#define hyper params here
image_size = 128
num_epochs = 100
lr = 1e-5

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                           ]))

def wasserstein_loss(y_true, y_pred):
    return -torch.mean(y_true * y_pred)
