import torch, torchvision
import models as my_models
from torchvision import models, transforms
import torchvision.datasets as dset



# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 500

#size of feature maps
ngf = 256
ndf = int(ngf/4)

#define hyper params here
image_size = 128
num_epochs = 100
lr = 1e-5

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


dataroot = "./datasets/dataset1"
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                           ]))

def wasserstein_loss(y_true, y_pred):
    return -torch.mean(y_true * y_pred)
