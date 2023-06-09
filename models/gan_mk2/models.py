import torch
import math
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #discriminator layers
        #torch.nn.Conv2d(in_channels=3, 64, stride=2)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

class GAN(nn.Module):
    def __init__(self, input_dim = (128, 128, 3), d_conv_filters=[64, 64, 128, 128], d_conv_kernel=[5, 5, 5, 5]):
        dis = Discriminator()
        gen = Generator()
