import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flattened_dimension_downsize = 500
pad = 1
dil = 1
ker = 3
stri = 2

def calc_out_conv_layers(in_h, in_w, layers):
    out_h = in_h
    out_w = in_w
    for i in range(layers):
        out_h = (out_h + 2*pad - dil * (ker-1) - 1)/stri + 1
        out_w = (out_w + 2*pad - dil * (ker-1) - 1)/stri + 1
    #print(out_h)
    #print(out_w)
    return round(out_h), round(out_w)

class Encoder(torch.nn.Module):
    def __init__(self, in_shape, flattened_dim, hidden_dims):
        super(Encoder, self).__init__()
        #encoder layers
        modules = []
        in_channels=in_shape[0]
        for h_dim in hidden_dims:
            modules.append(
                        torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels, out_channels = h_dim, kernel_size=ker, stride=stri, padding=pad),
                            torch.nn.BatchNorm2d(h_dim), 
                            torch.nn.LeakyReLU())
                           )
            in_channels = h_dim

        self.conv_layers = torch.nn.Sequential(*modules)

        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear_mu = torch.nn.Linear(flattened_dim, flattened_dimension_downsize)
        self.linear_log = torch.nn.Linear(flattened_dim, flattened_dimension_downsize)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.conv_layers(x)
        #print('last conv layer ', out.size())
        flattened = self.flatten(out)
        self.mu = self.linear_mu(flattened)
        self.log_var = self.linear_log(flattened)
        epsilon = torch.normal(mean=0., std=1., size=self.mu.size()).to(device)
        out = self.mu + torch.exp(self.log_var / 2) * epsilon
        return out

class Decoder(torch.nn.Module):
    def __init__(self, flattened_dim, rebuild_shape, hidden_dims):
        super(Decoder, self).__init__()
        #decoder layers
        self.rebuild_shape = rebuild_shape
        self.linear_in = torch.nn.Linear(flattened_dimension_downsize, flattened_dim)

        modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=ker,
                                       stride = stri,
                                       padding=pad,
                                       output_padding=1),
                    torch.nn.BatchNorm2d(hidden_dims[i + 1]),
                    torch.nn.LeakyReLU())
            )

        self.conv_layers = torch.nn.Sequential(*modules)

        self.final_layer = torch.nn.Sequential(
                            torch.nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=ker,
                                               stride=stri,
                                               padding=pad,
                                               output_padding=1),
                            torch.nn.BatchNorm2d(hidden_dims[-1]),
                            torch.nn.LeakyReLU(),
                            torch.nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= ker, padding= pad),
                            torch.nn.Tanh())

    def forward(self, x):
        out = self.linear_in(x)
        out = torch.reshape(out, self.rebuild_shape)

        out = self.conv_layers(out)

        out = self.final_layer(out)
        return out


class AE(torch.nn.Module):
    def __init__(self, input_dims, batch_size):
        super(AE, self).__init__()
        hidden_dims = [8, 16, 32, 64, 128, 256, 512]
        final_size = calc_out_conv_layers(input_dims[1], input_dims[2], len(hidden_dims))
        #print(final_size)
        flattened_dim = 8192#batch_size * hidden_dims[-1] * final_size[0] * final_size[1]
        shape = (batch_size, hidden_dims[-1], 4, 4)#(batch_size, hidden_dims[-1], final_size[0], final_size[1])

        self.encoder = Encoder(input_dims, flattened_dim, hidden_dims)
        self.decoder = Decoder(flattened_dim, shape, hidden_dims)

    def forward(self, x):
        h = x
        h = self.encoder(h)
        h = self.decoder(h)
        return h




from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import *

flatened_total_dim = 196608

class BetaVAE(torch.nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(flatened_total_dim, latent_dim)
        self.fc_var = nn.Linear(flatened_total_dim, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, flatened_total_dim)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        self.rebuild_shape = result.size()
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = torch.reshape(result, self.rebuild_shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        #print(recons.size())
        #print(input.size())
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]