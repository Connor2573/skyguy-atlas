import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flattened_dimension_downsize = 125

class Encoder(torch.nn.Module):
    def __init__(self, in_shape, flattened_dim):
        super(Encoder, self).__init__()
        #encoder layers
        self.conv1 = torch.nn.Conv2d(in_channels=in_shape[0], out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #self.batch1 = torch.nn.BatchNorm2d(num_features=64)
        self.dropout1 = torch.nn.Dropout()

        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout2 = torch.nn.Dropout()

        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.dropout3 = torch.nn.Dropout()

        self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.dropout4 = torch.nn.Dropout()

        self.flatten = torch.nn.Flatten(start_dim=0)
        self.linear_mu = torch.nn.Linear(flattened_dim, flattened_dimension_downsize)
        self.linear_log = torch.nn.Linear(flattened_dim, flattened_dimension_downsize)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        #out = self.pool1(out)
        #out = self.batch1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        
        out = self.conv3(out)
        out = self.activation(out)
        out = self.dropout3(out)
        
        out = self.conv4(out)
        out = self.activation(out)
        out = self.dropout4(out)

        flattened = self.flatten(out)
        self.mu = self.linear_mu(flattened)
        self.log = self.linear_log(flattened)
        epsilon = torch.normal(mean=0., std=1., size=self.mu.size()).to(device)
        out = self.mu + torch.exp(self.log / 2) * epsilon
        return out

    def kl_loss(self):
        kl_loss = -.5 * torch.sum(1 + self.log - torch.square(self.mu) - torch.exp(self.log))
        return kl_loss

class Decoder(torch.nn.Module):
    def __init__(self, flattened_dim, rebuild_shape):
        super(Decoder, self).__init__()
        #decoder layers
        self.rebuild_shape = rebuild_shape
        self.linear_in = torch.nn.Linear(flattened_dimension_downsize, flattened_dim)

        self.conv4 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.dropout4 = torch.nn.Dropout()

        self.conv3 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout3 = torch.nn.Dropout()

        self.conv2 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = torch.nn.Dropout()

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.linear_in(x)
        out = torch.reshape(out, self.rebuild_shape)

        out = self.conv4(out)
        out = self.activation(out)
        out = self.dropout4(out)

        out = self.conv3(out)
        out = self.activation(out)
        out = self.dropout3(out)

        out = self.conv2(out)
        out = self.activation(out)
        out = self.dropout2(out)

        out = self.conv1(out)
        out = self.activation(out)
        return out


class AE(torch.nn.Module):
    def __init__(self, input_dims):
        super(AE, self).__init__()
        last_conv_layer_size = 8 #the size of the last encoder convolution
        flattened_dim = last_conv_layer_size * input_dims[1] * input_dims[2]
        shape = (last_conv_layer_size, input_dims[1], input_dims[2])

        self.encoder = Encoder(input_dims, flattened_dim)
        self.decoder = Decoder(flattened_dim, shape)

    def forward(self, x):
        h = x
        h = self.encoder(h)
        h = self.decoder(h)
        return h





