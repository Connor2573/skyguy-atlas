import torch

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, flattened_dim):
        super(Encoder, self).__init__()
        #encoder layers
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.flatten = torch.nn.Flatten(start_dim=0)
        self.linear_out = torch.nn.Linear(flattened_dim, 2573) #2573 is made up
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.activation(out)

        out = self.conv4(out)
        out = self.activation(out)

        out = self.flatten(out)
        out = self.linear_out(out)
        return out

class Decoder(torch.nn.Module):
    def __init__(self, flattened_dim, rebuild_shape):
        super(Decoder, self).__init__()
        #decoder layers
        self.rebuild_shape = rebuild_shape
        self.linear_in = torch.nn.Linear(2573, flattened_dim)
        self.conv4 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.linear_in(x)
        out = torch.reshape(out, self.rebuild_shape)

        out = self.conv4(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv1(out)
        out = self.activation(out)
        return out


class AE(torch.nn.Module):
    def __init__(self, input_dims):
        super(AE, self).__init__()
        in_channels = input_dims[0]
        last_conv_layer_size = 8
        flattened_dim = last_conv_layer_size * input_dims[1] * input_dims[2] #64 is the out channels of the last convulutional layer of the encoder
        shape = (last_conv_layer_size, input_dims[1], input_dims[2])

        self.encoder = Encoder(in_channels, flattened_dim)
        self.decoder = Decoder(flattened_dim, shape)

    def forward(self, x):
        h = x
        h = self.encoder(h)
        h = self.decoder(h)
        return h





