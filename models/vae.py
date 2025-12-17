import torch
import torch.nn as nn
import torch.nn.functional as F


class ResDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResDownBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
        )

        self.shortcut = nn.Identity()
        if stride > 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual
        out = self.relu(out)
        return out

class ResUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super(ResUpBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, output_padding=(1 if stride > 1 else 0)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.relu = nn.ReLU()

        self.shortcut = nn.Identity()
        if stride > 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=1, stride=stride,
                                   output_padding=(1 if stride > 1 else 0)),
                nn.BatchNorm2d(out_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual
        out = self.relu(out)
        return out



class ResVAE(nn.Module):
    def __init__(self, channels, image_size, hidden_dim=200, z_diz=20):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        self.encoder = nn.Sequential(
            ResDownBlock(channels, 32, stride=2),
            ResDownBlock(32, 64, stride=2),
            ResDownBlock(64, 128, stride=2),
            ResDownBlock(128, 256, stride=1),
        )

        self.decoder = nn.Sequential(
            ResUpBlock(256, 128, stride=1),
            ResUpBlock(128, 64, stride=2),
            ResUpBlock(64, 32, stride=2),
            ResUpBlock(32, channels, stride=2),
        )

        _dummy_input = torch.zeros(2, channels, image_size, image_size)
        with torch.no_grad():
            dummy_output = self.encoder(_dummy_input)

        self.enc_out_shape = dummy_output.shape

        self.num_features = dummy_output.flatten(1).shape[1]

        self.hid_2mu = nn.Sequential(
            nn.Linear(self.num_features, hidden_dim),
            nn.Linear(hidden_dim, z_diz),
        )
        self.hid_2log_var = nn.Sequential(
            nn.Linear(self.num_features, hidden_dim),
            nn.Linear(hidden_dim, z_diz),
        )

        self.z_2hid = nn.Sequential(
            nn.Linear(z_diz, hidden_dim),
            nn.Linear(hidden_dim, self.num_features),
        )

        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.relu(self.encoder(x))
        h = x.flatten(1)

        mu, log_var = self.hid_2mu(h), self.hid_2log_var(h)

        return mu, log_var

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        x = h.reshape(-1, self.enc_out_shape[1], self.enc_out_shape[2], self.enc_out_shape[3])
        return torch.sigmoid(self.decoder(x)).reshape((-1, self.channels, self.image_size, self.image_size))

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z_perametrized = mu +eps*std
        x_recon = self.decode(z_perametrized)
        return x_recon, mu, log_var
