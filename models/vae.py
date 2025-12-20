import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=4096):
        return input.view(input.size(0), size, 1, -1)


class ResVAE(nn.Module):
    def __init__(self, image_channels, hidden_dim=1024, z_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2), #96 -> 47
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), #47 -> 22
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), #22 -> 10
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), #10 -> 4
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(hidden_dim, z_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, hidden_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(hidden_dim, 256, kernel_size=4, stride=2), #1 -> 4
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), #4 -> 10
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), #10 -> 22
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), #22 -> 46
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),  # 46 -> 96
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp.to(std.device)
        return z

    def bottleneck(self, h):
        mu, log_var = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def encode(self, x):
        h = self.encoder(x)
        z, mu, log_var = self.bottleneck(h)
        return z, mu, log_var

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, log_var = self.encode(x)
        z = self.decode(z)
        return z, mu, log_var
