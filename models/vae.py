import torch
import torch.nn as nn
import torch.nn.functional as F


class ResDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channel),
            nn.LeakyReLU(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channel),
        )

        self.shortcut = nn.Identity()
        if stride > 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.GroupNorm(8, out_channel)
            )

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.act(out)


class ResUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super().__init__()

        if stride > 1:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_channel),
                nn.LeakyReLU(0.2)
            )
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channel, out_channel, kernel_size=1),
                nn.GroupNorm(8, out_channel)
            )
        else:
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_channel),
                nn.LeakyReLU(0.2)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1),
                nn.GroupNorm(8, out_channel)
            ) if in_channel != out_channel else nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channel),
        )

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.upsample(x)
        out = self.conv2(out)
        out = out + residual
        return self.act(out)


class ResVAE(nn.Module):
    def __init__(self, channels, image_size, hidden_dim=256, z_dim=32):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.z_dim = z_dim

        # Encoder
        self.encoder = nn.Sequential(
            ResDownBlock(channels, 32, stride=2),  # 96 -> 48
            ResDownBlock(32, 64, stride=2),  # 48 -> 24
            ResDownBlock(64, 128, stride=2),  # 24 -> 12
            ResDownBlock(128, 256, stride=2),  # 12 -> 6
        )

        # Вычисляем размер после encoder
        with torch.no_grad():
            dummy = torch.zeros(1, channels, image_size, image_size)
            enc_out = self.encoder(dummy)
            self.enc_shape = enc_out.shape[1:]  # (256, 6, 6)
            self.flat_size = enc_out.flatten(1).shape[1]  # 256 * 6 * 6 = 9216

        # Bottleneck - добавляем дополнительное сжатие
        self.pre_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Mu и log_var
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

        # Инициализация для более стабильного latent space
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.1)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.1)
        nn.init.constant_(self.fc_logvar.bias, -2.0)  # Начинаем с маленького std

        # Decoder projection
        self.fc_decode = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.flat_size),
            nn.LeakyReLU(0.2),
        )

        # Decoder conv
        self.decoder_conv = nn.Sequential(
            ResUpBlock(256, 128, stride=2),  # 6 -> 12
            ResUpBlock(128, 64, stride=2),  # 12 -> 24
            ResUpBlock(64, 32, stride=2),  # 24 -> 48
            ResUpBlock(32, 32, stride=2),  # 48 -> 96
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.pre_latent(h)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        # Ограничиваем log_var для стабильности
        log_var = torch.clamp(log_var, min=-10, max=2)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, *self.enc_shape)
        h = self.decoder_conv(h)
        return self.final_conv(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def sample(self, n_samples, device):
        """Генерация новых изображений из prior"""
        z = torch.randn(n_samples, self.z_dim, device=device)
        return self.decode(z)