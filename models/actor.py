import torch
import torch.nn as nn
import torch.nn.functional as f


class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.loc_head = nn.Sequential(
            nn.Linear(latent_dim, action_dim),
            nn.Tanh()
        )

        self.scale_head = nn.Sequential(
            nn.Linear(latent_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # z: [batch, latent_dim]
        mean = self.loc_head(x)  # [batch, action_dim]
        log_std = self.scale_head(x)
        return mean, log_std


class Critic(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.layer1 = nn.Linear(z_dim, 64)
        self.layer2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, z):
        x = self.layer1(z)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class PolicyNetwork(nn.Module):
    """observation -> loc, scale"""

    def __init__(self, vae, actor):
        super().__init__()
        self.vae = vae
        self.actor = actor

    def forward(self, obs):
        with torch.no_grad():
            z, _, _ = self.vae.encode(obs)
        return self.actor(z)


class ValueNetwork(nn.Module):
    """observation -> value"""

    def __init__(self, vae, critic):
        super().__init__()
        self.vae = vae
        self.critic = critic

    def forward(self, obs):
        with torch.no_grad():
            z, _, _ = self.vae.encode(obs)
        return self.critic(z)