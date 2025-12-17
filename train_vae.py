import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import os

from models.vae_dataset import VAENpDataset
from models.vae import ResVAE

# ============== НАСТРОЙКИ ==============

# Data parameters
DATA_PATH = 'data'
IMG_SIZE = 96

# Model parameters
IN_CHANNELS = 3
HIDDEN_DIM = 256
Z_DIM = 32

# Training parameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0001
BETA = 0.5

# Checkpoint parameters
CHECKPOINT_DIR = 'checkpoints'
SAVE_EVERY = 2  # Сохранять каждые N эпох
RESUME_PATH = None  # Путь к чекпоинту для продолжения обучения (None = с нуля)

# Device: 'auto', 'cuda', 'cpu', 'mps'
DEVICE = 'auto'


# =======================================


def get_device(device_arg):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_arg)


def save_checkpoint(model, optimizer, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f'Checkpoint saved: {path}')


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    print(f'Checkpoint loaded: {path}, resuming from epoch {start_epoch}')
    return start_epoch, loss


def main():
    device = get_device(DEVICE)
    print(f'Using device: {device}')

    dataset = VAENpDataset(path_to_np_files=DATA_PATH, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    resvae = ResVAE(
        IN_CHANNELS,
        IMG_SIZE,
        hidden_dim=HIDDEN_DIM,
        z_dim=Z_DIM
    ).to(device)

    optimizer = optim.Adam(resvae.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss(reduction='mean')

    start_epoch = 0
    if RESUME_PATH:
        start_epoch, _ = load_checkpoint(resvae, optimizer, RESUME_PATH, device)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(start_epoch, EPOCHS):
        resvae.train()
        epoch_loss = 0.0

        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{EPOCHS}')
        for i, x in loop:
            x = x.to(device)

            x_recon, mu, log_var = resvae(x)

            rec_loss = loss_fn(x_recon, x)
            kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

            loss = rec_loss + (kl_div * BETA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()
            loop.set_postfix(full_loss=loss.item(), kl=kl_div.item(), rec_loss=rec_loss.item())

        if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pt')
            save_checkpoint(resvae, optimizer, epoch, epoch_loss, checkpoint_path)

            latest_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pt')
            save_checkpoint(resvae, optimizer, epoch, epoch_loss, latest_path)


if __name__ == '__main__':
    main()