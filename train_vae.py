import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import os
import math

from models.vae_dataset import VAENpDataset
from models.vae import ResVAE

# ============== НАСТРОЙКИ ==============
DATA_PATH = 'data'
IMG_SIZE = 96

IN_CHANNELS = 3
HIDDEN_DIM = 256
Z_DIM = 32  # Уменьшено! Для Car Racing достаточно

BATCH_SIZE = 64  # Увеличен для стабильности
EPOCHS = 50
LEARNING_RATE = 0.0003

# Beta-VAE параметры
BETA_MAX = 0.5  # Максимальный beta
BETA_WARMUP_EPOCHS = 10  # Постепенное увеличение beta
FREE_BITS = 0.5  # Минимальный KL на измерение (предотвращает collapse)

CHECKPOINT_DIR = 'checkpoints'
SAVE_EVERY = 5
RESUME_PATH = None
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


def compute_kl_divergence(mu, log_var, free_bits=0.0):
    """
    KL divergence с поддержкой free bits для предотвращения posterior collapse
    """
    # KL для каждого измерения
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

    if free_bits > 0:
        # Free bits: минимум KL на каждое измерение
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Сумма по латентным измерениям, среднее по батчу
    kl = torch.mean(torch.sum(kl_per_dim, dim=1))
    return kl


def get_beta(epoch, warmup_epochs, beta_max, schedule='linear'):
    """
    Постепенное увеличение beta (KL annealing)
    """
    if schedule == 'linear':
        return min(beta_max * (epoch / warmup_epochs), beta_max)
    elif schedule == 'cyclical':
        # Cyclical annealing
        cycle_length = warmup_epochs
        cycle = epoch % cycle_length
        return beta_max * min(cycle / (cycle_length / 2), 1.0)
    elif schedule == 'cosine':
        if epoch >= warmup_epochs:
            return beta_max
        return beta_max * (1 - math.cos(math.pi * epoch / warmup_epochs)) / 2
    return beta_max


def main():
    device = get_device(DEVICE)
    print(f'Using device: {device}')

    dataset = VAENpDataset(path_to_np_files=DATA_PATH, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)

    print(f'Dataset size: {len(dataset)} images')

    resvae = ResVAE(
        IN_CHANNELS,
        IMG_SIZE,
        hidden_dim=HIDDEN_DIM,
        z_dim=Z_DIM
    ).to(device)

    optimizer = optim.AdamW(resvae.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    start_epoch = 0
    if RESUME_PATH:
        start_epoch, _ = load_checkpoint(resvae, optimizer, RESUME_PATH, device)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(start_epoch, EPOCHS):
        resvae.train()

        # Beta annealing
        beta = get_beta(epoch, BETA_WARMUP_EPOCHS, BETA_MAX, schedule='cosine')

        epoch_rec_loss = 0.0
        epoch_kl_loss = 0.0

        loop = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc=f'Epoch {epoch + 1}/{EPOCHS}')

        for i, x in loop:
            x = x.to(device)

            x_recon, mu, log_var = resvae(x)

            # Reconstruction loss (можно использовать и BCE для [0,1])
            rec_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')

            # KL divergence с free bits
            kl_loss = compute_kl_divergence(mu, log_var, free_bits=FREE_BITS)

            # Total loss
            loss = rec_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(resvae.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_rec_loss += rec_loss.item()
            epoch_kl_loss += kl_loss.item()

            loop.set_postfix(
                rec=f'{rec_loss.item():.4f}',
                kl=f'{kl_loss.item():.2f}',
                beta=f'{beta:.4f}',
                mu_std=f'{mu.std().item():.3f}'  # Должен быть > 0!
            )

        scheduler.step()

        avg_rec = epoch_rec_loss / len(dataloader)
        avg_kl = epoch_kl_loss / len(dataloader)

        print(f'Epoch {epoch + 1}: rec_loss={avg_rec:.4f}, kl_loss={avg_kl:.2f}, beta={beta:.4f}')

        if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pt')
            save_checkpoint(resvae, optimizer, epoch, avg_rec + beta * avg_kl, checkpoint_path)

            latest_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pt')
            save_checkpoint(resvae, optimizer, epoch, avg_rec + beta * avg_kl, latest_path)


if __name__ == '__main__':
    main()