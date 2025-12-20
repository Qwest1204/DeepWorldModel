import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import os

from models.vae_dataset import VAENpDataset
from models.vae import ResVAE

# ============== НАСТРОЙКИ ==============
DATA_PATH = 'data'
IMG_SIZE = 96

IN_CHANNELS = 3
HIDDEN_DIM = 1024
Z_DIM = 32

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0003

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

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def main():
    device = get_device(DEVICE)
    print(f'Using device: {device}')

    dataset = VAENpDataset(path_to_np_files=DATA_PATH, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)

    print(f'Dataset size: {len(dataset)} images')

    resvae = ResVAE(
        image_channels=IN_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        z_dim=Z_DIM
    ).to(device)

    optimizer = optim.AdamW(resvae.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    start_epoch = 0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(start_epoch, EPOCHS):
        resvae.train()

        epoch_rec_loss = 0.0
        epoch_kl_loss = 0.0

        loop = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc=f'Epoch {epoch + 1}/{EPOCHS}')

        for i, x in loop:
            x = x.to(device)

            x_recon, mu, log_var = resvae(x)

            loss, bce, kld = loss_fn(x_recon, x, mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_rec_loss += loss.item()
            epoch_kl_loss += kld.item()

            loop.set_postfix(
                rec=f'{loss.item():.4f}',
                kl=f'{kld.item():.2f}',
                mu_std=f'{mu.std().item():.3f}'
            )

        avg_rec = epoch_rec_loss / len(dataloader)
        avg_kl = epoch_kl_loss / len(dataloader)

        print(f'Epoch {epoch + 1}: rec_loss={avg_rec:.4f}, kl_loss={avg_kl:.2f}')

        if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pt')
            save_checkpoint(resvae, optimizer, epoch, avg_rec + avg_kl, checkpoint_path)

            latest_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pt')
            save_checkpoint(resvae, optimizer, epoch, avg_rec + avg_kl, latest_path)


if __name__ == '__main__':
    main()