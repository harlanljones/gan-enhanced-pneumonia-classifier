import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import json
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import RSNAPneumoniaDataset, data_transforms
from cgan import Generator, Discriminator, weights_init

def plot_gan_losses(history, output_path):
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(history.get('G_losses_epoch', [])) + 1)
    g_losses = history.get('G_losses_epoch', [])
    d_losses = history.get('D_losses_epoch', [])
    perceptual_losses = history.get('perceptual_losses', [])
    fm_losses = history.get('feature_matching_losses', [])
    if not epochs or not g_losses or not d_losses:
        print("Warning: Loss data missing. Skipping plot.")
        plt.close()
        return

    plt.subplot(2, 1, 1)
    plt.plot(epochs, g_losses, label="Generator Loss", alpha=0.8)
    plt.plot(epochs, d_losses, label="Discriminator Loss", alpha=0.8)
    plt.title("Generator and Discriminator Loss During Training (Per Epoch)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, perceptual_losses, label="Perceptual Loss", alpha=0.8)
    plt.plot(epochs, fm_losses, label="Feature Matching Loss", alpha=0.8)
    plt.title("Additional Loss Components During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print(f"Saved GAN loss plot to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
    plt.close()

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:16]
        ]).eval()
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, x, y):
        x_feats, y_feats = [], []
        for block in self.blocks:
            x = block(x)
            y = block(y)
            x_feats.append(x)
            y_feats.append(y)
        return sum(torch.mean((a - b) ** 2) for a, b in zip(x_feats, y_feats))

def feature_matching_loss(real_features, fake_features):
    return sum(torch.mean((real - fake) ** 2) for real, fake in zip(real_features, fake_features))

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    gan_model_dir = os.path.join(args.model_dir, 'gan')
    os.makedirs(gan_model_dir, exist_ok=True)
    gan_output_dir = os.path.join(args.output_dir, 'gan_images')
    os.makedirs(gan_output_dir, exist_ok=True)
    metrics_dir = args.results_dir
    os.makedirs(metrics_dir, exist_ok=True)
    figures_dir = args.figures_dir
    os.makedirs(figures_dir, exist_ok=True)

    try:
        dataset = RSNAPneumoniaDataset(
            data_dir=os.path.join(args.data_dir, 'Training', 'Images'),
            metadata_file=os.path.join(args.data_dir, 'stage2_train_metadata.csv'),
            transform=data_transforms['train'],
            is_test=False
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        print(f"Loaded training data with {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    num_classes = 2
    netG = Generator(args.latent_dim, num_classes, args.num_channels, args.feature_maps_g).to(device)
    netD = Discriminator(num_classes, args.num_channels, args.feature_maps_d).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    perceptual_loss = PerceptualLoss().to(device)
    fixed_noise = torch.randn(args.vis_batch_size, args.latent_dim, device=device)

    fixed_labels = torch.tensor(
        np.tile(np.arange(num_classes), args.vis_batch_size // num_classes + 1)[:args.vis_batch_size],
        dtype=torch.long, device=device
    )

 
    real_label = 0.9
    fake_label = 0.1  
    
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    history = {'G_losses_iter': [], 'D_losses_iter': [], 'D_x_iter': [], 'D_G_z1_iter': [], 'D_G_z2_iter': [],
               'G_losses_epoch': [], 'D_losses_epoch': [], 'perceptual_losses': [], 'feature_matching_losses': []}
    iters = 0
    start_time = time.time()

    resolutions = [28, 56, 112, 224]
    epochs_per_resolution = args.epochs // len(resolutions)
    current_resolution_idx = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        num_batches = len(dataloader)

        current_resolution = resolutions[min(current_resolution_idx, len(resolutions)-1)]
        alpha = min(1.0, (epoch % epochs_per_resolution) / (epochs_per_resolution * 0.3))
        if epoch > 0 and epoch % epochs_per_resolution == 0 and current_resolution_idx < len(resolutions)-1:
            current_resolution_idx += 1
            print(f"\nProgressing to resolution {resolutions[current_resolution_idx]}x{resolutions[current_resolution_idx]}")

        D_losses, G_losses = [], []
        D_x_vals, D_G_z1_vals, D_G_z2_vals = [], [], []
        perceptual_losses, fm_losses = [], []

        progress_bar = tqdm(dataloader, total=num_batches, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for i, (real_images, real_labels) in enumerate(progress_bar):
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_images.size(0)

            real_labels_smooth = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            real_labels_smooth = real_labels_smooth - 0.1 * torch.rand(batch_size, device=device)  
            
            fake_labels_smooth = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
            fake_labels_smooth = fake_labels_smooth + 0.1 * torch.rand(batch_size, device=device)  
            netD.zero_grad()
            output_real = netD(real_images, real_labels, alpha)
            D_x = torch.sigmoid(output_real).mean().item()
            errD_real = criterion(output_real, real_labels_smooth)

            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_images = netG(noise, fake_labels, alpha)
            output_fake = netD(fake_images.detach(), fake_labels, alpha)
            D_G_z1 = torch.sigmoid(output_fake).mean().item()
            errD_fake = criterion(output_fake, fake_labels_smooth)


            errD = errD_real + errD_fake

            if D_x < 0.8 or D_G_z1 > 0.2 or epoch < 5:
                errD.backward()
                optimizerD.step()

            netG.zero_grad()
            output_fake = netD(fake_images, fake_labels, alpha)
            D_G_z2 = torch.sigmoid(output_fake).mean().item()

            errG_adv = criterion(output_fake, real_labels_smooth) 
            errG_perceptual = perceptual_loss(fake_images, real_images)
            errG_fm = feature_matching_loss(
                netD.get_intermediate_features(real_images, real_labels, alpha),
                netD.get_intermediate_features(fake_images, fake_labels, alpha)
            )
 
            errG = errG_adv + 10.0 * errG_perceptual + 5.0 * errG_fm
            errG.backward()
            optimizerG.step()

            D_losses.append(errD.item())
            G_losses.append(errG.item())
            D_x_vals.append(D_x)
            D_G_z1_vals.append(D_G_z1)
            D_G_z2_vals.append(D_G_z2)
            perceptual_losses.append(errG_perceptual.item())
            fm_losses.append(errG_fm.item())

            progress_bar.set_postfix({
                'D_loss': f'{np.mean(D_losses):.3f}',
                'G_loss': f'{np.mean(G_losses):.3f}',
                'D(x)': f'{np.mean(D_x_vals):.3f}',
                'D(G(z))': f'{np.mean(D_G_z2_vals):.3f}'
            })

            if (iters % args.save_interval == 0) or ((epoch == args.epochs-1) and (i == num_batches-1)):
                with torch.no_grad():
                    fake_vis = netG(fixed_noise, fixed_labels, alpha).detach().cpu()
                torchvision.utils.save_image(fake_vis, f"{gan_output_dir}/fake_samples_epoch_{epoch+1:03d}_iter_{iters:06d}.png", normalize=True, nrow=8)

            iters += 1

        epoch_time = time.time() - epoch_start_time
        history['G_losses_epoch'].append(np.mean(G_losses))
        history['D_losses_epoch'].append(np.mean(D_losses))
        history['perceptual_losses'].append(np.mean(perceptual_losses))
        history['feature_matching_losses'].append(np.mean(fm_losses))

        print(f"Epoch {epoch+1}/{args.epochs} Summary - Time: {epoch_time:.2f}s, Avg Loss_D: {np.mean(D_losses):.4f}, Avg Loss_G: {np.mean(G_losses):.4f}")

        if (epoch + 1) % args.checkpoint_interval == 0 or (epoch + 1) == args.epochs:
            g_path = os.path.join(gan_model_dir, f'generator_epoch_{epoch+1:03d}.pth')
            d_path = os.path.join(gan_model_dir, f'discriminator_epoch_{epoch+1:03d}.pth')
            torch.save(netG.state_dict(), g_path)
            torch.save(netD.state_dict(), d_path)
            print(f"Saved checkpoints for epoch {epoch+1} to {gan_model_dir}")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")
    torch.save(netG.state_dict(), os.path.join(gan_model_dir, 'generator_final.pth'))
    torch.save(netD.state_dict(), os.path.join(gan_model_dir, 'discriminator_final.pth'))
    print(f"Saved final models to {gan_model_dir}")

    history_filename = os.path.join(metrics_dir, 'gan_training_history.json')
    try:
        with open(history_filename, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Saved training history to {history_filename}")
    except Exception as e:
        print(f"Error saving training history to {history_filename}: {e}")

    plot_filename = os.path.join(figures_dir, 'gan_loss_curve.png')
    plot_gan_losses(history, plot_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train cDCGAN on RSNA Pneumonia Dataset with Enhanced Logging')
    parser.add_argument('--data-dir', type=str, default='./data/processed')
    parser.add_argument('--model-dir', type=str, default='./models')
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--results-dir', type=str, default='./results/metrics')
    parser.add_argument('--figures-dir', type=str, default='./results/figures')
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--feature-maps-g', type=int, default=32)
    parser.add_argument('--feature-maps-d', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--vis-batch-size', type=int, default=32)
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--checkpoint-interval', type=int, default=5)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    print("--- Training Arguments ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-------------------------")
    main(args)
