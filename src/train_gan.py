import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import numpy as np
import argparse
import time
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt # Added for plotting

# Project specific imports
from data_loader import RSNAPneumoniaDataset, data_transforms
from dcgan import Generator, Discriminator, weights_init

# --- Plotting Function --- Moved from classifier and adapted
def plot_gan_losses(history, output_path):
    """Plots Generator and Discriminator losses from training history."""
    plt.figure(figsize=(12, 6))

    # Use epoch-level losses instead of iteration-level
    epochs = range(1, len(history.get('G_losses_epoch', [])) + 1)
    g_losses = history.get('G_losses_epoch', [])
    d_losses = history.get('D_losses_epoch', [])
    perceptual_losses = history.get('perceptual_losses', [])
    fm_losses = history.get('feature_matching_losses', [])

    if not epochs or not g_losses or not d_losses:
        print("Warning: Loss data missing or empty in history. Skipping plot generation.")
        plt.close()
        return

    # Plot main losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, g_losses, label="Generator Loss", alpha=0.8)
    plt.plot(epochs, d_losses, label="Discriminator Loss", alpha=0.8)
    plt.title("Generator and Discriminator Loss During Training (Per Epoch)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot additional losses
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
    plt.close() # Close the figure to free memory

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
        ]).eval()
        
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x, y):
        x_feats = []
        y_feats = []
        for block in self.blocks:
            x = block(x)
            y = block(y)
            x_feats.append(x)
            y_feats.append(y)
        return sum(torch.mean((x - y) ** 2) for x, y in zip(x_feats, y_feats))

def feature_matching_loss(real_features, fake_features):
    return sum(torch.mean((real - fake) ** 2) for real, fake in zip(real_features, fake_features))

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # --- Create output directories --- #
    # Gan models subdir
    gan_model_dir = os.path.join(args.model_dir, 'gan')
    os.makedirs(gan_model_dir, exist_ok=True)
    # Gan sample images subdir
    gan_output_dir = os.path.join(args.output_dir, 'gan_images')
    os.makedirs(gan_output_dir, exist_ok=True)
    # Metrics subdir (shared with classifier potentially, use prefix)
    metrics_dir = args.results_dir
    os.makedirs(metrics_dir, exist_ok=True)
    # Figures subdir (shared with classifier potentially, use prefix)
    figures_dir = args.figures_dir
    os.makedirs(figures_dir, exist_ok=True)

    # --- Setup data loader --- #
    try:
        # Initialize dataset with correct parameters
        dataset = RSNAPneumoniaDataset(
            data_dir=os.path.join(args.data_dir, 'Training', 'Images'),
            metadata_file=os.path.join(args.data_dir, 'stage2_train_metadata.csv'),
            transform=data_transforms['train'],
            is_test=False
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        print(f"Loaded training data with {len(dataset)} samples.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure the dataset exists at '{args.data_dir}' and is structured correctly.")
        print("Run `python src/download_dataset.py` first if needed.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Create Generator and Discriminator --- #
    netG = Generator(args.latent_dim, args.num_channels, args.feature_maps_g).to(device)
    netD = Discriminator(args.num_channels, args.feature_maps_d).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    print("Generator Architecture Initialized.")
    print("Discriminator Architecture Initialized.")

    # --- Loss, Optimizers, Labels --- #
    criterion = nn.BCELoss()
    perceptual_loss = PerceptualLoss().to(device)
    # Generate fixed noise for visualization - ensure correct shape
    fixed_noise = torch.randn(args.vis_batch_size, args.latent_dim, device=device)
    real_label = 0.9 # Use label smoothing for real images
    fake_label = 0.
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # --- Training Loop --- #
    print("Starting Training Loop...")
    # Lists to keep track of progress
    history = {
        'G_losses_iter': [],
        'D_losses_iter': [],
        'D_x_iter': [],      # Avg D output for real batch per iter
        'D_G_z1_iter': [], # Avg D output for fake batch before G update
        'D_G_z2_iter': [], # Avg D output for fake batch after G update
        'G_losses_epoch': [], # Avg G loss per epoch
        'D_losses_epoch': [],  # Avg D loss per epoch
        'perceptual_losses': [],
        'feature_matching_losses': []
    }
    iters = 0
    start_time = time.time()

    # Progressive growing schedule
    resolutions = [28, 56, 112, 224]  # Progressive resolutions
    epochs_per_resolution = args.epochs // len(resolutions)
    current_resolution_idx = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_G_loss_accum = 0.0
        epoch_D_loss_accum = 0.0
        num_batches = len(dataloader)

        # Update alpha for progressive growing
        current_resolution = resolutions[min(current_resolution_idx, len(resolutions)-1)]
        alpha = min(1.0, (epoch % epochs_per_resolution) / (epochs_per_resolution * 0.3))
        
        # Progress to next resolution
        if epoch > 0 and epoch % epochs_per_resolution == 0 and current_resolution_idx < len(resolutions)-1:
            current_resolution_idx += 1
            print(f"\nProgressing to resolution {resolutions[current_resolution_idx]}x{resolutions[current_resolution_idx]}")

        D_losses, G_losses = [], []
        D_x_vals, D_G_z1_vals, D_G_z2_vals = [], [], []
        perceptual_losses, fm_losses = [], []

        progress_bar = tqdm(dataloader, total=num_batches, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for i, (real_images, _) in enumerate(progress_bar):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            netD.zero_grad()
            label_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            label_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)

            output_real = netD(real_images, alpha)
            D_x = output_real.mean().item()
            errD_real = criterion(output_real, label_real)

            # Generate noise vector - ensure correct shape
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = netG(noise, alpha)
            output_fake = netD(fake_images.detach(), alpha)
            D_G_z1 = output_fake.mean().item()
            errD_fake = criterion(output_fake, label_fake)

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            output_fake = netD(fake_images, alpha)
            D_G_z2 = output_fake.mean().item()
            
            # Calculate losses
            errG_adv = criterion(output_fake, label_real)
            errG_perceptual = perceptual_loss(fake_images, real_images)
            errG_fm = feature_matching_loss(
                netD.get_intermediate_features(real_images, alpha),
                netD.get_intermediate_features(fake_images, alpha)
            )
            
            # Combined loss
            errG = errG_adv + 10.0 * errG_perceptual + 5.0 * errG_fm
            errG.backward()
            optimizerG.step()

            # Record losses
            D_losses.append(errD.item())
            G_losses.append(errG.item())
            D_x_vals.append(D_x)
            D_G_z1_vals.append(D_G_z1)
            D_G_z2_vals.append(D_G_z2)
            perceptual_losses.append(errG_perceptual.item())
            fm_losses.append(errG_fm.item())

            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': f'{np.mean(D_losses):.3f}',
                'G_loss': f'{np.mean(G_losses):.3f}',
                'D(x)': f'{np.mean(D_x_vals):.3f}',
                'D(G(z))': f'{np.mean(D_G_z2_vals):.3f}'
            })

            # --- Save generated image samples --- #
            if (iters % args.save_interval == 0) or ((epoch == args.epochs-1) and (i == num_batches-1)):
                with torch.no_grad():
                    fake_vis = netG(fixed_noise, alpha).detach().cpu()
                torchvision.utils.save_image(fake_vis, f"{gan_output_dir}/fake_samples_epoch_{epoch+1:03d}_iter_{iters:06d}.png", normalize=True, nrow=8) # Use vis_batch_size indirectly via nrow

            iters += 1

        # --- End of epoch summary and recording --- #
        epoch_time = time.time() - epoch_start_time
        avg_G_loss_epoch = np.mean(G_losses)
        avg_D_loss_epoch = np.mean(D_losses)
        history['G_losses_epoch'].append(avg_G_loss_epoch)
        history['D_losses_epoch'].append(avg_D_loss_epoch)
        history['perceptual_losses'].append(np.mean(perceptual_losses))
        history['feature_matching_losses'].append(np.mean(fm_losses))

        print(f"Epoch {epoch+1}/{args.epochs} Summary - Time: {epoch_time:.2f}s, Avg Loss_D: {avg_D_loss_epoch:.4f}, Avg Loss_G: {avg_G_loss_epoch:.4f}")

        # --- Save model checkpoints --- #
        if (epoch + 1) % args.checkpoint_interval == 0 or (epoch + 1) == args.epochs:
            g_path = os.path.join(gan_model_dir, f'generator_epoch_{epoch+1:03d}.pth')
            d_path = os.path.join(gan_model_dir, f'discriminator_epoch_{epoch+1:03d}.pth')
            torch.save(netG.state_dict(), g_path)
            torch.save(netD.state_dict(), d_path)
            print(f"Saved checkpoints for epoch {epoch+1} to {gan_model_dir}")

    # --- End of training --- #
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")

    # --- Save final models --- #
    torch.save(netG.state_dict(), os.path.join(gan_model_dir, 'generator_final.pth'))
    torch.save(netD.state_dict(), os.path.join(gan_model_dir, 'discriminator_final.pth'))
    print(f"Saved final models to {gan_model_dir}")

    # --- Save training history --- #
    history_filename = os.path.join(metrics_dir, 'gan_training_history.json')
    try:
        with open(history_filename, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Saved training history to {history_filename}")
    except Exception as e:
        print(f"Error saving training history to {history_filename}: {e}")

    # --- Generate final plot --- #
    plot_filename = os.path.join(figures_dir, 'gan_loss_curve.png')
    plot_gan_losses(history, plot_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DCGAN on RSNA Pneumonia Dataset with Enhanced Logging')

    # --- Paths --- #
    parser.add_argument('--data-dir', type=str, default='./data/processed', help='Path to the processed dataset directory')
    parser.add_argument('--model-dir', type=str, default='./models', help='Base directory to save model checkpoints (GAN models saved to ./models/gan/)')
    parser.add_argument('--output-dir', type=str, default='./results', help='Base directory for outputs (generated images saved to ./results/gan_images/)')
    parser.add_argument('--results-dir', type=str, default='./results/metrics', help='Directory to save training history JSON (gan_training_history.json)')
    parser.add_argument('--figures-dir', type=str, default='./results/figures', help='Directory to save generated plot images (gan_loss_curve.png)')

    # --- Model Hyperparameters --- #
    parser.add_argument('--num-channels', type=int, default=3, help='Number of image channels (3 for RGB)')
    parser.add_argument('--latent-dim', type=int, default=100, help='Size of the latent z vector')
    parser.add_argument('--feature-maps-g', type=int, default=32, help='Base feature maps for Generator')
    parser.add_argument('--feature-maps-d', type=int, default=32, help='Base feature maps for Discriminator')

    # --- Training Hyperparameters --- #
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparameter for Adam optimizers')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')

    # --- Logging and Saving --- #
    parser.add_argument('--vis-batch-size', type=int, default=32, help='Batch size for generating visualization images')
    parser.add_argument('--save-interval', type=int, default=1000, help='Save generated image samples every N iterations')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='Save model checkpoints every N epochs')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU even if CUDA is available')

    args = parser.parse_args()

    # # Derived/Corrected Paths (No longer needed as we create subdirs manually)
    # args.gan_model_dir = os.path.join(args.model_dir, 'gan')
    # args.gan_output_dir = os.path.join(args.output_dir, 'gan_images')

    print("--- Training Arguments ---")
    # Print args in a cleaner format
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-------------------------")

    main(args)
