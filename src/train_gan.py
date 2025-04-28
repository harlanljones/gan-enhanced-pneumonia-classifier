import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import argparse
import time
import json # Added for saving history
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt # Added for plotting

# Project specific imports
from data_loader import get_dataloaders # Assuming get_dataloaders can provide just the train loader
from dcgan import Generator, Discriminator, weights_init

# --- Plotting Function --- Moved from classifier and adapted
def plot_gan_losses(history, output_path):
    """Plots Generator and Discriminator losses from training history."""
    plt.figure(figsize=(12, 6))

    iters = range(len(history.get('G_losses_iter', [])))
    g_losses = history.get('G_losses_iter', [])
    d_losses = history.get('D_losses_iter', [])

    if not iters or not g_losses or not d_losses:
        print("Warning: Loss data missing or empty in history. Skipping plot generation.")
        plt.close()
        return

    plt.plot(iters, g_losses, label="Generator Loss", alpha=0.8)
    plt.plot(iters, d_losses, label="Discriminator Loss", alpha=0.8)

    plt.title("Generator and Discriminator Loss During Training (Per Iteration)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss (BCELoss)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print(f"Saved GAN loss plot to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
    plt.close() # Close the figure to free memory


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
        train_loader, _ = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
        print(f"Loaded training data with {len(train_loader.dataset)} samples.")
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
    fixed_noise = torch.randn(args.vis_batch_size, args.latent_dim, 1, 1, device=device) # Smaller batch for vis
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
        'D_losses_epoch': []  # Avg D loss per epoch
    }
    iters = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_G_loss_accum = 0.0
        epoch_D_loss_accum = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for i, data in progress_bar:

            # (1) Update D network
            netD.zero_grad()
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output_real = netD(real_data).view(-1)
            errD_real = criterion(output_real, label)
            errD_real.backward()
            D_x = output_real.mean().item()

            noise = torch.randn(b_size, args.latent_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output_fake = netD(fake.detach()).view(-1)
            errD_fake = criterion(output_fake, label)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network
            netG.zero_grad()
            label.fill_(real_label)
            output_fake_for_G = netD(fake).view(-1)
            errG = criterion(output_fake_for_G, label)
            errG.backward()
            D_G_z2 = output_fake_for_G.mean().item()
            optimizerG.step()

            # --- Record losses and metrics per iteration --- #
            history['G_losses_iter'].append(errG.item())
            history['D_losses_iter'].append(errD.item())
            history['D_x_iter'].append(D_x)
            history['D_G_z1_iter'].append(D_G_z1)
            history['D_G_z2_iter'].append(D_G_z2)

            epoch_G_loss_accum += errG.item()
            epoch_D_loss_accum += errD.item()

            # Update progress bar
            progress_bar.set_postfix({'Loss_D': f"{errD.item():.4f}", 'Loss_G': f"{errG.item():.4f}",
                                      'D(x)': f"{D_x:.4f}", 'D(G(z))': f"{D_G_z1:.4f}/{D_G_z2:.4f}"})

            # --- Save generated image samples --- #
            if (iters % args.save_interval == 0) or ((epoch == args.epochs-1) and (i == num_batches-1)):
                with torch.no_grad():
                    fake_vis = netG(fixed_noise).detach().cpu()
                vutils.save_image(fake_vis, f"{gan_output_dir}/fake_samples_epoch_{epoch+1:03d}_iter_{iters:06d}.png", normalize=True, nrow=8) # Use vis_batch_size indirectly via nrow

            iters += 1

        # --- End of epoch summary and recording --- #
        epoch_time = time.time() - epoch_start_time
        avg_G_loss_epoch = epoch_G_loss_accum / num_batches
        avg_D_loss_epoch = epoch_D_loss_accum / num_batches
        history['G_losses_epoch'].append(avg_G_loss_epoch)
        history['D_losses_epoch'].append(avg_D_loss_epoch)

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
    parser.add_argument('--feature-maps-g', type=int, default=64, help='Base feature maps for Generator')
    parser.add_argument('--feature-maps-d', type=int, default=64, help='Base feature maps for Discriminator')

    # --- Training Hyperparameters --- #
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparameter for Adam optimizers')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')

    # --- Logging and Saving --- #
    parser.add_argument('--vis-batch-size', type=int, default=64, help='Batch size for generating visualization images')
    parser.add_argument('--save-interval', type=int, default=500, help='Save generated image samples every N iterations')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Save model checkpoints every N epochs')
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
