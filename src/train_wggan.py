import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from wggan import Generator, Discriminator, gradient_penalty, weights_init

def plot_gan_losses(history, out_path):
    plt.figure(figsize=(12,6))
    plt.plot(history['D_losses'], label='Critic (D) Loss')
    plt.plot(history['G_losses'], label='Generator Loss')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    model_dir = os.path.join(args.model_dir, 'wgan')
    os.makedirs(model_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, 'wgan_images')
    os.makedirs(image_dir, exist_ok=True)
    metrics_dir = args.results_dir
    os.makedirs(metrics_dir, exist_ok=True)
    figures_dir = args.figures_dir
    os.makedirs(figures_dir, exist_ok=True)

    train_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    netG = Generator(args.latent_dim, args.num_channels, args.feature_maps_g).to(device)
    netD = Discriminator(args.num_channels, args.feature_maps_d).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.9))

    fixed_noise = torch.randn(args.vis_batch_size, args.latent_dim, device=device).unsqueeze(2).unsqueeze(3)
    history = {'D_losses': [], 'G_losses': [], 'D_losses_epoch': [], 'G_losses_epoch': []}
    iters = 0

    for epoch in range(args.epochs):
       
        d_epoch_losses = []
        g_epoch_losses = []
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for i, data in progress_bar:
            real_images = data[0].to(device)
            b_size = real_images.size(0)

            for _ in range(args.critic_iters):
                netD.zero_grad()

                d_real = netD(real_images)
                d_real_loss = -d_real.mean()

                noise = torch.randn(b_size, args.latent_dim, device=device).unsqueeze(2).unsqueeze(3)
                fake_images = netG(noise)
                d_fake = netD(fake_images.detach())
                d_fake_loss = d_fake.mean()
                gp = gradient_penalty(netD, real_images.data, fake_images.data, device, lambda_gp=args.lambda_gp)
                d_loss = d_real_loss + d_fake_loss + gp
                d_loss.backward()
                optimizerD.step()
                history['D_losses'].append(d_loss.item())
                d_epoch_losses.append(d_loss.item())

            netG.zero_grad()
            noise = torch.randn(b_size, args.latent_dim, device=device).unsqueeze(2).unsqueeze(3)
            fake_images = netG(noise)
            g_loss = -netD(fake_images).mean()
            g_loss.backward()
            optimizerG.step()
            history['G_losses'].append(g_loss.item())
            g_epoch_losses.append(g_loss.item())

            progress_bar.set_postfix({
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item()
            })

            if (iters % args.save_interval == 0) or (epoch == args.epochs-1 and i == len(train_loader)-1):
                with torch.no_grad():
                    fake_vis = netG(fixed_noise).detach().cpu()
                vutils.save_image(fake_vis, f"{image_dir}/fake_samples_epoch_{epoch+1:03d}_iter_{iters:06d}.png", normalize=True, nrow=8)
            
            iters += 1

        avg_d_loss = np.mean(d_epoch_losses)
        avg_g_loss = np.mean(g_epoch_losses)
        history['D_losses_epoch'].append(avg_d_loss)
        history['G_losses_epoch'].append(avg_g_loss)
        print(f"Epoch {epoch+1}/{args.epochs} Summary -  Avg Loss_D: {avg_d_loss:.4f}, Avg Loss_G: {avg_g_loss:.4f}")

        if (epoch + 1) % args.checkpoint_interval == 0 or (epoch+1) == args.epochs:
            torch.save(netG.state_dict(), os.path.join(model_dir, f'generator_epoch_{epoch+1:03d}.pth'))
            torch.save(netD.state_dict(), os.path.join(model_dir, f'discriminator_epoch_{epoch+1:03d}.pth'))

    torch.save(netG.state_dict(), os.path.join(model_dir, 'generator_final.pth'))
    torch.save(netD.state_dict(), os.path.join(model_dir, 'discriminator_final.pth'))
    print("Saved final models.")

    with open(os.path.join(metrics_dir, 'wgan_training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    plot_gan_losses(history, os.path.join(figures_dir, 'wgan_loss_curve.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wasserstein GAN-GP on RSNA Pneumonia images")
    parser.add_argument('--data-dir', type=str, default='./data/processed')
    parser.add_argument('--model-dir', type=str, default='./models')
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--results-dir', type=str, default='./results/metrics')
    parser.add_argument('--figures-dir', type=str, default='./results/figures')
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--feature-maps-g', type=int, default=64)
    parser.add_argument('--feature-maps-d', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--vis-batch-size', type=int, default=64)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--critic-iters', type=int, default=5,
                        help='Number of D updates per G update')
    parser.add_argument('--lambda-gp', type=float, default=10.,
                        help='Gradient penalty coefficient')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    print("--- Args ---")
    for k,v in vars(args).items(): print(f"  {k}: {v}")

    main(args)
