import argparse
import os
import torch
import torchvision.utils as vutils
from torchvision import transforms
import numpy as np
import sys

# Add src directory to path to import dcgan and utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dcgan import Generator # Assuming Generator class is in dcgan.py
    from utils import check_create_dir # Assuming check_create_dir is in utils.py
except ImportError as e:
    print(f"Error importing modules. Make sure dcgan.py and utils.py are in the src directory: {e}")
    sys.exit(1)

def generate_images(generator_path, output_dir, num_images, latent_dim, feature_maps_g, batch_size, device):
    """Generates synthetic images using a trained generator."""
    check_create_dir(output_dir)

    # Load the generator model
    netG = Generator(latent_dim, 3, feature_maps_g).to(device) # 3 channels for RGB
    try:
        netG.load_state_dict(torch.load(generator_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Generator model not found at {generator_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading generator state dict: {e}")
        print("Ensure the Generator class definition matches the saved model.")
        sys.exit(1)
        
    netG.eval() # Set generator to evaluation mode

    print(f"Generating {num_images} synthetic images...")
    generated_count = 0
    with torch.no_grad():
        while generated_count < num_images:
            current_batch_size = min(batch_size, num_images - generated_count)
            if current_batch_size <= 0:
                break

            # Create batch of latent vectors
            noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            
            # Generate fake image batch with G
            fake_images = netG(noise)

            # Save images
            for i in range(fake_images.size(0)):
                if generated_count < num_images:
                    # Un-normalize from tanh output [-1, 1] to [0, 1] before saving
                    img = (fake_images[i] * 0.5) + 0.5 
                    vutils.save_image(img, os.path.join(output_dir, f'synthetic_{generated_count+1:05d}.png'), normalize=False)
                    generated_count += 1
                else:
                    break
            
            print(f"Generated {generated_count}/{num_images} images...")

    print(f"Finished generating {generated_count} images in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic images using a trained DCGAN generator.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained generator checkpoint (e.g., models/gan/generator_final.pth)")
    parser.add_argument('--output-dir', type=str, default='./data/synthetic', help="Directory to save generated images.")
    parser.add_argument('--num-images', type=int, default=5000, help="Number of synthetic images to generate.")
    parser.add_argument('--latent-dim', type=int, default=100, help="Size of the latent z vector (must match training).")
    parser.add_argument('--feature-maps-g', type=int, default=64, help="Generator base feature maps (must match training).")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for generation.")
    parser.add_argument('--cpu', action='store_true', help="Force CPU usage even if CUDA is available.")

    args = parser.parse_args()

    # Setup device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generate_images(
        generator_path=args.model_path,
        output_dir=args.output_dir,
        num_images=args.num_images,
        latent_dim=args.latent_dim,
        feature_maps_g=args.feature_maps_g,
        batch_size=args.batch_size,
        device=device
    ) 