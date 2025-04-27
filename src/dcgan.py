import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Custom weights initialization called on generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, latent_dim, num_channels, feature_maps_g):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.init_size = 7
        nf = feature_maps_g
        
        # Initial processing
        self.fc = nn.Linear(latent_dim, nf * 8 * self.init_size * self.init_size)
        
        # Main generation blocks
        self.main = nn.Sequential(
            # 7x7 -> 14x14
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 8, nf * 4, 3, 1, 1),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            
            # 14x14 -> 28x28
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            
            # 28x28 -> 56x56
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 2, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            
            # 56x56 -> 112x112
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf, nf // 2, 3, 1, 1),
            nn.BatchNorm2d(nf // 2),
            nn.ReLU(True),
            
            # 112x112 -> 224x224
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf // 2, num_channels, 3, 1, 1),
            nn.Tanh()
        )
        
        self.apply(weights_init)
    
    def forward(self, z, alpha=1.0):
        # Initial processing
        out = self.fc(z)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        
        # Process through main blocks
        out = self.main(out)
        
        return out

class Discriminator(nn.Module):
    def __init__(self, num_channels, feature_maps_d):
        super(Discriminator, self).__init__()
        nf = feature_maps_d
        
        # Main discriminator blocks
        self.main = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(num_channels, nf // 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 56x56
            nn.Conv2d(nf // 2, nf, 4, 2, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 28x28
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 14x14
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final classification
            nn.Conv2d(nf * 8, 1, 7, 1, 0),
            nn.Sigmoid()
        )
        
        self.apply(weights_init)
    
    def forward(self, x, alpha=1.0):
        features = []
        
        # Process through main blocks
        for i, layer in enumerate(self.main):
            x = layer(x)
            features.append(x)
        
        return x.view(-1, 1).squeeze(1)
    
    def get_intermediate_features(self, x, alpha=1.0):
        """Get intermediate feature maps for feature matching loss."""
        features = []
        
        for i, layer in enumerate(self.main):
            x = layer(x)
            features.append(x)
        
        return features[:-1]  # Exclude final output

# For backward compatibility
ProgressiveGenerator = Generator
ProgressiveDiscriminator = Discriminator

if __name__ == '__main__':
    # Example Usage (for testing the architecture)
    latent_size = 100
    num_chan = 3
    f_g = 32  # Generator feature maps base size
    f_d = 32  # Discriminator feature maps base size
    target_size = 224
    batch_size = 4

    # Test models
    netG = Generator(latent_size, num_chan, f_g)
    netD = Discriminator(num_chan, f_d)

    # Test with different alpha values
    alphas = [0.0, 0.5, 1.0]
    for alpha in alphas:
        print(f"\nTesting with alpha={alpha}")
        
        # Generate fake images
        noise = torch.randn(batch_size, latent_size)
        fake_images = netG(noise, alpha)
        print(f"Generated image shape: {fake_images.shape}")
        
        # Test discriminator
        disc_output = netD(fake_images, alpha)
        print(f"Discriminator output shape: {disc_output.shape}")
        print(f"Discriminator output values: {disc_output.detach().numpy()}")

    print(f"\nEnhanced DCGAN models defined successfully for {target_size}x{target_size} images.")
