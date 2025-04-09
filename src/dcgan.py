import torch
import torch.nn as nn
import torch.nn.functional as F # Import functional for potential resizing if needed

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
        """
        Generator network for DCGAN for 224x224 images.

        Args:
            latent_dim (int): Size of the latent z vector.
            num_channels (int): Number of channels in the output image (e.g., 3 for RGB).
            feature_maps_g (int): Base size of feature maps in the generator.
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            # Output size = (Input size - 1) * stride + kernel_size - 2 * padding
            nn.ConvTranspose2d(latent_dim, feature_maps_g * 8, 7, 1, 0, bias=False), # Output: (f_g*8) x 7 x 7
            nn.BatchNorm2d(feature_maps_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps_g * 8, feature_maps_g * 4, 4, 2, 1, bias=False), # Output: (f_g*4) x 14 x 14
            nn.BatchNorm2d(feature_maps_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps_g * 4, feature_maps_g * 2, 4, 2, 1, bias=False), # Output: (f_g*2) x 28 x 28
            nn.BatchNorm2d(feature_maps_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps_g * 2, feature_maps_g, 4, 2, 1, bias=False), # Output: f_g x 56 x 56
            nn.BatchNorm2d(feature_maps_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps_g, feature_maps_g // 2, 4, 2, 1, bias=False), # Output: (f_g//2) x 112 x 112
            nn.BatchNorm2d(feature_maps_g // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps_g // 2, num_channels, 4, 2, 1, bias=False), # Output: num_channels x 224 x 224
            nn.Tanh() # Output: num_channels x 224 x 224 (values between -1 and 1)
        )
        # Apply the weights initialization
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, num_channels, feature_maps_d):
        """
        Discriminator network for DCGAN for 224x224 images.

        Args:
            num_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            feature_maps_d (int): Base size of feature maps in the discriminator.
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: num_channels x 224 x 224
            # Output size = floor((Input size + 2 * padding - kernel_size) / stride) + 1
            nn.Conv2d(num_channels, feature_maps_d // 2, 4, 2, 1, bias=False), # Output: (f_d//2) x 112 x 112
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps_d // 2, feature_maps_d, 4, 2, 1, bias=False), # Output: f_d x 56 x 56
            nn.BatchNorm2d(feature_maps_d),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps_d, feature_maps_d * 2, 4, 2, 1, bias=False), # Output: (f_d*2) x 28 x 28
            nn.BatchNorm2d(feature_maps_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps_d * 2, feature_maps_d * 4, 4, 2, 1, bias=False), # Output: (f_d*4) x 14 x 14
            nn.BatchNorm2d(feature_maps_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps_d * 4, feature_maps_d * 8, 4, 2, 1, bias=False), # Output: (f_d*8) x 7 x 7
            nn.BatchNorm2d(feature_maps_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps_d * 8, 1, 7, 1, 0, bias=False), # Output: 1 x 1 x 1
            nn.Sigmoid() # Output: scalar probability
        )
        # Apply the weights initialization
        self.apply(weights_init)

    def forward(self, input):
        # Input validation - check if image size is 224x224
        # if input.shape[2] != 224 or input.shape[3] != 224:
        #     print(f"Warning: Discriminator received input of size {input.shape[2:]}, expected 224x224.")
            # Optionally resize here if needed, though generator should provide correct size
            # input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)

        return self.main(input).view(-1, 1).squeeze(1) # Flatten output

if __name__ == '__main__':
    # Example Usage (for testing the architecture)
    latent_size = 100
    num_chan = 3
    f_g = 64 # Generator feature maps base size
    f_d = 64 # Discriminator feature maps base size
    target_size = 224

    # Create the generator
    netG = Generator(latent_size, num_chan, f_g)
    print("Generator Architecture:")
    # print(netG) # Can be verbose

    # Create a dummy noise vector
    noise = torch.randn(4, latent_size, 1, 1) # Batch size 4

    # Pass noise through generator
    fake_image = netG(noise)
    print(f"Generated image batch shape: {fake_image.shape}") # Should be [4, 3, 224, 224]
    assert fake_image.shape[1] == num_chan and fake_image.shape[2] == target_size and fake_image.shape[3] == target_size, "Generator output shape is incorrect!"

    # Create the discriminator
    netD = Discriminator(num_chan, f_d)
    print("Discriminator Architecture:")
    # print(netD) # Can be verbose

    # Pass fake image through discriminator
    outputD = netD(fake_image)
    print(f"Discriminator output shape: {outputD.shape}") # Should be [4]
    assert outputD.shape[0] == fake_image.shape[0], "Discriminator output shape is incorrect!"
    print(f"Discriminator output values (should be between 0 and 1): {outputD.detach().numpy()}")

    print(f"DCGAN models defined successfully for {target_size}x{target_size} images.")
