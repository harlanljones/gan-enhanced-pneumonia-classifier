import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, num_channels, feature_maps_g):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_size = 7
        self.num_classes = num_classes
        nf = feature_maps_g

        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.fc = nn.Linear(latent_dim, nf * 8 * self.init_size * self.init_size)
        self.main = nn.Sequential(
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 8, nf * 4, 3, 1, 1),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 2, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf, nf // 2, 3, 1, 1),
            nn.BatchNorm2d(nf // 2),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf // 2, num_channels, 3, 1, 1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z, labels, alpha=1.0):
        cond = self.label_emb(labels)
        x = z + cond
        out = self.fc(x)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        out = self.main(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, num_classes, num_channels, feature_maps_d):
        super().__init__()
        nf = feature_maps_d
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, nf*8*7*7)

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, nf // 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf // 2, nf, 4, 2, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf * 8, 1, 7, 1, 0),

        )
        self.apply(weights_init)

    def forward(self, x, labels, alpha=1.0):
        features = []
        for i, layer in enumerate(self.main[:-2]):
            x = layer(x)
            features.append(x)

        x = self.main[-2](x)
        features.append(x)

        proj = (self.label_emb(labels) * x.view(x.size(0), -1)).sum(dim=1, keepdim=True)

        out = self.main[-1](x).view(-1, 1)  
        return (out + proj).squeeze(1)     

    def get_intermediate_features(self, x, labels, alpha=1.0):
        features = []
        for i, layer in enumerate(self.main[:-1]):
            x = layer(x)
            features.append(x)
        return features

ProgressiveGenerator = Generator
ProgressiveDiscriminator = Discriminator

if __name__ == '__main__':
    latent_size = 100
    num_classes = 2
    num_chan = 3
    f_g = 32
    f_d = 32
    target_size = 224
    batch_size = 4
    netG = Generator(latent_size, num_classes, num_chan, f_g)
    netD = Discriminator(num_classes, num_chan, f_d)
    alphas = [0.0, 0.5, 1.0]
    for alpha in alphas:
        labels = torch.randint(0, num_classes, (batch_size,))
        noise = torch.randn(batch_size, latent_size)
        fake_images = netG(noise, labels, alpha)
        print(f"Generated image shape: {fake_images.shape}")
        disc_output = netD(fake_images, labels, alpha)
        print(f"Discriminator output shape: {disc_output.shape}")
    print(f"\nConditional DCGAN models defined successfully for {target_size}x{target_size} images.")
