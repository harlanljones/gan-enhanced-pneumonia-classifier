import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and m.weight is not None:
        if 'Conv' in classname or 'Linear' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in classname or 'InstanceNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.)

class Generator(nn.Module):
    def __init__(self, latent_dim, num_channels, feature_maps_g):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps_g * 16, 7, 1, 0, bias=False), # 7x7
            nn.BatchNorm2d(feature_maps_g * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps_g * 16, feature_maps_g * 8, 4, 2, 1, bias=False), # 14x14
            nn.BatchNorm2d(feature_maps_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps_g * 8, feature_maps_g * 4, 4, 2, 1, bias=False), # 28x28
            nn.BatchNorm2d(feature_maps_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps_g * 4, feature_maps_g * 2, 4, 2, 1, bias=False), # 56x56
            nn.BatchNorm2d(feature_maps_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps_g * 2, feature_maps_g, 4, 2, 1, bias=False), # 112x112
            nn.BatchNorm2d(feature_maps_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps_g, num_channels, 4, 2, 1, bias=False), # 224x224
            nn.Tanh()
        )
        self.apply(weights_init)
        
    def forward(self, z):

        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, num_channels, feature_maps_d):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, feature_maps_d, 4, 2, 1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps_d, feature_maps_d * 2, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(feature_maps_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps_d * 2, feature_maps_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps_d * 4, feature_maps_d * 8, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(feature_maps_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps_d * 8, 1, 7, 1, 0, bias=False),  
        )
        self.apply(weights_init)

    def forward(self, x):
        out = self.main(x)       
        out = out.mean([2, 3])  
        return out.view(-1)     

def gradient_penalty(D, real_samples, fake_samples, device, lambda_gp=10.):

    batch_size = real_samples.size(0)

    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp
