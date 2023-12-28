# Implementation of generator and discriminator networks for PosterGAN.
# The architecture is that of a deep-convolutional-conditional-Wasserstein GAN... quite a mouthful.
# It's simply a DCGAN with an additional input of the class label and using a Wasserstein loss.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class Generator(nn.Module):
    """
    Generator network for conditional WGAN.
    
    Output image size for Poster Dataset: (batch_size, 3, 268, 182)
    """
    def __init__(self, latent_dim, n_classes, class_embedding_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.class_embedding_dim = class_embedding_dim

        self.class_embedding_layer = nn.Linear(self.n_classes, self.class_embedding_dim, bias=False) # No bias term to emulate nn.Embedding

        self.input_layer = nn.Linear(self.latent_dim + self.class_embedding_dim, 128*30*20)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(6, 5), stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(6, 5), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2),
            nn.Tanh()
        )
    
    def forward(self, z, class_multihot):
        assert z.shape == (z.shape[0], self.latent_dim), 'Incorrect latent variable shape.'
        assert class_multihot.shape[1] == self.n_classes, 'Incorrect class embedding shape.'
        class_embedding = self.class_embedding_layer(class_multihot)
        x = torch.cat((z, class_embedding), dim=1)
        x = self.input_layer(x)
        x = x.view(x.size(0), 128, 30, 20)  # Reshape
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        assert x.shape == (x.shape[0], 3, 268, 182), 'Incorrect output shape.'
        return x

    @torch.no_grad()
    def generate_poster(self, class_multihot, z=None):
        """
        Generate a movie poster given a class embedding.
        """
        assert class_multihot.shape[1] == self.n_classes, 'Incorrect class embedding shape.'
        if z is None:
            z = torch.randn(1, self.latent_dim)
        poster = self.forward(z, class_multihot)
        poster = (poster + 1) * 127.5
        poster = transforms.functional.to_pil_image(poster.squeeze(0).type(torch.uint8))
        return poster


class Discriminator(nn.Module):
    """
    Discriminator network for conditional WGAN.

    Input image size for Poster Dataset: (batch_size, 3, 268, 182)
    """
    def __init__(self, n_classes, class_embedding_dim):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.class_embedding_dim = class_embedding_dim

        self.class_embedding_layer = nn.Linear(self.n_classes, self.class_embedding_dim, bias=False) # No bias term to emulate nn.Embedding

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=6, stride=2),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(6, 5), stride=2),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(6, 5), stride=2),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.output_layer = nn.Linear(64*30*20 + self.class_embedding_dim, 1)
    
    def forward(self, x, class_multihot):
        assert x.shape == (x.shape[0], 3, 268, 182), 'Incorrect image shape.'
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        class_embedding = self.class_embedding_layer(class_multihot)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat((x, class_embedding), dim=1)
        x = self.output_layer(x)
        assert x.shape == (x.shape[0], 1), 'Incorrect output shape.'
        return x
