# Main training loop for PosterGAN
from os import write
import matplotlib.pyplot as plt
from sympy import beta
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from networks import Generator, Discriminator
from data_loader import PosterDataset

batch_size = 64
epochs = 5
lr = 0.0002
betas = (0.5, 0.999)
lambda_gp = 10
latent_dim = 100
n_classes = 28
normalize = lambda x: x / 127.5 - 1
g_update_freq = 5
genre_csv = 'MovieGenre_cleaned.csv'
poster_dir = 'posters'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

dataset = PosterDataset(genre_csv=genre_csv, poster_dir=poster_dir, transform=normalize)
train_size = int(1.0 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

G = Generator(latent_dim=latent_dim, n_classes=n_classes)
G.to(device)
D = Discriminator(n_classes=n_classes)
D.to(device)
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

writer = SummaryWriter()

D_losses = []
G_losses = []
for epoch in range(epochs):
    for i, (real_poster, genre_embedding) in enumerate(tqdm(dataloader)):
        real_poster = real_poster.to(device)
        genre_embedding = genre_embedding.to(device)
        batch_size = real_poster.shape[0]
        
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_poster = G(z, genre_embedding)

        # Train Descriminator using gradient penalty and Wasserstein loss
        epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolates = epsilon * real_poster + (1 - epsilon) * fake_poster
        interpolates.requires_grad_(True)
        disc_interpolates = D(interpolates, genre_embedding)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True)[0]
        # Add a small number to avoid problems with drivative of sqrt at 0
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        D_loss = -torch.mean(D(real_poster, genre_embedding)) \
            + torch.mean(D(fake_poster, genre_embedding)) \
            + lambda_gp * gradient_penalty
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()
        writer.add_scalar('D_loss', D_loss.item(), epoch * len(dataloader) + i)

        # Train Generator
        if i % g_update_freq == 0:
            z = torch.randn(batch_size, latent_dim, device=device)
            optimizer_G.zero_grad()
            fake_poster = G(z, genre_embedding)
            fake_out = D(fake_poster, genre_embedding)
            G_loss = -torch.mean(fake_out)
            G_loss.backward()
            optimizer_G.step()
            writer.add_scalar('G_loss', G_loss.item(), epoch * len(dataloader) + i)

            # Save losses
            # D_losses.append(D_loss.item())
            # G_losses.append(G_loss.item())

# Save Generator and Discriminator
torch.save(G.state_dict(), 'G.pt')
torch.save(D.state_dict(), 'D.pt')
writer.close()

# plt.plot(D_losses, label='Discriminator Loss')
# plt.plot(G_losses, label='Generator Loss')
# plt.legend()
# plt.show()

