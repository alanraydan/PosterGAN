# Main training loop for PosterGAN
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from networks import Generator, Discriminator
from trainer import Trainer
from data_loader import PosterDataset

if __name__ == '__main__':
    # Taken from WGAN gradient penalty paper
    lr = 0.0001
    betas = (0.0, 0.9)
    lambda_gp = 10
    g_update_freq = 5

    batch_size = 64
    epochs = 5
    latent_dim = 100
    class_embedding_dim = 16
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    dataset = PosterDataset()
    train_size = int(1.0 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    G = Generator(latent_dim=latent_dim, n_classes=dataset.n_genres, class_embedding_dim=class_embedding_dim)
    G.to(device)
    D = Discriminator(n_classes=dataset.n_genres, class_embedding_dim=class_embedding_dim)
    D.to(device)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    writer = SummaryWriter()

    trainer = Trainer(G, D, optimizer_G, optimizer_D,
                    gp_weight=lambda_gp, update_freq_G=g_update_freq,
                    device=device, writer=writer)

    trainer.train(epochs, dataloader, save_networks=True)
    writer.close()
