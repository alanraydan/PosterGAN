# Main training loop for PosterGAN
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from networks import Generator, Discriminator
from trainer import Trainer
from data_loader import PosterDataset

if __name__ == '__main__':
    start_from_checkpoint = False
    checkpoint_path = '/Users/alanraydan/Development/PosterGAN/runs/Dec28_23-09-05_Alans-MacBook-Pro-2.local'

    # Taken from WGAN gradient penalty paper
    lr = 0.0001
    betas = (0.0, 0.9)
    lambda_gp = 20
    g_update_freq = 5

    batch_size = 64
    epochs = 60
    latent_dim = 100
    class_embedding_dim = 16
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    global_step = 0

    dataset = PosterDataset()
    train_size = int(1.0 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    G = Generator(latent_dim=latent_dim, n_classes=dataset.n_genres, class_embedding_dim=class_embedding_dim)
    D = Discriminator(n_classes=dataset.n_genres, class_embedding_dim=class_embedding_dim)
    G = G.to(device)
    D = D.to(device)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    if start_from_checkpoint:
        G_checkpoint = torch.load(f'{checkpoint_path}/G.pt')
        D_checkpoint = torch.load(f'{checkpoint_path}/D.pt')
        G.load_state_dict(G_checkpoint['model_state_dict'])
        D.load_state_dict(D_checkpoint['model_state_dict'])
        optimizer_G.load_state_dict(G_checkpoint['optimizer_state_dict'])
        optimizer_D.load_state_dict(D_checkpoint['optimizer_state_dict'])
        global_step = G_checkpoint['global_step']

    writer = SummaryWriter()

    trainer = Trainer(G, D, optimizer_G, optimizer_D,
                    gp_weight=lambda_gp, update_freq_G=g_update_freq,
                    device=device, writer=writer, global_step=global_step)

    trainer.train(epochs, dataloader, save_networks=True)
    writer.close()
