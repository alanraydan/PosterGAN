import torch
from tqdm import tqdm, trange

class Trainer:
    """
    Trainer class for conditional WGAN.

    Model info_dict should contain keys:
        - epoch
        - global_step
        - model_state_dict
        - optimizer_state_dict
    """
    def __init__(self, G, D, optimizer_G, optimizer_D,
                 gp_weight=10.0, update_freq_G=5, device='cpu',
                 writer=None, global_step=0):
        self.G = G
        self.D = D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.gp_weight = gp_weight
        self.update_freq_G = update_freq_G
        self.device = device
        self.writer = writer
        self.latent_dim = G.latent_dim
        self.global_step = global_step

        self.G.to(self.device)
        self.D.to(self.device)

        self.G.train()
        self.D.train()

    def _update_D(self, real_image, genre_multihot):

        batch_size = real_image.shape[0]

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_image = self.G(z, genre_multihot)

        # Train Descriminator using gradient penalty
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolant = epsilon * real_image + (1 - epsilon) * fake_image
        interpolant.requires_grad_(True)
        interpolant_score = self.D(interpolant, genre_multihot)
        gradients = torch.autograd.grad(outputs=interpolant_score, inputs=interpolant,
                                        grad_outputs=torch.ones_like(interpolant_score).to(self.device),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        # Add a small number to avoid problems with drivative of sqrt at 0
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()

        # Compute Wasserstein loss
        real_image_score = self.D(real_image, genre_multihot)
        fake_image_score = self.D(fake_image, genre_multihot)
        wasserstein_loss = -torch.mean(real_image_score) + torch.mean(fake_image_score)

        D_loss = wasserstein_loss + self.gp_weight * gradient_penalty
        self.optimizer_D.zero_grad()
        D_loss.backward()
        self.optimizer_D.step()

        if self.writer is not None:
            self.writer.add_scalar('D_gp_loss', gradient_penalty.item(), self.global_step)
            self.writer.add_scalars('D_scores', {'real': real_image_score.mean().item(),
                                            'fake': fake_image_score.mean().item()}, self.global_step)

    def _update_G(self, genre_multihot):

        batch_size = genre_multihot.shape[0]

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_image = self.G(z, genre_multihot)
        fake_image_score = self.D(fake_image, genre_multihot)

        G_loss = -torch.mean(fake_image_score)
        self.optimizer_G.zero_grad()
        G_loss.backward()
        self.optimizer_G.step()

        if self.writer is not None:
            self.writer.add_scalar('G_loss', G_loss.item(), self.global_step)

    def train(self, n_epochs, dataloader, save_networks=False):

        print(f'Using device: {self.device}')

        for epoch in trange(n_epochs, desc='Epoch'):

            for i, (real_image, genre_multihot) in enumerate(tqdm(dataloader, desc='Batch', leave=False)):

                real_image = real_image.to(self.device)
                genre_multihot = genre_multihot.to(self.device)

                self._update_D(real_image, genre_multihot)

                if i % self.update_freq_G == 0:
                    self._update_G(genre_multihot)

                self.global_step += 1

        if save_networks:
            G_info_dict = {'epoch': epoch, 'global_step': self.global_step,
                           'model_state_dict': self.G.state_dict(),
                           'optimizer_state_dict': self.optimizer_G.state_dict()}
            D_info_dict = {'epoch': epoch, 'global_step': self.global_step,
                           'model_state_dict': self.D.state_dict(),
                            'optimizer_state_dict': self.optimizer_D.state_dict()}
            torch.save(G_info_dict, 'G.pt')
            torch.save(D_info_dict, 'D.pt')
