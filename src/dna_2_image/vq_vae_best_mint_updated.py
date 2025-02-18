from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np


from six.moves import xrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device shape is {device}')
## Load the data
from torchvision import datasets, transforms
import wandb


"""## Load Data"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Define transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 256x256
    transforms.ToTensor(),  # Convert to tensor [0,1]
])

# Load MNIST datasets
training_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

validation_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
print(validation_data[0][0].shape)
exit()
# Function to filter dataset
def filter_dataset(dataset, allowed_labels):
    indices = [i for i, (img, label) in enumerate(dataset) if label in allowed_labels]
    return Subset(dataset, indices)
allowed_labels = {0, 1, 2, 3}
training_data_filter = filter_dataset(training_data, allowed_labels)
validation_data_filter = filter_dataset(validation_data, allowed_labels)

# Create data loaders
training_loader = DataLoader(
    training_data_filter,
    batch_size=256,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    validation_data_filter,
    batch_size=256,
    shuffle=False,
    num_workers=4
)
std = 1

# from dna_dataset import training_loader, val_loader, std
# training_data = training_loader

# data_variance = std

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

"""We will also implement a slightly modified version  which will use exponential moving averages to update the embedding vectors instead of an auxillary loss. This has the advantage that the embedding updates are independent of the choice of optimizer for the encoder, decoder and other parts of the architecture. For most experiments the EMA version trains faster than the non-EMA version."""

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        print(f"input shaps is {inputs.shape}")
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        print(f"flat_input shape is {flat_input.shape}")
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        print(f"shape of encoding_indices is {encoding_indices.shape}")
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        print(f"shape of encodings is {encodings.shape}")
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        print(f"torch.matmul(encodings, self._embedding.weight) is {torch.matmul(encodings, self._embedding.weight).shape}")
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

"""## Encoder & Decoder Architecture

The encoder and decoder architecture is based on a ResNet and is implemented below:
"""

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(False),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens//2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                out_channels=1,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()

        self._encoder = Encoder(1, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        print(f"shape of encoded output z is {z.shape}")
        z = self._pre_vq_conv(z)
        print(f"shape after prev-vq-conv output z is {z.shape}")
        loss, quantized, perplexity, encoding_indices = self._vq_vae(z)
        # print(f"shape of quantized vector is {quantized.shape}")
        # print(f"shape of quantized encoding_indices is {encoding_indices.shape}")
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity, encoding_indices


import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

class VQVAE(pl.LightningModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0,
                 learning_rate=1e-3, data_variance=1.0):
        super().__init__()
        self.save_hyperparameters()

        # Create the VQ-VAE model
        self.model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                          num_embeddings, embedding_dim, commitment_cost, decay)

        self.data_variance = data_variance
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data = batch[0]
        vq_loss, data_recon, perplexity, encodings = self.model(data)

        # Original reconstruction error (MSE)
        recon_error = F.mse_loss(data_recon, data) / self.data_variance

        # Add L1 loss for sharper reconstructions
        # l1_loss = F.l1_loss(data_recon, data)

        # Calculate KL divergence to encourage diverse codebook usage
        # Get average usage of each codebook entry
        # avg_probs = torch.mean(encodings, dim=0)
        # KL divergence with uniform distribution
        # kl_loss = torch.sum(avg_probs * torch.log(avg_probs + 1e-10))

        # Combine all losses with weighting factors
        total_loss = recon_error + vq_loss # + 0.1 * l1_loss + 0.1 * kl_loss

        # Log all metrics
        codebook_usage = (encodings.sum(0) > 0).float().mean()
        self.log('train/codebook_usage', codebook_usage)
        self.log('train/reconstruction_error', recon_error, on_step=True, on_epoch=True)
        self.log('train/vq_loss', vq_loss, on_step=True, on_epoch=True)
        self.log('train/perplexity', perplexity, on_step=True, on_epoch=True)
        # self.log('train/l1_loss', l1_loss, on_step=True, on_epoch=True)
        # self.log('train/kl_loss', kl_loss, on_step=True, on_epoch=True)
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True)

        # Log reconstructed images periodically
        if batch_idx % 100 == 0:
            self._log_reconstructions(data, data_recon, "train")

        return total_loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        vq_loss, data_recon, perplexity, encodings = self.model(data)

        # Original reconstruction error (MSE)
        recon_error = F.mse_loss(data_recon, data) / self.data_variance

        # Add L1 loss
        # l1_loss = F.l1_loss(data_recon, data)

        # Calculate KL divergence
        # avg_probs = torch.mean(encodings, dim=0)
        # kl_loss = torch.sum(avg_probs * torch.log(avg_probs + 1e-10))

        # Combine all losses with same weights as training
        total_loss = recon_error + vq_loss # + 0.1 * l1_loss + 0.1 * kl_loss

        # Log all metrics
        self.log('val/reconstruction_error', recon_error)
        self.log('val/vq_loss', vq_loss)
        self.log('val/perplexity', perplexity)
        # self.log('val/l1_loss', l1_loss)
        # self.log('val/kl_loss', kl_loss)
        self.log('val/total_loss', total_loss)

        if batch_idx == 0:
            self._log_reconstructions(data, data_recon, "val")

    def _log_reconstructions(self, original, reconstruction, split):
        # Take first 8 images from batch
        original = original[:8].cpu()
        reconstruction = reconstruction[:8].cpu()

        # Create a grid of original and reconstructed images
        comparison = torch.cat([original, reconstruction])
        grid = make_grid(comparison, nrow=8, normalize=True, padding=2)

        # Log to wandb
        self.logger.experiment.log({
            f'{split}/reconstructions': wandb.Image(grid)
        })

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# Training configuration
config = {
    'batch_size': 128,
    'num_hiddens': 128,
    'num_residual_hiddens': 32,
    'num_residual_layers': 2,
    'embedding_dim': 64,
    'num_embeddings': 6,
    'commitment_cost': 0.25,
    'decay': 0.99,
    'learning_rate': 5e-4,
    'max_epochs': 100,
}


def main():
    # Initialize wandb
    wandb.init(
        project="vq-vae",
        config=config,
    )

    # Create the model
    model = VQVAE(
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        decay=config['decay'],
        learning_rate=config['learning_rate'],
        data_variance=std
    )

    # Create the wandb logger
    wandb_logger = WandbLogger(project="vq-vae")

    # Create the trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val/total_loss',
                dirpath='checkpoints-test',
                filename='vqvae-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                mode='min',
            ),
        ]
    )
    # Start training
    trainer.fit(model, train_dataloaders=training_loader, val_dataloaders=val_loader)

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
