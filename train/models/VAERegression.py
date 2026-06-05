import torch
from torch import nn
from torch.nn import functional as F
from models.DeepMLP import ResidualBlock

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class CVAERegression(nn.Module):
    def __init__(self, input_channel, latent_dim):
        super(CVAERegression, self).__init__()
        self.latent_dim = latent_dim
        # Translated comment
        self.value_embedding = nn.Linear(1, 1024)
        
        # Translated comment
        self.condition_network = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*8*8, 1024),
            nn.ReLU()
            # nn.Flatten(),
            # nn.Linear(64*16*16, 1024),
            # nn.ReLU()
        )

        self.condition_embedding = nn.Sequential(
            nn.Linear(1024, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU()
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(1024 + 1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.decoder_outlayer = nn.Sequential(
            nn.Linear(64, 2)
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x, condition_img):
        x_embedded = self.value_embedding(x)
        x_emb64 = self.condition_embedding(x_embedded)
        condition = self.condition_network(condition_img)
        
        x = torch.cat([x_embedded, condition], dim=1)
        mu, log_var = torch.chunk(self.encoder(x), 2, dim=1)
        z = self.reparameterize(mu, log_var)
        
        z = torch.cat([z, condition], dim=1)
        z_emb = self.decoder(z)
        output = self.decoder_outlayer(z_emb)
        pred_mu = output[:, 0]
        pred_sigma = torch.exp(output[:, 1])
        return pred_mu, mu, log_var, x_emb64, z_emb, pred_sigma

    def predict(self, z, condition_img):
        # x_embedded = self.value_embedding(torch.randn(1))
        condition = self.condition_network(condition_img)
        
        # x = torch.cat([x_embedded, condition], dim=1)
        # mu, log_var = torch.chunk(self.encoder(x), 2, dim=1)
        # z = self.reparameterize(mu, log_var)
        z = torch.cat([z, condition], dim=1)
        return self.decoder(z)

class VAERegression(nn.Module):
    def __init__(self, original_dim, latent_dim):
        super(VAERegression, self).__init__()
        self.original_dim = original_dim
        self.intermediate_dim = 512
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_intermediate = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(original_dim, 4096),
            nn.Tanh(),
            nn.Linear(4096, self.intermediate_dim),
            # nn.Tanh()
        )
        self.z_mean = nn.Linear(self.intermediate_dim, latent_dim)
        self.z_log_var = nn.Linear(self.intermediate_dim, latent_dim)
        self.r_mean = nn.Linear(self.intermediate_dim, 1)
        self.r_log_var = nn.Linear(self.intermediate_dim, 1)

        self.fc_p = nn.Linear(1, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.intermediate_dim),
            nn.Tanh(),
            nn.Linear(self.intermediate_dim, 4096),
            # nn.Tanh(),
            nn.Linear(4096, original_dim)
        )

    def encode(self, x):
        h = self.encoder_intermediate(x)
        return self.z_mean(h), self.z_log_var(h), self.r_mean(h), self.r_log_var(h)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_mean, z_log_var, r_mean, r_log_var = self.encode(x.view(-1, self.original_dim))
        z = self.reparameterize(z_mean, z_log_var)
        r = self.reparameterize(r_mean, r_log_var)
        return self.decode(z), z, z_mean, z_log_var, r, r_mean, r_log_var, self.fc_p(r)

# class VAERegression(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super(VAERegression, self).__init__()
#         self.input_dim = input_dim
#         # Encoder
#         self.fc1 = nn.Linear(input_dim, 4096)
#         self.bn1 = nn.BatchNorm1d(4096)
#         self.dropout = nn.Dropout(0.5)
        
#         self.resblock1 = ResidualBlock(4096, 512)

#         self.fc21 = nn.Linear(512, latent_dim)  # Mean output
#         self.fc22 = nn.Linear(512, latent_dim)  # Log variance output
#         # Decoder
#         self.fc3 = nn.Linear(latent_dim, 512)
#         self.resblock2 = ResidualBlock(512, 4096)
#         self.fc4 = nn.Linear(4096, input_dim)
        
#     def encode(self, x):
#         x = F.tanh(self.bn1(self.fc1(x)))
#         x = self.dropout(x)
        
#         h1 = self.resblock1(x)
#         # h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)
    
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std
    
#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         h3 = self.resblock2(h3)
#         return torch.sigmoid(self.fc4(h3))
    
#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, self.input_dim))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

class ConvVAERegression(nn.Module):
    def __init__(self, input_channels=1, latent_dim=2, r_latent_dim = 1, intermediate_dim = 2048):
        super(ConvVAERegression, self).__init__()
        self.intermediate_dim = intermediate_dim
        # Encoder
        self.encoder_intermediate = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Flatten(),
            # nn.Linear(256*8*8, intermediate_dim),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*16*16, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.z_mean = nn.Linear(self.intermediate_dim, latent_dim)
        self.z_log_var = nn.Linear(self.intermediate_dim, latent_dim)
        self.regressor = nn.Sequential(
            nn.Linear(self.intermediate_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )
        self.r_mean = nn.Linear(64, r_latent_dim)
        self.r_log_var = nn.Linear(64, r_latent_dim)
        
        self.fc_p = nn.Linear(r_latent_dim, latent_dim)
        self.fc_p = nn.Sequential(
            nn.Linear(r_latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, 128*16*16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),
            # nn.Linear(1024, 256*8*8),
            # nn.ReLU(),
            # nn.Unflatten(1, (256, 8, 8)),
            # nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # Output: 128x128
        )
        
    def encode(self, x):
        h = self.encoder_intermediate(x)
        r_emb = self.regressor(h)
        return self.z_mean(h), self.z_log_var(h), self.r_mean(r_emb), self.r_log_var(r_emb)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_mean, z_log_var, r_mean, r_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        r = self.reparameterize(r_mean, r_log_var)
        r_recon = self.fc_p(r)
        return self.decode(torch.cat([z, r_recon], dim=1)), z, z_mean, z_log_var, r, r_mean, r_log_var, r_recon


class SimpleConvVAERegression(nn.Module):
    def __init__(self, input_channels=1, latent_dim=2, r_latent_dim = 1, intermediate_dim = 2048):
        super(SimpleConvVAERegression, self).__init__()
        self.intermediate_dim = intermediate_dim
        # Encoder
        self.encoder_intermediate = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Flatten(),
            # nn.Linear(256*8*8, intermediate_dim),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*16*16, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.z_mean = nn.Linear(self.intermediate_dim, latent_dim)
        self.z_log_var = nn.Linear(self.intermediate_dim, latent_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(self.intermediate_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )
        
        self.fc_p = nn.Sequential(
            nn.Linear(r_latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, 128*16*16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),
            # nn.Linear(1024, 256*8*8),
            # nn.ReLU(),
            # nn.Unflatten(1, (256, 8, 8)),
            # nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # Output: 128x128
        )
        
    def encode(self, x):
        h = self.encoder_intermediate(x)
        r = self.regressor(h)
        return self.z_mean(h), self.z_log_var(h), r

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_mean, z_log_var, r = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        r_recon = self.fc_p(r)
        return self.decode(torch.cat([z, r_recon], dim=1)), z, z_mean, z_log_var, r, r_recon


class SplitVAERegression(nn.Module):
    def __init__(self, input_channels=1, latent_dim=2, r_latent_dim = 1, intermediate_dim = 2048):
        super(SplitVAERegression, self).__init__()
        self.intermediate_dim = intermediate_dim
        # Encoder
        self.encoder_intermediate = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*8*8, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.z_mean = nn.Linear(self.intermediate_dim, latent_dim)
        self.z_log_var = nn.Linear(self.intermediate_dim, latent_dim)
        
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*8*8, intermediate_dim),
            nn.ReLU(),
            # nn.Linear(intermediate_dim, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.r_mean = nn.Linear(intermediate_dim, r_latent_dim)
        self.r_log_var = nn.Linear(intermediate_dim, r_latent_dim)
        
        self.generator = nn.Sequential(
            nn.Linear(r_latent_dim, latent_dim),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Linear(128, latent_dim),
        )

        self.fc_p = nn.Linear(r_latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 256*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # Output: 128x128
        )
        
    def encode(self, x):
        h = self.encoder_intermediate(x)
        r_emb = self.regressor(h)
        return self.z_mean(h), self.z_log_var(h), self.r_mean(r_emb), self.r_log_var(r_emb)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        e_h = self.encoder_intermediate(x)
        z_mean, z_log_var = self.z_mean(e_h), self.z_log_var(e_h)

        r_h = self.regressor(x)
        r_mean, r_log_var = self.r_mean(r_h), self.r_log_var(r_h)

        z = self.reparameterize(z_mean, z_log_var)
        r = self.reparameterize(r_mean, r_log_var)
        gen_z = self.generator(r)

        return self.decode(torch.cat([z, gen_z], dim=1)), z, z_mean, z_log_var, r, r_mean, r_log_var, gen_z


class SimpleSplitVAERegression(nn.Module):
    def __init__(self, input_channels=1, latent_dim=2, r_latent_dim = 1, intermediate_dim = 2048):
        super(SimpleSplitVAERegression, self).__init__()
        self.intermediate_dim = intermediate_dim
        # Encoder
        self.encoder_intermediate = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*8*8, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.z_mean = nn.Linear(self.intermediate_dim, latent_dim)
        self.z_log_var = nn.Linear(self.intermediate_dim, latent_dim)
        
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*8*8, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        )
        # self.r_mean = nn.Linear(intermediate_dim, r_latent_dim)
        # self.r_log_var = nn.Linear(intermediate_dim, r_latent_dim)
        
        self.generator = nn.Sequential(
            nn.Linear(r_latent_dim, latent_dim),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Linear(128, latent_dim),
        )

        self.fc_p = nn.Linear(r_latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 256*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # Output: 128x128
        )
        
    def encode(self, x):
        h = self.encoder_intermediate(x)
        r = self.regressor(h)
        return self.z_mean(h), self.z_log_var(h), r

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        e_h = self.encoder_intermediate(x)
        z_mean, z_log_var = self.z_mean(e_h), self.z_log_var(e_h)

        r = self.regressor(x)

        z = self.reparameterize(z_mean, z_log_var)

        gen_z = self.generator(r)

        return self.decode(torch.cat([z, gen_z], dim=1)), z, z_mean, z_log_var, r, gen_z