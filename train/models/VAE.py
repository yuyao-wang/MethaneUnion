import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dims, latent_dim):
        super(Encoder, self).__init__()

        # 初始化卷积层
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            input_channels = h_dim

        self.conv_layers = nn.Sequential(*modules)

        # 输出隐空间的均值和方差
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_channels):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        # 反卷积层
        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.conv_layers = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], output_channels, kernel_size=3, padding=1),
                            nn.Tanh())

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 2, 2)  # 这里的维度需要根据实际情况调整
        x = self.conv_layers(x)
        return self.final_layer(x)
    

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*128, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 128*128),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        # x = x.view(x.shape[0], -1)
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=1)  # 分割获取均值和方差
        z = self.reparameterize(mu, log_var)
        # 解码
        return self.decoder(z), mu, log_var
    
    def loss(self, out):
        reconstructed_x, x, mu, log_var = out[0], out[1], out[2], out[3]
        # 重构损失
        recon_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
        # KL散度
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_div
    
class ConvVAE(nn.Module):
    def __init__(self, input_channel, latent_dim):
        super(ConvVAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1024, latent_dim))

        self.fc2 = nn.Sequential(
            nn.Linear(1024, latent_dim))
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.Linear(1024, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.Dropout(0.5),
            nn.BatchNorm2d(num_features=256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(num_features=1),
            # nn.ReLU(),
            # nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder(x)
        mu, log_var = self.fc1(x), self.fc2(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def forward(self, x):
        # 编码
        x = self.encoder(x)
        mu, log_var = self.fc1(x), self.fc2(x)
        z = self.reparameterize(mu, log_var)
        # 解码
        return self.decoder(z), mu, log_var
    
class ConvVAEGroupNorm(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAEGroupNorm, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            # nn.GroupNorm(16, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            # nn.GroupNorm(32, 32),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # nn.GroupNorm(64, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            # nn.GroupNorm(128, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1024, latent_dim))

        self.fc2 = nn.Sequential(
            nn.Linear(1024, latent_dim))
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.Linear(1024, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.Dropout(0.25),
            nn.GroupNorm(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.GroupNorm(64, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.GroupNorm(32, 32),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.GroupNorm(16, 16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(num_features=1),
            # nn.ReLU(),
            # nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        x = self.encoder(x)
        mu, log_var = self.fc1(x), self.fc2(x)
        z = self.reparameterize(mu, log_var)
        # 解码
        return self.decoder(z), mu, log_var