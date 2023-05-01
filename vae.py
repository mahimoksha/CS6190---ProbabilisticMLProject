import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import numpy as np
from torchvision.transforms import Normalize
from torch.nn import functional as F
from Dataloader.Dataloader import MonkeyPoxRandAugDataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def weight_init(module, initf):
	def foo(m):
		classname = m.__class__.__name__.lower()
		if isinstance(m, module):
			initf(m.weight)
	return foo

class VAE_CNN(nn.Module):
    def __init__(self, image_channels, hidden_dim, latent_dim):
        super(VAE_CNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(hidden_dim * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 7 * 7, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, hidden_dim * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        # print(h.shape)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        # print(z.shape)
        z = self.fc_decode(z).view(-1, self.hidden_dim, 7, 7)
        # print(z.shape)
        return self.decoder(z), mu, logvar

class Encoder(nn.Module):
    def __init__(self, image_size, intermediate_dim=512, latent_dim=2):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(image_size, intermediate_dim)
        self.fc_mean = nn.Linear(intermediate_dim, latent_dim)
        self.fc_log_var = nn.Linear(intermediate_dim, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var

class Decoder(nn.Module):
    def __init__(self, intermediate_dim=512, latent_dim=2):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, image_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class VAE(nn.Module):
    def __init__(self, image_size, intermediate_dim=512, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(image_size, intermediate_dim, latent_dim)
        self.decoder = Decoder(intermediate_dim, latent_dim)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var


def loss_function(recon_x, x, mu, logvar, kl_loss=True):
    # import pdb;pdb.set_trace()
    # loss = nn.BCELoss() # , reduction='sum')
    # print(recon_x.max(), recon_x.min(), x.max(), x.min())
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0) # loss(recon_x, x)  # 
    # print("bce", BCE)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) if kl_loss else 0.0
    return BCE + KLD

def one_epoch(epoch, model, dataloader, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()
        
    train_loss = 0
    for _, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        # print(data.shape)
        data = data.to(device) 
        recon_batch, mu, logvar = model(data)
        if epoch >= 5:
            loss = loss_function(recon_batch, data, mu, logvar)
        else:
            loss = loss_function(recon_batch, data, mu, logvar, kl_loss=False)
        if train:
            loss.backward()
            optimizer.step()
        train_loss += loss.item()
        
    return train_loss / len(dataloader.dataset), model

if __name__ == "__main__":
    params = {'batch_size': 32, 'shuffle': True}
    rootdir = '/usr/sci/scratch/Moksha/CS6190_project/'
    csv_dir = os.path.join(rootdir, "data")
    img_dir = "/usr/sci/scratch/Moksha/CS6190_project/OriginalImages/OriginalImages/Total_Data/"
    scratchDir = './Results'

    tr_csv_file = "data/trainMonkeypox_neg_only.csv" # os.path.join(csv_dir, "trainMonkeypox.csv")
    cv_csv_file = "data/cvMonkeypox_neg_only.csv"  # os.path.join(csv_dir, "cvMonkeypox.csv")
    # te_csv_file = os.path.join(csv_dir, "testMonkeypox.csv")

    trans = transforms.Compose(
        [  
            transforms.Grayscale(),
            transforms.ToTensor(), 
            # Normalize(mean=(0.485), std=(0.229))
        ]
    )
    test_trans = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(), 
            # Normalize(mean=(0.485), std=(0.229))
        ]
    )

    train = MonkeyPoxRandAugDataLoader(tr_csv_file, img_dir, transform=trans)
    train_dataloader = DataLoader(train, **params)
    cv = MonkeyPoxRandAugDataLoader(cv_csv_file, img_dir, transform=test_trans)
    cv_dataloader = DataLoader(cv, **params)

    image_channels = 1 # 3
    image_size = 224 * 224
    hidden_dim = 2056
    latent_dim = 256

    model = VAE(image_size).to(device)
    # model.apply(weight_init(module=nn.Conv2d, initf=nn.init.xavier_normal_))
    # model.apply(weight_init(module=nn.ConvTranspose2d, initf=nn.init.xavier_normal_))
    # model.apply(weight_init(module=nn.Linear, initf=nn.init.xavier_normal_))
    # model.fc_logvar.apply(weight_init(module=nn.Linear, initf=nn.init.zeros_))
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 1000
    best_val_loss = np.inf
    patience_limit = 500
    patience_count = 0
    print("Training on: ", device)
    for epoch in range(num_epochs):
        train_loss, model = one_epoch(epoch, model, train_dataloader, optimizer)
        val_loss, model = one_epoch(epoch, model, cv_dataloader, optimizer, train=False)
        print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f'{scratchDir}/vae_best_model.torch')
            patience_count = 0
            best_val_loss = val_loss
        else:
            patience_count += 1
        
        if epoch == 6:
            best_val_loss = val_loss

        if patience_count ==  patience_limit:
            print(f"Training Completed after {epoch + 1} epochs")
            break

    model = VAE(image_size).to(device)
    model.load_state_dict(torch.load(f'{scratchDir}/vae_best_model.torch'))
    model.eval()

    num_generated_images = 1000

    with torch.no_grad():
        for i in range(num_generated_images):
            z = torch.randn(1, 1, 224, 224).to(device)
            # generated_image = model.decoder(model.fc_decode(z).view(-1, hidden_dim, 7, 7)).cpu()
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1) )(
            generated_image, z_mean, z_log_var = model(z.flatten())
            generated_image.to(device)
            save_image(generated_image.view(image_channels, 224, 224), f"generated_images/neg_images/neg_image_{i + 1}.png")
            del z, generated_image





