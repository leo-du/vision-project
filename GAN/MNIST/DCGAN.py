from GAN.utils import *

# parameters from InfoGAN paper:
# url: https://arxiv.org/pdf/1606.03657.pdf

def get_discriminator():
    return nn.Sequential(
        Unflatten(128, 1, 28, 28),
        nn.Conv2d(1, 32, 5),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 5),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(4 * 4 * 64, 4 * 4 * 64),
        nn.LeakyReLU(),
        nn.Linear(4 * 4 * 64, 1)
    )


def get_generator(noise_dim=96):
    return nn.Sequential(
        nn.Linear(noise_dim,1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024,7*7*128),
        nn.BatchNorm1d(7*7*128),
        Unflatten(128,128,7,7),
        nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64,1,4,stride=2,padding=1),
        nn.Tanh(),
        Flatten()
    )