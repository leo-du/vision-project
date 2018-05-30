from GAN.utils import *

def get_discriminator():
    return nn.Sequential(
        Flatten(),
        nn.Linear(32*32*3, 1024),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(1024, 1024),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(1024, 1)
    )

def get_generator(noise_dim=96):
    return nn.Sequential(
        nn.Linear(noise_dim, 3072),
        nn.ReLU(inplace=True),
        nn.Linear(3072, 3072),
        nn.ReLU(inplace=True),
        nn.Linear(3072, 3072),
        nn.ReLU(inplace=True),
        nn.Linear(3072, 32*32*3),
        nn.Tanh()
    )