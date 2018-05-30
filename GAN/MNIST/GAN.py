from GAN.utils import *

def get_discriminator():
    return nn.Sequential(
        Flatten(),
        nn.Linear(1*28*28, 256),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(256, 1)
    )


def get_generator(noise_dim=96):
    return nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1*28*28),
        nn.Tanh()
    )

