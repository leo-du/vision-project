from GAN.utils import *

NOISE_DIM = 96
IMG_DIM = 32
IMG_CHAN = 1

def get_discriminator():
    return nn.Sequential(
        Flatten(),
        nn.Linear(IMG_DIM * IMG_DIM * IMG_CHAN, 256),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(256, 1)
    )


def get_generator(noise_dim=NOISE_DIM):
    return nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, IMG_DIM * IMG_DIM * IMG_CHAN),
        nn.Tanh()
    )

