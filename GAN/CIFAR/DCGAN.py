from GAN.utils import *

def get_discriminator():
    return nn.Sequential(
        Unflatten(128, 3, 32, 32),
        nn.Conv2d(3, 64, 5, padding=2, stride=2),
        nn.LeakyReLU(),
        nn.Conv2d(64, 128, 5, padding=2, stride=2),
        nn.LeakyReLU(),
        nn.Conv2d(128, 256, 5, padding=2, stride=2),
        nn.LeakyReLU(),
        nn.Conv2d(256, 512, 5, padding=2, stride=2),
        nn.LeakyReLU(),
        Flatten(),
        nn.Linear(2 * 2 * 512, 2 * 2 * 512),
        nn.LeakyReLU(),
        nn.Linear(2 * 2 * 512, 1)
    )

def get_generator():
    return nn.Sequential(
        nn.Linear(96, 2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048),
        nn.Linear(2048, 2 * 2 * 512),
        nn.BatchNorm1d(2 * 2 * 512),
        Unflatten(128, 512, 2, 2),
        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
        nn.Tanh(),
        Flatten()
    )