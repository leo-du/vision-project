import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from GAN.utils import sample_noise


def show_images(images, save=False, save_path=None):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([3, 32, 32]).transpose(1, 2, 0))
    if save:
        plt.savefig(save_path + ".png")


def train(D, G, D_optim, G_optim, D_loss_fn, G_loss_fn, dtype, loader, show_every=250,
          batch_size=128, noise_size=96, num_epochs=50, save=False):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in loader:
            if len(x) != batch_size:
                continue

            D_optim.zero_grad()
            x = x.type(dtype)
            D_x = D(2 * (x - 0.5)).type(dtype)

            z = sample_noise(batch_size, noise_size).type(dtype)
            G_z = G(z).detach()
            D_G_z = D(G_z.view(batch_size, 3, 32, 32))
            D_loss = D_loss_fn(D_x, D_G_z)
            D_loss.backward()
            D_optim.step()

            G_optim.zero_grad()
            z = sample_noise(batch_size, noise_size).type(dtype)
            G_z = G(z)
            D_G_z = D(G_z.view(batch_size, 3, 32, 32))
            G_loss = G_loss_fn(D_G_z)
            G_loss.backward()
            G_optim.step()

            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D: {:4}, G: {:.4}'.format(epoch, iter_count, D_loss.item(), G_loss.item()))
                gen_sample = G_z.data[:16].cpu().numpy() / 2.0 + 0.5
                if not save:
                    show_images(gen_sample)
                else:
                    show_images(gen_sample, True,
                                "cifar_gen/iter{}D{:4}G{:4}".
                                format(iter_count, D_loss.item(), G_loss.item()))
                plt.show()
                print()
            iter_count += 1
