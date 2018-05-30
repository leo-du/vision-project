# CSE 455 Computer Vision Final Project

## Generative Adversarial Models

### Demo

| Vanilla GAN on MNIST | Deep Convolutional GAN on MNIST |
| -------------------- | ------------------------------- |
| ![vg](assets/Vanilla.gif) |    ![dc](assets/DCGAN.gif) |


| Vanilla GAN on CIFAR | Deep Convolutional GAN on CIFAR |
| -------------------- | ------------------------------- |
| ![vg](assets/cifar_vanilla.gif) |    ![dc](assets/cifar_dcgan.gif) |

### Model

Our Vanilla GAN are just Multi-Layer Perceptrons (linear transformations followed by ReLU nonlinearities).
CIFAR's Vanilla model has more layers than MNIST; and Generator has more layers than Discriminators. For DCGAN,
Generator and Discriminator in both MNIST and CIFAR have deep convolutional structures. The parameters for MNIST's
both model is taken from __InfoGAN__ (Chen et al.). CIFAR's model is similar to MNIST's only deeper.

### Usage

To reproduce our results, first run `$ jupyter notebook` and copy over the following setup code:

___Note:___ if you run on a CPU please change the `dtype` to `torch.FloatTensor` (uncomment the last line). However,
beware that our model includes deep convolutional networks that would run extremely slow on CPU.

```python
import GAN
import GAN.MNIST.GAN, GAN.MNIST.DCGAN
import GAN.CIFAR.GAN, GAN.CIFAR.DCGAN
from GAN.utils import *
from dataloader import *

import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

mnist_loader, cifar_loader = get_data()
dtype = torch.cuda.FloatTensor
# dtype = torch.FloatTensor
```

To train a model and see the pictures for yourself, you could run the following code to train a vanilla GAN (Goodfellow et al.) on MNIST dataset:

```python
D = GAN.MNIST.GAN.get_discriminator().type(dtype)
G = GAN.MNIST.GAN.get_generator().type(dtype)

D_optim = GAN.utils.get_optimizer(D)
G_optim = GAN.utils.get_optimizer(G)

GAN.MNIST.train(D, G, D_optim, G_optim, discriminator_loss,
                generator_loss, dtype, mnist_loader)
```

To switch to CIFAR dataset, change all `MNIST` to `CIFAR` would do the trick. To switch to Deep Convolutional GAN (i.e., __DCGAN__, Radford et al.),
use `GAN.MNIST.DCGAN` instead in the first two lines (you could also do `CIFAR`).

## References

1. X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel. ["Infogan: Interpretable
    representation learning by information maximizing generative adversarial nets"](https://arxiv.org/abs/1606.03657), in _NIPS_, 2016.
2. I. Goodfellow, J. Pouget-Abdie, M.Mirza, B. Xu, D. Warde-Farley, S.Ozair, A. Courville, and Y. Bengio,
    ["Generative Adversarial Nets"](https://arxiv.org/abs/1406.2661), in _NIPS_, 2014, pp.2672-2680.
3. A. Radford, L. Metz, and S. Chintala, ["Unsupervised Representation Learning with
    Deep Convolutional Generative Adversarial Networks"](https://arxiv.org/abs/1511.06434), in _ICLR_, 2016.
