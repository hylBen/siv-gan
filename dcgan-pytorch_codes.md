## Pytorch implementation of DCGAN by Hao Zhang
### main-MNIST.py

```python
import yaml
import torch
from torch import nn
import time
import torchvision
import os
from os import path
from gan.MNIST_data_loader import get_mnist_loader
from gan.Model import build_models
from gan.Optimizer import build_optimizers
from gan.eval import Evaluator
from gan.distributions import get_zdist
from gan.train import Trainer

# config
config_path = './configs/MNIST_dcgan.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f)

#
is_cuda = torch.cuda.is_available()

# Create missing directories
if not path.exists(config['training']['out_dir']):
    os.makedirs(config['training']['out_dir'])

imgdir = os.path.join(config['training']['out_dir'], 'imgs')
if not os.path.exists(imgdir):
    os.makedirs(imgdir)


if is_cuda:
    device = config['gpu']['device']
else:
    device = "cpu"


# Dataset
train_loader = get_mnist_loader(batch_size = config['training']['batch_size'])

# Create models
generator, discriminator = build_models(config)
print(generator)
print(discriminator)

# Put models on gpu if needed
generator = generator.to(device)
discriminator = discriminator.to(device)

g_optimizer, d_optimizer = build_optimizers(
    generator, discriminator, config)

# Use multiple GPUs if possible
#generator = nn.DataParallel(generator)
#discriminator = nn.DataParallel(discriminator)

# Distributions
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

# Test generator
generator_test = generator

# Evaluator
evaluator = Evaluator(generator_test, zdist, device=device)

# Trainer
trainer = Trainer(generator, discriminator, g_optimizer, d_optimizer)

# Training loop
it = epoch_idx = -1
print('Start training...')
while True:
    tstart = time.time()
    epoch_idx += 1
    print('Start epoch %d...' % epoch_idx)


    for step, data in enumerate(train_loader, 0):
        x_real = data
        it += 1

        x_real = x_real.to(device)

        # Discriminator updates
        z = zdist.sample((config['training']['batch_size'],))
        dloss = trainer.discriminator_trainstep(x_real, z)

        # Generators updates
        z = zdist.sample((config['training']['batch_size'],))
        gloss = trainer.generator_trainstep(z)

        print('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f'
              % (epoch_idx, it, gloss, dloss))

        #Sample if necessary
        if (it % config['training']['sample_every']) == 0:
            print('Creating samples...')
            ztest = zdist.sample((64,))
            x_generate = evaluator.create_samples(ztest)

            outfile = os.path.join(imgdir, '%08d.png' % it)
            x_generate = x_generate / 2 + 0.5
            x_generate = torchvision.utils.make_grid(x_generate)
            torchvision.utils.save_image(x_generate, outfile, nrow=8)

    tend = time.time()

    print('[epoch %0d, time %4f]' % (epoch_idx, tend-tstart))

```

### train.py

```python
# coding: utf-8

from torch.nn import functional as F
#import torch.nn as nn


class Trainer(object):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer


    def generator_trainstep(self, z):
        toogle_grad(self.generator, True)
        toogle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z)
        d_fake = self.discriminator(x_fake)
        gloss = self.compute_loss(d_fake, 1)
        gloss.backward()

        self.g_optimizer.step()

        return gloss.item()

    def discriminator_trainstep(self, x_real, z):
        toogle_grad(self.generator, False)
        toogle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        d_real = self.discriminator(x_real)
        dloss_real = self.compute_loss(d_real, 1)

        # On fake data
        x_fake = self.generator(z)
        d_fake = self.discriminator(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0)

        dloss = dloss_real + dloss_fake
        dloss.backward()
        self.d_optimizer.step()

        return dloss.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.binary_cross_entropy_with_logits(d_out, targets)

        return loss


# Utility functions
def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

```python

### Model.py

```python
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, nz, ngf):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(nn.Linear(nz, 2 * 2 * ngf * 8), nn.ReLU(True))
        self.main = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(    ngf,      1, 4, 2, 1),
            nn.Tanh()
        )


    def forward(self, input):
        output = self.fc(input)
        output = output.view(output.size(0), -1, 2, 2)
        output = self.main(output)
        return output

class Discriminator(nn.Module):

    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 2),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(2 * 2 * ndf * 8, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

def build_models(config):

    generator = Generator(
        nz=config['z_dist']['dim'],
        ngf=64)
    discriminator = Discriminator(ndf=64)

    return generator, discriminator


```

### Optimzer.py

```python
from torch import optim


def build_optimizers(generator, discriminator, config):
    optimizer = config['training']['optimizer']
    lr_g = config['training']['lr_g']
    lr_d = config['training']['lr_d']

    g_params = generator.parameters()
    d_params = discriminator.parameters()

    # Optimizers
    if optimizer == 'rmsprop':
        g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
        d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)
    elif optimizer == 'adam':
        g_optimizer = optim.Adam(g_params, lr=lr_g, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(d_params, lr=lr_d, betas=(0.5, 0.999))
    elif optimizer == 'sgd':
        g_optimizer = optim.SGD(g_params, lr=lr_g, momentum=0.)
        d_optimizer = optim.SGD(d_params, lr=lr_d, momentum=0.)

    return g_optimizer, d_optimizer
    
```

### MNIST_data_loader.py

```python
import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
import os

class MNIST_Loader():
    def __init__(self, datapath='./data/mnist_train.mat'):
        data = sio.loadmat(datapath)
        X_train = np.expand_dims(data['X_train_img'], axis=1) * 2. - 1
        self.data = torch.from_numpy(X_train).float()

        #data = np.load(datapath)
        #X_train= data * 2. - 1
        #self.data = torch.from_numpy(X_train).float()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        sample = self.data[index]

        return sample

def get_mnist_loader(batch_size):
    dataset = MNIST_Loader()


    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True)
    return data_loader
    
```

### distributions.py

```python
import torch
from torch import distributions


def get_zdist(dist_name, dim, device=None):
    # Get distribution
    if dist_name == 'uniform':
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name == 'gauss':
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist

```

### MNIST_dcgan.yaml

```python


gpu:
  device: 0
data:
  type: mnist
  train_dir: data/
  test_dir: data/
  img_size: 28
z_dist:
  type: gauss
  dim: 256
training:
  out_dir: output/mnist_dcgan
  batch_size: 64
  nworkers: 0
  optimizer: adam
  lr_g: 0.0002
  lr_d: 0.0002
  sample_every: 1000


```


