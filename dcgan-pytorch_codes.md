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
