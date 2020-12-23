""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange

from PIL import Image


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
import torchvision.transforms as transforms
from utils import toggle_grad

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback
import torch.utils.data as data

data_dict = {'mnist_m': 'MNIST_M', 'mnist': 'MNIST', 'svhn': 'SVHN','syn_digits':'SYN_DIGITS','usps':'USPS', 'sign':'SIGN','syn_sign':'SYN_SIGN', 'sign64':'SIGN64','syn_sign64':'SYN_SIGN64'}

def test_acc(model, test_loader):

    model.eval()
    with torch.no_grad():
      test_loss = 0
      correct = 0
      for data, target in test_loader:
        # data = torch.as_tensor(data)
        # target = torch.as_tensor(target)
        data, target = data.cuda(), target.cuda().long()
        # if torch.sum(target == 1) > 1:
        #     plt.figure(1)
        #     plt.imshow(np.transpose((data[target == 1][0].cpu().numpy() + 1) / 2, [1, 2, 0]))
        #     plt.show()
        output = model(data)[2]
        test_loss += F.nll_loss(output, target).sum().item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset) * 1.0))
    model.train()

    return correct / len(test_loader.dataset)*1.0


class source_domain_numpy(data.Dataset):
  def __init__(self,root, root_list, transform=None, target_transform=None):  # last four are dummies

    self.transform = transform

    self.data_list = []
    self.label_list = []
    self.len_s = 20000
    root_list = root_list.split(',')
    for root_s in root_list:
        source_root = data_dict[root_s]
        data_source, labels_source = torch.load(os.path.join(root, root_s, source_root + '_train.pt'))
        l = data_source.shape[0]
        choosen_index = np.random.choice(l,self.len_s,replace=False)
        self.data_list.append(data_source[choosen_index])
        self.label_list.append(labels_source[choosen_index].squeeze())






    self.domain_num = len(root_list)
    print(self.len_s, self.domain_num)

  def inverse_data(self, data, labels):

      data = np.concatenate([data, 255 - data], axis=0)
      labels = np.concatenate([labels] * 2, axis=0)

      return data, labels


  def pre_prcess(self,img):

    if len(img.shape) == 2:
      img = np.concatenate([np.expand_dims(img,axis=2)]*3,axis=2)

    img = Image.fromarray(img)
    return img



  def __getitem__(self, index):


    # if self.transform is not None:
    # img = self.transform(img)
    # Apply my own transform
    index_data = np.random.choice(20000,1).item()
    chosen_d = np.random.choice(self.domain_num, 1).item()


    data_s = self.data_list[chosen_d][index_data]
    label_s = self.label_list[chosen_d][index_data]

    img_s = self.pre_prcess(data_s)

    if self.transform is not None:
      img_s = self.transform(img_s)

    return img_s, label_s, chosen_d

  def __len__(self):
    return self.len_s*4

class domain_test_numpy(data.Dataset):
  def __init__(self, root,  root_t, transform=None):  # last four are dummies

      self.transform = transform

      domain_root = data_dict[root_t]

      self.len_t = 9000

      self.data_domain, self.labels_domain = torch.load(os.path.join(root, root_t, domain_root + '_test.pt'))
      l = self.data_domain.shape[0]
      choosen_index = np.random.choice(l, self.len_t, replace=False)

      self.data_domain = self.data_domain[choosen_index]
      self.labels_domain = self.labels_domain[choosen_index]

      self.len_t = self.labels_domain.shape[0]

  def pre_prcess(self, img):
      if len(img.shape) == 2:
        img = np.concatenate([np.expand_dims(img, axis=2)] * 3, axis=2)

      img = Image.fromarray(img)
      return img



  def __getitem__(self, index):


    # if self.transform is not None:
    # img = self.transform(img)
    # Apply my own transform

    img_t = self.pre_prcess(self.data_domain[index])
    # label_t = self.labels_domain[chosen_t]

    if self.transform is not None:
      img_t = self.transform(img_t)

    return img_t, self.labels_domain[index].squeeze()

  def __len__(self):
    return self.len_t

# of this training run.
def run(config):

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object

  # for the activation specified as a string)
  config['resolution'] = 32#utils.imsize_dict[config['dataset']]
  config['n_classes'] = 10#utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  # By default, skip init if resuming training.
  if config['resume']:
    print('Skipping initialization for training resumption...')
    config['skip_init'] = True
  config = utils.update_config_roots(config)
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)

  # Next, build the model
  G = model.Generator(**config).to(device)
  D = model.Discriminator(**config).to(device)

   # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True,
                               'no_optim': True}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    ema = None
  
  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    print('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D)
  # print(G)
  # print(D)
  print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    print('Loading weights...')
    utils.load_weights(G, D, state_dict,
                       config['weights_root'], experiment_name, 
                       config['load_weights'] if config['load_weights'] else None,
                       G_ema if config['ema'] else None)

  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.DataParallel(GD)

    if config['cross_replica']:
      patch_replication_callback(GD)

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname, 
                                 reinitialize=(not config['resume']))
  print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
  # Write metadata
  utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])

  transforms_train = transforms.Compose([transforms.Resize(config['resolution']),transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

  print(config['base_root'])
  data_set = source_domain_numpy(root=config['base_root'], root_list=config['source_dataset'], transform=transforms_train)
  # loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
  #                                     'start_itr': state_dict['itr']})
  loaders = torch.utils.data.DataLoader(data_set, batch_size=D_batch_size, shuffle=True,
           num_workers=config['num_workers'],
           pin_memory=True,
           worker_init_fn=np.random.seed,drop_last=True)

  test_set_s = domain_test_numpy(root= config['base_root'],
                               root_t=config['target_dataset'], transform=transforms_train)
  test_loader_s = torch.utils.data.DataLoader(test_set_s, batch_size=D_batch_size, shuffle=False,
                                            num_workers=config['num_workers'],
                                            pin_memory=True,
                                            worker_init_fn=np.random.seed, drop_last=True)

  test_set_t = domain_test_numpy(root= config['base_root'],root_t=config['target_dataset'],transform=transforms_train)
  test_loader_t = torch.utils.data.DataLoader(test_set_t, batch_size=D_batch_size, shuffle=False,
           num_workers=config['num_workers'],
           pin_memory=True,
           worker_init_fn=np.random.seed,drop_last=True)

  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  z_, y_, yd_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],config['n_domain'],
                             device=device, fp16=config['G_fp16'])
  # Prepare a fixed z & y to see individual sample evolution throghout training
  fixed_z, fixed_y, fixed_yd = utils.prepare_z_y(G_batch_size, G.dim_z,
                                       config['n_classes'],config['n_domain'], device=device,
                                       fp16=config['G_fp16'])
  fixed_z.sample_()
  fixed_y.sample_()
  fixed_yd.sample_()
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':
    train = train_fns.GAN_training_function(G, D, GD, z_, y_,yd_,
                                            ema, state_dict, config)
  # Else, assume debugging and use the dummy train fn
  else:
    train = train_fns.dummy_training_function()
  # Prepare Sample function for use with inception metrics
  # sample = functools.partial(utils.sample,
  #                             G=(G_ema if config['ema'] and config['use_ema']
  #                                else G),
  #                             z_=z_, y_=y_, config=config)

  print('Beginning training at epoch %d...' % state_dict['epoch'])
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], config['num_epochs']):
    if epoch%10 == 0:
        test_acc(D, test_loader_s)
        test_acc(D, test_loader_t)
    # Which progressbar to use? TQDM or my own?
    if config['pbar'] == 'mine':
      pbar = utils.progress(loaders,displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    else:
      pbar = tqdm(loaders)
    for i, (x_s, y, yd) in enumerate(pbar):
      # Increment the iteration counter
      state_dict['itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      G.train()
      D.train()
      if config['ema']:
        G_ema.train()
      if config['D_fp16']:
        x_s,x_t, y = x_s.to(device).half(), x_t.to(device).half(), y.to(device)
      else:
        x_s, y, yd = x_s.to(device), y.to(device),yd.to(device)
      metrics = train(x_s, y, yd)
      # train_log.log(itr=int(state_dict['itr']), **metrics)
      
      # Every sv_log_interval, log singular values
      if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
        train_log.log(itr=int(state_dict['itr']), 
                      **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

      # If using my progbar, print metrics.
      if config['pbar'] == 'mine':
          print(', '.join(['itr: %d' % state_dict['itr']] 
                           + ['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]), end=' ')

      # Save weights and copies as configured at specified interval
      if not (state_dict['itr'] % config['save_every']):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
          if config['ema']:
            G_ema.eval()
        train_fns.save_and_sample(G, D, G_ema, z_, y_,yd_, fixed_z, fixed_y,fixed_yd,
                                  state_dict, config, experiment_name)

      # Test every specified interval
      if not (state_dict['itr'] % config['test_every']):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
        # train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
        #                get_inception_metrics, experiment_name, test_log)
    # Increment epoch counter at end of epoch
    state_dict['epoch'] += 1


def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)

if __name__ == '__main__':
  main()