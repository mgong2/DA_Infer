"""
DAGAN: Domain Adaptation GAN for single-source domain adaptation.
Directly minimizing maximum Mean Discrepancy (MMD), no discriminator D is involved.
"""
from __future__ import print_function
import sys
import os
import torch
import utils
import torchvision
import itertools
from torch.utils.tensorboard import SummaryWriter
from trainer import *
from sklearn.metrics import pairwise_distances
from os.path import join
import torch.nn.functional as F


def test_acc(model, test_loader, device):

    model.eval()
    with torch.no_grad():
      test_loss = 0
      correct = 0
      for data, target in test_loader:
        data, target = data.to(device), target.to(device).long()
        output = model(data)[1]
        test_loss += F.nll_loss(output, target).sum().item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset) * 1.0))
    model.train()

    return correct / len(test_loader.dataset)*1.0


def run(config):

    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)

    if config['cuda']:
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Import the model--this line allows us to dynamically select different files.
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    batch_size = config['batch_size']
    num_class = config['num_class']
    num_domain = config['num_domain']
    dim_z = config['dim_z']

    state_dict = {'epoch': 0, 'iterations': 0, 'best_score': 0, 'config': config}

    if config['trainer'] == 'ML':
        trainer = InferML(config)
    if config['trainer'] == 'Bayesian':
        trainer = InferBayesian(config)

    if config['resume']:
        state_dict = utils.resume_state(os.path.join(config['weights_root'], experiment_name))
        trainer.resume(os.path.join(config['weights_root'], experiment_name))
    trainer.to(device)

    iterations = state_dict['iterations']

    # config tensorboard writer
    log_folder = os.path.join(config['logs_root'], experiment_name)
    writer = SummaryWriter(log_folder)

    # load datasets
    dataset_specs = {'class_name': config['source_dataset'], 'seed': config['seed'], 'train': True,
                        'root': join(config['data_root'], config['source_dataset']), 'num_train': config['num_train'],
                        'num_domain': config['num_domain'], 'num_class': config['num_class'], 'dim': config['idim_a'],
                        'dim_d': config['dim_d']}
    train_loader = utils.get_data_loader(dataset_specs, batch_size, config['num_workers'])
    pair_dist = pairwise_distances(train_loader.dataset.data)
    config['base_x'] = np.median(pair_dist)
    if config['is_reg']:
        pair_dist = pairwise_distances(train_loader.dataset.y)
        config['base_y'] = np.median(pair_dist)

    # training
    best_score = state_dict['best_score']
    Diters = config['num_D_steps']
    Giters = config['num_G_steps']
    Diter = 0
    Giter = Giters
    best_score = state_dict['best_score']
    for ep in range(state_dict['epoch'], config['num_epochs']):
        state_dict['epoch'] = ep
        if config['cuda']:
            test_acc_target = test_acc(trainer.dis, train_loader, device=device)
            writer.add_scalar('test_acc_target', test_acc_target, ep)

        for it, (x, y) in enumerate(train_loader):
            if x.size(0) != batch_size:
                continue
            trainer.gen.train()
            trainer.dis.train()
            x = x.to(device)
            y = y.to(device).view(x.size(0), 2)

            if Diter < Diters:
                trainer.dis_update(x, y, config, device)
                Diter += 1
                if Diter == Diters:
                    Giter = 0

            if Giter < Giters and Diter > Diters:
                trainer.gen_update(x, y, config, device)
                Giter += 1
                if Giter == Giters:
                    Diter = 0

            if Diter == Diters:
                Diter += 1

            # Dump training stats in log file
            if (iterations + 1) % config['save_every'] == 0:
                writer.add_scalar('gan_loss', trainer.gan_loss, iterations)
                writer.add_scalar('aux_loss_c', trainer.aux_loss_c, iterations)
                writer.add_scalar('aux_loss_c1', trainer.aux_loss_c1, iterations)
                writer.add_scalar('aux_loss_ct', trainer.aux_loss_ct, iterations)
                writer.add_scalar('aux_loss_d', trainer.aux_loss_d, iterations)
                writer.add_scalar('aux_loss_d1', trainer.aux_loss_d1, iterations)
                writer.add_scalar('aux_loss_dt', trainer.aux_loss_dt, iterations)

            if (iterations + 1) % config['display_every'] == 0:
                print("Iteration: %08d, gan loss: %.2f, ac_fake: %.2f, ac_real: %.2f, ac_twin: %.2f, ad_fake: %.2f, "
                      "ad_real: %.2f, ad_twin: %.2f"
                      % (iterations + 1, trainer.gan_loss, trainer.aux_loss_c, trainer.aux_loss_c1, trainer.aux_loss_ct,
                         trainer.aux_loss_d, trainer.aux_loss_d1, trainer.aux_loss_dt))

            # Save network weights
            if (iterations + 1) % config['save_every'] == 0:
                trainer.save(os.path.join(config['weights_root'], experiment_name), state_dict)

            iterations += 1
            state_dict['iterations'] = iterations


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
