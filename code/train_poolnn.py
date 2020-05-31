"""
Domain Adaptation as a Problem of Inference on Graphical Models, baseline poolnn
"""
from __future__ import print_function
import sys
import os
import torch
import utils
import torchvision
import itertools
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
from dataset_flow import *
from dataset_wifi import *
from trainer import *


def test_acc(model, test_loader, device):

    model.eval()
    with torch.no_grad():
        test_loss_ac = 0 # classifier trained on all domains for minCh
        correct_ac = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).view(data.size(0), 2)
            output_ac = model(data)
            test_loss_ac += F.nll_loss(output_ac, target[:, 0]).sum().item()  # sum up batch loss
            pred_ac = output_ac.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_ac += pred_ac.eq(target[:, 0].view_as(pred_ac)).sum().item()

    test_loss_ac /= len(test_loader.dataset)
    print('\nTest set joint classifier: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss_ac, correct_ac, len(test_loader.dataset),
        100. * correct_ac / len(test_loader.dataset) * 1.0))

    model.train()

    return correct_ac / len(test_loader.dataset)*1.0


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
    state_dict = {'epoch': 0, 'iterations': 0, 'best_score': 0, 'final_score': 0, 'config': config}

    if config['trainer'] == 'DA_Poolnn':
        trainer = DA_Poolnn(config)
    trainer.to(device)

    if config['resume']:
        state_dict = utils.resume_state(os.path.join(config['weights_root'], experiment_name))
        trainer.resume(os.path.join(config['weights_root'], experiment_name))
        trainer.to(device)

    iterations = state_dict['iterations']

    # config tensorboard writer
    log_folder = os.path.join(config['logs_root'], experiment_name)
    writer = SummaryWriter(log_folder)

    # load datasets
    train_dataset_specs = {'class_name': config['dataset'], 'seed': config['seed'], 'train': True,
                           'root': join(config['data_root'], config['dataset']), 'num_train': config['num_train'],
                           'num_domain': config['num_domain'], 'num_class': config['num_class'], 'dim': config['idim'],
                           'dim_d': config['dim_d'], 'dag_mat_file': config['dag_mat_file'], 'useMB': config['useMB'],
                           'tar_id': config['tar_id'], 'resolution': config['resolution']}
    train_loader = utils.get_data_loader(train_dataset_specs, batch_size, config['num_workers'])
    test_dataset_specs = {'class_name': config['dataset'], 'seed': config['seed'], 'train': False,
                          'root': join(config['data_root'], config['dataset']), 'num_train': config['num_train'],
                          'num_domain': config['num_domain'], 'num_class': config['num_class'], 'dim': config['idim'],
                          'dim_d': config['dim_d'], 'dag_mat_file': config['dag_mat_file'], 'useMB': config['useMB'],
                          'tar_id': config['tar_id'], 'resolution': config['resolution']}
    test_loader = utils.get_data_loader(test_dataset_specs, batch_size, config['num_workers'])

    # training
    best_score = state_dict['best_score']
    for ep in range(state_dict['epoch'], config['num_epochs']):
        state_dict['epoch'] = ep

        test_acc_target_c = test_acc(trainer.dis, test_loader, device=device)
        writer.add_scalar('test_acc_target_ac', test_acc_target_c, ep)
        if test_acc_target_c > best_score:
            best_score = test_acc_target_c
            state_dict['best_score'] = best_score
        if ep == config['num_epochs'] - 1:
            state_dict['final_score'] = test_acc_target_c

        for it, (x, y) in enumerate(train_loader):
            if x.size(0) != batch_size:
                continue
            trainer.dis.train()
            x = x.to(device)
            y = y.to(device).view(x.size(0), 2)

            # trainer.gen_update(x, y, config, device)
            trainer.dis_update(x, y, config, device)

            # Dump training stats in log file
            if (iterations + 1) % config['save_every'] == 0:
                writer.add_scalar('aux_loss_c', trainer.aux_loss_c, iterations)

            if (iterations + 1) % config['display_every'] == 0:
                print("Epoch: %04d, Iteration: %08d, ce loss: %.2f"
                      % (ep+1, iterations + 1, trainer.aux_loss_c))

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
