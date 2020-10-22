from __future__ import print_function
from argparse import ArgumentParser
import torch
import torch.nn.init as init
import numpy as np
import sys
import os
import datetime
import time
from dataset_simul import *
from dataset_flow import *
from dataset_wifi import *
from dataset_mnistr import *
from dataset_digits import *
from dataset_simul import *
import torch.nn as nn

## Default value set for dataset flow and DA_Infer_MMD trainer

def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = ArgumentParser(description=usage)

    ### Dataset/Dataloader stuff ###
    parser.add_argument(
        '--dataset', type=str, default='DatasetFlow5',
        help='Multiple domain data (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=1,
        help='Number of dataloader workers; consider using less for HDF5 '
             '(default: %(default)s)')
    parser.add_argument('--cuda', action='store_true',
        help='Using cuda (default: %(default)s)')
    parser.add_argument(
        '--num_class', type=int, default=2,
        help='number of classes '
             '(default: %(default)s)')
    parser.add_argument(
        '--num_domain', type=int, default=6,
        help='number of domains '
             '(default: %(default)s)')
    parser.add_argument(
        '--num_train', type=int, default=500,
        help='Number of training examples in simulated data'
             '(default: %(default)s)')
    parser.add_argument(
        '--tar_id', type=int, default=1,
        help='target domain id (default: %(default)s)')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use; affects both initialization and '
             ' dataloading. (default: %(default)s)')
    parser.add_argument(
        '--resolution', type=int, default=32,
        help='input image size '
             '(default: %(default)s)')
    parser.add_argument(
        '--idim', type=int, default=4,
        help='input image channel in the source domain '
             '(default: %(default)s)')

    ### train function, hyperparameters ###
    parser.add_argument(
        '--trainer', type=str, default='DA_Infer_JMMD',
        help='train functions (default: %(default)s)')
    parser.add_argument(
        '--train_mode', type=str, default='m0',
        help='train modes (default: %(default)s)')
    parser.add_argument(
        '--estimate', type=str, default='ML',
        help='ML/Bayesian estimate (default: %(default)s)')
    parser.add_argument(
        '--gan_loss', type=str, default='mmd',
        help='Default location to store all weights, samples, data, and logs '
             ' (default: %(default)s)')
    parser.add_argument(
        '--AC_weight', type=float, default=1.0,
        help='auxiliary classifier weight '
             '(default: %(default)s)')
    parser.add_argument(
        '--SRC_weight', type=float, default=1.0,
        help='source domain classifier weight '
             '(default: %(default)s)')
    parser.add_argument(
        '--TAR_weight', type=float, default=0.1,
        help='target domain classifier weight '
             '(default: %(default)s)')
    parser.add_argument(
        '--warmup', type=int, default=200,
        help='warmp up epochs training in the source domain'
             '(default: %(default)s)')
    parser.add_argument(
        '--gp', type=float, default=10.0,
        help='is regression?: %(default)s)')
    parser.add_argument(
        '--sn', action='store_true',
        help='use spectral norm?: %(default)s)')

    ### Model stuff ###
    parser.add_argument(
        '--G_model', type=str, default='MLP_Generator',
        help='Name of the model module (default: %(default)s)')
    parser.add_argument(
        '--D_model', type=str, default='MLP_Classifier',
        help='Name of the model module (default: %(default)s)')
    parser.add_argument(
        '--G_mlp_layers', type=int, default=1,
        help='number of MLP hidden layers (default: %(default)s)')
    parser.add_argument(
        '--G_mlp_nodes', type=int, default=32,
        help='number of nodes in each MLP hidden layer (default: %(default)s)')
    parser.add_argument(
        '--D_mlp_layers', type=int, default=1,
        help='number of MLP hidden layers (default: %(default)s)')
    parser.add_argument(
        '--D_mlp_nodes', type=int, default=64,
        help='number of nodes in each MLP hidden layer (default: %(default)s)')
    parser.add_argument(
        '--dim_z', type=int, default=4,
        help='Noise dimensionality: %(default)s)')
    parser.add_argument(
        '--dim_y', type=int, default=2,
        help='class embedding dimensionality: %(default)s)')
    parser.add_argument(
        '--dim_d', type=int, default=1,
        help='domain embedding dimensionality: %(default)s)')
    parser.add_argument(
        '--is_reg', action='store_true',
        help='is regression?: %(default)s)')
    parser.add_argument(
        '--useMB', action='store_false',
        help='use Markov Blanket?: %(default)s)')
    parser.add_argument(
        '--dag_mat_file', type=str, default='dag_mat.npz',
        help='DAG matrix file: %(default)s)')

    ### Model init stuff ###
    parser.add_argument(
        '--skip_init', action='store_true',
        help='Skip initialization, ideal for testing when ortho init was used '
             '(default: %(default)s)')

    ### Optimizer stuff ###
    parser.add_argument(
        '--G_lr', type=float, default=1e-2,
        help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_lr', type=float, default=1e-2,
        help='Learning rate to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B1', type=float, default=0.5,
        help='Beta1 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B1', type=float, default=0.5,
        help='Beta1 to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B2', type=float, default=0.999,
        help='Beta2 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B2', type=float, default=0.999,
        help='Beta2 to use for Discriminator (default: %(default)s)')

    ### Batch size, parallel, and precision stuff ###
    parser.add_argument(
        '--batch_size', type=int, default=300,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--num_G_steps', type=int, default=1,
        help='Number of D steps per G step (default: %(default)s)')
    parser.add_argument(
        '--num_MI_steps', type=int, default=1,
        help='Number of D steps per G step (default: %(default)s)')
    parser.add_argument(
        '--num_D_steps', type=int, default=1,
        help='Number of D steps per G step (default: %(default)s)')
    parser.add_argument(
        '--num_epochs', type=int, default=500,
        help='Number of epochs to train for (default: %(default)s)')

    ### Bookkeping stuff ###
    parser.add_argument(
        '--display_every', type=int, default=10,
        help='display every X iterations (default: %(default)s)')
    parser.add_argument(
        '--save_every', type=int, default=10,
        help='Save every X iterations (default: %(default)s)')
    parser.add_argument(
        '--base_root', type=str, default='..',
        help='Default location to store all weights, samples, data, and logs '
             ' (default: %(default)s)')
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--weights_root', type=str, default='weights',
        help='Default location to store weights (default: %(default)s)')
    parser.add_argument(
        '--logs_root', type=str, default='logs',
        help='Default location to store logs (default: %(default)s)')
    parser.add_argument(
        '--samples_root', type=str, default='samples',
        help='Default location to store samples (default: %(default)s)')
    parser.add_argument(
        '--experiment_name', type=str, default='',
        help='Optionally override the automatic experiment naming with this arg. '
             '(default: %(default)s)')

    ### Resume training stuff
    parser.add_argument(
        '--load_weights', type=str, default='',
        help='Suffix for which weights to load (e.g. best0, copy0) '
             '(default: %(default)s)')
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='Resume training? (default: %(default)s)')

    return parser


# Utility to peg all roots to a base root
# If a base root folder is provided, peg all other root folders to it.
def update_config_roots(config):
    if config['base_root']:
        print('Pegging all root folders to base root %s' % config['base_root'])
        for key in ['data', 'weights', 'logs', 'samples']:
            config['%s_root' % key] = '%s/%s' % (config['base_root'], key)
    return config


# Utility to prepare root folders if they don't exist; parent folder must exist
def prepare_root(config):
    for key in ['weights_root', 'logs_root', 'samples_root']:
        if not os.path.exists(config[key]):
            print('Making directory %s for %s...' % (config[key], key))
            os.mkdir(config[key])

        if not os.path.exists(config[key]+'/' + config['dataset']):
            print('Making directory %s for %s...' % (config[key], key))
            os.mkdir(config[key]+'/' + config['dataset'])


def seed_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


# Name an experiment based on its config
def name_from_config(config):
    name = '_'.join([
        item for item in [
            config['dataset'] + '/',
            'tarId%d' % config['tar_id'],
            'seed%d' % config['seed'],
            'idim%d' % config['idim'],
            config['trainer'],
            config['estimate'],
            config['train_mode'],
            'warmup%d' % config['warmup'],
            config['G_model'],
            config['D_model'],
            'Diter%d' % config['num_D_steps'],
            'AC_weight%3.2f' % config['AC_weight'],
            'SRC_weight%3.2f' % config['SRC_weight'],
            'TAR_weight%3.2f' % config['TAR_weight'],
            'useMB%d' % config['useMB'],
            'G_mlp_nodes%d' % config['G_mlp_nodes'],
            'D_mlp_nodes%d' % config['D_mlp_nodes'],
            'bs%d' % config['batch_size'],
            'Glr%2.1e' % config['G_lr'],
            'Dlr%2.1e' % config['D_lr'],
            'numDomain%d' % config['num_domain'],
            'dimDomain%d' % config['dim_d'],
            'dimHidden%d' % config['dim_z'],
        ]
        if item is not None])

    return name


def get_data_loader(conf, batch_size, num_workers):
    print("dataset=%s(conf)" % conf['class_name'])
    exec("dataset=%s(conf)" % conf['class_name'])
    return torch.utils.data.DataLoader(dataset=locals()['dataset'], batch_size=batch_size, shuffle=True, num_workers=2)


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        # print m.__class__.__name__
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)


# Resume training status
def resume_state(snapshot_prefix):
    state_filename = snapshot_prefix + '_state.pkl'
    state_dict = torch.load(state_filename)
    return state_dict
    print('Resume the training status')
