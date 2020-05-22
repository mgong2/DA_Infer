"""
Zoo of GAN trainers
"""
from __future__ import print_function
from models import *
import torch.nn as nn
import os
import torch
import utils
import numpy as np
from mmd import *
from scipy.io import loadmat


# Maximum likelihood trainer, deterministic theta encoder, inspired by tac-gan
class InferML(object):
    def __init__(self, config):
        input_dim = config['idim_a']
        num_class = config['num_class']
        num_domain = config['num_domain']
        dim_class = config['dim_y']
        dim_domain = config['dim_d']
        dim_hidden = config['dim_z']
        num_layer = config['num_layer_mlp']
        num_nodes = config['num_nodes_mlp']
        is_reg = config['is_reg']

        if config['dag_mat_file'] != 'None':
            dag_mat_file = config['dag_mat_file']
            dag_wifi = loadmat(dag_mat_file)
            dag_mat = dag_wifi['gm']
            MB = dag_wifi['MB'].flatten() - 1
            self.MB = MB[:-2]
            dag_mat = dag_mat.T
            dag_mat = dag_mat[np.ix_(self.MB, self.MB)]
            self.num_var = len(self.MB)
        else:
            self.num_var = input_dim

        if config['G_model'] == 'MLP_Generator':
            self.gen = MLP_Generator(self.num_var, num_class, num_domain, dim_class, dim_domain, dim_hidden, num_layer, num_nodes, is_reg, dag_mat)
        if config['D_model'] == 'MLP_Discriminator':
            self.dis = MLP_Disriminator(self.num_var, num_class, num_domain, num_layer, num_nodes, is_reg, dag_mat)

        # set optimizers
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=config['G_lr'], betas=(config['G_B1'], config['G_B2']))
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=config['D_lr'], betas=(config['D_B1'], config['D_B2']))
        if not config['skip_init']:
            self.gen.apply(utils.xavier_weights_init)
            self.dis.apply(utils.xavier_weights_init)

        self.aux_loss_func = nn.CrossEntropyLoss()
        self.gan_loss = 0
        self.aux_loss_c = 0
        self.aux_loss_c1 = 0
        self.aux_loss_ct = 0
        self.aux_loss_d = 0
        self.aux_loss_d1 = 0
        self.aux_loss_dt = 0

    def to(self, device):
        self.gen.to(device)
        self.dis.to(device)

    def gen_update(self, x_a, y_a, config, device='cpu'):
        for p in self.dis.parameters():
            p.requires_grad_(False)
        self.gen.zero_grad()
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        num_domain = config['num_domain']
        num_class = config['num_class']
        is_reg = config['is_reg']

        # pickup Markov blanket variables
        if self.MB is not None:
            if len(self.MB) != 1:
                x_a = x_a[:, torch.LongTensor(np.int64(self.MB), device=device)]
            else:
                x_a = x_a[:, torch.LongTensor([np.int64(self.MB)], device=device)]

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        if not is_reg:
            y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class)
        else:
            y_a_onehot = y_a[:, 0].view(batch_size, 1)
        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain)

        fake_x_a = self.gen(noise, y_a_onehot, d_onehot, device)
        output_c, output_c_tw, output_d, output_d_tw = self.dis(fake_x_a)

        # sigma for MMD
        base_x = config['base_x']
        sigma_list = [0.125, 0.25, 0.5, 1, 2]
        sigma_listx = [sigma * base_x for sigma in sigma_list]

        gan_loss = mix_rbf_mmd2(fake_x_a, x_a, sigma_list=sigma_listx)
        aux_loss_c = self.aux_loss_func(output_c, y_a[:, 0])
        aux_loss_ct = self.aux_loss_func(output_c_tw, y_a[:, 0])
        aux_loss_d = self.aux_loss_func(output_d, y_a[:, 1])
        aux_loss_dt = self.aux_loss_func(output_d_tw, y_a[:, 1])

        errG = gan_loss + aux_loss_c - aux_loss_ct + aux_loss_d - aux_loss_dt
        errG.backward()
        self.gen_opt.step()
        self.gan_loss = gan_loss
        self.aux_loss_c = aux_loss_c
        self.aux_loss_ct = aux_loss_ct
        self.aux_loss_d = aux_loss_d
        self.aux_loss_dt = aux_loss_dt

    def dis_update(self, x_a, y_a, config, device='cpu'):
        for p in self.dis.parameters():
            p.requires_grad_(True)
        self.dis.zero_grad()
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        num_domain = config['num_domain']
        num_class = config['num_class']
        is_reg = config['is_reg']

        # pickup Markov blanket variables
        if self.MB is not None:
            if len(self.MB) != 1:
                x_a = x_a[:, torch.LongTensor(np.int64(self.MB), device=device)]
            else:
                x_a = x_a[:, torch.LongTensor([np.int64(self.MB)], device=device)]

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        if not is_reg:
            y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class)
        else:
            y_a_onehot = y_a[:, 0].view(batch_size, 1)
        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain)

        with torch.no_grad():
            fake_x_a = self.gen(noise, y_a_onehot, d_onehot, device).detach()
        output_c, output_c_tw, output_d, output_d_tw = self.dis(fake_x_a)
        output_c1, output_c_tw1, output_d1, output_d_tw1 = self.dis(x_a)

        ids_s = y_a[:, 1] != num_domain - 1

        aux_loss_c = self.aux_loss_func(output_c, y_a[:, 0])
        aux_loss_c1 = self.aux_loss_func(output_c1[ids_s], y_a[ids_s, 0])
        aux_loss_ct = self.aux_loss_func(output_c_tw, y_a[:, 0])
        aux_loss_d = self.aux_loss_func(output_d, y_a[:, 1])
        aux_loss_d1 = self.aux_loss_func(output_d1, y_a[:, 1])
        aux_loss_dt = self.aux_loss_func(output_d_tw, y_a[:, 1])

        errD = aux_loss_c + aux_loss_c1 + aux_loss_ct + aux_loss_d + aux_loss_d1 + aux_loss_dt
        errD.backward()
        self.dis_opt.step()
        self.aux_loss_c = aux_loss_c
        self.aux_loss_c1 = aux_loss_c1
        self.aux_loss_ct = aux_loss_ct
        self.aux_loss_d = aux_loss_d
        self.aux_loss_d1 = aux_loss_d1
        self.aux_loss_dt = aux_loss_dt

    def resume(self, snapshot_prefix):
            gen_filename = snapshot_prefix + '_gen.pkl'
            dis_filename = snapshot_prefix + '_dis.pkl'
            state_filename = snapshot_prefix + '_state.pkl'
            self.gen.load_state_dict(torch.load(gen_filename))
            self.dis.load_state_dict(torch.load(dis_filename))
            state_dict = torch.load(state_filename)
            print('Resume the model')
            return state_dict

    def save(self, snapshot_prefix, state_dict):
        gen_filename = snapshot_prefix + '_gen.pkl'
        dis_filename = snapshot_prefix + '_dis.pkl'
        state_filename = snapshot_prefix + '_state.pkl'
        torch.save(self.gen.state_dict(), gen_filename)
        torch.save(self.dis.state_dict(), dis_filename)
        torch.save(state_dict, state_filename)


# Bayesian trainer, probabilistic theta encoder, inspired by tac-gan
class InferBayesian(object):
    def __init__(self, config):
        input_dim = config['idim_a']
        num_class = config['num_class']
        num_domain = config['num_domain']
        dim_class = config['dim_y']
        dim_domain = config['dim_d']
        dim_hidden = config['dim_z']
        num_layer = config['num_layer_mlp']
        num_nodes = config['num_nodes_mlp']
        is_reg = config['is_reg']

        if config['dag_mat_file'] != 'None':
            dag_mat_file = config['dag_mat_file']
            dag_wifi = loadmat(dag_mat_file)
            dag_mat = dag_wifi['gm']
            MB = dag_wifi['MB'].flatten() - 1
            self.MB = MB[:-2]
            dag_mat = dag_mat.T
            dag_mat = dag_mat[np.ix_(self.MB, self.MB)]
            self.num_var = len(self.MB)
        else:
            self.MB = None
            dag_mat = None
            self.num_var = input_dim

        exec('self.gen = %s(self.num_var, num_class, num_domain, dim_class, dim_domain, dim_hidden, num_layer, num_nodes, is_reg, dag_mat)' % config['G_model'])
        exec('self.dis = %s(self.num_var, num_class, num_domain, num_layer, num_nodes, is_reg, dag_mat)' % config['D_model'])
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=config['G_lr'], betas=(config['G_B1'], config['G_B2']))
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=config['D_lr'], betas=(config['D_B1'], config['D_B2']))
        if not config['skip_init']:
            self.gen.apply(utils.xavier_weights_init)
            self.dis.apply(utils.xavier_weights_init)

        self.aux_loss_func = nn.CrossEntropyLoss()
        self.gan_loss = 0
        self.aux_loss_c = 0
        self.aux_loss_c1 = 0
        self.aux_loss_ct = 0
        self.aux_loss_d = 0
        self.aux_loss_d1 = 0
        self.aux_loss_dt = 0

    def to(self, device):
        self.gen.to(device)
        self.dis.to(device)

    def gen_update(self, x_a, y_a, config, device='cpu'):
        for p in self.dis.parameters():
            p.requires_grad_(False)
        self.gen.zero_grad()
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        num_domain = config['num_domain']
        num_class = config['num_class']
        is_reg = config['is_reg']

        # pickup Markov blanket variables
        if self.MB is not None:
            if len(self.MB) != 1:
                x_a = x_a[:, torch.LongTensor(np.int64(self.MB), device=device)]
            else:
                x_a = x_a[:, torch.LongTensor([np.int64(self.MB)], device=device)]

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        if not is_reg:
            y_a_onehot = torch.zeros((batch_size, num_class), device=device)
            y_a_onehot.scatter_(1, y_a[:, 0].view(batch_size, 1), 1)
        else:
            y_a_onehot = y_a[:, 0].view(batch_size, 1)
        d_onehot = torch.zeros(batch_size, num_domain, device=device)
        d_onehot.scatter_(1, y_a[:, 1].view(batch_size, 1), 1)

        fake_x_a = self.gen(noise, y_a_onehot, d_onehot, device)
        output_c, output_c_tw, output_d, output_d_tw = self.dis(fake_x_a)

        # sigma for MMD
        base_x = config['base_x']
        sigma_list = [0.125, 0.25, 0.5, 1, 2]
        sigma_listx = [sigma * base_x for sigma in sigma_list]

        gan_loss = mix_rbf_mmd2(fake_x_a, x_a, sigma_list=sigma_listx)
        aux_loss_c = self.aux_loss_func(output_c, y_a[:, 0])
        aux_loss_ct = self.aux_loss_func(output_c_tw, y_a[:, 0])
        aux_loss_d = self.aux_loss_func(output_d, y_a[:, 1])
        aux_loss_dt = self.aux_loss_func(output_d_tw, y_a[:, 1])

        errG = gan_loss + aux_loss_c - aux_loss_ct + aux_loss_d - aux_loss_dt
        errG.backward()
        self.gen_opt.step()
        self.gan_loss = gan_loss
        self.aux_loss_c = aux_loss_c
        self.aux_loss_ct = aux_loss_ct
        self.aux_loss_d = aux_loss_d
        self.aux_loss_dt = aux_loss_dt

    def dis_update(self, x_a, y_a, config, device='cpu'):
        for p in self.dis.parameters():
            p.requires_grad_(True)
        self.dis.zero_grad()
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        num_domain = config['num_domain']
        num_class = config['num_class']
        is_reg = config['is_reg']

        # pickup Markov blanket variables
        if self.MB is not None:
            if len(self.MB) != 1:
                x_a = x_a[:, torch.LongTensor(np.int64(self.MB), device=device)]
            else:
                x_a = x_a[:, torch.LongTensor([np.int64(self.MB)], device=device)]

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        if not is_reg:
            y_a_onehot = torch.zeros((batch_size, num_class), device=device)
            y_a_onehot.scatter_(1, y_a[:, 0].view(batch_size, 1), 1)
        else:
            y_a_onehot = y_a[:, 0].view(batch_size, 1)
        d_onehot = torch.zeros(batch_size, num_domain, device=device)
        d_onehot.scatter_(1, y_a[:, 1].view(batch_size, 1), 1)

        with torch.no_grad():
            fake_x_a = self.gen(noise, y_a_onehot, d_onehot, device).detach()
        output_c, output_c_tw, output_d, output_d_tw = self.dis(fake_x_a)
        output_c1, output_c_tw1, output_d1, output_d_tw1 = self.dis(x_a)

        ids_s = y_a[:, 1] != num_domain - 1

        aux_loss_c = self.aux_loss_func(output_c, y_a[:, 0])
        aux_loss_c1 = self.aux_loss_func(output_c1[ids_s], y_a[ids_s, 0])
        aux_loss_ct = self.aux_loss_func(output_c_tw, y_a[:, 0])
        aux_loss_d = self.aux_loss_func(output_d, y_a[:, 1])
        aux_loss_d1 = self.aux_loss_func(output_d1, y_a[:, 1])
        aux_loss_dt = self.aux_loss_func(output_d_tw, y_a[:, 1])

        errD = aux_loss_c + aux_loss_c1 + aux_loss_ct + aux_loss_d + aux_loss_d1 + aux_loss_dt
        errD.backward()
        self.dis_opt.step()
        self.aux_loss_c = aux_loss_c
        self.aux_loss_c1 = aux_loss_c1
        self.aux_loss_ct = aux_loss_ct
        self.aux_loss_d = aux_loss_d
        self.aux_loss_d1 = aux_loss_d1
        self.aux_loss_dt = aux_loss_dt

    def resume(self, snapshot_prefix):
            gen_filename = snapshot_prefix + '_gen.pkl'
            dis_filename = snapshot_prefix + '_dis.pkl'
            state_filename = snapshot_prefix + '_state.pkl'
            self.gen.load_state_dict(torch.load(gen_filename))
            self.dis.load_state_dict(torch.load(dis_filename))
            state_dict = torch.load(state_filename)
            print('Resume the model')
            return state_dict

    def save(self, snapshot_prefix, state_dict):
        gen_filename = snapshot_prefix + '_gen.pkl'
        dis_filename = snapshot_prefix + '_dis.pkl'
        state_filename = snapshot_prefix + '_state.pkl'
        torch.save(self.gen.state_dict(), gen_filename)
        torch.save(self.dis.state_dict(), dis_filename)
        torch.save(state_dict, state_filename)
