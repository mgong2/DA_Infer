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


# DA_Infer trainer, deterministic/probabilistic theta encoder, implemented by tac-gan/mmd-gan
class DA_Infer(object):
    def __init__(self, config):
        input_dim = config['idim']
        num_class = config['num_class']
        num_domain = config['num_domain']
        dim_class = config['dim_y']
        dim_domain = config['dim_d']
        dim_hidden = config['dim_z']
        num_layer = config['mlp_layers']
        num_nodes = config['mlp_nodes']
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
            self.MB = None

        if config['G_model'] == 'MLP_Generator' and config['trainer'] == 'ML':
            self.gen = MLP_Generator(self.num_var, num_class, num_domain, dim_class, dim_domain, dim_hidden, num_layer, num_nodes, is_reg, prob=False)
        if config['G_model'] == 'MLP_Generator' and config['trainer'] == 'Bayesian':
            self.gen = MLP_Generator(self.num_var, num_class, num_domain, dim_class, dim_domain, dim_hidden, num_layer, num_nodes, is_reg, prob=True)
        if config['D_model'] == 'MLP_Discriminator':
            self.dis = MLP_Disriminator(self.num_var, num_class, num_domain, num_layer, num_nodes, is_reg)

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
        dim_domain = config['dim_d']
        num_domain = config['num_domain']
        num_class = config['num_class']
        is_reg = config['is_reg']
        do_ss = config['do_ss']

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

        if config['trainer'] == 'ML':
            fake_x_a = self.gen(noise, y_a_onehot, d_onehot)
        elif config['trainer'] == 'Bayesian':
            noise_d = torch.randn(num_domain, dim_domain).to(device)
            fake_x_a, KL_reg = self.gen(noise, y_a_onehot, d_onehot, noise_d)
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

        if config['trainer'] == 'ML':
            errG = gan_loss + aux_loss_c - aux_loss_ct + aux_loss_d - aux_loss_dt
        elif config['trainer'] == 'Bayesian':
            errG = gan_loss + aux_loss_c - aux_loss_ct + aux_loss_d - aux_loss_dt + torch.dot(1.0/do_ss.squeeze(), KL_reg.squeeze())

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
        dim_domain = config['dim_d']
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
            if config['trainer'] == 'ML':
                fake_x_a = self.gen(noise, y_a_onehot, d_onehot)
            elif config['trainer'] == 'Bayesian':
                noise_d = torch.randn(num_domain, dim_domain).to(device)
                fake_x_a, _ = self.gen(noise, y_a_onehot, d_onehot, noise_d)
        output_c, output_c_tw, output_d, output_d_tw = self.dis(fake_x_a)
        output_c1, output_c_tw1, output_d1, output_d_tw1 = self.dis(x_a)

        ids_s = y_a[:, 1] != num_domain - 1

        aux_loss_c = self.aux_loss_func(output_c, y_a[:, 0])
        aux_loss_c1 = self.aux_loss_func(output_c1[ids_s], y_a[ids_s, 0])
        aux_loss_ct = self.aux_loss_func(output_c_tw[ids_s], y_a[ids_s, 0])
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


# DA_Infer trainer on a graph, deterministic/probabilistic theta encoder, implemented by tac-gan/mmd-gan
class DA_InferDAG(object):
    def __init__(self, config):
        input_dim = config['idim']
        num_class = config['num_class']
        num_domain = config['num_domain']
        dim_class = config['dim_y']
        dim_domain = config['dim_d']
        dim_hidden = config['dim_z']
        num_layer = config['num_layer_mlp']
        num_nodes = config['num_nodes_mlp']
        is_reg = config['is_reg']

        if config['dag_mat_file'] is not 'None':
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

        exec('self.gen = %s(self.num_var, num_class, num_domain, dim_class, dim_domain, dim_hidden, num_layer, num_nodes, is_reg, dag_mat, self.MB)' % config['G_model'])
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=config['G_lr'], betas=(config['G_B1'], config['G_B2']))
        if not config['skip_init']:
            self.gen.apply(utils.xavier_weights_init)

        self.mmd_loss = 0
        self.mmd_loss_s = 0
        self.mmd_loss_t = 0

    def to(self, device):
        self.gen.to(device)

    # separate training of each module, trained on source + target domain together
    def gen_update(self, x_a, y_a, hyperparameters, config, device='cpu'):
        self.gen.zero_grad()
        batch_size = hyperparameters['batch_size']
        dim_hidden = hyperparameters['dim_hidden']
        num_domain = hyperparameters['num_domain']
        num_class = hyperparameters['num_class']
        is_reg = hyperparameters['is_reg']
        input_dim = self.num_var

        # pickup Markov blanket variables
        if self.MB is not None:
            if len(self.MB) != 1:
                x_a = x_a[:, torch.LongTensor(np.int64(self.MB), device=device)]
            else:
                x_a = x_a[:, torch.LongTensor([np.int64(self.MB)], device=device)]

        # generate random Gaussian noise
        if dim_hidden != 0:
            noise = torch.randn((batch_size, dim_hidden * input_dim), device=device)

        # create domain labels
        if not is_reg:
            y_a_onehot = torch.zeros((batch_size, num_class), device=device)
            y_a_onehot.scatter_(1, y_a[:, 0].view(batch_size, 1), 1)
        else:
            y_a_onehot = y_a[:, 0].view(batch_size, 1)

        d_onehot = torch.zeros((batch_size, num_domain), device=device)
        d_onehot.scatter_(1, y_a[:, 1].long().view(batch_size, 1), 1)

        fake_x_a = self.gen(noise, y_a_onehot, d_onehot, x_a, device)

        # sigma for MMD
        base_x = config['base_x']
        base_y = config['base_y']
        sigma_list = [0.125, 0.25, 0.5, 1]
        sigma_listx = [sigma * base_x for sigma in sigma_list]
        sigma_listy = [sigma * base_y for sigma in sigma_list]

        ids_s = y_a[:, 1]!=num_domain-1
        ids_t = y_a[:, 1]==num_domain-1
        batch_size_s = len(y_a[ids_s, :])
        batch_size_t = len(y_a[ids_t, :])

        # MMD matching for each factor
        errG_s = torch.zeros(len(self.gen.nodeSort), device=device)
        errG_t = torch.zeros(len(self.gen.nodeSort), device=device)

        for i in self.gen.nodeSort:
            input_pDim = self.gen.numInput[i]
            if input_pDim > 0:
                if not self.gen.ischain:
                    output_dim = 1
                    index = np.argwhere(self.gen.dagMat[i, :])
                    index = index.flatten()
                    index = [int(j) for j in index]
                else:
                    output_dim = len(self.gen.nodesA[i])
                    if output_dim == 1:
                        index = np.argwhere(self.gen.dagMat[self.gen.nodesA[i][0], :])
                        index = index.flatten()
                        index = [int(j) for j in index]
                    else:
                        index = np.argwhere(self.gen.dagMatNew[i, :])
                        index = index.flatten()
                        index = [self.gen.nodesA[j] for j in index]
                        index = list(itertools.chain.from_iterable(index))
                        index = [int(j) for j in index]
                input_p = x_a[:, index].view(batch_size, len(index))
                if not is_reg:
                    errG_s[i] = mix_rbf_mmd2_joint(fake_x_a[ids_s, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                                 x_a[ids_s, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                                 y_a_onehot[ids_s], y_a_onehot[ids_s], d_onehot[ids_s],
                                                 d_onehot[ids_s], input_p[ids_s], input_p[ids_s], sigma_list=sigma_listx)
                else:
                    errG_s[i] = mix_rbf_mmd2_joint_regress(fake_x_a[ids_s, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                                         x_a[ids_s, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                                         y_a_onehot[ids_s], y_a_onehot[ids_s], d_onehot[ids_s],
                                                         d_onehot[ids_s], input_p[ids_s], input_p[ids_s], sigma_list=sigma_listx,
                                                         sigma_list1=sigma_listy)
                errG_t[i] = mix_rbf_mmd2_joint_regress(fake_x_a[ids_t, self.gen.nodesA[i]].view(batch_size_t, output_dim),
                                                     x_a[ids_t, self.gen.nodesA[i]].view(batch_size_t, output_dim),
                                                     input_p[ids_t], input_p[ids_t], sigma_list=sigma_listx, sigma_list1=sigma_listy)
            else:
                if not self.gen.ischain:
                    output_dim = 1
                else:
                    output_dim = len(self.gen.nodesA[i])
                if not is_reg:
                    errG_s[i] = mix_rbf_mmd2_joint(fake_x_a[ids_s][:, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                                 x_a[ids_s][:, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                                 y_a_onehot[ids_s], y_a_onehot[ids_s], d_onehot[ids_s],
                                                 d_onehot[ids_s], sigma_list=sigma_listx)
                else:
                    errG_s[i] = mix_rbf_mmd2_joint_regress(fake_x_a[ids_s][:, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                                         x_a[ids_s][:, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                                         y_a_onehot[ids_s], y_a_onehot[ids_s], d_onehot[ids_s],
                                                         d_onehot[ids_s], sigma_list=sigma_listx, sigma_list1=sigma_listy)
                errG_t[i] = mix_rbf_mmd2(fake_x_a[ids_t][:, self.gen.nodesA[i]].view(batch_size_t, output_dim),
                                       x_a[ids_t][:, self.gen.nodesA[i]].view(batch_size_t, output_dim), sigma_list=sigma_listx)
        errG_s = errG_s.sum()
        errG_t = errG_t.sum()
        errG = (num_domain-1)**2 * errG_s + errG_t

        errG.backward()
        self.gen_opt.step()
        self.mmd_loss = errG
        self.mmd_loss_s = errG_s
        self.mmd_loss_t = errG_t

    def resume(self, snapshot_prefix):
        gen_filename = snapshot_prefix + '_gen.pkl'
        state_filename = snapshot_prefix + '_state.pkl'
        self.gen.load_state_dict(torch.load(gen_filename))
        state_dict = torch.load(state_filename)
        print('Resume the model')
        return state_dict

    def save(self, snapshot_prefix, state_dict):
        gen_filename = snapshot_prefix + '_gen.pkl'
        state_filename = snapshot_prefix + '_state.pkl'
        torch.save(self.gen.state_dict(), gen_filename)
        torch.save(state_dict, state_filename)
