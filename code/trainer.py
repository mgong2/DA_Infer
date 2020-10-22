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
from os.path import join
import torch.autograd as autograd
from gan_losses import generator_loss, discriminator_loss


# DA baseline, pool neural network
class DA_Poolnn(object):
    def __init__(self, config):
        input_dim = config['idim']
        num_class = config['num_class']
        num_layer = config['D_mlp_layers']
        num_nodes = config['D_mlp_nodes']

        utils.seed_rng(config['seed'])
        if config['D_model'] == 'MLP_Classifier':
            self.dis = MLP_Classifier(input_dim, num_class, num_layer, num_nodes)
        if config['D_model'] == 'CNN_Classifier':
            self.dis = CNN_Classifier(input_dim, num_class, num_nodes)
        if config['D_model'] == 'CNN_Classifier_Exp':
            self.dis = CNN_Classifier_Exp(input_dim, num_class, num_nodes)
        if config['D_model'] == 'RES_Classifier':
            self.dis = RES_Classifier(input_dim, num_class, num_nodes)

        # set optimizers
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=config['D_lr'], betas=(config['D_B1'], config['D_B2']))

        self.aux_loss_func = nn.CrossEntropyLoss()
        self.aux_loss_c = 0

    def to(self, device):
        self.dis.to(device)

    def dis_update(self, x_a, y_a, config, device='cpu'):
        self.dis.zero_grad()

        num_domain = config['num_domain']
        ids_s = y_a[:, 1] != num_domain - 1
        output = self.dis(x_a[ids_s])
        aux_loss_c = self.aux_loss_func(output, y_a[ids_s, 0])
        aux_loss_c.backward()
        self.dis_opt.step()
        self.aux_loss_c = aux_loss_c

    def resume(self, snapshot_prefix):
        dis_filename = snapshot_prefix + '_dis.pkl'
        state_filename = snapshot_prefix + '_state.pkl'
        self.dis.load_state_dict(torch.load(dis_filename))
        state_dict = torch.load(state_filename)
        print('Resume the model')
        return state_dict

    def save(self, snapshot_prefix, state_dict):
        dis_filename = snapshot_prefix + '_dis.pkl'
        state_filename = snapshot_prefix + '_state.pkl'
        torch.save(self.dis.state_dict(), dis_filename)
        torch.save(state_dict, state_filename)


# DA_Infer trainer, deterministic/probabilistic theta encoder, implemented by tac-gan + mmd-gan
class DA_Infer_TAC(object):
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

        isProb = config['estimate'] == 'Bayesian'
        if config['G_model'] == 'MLP_Generator':
            self.gen = MLP_Generator(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden, num_layer,
                                     num_nodes, is_reg, prob=isProb)
        utils.seed_rng(config['seed'])
        if config['D_model'] == 'MLP_AuxClassifier':
            self.dis = MLP_AuxClassifier(input_dim, num_class, num_domain, num_layer, num_nodes, is_reg)

        # set optimizers
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=config['G_lr'], betas=(config['G_B1'], config['G_B2']))
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=config['D_lr'], betas=(config['D_B1'], config['D_B2']))
        # if not config['skip_init']:
        #     self.gen.apply(utils.xavier_weights_init)
        #     self.dis.apply(utils.xavier_weights_init)

        self.aux_loss_func = nn.CrossEntropyLoss()
        self.gan_loss = 0
        self.aux_loss_c = 0
        self.aux_loss_c1 = 0
        self.aux_loss_ct = 0
        self.aux_loss_d = 0
        self.aux_loss_d1 = 0
        self.aux_loss_dt = 0
        self.aux_loss_cls = 0
        self.aux_loss_cls1 = 0

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
        do_ss = config['do_ss']

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class).float()
        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain).float()

        if config['estimate'] == 'ML':
            fake_x_a = self.gen(noise, y_a_onehot, d_onehot)
        elif config['estimate'] == 'Bayesian':
            noise_d = torch.randn(num_domain, dim_domain).to(device)
            fake_x_a, KL_reg = self.gen(noise, y_a_onehot, d_onehot, noise_d)
        output_c, output_c_tw, output_d, output_d_tw, output_cls = self.dis(fake_x_a)

        # sigma for MMD
        base_x = config['base_x']
        sigma_list = [0.125, 0.25, 0.5, 1, 2]
        sigma_listx = [sigma * base_x for sigma in sigma_list]

        ids_s = y_a[:, 1] != num_domain - 1
        ids_t = y_a[:, 1] == num_domain - 1

        lambda_c = config['AC_weight']
        lambda_tar = config['TAR_weight']
        gan_loss = mix_rbf_mmd2(fake_x_a, x_a, sigma_list=sigma_listx)
        aux_loss_c = self.aux_loss_func(output_c[ids_s], y_a[ids_s, 0])
        aux_loss_ct = self.aux_loss_func(output_c_tw[ids_s], y_a[ids_s, 0])
        aux_loss_d = self.aux_loss_func(output_d, y_a[:, 1])
        aux_loss_dt = self.aux_loss_func(output_d_tw, y_a[:, 1])
        aux_loss_cls = self.aux_loss_func(output_cls[ids_t], y_a[ids_t, 0])

        if config['estimate'] == 'ML':
            errG = gan_loss + lambda_c * (aux_loss_c - aux_loss_ct + lambda_tar * aux_loss_cls + aux_loss_d - aux_loss_dt)
        elif config['estimate'] == 'Bayesian':
            errG = gan_loss + lambda_c * (aux_loss_c - aux_loss_ct + lambda_tar * aux_loss_cls + aux_loss_d - aux_loss_dt) + torch.dot(1.0/do_ss.to(device).squeeze(), KL_reg.squeeze())

        errG.backward()
        self.gen_opt.step()
        self.gan_loss = gan_loss
        self.aux_loss_c = aux_loss_c
        self.aux_loss_ct = aux_loss_ct
        self.aux_loss_d = aux_loss_d
        self.aux_loss_dt = aux_loss_dt
        self.aux_loss_cls = aux_loss_cls

    def dis_update(self, x_a, y_a, config, device='cpu'):
        for p in self.dis.parameters():
            p.requires_grad_(True)
        self.dis.zero_grad()
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        dim_domain = config['dim_d']
        num_domain = config['num_domain']
        num_class = config['num_class']

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class).float()
        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain).float()

        with torch.no_grad():
            if config['estimate'] == 'ML':
                fake_x_a = self.gen(noise, y_a_onehot, d_onehot)
            elif config['estimate'] == 'Bayesian':
                noise_d = torch.randn(num_domain, dim_domain).to(device)
                fake_x_a, _ = self.gen(noise, y_a_onehot, d_onehot, noise_d)
        output_c, output_c_tw, output_d, output_d_tw, output_cls = self.dis(fake_x_a.detach())
        output_c1, _, output_d1, _, output_cls1 = self.dis(x_a)

        ids_s = y_a[:, 1] != num_domain - 1
        ids_t = y_a[:, 1] == num_domain - 1

        lambda_tar = config['TAR_weight']
        # aux_loss_c = self.aux_loss_func(output_c[ids_s], y_a[ids_s, 0])
        aux_loss_c1 = self.aux_loss_func(output_c1[ids_s], y_a[ids_s, 0])
        aux_loss_ct = self.aux_loss_func(output_c_tw[ids_s], y_a[ids_s, 0])
        # aux_loss_d = self.aux_loss_func(output_d, y_a[:, 1])
        aux_loss_d1 = self.aux_loss_func(output_d1, y_a[:, 1])
        aux_loss_dt = self.aux_loss_func(output_d_tw, y_a[:, 1])
        aux_loss_cls = self.aux_loss_func(output_cls[ids_t], y_a[ids_t, 0])
        aux_loss_cls1 = self.aux_loss_func(output_cls1[ids_s], y_a[ids_s, 0])

        errD = aux_loss_c1 + aux_loss_ct + lambda_tar * aux_loss_cls + aux_loss_cls1 + aux_loss_d1 + aux_loss_dt

        errD.backward()
        self.dis_opt.step()
        # self.aux_loss_c = aux_loss_c
        self.aux_loss_c1 = aux_loss_c1
        self.aux_loss_ct = aux_loss_ct
        # self.aux_loss_d = aux_loss_d
        self.aux_loss_d1 = aux_loss_d1
        self.aux_loss_dt = aux_loss_dt
        self.aux_loss_cls = aux_loss_cls
        self.aux_loss_cls1 = aux_loss_cls1

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


# DA_Infer trainer, deterministic/probabilistic theta encoder, implemented by ac-gan
class DA_Infer_TAC_Adv(object):
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

        isProb = config['estimate'] == 'Bayesian'
        if config['G_model'] == 'MLP_Generator':
            self.gen = MLP_Generator(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden, num_layer,
                                     num_nodes, is_reg, prob=isProb)
        if config['G_model'] == 'CNN_Generator':
            self.gen = CNN_Generator(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden,
                                     num_nodes, prob=isProb)
        if config['G_model'] == 'CNN_Generator_Exp':
            self.gen = CNN_Generator_Exp(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden,
                                         num_nodes, prob=isProb)
        if config['G_model'] == 'RES_Generator':
            self.gen = RES_Generator(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden,
                                     num_nodes, prob=isProb)

        utils.seed_rng(config['seed'])
        if config['D_model'] == 'MLP_AuxClassifier':
            self.dis = MLP_AuxClassifier(input_dim, num_class, num_domain, num_layer, num_nodes, is_reg)
        if config['D_model'] == 'CNN_AuxClassifier':
            self.dis = CNN_AuxClassifier(input_dim, num_class, num_domain, num_nodes)
        if config['D_model'] == 'CNN_AuxClassifier_Exp':
            self.dis = CNN_AuxClassifier_Exp(input_dim, num_class, num_domain, num_nodes)
        if config['D_model'] == 'RES_AuxClassifier':
            self.dis = RES_AuxClassifier(input_dim, num_class, num_domain, num_nodes)

        # set optimizers
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=config['G_lr'], betas=(config['G_B1'], config['G_B2']))
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=config['D_lr'], betas=(config['D_B1'], config['D_B2']))
        # if not config['skip_init']:
        #     self.gen.apply(utils.xavier_weights_init)
        #     self.dis.apply(utils.xavier_weights_init)

        self.aux_loss_func = nn.CrossEntropyLoss()
        self.sigmoid_xent = nn.BCEWithLogitsLoss()

        self.gan_loss = 0
        self.aux_loss_c = 0
        self.aux_loss_c1 = 0
        self.aux_loss_ct = 0
        self.aux_loss_d = 0
        self.aux_loss_d1 = 0
        self.aux_loss_dt = 0
        self.aux_loss_cls = 0
        self.aux_loss_cls1 = 0

    def to(self, device):
        self.gen.to(device)
        self.dis.to(device)

    def gen_update(self, x_a, y_a, config, state, device='cpu'):
        for p in self.dis.parameters():
            p.requires_grad_(False)
        self.gen.zero_grad()
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        dim_domain = config['dim_d']
        num_domain = config['num_domain']
        num_class = config['num_class']
        do_ss = config['do_ss']

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class).float()
        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain).float()

        if config['estimate'] == 'ML':
            fake_x_a = self.gen(noise, y_a_onehot, d_onehot)
        elif config['estimate'] == 'Bayesian':
            noise_d = torch.randn(num_domain, dim_domain).to(device)
            fake_x_a, KL_reg = self.gen(noise, y_a_onehot, d_onehot, noise_d)
        output_c, output_c_tw, output_d, output_d_tw, output_cls, output_disc = self.dis(fake_x_a)

        ids_s = y_a[:, 1] != num_domain - 1
        ids_t = y_a[:, 1] == num_domain - 1

        lambda_c = config['AC_weight']
        # gan_loss = self.sigmoid_xent(output_disc, torch.ones_like(output_disc, device=device))
        # gan_loss = - output_disc.mean()
        gan_loss = generator_loss(output_disc)
        aux_loss_c = self.aux_loss_func(output_c[ids_s], y_a[ids_s, 0])
        aux_loss_d = self.aux_loss_func(output_d, y_a[:, 1])
        aux_loss_ct = self.aux_loss_func(output_c_tw[ids_s], y_a[ids_s, 0])
        aux_loss_dt = self.aux_loss_func(output_d_tw, y_a[:, 1])
        aux_loss_cls = self.aux_loss_func(output_cls[ids_t], y_a[ids_t, 0])

        if state['epoch'] < config['warmup']:
            lambda_tar = 0
        else:
            lambda_tar = config['TAR_weight']

        if config['estimate'] == 'ML':
            errG = gan_loss + lambda_c * (aux_loss_c - aux_loss_ct + aux_loss_d - aux_loss_dt + lambda_tar * aux_loss_cls)
        elif config['estimate'] == 'Bayesian':
            errG = gan_loss + lambda_c * (aux_loss_c - aux_loss_ct + aux_loss_d - aux_loss_dt + lambda_tar * aux_loss_cls)\
                   + torch.dot(1.0/do_ss.to(device).squeeze(), KL_reg.squeeze())

        errG.backward()
        self.gen_opt.step()
        # self.gan_loss = gan_loss
        self.aux_loss_c = aux_loss_c
        self.aux_loss_d = aux_loss_d
        self.aux_loss_ct = aux_loss_ct
        self.aux_loss_dt = aux_loss_dt

    def dis_update(self, x_a, y_a, config, state, device='cpu'):
        for p in self.dis.parameters():
            p.requires_grad_(True)
        self.dis.zero_grad()
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        dim_domain = config['dim_d']
        num_domain = config['num_domain']
        num_class = config['num_class']

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class).float()
        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain).float()

        if config['estimate'] == 'ML':
            fake_x_a = self.gen(noise, y_a_onehot, d_onehot)
        elif config['estimate'] == 'Bayesian':
            noise_d = torch.randn(num_domain, dim_domain).to(device)
            fake_x_a, _ = self.gen(noise, y_a_onehot, d_onehot, noise_d)
        output_c, output_c_tw, output_d, output_d_tw, output_cls, output_disc = self.dis(fake_x_a.detach())
        output_c1, output_c_tw1, output_d1, output_d_tw1, output_cls1, output_disc1 = self.dis(x_a)

        ids_s = y_a[:, 1] != num_domain - 1
        ids_t = y_a[:, 1] == num_domain - 1

        if state['epoch'] < config['warmup']:
            lambda_tar = 0
        else:
            lambda_tar = config['TAR_weight']
        lambda_c = config['AC_weight']
        # gan_loss = 0.5 * (
        #         self.sigmoid_xent(output_disc1, torch.ones_like(output_disc1, device=device)) +
        #         self.sigmoid_xent(output_disc, torch.zeros_like(output_disc, device=device))
        # )
        # gan_loss = output_disc.mean() - output_disc1.mean()
        gan_loss = discriminator_loss(output_disc, output_disc1)
        aux_loss_c1 = self.aux_loss_func(output_c1[ids_s], y_a[ids_s, 0])
        aux_loss_ct = self.aux_loss_func(output_c_tw[ids_s], y_a[ids_s, 0])
        aux_loss_d1 = self.aux_loss_func(output_d1, y_a[:, 1])
        aux_loss_dt = self.aux_loss_func(output_d_tw, y_a[:, 1])
        aux_loss_cls = self.aux_loss_func(output_cls[ids_t], y_a[ids_t, 0])
        aux_loss_cls1 = self.aux_loss_func(output_cls1[ids_s], y_a[ids_s, 0])

        gradient_penalty = self.calc_gradient_penalty(x_a, fake_x_a.detach(), device=device)
        errD = gan_loss + \
               lambda_c * (aux_loss_c1 + aux_loss_d1 + aux_loss_ct + aux_loss_dt + aux_loss_cls1 + lambda_tar * aux_loss_cls) \
               + config['gp'] * gradient_penalty

        errD.backward()
        self.dis_opt.step()
        self.aux_loss_c1 = aux_loss_c1
        self.aux_loss_d1 = aux_loss_d1
        self.aux_loss_ct = aux_loss_ct
        self.aux_loss_dt = aux_loss_dt
        self.aux_loss_cls = aux_loss_cls
        self.aux_loss_cls1 = aux_loss_cls1
        self.gan_loss = gan_loss

    def calc_gradient_penalty(self, real_data, fake_data, device):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        _, _, _, _, _, disc_interpolates = self.dis(interpolates)

        # witness = torch.exp(disc_interpolates) / torch.sum(torch.exp(disc_interpolates), dim=1)
        witness = disc_interpolates
        gradients = autograd.grad(outputs=witness, inputs=interpolates,
                                  grad_outputs=torch.ones(witness.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

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


# DA_Infer trainer, deterministic/probabilistic theta encoder, implemented by ac-gan
class DA_Infer_AC_Adv(object):
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

        isProb = config['estimate'] == 'Bayesian'
        if config['G_model'] == 'MLP_Generator':
            self.gen = MLP_Generator(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden, num_layer,
                                     num_nodes, is_reg, prob=isProb)
        if config['G_model'] == 'CNN_Generator':
            self.gen = CNN_Generator(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden,
                                     num_nodes, prob=isProb)
        if config['G_model'] == 'CNN_Generator_Exp':
            self.gen = CNN_Generator_Exp(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden,
                                     num_nodes, prob=isProb)
        if config['G_model'] == 'RES_Generator':
            self.gen = RES_Generator(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden,
                                     num_nodes, prob=isProb)

        utils.seed_rng(config['seed'])
        if config['D_model'] == 'MLP_AuxClassifier':
            self.dis = MLP_AuxClassifier(input_dim, num_class, num_domain, num_layer, num_nodes, is_reg)
        if config['D_model'] == 'CNN_AuxClassifier':
            self.dis = CNN_AuxClassifier(input_dim, num_class, num_domain, num_nodes)
        if config['D_model'] == 'CNN_AuxClassifier_Exp':
            self.dis = CNN_AuxClassifier_Exp(input_dim, num_class, num_domain, num_nodes)
        if config['D_model'] == 'RES_AuxClassifier':
            self.dis = RES_AuxClassifier(input_dim, num_class, num_domain, num_nodes)

        # set optimizers
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=config['G_lr'], betas=(config['G_B1'], config['G_B2']))
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=config['D_lr'], betas=(config['D_B1'], config['D_B2']))
        # if not config['skip_init']:
        #     self.gen.apply(utils.xavier_weights_init)
        #     self.dis.apply(utils.xavier_weights_init)

        self.aux_loss_func = nn.CrossEntropyLoss()
        self.sigmoid_xent = nn.BCEWithLogitsLoss()

        self.gan_loss = 0
        self.aux_loss_c = 0
        self.aux_loss_c1 = 0
        self.aux_loss_ct = 0
        self.aux_loss_d = 0
        self.aux_loss_d1 = 0
        self.aux_loss_dt = 0
        self.aux_loss_cls = 0
        self.aux_loss_cls1 = 0

    def to(self, device):
        self.gen.to(device)
        self.dis.to(device)

    def gen_update(self, x_a, y_a, config, state, device='cpu'):
        for p in self.dis.parameters():
            p.requires_grad_(False)
        self.gen.zero_grad()
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        dim_domain = config['dim_d']
        num_domain = config['num_domain']
        num_class = config['num_class']
        do_ss = config['do_ss']

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class).float()
        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain).float()

        if config['estimate'] == 'ML':
            fake_x_a = self.gen(noise, y_a_onehot, d_onehot)
        elif config['estimate'] == 'Bayesian':
            noise_d = torch.randn(num_domain, dim_domain).to(device)
            fake_x_a, KL_reg = self.gen(noise, y_a_onehot, d_onehot, noise_d)
        output_c, output_c_tw, output_d, output_d_tw, output_cls, output_disc = self.dis(fake_x_a)

        ids_s = y_a[:, 1] != num_domain - 1
        ids_t = y_a[:, 1] == num_domain - 1

        lambda_c = config['AC_weight']
        lambda_tar = config['TAR_weight']
        # gan_loss = self.sigmoid_xent(output_disc, torch.ones_like(output_disc, device=device))
        gan_loss = generator_loss(output_disc)
        aux_loss_c = self.aux_loss_func(output_c[ids_s], y_a[ids_s, 0])
        aux_loss_d = self.aux_loss_func(output_d, y_a[:, 1])
        aux_loss_cls = self.aux_loss_func(output_cls[ids_t], y_a[ids_t, 0])

        if state['epoch'] < config['warmup']:
            lambda_tar = 0
        else:
            lambda_tar = config['TAR_weight']

        if config['estimate'] == 'ML':
            errG = gan_loss + lambda_c * (aux_loss_c + aux_loss_d + lambda_tar * aux_loss_cls)
        elif config['estimate'] == 'Bayesian':
            errG = gan_loss + lambda_c * (aux_loss_c + aux_loss_d + lambda_tar * aux_loss_cls) + torch.dot(1.0/do_ss.to(device).squeeze(), KL_reg.squeeze())

        errG.backward()
        self.gen_opt.step()
        # self.gan_loss = gan_loss
        self.aux_loss_c = aux_loss_c
        self.aux_loss_d = aux_loss_d

    def dis_update(self, x_a, y_a, config, state, device='cpu'):
        for p in self.dis.parameters():
            p.requires_grad_(True)
        self.dis.zero_grad()
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        dim_domain = config['dim_d']
        num_domain = config['num_domain']
        num_class = config['num_class']

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class).float()
        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain).float()

        if config['estimate'] == 'ML':
            fake_x_a = self.gen(noise, y_a_onehot, d_onehot)
        elif config['estimate'] == 'Bayesian':
            noise_d = torch.randn(num_domain, dim_domain).to(device)
            fake_x_a, _ = self.gen(noise, y_a_onehot, d_onehot, noise_d)
        output_c, output_c_tw, output_d, output_d_tw, output_cls, output_disc = self.dis(fake_x_a.detach())
        output_c1, output_c_tw1, output_d1, output_d_tw1, output_cls1, output_disc1 = self.dis(x_a)

        ids_s = y_a[:, 1] != num_domain - 1
        ids_t = y_a[:, 1] == num_domain - 1

        lambda_tar = config['TAR_weight']
        lambda_c = config['AC_weight']
        # gan_loss = 0.5 * (
        #         self.sigmoid_xent(output_disc1, torch.ones_like(output_disc1, device=device)) +
        #         self.sigmoid_xent(output_disc, torch.zeros_like(output_disc, device=device))
        # )
        # gan_loss = output_disc.mean() - output_disc1.mean()
        gan_loss = discriminator_loss(output_disc, output_disc1)
        aux_loss_c1 = self.aux_loss_func(output_c1[ids_s], y_a[ids_s, 0])
        aux_loss_d1 = self.aux_loss_func(output_d1, y_a[:, 1])
        aux_loss_cls = self.aux_loss_func(output_cls[ids_t], y_a[ids_t, 0])
        aux_loss_cls1 = self.aux_loss_func(output_cls1[ids_s], y_a[ids_s, 0])

        if state['epoch'] < config['warmup']:
            lambda_tar = 0
        else:
            lambda_tar = config['TAR_weight']

        gradient_penalty = self.calc_gradient_penalty(x_a, fake_x_a.detach(), device=device)
        errD = gan_loss + lambda_c * (aux_loss_c1 + aux_loss_d1 + aux_loss_cls1 + lambda_tar * aux_loss_cls) + config['gp'] * gradient_penalty

        errD.backward()
        self.dis_opt.step()
        self.aux_loss_c1 = aux_loss_c1
        self.aux_loss_d1 = aux_loss_d1
        self.aux_loss_cls = aux_loss_cls
        self.aux_loss_cls1 = aux_loss_cls1
        self.gan_loss = gan_loss

    def calc_gradient_penalty(self, real_data, fake_data, device):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        _, _, _, _, _, disc_interpolates = self.dis(interpolates)

        # witness = torch.exp(disc_interpolates) / torch.sum(torch.exp(disc_interpolates), dim=1)
        witness = disc_interpolates
        gradients = autograd.grad(outputs=witness, inputs=interpolates,
                                  grad_outputs=torch.ones(witness.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

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


# DA_Infer trainer, deterministic/probabilistic theta encoder, implemented by joint MMD
class DA_Infer_JMMD(object):
    def __init__(self, config):
        input_dim = config['idim']
        num_class = config['num_class']
        num_domain = config['num_domain']
        dim_class = config['dim_y']
        dim_domain = config['dim_d']
        dim_hidden = config['dim_z']
        G_num_layer = config['G_mlp_layers']
        G_num_nodes = config['G_mlp_nodes']
        D_num_layer = config['D_mlp_layers']
        D_num_nodes = config['D_mlp_nodes']
        is_reg = config['is_reg']

        isProb = config['estimate'] == 'Bayesian'
        if config['G_model'] == 'MLP_Generator':
            self.gen = MLP_Generator(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden, G_num_layer,
                                     G_num_nodes, is_reg, prob=isProb)
        # Seed RNG
        utils.seed_rng(config['seed'])
        if config['D_model'] == 'MLP_Classifier':
            self.dis = MLP_Classifier(input_dim, num_class, D_num_layer, D_num_nodes)

        # set optimizers
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=config['G_lr'], betas=(config['G_B1'], config['G_B2']))
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=config['D_lr'], betas=(config['D_B1'], config['D_B2']))

        self.aux_loss_func = nn.CrossEntropyLoss()
        self.mmd_loss = 0
        self.mmd_loss_s = 0
        self.mmd_loss_t = 0
        self.aux_loss_c = 0

    def to(self, device):
        self.gen.to(device)
        self.dis.to(device)

    def gen_update(self, x_a, y_a, config, state, device='cpu'):
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

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        if not is_reg:
            y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class).float()
        else:
            y_a_onehot = y_a[:, 0].view(batch_size, 1)
        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain).float()

        if config['estimate'] == 'ML':
            fake_x_a = self.gen(noise, y_a_onehot, d_onehot)
        elif config['estimate'] == 'Bayesian':
            noise_d = torch.randn(num_domain, dim_domain).to(device)
            fake_x_a, KL_reg = self.gen(noise, y_a_onehot, d_onehot, noise_d)

        ids_s = y_a[:, 1] != num_domain - 1
        ids_t = y_a[:, 1] == num_domain - 1
        # output_cr = self.dis(x_a[ids_s])
        output_cf = self.dis(fake_x_a)

        # Train mode 0: only use MMD for G
        if state['epoch'] < config['warmup'] or config['train_mode'] == 'm0':
            lambda_tar = 0
        else:
            lambda_tar = config['TAR_weight']
        # lambda_src = config['SRC_weight']
        # aux_loss_c_src = lambda_src * self.aux_loss_func(output_cr, y_a[ids_s, 0])
        # aux_loss_c = lambda_src * aux_loss_c_src + lambda_tar * aux_loss_c_tar
        aux_loss_c_tar = lambda_tar * self.aux_loss_func(output_cf[ids_t], y_a[ids_t, 0])
        aux_loss_c = lambda_tar * aux_loss_c_tar

        # sigma for MMD
        base_x = config['base_x']
        base_y = config['base_y']
        # sigma_list = [0.125, 0.25, 0.5, 1]
        sigma_list = [0.1, 0.25, 0.5, 1, 2]
        sigma_listx = [sigma * base_x for sigma in sigma_list]
        sigma_listy = [sigma * base_y for sigma in sigma_list]

        if not is_reg:
            errG_s = mix_rbf_mmd2_joint(fake_x_a[ids_s], x_a[ids_s], y_a_onehot[ids_s], y_a_onehot[ids_s],
                                        d_onehot[ids_s], d_onehot[ids_s], sigma_list=sigma_listx)
        else:
            errG_s = mix_rbf_mmd2_joint_regress(fake_x_a[ids_s], x_a[ids_s], y_a_onehot[ids_s], y_a_onehot[ids_s],
                                                d_onehot[ids_s], d_onehot[ids_s], sigma_list=sigma_listx, sigma_list1=sigma_listy)
        errG_t = mix_rbf_mmd2(fake_x_a[ids_t], x_a[ids_t], sigma_list=sigma_listx)

        lambda_c = config['AC_weight']
        if config['estimate'] == 'ML':
            errG = (num_domain-1)**2 * errG_s + errG_t + lambda_c * aux_loss_c
        elif config['estimate'] == 'Bayesian':
            errG = (num_domain-1)**2 * errG_s + errG_t + lambda_c * aux_loss_c + torch.dot(1.0 / do_ss.to(device).squeeze(), KL_reg.squeeze())

        errG.backward()
        self.gen_opt.step()
        self.mmd_loss = errG
        self.mmd_loss_s = errG_s
        self.mmd_loss_t = errG_t
        self.aux_loss_c = aux_loss_c

    def dis_update(self, x_a, y_a, config, state, device='cpu'):
        for p in self.dis.parameters():
            p.requires_grad_(True)
        self.dis.zero_grad()
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        dim_domain = config['dim_d']
        num_domain = config['num_domain']
        num_class = config['num_class']
        is_reg = config['is_reg']
        do_ss = config['do_ss']

        # generate random Gaussian noise
        noise = torch.randn(batch_size, dim_hidden).to(device)

        # create domain labels
        if not is_reg:
            y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class).float()
        else:
            y_a_onehot = y_a[:, 0].view(batch_size, 1)
        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain).float()

        if config['estimate'] == 'ML':
            fake_x_a = self.gen(noise, y_a_onehot, d_onehot)
        elif config['estimate'] == 'Bayesian':
            noise_d = torch.randn(num_domain, dim_domain).to(device)
            fake_x_a, KL_reg = self.gen(noise, y_a_onehot, d_onehot, noise_d)

        ids_s = y_a[:, 1] != num_domain - 1
        ids_t = y_a[:, 1] == num_domain - 1
        output_cr = self.dis(x_a[ids_s])
        output_cf = self.dis(fake_x_a.detach())
        lambda_tar = config['TAR_weight']
        lambda_src = config['SRC_weight']
        aux_loss_c_src = lambda_src * self.aux_loss_func(output_cr, y_a[ids_s, 0])
        aux_loss_c_tar = lambda_tar * self.aux_loss_func(output_cf[ids_t], y_a[ids_t, 0])
        aux_loss_c = lambda_src * aux_loss_c_src + lambda_tar * aux_loss_c_tar

        aux_loss_c.backward()
        self.dis_opt.step()
        self.aux_loss_c = aux_loss_c

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


# DA_Infer trainer on a graph, deterministic/probabilistic theta encoder, implemented by joint MMD
class DA_Infer_JMMD_DAG(object):
    def __init__(self, config):
        input_dim = config['idim']
        num_class = config['num_class']
        num_domain = config['num_domain']
        dim_class = config['dim_y']
        dim_domain = config['dim_d']
        dim_hidden = config['dim_z']
        G_num_layer = config['G_mlp_layers']
        G_num_nodes = config['G_mlp_nodes']
        D_num_layer = config['D_mlp_layers']
        D_num_nodes = config['D_mlp_nodes']
        is_reg = config['is_reg']
        dag_mat_file = join(config['data_root'], config['dataset'], config['dag_mat_file'])
        npzfile = np.load(dag_mat_file)
        dag_mat = npzfile['mat']

        isProb = config['estimate'] == 'Bayesian'
        if config['G_model'] == 'DAG_Generator':
            self.gen = DAG_Generator(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden, G_num_layer,
                                     G_num_nodes, is_reg, dag_mat, prob=isProb)
        if config['G_model'] == 'PDAG_Generator':
            self.gen = PDAG_Generator(input_dim, num_class, num_domain, dim_class, dim_domain, dim_hidden, G_num_layer,
                                     G_num_nodes, is_reg, dag_mat, prob=isProb)

        utils.seed_rng(config['seed'])
        if config['D_model'] == 'MLP_Classifier':
            self.dis = MLP_Classifier(input_dim, num_class, D_num_layer, D_num_nodes)

        # set optimizer
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=config['G_lr'],
                                        betas=(config['G_B1'], config['G_B2']))
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=config['D_lr'],
                                        betas=(config['D_B1'], config['D_B2']))
        # if not config['skip_init']:
        #     self.gen.apply(utils.xavier_weights_init)
        #     self.dis.apply(utils.xavier_weights_init)

        self.aux_loss_func = nn.CrossEntropyLoss()
        self.mmd_loss = 0
        self.mmd_loss_s = 0
        self.mmd_loss_t = 0
        self.aux_loss_c = 0

    def to(self, device):
        self.gen.to(device)
        self.dis.to(device)

    # separate training of each module, trained on source + target domain together
    def gen_update(self, x_a, y_a, config, state, device='cpu'):
        # for p in self.dis.parameters():
        #     p.requires_grad_(False)
        self.gen.zero_grad()
        self.dis.zero_grad()
        input_dim = config['idim']
        dim_domain = config['dim_d']
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        num_domain = config['num_domain']
        num_class = config['num_class']
        is_reg = config['is_reg']
        do_ss = config['do_ss']

        # generate random Gaussian noise
        if dim_hidden != 0:
            noise = torch.randn((batch_size, dim_hidden * input_dim), device=device)

        # create domain labels
        if not is_reg:
            y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class).float()
        else:
            y_a_onehot = y_a[:, 0].view(batch_size, 1)

        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain).float()

        if config['estimate'] == 'ML':
            fake_x_a = self.gen.forward_indep(noise, y_a_onehot, d_onehot, x_a, device=device)
            fake_x_a_cls = self.gen(noise, y_a_onehot, d_onehot, device=device)
        elif config['estimate'] == 'Bayesian':
            noise_d = torch.randn(num_domain, dim_domain * input_dim).to(device)
            fake_x_a, KL_reg = self.gen.forward_indep(noise, y_a_onehot, d_onehot, x_a, device=device, noise_d=noise_d)
            fake_x_a_cls = self.gen(noise, y_a_onehot, d_onehot, device=device, noise_d=noise_d)

        # sigma for MMD
        base_x = config['base_x']
        sigma_list = [0.125, 0.25, 0.5, 1]
        # sigma_list = [0.25, 0.5, 1]
        sigma_listx = [sigma * base_x for sigma in sigma_list]

        ids_s = y_a[:, 1] != num_domain - 1
        ids_t = y_a[:, 1] == num_domain - 1

        # Train mode 0: only use MMD for G
        if config['train_mode'] == 'm0':  # no bp from classifier C to G
            output_cf = self.dis(fake_x_a_cls.detach())
            aux_loss_c = self.aux_loss_func(output_cf[ids_t], y_a[ids_t, 0])

        if config['train_mode'] == 'm1':  # bp from classifier C to G
            output_cr = self.dis(x_a[ids_s])
            output_cf = self.dis(fake_x_a_cls)
            lambda_src = config['SRC_weight']
            if state['epoch'] < config['warmup']:
                lambda_tar = 0
            else:
                lambda_tar = config['TAR_weight']
            aux_loss_c_src = lambda_src * self.aux_loss_func(output_cr, y_a[ids_s, 0])
            aux_loss_c_tar = lambda_tar * self.aux_loss_func(output_cf[ids_t], y_a[ids_t, 0])
            aux_loss_c = aux_loss_c_src + aux_loss_c_tar

        # MMD matching for each factor
        batch_size_s = len(y_a[ids_s, :])
        # batch_size_t = len(y_a[ids_t, :])
        errG_s = torch.zeros(len(self.gen.nodeSort), device=device)
        # errG_t = torch.zeros(len(self.gen.nodeSort), device=device)

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
                errG_s[i] = mix_rbf_mmd2_joint(fake_x_a[ids_s, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                             x_a[ids_s, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                             y_a_onehot[ids_s], y_a_onehot[ids_s], d_onehot[ids_s],
                                             d_onehot[ids_s], input_p[ids_s], input_p[ids_s], sigma_list=sigma_listx)

                # errG_t[i] = mix_rbf_mmd2_joint_regress(fake_x_a[ids_t, self.gen.nodesA[i]].view(batch_size_t, output_dim),
                #                                      x_a[ids_t, self.gen.nodesA[i]].view(batch_size_t, output_dim),
                #                                      input_p[ids_t], input_p[ids_t], sigma_list=sigma_listx, sigma_list1=sigma_listx)
            else:
                if not self.gen.ischain:
                    output_dim = 1
                else:
                    output_dim = len(self.gen.nodesA[i])
                errG_s[i] = mix_rbf_mmd2_joint(fake_x_a[ids_s][:, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                             x_a[ids_s][:, self.gen.nodesA[i]].view(batch_size_s, output_dim),
                                             y_a_onehot[ids_s], y_a_onehot[ids_s], d_onehot[ids_s],
                                             d_onehot[ids_s], sigma_list=sigma_listx)
                # errG_t[i] = mix_rbf_mmd2(fake_x_a[ids_t][:, self.gen.nodesA[i]].view(batch_size_t, output_dim),
                #                        x_a[ids_t][:, self.gen.nodesA[i]].view(batch_size_t, output_dim), sigma_list=sigma_listx)

        errG_t = mix_rbf_mmd2(fake_x_a_cls[ids_t], x_a[ids_t], sigma_list=sigma_listx)

        errG_s = errG_s.mean()
        # errG_t = errG_t.mean()

        lambda_c = config['AC_weight']
        if config['estimate'] == 'ML':
            errG = errG_s + errG_t + lambda_c * aux_loss_c
        elif config['estimate'] == 'Bayesian':
            errG = errG_s + errG_t + lambda_c * aux_loss_c + torch.dot(1.0 / do_ss.to(device).squeeze(), KL_reg.squeeze())

        errG.backward()
        self.gen_opt.step()
        self.dis_opt.step()
        self.mmd_loss = errG
        self.mmd_loss_s = errG_s
        self.mmd_loss_t = errG_t
        self.aux_loss_c = aux_loss_c

    def dis_update(self, x_a, y_a, config, state, device='cpu'):
        for p in self.dis.parameters():
            p.requires_grad_(True)
        self.dis.zero_grad()
        input_dim = config['idim']
        batch_size = config['batch_size']
        dim_hidden = config['dim_z']
        dim_domain = config['dim_d']
        num_domain = config['num_domain']
        num_class = config['num_class']
        is_reg = config['is_reg']
        do_ss = config['do_ss']

        # generate random Gaussian noise
        if dim_hidden != 0:
            noise = torch.randn((batch_size, dim_hidden * input_dim), device=device)

        # create domain labels
        if not is_reg:
            y_a_onehot = torch.nn.functional.one_hot(y_a[:, 0], num_class).float()
        else:
            y_a_onehot = y_a[:, 0].view(batch_size, 1)

        d_onehot = torch.nn.functional.one_hot(y_a[:, 1], num_domain).float()

        if config['estimate'] == 'ML':
            fake_x_a_cls = self.gen(noise, y_a_onehot, d_onehot, device=device)
        elif config['estimate'] == 'Bayesian':
            noise_d = torch.randn(num_domain, dim_domain * input_dim).to(device)
            fake_x_a_cls = self.gen(noise, y_a_onehot, d_onehot, device=device, noise_d=noise_d)

        ids_s = y_a[:, 1] != num_domain - 1
        ids_t = y_a[:, 1] == num_domain - 1
        output_cr = self.dis(x_a[ids_s])
        output_cf = self.dis(fake_x_a_cls.detach())
        lambda_src = config['SRC_weight']
        lambda_tar = config['TAR_weight']
        aux_loss_c_src = lambda_src * self.aux_loss_func(output_cr, y_a[ids_s, 0])
        aux_loss_c_tar = lambda_tar * self.aux_loss_func(output_cf[ids_t], y_a[ids_t, 0])
        aux_loss_c = aux_loss_c_src + aux_loss_c_tar
        aux_loss_c.backward()
        self.dis_opt.step()
        self.aux_loss_c = aux_loss_c

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
