''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os
import torch.nn.functional as F

import utils
import losses
import numpy as np


# Dummy training function for debugging
def dummy_training_function():
    def train(x, y):
        return {}

    return train


def hinge_multi(prob, y):
    len = prob.size()[0]

    index_list = [[], []]

    for i in range(len):
        index_list[0].append(i)
        index_list[1].append(np.asscalar(y[i].cpu().detach().numpy()))

    prob_choose = prob[index_list]
    prob_choose = (prob_choose.squeeze()).unsqueeze(dim=1)

    loss = ((1 - prob_choose + prob).clamp(min=0)).mean()

    return loss


def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"

        utils.toggle_grad(D, True)
        utils.toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                _, _, mi, c_cls, G_z = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                          x[counter], y[counter], train_G=False,
                                          split_D=config['split_D'], return_G_z=True)

                T_img = x[counter].view(-1, 4 * x[counter].size()[1], x[counter].size()[2], x[counter].size()[3])
                F_img = G_z.view(-1, 4 * G_z.size()[1], G_z.size()[2], G_z.size()[3])
                pack_img = torch.cat([T_img, F_img], dim=0)
                pack_out, _, _ = D(pack_img, pack=True)
                D_real = pack_out[:T_img.size()[0]]
                D_fake = pack_out[T_img.size()[0]:]

                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
                C_loss = 0
                if config['loss_type'] == 'Twin_AC':
                    C_loss += F.cross_entropy(c_cls[G_z.shape[0]:], y[counter]) + F.cross_entropy(mi[:G_z.shape[0]], y_)
                if config['loss_type'] == 'Twin_AC_M':
                    C_loss += hinge_multi(c_cls[G_z.shape[0]:], y[counter]) + hinge_multi(mi[:G_z.shape[0]], y_)
                if config['loss_type'] == 'AC':
                    C_loss += F.cross_entropy(c_cls[G_z.shape[0]:], y[counter])
                D_loss = (D_loss_real + D_loss_fake + C_loss * config['AC_weight']) / float(
                    config['num_D_accumulations'])
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        utils.toggle_grad(D, False)
        utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()
        for step_index in range(config['num_G_steps']):
            for accumulation_index in range(config['num_G_accumulations']):
                z_.sample_()
                y_.sample_()
                _, mi, c_cls, G_z = GD(z_, y_, train_G=True, split_D=config['split_D'], return_G_z=True)
                F_img = G_z.view(-1, 4 * G_z.size()[1], G_z.size()[2], G_z.size()[3])
                D_fake, _, _ = D(F_img, pack=True)

                C_loss = 0
                MI_loss = 0
                if config['loss_type'] == 'AC' or config['loss_type'] == 'Twin_AC':
                    C_loss = F.cross_entropy(c_cls, y_)
                    if config['loss_type'] == 'Twin_AC':
                        MI_loss = F.cross_entropy(mi, y_)
                if config['loss_type'] == 'Twin_AC_M':
                    C_loss = hinge_multi(c_cls, y_)
                    MI_loss = hinge_multi(mi, y_)

                G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])
                C_loss = C_loss / float(config['num_G_accumulations'])
                MI_loss = MI_loss / float(config['num_G_accumulations'])
                (G_loss + (C_loss - MI_loss) * config['AC_weight']).backward()

            # Optionally apply modified ortho reg in G
            if config['G_ortho'] > 0.0:
                print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
                # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
                utils.ortho(G, config['G_ortho'],
                            blacklist=[param for param in G.shared.parameters()])
            G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item()),
               'C_loss': C_loss,
               'MI_loss': MI_loss}
        # Return G's loss and the components of D's loss.
        return out

    return train


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''


def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                    state_dict, config, experiment_name):
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, None, G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name,
                           'copy%d' % state_dict['save_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_num'] = (state_dict['save_num'] + 1) % config['num_save_copies']

    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    # Accumulate standing statistics?
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])

    # Save a random sample sheet with fixed z and y
    with torch.no_grad():
        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
        else:
            fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    utils.sample_sheet(which_G,
                       classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                       num_classes=config['n_classes'],
                       samples_per_class=10, parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       z_=z_)
    # Also save interp sheets
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        utils.interp_sheet(which_G,
                           num_per_sheet=16,
                           num_midpoints=8,
                           num_classes=config['n_classes'],
                           parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=experiment_name,
                           folder_number=state_dict['itr'],
                           sheet_number=0,
                           fix_z=fix_z, fix_y=fix_y, device='cuda')


''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''


def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
    print('Gathering inception metrics...')
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])
    IS_mean, IS_std, FID = get_inception_metrics(sample,
                                                 config['num_inception_images'],
                                                 num_splits=10)
    print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (
    state_dict['itr'], IS_mean, IS_std, FID))
    # If improved over previous best metric, save approrpiate copy
    if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
            or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
        print('%s improved over previous best, saving checkpoint...' % config['which_best'])
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name, 'best%d' % state_dict['save_best_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num'] = (state_dict['save_best_num'] + 1) % config['num_best_copies']
    state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
                 IS_std=float(IS_std), FID=float(FID))