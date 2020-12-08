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

def hinge_multi(prob,y):

    len = prob.size()[0]

    index_list = [[],[]]

    for i in range(len):
        index_list[0].append(i)
        index_list[1].append(np.asscalar(y[i].cpu().detach().numpy()))

    prob_choose = prob[index_list]
    prob_choose = (prob_choose.squeeze()).unsqueeze(dim=1)

    loss = ((1-prob_choose+prob).clamp(min=0)).mean()

    return loss


def GAN_training_function(G, D, GD, z_, y_, yd_, ema, state_dict, config):
  def train(x_s, y, yd):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    y = y.long()
    yd = yd.long()
    x_s = torch.split(x_s, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    yd = torch.split(yd, config['batch_size'])
    counter = 0
    
    # Optionally toggle D and G's "require_grad"

    utils.toggle_grad(D, True)
    utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        yd_.sample_()



        D_fake, D_real, mi, c_cls, mid, c_clsd, G_z = GD(z_, y_, yd_,
                            x_s[counter], y[counter], yd[counter], train_G=False,
                            split_D=config['split_D'],return_G_z=True)

        D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)

        C_loss = 0

        if config['AC']:
            fake_mi = mi[:D_fake.shape[0]]
            fake_cls = c_cls[:D_fake.shape[0]]
            c_cls_rs = c_cls[D_fake.shape[0]:]

            fake_mid = mid[:D_fake.shape[0]]
            c_clsd = c_clsd[D_fake.shape[0]:]
            # print(yd)
            # print(yd_)

            if config['loss_type'] == 'Twin_AC':
                C_loss += F.cross_entropy(c_clsd, yd[counter]) + F.cross_entropy(fake_mid, yd_) + \
                          0.5*F.cross_entropy(c_cls_rs[yd[counter]!=0], y[counter][yd[counter]!=0]) + 0.5*F.cross_entropy(fake_cls, y_) + 1.0*F.cross_entropy(fake_mi, y_)
                # if state_dict['itr'] > 0000:
                #     C_loss += 0.2*F.cross_entropy(c_cls_ft, y_[yd_!=0]) + 0.2*F.cross_entropy(fake_mi_t[yd_!=0], y_[yd_!=0])#F.cross_entropy(fake_mi[yd_ == 0], y_[yd_ == 0])

            if config['loss_type'] == 'AC':
                C_loss += F.cross_entropy(c_cls_fs, y_f_s) + F.cross_entropy(c_clsd, yd)


        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations

        if config['Pac']:
            x_pack = torch.cat([x_s[counter],x_t[counter]],dim=0)
            T_img = x_pack.view(-1, 4 * x_pack.size()[1], x_pack.size()[2], x_pack.size()[3])
            F_img = G_z.view(-1, 4 * G_z.size()[1], G_z.size()[2], G_z.size()[3])
            pack_img = torch.cat([T_img, F_img], dim=0)
            pack_out, _, _ = D(pack_img, pack=True)
            D_real_pac = pack_out[:T_img.size()[0]]
            D_fake_pac = pack_out[T_img.size()[0]:]
            D_loss_real_pac, D_loss_fake_pac = losses.discriminator_loss(D_fake_pac, D_real_pac)
            D_loss_real += D_loss_real_pac
            D_loss_fake += D_loss_fake_pac

        D_loss = (D_loss_real + D_loss_fake + C_loss*config['AC_weight']) / float(config['num_D_accumulations'])
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
            yd_.sample_()
            D_fake, mi,cls, mid, clsd, G_z = GD(z_, y_, yd_, train_G=True, split_D=config['split_D'], return_G_z=True)

            C_loss = 0
            MI_loss = 0
            CD_loss = 0
            MID_loss = 0
            G_loss = losses.generator_loss(D_fake)
            if config['loss_type'] == 'AC' or config['loss_type'] == 'Twin_AC':
                C_loss = 1.0*F.cross_entropy(cls, y_) #+ 0.5*F.cross_entropy(cls[yd_!=0], y_[yd_!=0])
                CD_loss = F.cross_entropy(clsd,yd_)
                if config['loss_type'] == 'Twin_AC':
                    MI_loss = 1.0*F.cross_entropy(mi, y_)
                    # if state_dict['itr'] > 0000:
                    #     MI_loss += 0.5*F.cross_entropy(mi_t[yd_!=0], y_[yd_!=0])
                    MID_loss = F.cross_entropy(mid,yd_)

            if config['Pac']:
                F_img = G_z.view(-1, 4 * G_z.size()[1], G_z.size()[2], G_z.size()[3])
                D_fake_pac, _, _ = D(F_img, pack=True)
                G_loss_pac = losses.generator_loss(D_fake_pac)
                G_loss += G_loss_pac

            G_loss = G_loss / float(config['num_G_accumulations'])
            C_loss = C_loss / float(config['num_G_accumulations'])
            MI_loss = MI_loss / float(config['num_G_accumulations'])
            CD_loss = CD_loss / float(config['num_G_accumulations'])
            MID_loss = MID_loss / float(config['num_G_accumulations'])
            (G_loss + (C_loss - MI_loss  + CD_loss  - MID_loss)*config['AC_weight']).backward()

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
            'MI_loss': MI_loss,
            'CD_loss': CD_loss,
            'MID_loss': MID_loss}
    # Return G's loss and the components of D's loss.
    return out
  return train
  
''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''
def save_and_sample(G, D, G_ema, z_, y_,yd_, fixed_z, fixed_y, fixed_yd,
                    state_dict, config, experiment_name):
  utils.save_weights(G, D, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None)
  # Save an additional copy to mitigate accidental corruption if process
  # is killed during a save (it's happened to me before -.-)
  if config['num_save_copies'] > 0:
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name,
                       'copy%d' %  state_dict['save_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']
    
  # Use EMA G for samples or non-EMA?
  which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  # Accumulate standing statistics?
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_,yd_, config['n_classes'],
                           config['num_standing_accumulations'])
  
  # Save a random sample sheet with fixed z and y
  which_G.eval()
  # with torch.no_grad():
  #   if config['parallel']:
  #     fixed_yd[:]=0
  #     fixed_Gzs =  nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y),which_G.shared_d(fixed_yd)))
  #     fixed_yd[:] = 1
  #     fixed_Gzd = nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y), which_G.shared_d(fixed_yd)))
  #   else:
  #     fixed_yd[:] = 0
  #     fixed_Gzs = which_G(fixed_z, which_G.shared(fixed_y),which_G.shared_d(fixed_yd))
  #     fixed_yd[:] = 1
  #     fixed_Gzd1 = which_G(fixed_z, which_G.shared(fixed_y),which_G.shared_d(fixed_yd))
  #     fixed_yd[:] = 2
  #     fixed_Gzd2 = which_G(fixed_z, which_G.shared(fixed_y), which_G.shared_d(fixed_yd))
  #     fixed_yd[:] = 3
  #     fixed_Gzd3 = which_G(fixed_z, which_G.shared(fixed_y), which_G.shared_d(fixed_yd))
  # if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
  #   os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
  #
  # image_filename = '%s/%s/source_fixed_samples%d.jpg' % (config['samples_root'],
  #                                                 experiment_name,
  #                                                 state_dict['itr'])
  # torchvision.utils.save_image(fixed_Gzs.float().cpu(), image_filename,
  #                            nrow=int(fixed_Gzs.shape[0] **0.5), normalize=True)
  #
  # ####################################################
  #
  # image_filename = '%s/%s/target_fixed_samples%d.jpg' % (config['samples_root'],
  #                                                        experiment_name,
  #                                                        state_dict['itr'])
  # torchvision.utils.save_image(fixed_Gzd1.float().cpu(), image_filename,
  #                              nrow=int(fixed_Gzd.shape[0] ** 0.5), normalize=True)
  # For now, every time we save, also save sample sheets
  utils.sample_sheet(which_G,
                     classes_per_sheet=10,
                     num_classes=config['n_classes'],
                     num_domain=config['n_domain'],
                     samples_per_class=10, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['itr'],
                     z_=z_)
  #
  # utils.sample_sheet_inter(which_G,
  #                    classes_per_sheet=9,
  #                    num_classes=config['n_classes'],
  #                    num_domain=config['n_domain'],
  #                    samples_per_class=9, parallel=config['parallel'],
  #                    samples_root=config['samples_root'],
  #                    experiment_name=experiment_name,
  #                    folder_number=state_dict['itr'],
  #                    z_=z_)

  # utils.interp_sheet(which_G,
  #                    num_per_sheet=16,
  #                    num_midpoints=8,
  #                    num_classes=config['n_classes'],
  #                    parallel=config['parallel'],
  #                    samples_root=config['samples_root'],
  #                    experiment_name=experiment_name,
  #                    folder_number=state_dict['itr'],
  #                    sheet_number=0,
  #                    fix_z=True, fix_y=True, fix_yd=False, device='cuda')


  which_G.train()


  
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
  print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
  # If improved over previous best metric, save approrpiate copy
  if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
    or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
    print('%s improved over previous best, saving checkpoint...' % config['which_best'])
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, 'best%d' % state_dict['save_best_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
  state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
  state_dict['best_FID'] = min(state_dict['best_FID'], FID)
  # Log results to file
  test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
               IS_std=float(IS_std), FID=float(FID))