"""
Domain Adaptation as a Problem of Inference on Graphical Models
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
from trainer import *
import torchvision.transforms as transforms


def test_acc(model, test_loader, device):

    model.eval()
    with torch.no_grad():
        test_loss_ac = 0  # classifier trained on all domains for minCh
        correct_ac = 0
        test_loss_tac = 0  # classifier trained on the target domain
        correct_tac = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).view(data.size(0), 2)
            # output_tac, _, _, _, output_ac = model(data) # output_tac is actually output_cls !!
            # _, _, _, _, output_ac, _ = model(data) # output_tac is actually output_cls !!
            _, _, _, _, output_ac, _ = model(data) # output_tac is actually output_cls !!
            test_loss_ac += F.nll_loss(output_ac, target[:, 0]).sum().item()  # sum up batch loss
            pred_ac = output_ac.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_ac += pred_ac.eq(target[:, 0].view_as(pred_ac)).sum().item()
            # test_loss_tac += F.nll_loss(output_tac, target[:, 0]).sum().item()  # sum up batch loss
            # pred_tac = output_tac.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # correct_tac += pred_tac.eq(target[:, 0].view_as(pred_tac)).sum().item()

    test_loss_ac /= len(test_loader.dataset)
    # test_loss_tac /= len(test_loader.dataset)
    print('\nTest set ac: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss_ac, correct_ac, len(test_loader.dataset),
        100. * correct_ac / len(test_loader.dataset) * 1.0))
    # print('\nTest set tac: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss_tac, correct_tac, len(test_loader.dataset),
    #     100. * correct_tac / len(test_loader.dataset) * 1.0))

    model.train()

    # return correct_ac / len(test_loader.dataset)*1.0, correct_tac / len(test_loader.dataset)*1.0
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
    num_class = config['num_class']
    num_domain = config['num_domain']
    dim_z = config['dim_z']

    state_dict = {'epoch': 0, 'iterations': 0, 'best_score': 0, 'final_score': 0, 'config': config}

    if config['trainer'] == 'DA_Infer_TAC_Adv':
        trainer = DA_Infer_TAC_Adv(config)
    if config['trainer'] == 'DA_Infer_AC_Adv':
        trainer = DA_Infer_AC_Adv(config)
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

    # get sample size in each domain
    do_ss = torch.zeros((num_domain, 1))
    for do_i in range(num_domain):
        do_ss[do_i] = torch.Tensor([(train_loader.dataset.labels[:, 1] == do_i).sum()]).item()
    config['do_ss'] = do_ss

    # fixed noise for illustration
    fixed_noise = torch.randn(100, dim_z, device=device)
    label_gpu = torch.zeros(100, device=device, dtype=torch.int64)
    for i in range(num_class):
        label_gpu[i * num_class:(i + 1) * num_class] = i
    label_fixed = label_gpu.repeat(num_domain)
    y_onehot = torch.nn.functional.one_hot(label_fixed, num_class).float()
    label_fixed = y_onehot
    fixed_noise = fixed_noise.repeat(num_domain, 1)

    # training
    Diters = config['num_D_steps']
    Giters = config['num_G_steps']
    Diter = 0
    Giter = Giters
    best_score = state_dict['best_score']
    for ep in range(state_dict['epoch'], config['num_epochs']):
        state_dict['epoch'] = ep

        # test_acc_target_c = test_acc(trainer.dis, test_loader, device=device)
        # writer.add_scalar('test_acc_target_ac', test_acc_target_c, ep)
        # # writer.add_scalar('test_acc_target_tac', test_acc_target_ct, ep)
        # if test_acc_target_c > best_score:
        #     best_score = test_acc_target_c
        #     state_dict['best_score'] = best_score
        # if ep == config['num_epochs'] - 1:
        #     state_dict['final_score'] = test_acc_target_c

        for it, (x, y) in enumerate(train_loader):
            if x.size(0) != batch_size:
                continue
            trainer.gen.train()
            trainer.dis.train()
            x = x.to(device)
            y = y.to(device).view(x.size(0), 2)

            if Diter < Diters:
                trainer.dis_update(x, y, config, state_dict, device)
                Diter += 1
                if Diter == Diters:
                    Giter = 0

            if Giter < Giters and Diter > Diters:
                trainer.gen_update(x, y, config, state_dict, device)
                Giter += 1
                if Giter == Giters:
                    Diter = 0

            if Diter == Diters:
                Diter += 1

            # Dump training stats in log file
            if (iterations + 1) % config['save_every'] == 0:
                writer.add_scalar('gan_loss', trainer.gan_loss, iterations)
                writer.add_scalar('aux_loss_c_fake', trainer.aux_loss_c, iterations)
                writer.add_scalar('aux_loss_c_real', trainer.aux_loss_c1, iterations)
                writer.add_scalar('aux_loss_ct', trainer.aux_loss_ct, iterations)
                writer.add_scalar('aux_loss_d_fake', trainer.aux_loss_d, iterations)
                writer.add_scalar('aux_loss_d_real', trainer.aux_loss_d1, iterations)
                writer.add_scalar('aux_loss_dt', trainer.aux_loss_dt, iterations)
                writer.add_scalar('aux_loss_cls_real', trainer.aux_loss_cls1, iterations)
                writer.add_scalar('aux_loss_cls_fake', trainer.aux_loss_cls, iterations)
            if (iterations + 1) % config['display_every'] == 0:
                print("Epoch: %04d, Iteration: %08d, gan loss: %.2f, ac_fake: %.2f, ac_real: %.2f, ac_twin: %.2f, ad_fake: %.2f, "
                      "ad_real: %.2f, ad_twin: %.2f, cls_real: %.2f, cls_fake: %.2f"
                      % (ep+1, iterations + 1, trainer.gan_loss, trainer.aux_loss_c, trainer.aux_loss_c1, trainer.aux_loss_ct,
                         trainer.aux_loss_d, trainer.aux_loss_d1, trainer.aux_loss_dt, trainer.aux_loss_cls1, trainer.aux_loss_cls))

            # Save network weights
            if (iterations + 1) % config['save_every'] == 0:
                trainer.save(os.path.join(config['weights_root'], experiment_name), state_dict)

            iterations += 1
            state_dict['iterations'] = iterations

        # print variational parameters
        if config['estimate'] == 'Bayesian':
            print(trainer.gen.mu.squeeze())
            print(trainer.gen.sigma.squeeze())

        # save sample images
        one = torch.ones(100, 1, device=device, dtype=torch.int64)
        label_d = one * 0
        for i in range(1, num_domain):
            label_d = torch.cat((label_d, one * i))
        label_d_onehot = torch.nn.functional.one_hot(label_d.squeeze(), num_domain).float()
        trainer.gen.eval()
        with torch.no_grad():
            if config['estimate'] == 'ML':
                fake_img = trainer.gen(fixed_noise, label_fixed, label_d_onehot)
            elif config['estimate'] == 'Bayesian':
                noise_d = torch.randn(num_domain, config['dim_d']).to(device)
                fake_img, _ = trainer.gen(fixed_noise, label_fixed, label_d_onehot, noise_d)
        img_name = os.path.join(config['samples_root'], experiment_name + '_gen.jpg')
        torchvision.utils.save_image(fake_img.mul(0.5).add(0.5), img_name, nrow=10)
        trainer.gen.train()


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
