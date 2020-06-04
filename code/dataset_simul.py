import torch.utils.data as data
import torch
import os
from os.path import join
import numpy as np
import numpy.matlib
from models import Linear_Generator
import os
import utils


class dataset_simul_linear(data.Dataset):
    def __init__(self, specs):
        super(dataset_simul_linear, self).__init__()
        self.root = specs['root']
        self.id = specs['seed']
        self.num_class = specs['num_class']
        self.num_domain = specs['num_domain']
        self.sample_size = specs['num_train']
        self.dim = specs['dim']
        self.dim_d = specs['dim_d']
        full_path = join(self.root, str(self.id)+'_'+str(self.dim_d))
        device = torch.device('cpu')

        if not os.path.exists(self.root):
            os.mkdir(self.root)

        # extract data
        if os.path.isfile(full_path + '.npz'):
            npzfile = np.load(full_path + '.npz')
            x = npzfile['x']
            y = npzfile['y']
        else:
            # generate domain labels
            label_d = np.repeat(np.arange(self.num_domain), self.sample_size).T.reshape(self.sample_size * self.num_domain, 1)
            label_y = np.matlib.repmat(np.repeat(np.arange(self.num_class), self.sample_size / self.num_class), 1, self.num_domain).T
            noise = torch.randn(self.sample_size * self.num_domain, self.dim).to(device)

            label_d_tensor = torch.LongTensor(label_d)
            label_y_tensor = torch.LongTensor(label_y)

            # one-hot
            label_y_onehot = torch.zeros(self.sample_size * self.num_domain, self.num_class, device=device)
            label_y_onehot.scatter_(1, label_y_tensor.view(self.sample_size * self.num_domain, 1), 1)
            label_d_onehot = torch.zeros(self.sample_size * self.num_domain, self.num_domain, device=device)
            label_d_onehot.scatter_(1, label_d_tensor.view(self.sample_size * self.num_domain, 1), 1)

            # merge domain, label, and noise
            net = Linear_Generator(self.dim, self.num_class, self.num_domain, 2, self.dim_d, 1, 64).to(device)
            net.lc.weight.data.normal_(0, 2)
            net.ld.weight.data.normal_(0, 2)
            net.le.weight.data.normal_(0, 2)
            x = net.forward(noise, label_y_onehot, label_d_onehot, device)
            x = x.cpu().detach().numpy()
            y = np.concatenate((label_y, label_d), 1)

            # save data
            np.savez(full_path, x=x, y=y, lc=net.lc.weight.data, ld=net.ld.weight.data, le=net.le.weight.data)

        self.num = len(x)
        self.data = x
        self.labels = y

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        label = torch.LongTensor([np.int64(labels)])
        return np.float32(img), label

    def __len__(self):
        return self.num


class dataset_simul_mog(data.Dataset):
    def __init__(self, specs):
        super(dataset_simul_mog, self).__init__()
        self.root = specs['root']
        self.id = specs['seed']
        self.num_class = specs['num_class']
        self.num_domain = specs['num_domain']
        self.sample_size = specs['num_train']
        self.dim = specs['dim']
        full_path = join(self.root, str(self.id))

        # extract data

        # if not exist, generate data
        for l in range(self.num_domain):
            mean1 = [0, 0]
            mean2 = [4, 0]
            cov = [[1, 0], [0, 1]]
            x1 = np.random.multivariate_normal(mean1, cov, self.sample_size/2)
            x2 = np.random.multivariate_normal(mean2, cov, self.sample_size/2)
            x1 = x1 + np.ones((self.sample_size/2, 2)) * l * 4
            x2 = x2 + np.ones((self.sample_size/2, 2)) * l * 4
            y1 = np.zeros((self.sample_size/2, 1))
            y2 = np.ones((self.sample_size/2, 1))
            x = np.concatenate((x1, x2), 0)
            y = np.concatenate((y1, y2), 0)
            y = np.stack((y.squeeze(), np.repeat(l, y.shape[0])), 1)
            if l == 0:
                x_a = x
                y_a = y
            else:
                x_a = np.concatenate((x_a, x), 0)
                y_a = np.concatenate((y_a, y), 0)

        np.savez(full_path, x_a, y_a)
        self.num = len(x_a)
        self.data = x_a
        self.labels = y_a

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        label = torch.LongTensor([np.int64(labels)])
        return np.float32(img), label

    def __len__(self):
        return self.num


class DatasetSimuDAG2(data.Dataset):
    def __init__(self, specs):
        super(DatasetSimuDAG2, self).__init__()
        self.root = specs['root']
        self.id = specs['seed']
        self.train = specs['train']
        self.num_class = specs['num_class']
        self.num_domain = specs['num_domain']
        self.sample_size = specs['num_train']
        self.dim = specs['dim']
        self.dagMatFile = specs['dag_mat_file']
        full_path = join(self.root, 'to%d_numData%d' % (self.id, self.sample_size))

        if os.path.isfile(full_path+'.npz'):
            npzfile = np.load(full_path+'.npz')
            x = npzfile['x']
            y = npzfile['y']

        self.test_id = self.num_domain - 1

        if self.train:
            self.data = x
            self.labels = y
        else:
            self.data = x[y[:, 1] == self.test_id, :]
            self.labels = y[y[:, 1] == self.test_id, :]

        self.num = len(self.labels)

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        label = torch.LongTensor([np.int64(labels)])
        return np.float32(img), label

    def __len__(self):
        return self.num


class DatasetSimuDAG5(data.Dataset):
    def __init__(self, specs):
        super(DatasetSimuDAG5, self).__init__()
        self.root = specs['root']
        self.id = specs['seed']
        self.train = specs['train']
        self.num_class = specs['num_class']
        self.num_domain = specs['num_domain']
        self.sample_size = specs['num_train']
        self.dim = specs['dim']
        self.dagMatFile = specs['dag_mat_file']
        full_path = join(self.root, 'to%d_numData%d' % (self.id, self.sample_size))

        if os.path.isfile(full_path + '.npz'):
            npzfile = np.load(full_path + '.npz')
            x = npzfile['x']
            y = npzfile['y']

        self.test_id = self.num_domain - 1

        if self.train:
            self.data = x
            self.labels = y
        else:
            self.data = x[y[:, 1] == self.test_id, :]
            self.labels = y[y[:, 1] == self.test_id, :]

        self.num = len(self.labels)

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        label = torch.LongTensor([np.int64(labels)])
        return np.float32(img), label

    def __len__(self):
        return self.num


class DatasetSimuDAG9(data.Dataset):
    def __init__(self, specs):
        super(DatasetSimuDAG9, self).__init__()
        self.root = specs['root']
        self.id = specs['seed']
        self.train = specs['train']
        self.num_class = specs['num_class']
        self.num_domain = specs['num_domain']
        self.sample_size = specs['num_train']
        self.dim = specs['dim']
        self.dagMatFile = specs['dag_mat_file']
        full_path = join(self.root, 'to%d_numData%d' % (self.id, self.sample_size))

        if os.path.isfile(full_path + '.npz'):
            npzfile = np.load(full_path + '.npz')
            x = npzfile['x']
            y = npzfile['y']

        self.test_id = self.num_domain - 1

        if self.train:
            self.data = x
            self.labels = y
        else:
            self.data = x[y[:, 1] == self.test_id, :]
            self.labels = y[y[:, 1] == self.test_id, :]

        self.num = len(self.labels)

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        label = torch.LongTensor([np.int64(labels)])
        return np.float32(img), label

    def __len__(self):
        return self.num
