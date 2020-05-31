import os
import torch.utils.data as data
import torch
from scipy.io import loadmat
from os.path import join
import numpy as np


# flow data
class DatasetFlow5(data.Dataset):
    def __init__(self, specs):
        super(DatasetFlow5, self).__init__()
        self.root = specs['root']
        self.num_train = specs['num_train']
        self.train = specs['train']
        self.num_domain = specs['num_domain']
        self.seed = specs['seed']
        self.useMB = specs['useMB']
        dagFile = np.load(join(self.root, specs['dag_mat_file']))
        MB = dagFile['MB']
        full_path = join(self.root, 'unnorm_balanced', 'flow_'+str(self.seed)+'_'+str(self.num_train)+'_neqy_unnorm.npz')
        npzfile = np.load(full_path)
        x = npzfile['x']
        y = npzfile['y']

        self.test_id = self.num_domain - 1

        # extract Markov Blanket
        if self.useMB:
            x = x[:, MB]

        if self.train:
            self.data = x
            self.labels = y
        else:
            self.data = x[y[:, 1] == self.test_id, :]
            self.labels = y[y[:, 1] == self.test_id, :]

        self.num_training_samples = self.labels.shape[0]

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        label = torch.LongTensor([np.int64(labels)])
        return np.float32(img), label

    def __len__(self):
        return self.num_training_samples


# flow data
class DatasetFlow3(data.Dataset):
    def __init__(self, specs):
        super(DatasetFlow3, self).__init__()
        self.root = specs['root']
        self.num_train = specs['num_train']
        self.train = specs['train']
        self.num_domain = specs['num_domain']
        self.seed = specs['seed']
        self.useMB = specs['useMB']
        dagFile = np.load(join(self.root, specs['dag_mat_file']))
        MB = dagFile['MB']
        full_path = join(self.root, 'unnorm_balanced', 'flow_'+str(self.seed)+'_'+str(self.num_train)+'_neqy_unnorm.npz')
        npzfile = np.load(full_path)
        x = npzfile['x']
        y = npzfile['y']

        self.test_id = self.num_domain - 1

        # extract Markov Blanket
        if self.useMB:
            x = x[:, MB]

        if self.train:
            self.data = x
            self.labels = y
        else:
            self.data = x[y[:, 1] == self.test_id, :]
            self.labels = y[y[:, 1] == self.test_id, :]

        self.num_training_samples = self.labels.shape[0]

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        label = torch.LongTensor([np.int64(labels)])
        return np.float32(img), label

    def __len__(self):
        return self.num_training_samples