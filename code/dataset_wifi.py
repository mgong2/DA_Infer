import os
import torch.utils.data as data
import torch
from scipy.io import loadmat
from os.path import join
import numpy as np


# wifi data
class DatasetWifiMultiTot1(data.Dataset):
    def __init__(self, specs):
        super(DatasetWifiMultiTot1, self).__init__()
        self.root = specs['root']
        self.num_train = specs['num_train']
        self.train = specs['train']
        self.num_domain = specs['num_domain']
        self.id = specs['seed']
        full_path = join(self.root, 'tot1_dataId'+str(self.id)+'.npz')

        npzfile = np.load(full_path)
        x = npzfile['x']
        y = npzfile['y']
        self.test_id = self.num_domain - 1

        if self.train:
            self.data = x
            self.labels = y
        else:
            self.data = x[y[:, 1] == self.test_id, :]
            self.labels = y[y[:, 1] == self.test_id, :]

        self.num_training_samples = self.labels.shape[0]

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        labels[0] = int(labels[0] * 2.0 / 3)
        label = torch.LongTensor([labels])
        return np.float32(img), label

    def __len__(self):
        return self.num_training_samples


class DatasetWifiMultiTot2(data.Dataset):
    def __init__(self, specs):
        super(DatasetWifiMultiTot2, self).__init__()
        self.root = specs['root']
        self.num_train = specs['num_train']
        self.train = specs['train']
        self.num_domain = specs['num_domain']
        self.id = specs['seed']
        full_path = join(self.root, 'tot2_dataId'+self.id+'.npz')

        npzfile = np.load(full_path)
        x = npzfile['x']
        y = npzfile['y']
        self.test_id = self.num_domain - 1

        if self.train:
            self.data = x
            self.labels = y
        else:
            self.data = x[y[:, 1] == self.test_id, :]
            self.labels = y[y[:, 1] == self.test_id, :]

        self.num_training_samples = self.labels.shape[0]

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        labels[0] = int(labels[0] * 2.0 / 3)
        label = torch.LongTensor([labels])
        return np.float32(img), label

    def __len__(self):
        return self.num_training_samples


class DatasetWifiMultiTot3(data.Dataset):
    def __init__(self, specs):
        super(DatasetWifiMultiTot3, self).__init__()
        self.root = specs['root']
        self.num_train = specs['num_train']
        self.train = specs['train']
        self.num_domain = specs['num_domain']
        self.id = specs['seed']
        full_path = join(self.root, 'tot3_dataId'+self.id+'.npz')

        npzfile = np.load(full_path)
        x = npzfile['x']
        y = npzfile['y']
        self.test_id = self.num_domain - 1

        if self.train:
            self.data = x
            self.labels = y
        else:
            self.data = x[y[:, 1] == self.test_id, :]
            self.labels = y[y[:, 1] == self.test_id, :]

        self.num_training_samples = self.labels.shape[0]

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        labels[0] = int(labels[0] * 2.0 / 3)
        label = torch.LongTensor([labels])
        return np.float32(img), label

    def __len__(self):
        return self.num_training_samples


