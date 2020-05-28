import os
import torch.utils.data as data
import torch
from scipy.io import loadmat
from os.path import join
import numpy as np


# wifi data
class DatasetWifi(data.Dataset):
    def __init__(self, specs):
        super(DatasetWifi, self).__init__()
        self.root = specs['root']
        self.num_train = specs['num_train']
        self.train = specs['train']
        self.num_domain = specs['num_domain']
        self.id = specs['seed']
        self.tar_id = specs['tar_id']
        self.useMB = specs['useMB']
        dagFile = np.load(join(self.root, specs['dag_mat_file']))
        MB = dagFile['MB']
        full_path = join(self.root, 'tot%d_dataId%d.npz' % (self.tar_id, self.id))

        npzfile = np.load(full_path)
        x = npzfile['x']
        y = npzfile['y']
        self.test_id = self.num_domain - 1

        if self.useMB:
            x = x[:, MB[:-2]]

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


