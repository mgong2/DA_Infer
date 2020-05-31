import os
import torch.utils.data as data
import torch
from scipy.io import loadmat
from os.path import join
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


# mnist rotation data
class DatasetDigits4(data.Dataset):
    def __init__(self, specs):
        super(DatasetDigits4, self).__init__()
        self.root = specs['root']
        self.num_train = specs['num_train']
        self.num_test = 9000
        self.train = specs['train']
        self.num_domain = specs['num_domain']
        self.tar_id = specs['tar_id']
        self.resolution = specs['resolution']
        self.transforms_train = transforms.Compose(
            [transforms.Resize(self.resolution), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        self.order = [[1, 2, 3, 0], [0, 2, 3, 1], [0, 1, 3, 2], [0, 1, 2, 3]]

        if self.train:
            self.y_d = np.repeat(np.arange(self.num_domain), self.num_train).T.reshape(self.num_train * self.num_domain, 1).squeeze()
            img_mnist, label_mnist = torch.load(join(self.root, 'mnist_train.pt'))
            img_mnist_m, label_mnist_m = torch.load(join(self.root, 'mnist_m_train.pt'))
            img_svhn, label_svhn = torch.load(join(self.root, 'svhn_train.pt'))
            img_synd, label_synd = torch.load(join(self.root, 'synd_train.pt'))
            data_train = []
            label_train = []
            data_train.append(img_mnist)
            data_train.append(img_mnist_m)
            data_train.append(img_svhn)
            data_train.append(img_synd)
            label_train.append(label_mnist)
            label_train.append(label_mnist_m)
            label_train.append(label_svhn)
            label_train.append(label_synd)
            self.data = data_train
            self.label = label_train
        else:
            self.y_d = np.ones(self.num_test, dtype=np.int64) * (self.num_domain-1)
            img_mnist, label_mnist = torch.load(join(self.root, 'mnist_test.pt'))
            img_mnist_m, label_mnist_m = torch.load(join(self.root, 'mnist_m_test.pt'))
            img_svhn, label_svhn = torch.load(join(self.root, 'svhn_test.pt'))
            img_synd, label_synd = torch.load(join(self.root, 'synd_test.pt'))
            data_test = []
            label_test = []
            data_test.append(img_mnist)
            data_test.append(img_mnist_m)
            data_test.append(img_svhn)
            data_test.append(img_synd)
            label_test.append(label_mnist)
            label_test.append(label_mnist_m)
            label_test.append(label_svhn)
            label_test.append(label_synd)
            self.data = data_test
            self.label = label_test

        self.num_training_samples = self.y_d.shape[0]
        self.labels = np.stack((self.y_d, self.y_d), axis=1)

    def pre_process(self, img):

        if len(img.shape) == 2:
            img = np.concatenate([np.expand_dims(img, axis=2)] * 3, axis=2)

        img = Image.fromarray(img)
        return img

    def __getitem__(self, index):
        yd = self.y_d[index]
        if self.train:
            index1 = np.random.randint(self.num_train)
        else:
            index1 = np.random.randint(self.num_test)
        img = self.data[self.order[self.tar_id][yd]][index1]
        labels = self.label[self.order[self.tar_id][yd]][index1]
        img_pp = self.pre_process(img)
        imgr = self.transforms_train(img_pp)
        label = torch.LongTensor([labels, yd])
        return imgr, label

    def __len__(self):
        return self.num_training_samples