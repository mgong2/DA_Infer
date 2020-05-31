import os
import torch.utils.data as data
import torch
from scipy.io import loadmat
from os.path import join
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


# mnist rotation data
class DatasetMNISTR4(data.Dataset):
    def __init__(self, specs):
        super(DatasetMNISTR4, self).__init__()
        self.root = specs['root']
        self.num_train = specs['num_train']
        self.train = specs['train']
        self.num_domain = specs['num_domain']
        self.tar_id = specs['tar_id']
        self.resolution = specs['resolution']
        self.angles = [[30, 60, 90, 0], [0, 60, 90, 30], [0, 30, 90, 60], [0, 30, 60, 90]]
        full_path = join(self.root, 'MNIST_rot.npz')
        self.transforms_train = transforms.Compose(
            [transforms.Resize(self.resolution), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

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

    def pre_process(self, img):

        if len(img.shape) == 2:
            img = np.concatenate([np.expand_dims(img, axis=2)] * 3, axis=2)

        img = Image.fromarray(img)
        return img

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        img_pp = self.pre_process(img)
        angle = self.angles[self.tar_id][labels[1]]
        tran_rot = transforms.RandomRotation((angle, angle))
        imgr = self.transforms_train(tran_rot(img_pp))
        label = torch.LongTensor([labels])
        return imgr, label

    def __len__(self):
        return self.num_training_samples


class DatasetMNISTR3(data.Dataset):
    def __init__(self, specs):
        super(DatasetMNISTR3, self).__init__()
        self.root = specs['root']
        self.num_train = specs['num_train']
        self.train = specs['train']
        self.num_domain = specs['num_domain']
        self.tar_id = specs['tar_id']
        self.resolution = specs['resolution']
        self.angles = [[45, 90, 0], [0, 90, 45], [0, 45, 90]]
        full_path = join(self.root, 'MNIST_rot.npz')
        self.transforms_train = transforms.Compose(
            [transforms.Resize(self.resolution), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

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

    def pre_process(self, img):

        if len(img.shape) == 2:
            img = np.concatenate([np.expand_dims(img, axis=2)] * 3, axis=2)

        img = Image.fromarray(img)
        return img

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        img_pp = self.pre_process(img)
        angle = self.angles[self.tar_id][labels[1]]
        tran_rot = transforms.RandomRotation((angle, angle))
        imgr = self.transforms_train(tran_rot(img_pp))
        label = torch.LongTensor([labels])
        return imgr, label

    def __len__(self):
        return self.num_training_samples