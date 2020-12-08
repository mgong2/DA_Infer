


# Domain Adaptation as A Problem of Inference on Graphical Models

This experimental code provides digits adaptation results in Table 4.

## Requirements
* Environment
Python 3.7
Pytorch >= 1.0.0

* Data Preparation
Downloading address: [MNIST](http://yann.lecun.com/exdb/mnist/index.html) [SVHN](http://ufldl.stanford.edu/housenumbers/) [MNIST\_M](http://yaroslav.ganin.net/) [SythDigits](http://yaroslav.ganin.net/)
Please download the datasets and do preprocessing using following step.

1. Download the official dataset.
2. Extract the images and labels from the training split and the testing split. Resize the images to be [3,32,32] (channel, length, width). For Mnist, stack the images in the channel dimension three times.

(All of the images are keeping the same size and without any augmentation.)

## Training

To train the model(s) in the paper,  run the following command:
1. train LV-CGAN networks, specify the parameter --target_dataset to change the target domain dataset
```
bash scripts/twin_ac_launch_mnist_mnist.sh
```
(We randomly sample 20,000 paired images from the training set for each dataset. The subsampling process is implemented in the 'train.py'). To be notated, the labels for the target dataset are not allowed to use in our training code and this is implemented in the 'train_fns.py'.

## Evaluation
Evaluation is provided in the 'train.py' code.

1. We randomly sample 9,000 paired images from the extra testing set, which does not overlap with their training set.
2. We test the standard classification accuracy on the sampled testing set.

## Results
See table 1 in the main paper.