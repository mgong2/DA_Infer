


# Domain Adaptation as A Problem of Inference on Graphical Models

This experimental code provides digits adaptation results in Table 4.

## Requirements
* Environment
Python 3.7
Pytorch >= 1.0.0

* Data Preparation
Please download the datasets [here](https://drive.google.com/open?id=1yneOon1U5U8FjZNMXKUTtH6V5kp7DKAo). Put the data in the ../data folder.

## Training

To train the model(s) in the paper,  run the following command:
1. train LV-CGAN networks, specify the parameter --target_dataset to change the target domain dataset
```
bash twin_ac_launch_mnist_mnist.sh
```
(We randomly sample 20,000 paired images from the training set for each dataset. The subsampling process is implemented in the 'train.py'). To be notated, the labels for the target dataset are not allowed to use in our training code and this is implemented in the 'train_fns.py'.

## Evaluation
Evaluation is provided in the 'train.py' code.

1. We randomly sample 9,000 paired images from the extra testing set, which does not overlap with their training set.
2. We test the standard classification accuracy on the sampled testing set.

## Results
See table 4 in the main paper.
