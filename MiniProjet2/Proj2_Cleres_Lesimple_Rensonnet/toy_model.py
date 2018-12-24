# -*- coding: utf-8 -*-
from math import pi
from torch import FloatTensor, LongTensor

def generate_disc_set(nb, one_hot=False):
    """
    Generates a uniform distribution of points in [0,1]x[0,1]. Assigns
    label 1 if a point is inside the disk of radius sqrt(1/(2*pi)) centered
    at (0.5, 0.5), and a label 0 otherwise.
    Supports one_hot encoding with vector entries in {-1, 1} for use with
    MSE loss for instance.
    """
    data = FloatTensor(nb, 2).uniform_(0, 1)
    target = data.sub(0.5).pow(2).sum(1).sub(1/(2*pi)).sign().sub(1).div(-2).long() # 1 if inside disk, 0 if outside
    if one_hot:
        # Useful for MSE loss: convert from scalar labels to vector labels
        target_one_hot = LongTensor(nb, 2).fill_(-1)
        # A row is [-1, 1] if point inside disk and [1, -1] outside of it
        for k in range(target.size(0)):
            target_one_hot[k, target[k]] = 1
        target = target_one_hot
    return data, target


def generate_data_standardized(nb, one_hot=False):
    # Generation of the data
    train_input, train_target = generate_disc_set(nb, one_hot)
    test_input, test_target = generate_disc_set(nb, one_hot)

    # Calculation of mean and std dev of training set
    mean, std = train_input.mean(), train_input.std()

    # Standardization
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    return train_input, train_target, test_input, test_target

