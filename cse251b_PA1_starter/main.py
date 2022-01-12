import argparse
import network
import data
from network import Network
from pca import PCA
import numpy as np
import os, random, sys
import matplotlib.pyplot as plt
from data import traffic_sign, generate_k_fold_set


def main(hyperparameters):
    pass

parser = argparse.ArgumentParser(description = 'CSE251B PA1')
parser.add_argument('--batch-size', type = int, default = 128,
        help = 'input batch size for training (default: 128)')
parser.add_argument('--epochs', type = int, default = 300,
        help = 'number of epochs to train (default: 150)')
parser.add_argument('--learning-rate', type = float, default = 0.001,
        help = 'learning rate (default: 0.001)')
parser.add_argument('--z-score', dest = 'normalization', action='store_const',
        default = data.min_max_normalize, const = data.z_score_normalize,
        help = 'use z-score normalization on the dataset, default is min-max normalization')
parser.add_argument('--in-dim', type = int, default = 32*32, 
        help = 'number of principal components to use')
parser.add_argument('--out-dim', type = int, default = 43,
        help = 'number of outputs')
parser.add_argument('--k-folds', type = int, default = 5,
        help = 'number of folds for cross-validation')

hyperparameters = parser.parse_args()
main(hyperparameters)
    
def PCA_preprocess(k = 10, n_components = 40):
    # Keep 3 different component numbers. 40, 100, 150
    # 1. Init
    load_data = traffic_sign()
    # 2. Get train, valid and test set - only for first fold for training
    train_data, train_label, valid_data, valid_label, test_data, test_label = generate_k_fold_set(load_data)

    # 3. Apply the PCA - should only perform on the training set
    prob = PCA(train_data, n_components)
    projected, mean_image, sqrt_eigen_values, eigen_vectors  = prob.PCA_()

    # 4. The resulting projections and report the result
    print('1. Projected Training set >> mean, std ', np.mean(projected), 'and', np.std(projected))
    # Project the valid and test set
    valid_data = np.dot((valid_data - mean_image), eigen_vectors) / sqrt_eigen_values
    print('2. Projected Validation set >> mean, std ', np.mean(valid_data), 'and',np.std(valid_data))
    test_data = np.dot((test_data - mean_image), eigen_vectors) / sqrt_eigen_values
    print('3. Projected Test set >> mean, std ', np.mean(test_data), 'and',np.std(test_data))
    prob.plot_PC()

PCA_preprocess()


