import argparse
import network
import data
from network import Network
from pca import PCA
import numpy as np
import os, random, sys
import matplotlib.pyplot as plt
from data import traffic_sign, generate_k_fold_set,select_binarydata,generate_no_fold_set


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
    aligned = True
    load_data = traffic_sign(True)
    # 2. Get train, valid and test set - only for first fold for training
    train_data, train_label, valid_data, valid_label, test_data, test_label = generate_no_fold_set(load_data)
    
    # 3. Apply the PCA - should only perform on the training set
    prob = PCA(n_components)
    #projected  = prob.PCA_Emmet(),#PCA_generate
    projected  = prob.fit_transform(train_data)

    # 4. The resulting projections and report the result
    print('1. Training set >> mean : ', np.mean(projected), 'std : ', np.std(projected)*np.sqrt(projected.shape[0]))
    # Project the valid and test set
    valid_data = prob.transform(valid_data)
    print('2. Validation set >> mean : ', np.mean(valid_data), 'std : ',np.std(valid_data)* np.sqrt(projected.shape[0]))
    test_data = prob.transform(test_data)
    print('3. Test set >> mean : ', np.mean(test_data), 'std : ',np.std(test_data)* np.sqrt(projected.shape[0]))
    prob.plot_PC()


PCA_preprocess()


