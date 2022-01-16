import numpy as np
import sys
sys.path.append("..")
from data import onehot_decode, onehot_encode

class SoftmaxRegression():
    '''
    Softmax Regression
    '''

    def __init__(self, lr, num_features, n_class):
        '''
        Initialization
        '''
        self.lr = lr # learning rate
        self.n_class = n_class # number of classes
        self.W = np.zeros((num_features, n_class)) # weight layer
        self.b = np.zeros((1, n_class)) # bias layer

    def softmax(self, X):
        '''
        Softmax activation function

        Input: X (n elements x k classes)
        '''
        eX = np.exp(X) # e^X
        # [:, np.newaxis] is necessary for broadcasting to work properly
        partition = np.sum(eX, axis=1)[:, np.newaxis] # sum of each row
        return eX / partition
    
    def model(self, X):
        '''
        Model Network for Softmax Regression
        '''
        logits = np.dot(X, self.W) + self.b
        return self.softmax(logits)

    def model_w(self, X, W, b):
        '''
        Model Network for Softmax Regression w/ given weights and bias
        '''
        logits = np.dot(X, W) + b
        return self.softmax(logits)

    def cross_entropy(self, y_true, y_hat):
        '''
        Cross Entropy Loss

        y_true: true labels (onehot)
        y_hat: predicted labels

        Returns cross entropy without being averaged over examples or categories
        '''
        epsilon = 1e-5

        y_true = onehot_decode(y_true)
        ce = np.log(y_hat[range(len(y_hat)), y_true] + epsilon)
        return -np.sum(ce)

    def update_weights(self, X, y_true, y_hat):
        '''
        Update the weights

        Calculates the gradient w.r.t. the weights
        and updates the weights

        y_true: true labels (onehot)
        y_hat: predicted labels
        '''
        grad_w = np.dot(X.T, (y_hat - y_true)) 
        grad_b = np.sum(y_hat - y_true) 

        self.W -= self.lr * grad_w
        self.b -= self.lr * grad_b

    def accuracy(self, y_true, y_hat):
        '''
        Calculate accuracy

        y_true: true labels (onehot)
        y_hat: predicted labels
        '''
        true_labels = onehot_decode(y_true)
        pred_labels = np.argmax(y_hat, axis=1)
        return np.sum(true_labels == pred_labels) / y_true.shape[0]
