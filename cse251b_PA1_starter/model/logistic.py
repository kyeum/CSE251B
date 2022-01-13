import numpy as np


class LogisticRegression():
    '''
    Logistic Regression
    '''

    def __init__(self, lr):
        '''
        Args

        size: the size of input, should be n_components
        class_size: number of classes, should be 4
        lr: learning rate
        '''
        #size of the weight -> initialize to random
        #learning rate
        self.lr = lr

    def logistic_model(self, w,   x):
        ''' 
        x :  M x (d + 1)
        w :  (d + 1) x 1

        Returns
        y: dimention of (M, 1)
        '''
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)  #add 1 for x0
        x = np.dot(x,w.T)
        res = 1 / (1 + np.exp(-x)) # actvation fnc
        return res

    def loss_binary(self, y, true_y) :
        '''
        calculate the loss using cross-entropy cost function

        args

        y: forward result, dimention of (M, 1)

        true_y: true result, dimention of (M, 1)

        Returns

        loss: loss value
        '''
        cross_entropy_loss = -np.sum(true_y * np.log(y) + (1 - true_y) * np.log(1 - y))
        return cross_entropy_loss

    def update_weight(self, w, x, y, true_y):
        '''
        Args

        x: input data, which dimention is (M, d), means M pics each pixel number is d

        y: forward result, dimention of (M, 1)

        true_y: true result, dimention of (M, 1)
        '''
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1) #add 1 for x0

        gradient = np.dot((true_y - y) , x)
        w = w + self.lr * gradient
        return w

    def check_accuracy(self,y, y_t):
        '''
        accuracy for logistic regression
        '''
        y_round = np.round(y)
        correct = np.sum(y_round == y_t)
        accuracy = correct / y.shape[0]
        return accuracy
