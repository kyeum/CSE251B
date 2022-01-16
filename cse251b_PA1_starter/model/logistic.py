import numpy as np
from data import onehot_decode, onehot_encode


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
        cross-entropy cost function

        y: (M, 1)

        true_y: true result (M, 1)

        Returns
        loss: loss value
        '''
        
        loss = -(np.sum(true_y * np.log(y) + (1 - true_y) * np.log(1 - y)))/np.size(y)
        return loss



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
        y_checkprob = np.zeros(y.shape)
        y_checkprob[y > 0.5] = 1
        #y_round = np.round(y)
        correct = np.sum(y_checkprob == y_t)
        return correct / y_t.shape[0]


    '''
    def predict(self, X):
        probs = self.logistic(np.dot(X, self.w))
        prediction = np.zeros(probs.shape)
        prediction[probs > 0.5] = 1
        return prediction

   

    def accuracy(self, test_set=None):
        if not test_set:
            test_set = self.test_set
        return np.sum(self.predict(test_set.X) == test_set.y) / len(test_set.y)
    '''
