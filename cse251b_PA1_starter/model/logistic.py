import numpy as np
from data import onehot_decode, onehot_encode


class LogisticRegression():
    '''
    Logistic Regression
    '''

    def __init__(self, lr,components):
        '''
        Args
        lr: learning rate
        '''
        #size of the weight -> initialize to random
        #learning rate
        self.lr = lr
        self.b = np.zeros((1, components)) 

    def logistic_model(self, w, x,b):
        ''' 
        x :  M x (d + 1)
        w :  (d + 1) x 1

        Returns
        y: dimention of (M, 1)
        '''
        #x = np.append(x, np.ones((x.shape[0], 1)), axis=1) #add 1 for x0
        #w[-1] = b
        #x = np.dot(x,w.T) 

        x = np.dot(x, w.T) + b

        res = 1 / (1 + np.exp(-x)) # actvation fnc
        return res


    def loss_binary(self, y, true_y) :
        '''
        cross-entropy cost function
        y: (M, 1)

        true_y: true result (M, 1)

        Returns
        loss: (M,1)
        '''
        loss = -(np.sum(true_y * np.log(y) + (1 - true_y) * np.log(1 - y)))/np.size(y)
        return loss




    def update_weight(self, w, b,x, y, true_y):
        '''
        Args

        x: input data, which dimention is (M, d), means M pics each pixel number is d

        y: forward result, dimention of (M, 1)

        true_y: true result, dimention of (M, 1)
        '''

        
        #x = np.append(x, np.ones((x.shape[0], 1)), axis=1) #add 1 for x0
        gradient = np.dot((true_y - y) , x)
        gradient_b = np.sum((true_y - y))
        #w[-1] = b
        w  += self.lr * gradient
        b += self.lr * gradient_b

        return w, b

    def check_accuracy(self,y, y_t):
        '''
        accuracy for logistic regression
        '''
        y_checkprob = np.zeros(y.shape)
        y_checkprob[y > 0.5] = 1
        correct = np.sum(y_checkprob == y_t)
        return correct / y_t.shape[0]
