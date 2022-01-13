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

    def logistic_model(self,w,   x: np.ndarray) -> np.ndarray:
        ''' 
        logistic model forward

        Args

        x: input data, which dimention is (M, d), means M pics each pixel number is d

        Returns

        y: dimention of (M, 1)
        '''
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        x = np.dot(w,x.T)
        res = 1 / (1 + np.exp(-x)) # actvation
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
        cross_entropy_loss = -np.mean(true_y * np.log(y) + (1 - true_y) * np.log(1 - y))
        return cross_entropy_loss

    def update_weight(self, w, x: np.ndarray, y: np.ndarray, true_y: np.ndarray) -> None:
        '''
        logistic model backward

        Args

        x: input data, which dimention is (M, d), means M pics each pixel number is d

        y: forward result, dimention of (M, 1)

        true_y: true result, dimention of (M, 1)
        '''
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1) #add 1 for x0

        gradient = np.dot((true_y - y) , x)
        w += self.lr * gradient
        return w

    def check_accuracy(self,y, y_t):
        '''
        accuracy for logistic regression
        '''
        y_round = np.round(y)
        correct = np.sum(y_round == y_t)
        accuracy = correct / y.shape[0]
        return accuracy


    def simple_logistic_model_gradient_descent(self,model_output, input_vec, true_label):
        '''
        Args:
            model_output : calculated result
            input_vec : input image
            true_label : true category

        Returns:
            dw :gradient descent
        '''
        N = model_output.shape[0]
        # convert true label vector into binary
        tn = true_label.reshape((len(model_output), 1))
        #calculate the gradient
        dw = np.sum(np.dot((tn - model_output).T, input_vec), axis = 0, keepdims= True) / N
        return dw.T
