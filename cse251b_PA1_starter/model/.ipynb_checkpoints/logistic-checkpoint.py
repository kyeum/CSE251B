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
        x = w @ x.T
        return 1 / (1 + np.exp(-x))

    def loss(self, y: np.ndarray, true_y: np.ndarray) -> int:
        '''
        calculate the loss using cross-entropy cost function

        args

        y: forward result, dimention of (M, 1)

        true_y: true result, dimention of (M, 1)

        Returns

        loss: loss value
        '''

        return -np.mean(true_y * np.log(y) + (1 - true_y) * np.log(1 - y))


    def simple_logistic_model_loss(self,model_output, input_vec, true_label):
        loss = np.sum(true_label * np.log(model_output) \
                    + (1 - true_label) * np.log(1 - model_output)) * -1
        return loss.T

    def update_weight(self, w, x: np.ndarray, y: np.ndarray, true_y: np.ndarray) -> None:
        '''
        logistic model backward

        Args

        x: input data, which dimention is (M, d), means M pics each pixel number is d

        y: forward result, dimention of (M, 1)

        true_y: true result, dimention of (M, 1)
        '''
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1) #add 1 for x0

        gradient = (true_y - y) @ x
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

    '''
    def loss(self, X, y):
        y_hat = self.logistic(np.dot(X, self.w))
        return (- np.dot(y.T, np.log(y_hat)) / len(y_hat))[0][0]

    def gradient(self, X, y):
        y_hat = self.logistic(np.dot(X, self.w))
        return np.sum((y_hat - y) * X, axis=0).reshape(-1, 1)

    def accuracy(self, test_set=None):
        if not test_set:
            test_set = self.test_set
        return np.sum(self.predict(test_set.X) == test_set.y) / len(test_set.y)
    '''






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
