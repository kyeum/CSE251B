################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2022
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np
import pickle



def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    TODO: Normalize your inputs here to have 0 mean and unit variance by z scoring.

    input = batch_size x 32 x 32 x 3

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    inp : np.array
        The data to z-score normalize

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    print("inp:", inp.shape)
    mu = np.mean(inp, axis=(0,1,2)) # calculate mean for each feature col
    sigma = np.std(inp, axis=(0,1,2)) # calculate stddev for each feature col
    X_norm = (inp - mu) / sigma

    return X_norm, (mu, sigma)

def normalize_data_given(X, stats):
    """
    Z-score normalize a dataset given the mean and stddev of the training set.
    """
    mean, stddev = stats
    X = (X - mean) / stddev
    return X


def one_hot_encoding(labels):
    """
    TODO: Encode labels using one hot encoding and return them.

    Performs one-hot encoding on y.

    Assumes 0-indexed classes.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    
    k = np.max(labels) + 1
    onehot_encoded = np.eye(k)[labels]
    return onehot_encoded
    
def onehot_decode(y):
    indices = np.argmax(y, axis=1)
    return indices

def load_data(path, stats=None, mode='train'):
    """
    Load CIFAR-10 data.
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, "cifar-10-batches-py")

    if mode == "train":
        images = []
        labels = []
        for i in range(1,6):
            images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{i}"))
            data = images_dict[b'data'] # 10000 x 3072
            label = images_dict[b'labels'] # 10000
            labels.extend(label)
            images.extend(data.reshape((-1, 32, 32, 3)))
        images = np.array(images)
        print(images.shape)
        normalized_images, stats = normalize_data(images)
        one_hot_labels    = one_hot_encoding(labels) #(n,10)
        return np.array(normalized_images), np.array(one_hot_labels), stats
    elif mode == "test":
        test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
        test_data = test_images_dict[b'data']
        test_labels = test_images_dict[b'labels']
        test_data = test_data.reshape(-1, 32, 32, 3)
        normalized_images = normalize_data_given(test_data, stats)
        one_hot_labels    = one_hot_encoding(test_labels) #(n,10)
        return np.array(normalized_images), np.array(one_hot_labels)
    else:
        raise NotImplementedError(f"Provide a valid mode for load data (train/test)")




def softmax(x):
    """
    TODO: Implement the softmax function here.
    Remember to take care of the overflow condition.

    Softmax activation function

    Input: X (n elements x k classes)
    """
    
    eX = np.exp(x - np.max(x, axis=1)[:, np.newaxis]) # e^X
    # [:, np.newaxis] is necessary for broadcasting to work properly
    partition = np.sum(eX, axis=1)[:, np.newaxis] # sum of each row
    return eX / partition
    

def plot_PC(self):
    '''
    Plot top 4 principal components
    the eigenvector with the largest eigenvalue is the first principal component,
    '''
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(8, 8)
    fig.set_dpi(100)
    axs[0, 0].set_title('PC 1')
    axs[0, 0].imshow(self.principal_eigen_vectors.T[0].real.reshape((32, 32)))
    axs[0, 1].set_title('PC 2')
    axs[0, 1].imshow(self.principal_eigen_vectors.T[1].real.reshape((32, 32)))
    axs[1, 0].set_title('PC: 3')
    axs[1, 0].imshow(self.principal_eigen_vectors.T[2].real.reshape((32, 32)))
    axs[1, 1].set_title('PC: 4')
    axs[1, 1].imshow(self.principal_eigen_vectors.T[3].real.reshape((32, 32)))
    plt.show()


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            return self.leakyReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        sigmoid(z) = 1 / (1 + e^{-z})
        """
        self.x = x

        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        TODO: Implement tanh here.
        tanh(z) = (e^z - e^{-z})/(e^z + e^{-z})
        """
        self.x = x

        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        ReLU(z) = max(0, z)
        """
        self.x = x
        return np.maximum(0, x)

    def leakyReLU(self, x):
        """
        TODO: Implement leaky ReLU here.
        leakyReLU(z) = max(0.1*z, z)
        """
        self.x = x

        return np.maximum(0.1 * x, x)

    def grad_sigmoid(self):
        """
        TODO: Compute the gradient for sigmoid here.
        dsigmoid(z) = sigmoid(z) * (1 - sigmoid(z))
        """
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        TODO: Compute the gradient for tanh here.
        dtanh(z) = 1 - tanh(z)^2
        """
        return 1 - np.tanh(self.x) ** 2

    def grad_ReLU(self):
        """
        TODO: Compute the gradient for ReLU here.
        dReLU(z) = 1 if z > 0 else 0
        """
        return np.where(self.x > 0, 1, 0)

    def grad_leakyReLU(self):
        """
        TODO: Compute the gradient for leaky ReLU here.
        dleakyReLU(z) = 1 if z > 0 else 0.1
        """
        return np.where(self.x > 0, 1, 0.1)


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """
    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)    #input layer size  output layer size     # >>EY : add randomize 
        # self.w = np.zeros((in_units, out_units))    #input layer size  output layer size     # >>EY : add randomize 
        self.b = np.zeros((1, out_units)) # Create a placeholder for Bias        # >>EY : add randomize 

        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        print("IN Layer forward")
        self.x = x.reshape((-1, self.w.shape[0]))
        self.a = np.dot(self.x,self.w) + self.b
        return self.a

    def backward(self, delta):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        size = self.x.shape[0]

        self.d_x = np.dot(delta,self.w.T)
        self.d_w = -np.dot(self.x.T,delta) / size
        self.d_b = -delta.sum(axis=0) / size
        return self.d_x

class SoftmaxLayer(Layer):
    def __init__(self):
        pass

    def forward(self, x, targets=None):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        y_hat = self.softmax(x)
        self.y = y_hat

        if targets is None:
            return y_hat
        loss = self.loss(y_hat, targets)
        return y_hat, loss

    def backward(self, delta):
        delta = (delta -(delta * self.y).sum(axis=1)[:,None])
        return self.y * delta

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        epsilon = 1e-10
        y_true = np.argmax(targets, axis=1)# decode
        ce = np.log(logits[range(len(logits)), y_true] + epsilon)
        return -np.sum(ce)    

    def softmax(self, x):
        """
        TODO: Implement the softmax function here.
        Remember to take care of the overflow condition.

        Softmax activation function

        Input: X (n elements x k classes)
        """
        
        eX = np.exp(x - np.max(x, axis=1)[:, np.newaxis]) # e^X
        # [:, np.newaxis] is necessary for broadcasting to work properly
        partition = np.sum(eX, axis=1)[:, np.newaxis] # sum of each row
        return eX / partition

class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.l2_penalty = None

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))
        self.layers.append(SoftmaxLayer())

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        self.targets = targets

        out = self.x
        for layer in self.layers:
            out = layer.forward(out)

        # Softmax
        # print("OUT:", out)
        # self.y = softmax(out)
        self.y = out

        if targets is None:
            return self.y

        # Compute cross entropy loss
        loss = self.loss(self.y, targets)

        return self.y, loss

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        epsilon = 1e-10
        y_true = np.argmax(targets, axis=1)# decode
        ce = np.log(logits[range(len(logits)), y_true] + epsilon)
        return -np.sum(ce)       

    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        delta = self.targets - self.y
        for layer in self.layers[::-1]:
            if isinstance(layer, Layer):
                delta = layer.backward(delta) #update delta

        return delta


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """

    raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.
    """




    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./data")

    config_prob_b = {}
    config_prob_b['layer_specs'] = [3072, 50, 50, 11]
    config_prob_b['activation'] = 'ReLU'
    config_prob_b['learning_rate'] = 0.15 
    config_prob_b['batch_size'] = 128 
    config_prob_b['epochs'] = 100  
    config_prob_b['early_stop'] = True 
    config_prob_b['early_stop_epoch'] = 5  
    config_prob_b['L2_penalty'] = 0  
    config_prob_b['momentum'] = True  
    config_prob_b['momentum_gamma'] = 0.9  
    # Create the model
    model  = Neuralnetwork(config_prob_b)

    # Load the data
    x_train, y_train = load_data(path="./data", mode="train")
    #x_test,  y_test  = load_data(path="./data", mode="test")
    xt = []
    yt = []




    # TODO: Create splits for validation data here.
    # x_val, y_val = ...

    # TODO: train the model
    #train(model, x_train, y_train, x_valid, y_valid, config)

    #test_acc = test(model, x_test, y_test)

    # TODO: Plots
    # plt.plot(...)
