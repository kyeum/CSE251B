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
import random
import math


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
    # print("inp:", inp.shape)
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

def load_data(path, stats, mode='train'):
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
        normalized_images = normalized_images.reshape(-1, 3072)
        return np.array(normalized_images), np.array(one_hot_labels), stats
    elif mode == "test":
        test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
        test_data = test_images_dict[b'data']
        test_labels = test_images_dict[b'labels']
        test_data = test_data.reshape(-1, 32, 32, 3)
        normalized_images = normalize_data_given(test_data, stats)
        one_hot_labels    = one_hot_encoding(test_labels) #(n,10)
        
        normalized_images = normalized_images.reshape(-1, 3072)
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

        elif self.activation_type == "leakyReLU":
            grad = self.grad_leakyReLU()

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
        #(np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return np.tanh(self.x)

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
        self.w = 1*np.random.randn(in_units, out_units) #input layer size  output layer size     # >>EY : add randomize 
        self.b = 1*np.random.randn(1, out_units) # Create a placeholder for Bias        # >>EY : add randomize 

        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = np.zeros_like(self.w)  # Save the gradient w.r.t w in this
        self.d_b = np.zeros_like(self.b)  # Save the gradient w.r.t b in this
        self.w_best = np.zeros_like(self.w)
        self.b_best = np.zeros_like(self.b)
        self.pre_d_w = np.zeros_like(self.w)
        self.pre_d_b = np.zeros_like(self.b)

        self.in_units = in_units
        self.out_units = out_units

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def kaiming_init(self):
        scale = 1/max(1., (self.in_units+self.out_units)/2.)
        limit = math.sqrt(3.0 * scale)
        self.w = np.random.uniform(-limit, limit, size=(self.in_units,self.out_units))


    def forward(self, x):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        self.x = x.copy()
        self.a = np.dot(self.x,self.w) + self.b
        return self.a

    def backward(self, delta, l2_penalty):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        size = self.x.shape[0]
        self.d_x = np.dot(delta,self.w.T)
        self.d_w = np.dot(self.x.T,delta) + self.w * l2_penalty
        self.d_b = delta.sum(axis=0)
        return self.d_x

    def zero_grad(self):
        self.d_w = np.zeros_like(self.d_w)
        self.d_b = np.zeros_like(self.d_b)

    def updateweight(self, lr, l2_penalty, momentum, momentum_gamma, num_batches):
        """
        updating layer weight
        """
        # Average gradient by batch size.

        # TODO: put before or after gradient batch averaging
        #self.d_w += self.w * l2_penalty

        if (momentum) : 

            weight_change_w = self.d_w + momentum_gamma * self.pre_d_w
            weight_change_b = self.d_b + momentum_gamma * self.pre_d_b
            
            self.w -= lr * weight_change_w
            self.b -= lr * weight_change_b
            
            self.pre_d_w = weight_change_w
            self.pre_d_b = weight_change_b

        else : 
            self.w -= lr * self.d_w 
            self.b -= lr * self.d_b 

        self.zero_grad()

    def save_load_weight(self, save):
        if (save) : 
            self.w_best = self.w 
            self.b_best = self.b   
        else :     
            self.w = self.w_best
            self.b = self.b_best 


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
        self.lr = config['learning_rate']  # learning rate add
        self.momentum = config['momentum']  # momentum
        self.momentum_gamma = config['momentum_gamma']  # momentum
        self.l2_penalty = config['L2_penalty']  # momentum

        self.num_batches = 0

        self.input_size = config['layer_specs'][0]
        self.output_size = config['layer_specs'][-1]

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def kaiming_init(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.kaiming_init()

    def scale_weights(self, scale_factor):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.w *= scale_factor
                layer.b *= scale_factor

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x.copy()
        self.targets = targets
        out = x.copy()
        for layer in self.layers:
            out = layer.forward(out)
        # Softmax
        self.y = softmax(out)
        if targets is None:
            return self.y

        # Compute cross entropy loss
        loss = self.loss(self.y, targets)

        if self.l2_penalty :
            for layer in self.layers:
                if isinstance(layer, Layer):
                    loss += (np.mean(layer.w ** 2)) * self.l2_penalty / 2.

        return self.y, loss

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        targets = targets.reshape((-1, self.output_size))
        scale_size = targets.shape[0]
        epsilon = 1e-9
        y_true = np.argmax(targets, axis=1)# decode
        ce = np.log(logits[range(len(logits)), y_true]+epsilon)
        # Normalize loss by batch size and num classes
        return -np.sum(ce) / scale_size / self.output_size


    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        self.num_batches += 1
        delta = -(self.targets - self.y) / self.output_size 
        for layer in self.layers[::-1]:
            if isinstance(layer, Layer):
                delta = layer.backward(delta,self.l2_penalty) #update delta
            else : 
                delta = layer.backward(delta) #update delta


    def zero_grad(self):
        self.num_batches = 0.0
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.zero_grad()

    def updateweight(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        for layer in self.layers[::-1]:
            if isinstance(layer, Layer):
                layer.updateweight(self.lr,self.l2_penalty,self.momentum,self.momentum_gamma, self.num_batches)
                
    def save_load_weight(self, save):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.save_load_weight(save) #False : save , True : load

    def accuracy(self,x,target):
        '''
        Calculate accuracy
        y_true: true labels (onehot)
        y_hat: predicted labels
        '''
        y,l = self.forward(x,target)
        true_labels = onehot_decode(target)
        pred_labels = np.argmax(y, axis=1)
        return np.sum(true_labels == pred_labels) /target.shape[0]

   
def generate_minibatches(Data,labels, batch_size=128):
    order = np.random.permutation(len(Data)) # shuffle
    Data = Data[order]
    labels = labels[order]
    l_idx, r_idx = 0, batch_size
    while r_idx < len(Data):
        yield Data[l_idx:r_idx], labels[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield Data[l_idx:], labels[l_idx:]

def split_data(x, y):
    """
    :param x: Input data
    :param y: Input Label
    :param percentage: Train Validation Split Preventage
    :return: x_train, y_train, x_val, y_val
    """
    percentage = 0.1
    num_val = int(np.round(x.shape[0] * percentage))
    val_l = random.sample(list(range(x.shape[0])), num_val)
    l = list(range(x.shape[0]))
    t_l = [idx for idx in l if (idx not in val_l)]

    return x[t_l], y[t_l], x[val_l], y[val_l]



def train(model, x_train, y_train, x_valid, y_valid, config):
    epochs = config['epochs']
    batch_size = config['batch_size']
    momentum =    config['momentum']
    momentum_gamma = config['momentum_gamma']
    L2_penalty = config['L2_penalty']
    patience = config['early_stop_epoch']

    train_loss_record = []
    train_accuracy_record = []
    holdout_loss_record = []
    holdout_accuracy_record = []
    
    last_val_Loss = float('inf')
    epoch_stop = 0

    # How many times the validation loss has gone up in a row.
    cur_loss_up_sequence = 0

    for epoch in range(epochs):
        epoch_stop = epoch
        batch_loss = []
        batch_accuracy = []
        for x, y in generate_minibatches(x_train, y_train, batch_size):
            # Forward Pass
            train_y, loss = model.forward(x, y)
            batch_loss.append(loss) 
            # Backward Pass
            model.backward()
            model.updateweight() # update weight for each layer.
            batch_accuracy.append(model.accuracy(x,y))
        
        train_loss = np.mean(np.array(batch_loss))
        train_accuracy = np.mean(np.array(batch_accuracy))
        train_loss_record.append(train_loss)
        train_accuracy_record.append(train_accuracy)
        holdout_loss = model.forward(x_valid, y_valid)[1]
        holdout_accuracy = model.accuracy(x_valid, y_valid)
        holdout_loss_record.append(holdout_loss)
        holdout_accuracy_record.append(holdout_accuracy)

        print(f'Epoch: {epoch + 1}, train accuracy: {train_accuracy:.4f}, train_loss_norm:{train_loss:.4f}, '\
            f'valid_acc: {holdout_accuracy:.4f}, valid_loss_norm: {holdout_loss:.4f}')   
                
        
        # Save the best weights according to test set.
        if holdout_loss > last_val_Loss:
            cur_loss_up_sequence += 1
            print("Valid loss go up!")
            print(f"Current patience count: {cur_loss_up_sequence}")
            
            if cur_loss_up_sequence >= patience:
                print("Earlystop!")
                break
        else:
            print("Valid loss going down!")
            cur_loss_up_sequence = 0
            # Save the best weights.
            
        model.save_load_weight(save=True)

        last_val_Loss = holdout_loss
        
    
    return epoch_stop, train_loss_record, train_accuracy_record, holdout_loss_record, holdout_accuracy_record










def test(model, X_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.
    """
    y_hat = model.forward(X_test)
    true_labels = onehot_decode(y_test)
    pred_labels = np.argmax(y_hat, axis=1)
    return np.sum(true_labels == pred_labels) / true_labels.shape[0]

if __name__ == "__main__":
    # Load the configuration.
    # This is only for the testing.

    config = load_config("./data")

    config_prob_c = {}
    config_prob_c['layer_specs'] = [3072, 64, 64, 10]
    config_prob_c['activation'] = 'tanh'
    config_prob_c['learning_rate'] = 0.05 
    config_prob_c['batch_size'] = 128 
    config_prob_c['epochs'] = 100  
    config_prob_c['early_stop'] = True 
    config_prob_c['early_stop_epoch'] = 5  
    config_prob_c['L2_penalty'] = 0  
    config_prob_c['momentum'] = True  
    config_prob_c['momentum_gamma'] = 0.9  
    # Create the model
    model_c  = Neuralnetwork(config_prob_c)

    # Load the data
    x_train, y_train, stats = load_data(path="./data",stats = None, mode="train")
    x_test, y_test = load_data(path="./data",stats = stats, mode="test")

    # TODO(done): Create splits for validation data here.
    x_train, y_train, x_valid, y_valid = split_data(x_train,y_train)


    # TODO(on going): train the model
    trainacc = train(model_c, x_train, y_train, x_valid, y_valid, config_prob_c )
    print(trainacc)

    # TODO(done): test the model
    #test_acc = test(model, x_test, y_test)


    # TODO(on going): Plots