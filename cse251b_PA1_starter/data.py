import numpy as np
import os

from traffic_reader import load_traffic

def traffic_sign(aligned):
    if aligned:
        return load_traffic('data', kind='aligned') 
    return load_traffic('data', kind='unaligned') 
    # Image : (cnt, 32x32 byte = 1024) in one img data , 1 byte in one image data

def select_binarydata(dataset, class_a,class_b):
    Data, labels = dataset
    print(np.shape(Data),np.shape(labels))
    #only save class a, and class b.
    idx_class_a = np.where(labels == class_a)
    idx_class_b = np.where(labels == class_b)
    Data_class_a = Data[idx_class_a]
    Data_class_b = Data[idx_class_b]

    Data_ = np.concatenate([Data_class_a,Data_class_b])
    label_ = np.concatenate([labels[idx_class_a],labels[idx_class_b]])
    return   Data_ ,label_
    # Image : (cnt, 32x32 byte = 1024) in one img data , 1 byte in one image data


def z_score_normalize(X, u = None, sd = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to min-max normalize

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    pass

def min_max_normalize(X, _min = None, _max = None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """
    pass

def onehot_encode(y):
    """
    Performs one-hot encoding on y.

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
    pass

def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    pass

def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    pass

def append_bias(X):
    pass

def generate_minibatches(dataset, batch_size=64):
    Data, labels = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(Data):
        yield Data[l_idx:r_idx], labels[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield Data[l_idx:], labels[l_idx:]

def generate_k_fold_set(dataset, k = 10): 
    """
    Generate k fold sets.
    Eact sets are split in to train/val/test mutually exclusive()

    Parameters
    ----------
    dataset(img : 32x32 = 1024, label : 41), 
    k : int

    Returns
    -------
    Traindata, TrainLabel, Testdata,TestLabel, Valdata, ValLabel
    (ndarray :)
    """
   
    # Be sure to modify to include train/val/test and need to split train/val/test mutually exclusive(k-2/ 1/ 1)
    Data, labels = dataset
    
    order = np.random.permutation(len(Data)) # suffle
    fold_width = len(Data)//k # division with floor
    l_idx, r_idx = 0, fold_width # one fold data length
    # Training set : K-2, Validation set : 1, Test set : 1 first two sets are for validation, test, rest of them are for training
    Traindata = []
    TrainLabel = []
    Valdata = []
    ValLabel = []
    Testdata = []
    TestLabel = []
    #only testing with first fold in the beginning -> update later 
    for i in range(1):
        split = fold_width//k
        Valdata = (Data[order[l_idx:l_idx+split]])
        ValLabel = (labels[order[l_idx:l_idx+split]]) # first sets
        Testdata = (Data[order[l_idx+split:l_idx+split*2]])
        TestLabel = (labels[order[l_idx+split:l_idx+split*2]])# second sets
        Traindata = Data[order[l_idx+split*2:r_idx]]
        TrainLabel = labels[order[l_idx+split*2:r_idx]]
        l_idx, r_idx = r_idx, r_idx + fold_width

    return Traindata,TrainLabel,Valdata,ValLabel,Testdata,TestLabel

def generate_no_fold_set(dataset, k = 1): 
    """
    Generate k fold sets.
    Eact sets are split in to train/val/test mutually exclusive()

    Parameters
    ----------
    dataset(img : 32x32 = 1024, label : 41), 
    k : int

    Returns
    -------
    Traindata, TrainLabel, Testdata,TestLabel, Valdata, ValLabel
    (ndarray :)
    """
    # Be sure to modify to include train/val/test and need to split train/val/test mutually exclusive(k-2/ 1/ 1)
    Data, labels = dataset
    order = np.random.permutation(len(Data)) # suffle
    fold_width = len(Data) # division with floor
    l_idx, r_idx = 0, fold_width # one fold data length
    # Training set : K-2, Validation set : 1, Test set : 1 first two sets are for validation, test, rest of them are for training
    Traindata = []
    TrainLabel = []
    Valdata = []
    ValLabel = []
    Testdata = []
    TestLabel = []
    #only testing with first fold in the beginning -> update later 
    for i in range(1):
        split = fold_width
        Valdata = (Data[order[l_idx:l_idx+split]])
        ValLabel = (labels[order[l_idx:l_idx+split]]) # first sets
        Testdata = (Data[order[l_idx+split:l_idx+split*2]])
        TestLabel = (labels[order[l_idx+split:l_idx+split*2]])# second sets
        Traindata = Data[order[l_idx+split*2:r_idx]]
        TrainLabel = labels[order[l_idx+split*2:r_idx]]
        l_idx, r_idx = r_idx, r_idx + fold_width

    return Traindata,TrainLabel,Valdata,ValLabel,Testdata,TestLabel

#Traindata,TrainLabel,Valdata,ValLabel,Testdata,TestLabel = generate_k_fold_set(traffic_sign())
