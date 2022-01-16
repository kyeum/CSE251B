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

def balance_data(dataset):
    Data, labels = dataset
    print(np.shape(Data),np.shape(labels))
    #only save class a, and class b.
    idx_class_a = np.where(labels == 1)
    idx_class_b = np.where(labels == 2)
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
        The data to z-score normalize

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    mu = np.mean(X, axis=0) # calculate mean for each feature col
    sigma = np.std(X, axis=0) # calculate stddev for each feature col

    X_norm = (X - mu) / sigma # normalize each feature col
    return X_norm

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
    min = np.min(X, axis=0) # calculate min for each feature col
    max = np.max(X, axis=0) # calculate max for each feature col
    fx = (X - min) / (max - min) # min_max_normalize each feature col

    return fx, min, max

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
    k = np.max(y) + 1
    onehot_encoded = np.eye(k)[y]
    return onehot_encoded

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
    indices = np.argmax(y, axis=1)
    return indices

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
    X, y = dataset
    # Shuffle the indices
    indices = np.arange(len(X))
    indices = np.random.shuffle(indices)
    # Get images and labels according to shuffled indices
    X = X[indices]
    y = y[indices]
    return (X, y)

def append_bias(X):
    pass

def generate_minibatches(dataset, batch_size=64):
    Data, labels = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(Data):
        yield Data[l_idx:r_idx], labels[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield Data[l_idx:], labels[l_idx:]

def generate_k_fold_set(dataset, k=10): 
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
    
    order = np.random.permutation(len(Data)) # shuffle
    fold_width = len(Data) // k # division with floor
    l_idx, r_idx = 0, fold_width # one fold data length
    # Training set : K-2, Validation set : 1, Test set : 1 first two sets are for validation, test, rest of them are for training

    #only testing with first fold in the beginning -> update later 
    l_idx = 0
    m_idx = 1 * fold_width
    r_idx = 2 * fold_width
    
    for i in range(k):
        if r_idx < l_idx : 
            if l_idx < m_idx : 
                print("1")
                train = Data[order[r_idx:l_idx]], labels[order[r_idx:l_idx]]
                validation = Data[order[l_idx:m_idx]], labels[order[l_idx:m_idx]]
                test = np.concatenate([Data[order[:r_idx]], Data[order[m_idx:]]]), np.concatenate([labels[order[:r_idx]], labels[order[m_idx:]]])
                
            else : 
                print("2")
                train = Data[order[r_idx: l_idx]], labels[order[r_idx: l_idx]]
                validation = np.concatenate([Data[order[:m_idx]], Data[order[l_idx:]]]),np.concatenate([labels[order[:m_idx]], labels[order[l_idx:]]])
                test = Data[order[m_idx:r_idx]], labels[order[m_idx:r_idx]]  
        else :
            train = np.concatenate([Data[order[:l_idx]], Data[order[r_idx:]]]), np.concatenate([labels[order[:l_idx]], labels[order[r_idx:]]])
            validation = Data[order[l_idx:m_idx]], labels[order[l_idx:m_idx]]
            test = Data[order[m_idx:r_idx]], labels[order[m_idx:r_idx]]                

        yield train, validation, test
        l_idx, m_idx, r_idx = (l_idx + fold_width)% len(Data), (m_idx + fold_width )% len(Data), (r_idx + fold_width) % len(Data)

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
    fold_width = len(Data)//10 # division with floor
    l_idx, r_idx = 0, fold_width # one fold data length
    print("FOLDWIDTH:", fold_width)
    # Training set : K-2, Validation set : 1, Test set : 1 first two sets are for validation, test, rest of them are for training
    Traindata = []
    TrainLabel = []
    Valdata = []
    ValLabel = []
    Testdata = []
    TestLabel = []
    #only testing with first fold in the beginning -> update later 
    split = fold_width
    Valdata = (Data[order[l_idx:l_idx+split]])
    ValLabel = (labels[order[l_idx:l_idx+split]]) # first sets
    Testdata = (Data[order[l_idx+split:l_idx+split*2]])
    TestLabel = (labels[order[l_idx+split:l_idx+split*2]])# second sets
    Traindata = Data[order[l_idx+split*2:]]
    TrainLabel = labels[order[l_idx+split*2:]]

    return Traindata,TrainLabel,Valdata,ValLabel,Testdata,TestLabel

#Traindata,TrainLabel,Valdata,ValLabel,Testdata,TestLabel = generate_k_fold_set(traffic_sign())

def Balance_data(dataset, k=10): 
    
    Data, labels = dataset
    print(len(labels))
    #print(np.shape(dataset))




    return 0

