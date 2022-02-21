################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
from torchvision import models
import torch.nn as nn
# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want

    raise NotImplementedError("Model Factory Not Implemented")

def get_model_LSTM(config_data, vocab, DEBUG=True):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want

    encoder = models.resnet50(pretrained=True)
    fc_in_features = encoder.fc.in_features
    
    if DEBUG:
        print("fc_in_features:", fc_in_features)
        print("embedding_size:", embedding_size)
    encoder.fc = nn.Linear(fc_in_features, embedding_size)
    
    # Remove last 2 layers (pooling and fc)
        
    raise NotImplementedError("Model Factory LSTM Not Implemented")