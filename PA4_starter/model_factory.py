################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
from torchvision import models
import torch.nn as nn
from models import LSTM, LSTMEncoder, LSTMDecoder

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    model_type = config_data['model']['model_type']
    
    if model_type == 'LSTM':
        model = get_model_LSTM(config, vocab)
        if model:
            return get_model_LSTM(config, vocab)
        
    # You may add more parameters if you want

    raise NotImplementedError("Model Factory Not Implemented")

def get_model_LSTM(config_data, vocab, DEBUG=True):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    num_layers = 2
    vocab_size = len(vocab) # TODO
    
    if model_type != 'LSTM':
        return False
    
    encoder = LSTMEncoder(image_embedding_size=embedding_size)
    decoder = LSTMDecoder(vocab_size=vocab_size, hidden_size=hidden_size, word_embedding_size=embedding_size, num_layers=num_layers):
    model = LSTM(encoder, decoder)
        
    return model

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    