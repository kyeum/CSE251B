################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
from torchvision import models
import torch.nn as nn
from models import LSTM, Encoder, LSTMDecoder
import torch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("MODEL_FAC - DEVICE:", device)

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    print("Getting model...")
    model_type = config_data['model']['model_type']
    
    if model_type == 'LSTM':
        model = get_model_LSTM(config_data, vocab)
    elif model_type == 'vRNN':
        model = get_model_vRNN(config_data, vocab)
    
    if model:
        print("Got model!")
        return model

def get_model_LSTM(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    max_seq_len = config_data['generation']['max_length']
    num_layers = 2
    vocab_size = len(vocab)
    
    if model_type != 'LSTM':
        return False
    
    encoder = Encoder(image_embedding_size=embedding_size)
    decoder = LSTMDecoder(vocab_size=vocab_size, hidden_size=hidden_size, word_embedding_size=embedding_size, num_layers=num_layers, max_seq_len=max_seq_len)
    model = MODEL(config_data, encoder, decoder, device)
        
    return model

def get_model_vRNN(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    max_seq_len = config_data['generation']['max_length']
    num_layers = 2
    vocab_size = len(vocab)
    
    if model_type != 'vRNN':
        return False
    
    encoder = Encoder(image_embedding_size=embedding_size)
    decoder = vRNNDecoder(vocab_size=vocab_size, hidden_size=hidden_size, word_embedding_size=embedding_size, num_layers=num_layers, max_seq_len=max_seq_len)
    model = MODEL(config_data, encoder, decoder, device)
        
    return model
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    