################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
from torchvision import models
import torch.nn as nn
from models import LSTM, LSTMEncoder, LSTMDecoder
from constants import *
import torch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("MODEL_FAC - DEVICE:", device)

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    print("Getting model...")
    model_type = config_data['model']['model_type']
    
    if model_type == 'LSTM':
        model = get_model_LSTM(config_data, vocab)
        if model:
            print("Got model!")
            return model
        
    # You may add more parameters if you want

    raise NotImplementedError("Model Factory Not Implemented")

def get_model_LSTM(config_data, vocab, DEBUG=True):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    max_seq_len = config_data['generation']['max_length'] + 2
    num_layers = 2
    vocab_size = len(vocab)
    
    if model_type != 'LSTM':
        return False
    
    encoder = LSTMEncoder(image_embedding_size=embedding_size)
    
    # Get EOS token index from vocab for decoder to know when to stop generating
    EOS_TOK_INDEX = vocab(EOS_TOK)
    
    decoder = LSTMDecoder(vocab_size=vocab_size, eos_tok_index=EOS_TOK_INDEX, hidden_size=hidden_size, word_embedding_size=embedding_size, num_layers=num_layers, max_seq_len=max_seq_len)
    model = LSTM(config_data, encoder, decoder, device)
        
    return model

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    