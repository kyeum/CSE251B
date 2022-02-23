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
    
    if model_type is 'LSTM':
        return get_model_LSTM(config, vocab)
        
    # You may add more parameters if you want

    raise NotImplementedError("Model Factory Not Implemented")

def get_model_LSTM(config_data, vocab, DEBUG=True):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    num_layers = 2
    vocab_size = len(vocab) # TODO
    
    if model_type is not 'LSTM':
        return False
        
    ### Encoder
    encoder = models.resnet50(pretrained=True)
    
    ## Feature Extraction
    # Freeze all layers of pretrained encoder
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
    fc_in_features = encoder.fc.in_features
    
    if DEBUG:
        print("fc_in_features:", fc_in_features)
        print("embedding_size:", embedding_size)

    # Replace last layer with weight layer to embedding dimension
    encoder.fc = nn.Linear(fc_in_features, embedding_size)
    
    # training
    seq = [image_emb, sen_embed]
    # inf
    seq = [image_emb]
    
    ### Decoder
    # input shape = batch_size x sequence_length x input_size
    decoder = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, proj_size=hidden_size, batch_first=True)
    
    decoder2vocab = nn.Linear(hidden_size, vocab_size)
    
# batch x seq_len (2) x feature_size

    # Image to image embed
    # Word to word embedding
    model = nn.Sequential([
        encoder,
        decoder,
        decoder2vocab,
        nn.Softmax(dim=2)
    ])
    # output shape = batch_size x sequence_length x vocab_size
        
#     raise NotImplementedError("Model Factory LSTM Not Implemented")
    return model

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    