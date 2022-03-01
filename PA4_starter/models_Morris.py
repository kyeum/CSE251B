from torchvision import models
import torch.nn as nn
from constants import *
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import torch.utils.data as data


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("MODELS - DEVICE:", device)

class MODEL(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(MODEL, self).__init__()
        encoder.to(device)
        self.encoder = encoder
        decoder.to(device)
        self.decoder = decoder
    
    def forward(self, images, captions, lengths, sampling_mode, temperature):
        inference = False
        if captions is None:
            inference = True
        
        encoded_images = self.encoder(images)
        
        if inference:
            word_seq = self.decoder.generate_caption(encoded_images, lengths, sampling_mode = sampling_mode, temperature = temperature)
            return word_seq
        else:
            out = self.decoder(encoded_images, captions, lengths)
            return out
    
        
class Encoder(nn.Module):
    def __init__(self, image_embedding_size = 300):
        super(Encoder, self).__init__()
        ### Encoder
        '''
        self.encoder = models.resnet50(pretrained=True)

        ## Feature Extraction
        # Freeze all layers of pretrained encoder.
        for param in self.encoder.parameters():
            param.requires_grad = False

        fc_in_features = self.encoder.fc.in_features

        # Replace last layer with weight layer to embedding dimension.
        # Don't need to unfreeze as new linear layer has grads enabled.
        self.encoder.fc = nn.Linear(fc_in_features, image_embedding_size)
        '''
        self.model = models.resnet50(pretrained = True)
        self.in_feature = self.model.fc.in_features
        
        self.encoder = nn.Sequential(*list(self.model.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.linear = nn.Linear(in_features = self.in_feature, out_features = image_embedding_size)
        self.batch_norm = nn.BatchNorm1d(image_embedding_size)

    def forward(self, images):
        """
        Input = Batch_size x Image_height x Image_width
        Output = Batch_size x Image_embedding_size
        """
        encoded_images = self.encoder(images)
        encoded_images = encoded_images.squeeze()
        encoded_images = self.linear(encoded_images)
        encoded_images = self.batch_norm(encoded_images)
        return encoded_images
    
    
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size = -1, hidden_size = 512, word_embedding_size = 300, num_layers = 2, max_seq_len = 20):
        super(LSTMDecoder, self).__init__()
        
        # input: (N, L, H_in) = batch_size x seq_len x input_size
        # output: (N, L, D * H_out) = batch_size x seq_len, proj_size) 
        #     [here proj_size=hidden_size]
        # proj_size cannot be passed as hidden_size! 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder = nn.LSTM(input_size = word_embedding_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        ### Given a vocab word index, returns the word embedding
        # Input: (*) indices of embedding
        # Output: (*, H) where * is input shape and H = embedding_dim
        self.vocab2wordEmbed = nn.Embedding(num_embeddings = vocab_size, embedding_dim = word_embedding_size)
        
        # Converts from LSTM output to vocab size
        self.decoder2vocab = nn.Linear(hidden_size, vocab_size)
        
        # Softmax
        self.softmax = nn.Softmax(dim = 1)
        
        self.max_seq_len = max_seq_len
    
    # encoded_caption is in form of vocab word indices
    def forward(self, encoded_image, captions, lengths):
        
        encoded_image = encoded_image.unsqueeze(1)
        caption_embeddings = self.vocab2wordEmbed(captions)
        embeddings = torch.cat((encoded_image, caption_embeddings), 1)
        packed_input = pack_padded_sequence(embeddings, lengths, batch_first = True)
        
        # Get output
        packed_output, (ht, ct) = self.decoder(packed_input)
        output = self.decoder2vocab(packed_output.data)
        
        return output # shape: batch_sum_seq_len x vocab_size
    
    
    def generate_caption(self, encoded_image, lengths, states = None, sampling_mode = DETERMINISTIC, temperature = 0.1):
        caption_txt = []
        lstm_input = encoded_image.unsqueeze(1)
        
        for i in range(self.max_seq_len):
            output, states = self.decoder(lstm_input, states)
            output = self.decoder2vocab(output.squeeze(1))
            
            if sampling_mode == STOCHASTIC:
                prob = self.softmax(output / temperature)
                predicted = torch.multinomial(input = prob, num_samples = 1).squeeze(1)
            else:
                _, predicted = output.max(1)              

            caption_txt.append(predicted)
            lstm_input = self.vocab2wordEmbed(predicted).unsqueeze(1)
            
        caption_txt = torch.stack(caption_txt, 1)
        return caption_txt

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class vRNNDecoder(nn.Module):
    def __init__(self, vocab_size=-1, eos_tok_index=-1, hidden_size=512, word_embedding_size=300, num_layers=2, max_seq_len=22):
        super(RNNDecoder, self).__init__()
        
        # input: (N, L, H_in) = batch_size x seq_len x input_size
        # output: (N, L, D * H_out) = batch_size x seq_len, proj_size) 
        #     [here proj_size=hidden_size]
        # proj_size cannot be passed as hidden_size! 
        self.hidden_size = hidden_size
        self.decoder = nn.RNN(input_size=word_embedding_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity='relu', batch_first=True)
        ### Given a vocab word index, returns the word embedding
        # Input: (*) indices of embedding
        # Output: (*, H) where * is input shape and H = embedding_dim
        self.vocab2wordEmbed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_embedding_size)
        self.num_layers = num_layers
        # Converts from LSTM output to vocab size
        self.decoder2vocab = nn.Linear(hidden_size, vocab_size)
        
        # Softmax
        self.softmax = nn.Softmax(dim=2)
        
        self.max_seq_len = max_seq_len
        
        # Constants
        self.EOS_TOK_INDEX = eos_tok_index;
        
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    # encoded_caption is in form of vocab word indices
    def forward(self, encoded_image, captions):
    
        encoded_image = encoded_image.unsqueeze(1)
        # TODO: add case when captions is empty
        caption_embeddings = self.vocab2wordEmbed(captions)
        
#         print("caption_embed.shape:", caption_embeddings.shape)
        
        batch_size = encoded_image.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        initial_hidden_states = (h0, c0)
        
        
        # TODO: Initialize our LSTM with the image encoder output to bias our prediction.
        temp, image_hidden_states = self.decoder(encoded_image, initial_hidden_states)
        
        # Get output and hidden states
        out, hidden = self.decoder(caption_embeddings, image_hidden_states)
#         print("out1:", out.shape)
        
        out = self.decoder2vocab(out)
#         print("out.shape:", out.shape)

        
        return out # shape: batch_size x seq_len x vocab_size


    
    def generate_caption_ey(self, encoded_image,states=None,sampling_mode=STOCHASTIC, max_seq_len=22, temperature = 1):
        start_input = torch.ones((encoded_image.shape[0], 1)).long().to('cuda')
        # USE index instead of <start> -> not efficient!! #torch.tensor(1).to('cuda') # this is the '<start>'
        start_input = self.vocab2wordEmbed(start_input)
        encoded_image = encoded_image.unsqueeze(1)
        lstm_input = encoded_image
        caption_txt = []
        batch_size = encoded_image.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        initial_hidden_states = (h0, c0)
        temp,imh_hidden_states = self.decoder(lstm_input, initial_hidden_states)
        for i in range(max_seq_len):
            output,imh_hidden_states = self.decoder(start_input, imh_hidden_states)
            #output, _ = self.decoder(output,initial_hidden_states)
            output = self.decoder2vocab(output)
            if sampling_mode == STOCHASTIC :
                output = self.softmax(output/temperature)
                predicted = torch.multinomial(input=output, num_samples=1, replacement=True)
            else:
                _, predicted = output.max(1)
            caption_txt.append(predicted)
            start_input = self.vocab2wordEmbed(predicted)
            start_input = torch.unsqueeze(start_input, 1)
        caption_txt = torch.stack(caption_txt, 1)
        print(caption_txt)
        return captions
    

        
        
        
        
        
        
        
        
        
        
        
        