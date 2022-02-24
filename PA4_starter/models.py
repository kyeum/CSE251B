from torchvision import models
import torch.nn as nn
from constants import *
import torch

class LSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super(LSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, images, captions):
        inference = False
        if captions is None:
            inference = True
        
        encoded_images = self.encoder(images)
        
        print("image_shape:", images.shape)
        print("enc_image_shape:", encoded_images.shape)
        print("captions.shape:", captions.shape)
        # image_shape: torch.Size([64, 3, 256, 256])
        # enc_image_shape: torch.Size([64, 300])
        # captions.shape: torch.Size([64, 22])
        
        if inference:
            word_seq = self.decoder.generate_caption(encoded_images, sampling_mode=STOCHASTIC, max_seq_len=20, end_at_eos=True)
            return word_seq
        else:
            out = self.decoder(encoded_images, captions)
        
        return out
        
class LSTMEncoder(nn.Module):
    def __init__(self, image_embedding_size=300):
        super(LSTMEncoder, self).__init__()
        ### Encoder
        self.encoder = models.resnet50(pretrained=True)

        ## Feature Extraction
        # Freeze all layers of pretrained encoder.
        for param in self.encoder.parameters():
            param.requires_grad = False

        fc_in_features = self.encoder.fc.in_features

        # Replace last layer with weight layer to embedding dimension.
        # Don't need to unfreeze as new linear layer has grads enabled.
        self.encoder.fc = nn.Linear(fc_in_features, image_embedding_size)
        
    def forward(self, images):
        """
        Input = Batch_size x Image_height x Image_width
        Output = Batch_size x Image_embedding_size
        """
        return self.encoder(images)
    
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size=-1, eos_tok_index=-1, hidden_size=512, word_embedding_size=300, num_layers=2, max_seq_len=20):
        super(LSTMDecoder, self).__init__()
        
        # input: (N, L, H_in) = batch_size x seq_len x input_size
        # output: (N, L, D * H_out) = batch_size x seq_len, proj_size) 
        #     [here proj_size=hidden_size]
        # proj_size cannot be passed as hidden_size! 
        self.hidden_size = hidden_size
        self.decoder = nn.LSTM(input_size=word_embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
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
        
        print("caption_embed.shape:", caption_embeddings.shape)
        
        batch_size = encoded_image.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        initial_hidden_states = (h0, c0)
        # TODO: Initialize our LSTM with the image encoder output to bias our prediction.
        temp, image_hidden_states = self.decoder(encoded_image, initial_hidden_states)
        # TODO: Weight initialization
        print("temp.shape:", temp.shape)
        
        # Get output and hidden states
        out, hidden = self.decoder(caption_embeddings, image_hidden_states)
        print("out1:", out.shape)
        
        out = self.decoder2vocab(out)
        print("out.shape:", out.shape)
        
        # Get probabilities of each word
        out = self.softmax(out)
        return out # shape: batch_size x seq_len x vocab_size
        
    # Inference
    def generate_caption(self, encoded_image, sampling_mode=STOCHASTIC, end_at_eos=True):
        """
        Sampling_mode = 0 deterministic
                      = 1 stochastic
        """
        word_seq = [] # indices of words
        
        for i in range(self.max_seq_len):
            out = self.forward(encoded_image, word_seq)
            # Get word indice based on sampling_mode
            if sampling_mode == DETERMINISTIC:
                wordIndice = out[2].argmax()
                word_seq.append(wordIndice)
            else:
                gen_word_index = torch.multinomial(input=out, num_samples=1, replacement=True)
                word_seq.append(gen_word_index)
                
            if end_at_eos and wordIndice == self.EOS_TOK_INDEX:
                return word_seq
            
        return word_seq
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        