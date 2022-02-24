from torchvision import models
import torch.nn as nn

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
    def __init__(self, vocab_size=-1, eos_tok_index=-1, hidden_size=512, word_embedding_size=300, num_layers=2):
        super(LSTMDecoder, self).__init__()
        
        # input: (N, L, H_in) = batch_size x seq_len x input_size
        # output: (N, L, D * H_out) = batch_size x seq_len, proj_size) 
        #     [here proj_size=hidden_size]
        # proj_size cannot be passed as hidden_size! 
        self.decoder = nn.LSTM(input_size=word_embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        ### Given a vocab word index, returns the word embedding
        # Input: (*) indices of embedding
        # Output: (*, H) where * is input shape and H = embedding_dim
        self.vocab2wordEmbed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_embedding_size)
        
        # Converts from LSTM output to vocab size
        self.decoder2vocab = nn.Linear(hidden_size, vocab_size)
        
        # Softmax
        self.softmax = nn.Softmax(dim=2)
        
        # Constants
        self.EOS_TOK_INDEX = eos_tok_index;
        self.DETERMINISTIC = 0
        self.STOCHASTIC = 1
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    # encoded_caption is in form of vocab word indices
    def forward(self, encoded_image, captions):
        # TODO: add case when captions is empty
        caption_embeddings = self.vocab2wordEmbed(captions)
        
        # TODO: Initialize our LSTM with the image encoder output to bias our prediction.
        
        # TODO: Weight initialization
        
        
        # Get output and hidden states
        out, hidden = self.decoder(caption_embeddings, encoded_image)
        
        out = self.decoder2vocab(out)
        
        # Get probabilities of each word
        out = self.softmax(out)
        return out # shape: batch_size x seq_len x vocab_size
        
    # Inference
    def generate_caption(self, encoded_image, sampling_mode=1, max_seq_len=12, end_at_eos=True):
        """
        Sampling_mode = 0 deterministic
                      = 1 stochastic
        """
        word_seq = [] # indices of words
        
        for i in range(max_seq_len):
            out = self.forward(encoded_image, word_seq)
            # Get word indice based on sampling_mode
            if sampling_mode == self.DETERMINISTIC:
                wordIndice = out[2].argmax()
                word_seq.append(wordIndice)
            else:
                gen_word_index = torch.multinomial(input=out, num_samples=1, replacement=True)
                word_seq.append(gen_word_index)
                
            if end_at_eos and wordIndice == self.EOS_TOK_INDEX:
                return word_seq
            
        return word_seq
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        