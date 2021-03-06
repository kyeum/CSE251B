################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
import nltk
import caption_utils
import re
from datetime import datetime
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Experiment_LSTM(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__max_length = config_data['generation']['max_length']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__lr = config_data['experiment']['learning_rate']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__min_val_loss = float('inf')

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = torch.nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=self.__lr, weight_decay = 0.0001)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        print("In training loop...")
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model() # latest model only
        print("Finished training!")
        
    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0

        for i, (images, captions,_) in enumerate(self.__train_loader):
            images = images.to(device)
            captions = captions.to(device)
            self.__optimizer.zero_grad()
            y = self.__model(images,captions)
            # y : 8x 20 x vocab -> permute 8 x vocab x 20
            
            y = y.permute(0,2,1) # batch size change  # caption : 8 x 20
            # TODO : caption start from 1 - end, y start from 0 : end -1? for LSTM ???? IG???? 
            
            
            loss = self.__criterion(y, captions)
            training_loss += loss.item()
            loss.backward()
            self.__optimizer.step()
            if i % 1000 == 0: 
                train_str = "Epoch: {}, Batch: {} train_loss: {}".format(self.__current_epoch+1,i+1,loss)
                self.__log(train_str)
        training_loss = training_loss/len(self.__train_loader)
        
        return training_loss
            

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0
        bleu4 = 0

        with torch.no_grad():
            for i, (images, captions, img_ids) in enumerate(self.__val_loader):
                images = images.to(device)
                captions = captions.to(device)
                y = self.__model(images, captions)                
                y = y.permute(0,2,1) # batch size change  # caption : 8 x 20
                
                loss = self.__criterion(y, captions)
                val_loss += loss.item()
                
                if i % 100 == 0 : 
                    valid_str = "Epoch: {}, Batch: {} valid_loss: {}".format(self.__current_epoch+1,i+1,loss)
                    self.__log(valid_str) 
                    
            val_loss = val_loss/len(self.__val_loader)                   
            
            if(val_loss < self.__min_val_loss):
                self.__min_val_loss = val_loss
                print("Saving the model in {} epochs".format(self.__current_epoch+1))
                self.__best_model = self.__model
                self.__save_model(name = 'best_model')
                  
        return val_loss

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self, temperature):
        state_dict = torch.load(os.path.join(self.__experiment_dir, 'best_model'))
        self.__model.load_state_dict(state_dict['model'])
        self.__optimizer.load_state_dict(state_dict['optimizer'])
        self.__model.eval()
        self.__model.config_data['generation']['temperature'] = temperature
        
        test_loss = 0
        bleu1 = 0
        bleu4 = 0
        pred_text = []
        cnt = 0
        
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.to(device)
                captions = captions.to(device)
                y = self.__model(images, captions)  

                y = y.permute(0,2,1) # batch size change  # caption : 8 x 20
                
                loss = self.__criterion(y, captions)                
                test_loss += loss.item()

                # TODO: probably need to pad output to match true_size
                pred_text =  self.__model(images,None) # 8 x 20 x 1
                #print(pred_text.shape)                 #
                
                #pred_text = pred_text.permute(0,2,1) # batch size change  # caption : 8 x 20 x 1

                #print("pred_text shape",pred_text.shape,"img_ids", len(img_ids))# 8 
                
                for pred_, img_id in zip(pred_text, img_ids): # total 8 batches 
                    txt_true = []
                    for i in self.__coco_test.imgToAnns[img_id]:
                        caption = str(i['caption']).lower()
                        caption = re.sub(r'[^\w\s]', '', caption)
                        cap2tok = nltk.tokenize.word_tokenize(caption)
                        txt_true.append(cap2tok)
                
                    cnt = cnt + 1
                    
                    #1x20
                    pred_ = self.__cap2word(pred_, self.__vocab, max_length = self.__max_length)
                    if pred_ != "":
                        pred_ = pred_.split(' ')
                    
                    #print("bleu1:",caption_utils.bleu1(txt_true, pred_))
                    #print("true text:",txt_true)
                    #print("pred:",pred_)
                    bleu1 += caption_utils.bleu1(txt_true, pred_)
                    bleu4 += caption_utils.bleu4(txt_true, pred_)
                    
        test_loss = test_loss / cnt
        
        bleu1 = bleu1 / cnt
        bleu4 = bleu4 / cnt
        
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss, bleu1, bleu4)
        self.__log(result_str)

        return test_loss, bleu1, bleu4
    
    def __save_model(self, name = 'latest_model.pt'):
        root_model_path = os.path.join(self.__experiment_dir, name)
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
        
    def __cap2word(self, caption, vocab, max_length = 20):
        """
            Here, we generate the text caption for the given batch of images
        """
        batch_caption = []
        words = []
  
        img_caption = caption.cpu().numpy()
        
        for word_id in img_caption:
            word = vocab.idx2word[word_id[0]]
            
            if word == "<start>":
                continue
            elif word == "<end>":
                sentence = ' '.join(words)
                sentence = sentence.lower()
                batch_caption.append(sentence)
                words = []
                break
            else:
                words.append(word)
            
            # debug for max
            if(len(words) == max_length):
#               print('max')
                sentence = ' '.join(words)
                sentence = sentence.lower()
                batch_caption.append(sentence)
                words = []
        
        batch_caption = [re.sub(r"(^[^\w]+)|([^\w]+$)", "", b) for b in batch_caption]
        
        if len(batch_caption) > 0:
            return batch_caption[0]
        return ""
    
    
    def visualize(self, temperature):
        state_dict = torch.load(os.path.join(self.__experiment_dir, 'best_model'))
        self.__model.load_state_dict(state_dict['model'])
        self.__optimizer.load_state_dict(state_dict['optimizer'])
        self.__model.eval()
        self.__model.config_data['generation']['temperature'] = temperature
        
        test_loss = 0
        bleu1 = 0
        bleu4 = 0
        pred_text = []
        cnt = 0;
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.to(device)
                captions = captions.to(device)
                y = self.__model(images, captions)  

                y = y.permute(0,2,1) # batch size change  # caption : 8 x 20
                
                #captions = captions[:,1:]
                #y = y[:, :, :-1]     
                
                loss = self.__criterion(y, captions)                
                test_loss += loss

                # TODO: probably need to pad output to match true_size
                pred_text =  self.__model(images,None) # 8 x 20 x 1
                #print(pred_text.shape)                 #
                
                #pred_text = pred_text.permute(0,2,1) # batch size change  # caption : 8 x 20 x 1

                #print("pred_text shape",pred_text.shape,"img_ids", len(img_ids))# 8 
                batch = 0
                for pred_, img_id in zip(pred_text, img_ids): # total 8 batches 
                    #print(pred_.shape)
                    #break

                    txt_true = []
                    for i in self.__coco_test.imgToAnns[img_id]:
                        caption = str(i['caption']).lower()
                        caption = re.sub(r'[^\w\s]', '', caption)
                        txt_true.append(caption)
                
                    cnt = cnt + 1
                    
                    #1x20
                    pred_ = self.__cap2word(pred_,self.__vocab, max_length = self.__max_length)
                    
                    b1 = caption_utils.bleu1(txt_true, pred_)
                    b4 = caption_utils.bleu4(txt_true, pred_)
                    
                    if (b1 < 50):
                        plt.imshow(images[batch].cpu().permute(1, 2, 0))
                        plt.axis('off')
                        plt.show()
                        for it in range(0, 5):
                            print('True caption : {}'.format(txt_true[it]))

                        print('predict caption : {}'.format(pred_))
                        print('BLEU-1 : {}'.format(b1))
                        print('BLEU-4 : {}'.format(b4))

                    batch = batch + 1

                    #print("bleu1:",caption_utils.bleu1(txt_true, pred_))
                    #print("true text:",txt_true)
                    #print("pred:",pred_)

                    bleu1 += b1
                    bleu4 += b4

        test_loss = test_loss / cnt
        
        bleu1 = bleu1 / cnt
        bleu4 = bleu4 / cnt

        return 0
        
