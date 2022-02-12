from basic_fcn import *
from dataloader import *
from utils import *
import torch.optim as optim
import torchvision
import time
from torch.utils.data import DataLoader
import torch
import gc
import copy
import matplotlib.pyplot as plt
from torch.utilss.data import ConcatDataset as concat



# TODO: Some missing values are represented by '__'. You need to fill these up.
train_dataset_original = TASDataset('tas500v1.1') 
train_dataset_crop = TASDataset('tas500v1.1', transform_mode = 1) 
train_dataset_rotate = TASDataset('tas500v1.1',transform_mode = 2) 
train_dataset_flip = TASDataset('tas500v1.1',transform_mode = 3) 

train_dataset = concat([train_dataset_original,train_dataset_crop,train_dataset_rotate,train_dataset_flip])

val_dataset = TASDataset('tas500v1.1', eval=True, mode='val')
test_dataset = TASDataset('tas500v1.1', eval=True, mode='test')


train_loader = DataLoader(dataset=train_dataset, batch_size= 4, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 4, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= 4, shuffle=False)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   
#TODOO!!  weight normalization -> add to normalized data to crossentrophyloss

# 4-a
criterion = nn.CrossEntropyLoss()# Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

# 4-b
#pressuring the network to categorize the infrequently seen classes. 
# weighted loss, 
# dice coefficient loss.




#criterion = nn.CrossEntropyLoss()# Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html





epochs = 20       
criterion = nn.CrossEntropyLoss()# Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
n_class = 10
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)


gpu_status = torch.cuda.is_available()
gpu_status

# +
if gpu_status : 
    device = torch.device('cuda') # determine which device to use (gpu or cpu)
    print("status : GPU")
else : 
    device = torch.device('cpu')
    
fcn_model.to(device)
# -



def train(epochs, learning_rate):
    optimizer = optim.Adam(fcn_model.parameters(), lr = learning_rate) # choose an optimizer

    best_iou_score = 0.0
    train_loss_record = []
    valid_loss_record = []
    
    for epoch in range(epochs):
        train_loss = []
        ts = time.time()

        for iter, (inputs, labels) in enumerate(train_loader):
            # reset optimizer gradients
            optimizer.zero_grad()
            # both inputs and labels have to reside in the same device as the model's

            inputs = inputs.to(device) #transfer the input to the same device as the model's
            labels = labels.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            outputs = fcn_model(inputs) #we will not need to transfer the output, it will be automatically in the same device as the model's!
            
            loss = criterion(outputs,labels) #calculate loss
            
            # backpropagate
            loss.backward()
            # update the weights
            optimizer.step()
            train_loss.append(loss.item())

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        train_loss_record.append(np.mean(train_loss))
        

        current_miou_score = val(epoch)
        
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            #save the best model
            torch.save(fcn_model,'latest_model')

    return train_loss_record, valid_loss_record

    

def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            output = fcn_model(input)

            loss = criterion(output,label) #calculate the loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 

            pred = torch.argmax(output, dim = 1) # Make sure to include an argmax to get the prediction from the outputs of your model

            mean_iou_scores.append(np.nanmean(iou_ey(pred, label, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc_ey(pred, label)) # Complete this function in the util


    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores)

def test():
    #TODO: load the best model and complete the rest of the function for testing
    fcn_model = torch.load('latest_model')
    fcn_model.eval()
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = input.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            output = fcn_model(input)

            loss = criterion(output,label) #calculate the loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 

            pred = torch.argmax(output, dim = 1) # Make sure to include an argmax to get the prediction from the outputs of your model

            mean_iou_scores.append(np.nanmean(iou_ey(pred, label, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc_ey(pred, label)) # Complete this function in the util


    print(f"Loss :is {np.mean(losses)}")
    print(f"IoU is {np.mean(mean_iou_scores)}")
    print(f"Pixel is {np.mean(accuracy)}")
    
    visualization()
    
    return 0


   
def visualization(): # visualization of the segmented output for the first image in the tes set overlaid on the image. colorcoding mapping in the dataloader.py
    class2color = {}
    for k, v in test_dataset.color2class.items():
        class2color[v] = k

    fcn_model = torch.load('latest_model')
    fcn_model.eval()  # Don't forget to put in eval mode !
    ts = time.time()
    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label, real_image) in enumerate(plot_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            output = fcn_model(input)
            pred = torch.argmax(output, dim=1) 
            
        
            imgs = []
            for rows in pred[0]:
                for col in rows:
                    col = int(col)
                    imgs.append(class2color[col])
            imgs = np.asarray(imgs).reshape(pred.shape[1], pred.shape[2], 3)
            outputimg = PIL.Image.fromarray(np.array(imgs, dtype=np.uint8))
            plt.axis('off')
            plt.imshow(real_image[0])
            plt.imshow(outputimg, alpha=0.8)
            
            plt.title('Output Image')
            plt.show()

            imgs = []
            for rows in label[0]:
                for col in rows:
                    col = int(col)
                    imgs.append(class2color[col])
            imgs = np.asarray(imgs).reshape(pred.shape[1], pred.shape[2], 3)
            outputimg = PIL.Image.fromarray(np.array(imgs, dtype=np.uint8))
            plt.axis('off')
            plt.imshow(real_image[0])
            plt.imshow(outputimg, alpha=0.8)
            
            plt.title('Label Image')
            plt.show()


if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train_record, valid_record = train(epochs, 0.0001)
    #test()
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()


