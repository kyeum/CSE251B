# Used for Q4 onwards

from basic_fcn import *
from dataloader_4 import *
from utils import *
import torch.optim as optim
import torchvision
import time
from torch.utils.data import DataLoader
import torch
import gc
import copy
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset as concat
from tqdm import tqdm

print("in starter_4")

# TODO: Some missing values are represented by '__'. You need to fill these up.
train_dataset = TASDataset('tas500v1.1') 
# train_dataset.add_rand_crop()
# train_dataset.add_rand_rot()
# train_dataset.add_horz_flip()

val_dataset = TASDataset('tas500v1.1', eval_mode=True, mode='val')
test_dataset = TASDataset('tas500v1.1', eval_mode=True, mode='test')

batch_size = 8
train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=False)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if(m.bias is not None):
            torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   
#TODOO!!  weight normalization -> add to normalized data to crossentrophyloss



criterion = nn.CrossEntropyLoss()# Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html




epochs = 20       
n_class = 10


gpu_status = torch.cuda.is_available()
print("GPU_STATUS:", gpu_status)

if gpu_status : 
    device = torch.device('cuda') # determine which device to use (gpu or cpu)
else : 
    device = torch.device('cpu')



def train(fcn_model, epochs, learning_rate, save_fp="latest_model_4"):
    print("in train")
    optimizer = optim.Adam(fcn_model.parameters(), lr = learning_rate) # choose an optimizer
    best_iou_score = 0.0
    train_loss_record = []
    valid_loss_record = []
    
    for epoch in tqdm(range(epochs)):
        train_loss = []
        ts = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
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

            if i % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, i, loss.item()))


        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        train_loss_record.append(np.mean(train_loss))


        current_miou_score,valid_loss = val(fcn_model,epoch)
        valid_loss_record.append(valid_loss)

        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            #save the best model
            torch.save(fcn_model,save_fp)

    return train_loss_record, valid_loss_record

    

def val(fcn_model, epoch):
    print("in val")
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for i, (inputs, labels) in enumerate(val_loader):
            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device) #transfer the input to the same device as the model's
            labels = labels.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            output = fcn_model(inputs)

            loss = criterion(output,labels) #calculate the loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 

            pred = torch.argmax(output, axis = 1) # Make sure to include an argmax to get the prediction from the outputs of your model

            mean_iou_scores.append(np.nanmean(iou_ey(pred, labels, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc_ey(pred, labels)) # Complete this function in the util


    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores), np.mean(losses)

def test(fcn_model):
    print("in test")
    #TODO: load the best model and complete the rest of the function for testing
    fcn_model.eval()
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for i, (inputs, labels, rawimg) in enumerate(test_loader):

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device) #transfer the input to the same device as the model's
            labels = labels.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            output = fcn_model(inputs)

            loss = criterion(output,labels) #calculate the loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 

            pred = torch.argmax(output, axis = 1) # Make sure to include an argmax to get the prediction from the outputs of your model

            mean_iou_scores.append(np.nanmean(iou_ey(pred, labels, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc_ey(pred, labels)) # Complete this function in the util


    print(f"Loss :is {np.mean(losses)}")
    print(f"IoU is {np.mean(mean_iou_scores)}")
    print(f"Pixel is {np.mean(accuracy)}")
    return 0

def visualize(model_name,test_loader,device):
    #TODO: load the best model and complete the rest of the function for testing
    fcn_model = torch.load(model_name).to(device)
    fcn_model.eval()
    inputimg = []
    pred = []
    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
        for iter, (input, label, orgin_img) in enumerate(test_loader):
            inputimg = orgin_img[0]
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's
            output = fcn_model(input)
            pred = torch.argmax(output, axis = 1) # Make sure to include an argmax to get the prediction from the outputs of your model

    class2color = {}
    for r, c in test_dataset.color2class.items():
        class2color[c] = r    

    imgs = []
    for row in pred[0]:
        for col in row:
            imgs.append(class2color[int(col)])
    imgs = np.asarray(imgs).reshape(pred.shape[1], pred.shape[2], 3)
    outputimg = PIL.Image.fromarray(np.array(imgs, dtype=np.uint8))
    plt.axis('off')
    plt.imshow(inputimg)
    plt.imshow(outputimg, alpha=0.5)

    plt.title('Output Image')
    plt.show()
    
    imgs = []
    for rows in label[0]:
        for col in rows:
            imgs.append(class2color[int(col)])
    imgs = np.asarray(imgs).reshape(pred.shape[1], pred.shape[2], 3)
    outputimg = PIL.Image.fromarray(np.array(imgs, dtype=np.uint8))
    plt.axis('off')
    plt.imshow(inputimg)
    plt.imshow(outputimg, alpha=0.5)

    plt.title('Label Image')
    plt.show()    

    return 0
  


if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
    test()
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()
