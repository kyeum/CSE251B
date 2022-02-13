import numpy as np
import torch
import matplotlib.pyplot as plt


def iou(pred, target, n_classes = 10):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for undefined class ("9")
  for cls in range(n_classes-1):  # last class is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = sum(np.logical_and(pred_inds,target_inds)) #DONE: complete this, number of agreements, TP
    union = sum(np.logical_or(pred_inds,target_inds)) #DONE: complete this, total TP + FP + FN
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection/union)) 

  return np.array(ious)

def pixel_acc(pred, target):
    #DONE: TODO complete this function, make sure you don't calculate the accuracy for undefined class ("9")
    total=0
    correct=0
    
    for i in range(len(target)): # count correct prediction and total for non-undefined classes
      if target[i] == 9: 
        continue # ignore undefined class
      
      if target[i] == pred[i]:
        correct+=1

      total+=1
    
    return total/correct


def iou_ey(pred, target, n_classes = 10):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for undefined class ("9")
  for cls in range(n_classes-1):  # last class is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = torch.sum(pred_inds & target_inds) #DONE: complete this, number of agreements, TP
    union = torch.sum(pred_inds | target_inds) #DONE: complete this, total TP + FP + FN    
    
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection/union)) 

  return ious

def pixel_acc_ey(pred, target):
    #DONE: TODO complete this function, make sure you don't calculate the accuracy for undefined class ("9")

  res = pred == target
  res_undef = torch.mean(res[target != 9].to(torch.float)) 

  return float(res_undef)


def visualize(model_name,test_loader):
    #TODO: load the best model and complete the rest of the function for testing
    fcn_model = torch.load(model_name)
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
    for k, v in test_dataset.color2class.items():
        class2color[v] = k    

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
  
