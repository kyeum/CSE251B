import numpy as np
import torch


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


def pixel_acc_ey(pred, target):
    #DONE: TODO complete this function, make sure you don't calculate the accuracy for undefined class ("9")

  res = pred == target
  res_undef = torch.mean(res[target != 9].to(torch.float)) 

  return float(res_undef)