from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

import random
import os
import numpy as np
import glob
import PIL
from tqdm import tqdm 

import torch

def rgb2int(arr):
    """
    Convert (N,...M,3)-array of dtype uint8 to a (N,...,M)-array of dtype int32
    """
    return arr[...,0]*(256**2)+arr[...,1]*256+arr[...,2]

def rgb2vals(color, color2ind):
   
    int_colors = rgb2int(color)
#     print("int_color.shpe:", int_colors.shape)
    int_keys = rgb2int(np.array(list(color2ind.keys()), dtype='uint8'))
    int_array = np.r_[int_colors.ravel(), int_keys]
    uniq, index = np.unique(int_array, return_inverse=True)
    color_labels = index[:int_colors.size]
#     print("color2ind:", -len(color2ind))
    key_labels = index[-len(color2ind):]

    colormap = np.empty_like(int_keys, dtype='int32')
#     print("colormap.shape:", colormap.shape)
#     print("key_labls:", max(key_labels))
    colormap[key_labels] = list(color2ind.values())
    out = colormap[color_labels].reshape(color.shape[:2])

    return out


class TASDataset(Dataset):
    def __init__(self, data_folder, eval_mode=False, mode=None, transform_mode = 0): # mode 0 : origin, mode 1 : crop, mode 2 :flip, mode 3: rotate
        self.data_folder = data_folder
        self.eval_mode = eval_mode
        self.mode = mode
        self.transform_mode = transform_mode

        # You can use any valid transformations here
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))                 
                              ])
        # The following transformation normalizes each channel using the mean and std provided
        self.cur_transforms = [
        ]
        self.toTensor = transforms.Compose([
            transforms.ToTensor(),
                              ])
        self.ToPILImage = transforms.Compose([transforms.ToPILImage()])
        self.transform = transforms.Compose(self.cur_transforms)

        # we will use the following width and height to resize
        self.width = 768
        self.height = 384


        self.color2class = {
                #terrain
                (192,192,192): 0, (105,105,105): 0, (160, 82, 45):0, (244,164, 96): 0, \
                #vegatation
                ( 60,179,113): 1, (34,139, 34): 1, ( 154,205, 50): 1, ( 0,128,  0): 1, (0,100,  0):1, ( 0,250,154):1, (139, 69, 19): 1,\
                #construction
                (1, 51, 73):2, ( 190,153,153): 2, ( 0,132,111): 2,\
                #vehicle
                (0,  0,142):3, ( 0, 60,100):3, \
                #sky
                (135,206,250):4,\
                #object
                ( 128,  0,128): 5, (153,153,153):5, (255,255,  0 ):5, \
                #human
                (220, 20, 60):6, \
                #animal
                ( 255,182,193):7,\
                #void
                (220,220,220):8, \
                #undefined
                (0,  0,  0):9
        }

        self.input_folder = os.path.join(self.data_folder, 'train')
        self.label_folder = os.path.join(self.data_folder, 'train_labels')

        if self.eval_mode:
            self.input_folder = os.path.join(self.data_folder, 'val')
            self.label_folder = os.path.join(self.data_folder, 'val_labels')
        
        image_names = os.listdir(self.input_folder)
        
        invalid_labels = ['1537962190852671077.png','1539600515553691119.png', '1539600738459369245.png','1539600829359771415.png','1567611260762673589.png']
            
        image_names = list(set(image_names).difference(set(invalid_labels)))
            
        self.paths = [(os.path.join(self.input_folder, i), os.path.join(self.label_folder, i)) for i in image_names]
        
        if self.mode == 'val': # use first 50 images for validation
            self.paths = self.paths[:50]
            
        elif self.mode == 'test': # use last 50 images for test
            self.paths = self.paths[50:]
    def add_rand_crop(self):
        self.cur_transforms.insert(1, transforms.RandomResizedCrop(size=(384, 768), scale=(0.08, 1.0), ratio=(0.75, 1.3333)))
        self.transform = transforms.Compose(self.cur_transforms)
    
    def add_rand_rot(self):
        self.cur_transforms.insert(1, transforms.RandomRotation((-5, 5), fill=0))
        self.transform = transforms.Compose(self.cur_transforms)
        
    def add_horz_flip(self, p=0.5):
        self.cur_transforms.insert(1, transforms.RandomHorizontalFlip(p))
        self.transform = transforms.Compose(self.cur_transforms)
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):        
        ## updated here for crop, flip, rptate
        
        image = np.asarray(PIL.Image.open(self.paths[idx][0]).resize((self.width, self.height)))
        mask_image = np.asarray(PIL.Image.open(self.paths[idx][1]).resize((self.width, self.height), PIL.Image.NEAREST))

    
#         seed = np.random.randint(2147483647)
        if self.transform:
            image = self.normalize_transform(image).float()
            
                   
        mask_image = F.to_tensor(mask_image)
        
        # 50% chance to flip horizontally
        if random.random() < 0.5:
            image = F.hflip(image)
            mask_image = F.hflip(mask_image)
        # 50% chance to rotate
        if random.random() < 0.5:
            rot_deg = random.randint(-5, 5)
            image = F.rotate(image, rot_deg)
            mask_image = F.rotate(mask_image, rot_deg)
            
        # 50% to center crop and resize
        if random.random() < 0.5:
            image = F.center_crop(image, output_size=(192, 384))
            mask_image = F.center_crop(mask_image, output_size=(192, 384))
            image = F.resize(image, size=(self.height, self.width))
            # NEED THE INTERPOLATION MODE OR ELSE IT BREAKS RGB2VALS()
            mask_image = F.resize(mask_image, size=(self.height, self.width), interpolation=transforms.InterpolationMode.NEAREST)
            
        mask_image = np.asarray(F.to_pil_image(mask_image), dtype="uint8")
        
#         print("mask_image2:", mask_image.shape)
#         print("type_mask_image2:", type(mask_image))
#         print("mask_image_max2():", mask_image.max())
#         print("mask_image_min2():", mask_image.min())   
        mask =  rgb2vals(mask_image, self.color2class)
            
        if self.mode == 'test':
            return image, mask, np.asarray(PIL.Image.open(self.paths[idx][0]).resize((self.width, self.height)))

        return image, mask