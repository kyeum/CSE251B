import torch.nn as nn
import torch
from torchvision import models

   
class UNET(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        #1st
        #repeated two 3x3 conv(unpadded)
        #RELU
        #2x2 max pool, with stride 2, doubled the number of feature channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2) # maxpool 2x2 with stride 2 for downsampling
 
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )              
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )                
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )             
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )             
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )     
        
        #2nd 
        #up sampling
        #2x2 conv - concat will be in forward
        #
        self.Up5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )       
        self.upconv5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )     
        self.Up4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )       
        self.upconv4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )   
        self.Up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )       
        self.upconv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )   
        self.Up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )       
        self.upconv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )   
        self.classifier = nn.Conv2d(64,n_class,kernel_size=1)

    def forward(self, x):    
        x1 = self.conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.conv2(x2)      
        
        x3 = self.Maxpool(x2)
        x3 = self.conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.conv5(x5)

        y1 = self.Up5(x5)
        merge1 = torch.cat((x4,y1),dim=1)    # concat corresponding crop feature map from contracting path
        y1 = self.upconv5(merge1)            # two 3x3 conv , . 
        
        y2 = self.Up4(y1)
        merge2 = torch.cat((x3,y2),dim=1)
        y2 = self.upconv4(merge2)
        
        y3 = self.Up3(y2)
        merge3 = torch.cat((x2,y3),dim=1)
        y3 = self.upconv3(merge3)
        
        y4 = self.Up2(y3)
        merge4 = torch.cat((x1,y4),dim=1)
        y4 = self.upconv2(merge4)
        
        score = self.classifier(y4)  # In total the network has 23 convolutional layers.                
        return score 

