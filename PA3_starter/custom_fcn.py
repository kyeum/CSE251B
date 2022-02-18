import torch.nn as nn
from torchvision import models
from torchvision.transforms import functional as F
   
class Custom(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.leakyRelu = nn.LeakyReLU(0.01, inplace=True)
        
        self.dropout = nn.Dropout(p=0.2)
        self.Maxpool = nn.MaxPool2d(kernel_size=3,stride=1, padding=1) # maxpool 2x2 with stride 1
        
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        
    def forward(self, x):
        x1 = self.bnd1(self.leakyRelu(self.conv1(x)))
        
        x2 = self.bnd2(self.leakyRelu(self.conv2(x1)))
        # Downsize to output x2 size and average over channels
        down1 = F.resize(img=x, size=(x2.shape[2], x2.shape[3])).mean(dim=1, keepdim=True)
        # Duplicate for number of channels
        down1 = down1.expand(down1.shape[0], x2.shape[1], down1.shape[2], down1.shape[3])
        x2 += down1

        x3 = self.bnd3(self.leakyRelu(self.conv3(x2)))
        x3 = self.Maxpool(x3)
        
        x4 = self.bnd4(self.leakyRelu(self.conv4(x3)))
        #x4 = self.Maxpool(x4)
        down2 = F.resize(img=x2, size=(x4.shape[2], x4.shape[3])).mean(dim=1, keepdim=True)
        down2 = down2.expand(down2.shape[0], x4.shape[1], down2.shape[2], down2.shape[3])
        x4 += down2

        out_encoder =  self.bnd5(self.leakyRelu(self.conv5(x4)))

        # Complete the forward function for the rest of the encoder : Completed - ey

        y1 = self.bn1(self.leakyRelu(self.deconv1(self.relu(out_encoder))))   
        
        y2 = self.bn2(self.leakyRelu(self.deconv2(y1)))  

        # Upsize to output y2 size and average over channels
        up1 = F.resize(img=out_encoder, size=(y2.shape[2], y2.shape[3])).mean(dim=1, keepdim=True)
        # Duplicate for number of channels
        up1 = up1.expand(up1.shape[0], y2.shape[1], up1.shape[2], up1.shape[3])  
        y2 += up1
        
        y3 = self.bn3(self.dropout(self.leakyRelu(self.deconv3(y2))))    
        
        y4 = self.bn4(self.dropout(self.leakyRelu(self.deconv4(y3))))  
        up2 = F.resize(img=y2, size=(y4.shape[2], y4.shape[3])).mean(dim=1, keepdim=True)
        up2 = up2.expand(up2.shape[0], y4.shape[1], up2.shape[2], up2.shape[3])  
        y4 += up2

        out_decoder = self.bn5(self.leakyRelu(self.deconv5(y4)))    

        # Complete the forward function for the rest of the decoder
        score = self.classifier(out_decoder)    

        return score  # size=(N, n_class, x.H/1, x.W/1)

class Custom2(nn.Module):
    
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.leakyRelu = nn.LeakyReLU(0.01, inplace=True)
        
        self.dropout = nn.Dropout(p=0.5)
        self.Maxpool = nn.MaxPool2d(kernel_size=3,stride=1, padding=1) # maxpool 2x2 with stride 1
        
        self.conv1   = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        
    def forward(self, x):

        x1 = self.bnd1(self.leakyRelu(self.conv1(x)))
        x2 = self.bnd2(self.leakyRelu(self.conv2(x1)))
        x2 = self.Maxpool(x2)
        x3 = self.bnd3(self.leakyRelu(self.conv3(x2)))
        x3 = self.Maxpool(x3)
        x4 = self.bnd4(self.leakyRelu(self.conv4(x3)))

        out_encoder =  self.bnd5(self.leakyRelu(self.conv5(x4)))
        # Complete the forward function for the rest of the encoder : Completed - ey


        y1 = self.bn1(self.leakyRelu(self.deconv1(self.relu(out_encoder))))   
        y2 = self.bn2(self.leakyRelu(self.deconv2(y1)))    
        y3 = self.bn3(self.leakyRelu(self.deconv3(y2)))    
        y4 = self.bn4(self.leakyRelu(self.deconv4(y3)))    
        out_decoder = self.bn5(self.relu(self.deconv5(y4)))    

        # Complete the forward function for the rest of the decoder
        
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)
