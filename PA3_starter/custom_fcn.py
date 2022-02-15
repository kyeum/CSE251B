import torch.nn as nn
from torchvision import models

   
class Custom(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.leakyRelu = nn.LeakyReLU(0.01, inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.down = nn.Identity()
        self.up = nn.Identity()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2) # maxpool 2x2 with stride 2 for downsampling
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

        identity1 = self.down(x)
        x1 = self.bnd1(self.leakyRelu(self.conv1(x)))
        x1 += identity1
        x1 = self.Maxpool(x1)
        
        identity2 = self.down(x1)
        x2 = self.bnd2(self.leakyRelu(self.conv2(x1)))
        x2 += identity2
        x2 = self.Maxpool(x2)

        identity3 = self.down(x2)
        x3 = self.bnd3(self.dropout(self.leakyRelu(self.conv3(x2))))
        x3 += identity3
        x3 = self.Maxpool(x3)
        
        identity4 = self.down(x3)
        x4 = self.bnd4(self.dropout(self.leakyRelu(self.conv4(x3))))
        x4 += identity4
        x4 = self.Maxpool(x4)

        out_encoder =  self.bnd5(self.leakyRelu(self.conv5(x4)))
        # Complete the forward function for the rest of the encoder : Completed - ey

        
        upidentity1 = self.up(out_encoder)
        y1 = self.bn1(self.leakyRelu(self.deconv1(self.relu(out_encoder))))   
        y1 += upidentity1
        
        upidentity2 = self.up(y1)
        y2 = self.bn2(self.leakyRelu(self.deconv2(y1)))    
        y2 += upidentity2
        
        upidentity3 = self.up(y2)
        y3 = self.bn3(self.dropout(self.leakyRelu(self.deconv3(y2))))    
        y3 += upidentity3
        
        upidentity4 = self.up(y3)
        y4 = self.bn4(self.dropout(self.leakyRelu(self.deconv4(y3))))    
        y4 += upidentity4
        
        out_decoder = self.bn5(self.leakyRelu(self.deconv5(y4)))    

        # Complete the forward function for the rest of the decoder
        
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)

class Recurrent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        