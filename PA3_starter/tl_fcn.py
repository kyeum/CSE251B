### Transfer Learning to replace encoder of FCN architecture
import torch.nn as nn
from torchvision import models

   
class TL_FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        self.model_ft = models.resnet34(pretrained=True)
        # Remove last 2 layers (pooling and fc)
        self.model_ft = nn.Sequential(*list(self.model_ft.children())[:-2])
        # Freeze parameters of pretrained model.
#         for param in self.model_ft.parameters():
#             param.requires_grad = False
            
        # Create new linear layer to match dimensions.
        
       #
#         self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, 512)
#         self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
#         self.bnd1    = nn.BatchNorm2d(32)
#         self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
#         self.bnd2    = nn.BatchNorm2d(64)
#         self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
#         self.bnd3    = nn.BatchNorm2d(128)
#         self.conv4   = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
#         self.bnd4    = nn.BatchNorm2d(256)
#         self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
#         self.bnd5    = nn.BatchNorm2d(512)
        
        
        self.relu    = nn.ReLU(inplace=False)
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

#         x1 = self.bnd1(self.relu(self.conv1(x)))
#         x2 = self.bnd2(self.relu(self.conv2(x1)))
#         x3 = self.bnd3(self.relu(self.conv3(x2)))
#         x4 = self.bnd4(self.relu(self.conv4(x3)))

#         out_encoder =  self.bnd5(self.relu(self.conv5(x4)))

        out_encoder = self.model_ft(x)
#         print("out_encoder.shape:", out_encoder.shape)
        # Complete the forward function for the rest of the encoder : Completed - ey


        y1 = self.bn1(self.relu(self.deconv1(self.relu(out_encoder))))   
        y2 = self.bn2(self.relu(self.deconv2(y1)))    
        y3 = self.bn3(self.relu(self.deconv3(y2)))    
        y4 = self.bn4(self.relu(self.deconv4(y3)))    
        out_decoder = self.bn5(self.relu(self.deconv5(y4)))    

        # Complete the forward function for the rest of the decoder
        
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)
