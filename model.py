import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import gram_matrix

class VGGStyleExtractor(nn.Module):
    """Extracts style features from selected VGG-19 layers"""
    def __init__(self, vgg, style_layers):
        super(VGGStyleExtractor, self).__init__()
        self.vgg = vgg
        self.style_layers = style_layers  # List of layer indices
        self.selected_layers = nn.Sequential(*[self.vgg[i] for i in range(max(style_layers) + 1)])  # Cut the model at the deepest style layer

    def forward(self, x):
        """Passes the image through the model and extracts Gram matrices"""
        features = []
        for i, layer in enumerate(self.selected_layers):
            x = layer(x)
            if i in self.style_layers:
                features.append(gram_matrix(x))  # Extract Gram matrix for selected layers
        return features

class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, stride=1, padding=1):
        super(UpsampleConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)
        self.norm =  nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.lrelu(x)
        return x
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # First convolution and batch normalization
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Second convolution and batch normalization
        x = self.conv2(x)
        x = self.bn2(x)

        # Add skip connection (identity)
        x += identity
        x = self.relu(x)

        return x
class PseudoOptimizer1st(nn.Module):
    def __init__(self):
        super(PseudoOptimizer1st, self).__init__()

        # Encoder: Downsample the noisy image into feature maps at different scales

        self.decoder4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #output: [1,128,128,128]
        self.decoder2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # output: [1, 256, 64, 64]
        self.decoder1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # output: [1, 512, 32, 32]



        # Decoder: Upsample to the final image size of 256x256
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # [1, 256, 64, 64]
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # [1, 128, 128, 128]
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # [1, 64, 256, 256]


        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()

        self.relu = nn.LeakyReLU(negative_slope=0.01)

        self.sigmoid = nn.Sigmoid()

        # Final output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # output: [1, 3, 256, 256]

        self.resblock1 = ResidualBlock(512, 512)
        self.resblock2 = ResidualBlock(256, 256)
        self.resblock3 = ResidualBlock(128, 128)
        self.resblock4 = ResidualBlock(64, 64)



    def forward(self, gradients):

        x = gradients[3]
        x = self.decoder1(x)
        x = self.relu(x)

        for i in range(5):
          x = self.resblock1(x)

        upconv1_ = self.upconv1(x)

        x = gradients[2]
        x = self.decoder2(x)
        x = self.relu(x) + upconv1_
        for i in range(5):
          x = self.resblock2(x)

        upconv2_ = self.upconv2(x)

        x = gradients[1]
        x = self.decoder3(x)
        x = self.relu(x) + upconv2_

        for i in range(5):
          x = self.resblock3(x)

        upconv3_ = self.upconv3(x)

        x = gradients[0]
        x = self.decoder4(x)
        x = self.relu(x) + upconv3_

        for i in range(5):
          x = self.resblock4(x)
        delta_x = self.output_layer(x)
        delta_x = self.tanh(delta_x)

        return delta_x
class PseudoOptimizer(nn.Module):
    def __init__(self):
        super(PseudoOptimizer, self).__init__()

        # Decoder : Downsample the noisy image into feature maps at different scales
        self.decoder5_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder4_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #output: [1,128,128,128]
        self.decoder2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # output: [1, 256, 64, 64]
        self.decoder1_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # output: [1, 512, 32, 32]

        self.decoder5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #output: [1,128,128,128]
        self.decoder2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # output: [1, 256, 64, 64]
        self.decoder1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # output: [1, 512, 32, 32]

        # Upsamble using nearest neighbors and convolution
        self.upsamp1 = UpsampleConv(in_channels=512, out_channels=256)
        self.upsamp2 = UpsampleConv(in_channels=256, out_channels=128)
        self.upsamp3 = UpsampleConv(in_channels=128, out_channels=64)
        self.upsamp4 = UpsampleConv(in_channels=64, out_channels=64)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)


        # Instance normalization
        self.norm_1_1 = nn.InstanceNorm2d(512)
        self.norm_2_1 = nn.InstanceNorm2d(256)
        self.norm_3_1 = nn.InstanceNorm2d(128)
        self.norm_4_1 = nn.InstanceNorm2d(64)
        self.norm_5_1 = nn.InstanceNorm2d(64)


        self.norm_1_2 = nn.InstanceNorm2d(512)
        self.norm_2_2 = nn.InstanceNorm2d(256)
        self.norm_3_2 = nn.InstanceNorm2d(128)
        self.norm_4_2 = nn.InstanceNorm2d(64)
        self.norm_5_2 = nn.InstanceNorm2d(64)


        self.norm_5 = nn.InstanceNorm2d(3)




        # Final output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # output: [1, 3, 256, 256]

        # Resblocks
        self.resblock1_1 = ResidualBlock(512, 512)
        self.resblock2_1 = ResidualBlock(256, 256)
        self.resblock3_1 = ResidualBlock(128, 128)
        self.resblock4_1 = ResidualBlock(64, 64)
        self.resblock5_1 = ResidualBlock(64, 64)

        self.resblock1_2 = ResidualBlock(512, 512)
        self.resblock2_2 = ResidualBlock(256, 256)
        self.resblock3_2 = ResidualBlock(128, 128)
        self.resblock4_2 = ResidualBlock(64, 64)
        self.resblock5_2 = ResidualBlock(64, 64)


    def forward(self, gradients):

        x = gradients[4]

        x = self.decoder1_1(x)
        x = self.norm_1_1(x)
        x = self.lrelu(x)

        x = self.decoder1_2(x)
        x = self.norm_1_2(x)
        x = self.lrelu(x)

        x = self.resblock1_1(x)
        x = self.resblock1_2(x)

        upsamp1_ = self.upsamp1(x)


        x = gradients[3]

        x = self.decoder2_1(x)
        x = self.norm_2_1(x)
        x = self.lrelu(x)

        x = self.decoder2_2(x)
        x = self.norm_2_2(x)
        x = self.lrelu(x)

        x = x + upsamp1_

        x = self.resblock2_1(x)
        x = self.resblock2_2(x)

        upsamp2_ = self.upsamp2(x)

        x = gradients[2]

        x = self.decoder3_1(x)
        x = self.norm_3_1(x)
        x = self.lrelu(x)

        x = self.decoder3_2(x)
        x = self.norm_3_2(x)
        x = self.lrelu(x)

        x = x + upsamp2_

        x = self.resblock3_1(x)
        x = self.resblock3_2(x)

        upsamp3_ = self.upsamp3(x)


        x = gradients[1]

        x = self.decoder4_1(x)
        x = self.norm_4_1(x)
        x = self.lrelu(x)

        x = self.decoder4_2(x)
        x = self.norm_4_2(x)
        x = self.lrelu(x)

        x = x + upsamp3_

        x = self.resblock4_1(x)
        x = self.resblock4_2(x)


        upsamp4_ = self.upsamp4(x)

        x = gradients[0]

        x = self.decoder5_1(x)
        x = self.norm_5_1(x)
        x = self.lrelu(x)

        x = self.decoder5_2(x)
        x = self.norm_5_2(x)
        x = self.lrelu(x)

        x = x + upsamp4_

        x = self.resblock5_1(x)
        x = self.resblock5_2(x)


        delta_x = self.output_layer(x)


        return delta_x
class PseudoOptimizer_stage5(nn.Module):
    def __init__(self):
        super(PseudoOptimizer_stage5, self).__init__()

        # Decoder : Downsample the noisy image into feature maps at different scales
        self.decoder5_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder4_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #output: [1,128,128,128]
        self.decoder2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # output: [1, 256, 64, 64]
        self.decoder1_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # output: [1, 512, 32, 32]

        self.decoder5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #output: [1,128,128,128]
        self.decoder2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # output: [1, 256, 64, 64]
        self.decoder1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # output: [1, 512, 32, 32]

        # Upsamble using nearest neighbors and convolution
        self.upsamp1 = UpsampleConv(in_channels=512, out_channels=256)
        self.upsamp2 = UpsampleConv(in_channels=256, out_channels=128)
        self.upsamp3 = UpsampleConv(in_channels=128, out_channels=64)
        self.upsamp4 = UpsampleConv(in_channels=64, out_channels=64)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)


        # Instance normalization
        self.norm_1_1 = nn.InstanceNorm2d(512)
        self.norm_2_1 = nn.InstanceNorm2d(256)
        self.norm_3_1 = nn.InstanceNorm2d(128)
        self.norm_4_1 = nn.InstanceNorm2d(64)
        self.norm_5_1 = nn.InstanceNorm2d(64)


        self.norm_1_2 = nn.InstanceNorm2d(512)
        self.norm_2_2 = nn.InstanceNorm2d(256)
        self.norm_3_2 = nn.InstanceNorm2d(128)
        self.norm_4_2 = nn.InstanceNorm2d(64)
        self.norm_5_2 = nn.InstanceNorm2d(64)


        self.norm_5 = nn.InstanceNorm2d(3)




        # Final output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # output: [1, 3, 256, 256]

        # Resblocks
        self.resblock1_1 = ResidualBlock(512, 512)
        self.resblock2_1 = ResidualBlock(256, 256)
        self.resblock3_1 = ResidualBlock(128, 128)
        self.resblock4_1 = ResidualBlock(64, 64)
        self.resblock5_1 = ResidualBlock(64, 64)

        self.resblock1_2 = ResidualBlock(512, 512)
        self.resblock2_2 = ResidualBlock(256, 256)
        self.resblock3_2 = ResidualBlock(128, 128)
        self.resblock4_2 = ResidualBlock(64, 64)
        self.resblock5_2 = ResidualBlock(64, 64)


    def forward(self, gradients):

        x = gradients[4]

        x = self.decoder1_1(x)
        x = self.norm_1_1(x)
        x = self.lrelu(x)

        x = self.decoder1_2(x)
        x = self.norm_1_2(x)
        x = self.lrelu(x)

        x = self.resblock1_1(x)
        x = self.resblock1_2(x)

        upsamp1_ = self.upsamp1(x)


        x = gradients[3]

        x = self.decoder2_1(x)
        x = self.norm_2_1(x)
        x = self.lrelu(x)

        x = self.decoder2_2(x)
        x = self.norm_2_2(x)
        x = self.lrelu(x)

        x = x + upsamp1_

        x = self.resblock2_1(x)
        x = self.resblock2_2(x)

        upsamp2_ = self.upsamp2(x)

        x = gradients[2]

        x = self.decoder3_1(x)
        x = self.norm_3_1(x)
        x = self.lrelu(x)

        x = self.decoder3_2(x)
        x = self.norm_3_2(x)
        x = self.lrelu(x)

        x = x + upsamp2_

        x = self.resblock3_1(x)
        x = self.resblock3_2(x)

        upsamp3_ = self.upsamp3(x)


        x = gradients[1]

        x = self.decoder4_1(x)
        x = self.norm_4_1(x)
        x = self.lrelu(x)

        x = self.decoder4_2(x)
        x = self.norm_4_2(x)
        x = self.lrelu(x)

        x = x + upsamp3_

        x = self.resblock4_1(x)
        x = self.resblock4_2(x)


        upsamp4_ = self.upsamp4(x)

        x = gradients[0]

        x = self.decoder5_1(x)
        x = self.norm_5_1(x)
        x = self.lrelu(x)

        x = self.decoder5_2(x)
        x = self.norm_5_2(x)
        x = self.lrelu(x)

        x = x + upsamp4_

        x = self.resblock5_1(x)
        x = self.resblock5_2(x)


        delta_x = self.output_layer(x)


        return delta_x
class PseudoOptimizer_stage4(nn.Module):
    def __init__(self):
        super(PseudoOptimizer_stage4, self).__init__()

        # Decoder : Downsample the noisy image into feature maps at different scales
        self.decoder4_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #output: [1,128,128,128]
        self.decoder1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # output: [1, 256, 64, 64]

        self.decoder4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #output: [1,128,128,128]
        self.decoder1_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # output: [1, 256, 64, 64]

        # Upsamble using nearest neighbors and convolution
        self.upsamp1 = UpsampleConv(in_channels=256, out_channels=128)
        self.upsamp2 = UpsampleConv(in_channels=128, out_channels=64)
        self.upsamp3 = UpsampleConv(in_channels=64, out_channels=64)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)


        # Instance normalization
        self.norm_1_1 = nn.InstanceNorm2d(256)
        self.norm_2_1 = nn.InstanceNorm2d(128)
        self.norm_3_1 = nn.InstanceNorm2d(64)
        self.norm_4_1 = nn.InstanceNorm2d(64)


        self.norm_1_2 = nn.InstanceNorm2d(256)
        self.norm_2_2 = nn.InstanceNorm2d(128)
        self.norm_3_2 = nn.InstanceNorm2d(64)
        self.norm_4_2 = nn.InstanceNorm2d(64)


        self.norm_5 = nn.InstanceNorm2d(3)




        # Final output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # output: [1, 3, 256, 256]

        # Resblocks
        self.resblock1_1 = ResidualBlock(256, 256)
        self.resblock2_1 = ResidualBlock(128, 128)
        self.resblock3_1 = ResidualBlock(64, 64)
        self.resblock4_1 = ResidualBlock(64, 64)

        self.resblock1_2 = ResidualBlock(256, 256)
        self.resblock2_2 = ResidualBlock(128, 128)
        self.resblock3_2 = ResidualBlock(64, 64)
        self.resblock4_2 = ResidualBlock(64, 64)


    def forward(self, gradients):

        x = gradients[3]

        x = self.decoder1_1(x)
        x = self.norm_1_1(x)
        x = self.lrelu(x)

        x = self.decoder1_2(x)
        x = self.norm_1_2(x)
        x = self.lrelu(x)

        x = self.resblock1_1(x)
        x = self.resblock1_2(x)

        upsamp1_ = self.upsamp1(x)


        x = gradients[2]

        x = self.decoder2_1(x)
        x = self.norm_2_1(x)
        x = self.lrelu(x)

        x = self.decoder2_2(x)
        x = self.norm_2_2(x)
        x = self.lrelu(x)

        x = x + upsamp1_

        x = self.resblock2_1(x)
        x = self.resblock2_2(x)

        upsamp2_ = self.upsamp2(x)

        x = gradients[1]

        x = self.decoder3_1(x)
        x = self.norm_3_1(x)
        x = self.lrelu(x)

        x = self.decoder3_2(x)
        x = self.norm_3_2(x)
        x = self.lrelu(x)

        x = x + upsamp2_

        x = self.resblock3_1(x)
        x = self.resblock3_2(x)

        upsamp3_ = self.upsamp3(x)


        x = gradients[0]

        x = self.decoder4_1(x)
        x = self.norm_4_1(x)
        x = self.lrelu(x)

        x = self.decoder4_2(x)
        x = self.norm_4_2(x)
        x = self.lrelu(x)

        x = x + upsamp3_

        x = self.resblock4_1(x)
        x = self.resblock4_2(x)

        delta_x = self.output_layer(x)


        return delta_x
class PseudoOptimizer_stage3(nn.Module):
    def __init__(self):
        super(PseudoOptimizer_stage3, self).__init__()

        # Decoder : Downsample the noisy image into feature maps at different scales
        self.decoder3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder1_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #output: [1,128,128,128]


        self.decoder3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #output: [1,128,128,128]


        # Upsamble using nearest neighbors and convolution
        self.upsamp1 = UpsampleConv(in_channels=128, out_channels=64)
        self.upsamp2 = UpsampleConv(in_channels=64, out_channels=64)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)


        # Instance normalization
        self.norm_1_1 = nn.InstanceNorm2d(128)
        self.norm_2_1 = nn.InstanceNorm2d(64)
        self.norm_3_1 = nn.InstanceNorm2d(64)


        self.norm_1_2 = nn.InstanceNorm2d(128)
        self.norm_2_2 = nn.InstanceNorm2d(64)
        self.norm_3_2 = nn.InstanceNorm2d(64)


        self.norm_5 = nn.InstanceNorm2d(3)



        # Final output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # output: [1, 3, 256, 256]

        # Resblocks

        self.resblock1_1 = ResidualBlock(128, 128)
        self.resblock2_1 = ResidualBlock(64, 64)
        self.resblock3_1 = ResidualBlock(64, 64)


        self.resblock1_2 = ResidualBlock(128, 128)
        self.resblock2_2 = ResidualBlock(64, 64)
        self.resblock3_2 = ResidualBlock(64, 64)


    def forward(self, gradients):

        x = gradients[2]

        x = self.decoder1_1(x)
        x = self.norm_1_1(x)
        x = self.lrelu(x)

        x = self.decoder1_2(x)
        x = self.norm_1_2(x)
        x = self.lrelu(x)

        x = self.resblock1_1(x)
        x = self.resblock1_2(x)

        upsamp1_ = self.upsamp1(x)


        x = gradients[1]

        x = self.decoder2_1(x)
        x = self.norm_2_1(x)
        x = self.lrelu(x)

        x = self.decoder2_2(x)
        x = self.norm_2_2(x)
        x = self.lrelu(x)

        x = x + upsamp1_

        x = self.resblock2_1(x)
        x = self.resblock2_2(x)

        upsamp2_ = self.upsamp2(x)

        x = gradients[0]

        x = self.decoder3_1(x)
        x = self.norm_3_1(x)
        x = self.lrelu(x)

        x = self.decoder3_2(x)
        x = self.norm_3_2(x)
        x = self.lrelu(x)

        x = x + upsamp2_

        x = self.resblock3_1(x)
        x = self.resblock3_2(x)

        delta_x = self.output_layer(x)


        return delta_x
class PseudoOptimizer_stage2(nn.Module):
    def __init__(self):
        super(PseudoOptimizer_stage2, self).__init__()

        # Decoder : Downsample the noisy image into feature maps at different scales
        self.decoder1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]


        self.decoder1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]


        # Upsamble using nearest neighbors and convolution
        self.upsamp1 = UpsampleConv(in_channels=64, out_channels=64)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)


        # Instance normalization
        self.norm_1_1 = nn.InstanceNorm2d(64)
        self.norm_2_1 = nn.InstanceNorm2d(64)


        self.norm_1_2 = nn.InstanceNorm2d(64)
        self.norm_2_2 = nn.InstanceNorm2d(64)


        self.norm_5 = nn.InstanceNorm2d(3)




        # Final output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # output: [1, 3, 256, 256]

        # Resblocks
        self.resblock1_1 = ResidualBlock(64, 64)
        self.resblock2_1 = ResidualBlock(64, 64)

        self.resblock1_2 = ResidualBlock(64, 64)
        self.resblock2_2 = ResidualBlock(64, 64)


    def forward(self, gradients):

        x = gradients[1]

        x = self.decoder1_1(x)
        x = self.norm_1_1(x)
        x = self.lrelu(x)

        x = self.decoder1_2(x)
        x = self.norm_1_2(x)
        x = self.lrelu(x)

        x = self.resblock1_1(x)
        x = self.resblock1_2(x)

        upsamp1_ = self.upsamp1(x)


        x = gradients[0]

        x = self.decoder2_1(x)
        x = self.norm_2_1(x)
        x = self.lrelu(x)

        x = self.decoder2_2(x)
        x = self.norm_2_2(x)
        x = self.lrelu(x)

        x = x + upsamp1_

        x = self.resblock2_1(x)
        x = self.resblock2_2(x)

        delta_x = self.output_layer(x)


        return delta_x
class PseudoOptimizer_stage1(nn.Module):
    def __init__(self):
        super(PseudoOptimizer_stage1, self).__init__()

        # Decoder : Downsample the noisy image into feature maps at different scales
        self.decoder1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]


        self.decoder1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]




        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)


        # Instance normalization
        self.norm_1_1 = nn.InstanceNorm2d(64)

        self.norm_1_2 = nn.InstanceNorm2d(64)


        self.norm_5 = nn.InstanceNorm2d(3)




        # Final output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # output: [1, 3, 256, 256]

        # Resblocks
        self.resblock1_1 = ResidualBlock(64, 64)

        self.resblock1_2 = ResidualBlock(64, 64)


    def forward(self, gradients):

        x = gradients[0]

        x = self.decoder1_1(x)
        x = self.norm_1_1(x)
        x = self.lrelu(x)

        x = self.decoder1_2(x)
        x = self.norm_1_2(x)
        x = self.lrelu(x)

        x = self.resblock1_1(x)
        x = self.resblock1_2(x)

        delta_x = self.output_layer(x)


        return delta_x
class AdapPseudoOptimizer(nn.Module):
    def __init__(self):
        super(AdapPseudoOptimizer, self).__init__()

        # Decoder : Downsample the noisy image into feature maps at different scales
        self.decoder5_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder4_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #output: [1,128,128,128]
        self.decoder2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # output: [1, 256, 64, 64]
        self.decoder1_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # output: [1, 512, 32, 32]

        self.decoder5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # output: [1, 64, 256, 246]
        self.decoder3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #output: [1,128,128,128]
        self.decoder2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # output: [1, 256, 64, 64]
        self.decoder1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # output: [1, 512, 32, 32]

        # Upsamble using nearest neighbors and convolution
        self.upsamp1 = UpsampleConv(in_channels=512, out_channels=256)
        self.upsamp2 = UpsampleConv(in_channels=256, out_channels=128)
        self.upsamp3 = UpsampleConv(in_channels=128, out_channels=64)
        self.upsamp4 = UpsampleConv(in_channels=64, out_channels=64)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)


        # Instance normalization
        self.norm_1_1 = nn.InstanceNorm2d(512)
        self.norm_2_1 = nn.InstanceNorm2d(256)
        self.norm_3_1 = nn.InstanceNorm2d(128)
        self.norm_4_1 = nn.InstanceNorm2d(64)
        self.norm_5_1 = nn.InstanceNorm2d(64)


        self.norm_1_2 = nn.InstanceNorm2d(512)
        self.norm_2_2 = nn.InstanceNorm2d(256)
        self.norm_3_2 = nn.InstanceNorm2d(128)
        self.norm_4_2 = nn.InstanceNorm2d(64)
        self.norm_5_2 = nn.InstanceNorm2d(64)


        self.norm_5 = nn.InstanceNorm2d(3)




        # Final output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # output: [1, 3, 256, 256]

        # Resblocks
        self.resblock1_1 = ResidualBlock(512, 512)
        self.resblock2_1 = ResidualBlock(256, 256)
        self.resblock3_1 = ResidualBlock(128, 128)
        self.resblock4_1 = ResidualBlock(64, 64)
        self.resblock5_1 = ResidualBlock(64, 64)

        self.resblock1_2 = ResidualBlock(512, 512)
        self.resblock2_2 = ResidualBlock(256, 256)
        self.resblock3_2 = ResidualBlock(128, 128)
        self.resblock4_2 = ResidualBlock(64, 64)
        self.resblock5_2 = ResidualBlock(64, 64)


    def forward(self, gradients):

        x = gradients[4]

        x = self.decoder1_1(x)
        x = self.norm_1_1(x)
        x = self.lrelu(x)

        x = self.decoder1_2(x)
        x = self.norm_1_2(x)
        x = self.lrelu(x)

        x = self.resblock1_1(x)
        x = self.resblock1_2(x)

        upsamp1_ = self.upsamp1(x)


        x = gradients[3]

        x = self.decoder2_1(x)
        x = self.norm_2_1(x)
        x = self.lrelu(x)

        x = self.decoder2_2(x)
        x = self.norm_2_2(x)
        x = self.lrelu(x)

        x = x + upsamp1_

        x = self.resblock2_1(x)
        x = self.resblock2_2(x)

        upsamp2_ = self.upsamp2(x)

        x = gradients[2]

        x = self.decoder3_1(x)
        x = self.norm_3_1(x)
        x = self.lrelu(x)

        x = self.decoder3_2(x)
        x = self.norm_3_2(x)
        x = self.lrelu(x)

        x = x + upsamp2_

        x = self.resblock3_1(x)
        x = self.resblock3_2(x)

        upsamp3_ = self.upsamp3(x)


        x = gradients[1]

        x = self.decoder4_1(x)
        x = self.norm_4_1(x)
        x = self.lrelu(x)

        x = self.decoder4_2(x)
        x = self.norm_4_2(x)
        x = self.lrelu(x)

        x = x + upsamp3_

        x = self.resblock4_1(x)
        x = self.resblock4_2(x)


        upsamp4_ = self.upsamp4(x)

        x = gradients[0]

        x = self.decoder5_1(x)
        x = self.norm_5_1(x)
        x = self.lrelu(x)

        x = self.decoder5_2(x)
        x = self.norm_5_2(x)
        x = self.lrelu(x)

        x = x + upsamp4_

        x = self.resblock5_1(x)
        x = self.resblock5_2(x)


        delta_x = self.output_layer(x)


        return delta_x    

class ProgressivePseudoOptimizer(nn.Module):
  def __init__(self):
    super(ProgressivePseudoOptimizer, self).__init__()
    self.stage1 = PseudoOptimizer_stage1()
    self.stage2 = PseudoOptimizer_stage2()
    self.stage3 = PseudoOptimizer_stage3()
    self.stage4 = PseudoOptimizer_stage4()
    self.stage5 = PseudoOptimizer_stage5()

  def forward(self, gradients, ind):
    if ind == 1:
      x = self.stage1(gradients[:1])
    if ind == 2:
      x = self.stage2(gradients[:2])
    if ind == 3:
      x = self.stage3(gradients[:3])
    if ind == 4:
      x = self.stage4(gradients[:4])
    if ind == 5:
      x = self.stage5(gradients[:5])

    return x
  
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.selected_layers = layers
        self.model = nn.Sequential(*[vgg[i] for i in range(max(layers) + 1)])
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze VGG model

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            x_requires_grad = True
            if i in self.selected_layers:
                outputs.append(x)
        return outputs