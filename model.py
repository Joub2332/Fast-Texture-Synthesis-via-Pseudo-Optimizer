import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import gram_matrix
from utils import load_image
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
