import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize(256),  # Ajuste la plus grande dimension à 256, sans distorsion
    transforms.CenterCrop(256),  # Coupe les bords pour obtenir un carré parfait
    transforms.ToTensor()
])

def load_image(image_path):
    """Charge et transforme une image en tenseur PyTorch sans contours blancs"""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image


def gram_matrix(features):
    """Calcule la matrice de Gram d'une carte de caractéristiques"""
    (b, c, h, w) = features.size()
    features = features.view(b, c, h * w).clone()  # Cloner pour éviter l'erreur in-place
    gram = torch.bmm(features, features.transpose(1, 2))  # Produit matriciel
    return (gram / (c * h * w))  # Normalisation

def get_vgg_model():
    print("Loading VGG model...")
    return models.vgg19(pretrained=True).features.eval()
