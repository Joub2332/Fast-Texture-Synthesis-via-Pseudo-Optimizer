import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 

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

def flat_features(features):
    (b, c, h, w) = features.size()
    features = features.view(b, c, h * w)
    return features

def unflatten_features(features, h, w):
  (b, c, n) = features.size()
  assert n == h*w
  return features.view(b,c,h,w)

def denormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406],device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)

    image_tensor = image_tensor * std + mean  # Reverse normalization
    image_tensor = torch.clamp(image_tensor, 0, 1)  # Ensure values are valid
    return image_tensor



def show_image(tensor, epoch, filename=None):
    """Function to show an image given a tensor and save it."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)

    # Reverse normalization
    tensor = tensor * std + mean
    image = tensor.detach().cpu().clamp(0, 1).squeeze().permute(1, 2, 0).numpy()

    # Ensure the image has only 3 channels (RGB)
    if image.shape[2] == 4:  # If an alpha channel is present
        image = image[:, :, :3]  # Remove the alpha channel

    # Display the image using matplotlib
    plt.imshow(image)
    plt.title(f"Epoch {epoch}")
    plt.show()

    # Save the image using Pillow (ensures 3 channels)
    if filename:
        # Convert the image to uint8 and scale to 0-255
        image_uint8 = (image * 255).astype(np.uint8)
        # Create a PIL image from the NumPy array
        pil_image = Image.fromarray(image_uint8)
        # Save the image
        pil_image.save(filename)