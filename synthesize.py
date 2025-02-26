from model import VGGStyleExtractor
from train import Gatys_and_alTraining
from utils import load_image,get_vgg_model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
torch.autograd.set_detect_anomaly(True)
def Gatys_and_alSynthesize(texture_image):
    texture_image = load_image(texture_image)  # Load the texture image
    style_layers = [0, 5, 10, 19, 28]  # Corrected VGG-19 style layer indices
    vgg = get_vgg_model()# Load pretrained VGG-19
    style_extractor = VGGStyleExtractor(vgg, style_layers)
    style_features = style_extractor(texture_image)
    print(f"Extracted {len(style_features)} style features") 
    # Image de départ : bruit aléatoire
    input_image = torch.randn_like(texture_image, requires_grad=True)

    # Optimiseur (Adam ou LBFGS)
    optimizer = optim.Adam([input_image], lr=0.01)

    # Fonction de perte (MSE entre matrices de Gram)
    loss_fn = nn.MSELoss()

    num_iterations=500
    loss_list=Gatys_and_alTraining(input_image,num_iterations,optimizer,style_extractor,style_features,style_layers,loss_fn)
    final_image = input_image.detach().squeeze().permute(1, 2, 0).numpy()
    final_image_pil = Image.fromarray((final_image * 255).astype("uint8"))

# Save the output image
    output_path = "synthesized_texture.jpg"
    final_image_pil.save(output_path)   
    plt.imshow(final_image)
    plt.axis("off")
    plt.show()
if __name__ == "__main__":
    Gatys_and_alSynthesize("fleur.jpg")  # S'exécute seulement si le fichier est lancé directement

