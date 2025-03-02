import torch
import os
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from train import Gatys_and_alTraining
from utils import load_image,get_vgg_model
from model import VGGStyleExtractor

torch.autograd.set_detect_anomaly(True)
def Gatys_and_alSynthesize(texture_image,output_path,num_iterations,device):
    texture_image = load_image(texture_image).to(device)  # Load the texture image
    style_layers = [0, 5, 10, 19, 25,30]  # Corrected VGG-19 style layer indices
    vgg = get_vgg_model().to(device)# Load pretrained VGG-19
    style_extractor = VGGStyleExtractor(vgg, style_layers)
    style_features = style_extractor(texture_image)
    print(f"Extracted {len(style_features)} style features") 
    # Image de départ : bruit aléatoire
    input_image = torch.randn_like(texture_image, requires_grad=True).to(device)
    # Optimiseur (Adam ou LBFGS)
    optimizer = optim.LBFGS([input_image])

    # Fonction de perte (MSE entre matrices de Gram)
    loss_fn = nn.MSELoss()
    loss_list=Gatys_and_alTraining(input_image,num_iterations,optimizer,style_extractor,style_features,style_layers,loss_fn)
    final_image = input_image.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    final_image_pil = Image.fromarray((final_image * 255).clip(0,255).astype("uint8"))
    final_image_pil.save(output_path)
    

# Fonction pour charger une image et la transformer en tenseur
# Fonction principale pour exécuter la synthèse sur toutes les images du dataset
def synthesize_all_images(dataset_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Crée le dossier de sortie s'il n'existe pas

    for filename in os.listdir(dataset_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):  # Vérifie si c'est une image
            image_path = os.path.join(dataset_path, filename)
            print(f"Processing: {filename}")
            # Charger l'image
            Gatys_and_alSynthesize(image_path,output_folder)
    print("Synthèse terminée pour toutes les images.")