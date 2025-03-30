import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib as plt
from PIL import Image
from train import Gatys_and_alTraining
from utils import load_image,get_vgg_model,gram_matrix,flat_features,unflatten_features,show_image
from model import VGGStyleExtractor,VGGFeatureExtractor,PseudoOptimizer,AdapPseudoOptimizer,ProgressivePseudoOptimizer
from train import train_pseudo_optimizer

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def Pseudo_optimizerSynthesize(texture_image):
    target_texture = load_image(texture_image).to(device)

    noise = torch.randn_like(target_texture, requires_grad=True).to(device)*0.1

    layers = [1, 6, 11, 20]  # Select VGG layers for texture loss
    vgg_extractor = VGGFeatureExtractor(layers).to(device).eval()

    noise_features = vgg_extractor(noise)
    noise_grams = [gram_matrix(f) for f in noise_features]

    target_features = vgg_extractor(target_texture)
    target_grams = [gram_matrix(f) for f in target_features]

    loss_fn = nn.MSELoss()

    # Compute gradients w.r.t. texture loss
    grad_features = [torch.autograd.grad(loss_fn(noise_grams[i], target_grams[i]), noise_features[i], create_graph=True)[0] for i in range(len(layers))]
    model = train_pseudo_optimizer(texture_image,PseudoOptimizer(), num_epochs=5000, lr=2e-4, num_iter=5)
    gradient_x = model(grad_features)

    # Assuming noise and gradient_x are tensors and already on the same device
    # Update the noise with the predicted gradient
    noise = noise + gradient_x

    noise_features = vgg_extractor(noise)
    noise_grams = [gram_matrix(f) for f in noise_features]

    # Clamp values to be between 0 and 1 to ensure they are within a valid range for images
    output_image = noise.detach().cpu().clamp(0, 1)

    # Remove batch dimension if it's there (assuming single image in the batch)
    output_image = output_image.squeeze(0)

    # Convert tensor to numpy array (for visualization with matplotlib)
    output_image = output_image.permute(1, 2, 0).cpu().numpy()

    # Visualize the image
    plt.imshow(output_image)
    plt.axis('off')  # Turn off axis for cleaner display
    plt.show()

    loss = sum(loss_fn(noise_grams[i], target_grams[i]) for i in range(len(layers)))
    print(loss)

def Adaptive_Pseudo_Optimizer(liste_texture_images):
    target_texture = load_image(liste_texture_images[0], size=256).to(device)
    target_features = vgg_extractor(target_texture)
    target_grams = [gram_matrix(f) for f in target_features]
    size = [(noise_features[i].shape[2:4]) for i in range(len(layers))]

    noise = torch.randn_like(target_texture, requires_grad=True).to(device)*0.1

    layers = [0, 4, 9, 18,27]  # Select VGG layers for texture loss
    vgg_extractor = VGGFeatureExtractor(layers).to(device).eval()

    noise_features = vgg_extractor(noise)
    noise_grams = [gram_matrix(f) for f in noise_features]
    noise_flat_features = [flat_features(f) for f in noise_features]
    flat_grad_features = [4/(noise_flat_features[i].shape[2]) * torch.bmm((noise_grams[i]-target_grams[i]),noise_flat_features[i]) for i in range(len(layers))]
    grad_features = [unflatten_features(flat_grad_features[i], size[i][0], size[i][1]) for i in range(len(layers))]



    target_features = vgg_extractor(target_texture)
    target_grams = [gram_matrix(f) for f in target_features]

    loss_fn = nn.MSELoss()

    # Compute gradients w.r.t. texture loss

    size = [(noise_features[i].shape[2:4]) for i in range(len(layers))]

    model = train_pseudo_optimizer(liste_texture_images,AdapPseudoOptimizer(), num_epochs=1000, lr=2e-4, num_iter=3)

    delta_x = model(grad_features)

    refined_image = noise + delta_x
    show_image(refined_image, 0, filename=f"output_{0}.png")
    refined_features = vgg_extractor(refined_image)
    refined_grams = [gram_matrix(f) for f in refined_features]

def Progressive_Pseudo_Optimizer(texture_images):
    target_texture = load_image(texture_images[0], size=256).to(device)

    noise = torch.randn_like(target_texture, requires_grad=True).to(device)

    layers = [0, 4, 9, 18,27]  # Select VGG layers for texture loss
    vgg_extractor = VGGFeatureExtractor(layers).to(device).eval()

    noise_features = vgg_extractor(noise)
    noise_grams = [gram_matrix(f) for f in noise_features]
    noise_flat_features = [flat_features(f) for f in noise_features]


    target_features = vgg_extractor(target_texture)
    target_grams = [gram_matrix(f) for f in target_features]

    loss_fn = nn.MSELoss()

    # Compute gradients w.r.t. texture loss

    size = [(noise_features[i].shape[2:4]) for i in range(len(layers))]

    for lay in range(len(layers)):
        if lay == 0:
            noise = torch.randn_like(target_texture, requires_grad=True).to(device)*0.1
            noise_features = vgg_extractor(noise)
            noise_flat_features = [flat_features(f) for f in noise_features]
            noise_grams = [gram_matrix(f) for f in noise_features]
        else:
            noise = refined_image
            noise_features = refined_features
            noise_grams = refined_grams
            noise_flat_features = [flat_features(f) for f in noise_features]
                            # Compute current features and gradients



                        # Compute gradients w.r.t. texture loss
        flat_grad_features = [4/(noise_flat_features[i].shape[2]) * torch.bmm((noise_grams[i]-target_grams[i]),noise_flat_features[i]) for i in range(len(layers))]
        grad_features = [unflatten_features(flat_grad_features[i], size[i][0], size[i][1]) for i in range(len(layers))]
                            # Forward pass through Pseudo Optimizer
        model = train_pseudo_optimizer(texture_images, num_epochs=1000, lr=2e-4, num_iter=3)

        delta_x = model(grad_features, lay+1)

        refined_image = noise + delta_x
        show_image(refined_image, 0, filename=f"output_{lay+1}.png")
        refined_features = vgg_extractor(refined_image)
        refined_grams = [gram_matrix(f) for f in refined_features]

                # Compute gradients w.r.t. texture loss
