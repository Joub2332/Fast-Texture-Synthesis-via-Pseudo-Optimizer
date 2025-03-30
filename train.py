import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import load_image,gram_matrix,flat_features,unflatten_features
from model import VGGFeatureExtractor,ProgressivePseudoOptimizer

def Gatys_and_alTraining(input_image, num_iterations, optimizer, style_extractor, style_features, style_layers, loss,device):
    loss_list = []

    # Déplacement sur le bon appareil
    input_image = input_image.to(device)  # ⬅ Ajout de .to(device)
    style_features = [f.to(device) for f in style_features]  # ⬅ Ajout pour chaque feature
    def total_variation_loss(img):
        diff_x = img[:, :, 1:, :] - img[:, :, :-1, :]
        diff_y = img[:, :, :, 1:] - img[:, :, :, :-1]
        return torch.sum(torch.abs(diff_x)) + torch.sum(torch.abs(diff_y))

    def closure():
        optimizer.zero_grad()
        input_features = style_extractor(input_image)
        style_weight = 4e5
        tv_weight = 1e-5  # Ajuste pour plus ou moins de lissage
        current_loss = tv_weight * total_variation_loss(input_image)+(style_weight * sum(loss(input_features[j], style_features[j]) for j in range(len(style_layers))))
        loss_list.append(current_loss.item())
        current_loss.backward(retain_graph=True)
        return current_loss

    for i in range(num_iterations):
        optimizer.step(closure)
        
        if i % 50 == 0:
            print(f"Iteration {i}, Loss: {loss_list[-1]}")

    return loss_list

<<<<<<< HEAD
def train_pseudo_optimizer(texture_imgs_path,model, num_epochs=1000, lr=0.01, num_iter=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
        # Feature extractor
    layers = [0, 4, 9, 18,27]  # Select VGG layers for texture loss
    vgg_extractor = VGGFeatureExtractor(layers).to(device).eval()

    coefs = [
     1.0,  # Lower layer: More weight on fine-grained texture
     1.0,  # Mid-layer: Moderate weight on medium texture features
     1.0, # Higher layer: Less weight on high-level features
     1.0,  # Top layer: Minimal weight on very abstract features
     1.0
    ]
    # Initialize Pseudo Optimizer
    pseudo_optimizer = model().to(device)
    optimizer = optim.Adam(pseudo_optimizer.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_fn = nn.MSELoss()


    L = len(texture_imgs_path)
    def lr_lambda(epoch):
        if epoch < 1500:
            return 1.0  # No decay for the first 1500 epochs
        else:
            # Linear decay from epoch 1500 to 5000
            return 1.0 - (epoch - 1500) / (num_epochs - 1500)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    # Training Loop
    for epoch in range(num_epochs):
          for texture_img_path in texture_imgs_path:
              # Load texture image
              target_texture = load_image(texture_img_path, size=256).to(device)
              target_texture.requires_grad = True



              # Compute target Gram matrices
              target_features = vgg_extractor(target_texture)
              target_grams = [gram_matrix(f) for f in target_features]
              size = [(target_features[i].shape[2:4]) for i in range(len(layers))]

              loss = 0

              for _ in range(num_iter):
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
                  delta_x = pseudo_optimizer(grad_features, lay+1)

                  refined_image = noise + delta_x
                  refined_features = vgg_extractor(refined_image)
                  refined_grams = [gram_matrix(f) for f in refined_features]
                                      # Compute new texture loss
                  loss = loss + sum(loss_fn(refined_grams[i], target_grams[i])*coefs[i] for i in range(lay))

            # Backpropagation
          loss = loss/(num_iter*5*L)
          optimizer.zero_grad()
          loss.backward(retain_graph=True)
          optimizer.step()
          if epoch == 0 or loss.item() < best_loss:
            best_loss = loss.item()
            model_path = f"/kaggle/working/best_model_epoch.pth"
            torch.save(pseudo_optimizer.state_dict(), model_path)
            print(f"New best model saved at epoch {epoch} with loss {best_loss:.6f}")


          #scheduler.step()


          if epoch % 100 == 0:
              print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

    return pseudo_optimizer

=======
>>>>>>> cfcfe751498c9f6c23a2e12c161d2a7246cf4a6f
   