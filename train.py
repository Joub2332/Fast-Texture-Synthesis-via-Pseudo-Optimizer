import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

def Gatys_and_alTraining(input_image,num_iterations,optimizer,style_extractor,style_features,style_layers,loss):
    loss_list = []
    def closure():
        optimizer.zero_grad()
        input_features = style_extractor(input_image)
        style_weight = 1e6  # 🔥 Essaie d'ajuster cette valeur
        current_loss = style_weight * sum(loss(input_features[j], style_features[j]) for j in range(len(style_layers)))
        loss_list.append(current_loss.item())
        current_loss.backward(retain_graph=True)
        return current_loss
    
    for i in range(num_iterations):
        optimizer.step(closure)

        if i % 50 == 0:
            print(f"Iteration {i}, Loss: {loss_list[-1]}")

    return loss_list
    return loss_list
    # Afficher le résultat
    final_image = input_image.detach().squeeze().permute(1, 2, 0).numpy()
    plt.imshow(final_image)
    plt.axis("off")
    plt.show()