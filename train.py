import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

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

   