import random
import matplotlib.pyplot as plt
import torch
def show_images_with_predictions(model, dataset, num_images=5, device='cpu'):
    model.eval()
    indices = random.sample(range(len(dataset)), num_images)
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)

        plt.subplot(1, num_images, i + 1)
        plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'Predicted: {predicted.item()}\nActual: {label}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()