import random
import matplotlib.pyplot as plt
import torch
import numpy as np


def show_images_with_predictions(model, dataset, num_images=5, device='cpu', class_names=None):
    model.eval()
    indices = random.sample(range(len(dataset)), num_images)
    
    plt.figure(figsize=(3 * num_images, 4))
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        display_image = image.clone()
        
        image_input = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_input)
            _, predicted = torch.max(output.data, 1)
        
        img_np = display_image.cpu().numpy()
        
        if img_np.shape[0] == 1:  
            img_np = img_np.squeeze()
            cmap = 'gray'
        elif img_np.shape[0] == 3: 
            img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.min() < 0:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_np = np.clip(img_np, 0, 1)
            cmap = None
        else:
            raise ValueError(f"Unexpected image shape: {img_np.shape}")

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img_np, cmap=cmap)

        if class_names is not None:
            pred_name = class_names[predicted.item()]
            actual_name = class_names[label]
            title = f'Pred: {pred_name}\nActual: {actual_name}'
        else:
            title = f'Pred: {predicted.item()}\nActual: {label}'

        color = 'green' if predicted.item() == label else 'red'
        plt.title(title, color=color, fontsize=10, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def denormalize_image(image, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return image * std + mean


def show_mnist_predictions(model, dataset, num_images=5, device='cpu'):
    class_names = [str(i) for i in range(10)]
    show_images_with_predictions(model, dataset, num_images, device, class_names)


def show_cifar10_predictions(model, dataset, num_images=5, device='cpu'):
    """Convenience function for CIFAR-10"""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    show_images_with_predictions(model, dataset, num_images, device, class_names)


def show_cifar100_predictions(model, dataset, num_images=5, device='cpu'):
    show_images_with_predictions(model, dataset, num_images, device)


"""
# For MNIST
show_mnist_predictions(model, val_dataset, num_images=5, device='cuda')

# For CIFAR-10
show_cifar10_predictions(model, val_dataset, num_images=5, device='cuda')

# For any dataset with custom class names
class_names = ['class1', 'class2', 'class3', ...]
show_images_with_predictions(model, dataset, num_images=5, device='cuda', class_names=class_names)

# For any dataset without class names
show_images_with_predictions(model, dataset, num_images=5, device='cuda')
"""