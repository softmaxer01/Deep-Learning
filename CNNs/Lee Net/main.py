import torch
import torch.nn as nn 
import torchvision
import torch.optim as optim
from model import LeNet
from training import LOOPS
from plotting import plot_metrics
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from inference import show_images_with_predictions

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

val_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)
# model:
model = LeNet(num_channels=1)

# loss funtion
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(),lr= 0.001)

# epochs
eps = 10

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# training loop
l = LOOPS(model=model,loss_fn=loss_fn,optimizer=optimizer,eps=eps,device=device)


metrices = l.training_loop(train_loader=train_loader,val_loader=val_loader)

# plotting
plot_metrics(metrics=metrices)

# inference
show_images_with_predictions(model, val_dataset, num_images=5)