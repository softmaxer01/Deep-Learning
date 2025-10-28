import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt



# Load data
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# auto encoder model
class Encoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inp_to_hid = nn.Linear(inp_dim, hid_dim, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.inp_to_hid(x))

class Decoder(nn.Module):
    def __init__(self, hid_dim, out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hid_to_out = nn.Linear(hid_dim, out_dim, bias=True)
        self.sigmoid = nn.Sigmoid()  
    
    def forward(self, x):
        return self.sigmoid(self.hid_to_out(x))

class Model(nn.Module):
    def __init__(self, inp, hid, out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enc = Encoder(inp, hid)
        self.dec = Decoder(hid, out)
    
    def forward(self, x):
        return self.dec(self.enc(x))
    

# auto encoder training
model = Model(784,32,784)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr = 0.1, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
def train_autoencoder(model, dataloader, loss_fn, optimizer, epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in dataloader:
            X = torch.flatten(X,start_dim=1,end_dim=-1)
            X = X.to(device)
            preds = model(X)
            loss = loss_fn(preds, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        scheduler.step()
train_autoencoder(model, train_loader, loss_fn, optimizer, epochs=10)


# classifier head
class Classifier_Model(nn.Module):
  def __init__(self,encoder_model):
    super().__init__()
    self.encoder_model = encoder_model
    self.fc1 = nn.Linear(32,128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128,10)

  def forward(self,x):
    encoded = self.encoder_model.enc(x)
    return self.fc2(self.relu(self.fc1(encoded)))
  

# classifier training
model1 = Classifier_Model(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model1.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def calculate_accuracy(model, dataloader, device):
    """Calculate accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for X, y in dataloader:
            X = torch.flatten(X, start_dim=1, end_dim=-1)
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            total_loss += loss.item()
            
            _, predicted = torch.max(preds, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

def train(model, train_loader, val_loader, loss_fn, optimizer, epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X = torch.flatten(X, start_dim=1, end_dim=-1)
            X, y = X.to(device), y.to(device)
            
            preds = model(X)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        val_acc, val_loss = calculate_accuracy(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
    
    return history

# Train the model
history = train(model1, train_loader, test_loader, loss_fn, optimizer, epochs=10)



# plotting
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Val Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_history(history)

test_acc, test_loss = calculate_accuracy(model1, test_loader, "cuda" if torch.cuda.is_available() else "cpu")
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")


