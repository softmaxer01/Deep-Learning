import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
    
    def forward(self, x):
        return self.hid_to_out(x)
    

class Model(nn.Module):
    def __init__(self, inp, hid, out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enc = Encoder(inp, hid)
        self.dec = Decoder(hid, out)

    def forward(self, x):
        return self.dec(self.enc(x))


def train(model, dataloader, loss_fn, optimizer, epochs=10, device="cpu"):
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")


x = torch.randn((2000, 2000))*5
y = x.clone()
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = Model(2000, 50, 2000)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

train(model, dataloader, loss_fn, optimizer, epochs=200)
