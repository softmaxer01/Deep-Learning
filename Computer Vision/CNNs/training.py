import torch
import torch.nn as nn

class LOOPS:
    def __init__(self, model, loss_fn, optimizer, eps, device):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.eps = eps
        self.device = device

    def training_loop(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for ep in range(self.eps):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for img, tar in self.train_loader:
                img, tar = img.to(self.device), tar.to(self.device)
                output = self.model(img)
                loss = self.loss_fn(output, tar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += tar.size(0)
                correct += (predicted == tar).sum().item()

            avg_loss = total_loss / len(self.train_loader)
            train_acc = 100 * correct / total

            val_loss, val_acc = self.val_loop(self.val_loader)
            train_losses.append(avg_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f'Epoch [{ep+1}/{self.eps}], '
                  f'Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        return {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_acc": train_accs,
            "val_acc": val_accs
        }

    def val_loop(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for img, tar in val_loader:
                img, tar = img.to(self.device), tar.to(self.device)
                output = self.model(img)
                loss = self.loss_fn(output, tar)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += tar.size(0)
                correct += (predicted == tar).sum().item()

        avg_val_loss = total_loss / len(val_loader)
        val_acc = 100 * correct / total
        return avg_val_loss, val_acc
