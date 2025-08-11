import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.W_xh = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))

    def forward(self, inputs, state=None):
        if state is None:
            state = torch.zeros(inputs.shape[1], self.num_hiddens, device=inputs.device)

        outputs = []
        for X in inputs:  # X: (batch, num_inputs)
            state = torch.tanh(
                torch.matmul(X, self.W_xh) + torch.matmul(state, self.W_hh) + self.b_h
            )
            outputs.append(state)
        return outputs, state


class StackedRNNScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

        self.rnns = nn.ModuleList(
            [
                RNN(
                    self.num_inputs if i == 0 else self.num_hiddens,
                    self.num_hiddens,
                    sigma,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, inputs, Hs=None):
        outputs = inputs
        if Hs is None:
            Hs = [None] * self.num_layers

        for i in range(self.num_layers):
            outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
            outputs = torch.stack(outputs, 0)  # keep shape (T, batch, hidden)

        return outputs, Hs


model = StackedRNNScratch(3, 10, 5)
X = torch.randn(10, 2, 3)
y_pred, _ = model(X)
print(f"Shapes prediction: {y_pred.shape}  shape of the hidden layer: {_[0].shape}")
