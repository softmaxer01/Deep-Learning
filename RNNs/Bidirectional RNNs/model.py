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
        # inputs: (T, batch, num_inputs)
        if state is None:
            state = torch.zeros(inputs.shape[1], self.num_hiddens, device=inputs.device)

        outputs = []
        for X in inputs:  # iterate over time steps
            state = torch.tanh(
                torch.matmul(X, self.W_xh) + torch.matmul(state, self.W_hh) + self.b_h
            )
            outputs.append(state)
        return outputs, state


class BiRNN(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.forward_rnn = RNN(num_inputs, num_hiddens, sigma)
        self.backward_rnn = RNN(num_inputs, num_hiddens, sigma)
        self.out_hiddens = num_hiddens * 2  # after concatenation

    def forward(self, inputs, Hs=None):
        # inputs: (T, batch, num_inputs)
        f_H, b_H = Hs if Hs is not None else (None, None)

        # forward direction
        f_outputs, f_H = self.forward_rnn(inputs, f_H)

        # backward direction
        b_inputs = torch.flip(inputs, [0])  # reverse in time
        b_outputs, b_H = self.backward_rnn(b_inputs, b_H)
        b_outputs = list(reversed(b_outputs))  # flip outputs back

        # concatenate along hidden dimension
        outputs = [torch.cat((f, b), dim=-1) for f, b in zip(f_outputs, b_outputs)]

        return outputs, (f_H, b_H)


model = BiRNN(3, 10)
X = torch.randn(110, 2, 3)
y_pred, _ = model(X)

print(f"shape of the hidden state:{y_pred[0].shape}, len {len(y_pred)}")
