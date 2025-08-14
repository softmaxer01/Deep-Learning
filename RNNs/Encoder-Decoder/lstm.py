import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, sigma=0.001):
        super(LSTM, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.sigma = sigma

        self.W_xi, self.W_hi, self.b_i = self.triple()
        self.W_xf, self.W_hf, self.b_f = self.triple()
        self.W_xo, self.W_ho, self.b_o = self.triple()
        self.W_xc, self.W_hc, self.b_c = self.triple()

        self.W_fo = self.init_weights(self.num_hiddens, self.num_outputs)
        self.b_fo = nn.Parameter(torch.zeros(self.num_outputs))

    def init_weights(self, *shape):
        return nn.Parameter(torch.randn(*shape) * self.sigma)

    def triple(self):
        return (
            self.init_weights(self.num_inputs, self.num_hiddens),
            self.init_weights(self.num_hiddens, self.num_hiddens),
            nn.Parameter(torch.zeros(self.num_hiddens)),
        )

    def forward(self, inputs, H_C=None):
        device = inputs.device
        if H_C is None:
            H = torch.zeros((inputs.shape[1], self.num_hiddens), device=device)
            C = torch.zeros((inputs.shape[1], self.num_hiddens), device=device)
        else:
            H, C = H_C

        outputs = []
        for X in inputs:
            I = torch.sigmoid(
                torch.matmul(X, self.W_xi) + torch.matmul(H, self.W_hi) + self.b_i
            )
            F = torch.sigmoid(
                torch.matmul(X, self.W_xf) + torch.matmul(H, self.W_hf) + self.b_f
            )
            C_hat = torch.tanh(
                torch.matmul(X, self.W_xc) + torch.matmul(H, self.W_hc) + self.b_c
            )
            O = torch.sigmoid(
                torch.matmul(X, self.W_xo) + torch.matmul(H, self.W_ho) + self.b_o
            )

            C = F * C + I * C_hat
            H = O * torch.tanh(C)
            output = torch.matmul(H, self.W_fo) + self.b_fo
            outputs.append(output)

        return outputs, (H, C)
