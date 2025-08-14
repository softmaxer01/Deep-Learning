import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, sigma=0.001):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigma = sigma

        self.W_xi = self.init_weights(input_size, hidden_size)
        self.W_hi = self.init_weights(hidden_size, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        self.W_xf = self.init_weights(input_size, hidden_size)
        self.W_hf = self.init_weights(hidden_size, hidden_size)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        self.W_xo = self.init_weights(input_size, hidden_size)
        self.W_ho = self.init_weights(hidden_size, hidden_size)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        self.W_xc = self.init_weights(input_size, hidden_size)
        self.W_hc = self.init_weights(hidden_size, hidden_size)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

    def init_weights(self, *shape):
        return nn.Parameter(torch.randn(*shape) * self.sigma)

    def forward(self, x, hidden_state=None):
        device = x.device
        batch_size = x.size(0)

        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            c = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h, c = hidden_state
        i = torch.sigmoid(
            torch.matmul(x, self.W_xi) + torch.matmul(h, self.W_hi) + self.b_i
        )

        f = torch.sigmoid(
            torch.matmul(x, self.W_xf) + torch.matmul(h, self.W_hf) + self.b_f
        )

        c_hat = torch.tanh(
            torch.matmul(x, self.W_xc) + torch.matmul(h, self.W_hc) + self.b_c
        )

        o = torch.sigmoid(
            torch.matmul(x, self.W_xo) + torch.matmul(h, self.W_ho) + self.b_o
        )

        c = f * c + i * c_hat
        h = o * torch.tanh(c)

        return h, c


class LSTM(nn.Module):

    def __init__(
        self,
        num_inputs,
        num_hiddens,
        num_outputs,
        num_layers=1,
        dropout=0.0,
        sigma=0.001,
    ):
        super(LSTM, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.dropout = dropout
        self.sigma = sigma

        self.lstm_layers = nn.ModuleList()

        self.lstm_layers.append(LSTMCell(num_inputs, num_hiddens, sigma))

        for _ in range(1, num_layers):
            self.lstm_layers.append(LSTMCell(num_hiddens, num_hiddens, sigma))

        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

        self.output_projection = nn.Linear(num_hiddens, num_outputs)

    def init_hidden(self, batch_size, device):
        hidden_states = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.num_hiddens, device=device)
            c = torch.zeros(batch_size, self.num_hiddens, device=device)
            hidden_states.append((h, c))
        return hidden_states

    def forward(self, inputs, H_C=None):

        device = inputs.device
        seq_len, batch_size, _ = inputs.shape

        if H_C is None:
            hidden_states = self.init_hidden(batch_size, device)
        else:
            if isinstance(H_C, tuple) and len(H_C) == 2:
                h, c = H_C
                hidden_states = [(h, c) for _ in range(self.num_layers)]
            else:
                hidden_states = H_C

        outputs = []

        for t in range(seq_len):
            x = inputs[t]
            for layer_idx in range(self.num_layers):
                h, c = self.lstm_layers[layer_idx](x, hidden_states[layer_idx])
                hidden_states[layer_idx] = (h, c)

                if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                    x = self.dropout_layer(h)
                else:
                    x = h

            output = self.output_projection(h)
            outputs.append(output)

        final_h, final_c = hidden_states[-1]
        return outputs, (final_h, final_c)

    def one_layer_forward(self, inputs, H_C=None):

        return self.forward(inputs, H_C)

