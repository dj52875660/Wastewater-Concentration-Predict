import torch
import torch.nn as nn


class CustomLSTM_With_Peephole(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, num_layers: int = 1, bidirectional: bool = True):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm_cells = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = input_sz if layer == 0 else hidden_sz
            self.lstm_cells.append(LSTMCellWithPeephole(input_dim, hidden_sz))

        if bidirectional:
            self.lstm_cells_back = nn.ModuleList()
            for layer in range(num_layers):
                input_dim = input_sz if layer == 0 else hidden_sz
                self.lstm_cells_back.append(LSTMCellWithPeephole(input_dim, hidden_sz))

    def forward(self, x, init_states=None):
        bs, seq_sz, _ = x.shape
        if init_states is None:
            init_states = [
                (torch.zeros(bs, self.hidden_size).to(x.device), torch.zeros(bs, self.hidden_size).to(x.device))
                for _ in range(self.num_layers * self.num_directions)
            ]

        hidden_seq = []
        for layer in range(self.num_layers):
            h_t, c_t = init_states[layer]
            forward_seq = []
            for t in range(seq_sz):
                x_t = x[:, t, :] if layer == 0 else hidden_seq[layer - 1][:, t, :]
                h_t, c_t = self.lstm_cells[layer](x_t, (h_t, c_t))
                forward_seq.append(h_t.unsqueeze(1))
            forward_seq = torch.cat(forward_seq, dim=1)
            hidden_seq.append(forward_seq)

        if self.bidirectional:
            hidden_seq_back = []
            for layer in range(self.num_layers):
                h_t, c_t = init_states[self.num_layers + layer]
                backward_seq = []
                for t in range(seq_sz - 1, -1, -1):
                    x_t = x[:, t, :] if layer == 0 else hidden_seq_back[layer - 1][:, seq_sz - 1 - t, :]
                    h_t, c_t = self.lstm_cells_back[layer](x_t, (h_t, c_t))
                    backward_seq.append(h_t.unsqueeze(1))
                backward_seq = torch.cat(backward_seq, dim=1)
                hidden_seq_back.append(backward_seq)

            hidden_seq = [torch.cat((f_seq, b_seq), dim=2) for f_seq, b_seq in zip(hidden_seq, hidden_seq_back)]

        return hidden_seq[-1], init_states

class LSTMCellWithPeephole(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        self.W_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_f = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        self.W_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_i = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        self.W_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        self.W_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_o = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight)
            else:
                nn.init.zeros_(weight)

    def forward(self, x, states):
        h_t, c_t = states
        f_t = torch.sigmoid(x @ self.U_f + h_t @ self.W_f + self.b_f + self.V_f * c_t)
        i_t = torch.sigmoid(x @ self.U_i + h_t @ self.W_i + self.b_i + self.V_i * c_t)
        c_t = f_t * c_t + i_t * torch.tanh(x @ self.U_c + h_t @ self.W_c + self.b_c)
        o_t = torch.sigmoid(x @ self.U_o + h_t @ self.W_o + self.b_o + self.V_o * c_t)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t



class LSTMPeepholebid(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        forecast_steps: int,
        hidden_size: int,
        num_layers: int = 2,
        bidirectional: bool = True,
    ) -> None:
        """
        Initialize the LSTMPeepholebid module.

        Args:
            input_size (int): The size of the input.
            output_size (int): The size of the output.
            forecast_steps (int): The number of forecast steps.
            hidden_size (int): The size of the hidden state.
            num_layers (int, optional): The number of layers. Defaults to 2.
            bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
        """
        super().__init__()
        self.output_size = output_size
        self.forecast_steps = forecast_steps

        self.lstm_peephole1 = CustomLSTM_With_Peephole(
            hidden_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional
        )
        self.lstm_peephole2 = CustomLSTM_With_Peephole(
            hidden_size * (2 if bidirectional else 1),
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.project_x1 = nn.Linear(input_size, hidden_size)
        self.add_norm1 = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))
        self.add_norm2 = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))
        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size * (2 if bidirectional else 1))
        self.fc2 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size * (2 if bidirectional else 1))
        self.fc3 = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.fc4 = nn.Linear(
            self.output_size * self.forecast_steps * 2,
            self.output_size * self.forecast_steps,
        )

    def forward(self, x, _):
        residual = self.project_x1(x)
        out, hidden = self.lstm_peephole1(residual, None)
        
        residual = self.fc1(out)
        residual = self.add_norm1(out + residual)

        out, hidden = self.lstm_peephole2(residual, hidden)
        
        residual = self.fc2(out)
        out = self.add_norm2(out + residual)
        
        out = self.fc3(out)
        out = out.view(-1, 1, self.output_size * self.forecast_steps * 2)
        out = self.fc4(out)
        out = out.view(-1, self.output_size * self.forecast_steps, 1)
        
        return out
