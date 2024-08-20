import torch
import torch.nn as nn


class CustomLSTM_With_Peephole(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.W_f = nn.Parameter(
            torch.Tensor(hidden_sz, hidden_sz)
        )  # switched wf an uf creation (hidden hidde) instead of (input, hidden)
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
            if len(weight.shape) > 1:  # Weight matrix
                nn.init.xavier_normal_(weight)
            else:  # Bias vector
                nn.init.zeros_(weight)

    def forward(self, x, init_states=None):
        bs, seq_sz, _ = x.shape
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size).to(x.device)
            c_t = torch.zeros(bs, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            f_t = torch.sigmoid(
                x_t @ self.U_f + h_t @ self.W_f + self.b_f + self.V_f * c_t
            )
            i_t = torch.sigmoid(
                x_t @ self.U_i + h_t @ self.W_i + self.b_i + self.V_i * c_t
            )
            c_t = f_t * c_t + i_t * torch.tanh(
                x_t @ self.U_c + h_t @ self.W_c + self.b_c
            )
            o_t = torch.sigmoid(
                x_t @ self.U_o + h_t @ self.W_o + self.b_o + self.V_o * c_t
            )
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


# class LSTMPeephole(nn.Module):

#     def __init__(
#         self, input_size: int, output_size: int, forecast_steps: int, hidden_size: int
#     ):
#         super(LSTMPeephole, self).__init__()
#         self.output_size = output_size
#         self.forecast_steps = forecast_steps

#         self.lstm_peephole1 = CustomLSTM_With_Peephole(input_size, hidden_size)
#         self.lstm_peephole2 = CustomLSTM_With_Peephole(hidden_size, hidden_size)
#         self.project_x1 = nn.Linear(input_size, hidden_size)
#         self.add_norm1 = nn.LayerNorm(hidden_size)
#         self.add_norm2 = nn.LayerNorm(hidden_size)
#         self.add_norm3 = nn.LayerNorm(hidden_size)
#         self.add_norm4 = nn.LayerNorm(hidden_size)
#         self.fc1 = nn.Linear(hidden_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
#         self.fc4 = nn.Linear(
#             self.output_size * self.forecast_steps * 2,
#             self.output_size * self.forecast_steps,
#         )

#     def forward(self, x, _):
#         residual = self.project_x1(x)
#         out, hidden = self.lstm_peephole1(x, None)
#         out = self.add_norm1(out + residual)
#         residual = self.fc1(out)
#         residual = self.add_norm2(out + residual)
#         out, hidden = self.lstm_peephole2(residual, hidden)
#         out = self.add_norm2(out + residual)
#         residual = self.fc2(out)
#         out = self.add_norm4(out + residual)
#         # out = self.layer_3(out[:, -self.forecast_steps :, :])

#         out = self.fc3(out)
#         out = out.view(-1, 1, self.output_size * self.forecast_steps * 2)
#         out = self.fc4(out)
#         out = out.view(-1, self.output_size * self.forecast_steps, 1)
#         return out


class LSTMPeephole(nn.Module):

    def __init__(
        self, input_size: int, output_size: int, forecast_steps: int, hidden_size: int
    ):
        super(LSTMPeephole, self).__init__()
        self.output_size = output_size
        self.forecast_steps = forecast_steps

        self.embedding = nn.Linear(input_size, hidden_size)
        self.lstm_encoder = CustomLSTM_With_Peephole(hidden_size, hidden_size)
        self.lstm_decoder = CustomLSTM_With_Peephole(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size * forecast_steps)
        self.residual_fc = nn.Linear(forecast_steps,  forecast_steps)

    def forward(self, x, tgt_hist):
        x_embed = self.embedding(x)
        out, (hidden, cell) = self.lstm_encoder(x_embed)
        out, _ = self.lstm_decoder(x_embed, (hidden, cell))
        out = self.fc(out[:, -1, :])
        tgt_hist = tgt_hist.view(-1, self.output_size, self.forecast_steps)
        residual_out = self.residual_fc(tgt_hist)
        out += residual_out[:, 0, :]
        out = out.view(-1, self.forecast_steps, self.output_size)

        return out
