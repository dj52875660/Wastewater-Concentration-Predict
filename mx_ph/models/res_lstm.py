import torch.nn as nn


class ResLSTMModel(nn.Module):
    def __init__(
        self, input_size, output_size, forecast_steps, hidden_size, num_layers
    ):
        super(ResLSTMModel, self).__init__()
        self.output_size = output_size
        self.forecast_steps = forecast_steps

        self.embedding = nn.Linear(input_size, hidden_size)
        self.lstm_encoder = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm_decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size * forecast_steps)

    def forward(self, x, tgt_hist):
        x_embed = self.embedding(x)
        out, (hidden, cell) = self.lstm_encoder(x_embed, None)
        out, _ = self.lstm_decoder(x_embed, (hidden, cell))
        out = self.fc(out[:, -1, :])
        residual_out = tgt_hist[:, :, 0]
        out += residual_out
        out = out.view(-1, self.forecast_steps, self.output_size)
        return out
