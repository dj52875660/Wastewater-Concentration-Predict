import torch.nn as nn


# class LSTMModel(nn.Module):
#     def __init__(
#         self, input_size, output_size, forecast_steps, hidden_size, num_layers
#     ):
#         super(LSTMModel, self).__init__()
#         self.output_size = output_size
#         self.forecast_steps = forecast_steps

#         self.embedding = nn.Linear(input_size, hidden_size)
#         self.lstm_encoder = nn.LSTM(
#             hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True
#         )
#         self.fc = nn.Linear(hidden_size * 2, output_size * forecast_steps)

#     def forward(self, x, _):
#         x = self.embedding(x)
#         out, (hidden, cell) = self.lstm_encoder(x, None)
#         out = self.fc(out[:, -1, :])
#         out = out.view(-1, self.forecast_steps, self.output_size)
#         return out


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        forecast_steps,
        hidden_size,
        num_layers,
        dropout=0.5,
    ):
        super(LSTMModel, self).__init__()
        self.output_size = output_size
        self.forecast_steps = forecast_steps

        self.embedding = nn.Linear(input_size, hidden_size)
        self.lstm_encoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size * forecast_steps)

    def forward(self, x, _):
        x = self.embedding(x)
        out, (hidden, cell) = self.lstm_encoder(x, None)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = out.view(-1, self.forecast_steps, self.output_size)
        return out
