import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.residual_linear = nn.Linear(input_size, hidden_size * 2)  # Adjust dimension

    def forward(self, x):
        residual = self.residual_linear(x)  # Match dimension for residual
        x = self.linear(x)
        out, (hidden, cell) = self.lstm(x)
        out = torch.add(out, residual)  # Use torch.add() instead of inplace operation
        return out, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            hidden_size * 2, num_heads, batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.residual_linear = nn.Linear(input_size, hidden_size * 2)  # Adjust dimension

    def forward(self, tgt_hist, hidden, cell, encoder_outputs):
        residual = self.residual_linear(tgt_hist)  # Match dimension for residual
        tgt_hist = self.linear(tgt_hist)
        out, _ = self.lstm(tgt_hist, (hidden, cell))
        out = torch.add(out, residual)  # Use torch.add() instead of inplace operation
        attn_output, _ = self.attention(out, encoder_outputs, encoder_outputs)
        out = self.fc(attn_output)
        return out


class ResSeq2Seq(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        num_heads,
    ):
        super(ResSeq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(
            output_size, hidden_size, output_size, num_layers, num_heads
        )

    def forward(self, src, tgt_hist):
        out, hidden, cell = self.encoder(src)
        output = self.decoder(tgt_hist, hidden, cell, out)
        return output
