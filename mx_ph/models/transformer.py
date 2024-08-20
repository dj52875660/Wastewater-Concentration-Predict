import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(TransformerModel, self).__init__()
        self.encode_embedding = nn.Linear(input_size, hidden_size)
        self.decode_embedding = nn.Linear(output_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=512,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt_hist):
        src = self.encode_embedding(src) * torch.sqrt(
            torch.tensor(src.size(-1), dtype=torch.float32)
        )
        src = self.positional_encoding(src)
        tgt_hist = self.decode_embedding(tgt_hist) * torch.sqrt(
            torch.tensor(tgt_hist.size(-1), dtype=torch.float32)
        )
        tgt_hist = self.positional_encoding(tgt_hist)
        output = self.transformer(src, tgt_hist)
        output = self.fc_out(output)
        return output
