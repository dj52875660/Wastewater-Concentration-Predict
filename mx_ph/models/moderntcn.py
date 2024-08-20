import torch
import torch.nn as nn
from moderntcn import ModernTCN


class ModernTCNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(ModernTCNModel, self).__init__()
        self.tcn = ModernTCN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, x):
        return self.tcn(x)
