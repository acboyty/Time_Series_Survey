import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = nn.LSTM(1, 10)
        self.dec = nn.Linear(500, 1)

    def forward(self, x):
        # x: [seq_len, batch, input_size]
        # hids: [seq_len, batch, hid_size] -> [batch, seq_len * hid_size]
        _, batch, _ = x.size()

        hids, _ = self.enc(x)
        hids = hids.permute(1, 0, 2).reshape(batch, -1)
        outputs = self.dec(hids)

        return outputs