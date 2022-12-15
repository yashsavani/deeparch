from torch import nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_hid):
        super(MLP, self).__init__()
        self.fcin = nn.Linear(input_size, hidden_size)
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for i in range(num_hid)
        ])
        self.fcout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fcin(x)
        for i, fc in enumerate(self.fcs):
            x = F.relu(fc(x))
        x = self.fcout(x)
        return x

class ViT(nn.Module):

    def __init__(self, dim, depth, heads):
        super(ViT, self).__init__()
        self.to_patches = nn.Conv2d(3, dim, kernel_size=2, stride=2)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim*4,
                batch_first=True,
            ),
            num_layers=depth
        )
        self.to_logits = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.to_patches(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.to_logits(x)
        return x
