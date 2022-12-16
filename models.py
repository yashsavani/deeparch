import math
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


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


def Mlp(in_features, hidden_features=None, out_features=None, drop=0.):
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.GELU(),
        nn.Dropout(drop),
        nn.Linear(hidden_features, out_features),
        nn.Dropout(drop)
    )


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """

    def __init__(self, dim, seq_len, mlp_ratio=(0.5, 4.0), drop=0.):
        super().__init__()
        tokens_dim, channels_dim = int(dim*mlp_ratio[0]), int(dim * mlp_ratio[1])
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_tokens = Mlp(seq_len, tokens_dim, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_channels = Mlp(dim, channels_dim, drop=drop)

    def forward(self, x):
        x = x + self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.mlp_channels(self.norm2(x))
        return x


class MlpMixer(nn.Module):

    def __init__(
            self,
            num_classes=1000,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            drop_rate=0.,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.grad_checkpointing = False

        self.stem = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.blocks = nn.Sequential(*[
            MixerBlock(embed_dim, 16, mlp_ratio, drop=drop_rate)
            for _ in range(num_blocks)])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, self.num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
