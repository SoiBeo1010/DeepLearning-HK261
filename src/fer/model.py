"""Checkpoint-compatible CCT model; inference never downloads ImageNet weights."""

import torch
from torch import nn
from torchvision import models


class ResNetTokenizer(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x).flatten(2).transpose(1, 2)


class SeqPool(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.attention_pool = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention_pool(x).softmax(dim=1)
        return torch.matmul(weights.transpose(1, 2), x).squeeze(1)


class CCT_ResNet(nn.Module):
    """Keep parameter names and shapes compatible with the original checkpoint."""

    def __init__(
        self,
        num_classes: int = 7,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        pretrained: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tokenizer = ResNetTokenizer(pretrained=pretrained)
        self.positional_emb = nn.Parameter(torch.zeros(1, 49, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_pool = SeqPool(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x) + self.positional_emb
        return self.classifier(self.seq_pool(self.transformer(x)))
