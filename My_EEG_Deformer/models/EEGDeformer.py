import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Shallow Feature Encoder
# ----------------------------
class ShallowEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, out_channels, (1, kernel_length), padding=(0, kernel_length // 2))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.spatial_conv = nn.Conv2d(out_channels, out_channels, (in_channels, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()

    def forward(self, x):
        # x shape: [batch, 1, channels, time]
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        return x  # shape: [batch, features, 1, time]

# ----------------------------
# Fine Branch Block (1D conv)
# ----------------------------
class FineBranch(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return self.act(x)

# ----------------------------
# Coarse Branch Block (Transformer)
# ----------------------------
class CoarseBranch(nn.Module):
    def __init__(self, dim, heads, dropout=0.5):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        return self.attn(x)


# ----------------------------
# HCT Block = Fine + Coarse
# ----------------------------
class HCTBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.5):
        super().__init__()
        self.coarse = CoarseBranch(dim, heads, dropout=dropout)
        self.fine = FineBranch(dim, dropout=dropout)

    def forward(self, x):
        x_coarse = self.coarse(x)
        x_fine = self.fine(x.transpose(1, 2)).transpose(1, 2)  # Conv1d expects (B, C, T)
        return x_coarse + x_fine, x_fine

# ----------------------------
# DIP (Dense Information Purification)
# ----------------------------
class DIP(nn.Module):
    def __init__(self, dim, num_blocks):
        super().__init__()
        self.conv = nn.Conv1d(dim * num_blocks, dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.ELU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, fine_outputs):
        x = torch.cat(fine_outputs, dim=1)  # concat on channel dim
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x).squeeze(-1)
        return x  # shape: [batch, dim]

# ----------------------------
# EEG-Deformer Model
# ----------------------------
class EEGDeformer(nn.Module):
    def __init__(self, in_channels=30, seq_len=384, num_classes=3, dropout=0.5):
        super().__init__()
        dim = 64  # feature dimension
        num_blocks = 4  # number of HCT blocks

        self.encoder = ShallowEncoder(in_channels, dim, kernel_length=64)
        self.proj = nn.Conv2d(dim, dim, (1, 1))
        self.hct_blocks = nn.ModuleList([HCTBlock(dim, heads=4, dropout=dropout) for _ in range(num_blocks)])
        self.dip = DIP(dim, num_blocks)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Input shape: [batch, 1, channels, time]
        x = self.encoder(x)  # [B, dim, 1, T]
        x = self.proj(x)  # [B, dim, 1, T]
        x = x.squeeze(2).transpose(1, 2)  # [B, T, dim]

        fine_features = []
        for block in self.hct_blocks:
            x, fine = block(x)
            fine_features.append(fine.transpose(1, 2))  # to [B, dim, T] for DIP

        x = self.dip(fine_features)
        return self.classifier(x)

# Exemple d'appel
if __name__ == "__main__":
    model = EEGDeformer()
    dummy = torch.randn(8, 1, 30, 384)  # batch de 8 EEG
    output = model(dummy)
    print(output.shape)  # [8, 2] pour classification binaire
