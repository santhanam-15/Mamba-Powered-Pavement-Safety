import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBranch(nn.Module):
    """Local-detail extraction branch with 3x3, 5x5, and 7x7 filters."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = channels // 3
        remainder = channels - 2 * branch_channels
        self.conv3 = ConvBNAct(channels, branch_channels, kernel_size=3)
        self.conv5 = ConvBNAct(channels, branch_channels, kernel_size=5)
        self.conv7 = ConvBNAct(channels, remainder, kernel_size=7)
        self.fuse = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1))


class OptimizedMambaBlock(nn.Module):
    """
    Lightweight Mamba-style block for global context.

    This keeps the stage-2 structure requested by the flow while staying
    runnable with plain PyTorch and no external Mamba package.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.token_mixer = nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.state_proj = nn.Conv1d(channels, channels * 2, kernel_size=1, bias=False)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens).transpose(1, 2)
        mixed = self.token_mixer(tokens)
        state, gate = self.state_proj(mixed).chunk(2, dim=1)
        mixed = self.out_proj(torch.tanh(state) * torch.sigmoid(gate))
        mixed = mixed.view(batch_size, channels, height, width)
        return x + self.scale * mixed


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))


class MSAAModule(nn.Module):
    """Multi-scale attention aggregation block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre_fuse = ConvBNAct(channels * 2, channels, kernel_size=1)
        self.scale3 = DepthwiseSeparableConv(channels, channels)
        self.scale5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.scale7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
        self.out_fuse = ConvBNAct(channels * 3, channels, kernel_size=1)

    def forward(self, local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        fused = self.pre_fuse(torch.cat([local_features, global_features], dim=1))
        multi_scale = torch.cat([self.scale3(fused), self.scale5(fused), self.scale7(fused)], dim=1)
        multi_scale = self.out_fuse(multi_scale)
        multi_scale = multi_scale * self.channel_attention(multi_scale)
        multi_scale = multi_scale * self.spatial_attention(multi_scale)
        return multi_scale


class LightweightMLPDecoder(nn.Module):
    """Upsampling segmentation head."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.reduce = ConvBNAct(in_channels, in_channels // 2, kernel_size=1)
        self.skip = ConvBNAct(skip_channels, in_channels // 2, kernel_size=1)
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels // 2),
            DepthwiseSeparableConv(in_channels // 2, in_channels // 4),
            ConvBNAct(in_channels // 4, in_channels // 4, kernel_size=3),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

    def forward(self, fused_features: torch.Tensor, skip_features: torch.Tensor, output_size: int) -> torch.Tensor:
        upsampled = F.interpolate(self.reduce(fused_features), size=skip_features.shape[-2:], mode="bilinear", align_corners=False)
        merged = torch.cat([upsampled, self.skip(skip_features)], dim=1)
        refined = self.dropout(self.refine(merged))
        logits = self.head(refined)
        return F.interpolate(logits, size=(output_size, output_size), mode="bilinear", align_corners=False)


class EdgeDetectionHead(nn.Module):
    """Lightweight auxiliary boundary head operating on MSAA features."""

    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, output_size: int) -> torch.Tensor:
        edge_map = self.block(x)
        return F.interpolate(edge_map, size=(output_size, output_size), mode="bilinear", align_corners=False)


class CMEncoder(nn.Module):
    """Shared encoder feeding both the CNN branch and the Mamba branch."""

    def __init__(self, in_channels: int = 3, stem_channels: int = 64, out_channels: int = 128) -> None:
        super().__init__()
        self.stem1 = ConvBNAct(in_channels, stem_channels, kernel_size=3, stride=2)
        self.skip = ConvBNAct(stem_channels, stem_channels, kernel_size=3)
        self.stem2 = ConvBNAct(stem_channels, stem_channels, kernel_size=3)
        self.stem3 = ConvBNAct(stem_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem1(x)
        skip_features = self.skip(x)
        x = self.stem2(x)
        encoded = self.stem3(x)
        return encoded, skip_features


class CMSegNet(nn.Module):
    """Stage-2 CMSegNet-style segmentation model."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, img_size: int = 256) -> None:
        super().__init__()
        self.img_size = img_size
        self.encoder = CMEncoder(in_channels=in_channels, stem_channels=64, out_channels=128)
        self.cnn_branch = CNNBranch(128)
        self.optimized_mamba = OptimizedMambaBlock(128)
        self.msaa = MSAAModule(128)
        self.decoder = LightweightMLPDecoder(in_channels=128, skip_channels=64, out_channels=out_channels)
        self.edge_head = EdgeDetectionHead(in_channels=128)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, skip_features = self.encoder(x)
        local_features = self.cnn_branch(encoded)
        global_features = self.optimized_mamba(encoded)
        fused = self.msaa(local_features, global_features)
        segmentation_logits = self.decoder(fused, skip_features, output_size=self.img_size)
        edge_map = self.edge_head(fused, output_size=self.img_size)
        return segmentation_logits, edge_map


class ShadowNet(nn.Module):
    """Lightweight U-Net for shadow probability masks."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 6, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        return self.out(dec1)
