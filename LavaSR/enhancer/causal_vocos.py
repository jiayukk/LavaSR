import torch
from torch import nn
from torch.nn.utils import weight_norm
from typing import Optional

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_padding = (self.kernel_size[0] - 1) * self.dilation[0]

    def forward(self, x):
        x = nn.functional.pad(x, (self.left_padding, 0))
        return super().forward(x)
      
class CausalConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim,
        intermediate_dim,
        layer_scale_init_value=1e-6,
        adanorm_num_embeddings=None,
    ):
        super().__init__()

        self.dwconv = CausalConv1d(
            dim,
            dim,
            kernel_size=7,
            groups=dim
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, x, cond_embedding_id=None):
        shortcut = x

        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.transpose(1, 2)
        return shortcut + x

class CausalResBlock1(nn.Module):
    def __init__(self, dim, layer_scale_init_value=None):
        super().__init__()

        self.conv1 = weight_norm(CausalConv1d(dim, dim, kernel_size=3))
        self.conv2 = weight_norm(CausalConv1d(dim, dim, kernel_size=3))

        self.act = nn.LeakyReLU(0.1)

        scale = layer_scale_init_value or 1e-2
        self.gamma = nn.Parameter(scale * torch.ones(dim))

    def forward(self, x):
        residual = x

        x = self.act(self.conv1(x))
        x = self.conv2(x)

        x = self.gamma.view(1, -1, 1) * x
        return residual + x
      
class VocosBackbone(nn.Module):

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()

        self.embed = CausalConv1d(
            input_channels,
            dim,
            kernel_size=7
        )

        self.adanorm = adanorm_num_embeddings is not None

        if self.adanorm:
            from modules import AdaLayerNorm
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)

        layer_scale_init_value = layer_scale_init_value or 1 / num_layers

        self.convnext = nn.ModuleList([
            CausalConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
                adanorm_num_embeddings=adanorm_num_embeddings,
            )
            for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs):
        bandwidth_id = kwargs.get('bandwidth_id', None)

        x = self.embed(x)

        if self.adanorm:
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))

        x = x.transpose(1, 2)

        for block in self.convnext:
            x = block(x, cond_embedding_id=bandwidth_id)

        x = self.final_layer_norm(x.transpose(1, 2))
        return x

class VocosResNetBackbone(nn.Module):

    def __init__(self, input_channels, dim, num_blocks, layer_scale_init_value=None):
        super().__init__()

        self.embed = weight_norm(
            CausalConv1d(input_channels, dim, kernel_size=3)
        )

        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3

        self.resnet = nn.Sequential(
            *[
                CausalResBlock1(
                    dim=dim,
                    layer_scale_init_value=layer_scale_init_value
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x

def _check_causality(model: nn.Module, input_channels: int, T_prefix: int, T_suffix: int, name: str = "model"):
    torch.set_grad_enabled(False)
    model.eval()
    # Build x1 = [a, b], x2 = [a, c]; time dim is last (B, C, T)
    a = torch.randn(1, input_channels, T_prefix)
    b = torch.randn(1, input_channels, T_suffix)
    c = torch.randn(1, input_channels, T_suffix)
    x1 = torch.cat([a, b], dim=2)
    x2 = torch.cat([a, c], dim=2)

    out1 = model(x1)
    out2 = model(x2)

    # Causal: first T_prefix output frames depend only on first T_prefix input frames (a), so must be identical
    prefix_out1 = out1[:, :T_prefix, :]
    prefix_out2 = out2[:, :T_prefix, :]
    diff_prefix = (prefix_out1 - prefix_out2).abs().max().item()
    # Suffix may differ (input b != c)
    suffix_out1 = out1[:, T_prefix:, :]
    suffix_out2 = out2[:, T_prefix:, :]
    diff_suffix = (suffix_out1 - suffix_out2).abs().max().item()

    causal_ok = diff_prefix < 1e-5
    print(f"[{name}] prefix output max diff: {diff_prefix:.2e} (expect ~0) | suffix output max diff: {diff_suffix:.2e}")
    print(f"[{name}] causality: {'PASS' if causal_ok else 'FAIL'}")
    return causal_ok

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    backbone = VocosBackbone(
        input_channels=32,
        dim=64,
        intermediate_dim=128,
        num_layers=2,
        adanorm_num_embeddings=None,
    ).eval()
    ok1 = _check_causality(backbone, input_channels=32, T_prefix=50, T_suffix=50, name="VocosBackbone")

    resnet_backbone = VocosResNetBackbone(
        input_channels=32,
        dim=64,
        num_blocks=2,
    ).eval()
    ok2 = _check_causality(resnet_backbone, input_channels=32, T_prefix=50, T_suffix=50, name="VocosResNetBackbone")

    if ok1 and ok2:
        print("\nAll causality checks passed.")
    else:
        print("\nSome causality checks failed.")
