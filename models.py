import math
from typing import Tuple, Optional, Dict, Iterable, Union, Sequence, List, Any

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, convnext_tiny, ConvNeXt_Tiny_Weights


def _normalize_layer_list(layers: Union[str, Sequence[str]]) -> Sequence[str]:
    if isinstance(layers, str):
        items: List[str] = []
        for token in layers.replace(';', ',').split(','):
            token = token.strip()
            if token:
                items.append(token)
        return items or ['layer3', 'layer4']
    return list(layers)


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int = 4, alpha: Optional[float] = None):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.rank = max(1, int(rank))
        alpha = float(alpha) if alpha is not None else float(self.rank)
        self.scale = alpha / self.rank

        self.lora_down = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=self.rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=False,
        )
        self.lora_up = nn.Conv2d(
            in_channels=self.rank,
            out_channels=base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale


def _inject_lora_into_module(module: nn.Module, rank: int, alpha: Optional[float]) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            module._modules[name] = LoRAConv2d(child, rank=rank, alpha=alpha)
        else:
            _inject_lora_into_module(child, rank, alpha)


def inject_lora_modules(model: nn.Module,
                        layers: Iterable[str],
                        rank: int = 4,
                        alpha: Optional[float] = None) -> None:
    for layer_name in layers:
        if not hasattr(model, layer_name):
            continue
        layer = getattr(model, layer_name)
        _inject_lora_into_module(layer, rank=rank, alpha=alpha)


def freeze_params_for_lora(model: nn.Module, train_bn: bool = False) -> None:
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    if train_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True


class SinusoidalPE(nn.Module):
    def __init__(self, in_dims: int = 2, L: int = 6, scale: float = 2 * math.pi):
        super().__init__()
        self.L = L
        self.scale = scale
        self.in_dims = in_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x]
        for k in range(self.L):
            freq = (2.0 ** k) * self.scale
            outs += [torch.sin(freq * x), torch.cos(freq * x)]
        return torch.cat(outs, dim=-1)


class BranchCNN(nn.Module):
    def __init__(self,
                 out_dim: int = 512,
                 pretrained: bool = True,
                 dropout: float = 0.1,
                 variant: str = 'resnet18',
                 **variant_kwargs):
        super().__init__()
        self.variant = (variant or 'resnet18').lower()
        self.out_dim = out_dim
        self.variant_kwargs = variant_kwargs
        self.backbone: nn.Module
        self.feature_extractor: nn.Module
        self.stem: nn.Module
        self._feature_dim: int

        if self.variant in ('resnet18', 'resnet'):
            self._build_resnet(pretrained, use_lora=False)
        elif self.variant in ('resnet18_lora', 'reslora', 'res-lora'):
            self._build_resnet(pretrained, use_lora=True)
        elif self.variant in ('convnext_t', 'convnext-t', 'convnext_tiny', 'convnext'):
            self._build_convnext(pretrained)
        else:
            raise ValueError(f'Unsupported branch variant "{variant}".')

        proj_dropout = float(variant_kwargs.get('proj_dropout', dropout))
        self.proj = nn.Sequential(
            nn.Dropout(proj_dropout),
            nn.Linear(self._feature_dim, out_dim)
        )
        nn.init.xavier_uniform_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def _build_resnet(self, pretrained: bool, use_lora: bool) -> None:
        kwargs = self.variant_kwargs
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        old_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

        if use_lora:
            layers = _normalize_layer_list(kwargs.get('lora_layers', ('layer3', 'layer4')))
            rank = int(kwargs.get('lora_rank', 4))
            alpha = kwargs.get('lora_alpha')
            inject_lora_modules(backbone, layers=layers, rank=rank, alpha=alpha)
            freeze_params_for_lora(backbone, train_bn=bool(kwargs.get('lora_train_bn', False)))

        self.backbone = backbone
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.stem = self.feature_extractor
        self._feature_dim = 512

    def _build_convnext(self, pretrained: bool) -> None:
        kwargs = self.variant_kwargs
        drop_path = float(kwargs.get('drop_path_rate', kwargs.get('convnext_drop_path_rate', 0.0)))
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights, drop_path_rate=drop_path)

        patch_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1,
            patch_conv.out_channels,
            kernel_size=patch_conv.kernel_size,
            stride=patch_conv.stride,
            padding=patch_conv.padding,
            bias=patch_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(patch_conv.weight.mean(dim=1, keepdim=True))
                if patch_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(patch_conv.bias)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)
        backbone.features[0][0] = new_conv

        self.backbone = backbone
        self.feature_extractor = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.stem = self.feature_extractor
        self._feature_dim = backbone.classifier[2].in_features
        train_stages = kwargs.get('train_stages', kwargs.get('convnext_train_stages'))
        self._apply_convnext_stage_freeze(train_stages)

    def _apply_convnext_stage_freeze(self, train_stages: Optional[Union[str, Iterable[int]]]) -> None:
        if train_stages is None:
            return
        stages = list(enumerate(self.backbone.features))
        if not stages:
            return

        if isinstance(train_stages, str):
            key = train_stages.lower()
            if key in ('all', '*'):
                train_idx = {idx for idx, _ in stages}
            elif key in ('last', 'last1'):
                train_idx = {len(stages) - 1}
            elif key == 'last2':
                train_idx = {max(0, len(stages) - 2), len(stages) - 1}
            elif key in ('none', 'freeze_all', ''):
                train_idx = set()
            else:
                train_idx = {int(token.strip()) for token in key.split(',') if token.strip().isdigit()}
        else:
            train_idx = {int(idx) for idx in train_stages}

        for idx, stage in stages:
            requires_grad = idx in train_idx
            for param in stage.parameters():
                param.requires_grad = requires_grad
        if 0 not in train_idx:
            for param in self.backbone.features[0].parameters():
                param.requires_grad = False

    def apply_freeze(self, key: str) -> None:
        if not key:
            return
        key = str(key).lower()
        if key == 'none':
            return
        if self.variant.startswith('resnet'):
            layer_map = {
                'layer1': self.backbone.layer1,
                'layer2': self.backbone.layer2,
                'layer3': self.backbone.layer3,
                'layer4': self.backbone.layer4,
            }
            targets = set()
            if key in ('layer1', 'layer1-2', 'layer12', 'layer1_2'):
                targets.add('layer1')
            if key in ('layer2', 'layer1-2', 'layer12', 'layer1_2'):
                targets.add('layer2')
            if key in ('layer3', 'layer34', 'all'):
                targets.add('layer3')
            if key in ('layer4', 'layer34', 'all'):
                targets.add('layer4')
            for name in targets:
                module = layer_map.get(name)
                if module is None:
                    continue
                for param in module.parameters():
                    param.requires_grad = False
        elif self.variant.startswith('convnext'):
            if key == 'all':
                for param in self.backbone.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.proj(features)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


class BranchCond(nn.Module):
    def __init__(self, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim), nn.GELU(),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.mlp(c)


class TrunkMLP(nn.Module):
    def __init__(self,
                 K: int = 256,
                 L: int = 6,
                 hidden: int = 256,
                 depth: int = 4,
                 act: str = 'gelu',
                 film_layers: Optional[Iterable[int]] = None,
                 cond_dim: Optional[int] = None,
                 film_gain: float = 1.0,
                 cond_mode: str = 'film'):
        super().__init__()
        self.pe = SinusoidalPE(2, L)
        self.cond_mode = (cond_mode or 'film').lower()
        base_dim = 2 + 4 * L
        in_dim = base_dim + (int(cond_dim) if (self.cond_mode == 'concat' and cond_dim is not None) else 0)
        self.activation = nn.GELU() if act == 'gelu' else nn.ReLU()
        depth = max(1, int(depth))
        self.hidden_layers = nn.ModuleList()
        input_dim = in_dim
        for layer_idx in range(depth):
            self.hidden_layers.append(nn.Linear(input_dim, hidden))
            input_dim = hidden
        self.output_layer = nn.Linear(hidden, K)

        raw_layers = []
        if film_layers is not None:
            if isinstance(film_layers, Iterable) and not isinstance(film_layers, (str, bytes)):
                raw_layers = list(film_layers)
            else:
                raw_layers = [film_layers]
        self.film_layers = sorted({int(idx) for idx in raw_layers if int(idx) >= 0})
        self.film_gain = float(film_gain)
        self.film_scale = nn.ModuleDict()
        self.film_shift = nn.ModuleDict()
        if self.cond_mode == 'film' and self.film_layers and cond_dim is not None:
            for idx in self.film_layers:
                if idx >= len(self.hidden_layers):
                    continue
                key = str(idx)
                self.film_scale[key] = nn.Linear(int(cond_dim), hidden)
                self.film_shift[key] = nn.Linear(int(cond_dim), hidden)
        elif self.cond_mode != 'film':
            self.film_layers = []

    def forward(self, coords: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            squeeze_output = True
        elif coords.dim() == 3:
            squeeze_output = False
        else:
            raise ValueError(f'Unsupported coordinate shape: {tuple(coords.shape)}')

        if cond is not None and cond.dim() == 1:
            cond = cond.unsqueeze(0)

        if cond is not None and cond.shape[0] != coords.shape[0]:
            if cond.shape[0] == 1:
                cond = cond.expand(coords.shape[0], -1)
            elif coords.shape[0] == 1:
                coords = coords.expand(cond.shape[0], -1, -1)
            else:
                raise ValueError('Mismatch between coords batch and cond batch sizes.')

        pe = self.pe(coords)
        if self.cond_mode == 'concat' and cond is not None:
            cond_exp = cond.unsqueeze(1).expand(-1, pe.shape[1], -1)
            h = torch.cat([pe, cond_exp], dim=-1)
        else:
            h = pe
        for idx, layer in enumerate(self.hidden_layers):
            h = layer(h)
            h = self.activation(h)
            if self.cond_mode == 'film' and cond is not None and str(idx) in self.film_scale:
                scale = torch.tanh(self.film_scale[str(idx)](cond)) * self.film_gain
                shift = self.film_shift[str(idx)](cond)
                h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.output_layer(h)
        if squeeze_output and (cond is None or cond.shape[0] == 1):
            out = out.squeeze(0)
        return out


class TrunkFFMLP(nn.Module):
    """Random Fourier Feature + MLP trunk."""
    def __init__(self,
                 K: int = 256,
                 L: int = 6,
                 hidden: int = 256,
                 depth: int = 4,
                 fourier_dim: int = 256,
                 fourier_sigma: float = 1.0,
                 cond_dim: Optional[int] = None,
                 cond_mode: str = 'film',
                 film_gain: float = 1.0):
        super().__init__()
        self.cond_mode = (cond_mode or 'film').lower()
        self.film_gain = float(film_gain)
        base_dim = 2 + 4 * L
        self.pe = SinusoidalPE(2, L)
        self.fourier_dim = int(fourier_dim)
        self.register_buffer('rff_w', torch.randn(base_dim + (cond_dim or 0), self.fourier_dim) / max(1e-6, float(fourier_sigma)))
        self.register_buffer('rff_b', torch.rand(self.fourier_dim) * 2 * math.pi)
        in_dim = self.fourier_dim * 2
        if self.cond_mode == 'concat' and cond_dim is not None:
            in_dim += cond_dim
        self.hidden_layers = nn.ModuleList()
        input_dim = in_dim
        depth = max(1, int(depth))
        for _ in range(depth):
            self.hidden_layers.append(nn.Linear(input_dim, hidden, bias=False))
            input_dim = hidden
        self.output_layer = nn.Linear(hidden, K, bias=False)

        raw_layers = []
        self.film_layers = []
        self.film_scale = nn.ModuleDict()
        self.film_shift = nn.ModuleDict()
        if self.cond_mode == 'film' and cond_dim is not None:
            raw_layers = list(range(depth))
            self.film_layers = sorted({int(idx) for idx in raw_layers if int(idx) >= 0})
            for idx in self.film_layers:
                key = str(idx)
                self.film_scale[key] = nn.Linear(int(cond_dim), hidden)
                self.film_shift[key] = nn.Linear(int(cond_dim), hidden)

    def forward(self, coords: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            squeeze_output = True
        elif coords.dim() == 3:
            squeeze_output = False
        else:
            raise ValueError(f'Unsupported coordinate shape: {tuple(coords.shape)}')

        if cond is not None and cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if cond is not None and cond.shape[0] != coords.shape[0]:
            if cond.shape[0] == 1:
                cond = cond.expand(coords.shape[0], -1)
            elif coords.shape[0] == 1:
                coords = coords.expand(cond.shape[0], -1, -1)
            else:
                raise ValueError('Mismatch between coords batch and cond batch sizes.')

        pe = self.pe(coords)
        if cond is not None:
            pe_cond = torch.cat([pe, cond.unsqueeze(1).expand(-1, pe.shape[1], -1)], dim=-1)
        elif pe.shape[-1] == self.rff_w.shape[0]:
            pe_cond = pe
        else:
            # cond missing but rff expects extra dims -> pad zeros
            pad_dim = self.rff_w.shape[0] - pe.shape[-1]
            pad = torch.zeros(pe.shape[0], pe.shape[1], pad_dim, device=pe.device, dtype=pe.dtype)
            pe_cond = torch.cat([pe, pad], dim=-1)
        proj = pe_cond @ self.rff_w  # [B,N,m]
        feats = torch.cat([torch.sin(proj + self.rff_b), torch.cos(proj + self.rff_b)], dim=-1)
        h = feats
        if self.cond_mode == 'concat' and cond is not None:
            cond_exp = cond.unsqueeze(1).expand(-1, h.shape[1], -1)
            h = torch.cat([h, cond_exp], dim=-1)
        for idx, layer in enumerate(self.hidden_layers):
            h = layer(h)
            h = torch.relu(h)
            if self.cond_mode == 'film' and cond is not None and str(idx) in self.film_scale:
                scale = torch.tanh(self.film_scale[str(idx)](cond)) * self.film_gain
                shift = self.film_shift[str(idx)](cond)
                h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.output_layer(h)
        if squeeze_output and (cond is None or cond.shape[0] == 1):
            out = out.squeeze(0)
        return out


class TrunkSIREN(nn.Module):
    """SIREN trunk with optional FiLM/concat conditioning. Bias disabled."""
    def __init__(self,
                 K: int = 256,
                 hidden: int = 256,
                 depth: int = 5,
                 w0: float = 30.0,
                 cond_dim: Optional[int] = None,
                 cond_mode: str = 'film',
                 film_gain: float = 1.0):
        super().__init__()
        self.w0 = float(w0)
        self.cond_mode = (cond_mode or 'film').lower()
        self.film_gain = float(film_gain)
        in_dim = 2 + (int(cond_dim) if (self.cond_mode == 'concat' and cond_dim is not None) else 0)
        self.first = nn.Linear(in_dim, hidden, bias=False)
        self.hidden_layers = nn.ModuleList()
        depth = max(1, int(depth))
        for _ in range(depth - 1):
            self.hidden_layers.append(nn.Linear(hidden, hidden, bias=False))
        self.output_layer = nn.Linear(hidden, K, bias=False)

        raw_layers = []
        self.film_layers = []
        self.film_scale = nn.ModuleDict()
        self.film_shift = nn.ModuleDict()
        if self.cond_mode == 'film' and cond_dim is not None:
            raw_layers = list(range(depth))
            self.film_layers = sorted({int(idx) for idx in raw_layers if int(idx) >= 0})
            for idx in self.film_layers:
                key = str(idx)
                self.film_scale[key] = nn.Linear(int(cond_dim), hidden)
                self.film_shift[key] = nn.Linear(int(cond_dim), hidden)
        self._apply_siren_init()

    def sine(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)

    def _apply_siren_init(self) -> None:
        def init_linear(layer: nn.Linear, w0: float, is_first: bool = False):
            fan_in = layer.in_features
            if is_first:
                bound = 1.0 / fan_in
            else:
                bound = math.sqrt(6.0 / fan_in) / w0
            nn.init.uniform_(layer.weight, -bound, bound)

        init_linear(self.first, self.w0, is_first=True)
        for layer in self.hidden_layers:
            init_linear(layer, self.w0, is_first=False)
        init_linear(self.output_layer, self.w0, is_first=False)

    def forward(self, coords: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            squeeze_output = True
        elif coords.dim() == 3:
            squeeze_output = False
        else:
            raise ValueError(f'Unsupported coordinate shape: {tuple(coords.shape)}')

        if cond is not None and cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if cond is not None and cond.shape[0] != coords.shape[0]:
            if cond.shape[0] == 1:
                cond = cond.expand(coords.shape[0], -1)
            elif coords.shape[0] == 1:
                coords = coords.expand(cond.shape[0], -1, -1)
            else:
                raise ValueError('Mismatch between coords batch and cond batch sizes.')

        coords = coords.clamp(0.0, 1.0)
        if cond is not None:
            cond = cond.clamp(0.0, 1.0)

        if self.cond_mode == 'concat' and cond is not None:
            cond_exp = cond.unsqueeze(1).expand(-1, coords.shape[1], -1)
            h = torch.cat([coords, cond_exp], dim=-1)
        else:
            h = coords
        h = self.first(h)
        h = self.sine(h)
        for idx, layer in enumerate(self.hidden_layers):
            h = layer(h)
            if self.cond_mode == 'film' and cond is not None and str(idx) in self.film_scale:
                scale = torch.tanh(self.film_scale[str(idx)](cond)) * self.film_gain
                shift = self.film_shift[str(idx)](cond)
                h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            h = self.sine(h)
        out = self.output_layer(h)
        if squeeze_output and (cond is None or cond.shape[0] == 1):
            out = out.squeeze(0)
        return out


class RDeepONetV2(nn.Module):
    def __init__(self,
                 K: int = 256,
                 pretrained: bool = True,
                 dropout: float = 0.1,
                 L: int = 6,
                 hidden: int = 256,
                 depth: int = 4,
                 cond_hidden: int = 128,
                 cond_out: int = 64,
                 branch_variant: str = 'resnet18',
                 branch_params: Optional[Dict[str, Any]] = None,
                 branch_out_dim: int = 512,
                 film_layers: Optional[Iterable[int]] = None,
                 film_gain: float = 1.0,
                 trunk_type: str = 'mlp',
                 trunk_cond_mode: str = 'film',
                 trunk_fourier_dim: int = 256,
                 trunk_fourier_sigma: float = 1.0,
                 trunk_w0: float = 30.0):
        super().__init__()
        branch_params = dict(branch_params or {})
        self.branch_variant = branch_variant
        self.branch_out_dim = int(branch_out_dim)
        self.trunk_type = (trunk_type or 'mlp').lower()

        self.branch_cnn = BranchCNN(
            out_dim=self.branch_out_dim,
            pretrained=pretrained,
            dropout=dropout,
            variant=branch_variant,
            **branch_params
        )
        self.branch_cond = BranchCond(hidden_dim=cond_hidden, out_dim=cond_out)
        self.branch_proj = nn.Linear(self.branch_out_dim + cond_out, K)

        if self.trunk_type in ('mlp', 'mlp_pe', 'baseline'):
            self.trunk = TrunkMLP(
                K=K,
                L=L,
                hidden=hidden,
                depth=depth,
                film_layers=film_layers,
                cond_dim=cond_out,
                film_gain=film_gain,
                act='gelu',
                cond_mode=trunk_cond_mode
            )
        elif self.trunk_type in ('ff', 'ffmlp', 'rff'):
            self.trunk = TrunkFFMLP(
                K=K,
                L=L,
                hidden=hidden,
                depth=depth,
                fourier_dim=trunk_fourier_dim,
                fourier_sigma=trunk_fourier_sigma,
                cond_dim=cond_out,
                cond_mode=trunk_cond_mode,
                film_gain=film_gain
            )
        elif self.trunk_type in ('siren',):
            self.trunk = TrunkSIREN(
                K=K,
                hidden=hidden,
                depth=depth,
                w0=trunk_w0,
                cond_dim=cond_out,
                cond_mode=trunk_cond_mode,
                film_gain=film_gain
            )
        else:
            raise ValueError(f"Unsupported trunk_type: {self.trunk_type}")

    def forward_full(self, ray: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b_img = self.branch_cnn(ray)
        b_cond = self.branch_cond(cond)
        b = self.branch_proj(torch.cat([b_img, b_cond], dim=-1))
        device = ray.device
        grid_size = ray.shape[-1]
        lin = torch.linspace(0.0, 1.0, grid_size, device=device)
        R, Z = torch.meshgrid(lin, lin, indexing='xy')
        coords = torch.stack([R, Z], dim=-1).view(-1, 2)
        phi = self.trunk(coords, b_cond)
        if phi.dim() == 2:
            out = (phi @ b.unsqueeze(-1)).view(grid_size, grid_size)
            return out
        pred = torch.einsum('bnk,bk->bn', phi, b)
        pred = pred.view(b.shape[0], grid_size, grid_size)
        return pred if pred.shape[0] > 1 else pred[0]

    def forward_coord(self, ray: torch.Tensor, cond: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        b_img = self.branch_cnn(ray)
        b_cond = self.branch_cond(cond)
        b = self.branch_proj(torch.cat([b_img, b_cond], dim=-1))
        phi = self.trunk(coords, b_cond)
        if phi.dim() == 2:
            pred = phi @ b.unsqueeze(-1)
            return pred.squeeze(-1)
        return torch.einsum('bnk,bk->bn', phi, b)

