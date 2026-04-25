import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from macls.models.cmoe_text_common import (
    build_projection_head,
    compute_proto_logits,
    gate_entropy,
    load_text_prototypes,
    uatr_cmoe_balance_loss,
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class MLPExpert(nn.Module):
    def __init__(self, embd_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_dim, embd_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embd_dim, embd_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class UATRCMoETextResNet18(nn.Module):
    def __init__(self,
                 num_class,
                 input_size,
                 base_channels=32,
                 embd_dim=256,
                 num_experts=4,
                 dropout=0.1,
                 text_dim=384,
                 text_prototype_path=None,
                 balance_loss_weight=0.01,
                 text_proto_weight=0.05,
                 proto_temperature=0.5,
                 learnable_temperature=False,
                 balance_loss_type="uatr_cmoe",
                 projection_head="mlp",
                 logit_scale_min=0.01,
                 logit_scale_max=100.0):
        super().__init__()
        del input_size
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if balance_loss_type != "uatr_cmoe":
            raise ValueError("Only balance_loss_type='uatr_cmoe' is supported in the first version")

        self.emb_size = embd_dim
        self.text_dim = int(text_dim)
        self.balance_loss_weight = float(balance_loss_weight)
        self.text_proto_weight = float(text_proto_weight)
        self.proto_temperature = max(float(proto_temperature), 1e-8)
        self.learnable_temperature = bool(learnable_temperature)
        self.balance_loss_type = balance_loss_type
        self.logit_scale_min = float(logit_scale_min)
        self.logit_scale_max = float(logit_scale_max)

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_channels = base_channels
        self.layer1 = self._make_layer(base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, blocks=2, stride=2)

        backbone_out_channels = base_channels * 8
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pre_moe_proj = nn.Linear(backbone_out_channels, embd_dim)
        self.experts = nn.ModuleList([
            MLPExpert(embd_dim=embd_dim, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(embd_dim, num_experts)
        self.gate = self.router
        self.classifier = nn.Sequential(
            nn.Linear(embd_dim, embd_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embd_dim, num_class),
        )
        self.audio_proj = build_projection_head(
            input_dim=embd_dim,
            output_dim=embd_dim,
            dropout=dropout,
            head_type=projection_head,
        )
        self.text_proj = build_projection_head(
            input_dim=self.text_dim,
            output_dim=embd_dim,
            dropout=dropout,
            head_type=projection_head,
        )

        initial_logit_scale = 1.0 / self.proto_temperature
        if self.learnable_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(initial_logit_scale), dtype=torch.float32))
        else:
            self.register_buffer("fixed_logit_scale", torch.tensor(initial_logit_scale, dtype=torch.float32))

        self.last_balance_loss = None
        self.last_gate_importance = None
        self.last_gate_fraction = None
        self.last_gate_entropy = None
        self.last_text_proto_loss = None
        text_prototypes_raw = load_text_prototypes(text_prototype_path, self.text_dim) if text_prototype_path else None
        self.register_buffer("text_prototypes_raw", text_prototypes_raw)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _forward_backbone(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def _current_logit_scale(self, reference):
        if self.learnable_temperature:
            scale = self.logit_scale.exp()
        else:
            scale = self.fixed_logit_scale
        scale = scale.clamp(min=self.logit_scale_min, max=self.logit_scale_max)
        return scale.to(device=reference.device, dtype=reference.dtype)

    def forward(self,
                x,
                labels=None,
                text_prototypes=None,
                return_aux=False,
                balance_weight=None,
                text_proto_weight=None,
                proto_temperature=None):
        backbone_feat = self._forward_backbone(x)
        audio_feat = self.pre_moe_proj(backbone_feat)

        router_logits = self.router(audio_feat)
        router_probs = F.softmax(router_logits, dim=1)
        expert_outputs = torch.stack([expert(audio_feat) for expert in self.experts], dim=1)
        mixed_feat = torch.sum(expert_outputs * router_probs.unsqueeze(-1), dim=1)

        logits = self.classifier(mixed_feat)

        balance_loss, gate_fraction, gate_importance = uatr_cmoe_balance_loss(router_probs)
        entropy = gate_entropy(router_probs)
        self.last_balance_loss = balance_loss
        self.last_gate_fraction = gate_fraction.detach()
        self.last_gate_importance = gate_importance.detach()
        self.last_gate_entropy = entropy.detach()
        self.last_text_proto_loss = None

        if labels is None:
            if return_aux:
                return logits, self.get_last_moe_aux()
            return logits

        text_prototypes_raw = self._resolve_text_prototypes(text_prototypes, mixed_feat)
        if text_prototypes_raw is None:
            raise RuntimeError("UATRCMoETextResNet18 requires text_prototypes_raw for text_proto_loss")

        if proto_temperature is None:
            logit_scale = self._current_logit_scale(mixed_feat)
        else:
            logit_scale = mixed_feat.new_tensor(1.0 / max(float(proto_temperature), 1e-8))
            logit_scale = logit_scale.clamp(min=self.logit_scale_min, max=self.logit_scale_max)

        proto_logits, audio_z, text_z = compute_proto_logits(
            audio_proj=self.audio_proj,
            text_proj=self.text_proj,
            mixed_feat=mixed_feat,
            text_prototypes_raw=text_prototypes_raw,
            logit_scale=logit_scale,
        )
        labels = labels.to(device=x.device, dtype=torch.long)
        ce_loss = F.cross_entropy(logits, labels)
        text_proto_loss = F.cross_entropy(proto_logits, labels)
        self.last_text_proto_loss = text_proto_loss

        balance_weight = self.balance_loss_weight if balance_weight is None else float(balance_weight)
        text_proto_weight = self.text_proto_weight if text_proto_weight is None else float(text_proto_weight)
        total_loss = ce_loss + balance_weight * balance_loss + text_proto_weight * text_proto_loss

        output = {
            "logits": logits,
            "ce_loss": ce_loss,
            "balance_loss": balance_loss,
            "text_proto_loss": text_proto_loss,
            "total_loss": total_loss,
            "gate_importance": self.last_gate_importance,
            "gate_fraction": self.last_gate_fraction,
            "gate_entropy": self.last_gate_entropy,
            "audio_z": audio_z,
            "text_z": text_z,
            "proto_logits": proto_logits,
            "logit_scale": logit_scale.detach(),
        }
        if return_aux:
            return logits, output
        return output

    def get_last_moe_aux(self):
        aux = {
            "balance_loss": self.last_balance_loss,
            "gate_importance": self.last_gate_importance,
            "gate_fraction": self.last_gate_fraction,
            "gate_entropy": self.last_gate_entropy,
        }
        if self.last_text_proto_loss is not None:
            aux["text_proto_loss"] = self.last_text_proto_loss
        return aux

    def _resolve_text_prototypes(self, text_prototypes, reference):
        if text_prototypes is None:
            text_prototypes = self.text_prototypes_raw
        if text_prototypes is None:
            return None
        if text_prototypes.size(1) != self.text_dim:
            raise RuntimeError(
                f"Expected text prototypes with dim {self.text_dim}, got {text_prototypes.size(1)}"
            )
        return text_prototypes.to(device=reference.device, dtype=reference.dtype)
