import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from macls.models.cmoe_text_common import (
    build_projection_head,
    compute_proto_logits,
    load_text_prototypes,
)
from macls.models.uatr_cmoe_resnet_ap import AttentionPooling2D, BasicBlock


class Top1LogitExpert(nn.Module):
    def __init__(self, embd_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_class),
        )

    def forward(self, x):
        return self.net(x)


class ResNetAPBackbone(nn.Module):
    def __init__(self, input_size, base_channels=32, attention_heads=8,
                 attention_dropout=0.1, attention_pool_type="learnable_query"):
        super().__init__()
        self.input_size = int(input_size)
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
        self.out_channels = base_channels * 8
        self.attention_pooling = AttentionPooling2D(
            embed_dim=self.out_channels,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            attention_pool_type=attention_pool_type,
        )

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

    def _to_image(self, x):
        if x.dim() != 3:
            raise RuntimeError(f"Expected audio feature [B,T,F] or [B,F,T], got shape {tuple(x.shape)}")
        if x.size(-1) == self.input_size:
            return x.unsqueeze(1)
        if x.size(1) == self.input_size:
            return x.transpose(1, 2).unsqueeze(1)
        return x.unsqueeze(1)

    def forward(self, x):
        x = self._to_image(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.attention_pooling(x)


class UATRCMoETextResNetAPTop1(nn.Module):
    def __init__(self,
                 num_class,
                 input_size,
                 base_channels=32,
                 embd_dim=256,
                 expert_hidden_dim=128,
                 num_experts=4,
                 dropout=0.1,
                 text_dim=384,
                 text_prototype_path=None,
                 balance_loss_weight=0.01,
                 text_proto_weight=0.0,
                 proto_temperature=0.5,
                 learnable_temperature=False,
                 balance_loss_type="uatr_cmoe",
                 projection_head="mlp",
                 attention_heads=8,
                 attention_dropout=0.1,
                 attention_pool_type="learnable_query",
                 logit_scale_min=0.01,
                 logit_scale_max=100.0):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if balance_loss_type != "uatr_cmoe":
            raise ValueError("Only balance_loss_type='uatr_cmoe' is supported")

        self.emb_size = embd_dim
        self.text_dim = int(text_dim)
        self.balance_loss_weight = float(balance_loss_weight)
        self.text_proto_weight = float(text_proto_weight)
        self.proto_temperature = max(float(proto_temperature), 1e-8)
        self.learnable_temperature = bool(learnable_temperature)
        self.balance_loss_type = balance_loss_type
        self.logit_scale_min = float(logit_scale_min)
        self.logit_scale_max = float(logit_scale_max)

        self.backbone = ResNetAPBackbone(
            input_size=input_size,
            base_channels=base_channels,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            attention_pool_type=attention_pool_type,
        )
        self.attention_pooling = self.backbone.attention_pooling
        self.pre_moe_proj = nn.Linear(self.backbone.out_channels, embd_dim)
        self.router = nn.Linear(embd_dim, num_experts)
        self.gate = self.router
        self.experts = nn.ModuleList([
            Top1LogitExpert(
                embd_dim=embd_dim,
                hidden_dim=expert_hidden_dim,
                num_class=num_class,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])
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
        pooled_feat = self.backbone(x)
        audio_feat = self.pre_moe_proj(pooled_feat)

        router_logits = self.router(audio_feat)
        router_probs = F.softmax(router_logits, dim=1)
        expert_id = router_probs.argmax(dim=1)

        expert_logits = torch.stack([expert(audio_feat) for expert in self.experts], dim=1)
        batch_index = torch.arange(expert_logits.size(0), device=expert_logits.device)
        logits = expert_logits[batch_index, expert_id]

        num_experts = router_probs.size(1)
        expert_mask = F.one_hot(expert_id, num_classes=num_experts).to(dtype=router_probs.dtype)
        gate_fraction = expert_mask.mean(dim=0)
        gate_importance = router_probs.mean(dim=0)
        gate_entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-8))).sum(dim=1).mean()
        balance_loss = num_experts * torch.sum(gate_fraction * gate_importance)

        self.last_balance_loss = balance_loss
        self.last_gate_fraction = gate_fraction.detach()
        self.last_gate_importance = gate_importance.detach()
        self.last_gate_entropy = gate_entropy.detach()
        self.last_text_proto_loss = None

        if labels is None:
            if return_aux:
                return logits, self.get_last_moe_aux()
            return logits

        text_prototypes_raw = self._resolve_text_prototypes(text_prototypes, audio_feat)
        if text_prototypes_raw is None:
            raise RuntimeError("UATRCMoETextResNetAPTop1 requires text_prototypes_raw for text_proto_loss")

        if proto_temperature is None:
            logit_scale = self._current_logit_scale(audio_feat)
        else:
            logit_scale = audio_feat.new_tensor(1.0 / max(float(proto_temperature), 1e-8))
            logit_scale = logit_scale.clamp(min=self.logit_scale_min, max=self.logit_scale_max)

        proto_logits, audio_z, text_z = compute_proto_logits(
            audio_proj=self.audio_proj,
            text_proj=self.text_proj,
            mixed_feat=audio_feat,
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
            "proto_logits": proto_logits,
            "audio_z": audio_z,
            "text_z": text_z,
            "logit_scale": logit_scale.detach(),
            "expert_id": expert_id.detach(),
            "attention_pooled_shape": self.attention_pooling.last_output_shape,
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


class ResNetAPClassifier(nn.Module):
    def __init__(self,
                 num_class,
                 input_size,
                 base_channels=64,
                 embd_dim=512,
                 dropout=0.2,
                 attention_heads=8,
                 attention_dropout=0.1,
                 attention_pool_type="mean_query",
                 classifier_use_bn=False,
                 classifier_hidden_dim=128):
        super().__init__()
        del embd_dim
        self.backbone = ResNetAPBackbone(
            input_size=input_size,
            base_channels=base_channels,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            attention_pool_type=attention_pool_type,
        )
        self.attention_pooling = self.backbone.attention_pooling

        if classifier_hidden_dim is None:
            self.classifier = nn.Linear(self.backbone.out_channels, num_class)
        else:
            classifier_layers = [
                nn.Linear(self.backbone.out_channels, classifier_hidden_dim),
            ]
            if classifier_use_bn:
                classifier_layers.append(nn.BatchNorm1d(classifier_hidden_dim))
            classifier_layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, num_class),
            ])
            self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        pooled_feat = self.backbone(x)
        return self.classifier(pooled_feat)
