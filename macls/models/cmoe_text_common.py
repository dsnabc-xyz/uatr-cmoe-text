from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPECTED_TEXT_CLASS_NAMES = ["cargo", "tanker", "passenger", "tug"]

CLASS_NAME_ALIASES = {
    "cargo": "cargo",
    "cargo ship": "cargo",
    "tanker": "tanker",
    "oil tanker": "tanker",
    "passenger": "passenger",
    "passenger ship": "passenger",
    "tug": "tug",
    "tugboat": "tug",
    "tug boat": "tug",
}


def normalize_ship_class_name(name):
    normalized = " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())
    return CLASS_NAME_ALIASES.get(normalized, normalized)


def normalize_ship_class_names(class_names):
    return [normalize_ship_class_name(name) for name in class_names]


def validate_ship_class_names(class_names, expected_class_names=None):
    expected_class_names = expected_class_names or EXPECTED_TEXT_CLASS_NAMES
    normalized = normalize_ship_class_names(class_names)
    expected = normalize_ship_class_names(expected_class_names)
    if normalized != expected:
        raise RuntimeError(
            "Ship class names mismatch. "
            f"raw labels={list(class_names)}, "
            f"normalized labels={normalized}, "
            f"expected labels={expected}"
        )
    return normalized


def load_text_prototypes(path, text_dim=384, expected_class_names=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Text prototype file not found: {path}. "
            "Please run: python tools/build_ship_text_prototypes.py --template_set original"
        )
    payload = torch.load(path, map_location="cpu")
    embeddings = payload.get("embeddings")
    class_names = payload.get("class_names")
    if embeddings is None:
        raise RuntimeError(f"{path} does not contain payload['embeddings']")
    if class_names is None:
        raise RuntimeError(f"{path} does not contain payload['class_names']")
    if tuple(embeddings.shape) != (len(EXPECTED_TEXT_CLASS_NAMES), int(text_dim)):
        raise RuntimeError(
            f"Expected payload['embeddings'] shape "
            f"({len(EXPECTED_TEXT_CLASS_NAMES)}, {int(text_dim)}), got {tuple(embeddings.shape)}"
        )
    validate_ship_class_names(class_names, expected_class_names)
    return embeddings.float()


def build_projection_head(input_dim, output_dim, dropout=0.0, head_type="mlp"):
    if head_type == "linear":
        return nn.Linear(input_dim, output_dim)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
    raise ValueError(f"Unsupported projection_head: {head_type}")


def uatr_cmoe_balance_loss(router_probs):
    num_experts = router_probs.size(1)
    expert_id = torch.argmax(router_probs, dim=1)
    expert_mask = F.one_hot(expert_id, num_classes=num_experts).to(dtype=router_probs.dtype)
    f_i = expert_mask.mean(dim=0)
    p_i = router_probs.mean(dim=0)
    return num_experts * torch.sum(f_i * p_i), f_i, p_i


def gate_entropy(router_probs):
    return -(router_probs * (router_probs + 1e-8).log()).sum(dim=1).mean()


def compute_proto_logits(audio_proj, text_proj, mixed_feat, text_prototypes_raw, logit_scale):
    audio_z = F.normalize(audio_proj(mixed_feat), dim=1)
    text_z = F.normalize(text_proj(text_prototypes_raw), dim=1)
    proto_logits = audio_z @ text_z.T * logit_scale
    return proto_logits, audio_z, text_z
