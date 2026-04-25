from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macls.models.uatr_cmoe_resnet_ap_top1 import UATRCMoETextResNetAPTop1


TEXT_PROTOTYPE_PATH = ROOT / "assets" / "text_prototypes" / "ship_type_sbert_384_original.pt"


def check_finite(name, value):
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN/Inf")


def grad_norm(module):
    total = None
    for param in module.parameters():
        if param.requires_grad and param.grad is not None:
            value = param.grad.detach().norm(p=2)
            total = value if total is None else total + value
    if total is None:
        return torch.tensor(float("nan"))
    return total.cpu()


def module_has_finite_grad(module):
    value = grad_norm(module)
    return bool(torch.isfinite(value) and value > 0)


def parameters_overlap(left, right):
    left_params = {id(param) for param in left.parameters()}
    right_params = {id(param) for param in right.parameters()}
    return bool(left_params.intersection(right_params))


def main():
    torch.manual_seed(20260425)

    batch_size, time_steps, feature_dim = 2, 64, 80
    num_class = 4
    num_experts = 4
    balance_weight = 0.01
    text_proto_weight = 0.05

    model = UATRCMoETextResNetAPTop1(
        num_class=num_class,
        input_size=feature_dim,
        base_channels=8,
        embd_dim=64,
        expert_hidden_dim=128,
        num_experts=num_experts,
        dropout=0.0,
        text_dim=384,
        text_prototype_path=TEXT_PROTOTYPE_PATH,
        balance_loss_weight=balance_weight,
        text_proto_weight=text_proto_weight,
        proto_temperature=0.5,
        learnable_temperature=False,
        balance_loss_type="uatr_cmoe",
        projection_head="linear",
        attention_heads=8,
        attention_dropout=0.0,
        attention_pool_type="learnable_query",
    )
    model.train()

    if model.text_prototypes_raw.requires_grad:
        raise RuntimeError("text_prototypes_raw must be frozen")
    if parameters_overlap(model.experts, model.audio_proj):
        raise RuntimeError("experts and audio_proj share parameters")

    audio = torch.randn(batch_size, time_steps, feature_dim)
    labels = torch.tensor([0, 1], dtype=torch.long)

    inference_logits = model(audio)
    if tuple(inference_logits.shape) != (batch_size, num_class):
        raise RuntimeError(f"Unexpected inference logits shape: {tuple(inference_logits.shape)}")

    outputs = model(audio, labels=labels)
    if not isinstance(outputs, dict):
        raise RuntimeError("model(audio, labels) must return a dict")

    required_keys = [
        "logits",
        "ce_loss",
        "balance_loss",
        "text_proto_loss",
        "total_loss",
        "gate_importance",
        "gate_fraction",
        "gate_entropy",
        "proto_logits",
    ]
    missing = [key for key in required_keys if key not in outputs]
    if missing:
        raise RuntimeError(f"Missing output keys: {missing}")

    logits = outputs["logits"]
    ce_loss = outputs["ce_loss"]
    balance_loss = outputs["balance_loss"]
    text_proto_loss = outputs["text_proto_loss"]
    total_loss = outputs["total_loss"]
    proto_logits = outputs["proto_logits"]
    gate_importance = outputs["gate_importance"]
    gate_fraction = outputs["gate_fraction"]
    gate_entropy = outputs["gate_entropy"]

    if tuple(logits.shape) != (batch_size, num_class):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if tuple(proto_logits.shape) != (batch_size, num_class):
        raise RuntimeError(f"Unexpected proto_logits shape: {tuple(proto_logits.shape)}")
    if tuple(gate_importance.shape) != (num_experts,):
        raise RuntimeError(f"Unexpected gate_importance shape: {tuple(gate_importance.shape)}")
    if tuple(gate_fraction.shape) != (num_experts,):
        raise RuntimeError(f"Unexpected gate_fraction shape: {tuple(gate_fraction.shape)}")
    if float(balance_loss.detach().cpu()) < 0:
        raise RuntimeError("balance_loss must be >= 0")

    expected_total = ce_loss + balance_weight * balance_loss + text_proto_weight * text_proto_loss
    if not torch.allclose(total_loss, expected_total, atol=1e-5):
        raise RuntimeError("total_loss formula mismatch")

    for name, value in {
        "logits": logits,
        "ce_loss": ce_loss,
        "balance_loss": balance_loss,
        "text_proto_loss": text_proto_loss,
        "total_loss": total_loss,
        "proto_logits": proto_logits,
        "gate_importance": gate_importance,
        "gate_fraction": gate_fraction,
        "gate_entropy": gate_entropy,
    }.items():
        check_finite(name, value)

    total_loss.backward()

    checks = {
        "experts": module_has_finite_grad(model.experts),
        "audio_proj": module_has_finite_grad(model.audio_proj),
        "text_proj": module_has_finite_grad(model.text_proj),
        "router": module_has_finite_grad(model.router),
        "attention_pooling": module_has_finite_grad(model.attention_pooling),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Gradient check failed: {failed}")

    print("uatr_cmoe_text_resnet_ap_top1_smoke: PASS")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"proto_logits shape: {tuple(proto_logits.shape)}")
    print(f"attention pooled shape: {outputs['attention_pooled_shape']}")
    print(f"expert_id: {outputs['expert_id'].detach().cpu().tolist()}")
    print(f"loss_total: {float(total_loss.detach().cpu()):.6f}")
    print(f"loss_ce: {float(ce_loss.detach().cpu()):.6f}")
    print(f"loss_balance: {float(balance_loss.detach().cpu()):.6f}")
    print(f"loss_text_proto: {float(text_proto_loss.detach().cpu()):.6f}")
    print(f"gate_importance: {[round(v, 6) for v in gate_importance.detach().cpu().tolist()]}")
    print(f"gate_fraction: {[round(v, 6) for v in gate_fraction.detach().cpu().tolist()]}")
    print(f"gate_entropy: {float(gate_entropy.detach().cpu()):.6f}")
    print(f"grad_norm/experts: {float(grad_norm(model.experts)):.6f}")
    print(f"grad_norm/audio_proj: {float(grad_norm(model.audio_proj)):.6f}")
    print(f"grad_norm/text_proj: {float(grad_norm(model.text_proj)):.6f}")
    print(f"grad_norm/router: {float(grad_norm(model.router)):.6f}")
    print(f"grad_norm/attention_pooling: {float(grad_norm(model.attention_pooling)):.6f}")
    print("experts_audio_proj_parameter_overlap: PASS")
    print("all finite checks: PASS")


if __name__ == "__main__":
    main()
