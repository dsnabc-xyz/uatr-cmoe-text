from pathlib import Path

import torch

from macls.models.cmoe_text_common import load_text_prototypes
from macls.models.uatr_cmoe_resnet18 import UATRCMoETextResNet18


TEXT_PROTOTYPE_PATH = Path("assets/text_prototypes/ship_type_sbert_384_original.pt")


def tensor_has_nan_or_inf(x):
    return bool(torch.isnan(x).any() or torch.isinf(x).any())


def check_finite(name, x):
    if tensor_has_nan_or_inf(x):
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
    torch.manual_seed(20260424)

    batch_size, time_steps, feature_dim = 2, 1198, 300
    num_class = 4
    balance_weight = 0.01
    text_proto_weight = 0.05
    proto_temperature = 0.5

    text_prototypes = load_text_prototypes(TEXT_PROTOTYPE_PATH, text_dim=384)
    model = UATRCMoETextResNet18(
        num_class=num_class,
        input_size=feature_dim,
        text_dim=384,
        embd_dim=256,
        text_prototype_path=TEXT_PROTOTYPE_PATH,
        balance_loss_weight=balance_weight,
        text_proto_weight=text_proto_weight,
        proto_temperature=proto_temperature,
        learnable_temperature=False,
        balance_loss_type="uatr_cmoe",
    )
    model.train()

    if model.text_prototypes_raw is None:
        raise RuntimeError("text_prototypes_raw buffer is missing")
    if model.text_prototypes_raw.requires_grad:
        raise RuntimeError("text_prototypes_raw must be frozen")
    if parameters_overlap(model.classifier, model.audio_proj):
        raise RuntimeError("classifier and audio_proj share parameters")

    x = torch.randn(batch_size, time_steps, feature_dim)
    labels = torch.tensor([0, 1], dtype=torch.long)

    audio_only_logits = model(x)
    if tuple(audio_only_logits.shape) != (batch_size, num_class):
        raise RuntimeError(f"Unexpected audio-only logits shape: {tuple(audio_only_logits.shape)}")

    output = model(
        x,
        labels=labels,
        text_prototypes=text_prototypes,
        balance_weight=balance_weight,
        text_proto_weight=text_proto_weight,
        proto_temperature=proto_temperature,
    )
    if not isinstance(output, dict):
        raise RuntimeError("model(audio, labels) must return a dict")

    logits = output["logits"]
    ce_loss = output["ce_loss"]
    balance_loss = output["balance_loss"]
    text_proto_loss = output["text_proto_loss"]
    total_loss = output["total_loss"]
    gate_importance = output["gate_importance"]
    gate_fraction = output["gate_fraction"]
    gate_entropy = output["gate_entropy"]
    audio_z = output["audio_z"]
    text_z = output["text_z"]
    proto_logits = output["proto_logits"]

    if tuple(proto_logits.shape) != (batch_size, num_class):
        raise RuntimeError(f"Unexpected proto_logits shape: {tuple(proto_logits.shape)}")

    expected_total_loss = ce_loss + balance_weight * balance_loss + text_proto_weight * text_proto_loss
    total_loss_ok = torch.allclose(total_loss, expected_total_loss, atol=1e-5)
    if not total_loss_ok:
        raise RuntimeError(
            f"total_loss mismatch: got {float(total_loss.detach().cpu()):.6f}, "
            f"expected {float(expected_total_loss.detach().cpu()):.6f}"
        )

    for name, tensor in {
        "logits": logits,
        "audio_z": audio_z,
        "text_z": text_z,
        "proto_logits": proto_logits,
        "ce_loss": ce_loss,
        "balance_loss": balance_loss,
        "text_proto_loss": text_proto_loss,
        "total_loss": total_loss,
    }.items():
        check_finite(name, tensor)

    if not torch.allclose(audio_z.norm(dim=1), torch.ones(batch_size), atol=1e-5):
        raise RuntimeError("audio_z is not normalized")
    if not torch.allclose(text_z.norm(dim=1), torch.ones(num_class), atol=1e-5):
        raise RuntimeError("text_z is not normalized")

    total_loss.backward()

    classifier_grad = grad_norm(model.classifier)
    audio_proj_grad = grad_norm(model.audio_proj)
    text_proj_grad = grad_norm(model.text_proj)
    router_grad = grad_norm(model.router)
    expert_grad = grad_norm(model.experts)

    checks = {
        "classifier": module_has_finite_grad(model.classifier),
        "audio_proj": module_has_finite_grad(model.audio_proj),
        "text_proj": module_has_finite_grad(model.text_proj),
        "router": module_has_finite_grad(model.router),
        "experts": module_has_finite_grad(model.experts),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Gradient check failed: {failed}")

    grad_ratio = text_proj_grad / classifier_grad
    if not torch.isfinite(grad_ratio) or grad_ratio > 50:
        raise RuntimeError(f"grad_ratio/text_proj_to_classifier invalid: {float(grad_ratio):.6f}")
    ratio_status = "WARN" if grad_ratio > 10 else "PASS"

    print(f"text prototype path: {TEXT_PROTOTYPE_PATH.as_posix()}")
    print(f"text_prototypes shape: {tuple(text_prototypes.shape)}")
    print(f"text_prototypes_raw.requires_grad: {model.text_prototypes_raw.requires_grad}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"audio_z shape: {tuple(audio_z.shape)}")
    print(f"text_z shape: {tuple(text_z.shape)}")
    print(f"proto_logits shape: {tuple(proto_logits.shape)}")
    print(f"ce_loss: {float(ce_loss.detach().cpu()):.6f}")
    print(f"balance_loss: {float(balance_loss.detach().cpu()):.6f}")
    print(f"text_proto_loss: {float(text_proto_loss.detach().cpu()):.6f}")
    print(f"total_loss: {float(total_loss.detach().cpu()):.6f}")
    print(f"total_loss check: {'PASS' if total_loss_ok else 'FAIL'}")
    print(f"gate_importance: {[round(v, 6) for v in gate_importance.detach().cpu().tolist()]}")
    print(f"gate_fraction: {[round(v, 6) for v in gate_fraction.detach().cpu().tolist()]}")
    print(f"gate_entropy: {float(gate_entropy.detach().cpu()):.6f}")
    print("NaN/Inf checks: PASS")
    print(f"grad_norm/classifier: {float(classifier_grad):.6f}")
    print(f"grad_norm/audio_proj: {float(audio_proj_grad):.6f}")
    print(f"grad_norm/text_proj: {float(text_proj_grad):.6f}")
    print(f"grad_norm/router: {float(router_grad):.6f}")
    print(f"grad_norm/experts: {float(expert_grad):.6f}")
    print(f"grad_ratio/text_proj_to_classifier: {float(grad_ratio):.6f} [{ratio_status}]")
    print("classifier_audio_proj_parameter_overlap: PASS")


if __name__ == "__main__":
    main()
