from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macls.models.uatr_cmoe_resnet18 import UATRCMoETextResNet18


TEXT_PROTOTYPE_PATH = ROOT / "assets" / "text_prototypes" / "ship_type_sbert_384_original.pt"


def check_finite(name, value):
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN/Inf")


def main():
    torch.manual_seed(20260425)

    batch_size, time_steps, feature_dim = 2, 64, 80
    labels = torch.tensor([0, 1], dtype=torch.long)

    model = UATRCMoETextResNet18(
        num_class=4,
        input_size=feature_dim,
        base_channels=8,
        embd_dim=64,
        num_experts=4,
        dropout=0.0,
        text_dim=384,
        text_prototype_path=TEXT_PROTOTYPE_PATH,
        balance_loss_weight=0.01,
        text_proto_weight=0.05,
        proto_temperature=0.5,
        learnable_temperature=False,
        balance_loss_type="uatr_cmoe",
        projection_head="linear",
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)

    features = torch.randn(batch_size, time_steps, feature_dim)

    optimizer.zero_grad()
    outputs = model(features, labels=labels)
    loss = outputs["total_loss"]
    logits = outputs["logits"]

    required = {
        "loss_total": loss,
        "loss_ce": outputs["ce_loss"],
        "loss_balance": outputs["balance_loss"],
        "loss_text_proto": outputs["text_proto_loss"],
        "logit_scale": outputs["logit_scale"],
        "gate_entropy": outputs["gate_entropy"],
        "gate_importance": outputs["gate_importance"],
        "logits": logits,
        "proto_logits": outputs["proto_logits"],
    }
    for name, value in required.items():
        check_finite(name, value)

    if tuple(logits.shape) != (batch_size, 4):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

    loss.backward()
    optimizer.step()

    print("one_batch_trainer_integration: PASS")
    print(f"features shape: {tuple(features.shape)}")
    print(f"labels: {labels.tolist()}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss_total: {float(loss.detach().cpu()):.6f}")
    print(f"loss_ce: {float(outputs['ce_loss'].detach().cpu()):.6f}")
    print(f"loss_balance: {float(outputs['balance_loss'].detach().cpu()):.6f}")
    print(f"loss_text_proto: {float(outputs['text_proto_loss'].detach().cpu()):.6f}")
    print(f"logit_scale: {float(outputs['logit_scale'].detach().cpu()):.6f}")
    print(f"gate_entropy: {float(outputs['gate_entropy'].detach().cpu()):.6f}")
    print(f"gate_importance: {[round(v, 6) for v in outputs['gate_importance'].detach().cpu().tolist()]}")
    print("all finite checks: PASS")
    print("backward: PASS")
    print("optimizer.step: PASS")


if __name__ == "__main__":
    main()
