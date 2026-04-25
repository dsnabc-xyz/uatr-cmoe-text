import torch
import torch.nn.functional as F

from macls.models.uatr_cmoe_resnet18 import UATRCMoETextResNet18


def tensor_has_nan_or_inf(x):
    return bool(torch.isnan(x).any() or torch.isinf(x).any())


def main():
    torch.manual_seed(20260424)

    batch_size, time_steps, feature_dim = 2, 1198, 300
    num_class = 4
    balance_loss_weight = 0.01

    model = UATRCMoETextResNet18(num_class=num_class, input_size=feature_dim)
    model.train()

    x = torch.randn(batch_size, time_steps, feature_dim)
    labels = torch.tensor([0, 1], dtype=torch.long)

    logits, aux = model(x, labels=labels, return_aux=True)
    ce_loss = F.cross_entropy(logits, labels)
    balance_loss = aux["balance_loss"]
    gate_importance = aux["gate_importance"]
    gate_entropy = aux["gate_entropy"]
    total_loss = ce_loss + balance_loss_weight * balance_loss
    total_loss.backward()

    print(f"logits shape: {tuple(logits.shape)}")
    print(f"ce_loss: {float(ce_loss.detach().cpu()):.6f}")
    print(f"balance_loss: {float(balance_loss.detach().cpu()):.6f}")
    print(f"total_loss: {float(total_loss.detach().cpu()):.6f}")
    print(f"gate_importance: {[round(v, 6) for v in gate_importance.detach().cpu().tolist()]}")
    print(f"gate_entropy: {float(gate_entropy.detach().cpu()):.6f}")
    print(f"logits has NaN/Inf: {tensor_has_nan_or_inf(logits)}")
    print(f"ce_loss has NaN/Inf: {tensor_has_nan_or_inf(ce_loss)}")
    print(f"balance_loss has NaN/Inf: {tensor_has_nan_or_inf(balance_loss)}")
    print(f"total_loss has NaN/Inf: {tensor_has_nan_or_inf(total_loss)}")


if __name__ == "__main__":
    main()
