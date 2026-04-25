# AGENTS.md

## Project Goal

This repository is being adapted from an AudioClassification-Pytorch style framework into a 4-class underwater acoustic target recognition model for ship-radiated noise classification.

The current main model is:

`UATRCMoETextResNet18`

Target classes:

- cargo
- tanker
- passenger
- tug

The model must support:

- audio-only inference
- text-prototype-guided training
- cMoE routing based only on audio features
- no text input during prediction

## Python Environment Rules

This project must use the following Python interpreter for all local Windows commands:

```text
D:\study\VirtualEnvironment\uatr_clean\python.exe
```

Do not use:

```text
D:\Anaconda3\python.exe
```

All Python commands, smoke tests, dependency checks, and package installs must explicitly use the project interpreter.

Use this pattern for compile checks:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -m compileall macls tools
```

Use this pattern for running scripts:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -u "e:\Projects\uatr-cmoe-text\tools\smoke_test_cmoe_text_common.py"
```

Use this pattern for installing dependencies:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -m pip install <package>
```

Before running smoke tests, verify the interpreter:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -c "import sys; print(sys.executable)"
```

The output must be:

```text
D:\study\VirtualEnvironment\uatr_clean\python.exe
```

Do not run bare commands such as:

```text
python script.py
pip install package
```

because they may use the wrong interpreter.

---

## Core Terminology

### Do not call the text prototype loss ITC

The loss defined as:

```python
audio_z = normalize(audio_proj(mixed_feat))
text_z = normalize(text_proj(text_prototypes_raw))
proto_logits = audio_z @ text_z.T / proto_temperature
text_proto_loss = cross_entropy(proto_logits, labels)
```

is not CLIP-style ITC.

It must be called:

- `text_proto_loss`
- or `text_proto_alignment_loss`

The logits must be called:

- `proto_logits`

Do not use these names as official fields:

- `itc_loss`
- `itc_weight`
- `audio_text_logits`

If old names exist only for backward compatibility, mark them as deprecated and do not use them in new code.

---

## Loss Definition

The total loss must be:

```text
total_loss = ce_loss + balance_weight * balance_loss + text_proto_weight * text_proto_loss
```

Required output field names:

```text
ce_loss
balance_loss
text_proto_loss
total_loss
proto_logits
```

Required config names:

```text
balance_weight
text_proto_weight
proto_temperature
learnable_temperature
balance_loss_type: "uatr_cmoe"
```

Do not introduce these names in new code:

```text
itc_weight
contrastive_temperature
```

---

## Model Architecture Constraints

The model structure must be:

```text
audio feature
  -> ResNet18 backbone
  -> cMoE router / experts
  -> mixed_feat
       |-- classifier -> logits -> CE loss
       `-- audio_proj -> audio_z -> text_proto_loss
```

Hard constraints:

1. The router must only receive audio features.
2. Text prototypes must not enter the router.
3. Text prototypes must not enter experts.
4. Text prototypes must not enter the classifier.
5. `classifier` and `audio_proj` must be independent modules.
6. `classifier` and `audio_proj` must not share parameters.
7. `proto_logits` must not replace classifier logits.
8. Inference must work with `model(audio)` only.

---

## Text Prototype Rules

Raw SBERT text prototypes must be frozen.

Implementation requirement:

```python
self.register_buffer("text_prototypes_raw", embeddings)
```

Rules:

1. `text_prototypes_raw.requires_grad` must be `False`.
2. `text_prototypes_raw` must not appear in `named_parameters()`.
3. `text_proj` is trainable.
4. `audio_proj` is trainable.
5. Text prototype file path defaults to:

```text
assets/text_prototypes/ship_type_sbert_384_original.pt
```

Expected raw shape:

```text
[4, 384]
```

---

## Class Name Normalization

Do not compare class names by raw string equality.

Before comparing labels or prototype class names, normalize with:

```text
strip().lower()
```

Also normalize `_` and `-` to spaces.

Aliases:

```text
cargo / cargo ship -> cargo
tanker / oil tanker -> tanker
passenger / passenger ship -> passenger
tug / tugboat / tug boat -> tug
```

Expected canonical order:

```python
["cargo", "tanker", "passenger", "tug"]
```

If mismatch occurs, error messages must show:

- raw labels
- normalized labels
- expected labels

---

## UATR-CMoE Balance Loss

The first version must use the UATR-CMoE style balance loss.

Formula:

```python
expert_id = torch.argmax(router_probs, dim=-1)
f_i = one_hot(expert_id).float().mean(dim=0)
p_i = router_probs.mean(dim=0)
balance_loss = num_experts * torch.sum(f_i * p_i)
```

Important:

1. `balance_loss` must be returned unweighted.
2. `balance_weight` is applied only inside `total_loss`.
3. Do not mix Switch Transformer `importance_l2` balance loss in the first version.
4. If another balance loss is added later, expose it through `balance_loss_type`.

---

## Temperature Rules

Default:

```yaml
proto_temperature: 0.5
learnable_temperature: false
```

If `learnable_temperature=true`:

1. Use a trainable log scale.
2. Clamp the value to avoid exploding `proto_logits`.
3. Log the effective temperature or logit scale.

Experiment candidates:

```yaml
proto_temperature: [0.07, 0.1, 0.5, 1.0]
learnable_temperature: [false, true]
```

---

## Forward Behavior

`model(audio)` must work for inference.

Recommended inference behavior:

```python
model(audio)
```

returns classifier logits tensor.

Training behavior:

```python
model(audio, labels=labels)
```

returns a dict containing at least:

```python
{
    "logits": logits,
    "ce_loss": ce_loss,
    "balance_loss": balance_loss,
    "text_proto_loss": text_proto_loss,
    "total_loss": total_loss,
    "proto_logits": proto_logits,
    "audio_emb": audio_z,
    "text_emb": text_z,
    "gate_importance": gate_importance,
    "gate_entropy": gate_entropy,
}
```

---

## Implementation Order

Follow this order. Do not skip ahead.

1. Implement or update `macls/models/cmoe_text_common.py`.
2. Implement or update `UATRCMoETextResNet18`.
3. Run model-level smoke test before touching trainer.
4. Update config and model registry.
5. Update trainer integration.
6. Run one-batch train smoke test.
7. Only after all tests pass, delete old plain cMoE+text entrypoints.
8. CQT frontend is future work and must not be implemented in this stage.

---

## Files Allowed in Current Stage

Expected files for this work:

```text
macls/models/cmoe_text_common.py
macls/models/uatr_cmoe_resnet18.py
tools/smoke_test_cmoe_text_common.py
tools/smoke_test_uatr_cmoe_text_resnet18.py
tools/smoke_train_one_batch_uatr_cmoe_text_resnet18.py
configs/deepship/uatr_cmoe_text_resnet18.yml
macls/models/__init__.py
macls/trainer.py
```

Do not modify unless explicitly requested:

```text
macls/predict.py
macls/data_utils/reader.py
macls/data_utils/collate_fn.py
assets/text_prototypes/*.pt
data split scripts
CQT/frontend implementation
```

---

## Required Smoke Tests

### Compile check

Command:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -m compileall macls tools
```

---

### Common smoke test

Command:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -u "e:\Projects\uatr-cmoe-text\tools\smoke_test_cmoe_text_common.py"
```

Must check:

- text prototype file exists
- raw embedding shape is `(4, 384)`
- `proto_logits.shape == (2, 4)`
- `text_proto_loss` is finite
- `audio_proj` has finite gradients
- `text_proj` has finite gradients

---

### Model-level smoke test

Command:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -u "e:\Projects\uatr-cmoe-text\tools\smoke_test_uatr_cmoe_text_resnet18.py"
```

Must check:

- `model(audio)` inference works
- classifier logits shape is `(2, 4)`
- `model(audio, labels)` returns dict
- `proto_logits.shape == (2, 4)`
- `text_proto_loss` is finite
- `total_loss` formula is correct
- `text_prototypes_raw.requires_grad == False`
- `text_prototypes_raw` is not in `named_parameters()`
- `text_proj` has finite gradients
- `audio_proj` has finite gradients
- `classifier` has finite gradients
- router or expert has finite gradients
- `classifier` and `audio_proj` do not share parameters

Gradient logs required:

```text
grad_norm/classifier
grad_norm/audio_proj
grad_norm/text_proj
grad_ratio/text_proj_to_classifier
```

If ratio > 10, print WARN.
If ratio > 50 or NaN/Inf, fail.

---

## Trainer Integration Rules

Only after model-level smoke test passes:

1. Training calls:

```python
model(features, labels=labels)
```

2. Evaluation and prediction call:

```python
model(features)
```

3. If model returns dict:

```python
logits = outputs["logits"]
loss = outputs["total_loss"]
```

4. Required logs:

```text
loss_total
loss_ce
loss_balance
loss_text_proto
gate_entropy
gate_importance
```

Do not log the main text loss as `loss_itc`.

---

## Experiment Matrix

Initial experiments:

```text
ResNet18 + CE
ResNet18 + cMoE + CE
ResNet18 + cMoE + CE + balance
UATRCMoETextResNet18 + CE + balance + text_proto_loss
```

Ablations:

```yaml
text_proto_weight: [0.01, 0.03, 0.05, 0.1]
proto_temperature: [0.07, 0.1, 0.5, 1.0]
learnable_temperature: [false, true]
projection_head: linear vs mlp
text_template: class_name vs original vs contrastive vs shuffled
balance_weight: [0.001, 0.003, 0.01, 0.03]
frontend: MelSpectrogram first, CQT later
```

---

## Done Criteria

A change is not done unless:

1. `compileall` passes.
2. Common smoke test passes.
3. Model-level smoke test passes.
4. No official field uses `itc_loss`, `itc_weight`, or `audio_text_logits`.
5. `model(audio)` still works for audio-only inference.
6. Text does not enter router, experts, or classifier.
7. `total_loss` formula is verified by test.
8. `classifier` and `audio_proj` are independent and do not share parameters.
9. `text_prototypes_raw` is frozen and not a trainable parameter.
10. The project interpreter check must pass before running smoke tests:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -c "import sys; print(sys.executable)"
```
