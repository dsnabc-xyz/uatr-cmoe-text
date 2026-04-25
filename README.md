# UATR-CMoE Text for DeepShip

This repository adapts an AudioClassification-Pytorch style framework for 4-class underwater ship-radiated noise recognition on DeepShip.

Target classes are fixed:

```text
cargo
tanker
passenger
tug
```

The main model is `UATRCMoETextResNet18`.

## Model

The model keeps inference audio-only and uses frozen text prototypes only as a training-side class-prototype alignment target.

```text
audio
  -> frontend feature, e.g. MelSpectrogram
  -> ResNet18 backbone
  -> cMoE router / experts
  -> mixed_feat
       |-- classifier -> logits -> ce_loss
       `-- audio_proj -> audio_z -> proto_logits -> text_proto_loss
```

Important constraints:

- The router is audio-only.
- Experts are audio-only.
- Text prototypes do not enter the router, experts, or classifier.
- `classifier` and `audio_proj` are independent modules.
- `text_proto_loss` must not be called ITC.
- Raw SBERT text prototypes are frozen buffers, not trainable parameters.

The text-prototype loss is:

```python
audio_z = normalize(audio_proj(mixed_feat))
text_z = normalize(text_proj(text_prototypes_raw))
proto_logits = audio_z @ text_z.T / proto_temperature
text_proto_loss = cross_entropy(proto_logits, labels)
```

Total loss:

```text
total_loss = ce_loss + balance_weight * balance_loss + text_proto_weight * text_proto_loss
```

## Important Files

```text
macls/models/uatr_cmoe_resnet18.py
  UATRCMoETextResNet18 model.

macls/models/cmoe_text_common.py
  Text prototype loading, class-name normalization, projection helpers, and UATR-CMoE balance helpers.

macls/trainer.py
  Minimal dict-output trainer integration for UATRCMoETextResNet18.

assets/text_prototypes/ship_type_sbert_384_original.pt
  Frozen class text prototypes, expected shape [4, 384].

tools/smoke_test_uatr_cmoe_text_resnet18.py
  Model-level smoke test.

tools/smoke_train_one_batch_uatr_cmoe_text_resnet18.py
  One-batch forward/backward/optimizer smoke test with a mock batch.

tools/make_deepship_demonet_split.py
  DEMONet Table 10 DeepShip train/test split generator.

configs/deepship/uatr_cmoe_text_resnet18.yml
  DeepShip training configuration.
```

## Smoke Tests

Use the project Python interpreter on Windows:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -c "import sys; print(sys.executable)"
```

Compile checks:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -m compileall macls tools
```

Model-level smoke:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -u "e:\Projects\uatr-cmoe-text\tools\smoke_test_uatr_cmoe_text_resnet18.py"
```

One-batch trainer integration smoke:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -u "e:\Projects\uatr-cmoe-text\tools\smoke_train_one_batch_uatr_cmoe_text_resnet18.py"
```

## DeepShip DEMONet Split

Generate the DEMONet Table 10 split on the server after DeepShip wav files are available:

```bash
python tools/make_deepship_demonet_split.py \
  --raw_root /root/autodl-tmp/DeepShip \
  --out_root /root/autodl-tmp/deepship_demonet_30s_overlap50 \
  --clip_seconds 30 \
  --overlap_ratio 0.5 \
  --mode cut \
  --overwrite
```

For a 15 second non-overlap split using script defaults:

```bash
python tools/make_deepship_demonet_split.py --overwrite
```

The split must be made by `class + parent_id` before slicing audio clips.

## Training

After generating split files and updating the config paths if needed:

```powershell
& "D:\study\VirtualEnvironment\uatr_clean\python.exe" -u train.py --configs configs/deepship/uatr_cmoe_text_resnet18.yml --use_gpu True
```

For server runs, use the environment Python selected on the server and the same config:

```bash
python train.py --configs configs/deepship/uatr_cmoe_text_resnet18.yml --use_gpu True
```

Do not run training before the model-level smoke and one-batch trainer smoke pass.
