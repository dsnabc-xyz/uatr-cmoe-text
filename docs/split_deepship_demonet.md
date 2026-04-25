# DeepShip DEMONet Table 10 Split

This document records the split policy used for DeepShip 4-class ship-radiated noise recognition.

## Principle

The split follows DEMONet Table 10 test IDs.

Do not slice first and randomly split clips. The correct order is:

1. Discover original wav files.
2. Infer class.
3. Extract `parent_id`.
4. Split by `class + parent_id`.
5. Slice audio only after train/test split is fixed.

This prevents clips from the same original vessel recording group from leaking across train and test.

## parent_id Rule

`parent_id` is the last number in the parent directory name.

Examples:

```text
20171115d-20 -> 20
20170202-54  -> 54
```

The server-side check script has verified that this rule matches the DEMONet Table 10 IDs for the local DeepShip wav layout.

## Test IDs

The IDs listed in DEMONet Table 10 enter `test`.

All other IDs enter `train`.

The split is class-aware, so the key is:

```text
class_name + parent_id
```

## Label Order

The label order is fixed:

```text
cargo
tanker
passenger
tug
```

Label mapping:

```text
cargo     -> 0
tanker    -> 1
passenger -> 2
tug       -> 3
```

`label.txt` must contain exactly:

```text
cargo
tanker
passenger
tug
```

## Clip Policy

The paper-aligned comparison split should use:

```text
clip_seconds = 30
overlap_ratio = 0.5
```

This is 30 seconds with 15 seconds overlap.

The split generator also supports 15 second non-overlap clips for faster debug or smaller experiments.

Tail clips shorter than `min_clip_ratio * clip_seconds` are dropped.

## Script

Use:

```text
tools/make_deepship_demonet_split.py
```

The script writes:

```text
out_root/
  train.txt
  test.txt
  label.txt
  metadata.csv
```

In `cut` mode, clips are written under:

```text
out_root/audio/train/<class_name>/<parent_id>_<seg_idx>.wav
out_root/audio/test/<class_name>/<parent_id>_<seg_idx>.wav
```

## Example Commands

15 second non-overlap split using defaults:

```bash
python tools/make_deepship_demonet_split.py --overwrite
```

30 seconds with 15 seconds overlap:

```bash
python tools/make_deepship_demonet_split.py \
  --raw_root /root/autodl-tmp/DeepShip \
  --out_root /root/autodl-tmp/deepship_demonet_30s_overlap50 \
  --clip_seconds 30 \
  --overlap_ratio 0.5 \
  --mode cut \
  --min_clip_ratio 0.95 \
  --overwrite
```

List-only mode for checking paths without cutting audio:

```bash
python tools/make_deepship_demonet_split.py \
  --raw_root /root/autodl-tmp/DeepShip \
  --out_root /root/autodl-tmp/deepship_demonet_list_only \
  --mode list_only \
  --overwrite
```

## Safety Checks

The split script must fail if:

- any class is unknown
- any DEMONet Table 10 test ID is missing
- one original path appears in both train and test
- one `class + parent_id` appears in both train and test
- any label ID is outside `0,1,2,3`
