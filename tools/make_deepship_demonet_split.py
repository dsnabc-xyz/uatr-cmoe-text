import argparse
import csv
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path


TEST_IDS = {
    "Cargo": {
        1, 2, 4, 5, 18, 30, 32, 35, 40, 48, 56, 62,
        63, 67, 68, 72, 74, 79, 83, 91, 92, 93, 95, 97,
        100, 104,
    },
    "Passenger": {
        2, 4, 5, 7, 11, 15, 17, 19, 20, 23, 30, 31,
        37, 45, 46, 52, 53, 60, 61, 62, 64, 67, 68, 70,
        75, 76, 77, 84, 86, 91, 101, 106, 113, 117, 122,
        125, 129, 130, 134, 135, 142, 144, 152, 157,
        159, 161, 167, 168, 177, 179, 187, 188, 189,
    },
    "Tanker": {
        2, 3, 4, 7, 8, 13, 14, 15, 19, 22, 25, 28,
        35, 37, 46, 58, 62, 71, 73, 79, 82, 84, 88, 89,
        92, 99, 106, 115, 118, 124, 126, 127, 131, 134,
        141, 144, 147, 151, 153, 156, 158, 167, 171,
        178, 179, 185, 186, 190, 192, 193, 201, 205,
        213, 217, 228, 233,
    },
    "Tug": {
        7, 8, 18, 20, 24, 25, 27, 29, 32, 33, 37, 39,
        40, 44, 45, 56, 59, 70,
    },
}

CLASS_TO_LABEL = {
    "Cargo": ("cargo", 0),
    "Tanker": ("tanker", 1),
    "Passenger": ("passenger", 2),
    "Tug": ("tug", 3),
}
LABEL_NAMES = ["cargo", "tanker", "passenger", "tug"]
METADATA_FIELDS = [
    "split",
    "class_name",
    "label_id",
    "parent_id",
    "parent_folder",
    "original_path",
    "clip_path",
    "start_sec",
    "end_sec",
    "duration_sec",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build DEMONet Table 10 DeepShip train/test split by parent folder ID."
    )
    parser.add_argument("--raw_root", default="/root/autodl-tmp/DeepShip")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--clip_seconds", type=float, default=30.0)
    parser.add_argument("--stride_seconds", type=float, default=None)
    parser.add_argument("--mode", choices=["list_only", "cut"], default="list_only")
    parser.add_argument("--min_clip_ratio", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def norm_text(value):
    return str(value).lower().replace("-", "_").replace(" ", "_")


def infer_class_from_path(path):
    parts = [norm_text(part) for part in path.parts]
    full = norm_text(str(path))
    for part in parts:
        if part == "cargo":
            return "Cargo"
        if part in {"passenger", "passengership", "passenger_ship", "passengers"}:
            return "Passenger"
        if part in {"tanker", "oil_tanker", "oiltanker", "oil"}:
            return "Tanker"
        if part in {"tug", "tugboat", "tug_boat"}:
            return "Tug"
    if "cargo" in full:
        return "Cargo"
    if "passenger" in full:
        return "Passenger"
    if "tanker" in full or "oil_tanker" in full or "oiltanker" in full:
        return "Tanker"
    if "tug" in full:
        return "Tug"
    return "Unknown"


def extract_parent_id(path):
    parent = path.parent.name.strip()
    numbers = re.findall(r"\d+", parent)
    if not numbers:
        raise ValueError(f"Cannot extract parent_id from folder: {path.parent}")
    return int(numbers[-1])


def audio_info(path):
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return int(info.frames), int(info.samplerate)
    except Exception:
        pass
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        return int(info.num_frames), int(info.sample_rate)
    except Exception:
        pass
    with wave.open(str(path), "rb") as handle:
        return int(handle.getnframes()), int(handle.getframerate())


def read_audio_segment(path, start_frame, num_frames):
    try:
        import soundfile as sf
        data, sample_rate = sf.read(
            str(path),
            start=start_frame,
            frames=num_frames,
            always_2d=False,
        )
        return "soundfile", data, int(sample_rate)
    except Exception:
        pass
    try:
        import torchaudio
        data, sample_rate = torchaudio.load(
            str(path),
            frame_offset=start_frame,
            num_frames=num_frames,
        )
        return "torchaudio", data, int(sample_rate)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio segment from {path}") from exc


def write_audio(path, backend, data, sample_rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "soundfile":
        import soundfile as sf
        sf.write(str(path), data, sample_rate)
        return
    if backend == "torchaudio":
        import torchaudio
        torchaudio.save(str(path), data, sample_rate)
        return
    raise RuntimeError(f"Unsupported audio backend: {backend}")


def discover_wavs(raw_root):
    wav_paths = sorted(raw_root.rglob("*.wav"))
    if not wav_paths:
        raise RuntimeError(f"No wav files found under raw_root: {raw_root}")

    records = []
    unknown = []
    for wav_path in wav_paths:
        class_key = infer_class_from_path(wav_path)
        if class_key == "Unknown":
            unknown.append(wav_path)
            continue
        parent_id = extract_parent_id(wav_path)
        class_name, label_id = CLASS_TO_LABEL[class_key]
        records.append({
            "class_key": class_key,
            "class_name": class_name,
            "label_id": label_id,
            "parent_id": parent_id,
            "parent_folder": wav_path.parent.name,
            "original_path": wav_path,
        })

    if unknown:
        examples = "\n".join(f"  {path}" for path in unknown[:20])
        raise RuntimeError(f"Unknown class wav files: {len(unknown)}\n{examples}")
    return records


def assign_splits(records):
    ids_by_class = {class_key: set() for class_key in TEST_IDS}
    for record in records:
        ids_by_class[record["class_key"]].add(record["parent_id"])

    missing_messages = []
    for class_key, expected_ids in TEST_IDS.items():
        missing = sorted(expected_ids - ids_by_class[class_key])
        if missing:
            missing_messages.append(f"{class_key}: {missing}")
    if missing_messages:
        raise RuntimeError(
            "Missing DEMONet Table 10 test IDs:\n" + "\n".join(missing_messages)
        )

    for record in records:
        test_ids = TEST_IDS[record["class_key"]]
        record["split"] = "test" if record["parent_id"] in test_ids else "train"
    return records


def validate_split(records):
    by_original = defaultdict(set)
    by_class_parent = defaultdict(set)
    bad_labels = []
    for record in records:
        by_original[str(record["original_path"])].add(record["split"])
        by_class_parent[(record["class_name"], record["parent_id"])].add(record["split"])
        if record["label_id"] not in {0, 1, 2, 3}:
            bad_labels.append(record)

    overlap_original = [path for path, splits in by_original.items() if len(splits) > 1]
    if overlap_original:
        raise RuntimeError(f"original_path appears in both train and test: {overlap_original[:10]}")

    overlap_parent = [key for key, splits in by_class_parent.items() if len(splits) > 1]
    if overlap_parent:
        raise RuntimeError(f"class + parent_id appears in both train and test: {overlap_parent[:10]}")

    if bad_labels:
        raise RuntimeError(f"Invalid label_id records found: {bad_labels[:10]}")


def ensure_output_root(out_root, overwrite):
    if out_root.exists() and any(out_root.iterdir()) and not overwrite:
        raise FileExistsError(f"out_root is not empty: {out_root}. Use --overwrite to continue.")
    out_root.mkdir(parents=True, exist_ok=True)


def iter_clip_windows(num_frames, sample_rate, clip_seconds, stride_seconds, min_clip_ratio):
    clip_frames = int(round(clip_seconds * sample_rate))
    stride_frames = int(round(stride_seconds * sample_rate))
    min_frames = int(round(min_clip_ratio * clip_frames))
    if clip_frames <= 0 or stride_frames <= 0:
        raise ValueError("clip_seconds and stride_seconds must be positive.")

    start = 0
    while start < num_frames:
        remaining = num_frames - start
        if remaining < min_frames:
            break
        frames = min(clip_frames, remaining)
        yield start, frames
        start += stride_frames


def write_label_file(out_root):
    label_path = out_root / "label.txt"
    label_path.write_text("\n".join(LABEL_NAMES) + "\n", encoding="utf-8")


def write_lists_and_metadata(records, args):
    out_root = Path(args.out_root)
    train_lines = []
    test_lines = []
    metadata_rows = []
    clip_counts = Counter()
    segment_index = defaultdict(int)

    for record in records:
        original_path = record["original_path"]
        frames, sample_rate = audio_info(original_path)
        duration_sec = frames / sample_rate if sample_rate > 0 else 0.0

        if args.mode == "list_only":
            clip_path = original_path
            line = f"{clip_path}\t{record['label_id']}"
            if record["split"] == "train":
                train_lines.append(line)
            else:
                test_lines.append(line)
            metadata_rows.append({
                "split": record["split"],
                "class_name": record["class_name"],
                "label_id": record["label_id"],
                "parent_id": record["parent_id"],
                "parent_folder": record["parent_folder"],
                "original_path": str(original_path),
                "clip_path": str(clip_path),
                "start_sec": "0.000000",
                "end_sec": f"{duration_sec:.6f}",
                "duration_sec": f"{duration_sec:.6f}",
            })
            clip_counts[(record["split"], record["class_name"])] += 1
            continue

        for start_frame, num_frames in iter_clip_windows(
                frames,
                sample_rate,
                args.clip_seconds,
                args.stride_seconds,
                args.min_clip_ratio):
            key = (record["split"], record["class_name"], record["parent_id"])
            seg_idx = segment_index[key]
            segment_index[key] += 1

            clip_path = (
                out_root / "audio" / record["split"] / record["class_name"] /
                f"{record['parent_id']}_{seg_idx:06d}.wav"
            )
            if clip_path.exists() and not args.overwrite:
                raise FileExistsError(f"Clip already exists: {clip_path}. Use --overwrite.")

            backend, audio, read_sample_rate = read_audio_segment(original_path, start_frame, num_frames)
            if read_sample_rate != sample_rate:
                raise RuntimeError(
                    f"Sample rate changed while reading {original_path}: {sample_rate} -> {read_sample_rate}"
                )
            write_audio(clip_path, backend, audio, sample_rate)

            start_sec = start_frame / sample_rate
            end_sec = (start_frame + num_frames) / sample_rate
            line = f"{clip_path}\t{record['label_id']}"
            if record["split"] == "train":
                train_lines.append(line)
            else:
                test_lines.append(line)
            metadata_rows.append({
                "split": record["split"],
                "class_name": record["class_name"],
                "label_id": record["label_id"],
                "parent_id": record["parent_id"],
                "parent_folder": record["parent_folder"],
                "original_path": str(original_path),
                "clip_path": str(clip_path),
                "start_sec": f"{start_sec:.6f}",
                "end_sec": f"{end_sec:.6f}",
                "duration_sec": f"{end_sec - start_sec:.6f}",
            })
            clip_counts[(record["split"], record["class_name"])] += 1

    (out_root / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (out_root / "test.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")
    with (out_root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(metadata_rows)

    return clip_counts


def print_summary(records, clip_counts):
    parent_ids = defaultdict(set)
    wav_counts = Counter()
    for record in records:
        key = (record["split"], record["class_name"])
        parent_ids[key].add(record["parent_id"])
        wav_counts[key] += 1

    print("\n================ Split Summary ================")
    for class_name in LABEL_NAMES:
        for split in ["train", "test"]:
            key = (split, class_name)
            print(
                f"{class_name:10s} {split:5s} | "
                f"parent IDs={len(parent_ids[key]):4d}, "
                f"wavs={wav_counts[key]:5d}, "
                f"clips={clip_counts[key]:6d}"
            )


def main():
    args = parse_args()
    if args.stride_seconds is None:
        args.stride_seconds = args.clip_seconds
    if not 0 < args.min_clip_ratio <= 1:
        raise ValueError("--min_clip_ratio must be in (0, 1].")

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root does not exist: {raw_root}")

    ensure_output_root(out_root, args.overwrite)
    records = assign_splits(discover_wavs(raw_root))
    validate_split(records)
    write_label_file(out_root)
    clip_counts = write_lists_and_metadata(records, args)
    print_summary(records, clip_counts)
    print(f"\n[OK] Wrote split to: {out_root}")
    print(f"Mode: {args.mode}")
    print("Label order: cargo, tanker, passenger, tug")


if __name__ == "__main__":
    main()
