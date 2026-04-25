from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


# ====== 你只需要改这里 ======
SPLIT_DIR = Path(r"E:\dataset\shipsEar\shipsear_split_9class")

TRAIN_LIST_IN = SPLIT_DIR / "train.txt"
TEST_LIST_IN = SPLIT_DIR / "test.txt"
LABEL_LIST_IN = SPLIT_DIR / "label.txt"

OUT_ROOT = Path(r"E:\dataset\shipsEar\shipsear_split_9class_clips_30s")

TARGET_SR = 52734
SEG_SEC = 30.0
HOP_SEC = 15.0       # 5s 无重叠；如果想 30s/15s overlap，则 SEG_SEC=30, HOP_SEC=15
PAD_LAST = False    # True 表示最后不足一段也补零保留；False 表示丢掉不足一段
# ===========================


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def resample_if_needed(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    audio = resample_poly(audio, up, down)
    return audio.astype(np.float32)


def read_list(list_path: Path):
    items = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 兼容 tab 或空格分隔
            if "\t" in line:
                wav_path, label = line.split("\t")
            else:
                parts = line.split()
                wav_path, label = " ".join(parts[:-1]), parts[-1]

            items.append((Path(wav_path), int(label)))
    return items


def cut_one_file(wav_path: Path, label: int, subset: str, global_index: int):
    audio, sr = sf.read(str(wav_path), dtype="float32")
    audio = ensure_mono(audio)
    audio = resample_if_needed(audio, sr, TARGET_SR)

    seg_len = int(SEG_SEC * TARGET_SR)
    hop_len = int(HOP_SEC * TARGET_SR)

    out_audio_dir = OUT_ROOT / subset / f"class_{label}"
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    n = len(audio)
    start = 0
    clip_idx = 0

    while start < n:
        end = start + seg_len

        if end <= n:
            clip = audio[start:end]
        else:
            if not PAD_LAST:
                break
            clip = np.zeros(seg_len, dtype=np.float32)
            remain = audio[start:n]
            clip[:len(remain)] = remain

        out_name = f"{wav_path.stem}_clip{clip_idx:04d}_{int(SEG_SEC)}s.wav"
        out_path = out_audio_dir / out_name

        sf.write(str(out_path), clip, TARGET_SR)
        rows.append(f"{out_path}\t{label}\n")

        clip_idx += 1
        start += hop_len

    print(f"[{subset}] {wav_path.name} -> {clip_idx} clips")
    return rows


def cut_split(list_in: Path, subset: str):
    items = read_list(list_in)
    all_rows = []

    for i, (wav_path, label) in enumerate(items):
        if not wav_path.exists():
            print(f"[MISSING] {wav_path}")
            continue

        rows = cut_one_file(wav_path, label, subset, i)
        all_rows.extend(rows)

    out_list = OUT_ROOT / f"{subset}.txt"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with open(out_list, "w", encoding="utf-8") as f:
        f.writelines(all_rows)

    print(f"\n[OK] write {out_list}")
    print(f"     total clips = {len(all_rows)}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 复制 label.txt
    label_out = OUT_ROOT / "label.txt"
    label_out.write_text(LABEL_LIST_IN.read_text(encoding="utf-8"), encoding="utf-8")

    cut_split(TRAIN_LIST_IN, "train")
    cut_split(TEST_LIST_IN, "test")

    print("\nDone.")
    print("Output root:", OUT_ROOT)


if __name__ == "__main__":
    main()