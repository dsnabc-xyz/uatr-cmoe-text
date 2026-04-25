from pathlib import Path

# 你的 ShipsEar 原始 wav 文件夹
AUDIO_ROOT = Path(r"E:\dataset\shipsEar\shipsEar_AUDIOS")

# 输出 train.txt / test.txt / label.txt 的目录
OUT_DIR = Path(r"E:\dataset\shipsEar\shipsear_split_9class")
OUT_DIR.mkdir(parents=True, exist_ok=True)

labels = [
    "Dredger",
    "Fishboat",
    "Motorboat",
    "Musselboat",
    "Naturalambientnoise",
    "Oceanliner",
    "Passengers",
    "RORO",
    "Sailboat",
]

label2id = {name: i for i, name in enumerate(labels)}

train_split = {
    "Dredger": [80, 93, 94, 96],
    "Fishboat": [73, 74, 76],
    "Motorboat": [21, 26, 33, 39, 45, 51, 52, 70, 77, 79],
    "Musselboat": [46, 47, 49, 66],
    "Naturalambientnoise": [81, 82, 84, 85, 86, 88, 89, 90, 91],
    "Oceanliner": [16, 22, 23, 25, 69],
    "Passengers": [6, 7, 8, 10, 11, 12, 14, 17, 32, 34, 36, 38, 40, 41, 43, 53, 54, 59, 60, 61, 63, 64, 67],
    "RORO": [18, 19, 58],
    "Sailboat": [37, 56, 68],
}

test_split = {
    "Dredger": [95],
    "Fishboat": [75],
    "Motorboat": [27, 50, 72],
    "Musselboat": [48],
    "Naturalambientnoise": [83, 87, 92],
    "Oceanliner": [24, 71],
    "Passengers": [9, 13, 35, 42, 55, 62, 65],
    "RORO": [20, 78],
    "Sailboat": [57],
}


def find_wav_by_id(wav_id: int):
    """
    适配你当前的 ShipsEar 文件名格式，例如：
    80_04_10_12_adricristuy.wav
    93_A_Draga_1.wav
    06_10_07_13_marDeCangas_Entra.wav
    """

    patterns = [
        f"{wav_id}_*.wav",       # 80_...
        f"{wav_id:02d}_*.wav",   # 06_...
        f"{wav_id} *.wav",       # 93 A Draga 1.wav
        f"{wav_id:02d} *.wav",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(list(AUDIO_ROOT.glob(pattern)))

    # 去重
    matches = list(dict.fromkeys(matches))

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        print(f"[WARNING] ID {wav_id} matched multiple files:")
        for m in matches:
            print("   ", m)
        print("Use first one:", matches[0])

    return matches[0]


def write_split(split_dict, out_file):
    rows = []
    missing = []

    for label, ids in split_dict.items():
        for wav_id in ids:
            wav_path = find_wav_by_id(wav_id)

            if wav_path is None:
                missing.append((label, wav_id))
                continue

            rows.append(f"{wav_path}\t{label2id[label]}\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(rows)

    print(f"[OK] {out_file}")
    print(f"     num = {len(rows)}")

    if missing:
        print(f"[WARNING] missing {len(missing)} files:")
        for label, wav_id in missing:
            print(f"  label={label}, id={wav_id}")


write_split(train_split, OUT_DIR / "train.txt")
write_split(test_split, OUT_DIR / "test.txt")

with open(OUT_DIR / "label.txt", "w", encoding="utf-8") as f:
    for name in labels:
        f.write(name + "\n")

print("Done.")
print("Output dir:", OUT_DIR)