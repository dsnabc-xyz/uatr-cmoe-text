from pathlib import Path
from collections import defaultdict, Counter
import re


# ==========================
# 改这里
# ==========================
RAW_ROOT = Path(r"E:\dataset\deepship")
# ==========================


# DEMONet Table 10 DeepShip test IDs
# 规则：这些 ID 是 test，其余 Else 是 train
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

LABELS = ["Cargo", "Passenger", "Tanker", "Tug"]


def norm(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_")


def infer_class_from_path(path: Path) -> str:
    """
    从路径中识别类别。

    适配：
    E:\\dataset\\deepship\\Cargo-...\\Cargo\\20171115d-20\\20.wav
    E:\\dataset\\deepship\\Passengership-...\\Passengership\\20170202-54\\183738.wav
    """
    parts = [norm(p) for p in path.parts]
    full = norm(str(path))

    # 优先精确匹配路径层级中的类别目录
    for p in parts:
        if p == "cargo":
            return "Cargo"
        if p in {"passenger", "passengership", "passenger_ship", "passengers"}:
            return "Passenger"
        if p in {"tanker", "oil_tanker", "oiltanker", "oil"}:
            return "Tanker"
        if p in {"tug", "tugboat", "tug_boat"}:
            return "Tug"

    # 再宽松匹配 zip 解压文件夹名
    if "cargo" in full:
        return "Cargo"
    if "passenger" in full:
        return "Passenger"
    if "tanker" in full or "oil_tanker" in full or "oiltanker" in full:
        return "Tanker"
    if "tug" in full:
        return "Tug"

    return "Unknown"


def extract_parent_id(path: Path) -> int:
    """
    按父目录最后一个数字作为 DEMONet Table 10 ID。

    例：
    ...\\Cargo\\20171115d-20\\20.wav -> 20
    ...\\Passengership\\20170202-54\\183738.wav -> 54
    """
    parent = path.parent.name.strip()
    nums = re.findall(r"\d+", parent)
    if not nums:
        raise ValueError(f"Cannot extract parent_id from folder: {path.parent}")
    return int(nums[-1])


def extract_parent_date(path: Path) -> str | None:
    """
    尝试从父目录提取日期，如：
    20170202-54 -> 20170202
    20171115d-20 -> 20171115
    """
    parent = path.parent.name.strip()
    m = re.search(r"(20\d{6})", parent)
    return m.group(1) if m else None


def main():
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"RAW_ROOT does not exist: {RAW_ROOT}")

    wavs = sorted(RAW_ROOT.rglob("*.wav"))
    print(f"RAW_ROOT: {RAW_ROOT}")
    print(f"Total wav files found: {len(wavs)}")

    if not wavs:
        raise RuntimeError("No wav files found. Please unzip all DeepShip zip files first.")

    # class -> parent_id -> list[wav_path]
    by_class_id = {cls: defaultdict(list) for cls in LABELS}
    unknown = []

    for wav_path in wavs:
        cls = infer_class_from_path(wav_path)
        if cls == "Unknown":
            unknown.append(wav_path)
            continue

        pid = extract_parent_id(wav_path)
        by_class_id[cls][pid].append(wav_path)

    print("\n================ Class / ID Summary ================")

    all_ok = True

    for cls in LABELS:
        id_to_paths = by_class_id[cls]
        ids_found = set(id_to_paths.keys())
        expected_test = TEST_IDS[cls]

        missing = sorted(expected_test - ids_found)
        hit = sorted(expected_test & ids_found)
        extra = sorted(ids_found - expected_test)

        print(f"\n[{cls}]")
        print(f"  unique parent IDs found: {len(ids_found)}")
        if ids_found:
            print(f"  ID range: {min(ids_found)} - {max(ids_found)}")
        print(f"  DEMONet Table 10 test IDs: {len(expected_test)}")
        print(f"  hit test IDs: {len(hit)} / {len(expected_test)}")
        print(f"  missing test IDs: {missing if missing else 'None'}")

        # 统计每个 ID 对应几个 wav
        wav_count_per_id = {pid: len(paths) for pid, paths in id_to_paths.items()}
        multi_wav_ids = {pid: n for pid, n in wav_count_per_id.items() if n > 1}

        print(f"  IDs with multiple wav files: {len(multi_wav_ids)}")
        if multi_wav_ids:
            examples = list(sorted(multi_wav_ids.items()))[:10]
            print(f"  first multi-wav ID examples: {examples}")

        # 打印命中的 test ID 对应的样例路径
        print("  first hit examples:")
        for pid in hit[:10]:
            p = id_to_paths[pid][0]
            print(f"    ID={pid:3d} -> {p}")

        # 打印缺失的 ID
        if missing:
            all_ok = False
            print("  [ERROR] Missing DEMONet test IDs in local files:")
            for pid in missing:
                print(f"    missing {cls} ID={pid}")

        # 检查同一 ID 是否有多个父目录
        parent_dirs_per_id = {}
        for pid, paths in id_to_paths.items():
            parent_dirs = sorted({str(p.parent) for p in paths})
            parent_dirs_per_id[pid] = parent_dirs

        duplicate_parent_id = {
            pid: dirs for pid, dirs in parent_dirs_per_id.items()
            if len(dirs) > 1
        }

        if duplicate_parent_id:
            print(f"  [WARNING] IDs appearing in multiple parent folders: {len(duplicate_parent_id)}")
            for pid, dirs in list(sorted(duplicate_parent_id.items()))[:10]:
                print(f"    ID={pid} appears in:")
                for d in dirs[:5]:
                    print(f"      {d}")

    print("\n================ Unknown class files ================")
    print(f"Unknown wav files: {len(unknown)}")
    for p in unknown[:20]:
        print(f"  {p}")

    print("\n================ Train/Test Prediction ================")
    for cls in LABELS:
        id_to_paths = by_class_id[cls]
        train_ids = sorted(set(id_to_paths.keys()) - TEST_IDS[cls])
        test_ids = sorted(set(id_to_paths.keys()) & TEST_IDS[cls])

        train_wavs = sum(len(id_to_paths[i]) for i in train_ids)
        test_wavs = sum(len(id_to_paths[i]) for i in test_ids)

        print(f"{cls:10s} | train IDs={len(train_ids):4d}, test IDs={len(test_ids):4d}, "
              f"train wavs={train_wavs:5d}, test wavs={test_wavs:5d}")

    print("\n================ Final Decision ================")
    if all_ok and not unknown:
        print("[OK] 本地 DeepShip 文件可以按 parent_id 匹配 DEMONet Table 10。")
        print("     后续可使用 parent folder 最后一个数字作为 Table 10 ID 来生成 30s/15s split。")
    elif all_ok and unknown:
        print("[PARTIAL OK] Table 10 test IDs 都能找到，但存在无法识别类别的 wav。")
        print("             请先处理 Unknown class files，再生成 split。")
    else:
        print("[NOT SAFE] 至少有一部分 DEMONet Table 10 test IDs 在本地文件中找不到。")
        print("           不建议直接生成 DEMONet split。请先检查是否所有 zip 已解压、类别名是否识别正确、或 ID 规则是否需要调整。")


if __name__ == "__main__":
    main()