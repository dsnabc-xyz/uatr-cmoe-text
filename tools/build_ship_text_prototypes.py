import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLASS_NAMES = ["cargo", "tanker", "passenger", "tug"]
TEMPLATE_SET_TO_SUFFIX = {
    "original": "original",
    "contrastive": "contrastive",
}

SHIP_TEXT_TEMPLATES = {
    "cargo": (
        "Cargo ship underwater radiated noise. "
        "Usually produced by a large merchant vessel with a low-speed propeller. "
        "Often contains stable low-frequency tonal and continuous components below about 100 Hz. "
        "May show blade-rate harmonics and diesel machinery components around low and mid-low frequencies. "
        "The acoustic pattern is relatively steady over time."
    ),
    "tanker": (
        "Tanker underwater radiated noise. "
        "Usually produced by a large heavily loaded vessel with slow shaft speed and a large propeller. "
        "Often has very strong low-frequency tonal components and low blade-rate harmonics. "
        "The spectrum may concentrate more energy in the very-low-frequency region than other merchant ships. "
        "Cavitation and broadband components may appear when operating under load."
    ),
    "passenger": (
        "Passenger ship underwater radiated noise. "
        "Usually produced by a vessel with relatively higher shaft speed, auxiliary machinery, and sometimes multiple propellers. "
        "May contain clearer tonal components from auxiliary machinery and propeller rotation. "
        "Compared with large cargo or tanker ships, it can show more broadband energy and higher modulation frequencies. "
        "The acoustic pattern is often more regular than a tugboat."
    ),
    "tug": (
        "Tugboat underwater radiated noise. "
        "Usually produced by a small hull with a high-power diesel engine and strong propulsion load. "
        "Often contains strong engine harmonics, broadband components, and irregular time-varying spectral patterns. "
        "Variable-speed operation can cause changing modulation and transient noise. "
        "Compared with passenger ships, the acoustic pattern may be less steady and more dynamic."
    ),
}

SHIP_TEXT_TEMPLATES_CONTRASTIVE = {
    "cargo": (
        "Cargo ship underwater radiated noise. "
        "Compared with tugboats, the propeller and engine pattern is usually slower and more stable. "
        "Compared with passenger ships, cargo ships often have less high-speed auxiliary machinery activity. "
        "Compared with tankers, cargo ships may have slightly higher blade-rate harmonics and less extreme very-low-frequency dominance. "
        "The key acoustic cues are stable low-frequency tonals, slow propeller modulation, and steady diesel machinery noise."
    ),
    "tanker": (
        "Tanker underwater radiated noise. "
        "Compared with cargo ships, tankers often have slower shaft speed, heavier loading, and stronger very-low-frequency energy. "
        "Compared with passenger ships and tugboats, tankers usually show lower blade-rate harmonics and less rapid time variation. "
        "The key acoustic cues are very-low-frequency tonal components, slow propeller modulation, heavy-hull loading, and low shaft-speed machinery noise."
    ),
    "passenger": (
        "Passenger ship underwater radiated noise. "
        "Compared with cargo ships and tankers, passenger ships often have higher shaft speed and more auxiliary machinery components. "
        "Compared with tugboats, passenger ships usually have a more regular and stable operating pattern. "
        "The key acoustic cues are mid-frequency tonal components, higher propeller modulation than large merchant ships, auxiliary machinery noise, and relatively regular broadband energy."
    ),
    "tug": (
        "Tugboat underwater radiated noise. "
        "Compared with cargo ships and tankers, tugboats have a smaller hull, stronger propulsion load, and more variable engine operation. "
        "Compared with passenger ships, tugboats often show more irregular transients and stronger time-varying broadband noise. "
        "The key acoustic cues are high-power diesel harmonics, rapid propeller modulation, variable-speed operation, broadband bursts, and unstable spectral patterns."
    ),
}

TEMPLATE_SETS = {
    "original": SHIP_TEXT_TEMPLATES,
    "contrastive": SHIP_TEXT_TEMPLATES_CONTRASTIVE,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SBERT text prototypes for four ship types."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"SentenceTransformer model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--template_set",
        choices=sorted(TEMPLATE_SETS.keys()),
        default="original",
        help="Template set to encode.",
    )
    return parser.parse_args()


def format_similarity_matrix(similarity, class_names):
    header = [" " * 12] + [f"{name:>12s}" for name in class_names]
    lines = ["".join(header)]
    for idx, row in enumerate(similarity.tolist()):
        values = "".join(f"{value:12.4f}" for value in row)
        lines.append(f"{class_names[idx]:>12s}{values}")
    return "\n".join(lines)


def main():
    args = parse_args()
    template_set = TEMPLATE_SETS[args.template_set]
    output = Path(
        f"assets/text_prototypes/ship_type_sbert_384_{TEMPLATE_SET_TO_SUFFIX[args.template_set]}.pt"
    )
    descriptions = [template_set[name] for name in CLASS_NAMES]
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. "
            "Please install it before running this script."
        ) from exc

    model = SentenceTransformer(args.model_name)
    embeddings = model.encode(
        descriptions,
        convert_to_tensor=True,
        normalize_embeddings=False,
    ).to(dtype=torch.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(CLASS_NAMES):
        raise RuntimeError(f"Unexpected embedding shape: {tuple(embeddings.shape)}")

    payload = {
        "class_names": CLASS_NAMES,
        "descriptions": descriptions,
        "embeddings": embeddings,
        "model_name": args.model_name,
    }
    torch.save(payload, output)

    similarity = F.normalize(embeddings, dim=1) @ F.normalize(embeddings, dim=1).T
    off_diagonal = similarity[~torch.eye(len(CLASS_NAMES), dtype=torch.bool)]
    off_diagonal_mean = off_diagonal.mean().item()

    cargo_idx = CLASS_NAMES.index("cargo")
    tanker_idx = CLASS_NAMES.index("tanker")
    passenger_idx = CLASS_NAMES.index("passenger")
    tug_idx = CLASS_NAMES.index("tug")

    cargo_tanker = similarity[cargo_idx, tanker_idx].item()
    cargo_passenger = similarity[cargo_idx, passenger_idx].item()
    passenger_tug = similarity[passenger_idx, tug_idx].item()

    print(f"[OK] saved to {output}")
    print(f"sys.executable: {sys.executable}")
    print(f"sentence_transformers.__version__: {sentence_transformers.__version__}")
    print(f"embeddings.shape: {tuple(embeddings.shape)}")
    print("cosine similarity matrix:")
    print(format_similarity_matrix(similarity, CLASS_NAMES))
    print(f"off_diagonal_mean: {off_diagonal_mean:.4f}")
    print(f"cargo_tanker: {cargo_tanker:.4f}")
    print(f"cargo_passenger: {cargo_passenger:.4f}")
    print(f"passenger_tug: {passenger_tug:.4f}")

    if cargo_passenger > cargo_tanker:
        print("[WARN] cargo-passenger similarity is higher than cargo-tanker; text space does not match expected physical prior.")

    if off_diagonal_mean > 0.85:
        print("[WARN] class text prototypes are still too similar.")


if __name__ == "__main__":
    main()
