import argparse

import torch
from loguru import logger

from macls.data_utils.featurizer import AudioFeaturizer


PRESETS = {
    '16k': {
        'sample_rate': 16000,
        'frame_length': 50.0,
        'frame_shift': 25.0,
        'n_fft': 2048,
        'n_mels': 160,
        'f_min': 20.0,
        'f_max': None,
    },
    '52734': {
        'sample_rate': 52734,
        'frame_length': 50.0,
        'frame_shift': 25.0,
        'n_fft': 4096,
        'n_mels': 300,
        'f_min': 100.0,
        'f_max': 26360.0,
    },
}


def build_method_args(preset_name):
    preset = PRESETS[preset_name]
    return {
        'sample_rate': preset['sample_rate'],
        'frame_length': preset['frame_length'],
        'frame_shift': preset['frame_shift'],
        'n_fft': preset['n_fft'],
        'n_mels': preset['n_mels'],
        'f_min': preset['f_min'],
        'f_max': preset['f_max'],
        'window_type': 'hanning',
        'power': 2.0,
        'center': False,
        'pad': 0,
        'pad_mode': 'reflect',
        'normalized': False,
        'norm': None,
        'mel_scale': 'htk',
        'log_type': 'log',
        'eps': 1.0e-10,
        'transpose': True,
    }


def run_case(preset_name, seconds=30):
    method_args = build_method_args(preset_name)
    warnings = []
    if (method_args['sample_rate'] == 16000 and
            method_args['n_mels'] >= 300 and
            method_args['n_fft'] <= 1024):
        warnings.append(
            "[Warning] 16k + n_fft<=1024 + n_mels>=300 may produce zero mel filterbanks. "
            "For local debug, use n_fft=2048 and n_mels=160/200. "
            "For paper-aligned ShipsEar, use sample_rate=52734, n_fft=4096, n_mels=300."
        )

    featurizer = AudioFeaturizer(
        feature_method='MelSpectrogram',
        method_args=method_args,
        expected_sample_rate=method_args['sample_rate'],
    )
    waveform = torch.randn(method_args['sample_rate'] * seconds, dtype=torch.float32)
    feature = featurizer(waveform)
    feature_2d = feature.squeeze(0)
    resolved = featurizer.feat_fun.resolved_args

    print(f"preset={preset_name}")
    print(f"sample_rate={resolved['sample_rate']}")
    print(f"win_length={resolved['win_length']}")
    print(f"hop_length={resolved['hop_length']}")
    print(f"n_fft={resolved['n_fft']}")
    print(f"n_mels={resolved['n_mels']}")
    print(f"feature_2d_shape={tuple(feature_2d.shape)}")
    print(f"has_nan_inf={bool(~torch.isfinite(feature_2d).all())}")
    if warnings:
        for warning in warnings:
            print(warning)
    else:
        print("warning=None")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', choices=sorted(PRESETS.keys()), required=True)
    args = parser.parse_args()
    logger.remove()
    run_case(args.preset)


if __name__ == '__main__':
    main()
