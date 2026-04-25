import numpy as np
import torch
import torchaudio.compliance.kaldi as Kaldi
from torch import nn
from torchaudio.transforms import MelSpectrogram, Spectrogram, MFCC
from loguru import logger


class AudioFeaturizer(nn.Module):
    """音频特征器

    :param feature_method: 所使用的预处理方法
    :type feature_method: str
    :param use_hf_model: 是否使用HF上的Wav2Vec2类似模型提取音频特征
    :type use_hf_model: bool
    :param method_args: 预处理方法的参数
    :type method_args: dict
    """

    def __init__(self, feature_method='MelSpectrogram', use_hf_model=False, method_args={}):
        super().__init__()
        self._method_args = dict(method_args or {})
        self._feature_method = feature_method
        self.use_hf_model = use_hf_model
        if self.use_hf_model:
            from transformers import AutoModel, AutoFeatureExtractor
            # 判断是否使用GPU提取特征
            use_gpu = torch.cuda.is_available() and method_args.get('use_gpu', True)
            self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
            # 加载Wav2Vec2类似模型
            self.processor = AutoFeatureExtractor.from_pretrained(feature_method)
            self.feature_model = AutoModel.from_pretrained(feature_method).to(self.device)
            logger.info(f'使用模型【{feature_method}】提取特征，使用【{self.device}】设备提取')
            # 获取模型的输出通道数
            inputs = self.processor(np.ones(16000 * 1, dtype=np.float32), sampling_rate=16000,
                                    return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.feature_model(**inputs)
                self.output_channels = outputs.extract_features.shape[2]
        else:
            if feature_method == 'MelSpectrogram':
                self.feat_fun = CompatibleMelSpectrogram(**self._method_args)
                self._method_args = self.feat_fun.resolved_args
            elif feature_method == 'Spectrogram':
                self.feat_fun = Spectrogram(**self._method_args)
            elif feature_method == 'MFCC':
                self.feat_fun = MFCC(**self._method_args)
            elif feature_method == 'Fbank':
                self.feat_fun = KaldiFbank(**self._method_args)
            else:
                raise Exception(f'预处理方法 {self._feature_method} 不存在!')
            logger.info(f'使用【{feature_method}】提取特征')

    def forward(self, waveforms, input_lens_ratio=None):
        """从AudioSegment中提取音频特征

        :param waveforms: Audio segment to extract features from.
        :type waveforms: AudioSegment
        :param input_lens_ratio: input length ratio
        :type input_lens_ratio: tensor
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)
        if self.use_hf_model:
            # 使用HF上的Wav2Vec2类似模型提取音频特征
            if isinstance(waveforms, torch.Tensor):
                waveforms = waveforms.numpy()
            inputs = self.processor(waveforms, sampling_rate=16000,
                                    return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.feature_model(**inputs)
                feature = outputs.extract_features.cpu().detach()
        else:
            # 使用普通方法提取音频特征
            feature = self.feat_fun(waveforms)
            if self._feature_method != 'MelSpectrogram':
                feature = feature.transpose(2, 1)
        # 归一化
        feature = feature - feature.mean(1, keepdim=True)
        if input_lens_ratio is not None:
            # 对掩码比例进行扩展
            input_lens = (input_lens_ratio * feature.shape[1])
            mask_lens = torch.round(input_lens).long()
            mask_lens = mask_lens.unsqueeze(1)
            # 生成掩码张量
            idxs = torch.arange(feature.shape[1], device=feature.device).repeat(feature.shape[0], 1)
            mask = idxs < mask_lens
            mask = mask.unsqueeze(-1)
            # 对特征进行掩码操作
            feature = torch.where(mask, feature, torch.zeros_like(feature))
        return feature

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self.use_hf_model:
            return self.output_channels
        if self._feature_method == 'MelSpectrogram':
            return self._method_args.get('n_mels', 128)
        elif self._feature_method == 'Spectrogram':
            return self._method_args.get('n_fft', 400) // 2 + 1
        elif self._feature_method == 'MFCC':
            return self._method_args.get('n_mfcc', 40)
        elif self._feature_method == 'Fbank':
            return self._method_args.get('num_mel_bins', 23)
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))


class CompatibleMelSpectrogram(nn.Module):
    def __init__(self, **method_args):
        super().__init__()
        args = dict(method_args or {})

        self.log_type = args.pop('log_type', None)
        self.transpose = bool(args.pop('transpose', True))

        frame_length = args.pop('frame_length', None)
        frame_shift = args.pop('frame_shift', None)
        window_type = args.pop('window_type', None)

        sample_rate = args.get('sample_rate', None)
        if sample_rate is None and (frame_length is not None or frame_shift is not None):
            raise ValueError("sample_rate is required when using frame_length or frame_shift.")
        if sample_rate is not None:
            sample_rate = int(sample_rate)
            args['sample_rate'] = sample_rate

        if frame_length is not None and args.get('win_length', None) is None:
            args['win_length'] = int(round(sample_rate * float(frame_length) / 1000.0))
        if frame_shift is not None and args.get('hop_length', None) is None:
            args['hop_length'] = int(round(sample_rate * float(frame_shift) / 1000.0))

        if window_type is not None:
            window_type = str(window_type).lower()
            if window_type in {'hann', 'hanning'}:
                args['window_fn'] = torch.hann_window
            elif window_type == 'hamming':
                args['window_fn'] = torch.hamming_window
            else:
                raise ValueError("window_type must be one of: hann, hanning, hamming.")

        win_length = args.get('win_length', None)
        n_fft = args.get('n_fft', None)
        if win_length is not None:
            win_length = int(win_length)
            args['win_length'] = win_length
            if n_fft is None:
                args['n_fft'] = win_length
            elif int(n_fft) < win_length:
                raise ValueError("n_fft must be >= win_length.")

        self.mel = MelSpectrogram(**args)

        resolved = dict(args)
        resolved['log_type'] = self.log_type
        resolved['transpose'] = self.transpose
        self.resolved_args = resolved
        self._log_config()

    def _log_config(self):
        logger.info(
            "[MelSpectrogram] "
            f"sample_rate={self.resolved_args.get('sample_rate')}, "
            f"n_fft={self.resolved_args.get('n_fft')}, "
            f"win_length={self.resolved_args.get('win_length')}, "
            f"hop_length={self.resolved_args.get('hop_length')}, "
            f"n_mels={self.resolved_args.get('n_mels')}, "
            f"f_min={self.resolved_args.get('f_min')}, "
            f"f_max={self.resolved_args.get('f_max')}, "
            f"center={self.resolved_args.get('center')}, "
            f"power={self.resolved_args.get('power')}, "
            f"log_type={self.log_type}, "
            f"transpose={self.transpose}"
        )

    def forward(self, waveforms):
        mel = self.mel(waveforms)
        log_type = None if self.log_type is None else str(self.log_type).lower()
        if log_type in {'log', 'ln'}:
            mel = torch.log(mel.clamp_min(1e-10))
        elif log_type == 'log10':
            mel = torch.log10(mel.clamp_min(1e-10))
        elif log_type in {None, 'none', 'false'}:
            pass
        else:
            raise ValueError("log_type must be one of: log, ln, log10, none, false.")

        if self.transpose:
            mel = mel.transpose(-1, -2)
        return mel


class KaldiFbank(nn.Module):
    def __init__(self, **kwargs):
        super(KaldiFbank, self).__init__()
        self.kwargs = kwargs

    def forward(self, waveforms):
        """
        :param waveforms: [Batch, Length]
        :return: [Batch, Feature, Length]
        """
        log_fbanks = []
        for waveform in waveforms:
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            log_fbank = Kaldi.fbank(waveform, **self.kwargs)
            log_fbank = log_fbank.transpose(0, 1)
            log_fbanks.append(log_fbank)
        log_fbank = torch.stack(log_fbanks)
        return log_fbank
