import math

import numpy as np


class FBank:
    """see also `torchaudio.compliance.kaldi.fbank`"""
    blackman_coeff: float = 0.42
    dither: float = 0.0
    energy_floor: float = 1.0
    frame_length: float = 25.0
    frame_shift: float = 10.0
    high_freq: float = 0.0
    htk_compat: bool = False
    low_freq: float = 20.0
    min_duration: float = 0.0
    num_mel_bins: int = 23
    preemphasis_coefficient: float = 0.97
    raw_energy: bool = True
    remove_dc_offset: bool = True
    round_to_power_of_two: bool = True
    sample_frequency: float = 16000.0
    snip_edges: bool = True
    subtract_mean: bool = False
    use_energy: bool = False
    use_log_fbank: bool = True
    use_power: bool = True
    vtln_high: float = -500.0
    vtln_low: float = 100.0
    vtln_warp: float = 1.0
    window_type: str = "povey"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, audio, **kwargs):
        return {
            'audio': self.apply_audio(audio),
        }

    def apply_audio(self, audio: np.ndarray, *args):
        audio, window_shift, window_size, padded_window_size = self._get_waveform_and_window_properties(audio)

        if len(audio) < self.min_duration * self.sample_frequency:
            # signal is too short
            return np.empty((0, self.num_mel_bins + self.use_energy))

        # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
        strided_input, signal_log_energy = self._get_window(audio, padded_window_size, window_size, window_shift)

        # size (m, padded_window_size // 2 + 1)
        spectrum = np.abs(np.fft.rfft(strided_input))
        if self.use_power:
            spectrum = np.power(spectrum, 2.0)

        # size (num_mel_bins, padded_window_size // 2)
        mel_energies, _ = self.get_mel_banks(padded_window_size)

        # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
        mel_energies = np.pad(mel_energies, ((0, 0), (0, 1)), mode="constant", constant_values=0)

        # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
        mel_energies = np.dot(spectrum, mel_energies.T)
        if self.use_log_fbank:
            # avoid log of zero (which should be prevented anyway by dithering)
            mel_energies = np.log(np.maximum(mel_energies, np.finfo(np.float32).eps))

        # if use_energy then add it as the last column for htk_compat == true else first column
        if self.use_energy:
            signal_log_energy = signal_log_energy[:, np.newaxis]  # size (m, 1)
            # returns size (m, num_mel_bins + 1)
            if self.htk_compat:
                mel_energies = np.concatenate((mel_energies, signal_log_energy), axis=1)
            else:
                mel_energies = np.concatenate((signal_log_energy, mel_energies), axis=1)

        mel_energies = self._subtract_column_mean(mel_energies)
        return mel_energies

    def _get_waveform_and_window_properties(self, audio: np.ndarray, milliseconds_to_seconds=0.001):
        def _next_power_of_2(x: int) -> int:
            return 1 if x == 0 else 2 ** (x - 1).bit_length()

        window_shift = int(self.sample_frequency * self.frame_shift * milliseconds_to_seconds)
        window_size = int(self.sample_frequency * self.frame_length * milliseconds_to_seconds)
        padded_window_size = _next_power_of_2(window_size) if self.round_to_power_of_two else window_size
        return audio, window_shift, window_size, padded_window_size

    def _get_window(self, audio: np.ndarray, padded_window_size: int, window_size: int, window_shift: int):
        epsilon = np.finfo(np.float32).eps

        # size (m, window_size)
        strided_input = self._get_strided(audio, window_size, window_shift)

        if self.dither != 0.0:
            rand_gauss = np.random.randn(*strided_input.shape) * self.dither
            strided_input = strided_input + rand_gauss

        if self.remove_dc_offset:
            # Subtract each row/frame by its mean
            row_means = np.mean(strided_input, axis=1, keepdims=True)  # size (m, 1)
            strided_input = strided_input - row_means

        if self.raw_energy:
            # Compute the log energy of each row/frame before applying preemphasis and window function
            signal_log_energy = self._get_log_energy(strided_input, epsilon)  # size (m)

        if self.preemphasis_coefficient != 0.0:
            # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
            offset_strided_input = np.pad(strided_input, ((0, 0), (1, 0)), mode='edge')
            strided_input = strided_input - self.preemphasis_coefficient * offset_strided_input[:, :-1]

        # Apply window_function to each row/frame
        window_function = self._feature_window_function(window_size)
        window_function = window_function[np.newaxis, :]  # size (1, window_size)
        strided_input = strided_input * window_function  # size (m, window_size)

        # Pad columns with zero until we reach size (m, padded_window_size)
        if padded_window_size != window_size:
            padding_right = padded_window_size - window_size
            strided_input = np.pad(strided_input, ((0, 0), (0, padding_right)), mode='constant', constant_values=0)

        # Compute energy after window function (not the raw one)
        if not self.raw_energy:
            signal_log_energy = self._get_log_energy(strided_input, epsilon)  # size (m)

        return strided_input, signal_log_energy

    def _get_strided(self, waveform: np.ndarray, window_size: int, window_shift: int) -> np.ndarray:
        num_samples = len(waveform)

        if self.snip_edges:
            if num_samples < window_size:
                return np.empty((0, 0), dtype=waveform.dtype)
            else:
                m = 1 + (num_samples - window_size) // window_shift
        else:
            reversed_waveform = np.flip(waveform)
            m = (num_samples + (window_shift // 2)) // window_shift
            pad = window_size // 2 - window_shift // 2
            if pad > 0:
                pad_left = reversed_waveform[-pad:]
                waveform = np.concatenate((pad_left, waveform, reversed_waveform))
            else:
                waveform = np.concatenate((waveform[-pad:], reversed_waveform))

        # Create strided array manually
        result = np.zeros((m, window_size), dtype=waveform.dtype)
        for i in range(m):
            result[i] = waveform[i * window_shift: i * window_shift + window_size]
        return result

    def _get_log_energy(self, strided_input: np.ndarray, epsilon: float) -> np.ndarray:
        log_energy = np.log(np.maximum(np.sum(strided_input ** 2, axis=1), epsilon))
        if self.energy_floor == 0.0:
            return log_energy
        return np.maximum(log_energy, math.log(self.energy_floor))

    def _feature_window_function(self, window_size: int) -> np.ndarray:
        if self.window_type == 'hanning':
            return np.hanning(window_size)
        if self.window_type == 'hamming':
            return np.hamming(window_size)
        elif self.window_type == 'povey':
            return np.power(np.hanning(window_size), 0.85)
        elif self.window_type == 'rectangular':
            return np.ones(window_size)
        elif self.window_type == 'blackman':
            a = 2 * math.pi / (window_size - 1)
            window_function = np.arange(window_size)
            return self.blackman_coeff - 0.5 * np.cos(a * window_function) + (0.5 - self.blackman_coeff) * np.cos(2 * a * window_function)
        else:
            raise Exception("Invalid window type " + self.window_type)

    def get_mel_banks(self, window_length_padded: int):
        num_fft_bins = window_length_padded // 2
        nyquist = 0.5 * self.sample_frequency

        high_freq = self.high_freq
        if high_freq <= 0.0:
            high_freq += nyquist

        # fft-bin width [think of it as Nyquist-freq / half-window-length]
        fft_bin_width = self.sample_frequency / window_length_padded
        mel_low_freq = 1127.0 * math.log(1.0 + self.low_freq / 700.0)
        mel_high_freq = 1127.0 * math.log(1.0 + high_freq / 700.0)

        # divide by num_bins+1 in next line because of end-effects where the bins
        # spread out to the sides.
        mel_freq_delta = (mel_high_freq - mel_low_freq) / (self.num_mel_bins + 1)

        if self.vtln_high < 0.0:
            self.vtln_high += nyquist

        bin = np.arange(self.num_mel_bins)[:, np.newaxis]
        left_mel = mel_low_freq + bin * mel_freq_delta  # size(num_bins, 1)
        center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta  # size(num_bins, 1)
        right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta  # size(num_bins, 1)

        if self.vtln_warp != 1.0:
            left_mel = self.vtln_warp_mel_freq(high_freq, left_mel)
            center_mel = self.vtln_warp_mel_freq(high_freq, center_mel)
            right_mel = self.vtln_warp_mel_freq(high_freq, right_mel)

        center_freqs = self.inverse_mel_scale(center_mel)  # size (num_bins)
        # size(1, num_fft_bins)
        mel = self.mel_scale(fft_bin_width * np.arange(num_fft_bins))[np.newaxis, :]

        # size (num_bins, num_fft_bins)
        up_slope = (mel - left_mel) / (center_mel - left_mel)
        down_slope = (right_mel - mel) / (right_mel - center_mel)

        if self.vtln_warp == 1.0:
            # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
            bins = np.maximum(0, np.minimum(up_slope, down_slope))
        else:
            # warping can move the order of left_mel, center_mel, right_mel anywhere
            bins = np.zeros_like(up_slope)
            up_idx = (mel > left_mel) & (mel <= center_mel)  # left_mel < mel <= center_mel
            down_idx = (mel > center_mel) & (mel < right_mel)  # center_mel < mel < right_mel
            bins[up_idx] = up_slope[up_idx]
            bins[down_idx] = down_slope[down_idx]

        return bins, center_freqs

    def vtln_warp_mel_freq(self, high_freq: float, mel_freq: np.ndarray) -> np.ndarray:
        return self.mel_scale(self.vtln_warp_freq(high_freq, self.inverse_mel_scale(mel_freq)))

    def vtln_warp_freq(self, high_freq: float, freq: np.ndarray, ) -> np.ndarray:
        l = self.vtln_low * max(1.0, self.vtln_warp)
        h = self.vtln_high * min(1.0, self.vtln_warp)
        scale = 1.0 / self.vtln_warp
        Fl = scale * l  # F(l)
        Fh = scale * h  # F(h)
        assert l > self.low_freq and h < high_freq
        # slope of left part of the 3-piece linear function
        scale_left = (Fl - self.low_freq) / (l - self.low_freq)
        # slope of right part of the 3-piece linear function
        scale_right = (high_freq - Fh) / (high_freq - h)

        res = np.empty_like(freq)
        outside_low_high_freq = (freq < self.low_freq) | (freq > high_freq)
        before_l = freq < l
        before_h = freq < h
        after_h = freq >= h

        # order of operations matter here (since there is overlapping frequency regions)
        res[after_h] = high_freq + scale_right * (freq[after_h] - high_freq)
        res[before_h] = scale * freq[before_h]
        res[before_l] = self.low_freq + scale_left * (freq[before_l] - self.low_freq)
        res[outside_low_high_freq] = freq[outside_low_high_freq]

        return res

    @staticmethod
    def inverse_mel_scale(mel_freq: np.ndarray) -> np.ndarray:
        return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)

    @staticmethod
    def mel_scale(freq: np.ndarray) -> np.ndarray:
        return 1127.0 * np.log(1.0 + freq / 700.0)

    def _subtract_column_mean(self, tensor: np.ndarray) -> np.ndarray:
        # subtracts the column mean of the tensor size (m, n) if subtract_mean=True
        if self.subtract_mean:
            col_means = np.mean(tensor, axis=0, keepdims=True)
            tensor = tensor - col_means
        return tensor


class LFR:
    def __init__(self, lfr_m: int = 7, lfr_n: int = 6):
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n

    def __call__(self, audio, **kwargs):
        return {
            'audio': self.apply_audio(audio),
        }

    def apply_audio(self, audio, *args):
        lfr_m = self.lfr_m
        lfr_n = self.lfr_n
        T = audio.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))

        # Create left padding by repeating the first frame
        left_padding = np.tile(audio[0], (int((lfr_m - 1) // 2), 1))
        inputs = np.vstack((left_padding, audio))
        T = T + (lfr_m - 1) // 2
        feat_dim = inputs.shape[-1]

        # Calculate parameters for strided view
        last_idx = (T - lfr_m) // lfr_n + 1
        num_padding = lfr_m - (T - last_idx * lfr_n)

        if num_padding > 0:
            num_padding = (2 * lfr_m - 2 * T + (T_lfr - 1 + last_idx) * lfr_n) / 2 * (T_lfr - last_idx)
            padding_rows = np.tile(inputs[-1:], (int(num_padding), 1))
            inputs = np.vstack([inputs, padding_rows])

        # Create strided view manually since NumPy doesn't have as_strided
        lfr_outputs = np.zeros((T_lfr, lfr_m * feat_dim), dtype=np.float32)
        for i in range(T_lfr):
            start_idx = i * lfr_n
            end_idx = start_idx + lfr_m
            lfr_outputs[i] = inputs[start_idx:end_idx].flatten()

        return lfr_outputs
