class UpSamples:
    def __call__(self, audio, **kwargs):
        return {
            'audio': self.apply_audio(audio),
        }

    def apply_audio(self, audio, *args):
        return audio * (2 ** 15)


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, audio, **kwargs):
        return {
            'audio': self.apply_audio(audio),
        }

    def apply_audio(self, audio, *args):
        return (audio + self.mean) / self.std


class ReSamples:
    def __init__(self):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__

    def __call__(self, audio, ori_sr, tgt_sr=16000, **kwargs):
        add_params = self.get_add_params(ori_sr, tgt_sr)

        return {
            'audio': self.apply_audio(audio, add_params),
            **add_params
        }

    def get_add_params(self, ori_sr, tgt_sr):
        return {self.name: dict(
            ori_sr=ori_sr,
            tgt_sr=tgt_sr
        )}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['ori_sr'], info['tgt_sr']

    def apply_audio(self, audio, ret, *args):
        import librosa  # pip install librosa
        ori_sr, tgt_sr = self.parse_add_params(ret)
        return librosa.resample(audio, orig_sr=ori_sr, target_sr=tgt_sr)
