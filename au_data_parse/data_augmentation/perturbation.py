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
