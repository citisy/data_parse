from .base import DataRegister, DataLoader, DataSaver
from utils import os_lib


info = {
    'simple': {
        'url': '',
        'fn': 'data.json'
    },

    # https://zhuanlan.zhihu.com/p/634873585
    'stanford': {
        'url': 'https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json',
        'fn': 'alpaca_data.json',
    },

    # https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM?tab=readme-ov-file
    'GPT-4-LLM_en': {
        'url': 'https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json',
        'fn': 'alpaca_gpt4_data.json',
    },

    'GPT-4-LLM_zh': {
        'url': 'https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json',
        'fn': 'alpaca_gpt4_data_zh.json',
    },
}


class Loader(DataLoader):
    """

    Data structure:
        .
        └── [xxx.json]

    Data format:
        [{
            'instruction': '',
            'input': '',
            'output': ''
        }]
    """

    default_set_type = [DataRegister.MIX]
    dataset_info = info['simple']
    classes = ['instruction', 'input', 'output']

    def _call(self, fn=None, **gen_kwargs):
        fn = fn or self.dataset_info["fn"]

        gen_func = enumerate(os_lib.loader.load_json(f'{self.data_dir}/{fn}'))
        return self.gen_data(gen_func, **gen_kwargs)

    def get_ret(self, obj, **kwargs) -> dict:
        i, d = obj
        classes = []
        texts = []
        for k, v in d.items():
            classes.append(k)
            texts.append(v)

        return dict(
            _id=i,
            classes=classes,
            texts=texts,
        )


class Saver(DataSaver):
    default_set_type = [DataRegister.MIX]

    def _call(self, iter_data, fn='data.json', **kwargs):
        _iter_data = []
        for d in iter_data:
            _iter_data.append(dict(zip(d['classes'], d['texts'])))
        os_lib.saver.save_json(_iter_data, f'{self.data_dir}/{fn}')
