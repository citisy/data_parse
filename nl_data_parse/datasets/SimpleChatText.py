import json

from .base import DataRegister, DataLoader, DataSaver
from utils import os_lib

info = {
    'simple': {
        'url': '',
        'fn': 'data.jsonl'
    },
}


class Loader(DataLoader):
    """

    Data structure:
        .
        └── [xxx.jsonl]

    Data format:
        [{
            messages_key: [{
                "role": "",
                "content": ""
            }]
        }]
    """

    default_set_type = [DataRegister.MIX]
    dataset_info = info['simple']
    classes = ['system', 'user', 'assistant']
    messages_key = 'messages'

    def _call(self, fn=None, **gen_kwargs):
        fn = fn or self.dataset_info["fn"]

        def gen_func():
            with open(f'{self.data_dir}/{fn}', 'r', encoding='utf-8') as f:
                for i, line in enumerate(f.readlines()):
                    yield i, json.loads(line)

        return self.gen_data(gen_func(), **gen_kwargs)

    def get_ret(self, obj, **kwargs) -> dict:
        i, d = obj
        # classes = []
        # texts = []
        #
        # for dd in d[self.messages_key]:
        #     classes.append(dd['role'])
        #     texts.append(dd['content'])

        return dict(
            _id=i,
            # classes=classes,
            # texts=texts,
            messages=d[self.messages_key],
        )


class Saver(DataSaver):
    def _call(self, iter_data, fn='data.jsonl', **kwargs):
        _iter_data = []
        for d in iter_data:
            _iter_data.append(dict(
                messages=[dict(role=_cls, content=_text) for _cls, _text in zip(d['classes'], d['texts'])]
            ))

        os_lib.saver.save_jsonl(_iter_data, f'{self.data_dir}/{fn}')
