from .base import DataRegister, DataLoader, DataSaver
from utils import os_lib

info = {
    'simple': {
        'url': '',
        'fn': 'data.json'
    },
}


class Loader(DataLoader):
    """

    Data structure:
        .
        └── [xxx.json]

    Data format:
        [{
            "conversations": [{
                "from": "",
                "value": ""
              }],
            "system": "",   # not necessary
            "tools": ""
        }]
    """

    default_set_type = [DataRegister.MIX]
    dataset_info = info['simple']
    classes = ['human', 'function_call', 'observation', 'gpt']

    def _call(self, fn=None, **gen_kwargs):
        fn = fn or self.dataset_info["fn"]
        gen_func = enumerate(os_lib.loader.load_json(f'{self.data_dir}/{fn}'))
        return self.gen_data(gen_func, **gen_kwargs)

    def get_ret(self, obj, **kwargs) -> dict:
        i, d = obj
        classes = []
        texts = []

        if 'system' in d:
            classes.append('system')
            texts.append(d['system'])

        for dd in d['conversations']:
            classes.append(dd['from'])
            texts.append(dd['value'])

        return dict(
            _id=i,
            classes=classes,
            texts=texts,
            tools=d['tools']
        )


class Saver(DataSaver):
    def _call(self, iter_data, fn='data.json', **kwargs):
        _iter_data = []
        for d in iter_data:
            ret = dict()
            if d['classes'][0] == 'system':
                ret['system'] = d['texts'][0]
                d['classes'].pop(0)
                d['texts'].pop(0)

            ret['conversations'] = [{'from': c, 'value': t} for c, t in zip(d['classes'], d['texts'])]
            ret['tools'] = d['tools']
            _iter_data.append(ret)

        os_lib.saver.save_json(_iter_data, f'{self.data_dir}/{fn}')
