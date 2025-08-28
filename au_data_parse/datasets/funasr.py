import os
import json
import numpy as np
from utils import os_lib
from .base import DataRegister, DataLoader, DataSaver, DatasetGenerator, get_audio, save_audio
from pathlib import Path


class Loader(DataLoader):
    """https://github.com/modelscope/FunASR

    Data structure:
        data_dir
        ├── audios
        │   └── [set_task]
        │       └── [xxx.wav]
        └── labels
            └── [set_task]
                ├── train.jsonl
                └── test.jsonl
    Usage:
        .. code-block:: python

            # get data
            from data_parse.au_data_parse.funasr import DataRegister, Loader

            loader = Loader(data_dir)
            data = loader(set_type=DataRegister.FULL, generator=True, audio_type=DataRegister.ARRAY)
            r = next(data[0])

    """

    default_set_type = [DataRegister.TRAIN, DataRegister.TEST]

    def _call(self, set_type=DataRegister.TRAIN, set_task='', label_dir='labels', **kwargs):
        gen_func = os_lib.loader.load_jsonl(f'{self.data_dir}/labels/{set_task}/{set_type.value}.jsonl')
        return self.gen_data(gen_func, **kwargs)

    def get_ret(self, ret, audio_type=DataRegister.PATH, **kwargs) -> dict:
        """ret={{"source": "", "target": ""}"""
        audio_path = ret['source']
        audio = get_audio(audio_path, audio_type)

        return dict(
            _id=Path(audio_path).name,
            audio=audio,
            text=ret['target'],
        )


class Saver(DataSaver):
    """https://github.com/modelscope/FunASR

    Data structure:
        data_dir
        ├── audios
        │   └── [set_task]
        │       └── [xxx.wav]
        └── labels
            └── [set_task]
                ├── train.jsonl
                └── test.jsonl

    Usage:
        .. code-block:: python
            # save as PaddleOcr type
            from au_data_parse.funasr import Saver

            saver = Saver(data_dir)
            saver(data, set_type=DataRegister.TRAIN, audio_type=DataRegister.PATH)
    """

    def mkdirs(self, set_types, **kwargs):
        set_task = kwargs.get('set_task', '')
        os_lib.mk_dir(f'{self.data_dir}/labels/{set_task}')

    def _call(self, iter_data, set_task='', set_type=DataRegister.TRAIN, **gen_kwargs):
        """ret={{"key": "", "source": "", "source_len": -1, "target": "", "target_len": -1}"""
        f = open(f'{self.data_dir}/labels/{set_task}/{set_type.value}.jsonl', 'w', encoding='utf8')
        self.gen_data(iter_data, f=f, **gen_kwargs)
        f.close()

    def parse_ret(self, ret, f=None, audio_type=DataRegister.PATH, set_task='', **get_kwargs):
        """save_dict={{"source": "", "target": ""}"""
        _id = ret['_id']
        audio_path = os.path.abspath(f'{self.data_dir}/audios/{set_task}/{_id}')
        save_audio(ret['audio'], audio_path, audio_type)
        save_dict = dict(
            source=audio_path,
            target=ret['text'],
        )
        f.write(json.dumps(save_dict, ensure_ascii=False) + '\n')
