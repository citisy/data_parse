import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

from utils import os_lib, converter
from .base import DataLoader, DataSaver, DataRegister, get_image, save_image, DatasetGenerator, DataVisualizer


class Loader(DataLoader):
    """a simple loader for image classification, image generation and so on

    Data structure:
        .
        ├── images
        │   └── [task]
        └── image_sets
            └── [set_task]
                  ├── train.json  # {'image_path': 'class'}
                  ├── test.json   # would like to be empty or same to val.txt
                  └── val.json

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.SimpleClassImage import DataRegister, Loader, DataVisualizer

            data_dir = 'data/xxx'
            loader = Loader(data_dir)
            data = loader(generator=True, image_type=DataRegister.ARRAY)

            # visual train dataset
            DataVisualizer(f'{data_dir}/visuals', verbose=False)(data[0])

    """
    default_set_type = [DataRegister.MIX]
    label_suffix = '.json'
    loader = os_lib.Loader(verbose=False)

    def _call(self, set_task='', set_type=DataRegister.TRAIN, **gen_kwargs):
        d = os_lib.loader.load_json(f'{self.data_dir}/image_sets/{set_task}/{set_type.value}.json')

        def gen_func():
            for fp, _class in d.items():
                yield fp, _class

        return self.gen_data(gen_func(), **gen_kwargs)

    def get_ret(self, obj, image_type=DataRegister.PATH, **kwargs) -> dict:
        fp, _class = obj
        image_path = os.path.abspath(fp)
        image = get_image(image_path, image_type)

        if hasattr(self, 'classes'):
            _class = self.classes[_class]

        return dict(
            _id=Path(fp).name,
            image=image,
            _class=_class
        )
