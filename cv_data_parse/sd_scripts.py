import os
from pathlib import Path

from utils import os_lib
from .base import DataLoader, DataRegister, get_image


class Loader(DataLoader):
    """https://github.com/kohya-ss/sd-scripts.git

    Data structure:
        .
        └── [task]
            └── [repeat]_[subtask]
                ├── *.[png]
                └── *.txt

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.sd_scripts import DataRegister, Loader

            data_dir = 'data/sd_scripts'
            loader = Loader(data_dir)
            data = loader(generator=True, image_type=DataRegister.ARRAY)

    """

    txt_suffix = '.txt'
    default_data_type = DataRegister.MIX

    def _call(self, task='original', **gen_kwargs):
        def gen_func():
            for fp1 in Path(f'{self.data_dir}/{task}').glob('*'):
                # repeat, subtask = fp1.name.split('_', 1)
                for fp2 in fp1.glob('*'):
                    if fp2.suffix in self.image_suffixes:
                        yield fp2

        return self.gen_data(gen_func(), **gen_kwargs)

    def get_ret(self, fp, image_type=DataRegister.PATH, **kwargs) -> dict:
        image_path = os.path.abspath(fp)
        image = get_image(image_path, image_type)

        txt_path = fp.with_suffix(self.txt_suffix)
        txt_path = os.path.abspath(txt_path)
        text = os_lib.loader.load_txt(txt_path, split_line=False)

        return dict(
            _id=fp.name,
            image=image,
            text=text,
        )
