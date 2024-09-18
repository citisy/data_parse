import os
from pathlib import Path
from .base import DataLoader, DataSaver, DataRegister, get_image, save_image
from utils import os_lib


class Loader(DataLoader):
    """a simple loader for image generation and so on

    Data structure:
        .
        ├── [task]
        └── [text_task]

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.SimpleTextImage import DataRegister, Loader, DataVisualizer

            data_dir = 'data/simple_text_image'
            loader = Loader(data_dir)
            data = loader(generator=True, image_type=DataRegister.ARRAY)

            # visual train dataset
            DataVisualizer(f'{data_dir}/visuals', verbose=False)(data[0])

    """
    default_set_type = [DataRegister.MIX]

    def _call(self, task='images', **kwargs):
        gen_func = os_lib.find_all_suffixes_files(f'{self.data_dir}/{task}', self.image_suffixes)
        return self.gen_data(gen_func, task=task, **kwargs)

    def get_ret(self, fp, image_type=DataRegister.PATH, task='images', text_task='texts', **kwargs) -> dict:
        image_path = os.path.abspath(fp)
        image = get_image(image_path, image_type)

        text_path = image_path.replace(task, text_task).replace(Path(image_path).suffix, '.txt')
        with open(text_path, 'r', encoding='utf8') as f:
            text = f.read()

        return dict(
            _id=fp.name,
            image=image,
            text=text,
        )


class Saver(DataSaver):
    """

    Data structure:
        .
        ├── [task]
        └── [text_task]

    """

    image_suffix = '.png'

    def mkdirs(self, set_types, task='images', text_task='texts', **kwargs):
        os_lib.mk_dir(f'{self.data_dir}/{task}')
        os_lib.mk_dir(f'{self.data_dir}/{text_task}')

    def _call(self, iter_data, **gen_kwargs):
        return self.gen_data(iter_data, **gen_kwargs)

    def parse_ret(self, ret, image_type=DataRegister.PATH, task='images', text_task='texts', **get_kwargs):
        image = ret['image']
        text = ret['text']
        _id = ret['_id']

        image_path = f'{self.data_dir}/{task}/{_id}'
        text_path = image_path.replace(task, text_task).replace(Path(_id).suffix, '.txt')
        save_image(image, image_path, image_type)
        with open(text_path, 'w', encoding='utf8') as f:
            f.write(text)
