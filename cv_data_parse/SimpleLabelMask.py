import os
from pathlib import Path
from .base import DataLoader, DataSaver, DataRegister, get_image, save_image
from utils import os_lib


class Loader(DataLoader):
    """a simple loader for image segmentation, image generation and so on

    Data structure:
        .
        ├── [task]
        └── [mask_task]

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.SimpleLabelMask import DataRegister, Loader, DataVisualizer

            data_dir = 'data/simple_label_mask'
            loader = Loader(data_dir)
            data = loader(generator=True, image_type=DataRegister.ARRAY)

            # visual train dataset
            DataVisualizer(f'{data_dir}/visuals', verbose=False)(data[0])

    """
    default_set_type = [DataRegister.MIX]

    def _call(self, task='original', **kwargs):
        gen_func = os_lib.find_all_suffixes_files(f'{self.data_dir}/{task}', self.image_suffixes)
        return self.gen_data(gen_func, task=task, **kwargs)

    def get_ret(self, fp, image_type=DataRegister.PATH, task='original', mask_task='original_masks', **kwargs) -> dict:
        image_path = os.path.abspath(fp)
        image = get_image(image_path, image_type)

        label_mask_path = image_path.replace(task, mask_task)
        label_mask = get_image(label_mask_path, image_type)

        return dict(
            _id=fp.name,
            image=image,
            label_mask=label_mask,
        )


class Saver(DataSaver):
    """https://github.com/ultralytics/yolov5

    Data structure:
        .
        ├── [task]
        └── [mask_task]

    Usage:
        .. code-block:: python

            # convert cmp_facade to SimpleLabelMask
            # load data from cmp_facade
            from data_parse.cv_data_parse.cmp_facade import Loader
            from utils.register import DataRegister
            loader = Loader('data/cmp_facade')
            data = loader()

            # save as SimpleLabelMask type
            from data_parse.cv_data_parse.SimpleLabelMask import Saver
            saver = Saver('data/simple_label_mask')
            saver(data)

    """

    def mkdirs(self, set_types, task='original', mask_task='original_masks', **kwargs):
        os_lib.mk_dir(f'{self.data_dir}/{task}')
        os_lib.mk_dir(f'{self.data_dir}/{mask_task}')

    def _call(self, iter_data, **gen_kwargs):
        return self.gen_data(iter_data, **gen_kwargs)

    def parse_ret(self, ret, image_type=DataRegister.PATH, task='original', mask_task='original_masks', **kwargs):
        image = ret['image']
        label_mask = ret['label_mask']
        _id = ret['_id']

        image_path = f'{self.data_dir}/{task}/{_id}'
        label_mask_path = image_path.replace(task, mask_task)
        save_image(image, image_path, image_type)
        save_image(label_mask, label_mask_path, image_type)
