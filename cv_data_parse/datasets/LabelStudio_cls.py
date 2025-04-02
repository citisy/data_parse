import os
import json
import cv2
import shutil
import numpy as np
from utils import os_lib, cv_utils
from .base import DataRegister, DataLoader, DataSaver, get_image, save_image
from tqdm import tqdm
from pathlib import Path


class Loader(DataLoader):
    """https://labelstud.io/

    Data structure:
        .
        ├── images
        └── [set_task].json

    """

    image_suffix = '.png'
    classes = []
    default_set_type = [DataRegister.MIX]

    def _call(self, set_task='label_studio', **kwargs):
        with open(f'{self.data_dir}/{set_task}.json', 'r', encoding='utf8') as f:
            gen_func = json.load(f)
        return self.gen_data(gen_func, set_task=set_task, **kwargs)

    def get_ret(self, js, image_type=DataRegister.PATH, set_task='label_studio', **kwargs) -> dict:
        image_path = Path(js['data']['image'])
        image_root = str(image_path.parent)
        sub_id, _id = image_path.name.split('-', 1)
        image_path = f'{self.data_dir}/images/{set_task}/{_id}'
        image_path = os.path.abspath(image_path)
        image = get_image(image_path, image_type)

        classes = []
        for a in js['annotations']:
            for r in a['result']:
                v = r['value']
                for c in v['choices']:
                    classes.append(self.classes.index(c))

        return dict(
            _id=_id,
            sub_id=sub_id,
            image_root=image_root,
            image=image,
            classes=classes
        )


class Saver(DataSaver):
    classes = []

    def __call__(self, data, set_type=DataRegister.FULL, image_type=DataRegister.PATH, **kwargs):
        os_lib.mk_dir(f'{self.data_dir}/images')
        super().__call__(data, set_type, image_type, **kwargs)

    def _call(self, iter_data, image_type=DataRegister.PATH, set_task='label_studio', cls_alias=None, is_save_image=True, **kwargs):
        rets = []
        for dic in iter_data:
            _id = dic['_id']
            sub_id = dic['sub_id']
            image_root = dic['image_root']
            image = dic['image']
            if is_save_image:
                image_path = f'{self.data_dir}/images/{set_task}/{_id}'
                save_image(image, image_path, image_type)

            classes = dic['classes']
            if cls_alias:
                classes = [cls_alias[i] for i in classes]

            result = []
            for cls in classes:
                result.append(dict(
                    value=dict(
                        choices=[cls],
                    ),
                    type='choices'
                ))

            rets.append(dict(
                data=dict(image=f'{image_root}/{sub_id}-{_id}'),
                annotations=[dict(
                    result=result
                )]
            ))

        os_lib.saver.save_json(rets, f'{self.data_dir}/{set_task}.json')

