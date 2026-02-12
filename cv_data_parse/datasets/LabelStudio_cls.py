import json
import os
from pathlib import Path

from utils import os_lib
from .base import DataLoader, DataRegister, DataSaver, DatasetGenerator, get_image, save_image


class Loader(DataLoader):
    """https://labelstud.io/

    Data structure:
        .
        ├── images
        │   └── [set_task]
        │       └── xxx.png
        └── [set_task].[set_type].json

    """

    image_suffix = '.png'
    classes = []
    default_set_type = [DataRegister.MIX]

    def _call(self, set_type=None, set_task='label_studio', **kwargs):
        with open(f'{self.data_dir}/{set_task}.{set_type.value}.json', 'r', encoding='utf8') as f:
            gen_func = json.load(f)
        return self.gen_data(gen_func, set_task=set_task, **kwargs)

    def get_ret(self, js, image_type=DataRegister.PATH, set_task='label_studio', task='annotations', **kwargs) -> dict:
        image_path = Path(js['data']['image'])
        image_root = str(image_path.parent)
        image_name = image_path.name
        if '-' in image_name:
            sub_id, _id = image_name.split('-', 1)
        else:
            sub_id, _id = '', image_name
        if not os.path.exists(image_path):
            image_path = f'{self.data_dir}/images/{set_task}/{_id}'
        image_path = os.path.abspath(image_path)
        image = get_image(image_path, image_type)

        classes = []
        for a in js[task]:
            for r in a['result']:
                v = r['value']
                if r['type'] == 'choices':
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
    default_set_type = [DataRegister.MIX]

    def _call(
            self,
            iter_data, set_type=None, image_type=DataRegister.PATH, set_task='label_studio',
            task='annotations', cls_alias=None, is_save_image=True, is_multi_label=False,
            **kwargs
    ):
        assert task in ['annotations', 'predictions']
        rets = []
        for dic in iter_data:
            _id = dic['_id']
            sub_id = dic.get('sub_id')
            image_root = dic['image_root']
            image = dic['image']
            if is_save_image:
                image_path = f'{self.data_dir}/images/{set_task}/{_id}'
                save_image(image, image_path, image_type)

            classes = dic['classes']
            if cls_alias:
                classes = [cls_alias[i] for i in classes]

            result = []
            if is_multi_label:
                result.append(dict(
                    value=dict(
                        choices=classes,
                    ),
                    type='choices',
                    from_name="category",
                    to_name="image",
                ))
            else:
                for cls in classes:
                    result.append(dict(
                        value=dict(
                            choices=[cls],
                        ),
                        type='choices',
                        from_name="category",
                        to_name="image",
                    ))

            if sub_id:
                image = f'{image_root}/{sub_id}-{_id}'
            else:
                image = f'{image_root}/{_id}'
            rets.append({
                'data': dict(image=image),
                task: [dict(
                    result=result
                )]
            })

        os_lib.saver.save_json(rets, f'{self.data_dir}/{set_task}.{set_type.value}.json')


class Generator(DatasetGenerator):
    def gen_sets(self, list_iter_data, *args, **kwargs):
        _iter_data = []
        for iter_data in list_iter_data:
            for ret in iter_data:
                if 'image' in ret:
                    assert isinstance(ret['image'], str)
                _iter_data.append(ret)
        return super().gen_sets(_iter_data, *args, **kwargs)

    def save_func(self, iter_data, candidate_ids, set_name, set_task=None, **kwargs):
        save_data = [iter_data[i] for i in candidate_ids]
        os_lib.saver.save_json(save_data, f'{self.data_dir}/{set_task}.{set_name}.json')
