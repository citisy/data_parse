import json
import os
from pathlib import Path

import numpy as np

from utils import cv_utils, os_lib
from .base import DataLoader, DataRegister, DataSaver, DatasetGenerator, get_image, save_image


class Loader(DataLoader):
    """https://labelstud.io/

    Data structure:
        .
        ├── images
        │   └── [set_task]
        └── [set_task].json

    """

    image_suffix = '.png'
    classes = []

    def _call(self, set_task='label_studio', **kwargs):
        with open(f'{self.data_dir}/{set_task}.json', 'r', encoding='utf8') as f:
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
        image_path = f'{self.data_dir}/images/{set_task}/{_id}'
        image_path = os.path.abspath(image_path)
        image = get_image(image_path, image_type)

        bboxes = []
        segmentations = []
        classes = []
        for a in js[task]:
            for r in a['result']:
                v = r['value']
                if v['type'] == 'rectanglelabels':
                    bboxes.append([v['x'], v['y'], v['width'], v['height']])
                    classes.append(self.classes.index(v['rectanglelabels'][0]))
                    size = (r['original_height'], r['original_width'], 3)
                elif v['type'] == 'polygonlabels':
                    points = r['value']['points']
                    points = np.array(points)
                    points /= 100
                    h, w = image.shape[:2]
                    points[:, 0] *= w
                    points[:, 1] *= h
                    points = points.astype(int)

                    points = np.clip(points, 0, [w, h], points)

                    segmentations.append(points.tolist())
                    classes.append(self.classes.index(v['polygonlabels'][0]))
                    size = (r['original_height'], r['original_width'], 3)

        bboxes = np.array(bboxes)
        bboxes /= 100
        bboxes = cv_utils.CoordinateConvert.top_xywh2top_xyxy(bboxes, wh=(size[1], size[0]), blow_up=True)
        classes = np.array(classes)

        return dict(
            _id=_id,
            sub_id=sub_id,
            image_root=image_root,
            image=image,
            size=size,
            bboxes=bboxes,
            segmentations=segmentations,
            classes=classes
        )


class Saver(DataSaver):
    def _call(self, iter_data, image_type=DataRegister.PATH, set_task='label_studio', task='annotations', cls_alias=None, is_save_image=True, **kwargs):
        rets = []
        for dic in iter_data:
            _id = dic['_id']
            sub_id = dic['sub_id']
            image_root = dic['image_root']
            image = dic['image']
            if is_save_image:
                image_path = f'{self.data_dir}/images/{set_task}/{_id}'
                save_image(image, image_path, image_type)

            if 'size' in dic:
                size = dic['size']
            elif isinstance(image, np.ndarray):
                size = image.shape[:2]
            else:
                raise 'must be set size or make image the type of np.ndarray'

            result = []

            classes = dic['classes']
            if cls_alias:
                classes = [cls_alias[i] for i in classes]

            bboxes = np.array(dic['bboxes']).reshape(-1, 4)
            bboxes = cv_utils.CoordinateConvert.top_xyxy2top_xywh(bboxes, wh=(size[1], size[0]), blow_up=False)
            bboxes *= 100
            bboxes = bboxes.tolist()

            for box, cls in zip(bboxes, classes):
                result.append(dict(
                    original_width=size[1],
                    original_height=size[0],
                    image_rotation=0,
                    value=dict(
                        x=box[0],
                        y=box[1],
                        width=box[2],
                        height=box[3],
                        rotation=0,
                        rectanglelabels=[cls]
                    ),
                    type='rectanglelabels',
                    from_name="label",
                    to_name="image",
                ))

            segmentations = dic['segmentations']
            for points, cls in zip(segmentations, classes):
                w = size[1]
                h = size[0]
                points[:, 0] /= w
                points[:, 1] /= h
                points *= 100
                points = points.tolist()
                result.append(dict(
                    original_width=size[1],
                    original_height=size[0],
                    image_rotation=0,
                    value=dict(
                        points=points,
                        rotation=0,
                        polygonlabels=[cls]
                    ),
                    type='polygonlabels',
                    from_name="label",
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

        os_lib.saver.save_json(rets, f'{self.data_dir}/{set_task}.json')



class MyGenerator(DatasetGenerator):
    def gen_sets(self, list_iter_data, *args, **kwargs):
        _iter_data = []
        for iter_data in list_iter_data:
            for ret in iter_data:
                ret.pop('image')
                _iter_data.append(ret)
        return super().gen_sets(_iter_data, *args, **kwargs)

    def save_func(self, iter_data, candidate_ids, set_name, set_task=None, **kwargs):
        save_data = [iter_data[i] for i in candidate_ids]
        os_lib.saver.save_json(save_data, f'{self.data_dir}/{set_task}.{set_name}.json')