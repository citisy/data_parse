import os

import cv2
import shutil
import numpy as np
from tqdm import tqdm
from utils import os_lib, converter, visualize, cv_utils
from typing import List
from numbers import Number
from ... import DataRegister, DataLoader, DataSaver, DatasetGenerator


def get_image(obj: str, image_type):
    if image_type == DataRegister.PATH:
        image = obj
    elif image_type == DataRegister.ARRAY:
        image = os_lib.loader.load_img(obj)
    elif image_type == DataRegister.GRAY_ARRAY:
        image = cv2.imread(obj, cv2.IMREAD_GRAYSCALE)
    elif image_type == DataRegister.NPY:
        image = np.load(obj)
    elif image_type == DataRegister.BASE64:
        image = cv2.imread(obj)
        image = converter.DataConvert.image_to_base64(image)
    elif image_type == DataRegister.URL:
        import requests
        r = requests.get(obj)
        image = r.content
        image = converter.DataConvert.bytes_to_image(image)
    else:
        raise ValueError(f'Unknown input {image_type = }')

    return image


def save_image(obj, save_path, image_type):
    if image_type == DataRegister.PATH:
        if os.path.abspath(obj) != os.path.abspath(save_path):
            shutil.copy(obj, save_path)
    elif image_type == DataRegister.ARRAY:
        os_lib.saver.save_img(obj, save_path)
    elif image_type == DataRegister.BASE64:
        obj = converter.DataConvert.base64_to_image(obj)
        os_lib.saver.save_img(obj, save_path)
    else:
        raise ValueError(f'Unknown input {image_type = }')


class ImgClsDict:
    # image file name, e.g. 'xxx.png'
    _id: str

    # if `image_type = DataRegister.PATH`,, return a string
    # if `image_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    image: str or np.ndarray

    # usually an int number giving the pic class
    _class: Number


class ObjDetDict:
    # image file name, e.g. 'xxx.png'
    _id: str

    # if `image_type = DataRegister.PATH`,, return a string
    # if `image_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    image: str or np.ndarray

    # a np.ndarray with shape of (n_obj, 4), 4 gives [top_left_x, top_left_y, right_down_x, right_down_y] usually
    bboxes: np.ndarray

    # a np.ndarray with shape of (n_obj, )
    classes: np.ndarray


class ImgSegDict:
    # image file name, e.g. 'xxx.png'
    _id: str

    # if `image_type = DataRegister.PATH`,, return a string
    # if `image_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    image: str or np.ndarray

    # if `image_type = DataRegister.PATH`,, return a string
    # if `image_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    label_mask: str or np.ndarray


class DataLoader(DataLoader):
    """
    for implementation, usually override the following methods:
        _call(): called by `load()`, which preparing the data, and returning an iterable function warped by `gen_data()`
        get_ret(): called by `gen_data()`, which is logic of parsing the data, and return a dict of result
    """
    default_image_type = DataRegister.PATH
    image_suffix = '.png'
    image_suffixes = os_lib.suffixes_dict['img']

    def load(self, image_type=None, **load_kwargs):
        """
        Args:
            image_type(DataRegister): `DataRegister.PATH` or `DataRegister.ARRAY`
                PATH -> a str of image abs path
                ARRAY -> a np.ndarray of image, read from cv2, as (h, w, c)
            load_kwargs:
                see also `_call` function to get more details of load_kwargs
                see also 'gen_data' function to get more details of gen_kwargs
                see also 'get_ret' function to get more details of get_kwargs
        """
        image_type = image_type or self.default_image_type
        return super().load(image_type=image_type, **load_kwargs)


class DatasetGenerator(DatasetGenerator):
    """generate datasets for training, testing and valuating"""
    image_suffix = '.jpg'
    image_suffixes = os_lib.suffixes_dict['img']


class DataVisualizer:
    """
    Usage:
        .. code-block:: python

            visualizer = DataVisualizer('visuals', verbose=False)

            rets1 = [
                {'_id': '0.png', 'image': image, 'bboxes': bboxes, 'classes': classes},
                {'_id': '1.png', 'image': image, 'bboxes': bboxes, 'classes': classes},
                {'_id': '2.png', 'image': image, 'bboxes': bboxes, 'classes': classes},
            ]
            # will save 3 images, each image contains 1 sub image
            visualizer(rets1)

            rets2 = [
                {'image': image, 'bboxes': bboxes, 'classes': classes},
                {'image': image, 'bboxes': bboxes, 'classes': classes},
                {'image': image, 'bboxes': bboxes, 'classes': classes},
            ]
            # will save 3 images, each image contains 2 sub images
            # note, rets2 do not need key of `_id`, the value get from rets1
            visualizer(rets1, rets2)

            # use special visual method
            # not necessary
            def visual_one_image(r, **visual_kwargs):
                image = r['image']
                bboxes = r['bboxes']
                classes = r['classes']
                colors = [visualize.get_color_array(int(cls)) for cls in classes]
                image = visualize.ImageVisualize.block(image, bboxes, colors=colors)
                return image

            visualizer.visual_one_image = visual_one_image
    """

    def __init__(self, save_dir, pbar=True, **saver_kwargs):
        self.save_dir = save_dir
        saver_kwargs.setdefault('verbose', not pbar)
        self.saver = os_lib.Saver(**saver_kwargs)
        self.pbar = pbar and not saver_kwargs['verbose']  # do not use pbar directly, 'cause they clash
        os_lib.mk_dir(save_dir)

    def __call__(self, *iter_data, max_vis_num=None, return_image=False, **visual_kwargs):
        """

        Args:
            iter_data (List[dict]):
                each data dict must have the key of '_id', 'image' at lease
                    - _id (str): the name to save the image
                    - image (np.ndarray): must have the same shape
                    - label_mask:
                    - bboxes:
                    - classes:
                    - confs:
                    - colors:

            **visual_kwargs:
                cls_alias:
                return_image:

        Returns:

        """
        pbar = zip(*iter_data)
        if self.pbar:
            pbar = tqdm(pbar, desc='visual')

        vis_num = 0
        max_vis_num = max_vis_num or float('inf')
        cache_image = []
        for rets in pbar:
            if vis_num >= max_vis_num:
                return

            images = []
            _id = rets[0]['_id']
            for r in rets:
                images.append(self.visual_one_image(r, **visual_kwargs))

                if 'label_mask' in r and r['label_mask'] is not None:
                    images.append(self.visual_one_image({'image': r['label_mask']}, **visual_kwargs))

            image = cv_utils.splice_image(images)
            self.saver.save_img(image, f'{self.save_dir}/{_id}')
            vis_num += 1
            if return_image:
                cache_image.append(image)

        return cache_image

    def visual_one_image(self, r, **visual_kwargs):
        image = r['image']

        if 'bboxes' in r and r['bboxes'] is not None:
            bboxes = r['bboxes']

            if 'classes' in r and r['classes'] is not None:
                classes = r['classes']

                if 'colors' in r:
                    colors = r['colors']
                else:
                    colors = [visualize.get_color_array(int(cls)) for cls in classes]

                if 'cls_alias' in visual_kwargs and visual_kwargs['cls_alias']:
                    cls_alias = visual_kwargs['cls_alias']
                    classes = [cls_alias[_] for _ in classes]

                if 'confs' in r:
                    classes = [f'{cls} {conf:.6f}' for cls, conf in zip(classes, r['confs'])]

                image = visualize.ImageVisualize.label_box(image, bboxes, classes, colors=colors)

            else:
                if 'colors' in r:
                    colors = r['colors']
                else:
                    colors = [visualize.cmap['Black']['array'] for _ in bboxes]

                image = visualize.ImageVisualize.box(image, bboxes, colors=colors)

        return image
