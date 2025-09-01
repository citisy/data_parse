from typing import List

import numpy as np


class BaseAug:
    def __init__(self, **kwargs):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.__dict__.update(kwargs)

    def __call__(self, image, *args, **kwargs) -> dict:
        raise NotImplementedError

    def get_add_params(self, *args) -> dict:
        raise NotImplementedError

    def parse_add_params(self, ret: dict) -> tuple:
        raise NotImplementedError

    def apply_image(self, image: np.ndarray, *args) -> np.ndarray:
        raise NotImplementedError

    def apply_bboxes(self, bboxes: np.ndarray | list, *args) -> np.ndarray:
        raise NotImplementedError

    def apply_classes(self, classes: np.ndarray | list, *args) -> np.ndarray:
        raise NotImplementedError

    def apply_image_list(self, image_list: List[np.ndarray], *args) -> List[np.ndarray]:
        raise NotImplementedError

    def apply_bboxes_list(self, bboxes_list: List[np.ndarray], *args) -> List[np.ndarray]:
        raise NotImplementedError

    def apply_classes_list(self, classes_list: List[np.ndarray], *args) -> List[np.ndarray]:
        raise NotImplementedError

    def restore(self, ret: dict) -> dict:
        args = self.parse_add_params(ret)
        if 'image' in ret and ret['image'] is not None:
            ret['image'] = self.restore_image(ret['image'], *args)
        if 'bboxes' in ret and ret['bboxes'] is not None:
            ret['bboxes'] = self.restore_bboxes(ret['bboxes'], *args)
        if 'classes' in ret and ret['classes'] is not None:
            ret['classes'] = self.restore_classes(ret['classes'], *args)
        return ret

    def restore_image(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def restore_bboxes(self, bboxes: np.ndarray | list, *args, **kwargs) -> np.ndarray:
        raise NotImplemented

    def restore_classes(self, classes: np.ndarray | list, *args, **kwargs) -> np.ndarray:
        raise NotImplemented

