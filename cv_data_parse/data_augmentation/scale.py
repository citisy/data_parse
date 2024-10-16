"""change the shape of image by resizing the image"""
import numbers
from typing import List, Optional

import cv2
import numpy as np

from . import crop

interpolation_mode = [
    cv2.INTER_LINEAR,
    cv2.INTER_NEAREST,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
    cv2.INTER_LANCZOS4
]

SHORTEST, LONGEST = 1, 2
MAX, MIN = 3, 4
AUTO = 0


class Proportion:
    """choice an edge, count the scale ratio, and then scale the image proportional
    See Also `torchvision.transforms.Resize` or `albumentations.Resize`"""

    def __init__(self, interpolation=0, choice_type=MIN, max_ratio=None):
        """

        Args:
            interpolation (int):
            choice_type (int): see `Proportion.get_params()`
            max_ratio (float | List[float]): make sure the scale ratio falls in [1 / (1 + max_ratio_0), 1 + max_ratio_1]

        """
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.interpolation = interpolation_mode[interpolation]
        self.choice_type = choice_type
        self.max_ratio = max_ratio

    def get_params(self, dst_w, dst_h, w, h):
        if self.choice_type == MIN:  # choice the most min scale factor
            p = min(dst_w / w, dst_h / h)
        elif self.choice_type == MAX:  # choice the most max scale factor
            p = max(dst_w / w, dst_h / h)
        elif self.choice_type == SHORTEST:  # scale to be the same length as the shortest edge
            p = min((w, dst_w / w), (h, dst_h / h), key=lambda x: x[0])[1]
        elif self.choice_type == LONGEST:  # scale to be the same length as the longest edge
            p = max((w, dst_w / w), (h, dst_h / h), key=lambda x: x[0])[1]
        elif self.choice_type == AUTO:  # choice the most abs min scale factor
            p1 = abs(dst_w - w) / w
            p2 = abs(dst_h - h) / h
            p = dst_w / w if p1 < p2 else dst_h / h
        else:
            raise ValueError(f'dont support {self.choice_type = }')

        if self.max_ratio:
            if isinstance(self.max_ratio, numbers.Number):
                max_ratio = (self.max_ratio, self.max_ratio)
            else:
                max_ratio = self.max_ratio

            # set in [1 / (1 + max_ratio_0), 1 + max_ratio_1]
            p = max(min(p, 1 + max_ratio[1]), 1 / (1 + max_ratio[0]))
        return p

    def get_add_params(self, dst, w, h):
        if isinstance(dst, int):
            dst = (dst, dst)
        dst_w, dst_h = dst
        p = self.get_params(dst_w, dst_h, w, h)
        return {self.name: dict(p=p, w=w, h=h)}

    def parse_add_params(self, ret):
        return ret[self.name]['p'], ret[self.name]['w'], ret[self.name]['h']

    def __call__(self, image, dst, bboxes=None, **kwargs):
        h, w = image.shape[:2]
        add_params = self.get_add_params(dst, w, h)

        return {
            'image': self.apply_image(image, add_params),
            'bboxes': self.apply_bboxes(bboxes, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        p, _, _ = self.parse_add_params(ret)
        return cv2.resize(image, None, fx=p, fy=p, interpolation=self.interpolation)

    def apply_bboxes(self, bboxes, ret):
        p, _, _ = self.parse_add_params(ret)
        if bboxes is not None:
            bboxes = np.array(bboxes, dtype=float) * p
            bboxes = bboxes.astype(int)
        return bboxes

    def restore(self, ret):
        p, w, h = self.parse_add_params(ret)
        if 'image' in ret and ret['image'] is not None:
            image = ret['image']
            # note, use dsize to avoid float accuracy missing
            image = cv2.resize(image, (w, h), interpolation=self.interpolation)
            ret['image'] = image

        if 'bboxes' in ret and ret['bboxes'] is not None:
            bboxes = ret['bboxes']
            bboxes = np.array(bboxes, dtype=float) / p
            bboxes = bboxes.astype(int)
            ret['bboxes'] = bboxes

        return ret


class RandomProportion(Proportion):
    def __init__(self, size_range: List[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.size_range = size_range

    def get_params(self, dst_w, dst_h, w, h):
        # note, if size_range not set, default random ratio is [1.14, 1.71]
        # consider the future process is cropping usually,
        # to prevent over cutting, the default min ratio is more than 1
        size_range_w = self.size_range if self.size_range else (int(dst_w * 1.14), int(dst_w * 1.71))
        size_range_h = self.size_range if self.size_range else (int(dst_h * 1.14), int(dst_h * 1.71))
        dst_w = np.random.randint(*size_range_w)
        dst_h = np.random.randint(*size_range_h)
        return super().get_params(dst_w, dst_h, w, h)


class Rectangle:
    """scale [h, w] to [dst_h, dst_w]
    See Also `torchvision.transforms.Resize` or `albumentations.Resize`"""

    def __init__(self, interpolation=0):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.interpolation = interpolation_mode[interpolation]

    def get_params(self, dst_w, dst_h, w, h):
        return dst_w / w, dst_h / h

    def get_add_params(self, dst_w, dst_h, w, h):
        pw, ph = self.get_params(dst_w, dst_h, w, h)
        return {self.name: dict(pw=pw, ph=ph)}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['pw'], info['ph']

    def __call__(self, image, dst, bboxes=None, **kwargs):
        if isinstance(dst, int):
            dst = (dst, dst)

        h, w, c = image.shape
        dst_w, dst_h = dst
        add_params = self.get_add_params(dst_w, dst_h, w, h)

        return {
            'image': self.apply_image(image, add_params),
            'bboxes': self.apply_bboxes(bboxes, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        pw, ph = self.parse_add_params(ret)
        return cv2.resize(image, None, fx=pw, fy=ph, interpolation=self.interpolation)

    def apply_bboxes(self, bboxes, ret):
        if bboxes is not None:
            pw, ph = self.parse_add_params(ret)
            bboxes = np.array(bboxes, dtype=float) * np.array([pw, ph, pw, ph])
            bboxes = bboxes.astype(int)

        return bboxes

    def restore(self, ret):
        pw, ph = self.parse_add_params(ret)
        if 'image' in ret and ret['image'] is not None:
            image = ret['image']
            image = cv2.resize(image, None, fx=1 / pw, fy=1 / ph, interpolation=self.interpolation)
            ret['image'] = image

        if 'bboxes' in ret and ret['bboxes'] is not None:
            bboxes = ret['bboxes']
            bboxes = np.array(bboxes, dtype=float) / np.array([pw, ph, pw, ph])
            bboxes = bboxes.astype(int)
            ret['bboxes'] = bboxes

        return ret


class RandomRectangle(Rectangle):
    def __init__(self, size_range=None, **kwargs):
        super().__init__(**kwargs)
        self.size_range = size_range

    def get_params(self, dst_w, dst_h, w, h):
        # note, if size_range not set, default random ratio is [1.14, 1.71]
        # consider the future process is cropping usually,
        # to prevent over cutting, the default min ratio is more than 1
        size_range_w = self.size_range if self.size_range else (int(dst_w * 1.14), int(dst_w * 1.71))
        size_range_h = self.size_range if self.size_range else (int(dst_h * 1.14), int(dst_h * 1.71))
        dst_w = np.random.randint(*size_range_w)
        dst_h = np.random.randint(*size_range_h)
        return super().get_params(dst_w, dst_h, w, h)


class Complex:
    """one resize + one crop"""

    resize: Optional
    crop: Optional

    def __call__(self, image, dst, bboxes=None, **kwargs):
        ret = dict(image=image, bboxes=bboxes, dst=dst, **kwargs)
        ret.update(self.resize(**ret))
        ret.update(self.crop(**ret))

        return ret

    def apply_image(self, image, ret):
        image = self.resize.apply_image(image, ret)
        image = self.crop.apply_image(image, ret)
        return image

    def restore(self, ret):
        ret = self.crop.restore(ret)
        ret = self.resize.restore(ret)

        return ret


class LetterBox(Complex):
    """proportion resize, center crop with pad"""

    def __init__(self, interpolation=0, max_ratio=None, **pad_kwargs):
        self.resize = Proportion(choice_type=MIN, interpolation=interpolation, max_ratio=max_ratio)  # choice the min scale factor
        pad_kwargs.setdefault('is_pad', True)
        pad_kwargs.setdefault('pad_type', crop.CENTER)
        self.crop = crop.Center(**pad_kwargs)


class RuderLetterBox(Complex):
    """proportion resize, center crop without pad"""

    def __init__(self, interpolation=0, max_ratio=None, **ig_kwargs):
        self.resize = Proportion(choice_type=SHORTEST, interpolation=interpolation, max_ratio=max_ratio)  # choice the short scale factor
        self.crop = crop.Center(is_pad=False)


class Jitter(Complex):
    """random resize, random crop with pad
    See Also `torchvision.transforms.RandomResizedCrop`"""

    def __init__(self, size_range=None, interpolation=0, **pad_kwargs):
        self.resize = RandomProportion(choice_type=LONGEST, size_range=size_range, interpolation=interpolation)
        self.crop = crop.Random(is_pad=True, pad_type=crop.CENTER, **pad_kwargs)


class RuderJitter(Complex):
    """random resize, random crop without pad"""

    def __init__(self, size_range=None, interpolation=0, **ig_kwargs):
        self.resize = RandomProportion(choice_type=SHORTEST, size_range=size_range, interpolation=interpolation)
        self.crop = crop.Random(is_pad=False)


class BatchLetterBox:
    def __init__(self, interpolation=0, choice_edge=SHORTEST):
        self.interpolation = interpolation_mode[interpolation]
        self.choice_edge = choice_edge
        self.aug = LetterBox(interpolation=interpolation)

    def __call__(self, image_list, bboxes_list=None, classes_list=None, **kwargs):
        image_sizes = [image[:2] for image in image_list]

        if self.choice_edge == SHORTEST:
            dst = np.min(image_sizes)
        elif self.choice_edge == LONGEST:
            dst = np.max(image_sizes)
        elif self.choice_edge == AUTO:
            dst = np.mean(image_sizes)
        else:
            raise ValueError(f'dont support {self.choice_edge = }')

        rets = []

        for i in range(len(image_list)):
            ret = dict(
                image=image_list[i],
                bboxes=bboxes_list[i],
                classes=classes_list[i],
                dst=dst
            )

            for k, v in kwargs.items():
                ret[k] = v

            ret.update(self.aug(**ret))
            rets.append(ret)

        return rets
