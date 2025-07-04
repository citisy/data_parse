"""change the channels of image, it is best to apply them in final"""
import numpy as np
import cv2


class HWC2CHW:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    @staticmethod
    def apply_image(image, *args):
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
            image = np.ascontiguousarray(image)
        return image

    def restore(self, ret):
        if 'image' in ret:
            ret['image'] = CHW2HWC.apply_image(ret['image'])
        return ret


class CHW2HWC:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    @staticmethod
    def apply_image(image, *args):
        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
            image = np.ascontiguousarray(image)
        return image

    def restore(self, ret):
        ret['image'] = HWC2CHW.apply_image(ret['image'])
        return ret


class BGR2RGB:
    """
    cv2 -> bgr
    PIL.Image -> rgb
    """
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        image = image.copy()
        if self.axis == 0:
            image = image[::-1]
        elif self.axis == -1:
            image[..., :] = image[..., ::-1]
        return image


class RGB2BGR(BGR2RGB):
    """same op to BGR2RGB"""


class Gray2BGR:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image


class BGR2Gray:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, None]
        return image


class Keep3Dims:
    """input an array which have any(2, 3 or 4) dims, output an array which have 3-dims"""

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        shape = image.shape
        if len(shape) == 2:
            image = image[:, :, None]
        elif len(shape) == 3:
            pass
        else:
            raise ValueError(f'Do not support shape = {shape}')
        return image


class Keep3Channels:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        c = image.shape[-1]
        if c == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif c == 3:
            pass
        elif c == 4:
            color = image[:, :, 0:3].astype(np.float32)
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            image = color * alpha + 255.0 * (1.0 - alpha)
            image = image.clip(0, 255).astype(np.uint8)
        else:
            raise ValueError(f'Do not support channels = {c}')
        return image


class AddXY:
    def __init__(self, axis=-1):
        # y = -cot(x)
        # y in (-1.9, 1.4) where x in (0.5, 2.5)
        self.func = lambda x: (-1 / np.tan(np.linspace(0.5, 2.5, x)) + 1.9) / 3.3 * 255
        self.axis = axis

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        if self.axis in (-1, 3):
            h, w = image.shape[:2]
        elif self.axis == 0:
            h, w = image.shape[1:]
        else:
            raise ValueError(f'Do not support axis = {self.axis}')

        x, y = self.func(w), self.func(h)
        xv, yv = np.meshgrid(x, y)  # (h, w)
        xv = np.expand_dims(xv, axis=self.axis).astype(np.uint8)
        yv = np.expand_dims(yv, axis=self.axis).astype(np.uint8)
        img_xy = np.concatenate((image, xv, yv), axis=self.axis)
        return img_xy

    def restore(self, ret):
        if self.axis in (-1, 3):
            ret['image'] = ret['image'][:, :, :3]
        elif self.axis == 0:
            ret['image'] = ret['image'][:3]

        return ret
