"""change the pixel of image"""
import numbers

import cv2
import numpy as np

from . import RandomChoice


class MinMax:
    """[0, 255] to [0, 1]"""

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        return image / 255.

    def restore(self, ret):
        if 'image' in ret:
            ret['image'] = ret['image'] * 255
        return ret


class Clip:
    """all falls in [a_min, a_max]"""
    def __init__(self, a_min=0, a_max=255):
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        return np.clip(image, self.a_min, self.a_max)


class GaussNoise:
    """添加高斯噪声

    Args:
        mean: 高斯分布的均值
        sigma: 高斯分布的标准差
    """

    def __init__(self, mean=0, sigma=25):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        gauss = np.random.normal(self.mean, self.sigma, image.shape)
        image = (image + gauss).clip(min=0, max=255).astype(image.dtype)

        return image


class SaltNoise:
    """添加椒盐噪声

    Args:
        s_vs_p: 添加椒盐噪声的数目比例
        amount: 添加噪声图像像素的数目
    """

    def __init__(self, s_vs_p=0.5, amount=0.04):
        self.s_vs_p = s_vs_p
        self.amount = amount

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        image = np.copy(image)
        dtype = image.dtype
        num_salt = np.ceil(self.amount * image.size * self.s_vs_p)
        coords = tuple(np.random.randint(0, i - 1, int(num_salt)) for i in image.shape)
        image[coords] = 255
        num_pepper = np.ceil(self.amount * image.size * (1. - self.s_vs_p))
        coords = tuple(np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape)
        image[coords] = 0
        image = image.clip(min=0, max=255).astype(dtype)
        return image


class PoissonNoise:
    """添加泊松噪声"""

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        dtype = image.dtype
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        image = np.random.poisson(image * vals) / float(vals)
        image = image.clip(min=0, max=255).astype(dtype)

        return image


class SpeckleNoise:
    """添加斑点噪声"""

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        gauss = np.random.randn(*image.shape)
        noisy_image = image + image * gauss
        image = np.clip(noisy_image, a_min=0, a_max=255)
        return image


class Normalize:
    """Normalizes a ndarray image or image with mean and standard deviation.
    See Also `torchvision.transforms.Normalize`
    """

    def __init__(self, mean=None, std=None):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.mean = mean
        self.std = std

    def get_params(self, image):
        mean = self.mean if self.mean is not None else np.mean(image, axis=(0, 1))
        std = self.std if self.std is not None else np.std(image, axis=(0, 1))

        return mean, std

    def get_add_params(self, image):
        mean, std = self.get_params(image)
        return {self.name: dict(mean=mean, std=std)}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['mean'], info['std']

    def __call__(self, image, **kwargs):
        add_params = self.get_add_params(image)

        return {
            'image': self.apply_image(image, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        mean, std = self.parse_add_params(ret)
        return (image - mean) / std

    def restore(self, ret):
        if 'image' in ret:
            mean, std = self.parse_add_params(ret)
            image = ret['image']
            ret['image'] = image * std + mean
        return ret


class Pca:
    """after dimensionality reduction by pca
    add the random scale factor
    todo: still not fix yet"""

    def __init__(self, eigenvectors=None, eigen_values=None):
        self.eigenvectors = eigenvectors
        self.eigen_values = eigen_values

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        a = np.random.normal(0, 0.1)

        noisy_image = np.array(image, dtype=float)

        if self.eigenvectors is None:
            eigenvectors = []
            eigen_values = []

            for i in range(noisy_image.shape[-1]):
                x = noisy_image[:, :, i]

                for j in range(x.shape[0]):
                    n = np.mean(x[j])
                    s = np.std(x[j], ddof=1)
                    x[j] = (x[j] - n) / s

                cov = np.cov(x, rowvar=False)

                # todo: Complex return, is that wrong?
                eigen_value, eigen_vector = np.linalg.eig(cov)

                # arg = np.argsort(eigen_value)[::-1]
                # eigen_vector = eigen_vector[:, arg]
                # eigen_value = eigen_value[arg]

                eigen_values.append(eigen_value)
                eigenvectors.append(eigen_vector)

        for i in range(image.shape[-1]):
            noisy_image[:, :, i] = noisy_image[:, :, i] @ (self.eigenvectors[i].T * self.eigen_values[i] * a).T

        return noisy_image


class AdjustHsv:
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return image


class AdjustBrightness:
    """Adjusts brightness of an image.
    See Also `torchvision.transforms.functional.adjust_brightness` or `albumentations.adjust_brightness_torchvision`

    Args:
        offset (float):
            factor = 1 + offset
            factor in [0, inf], where 0 give black, 1 give original, 2 give brightness doubled

    """

    def __init__(self, offset=.5):
        self.offset = offset

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        factor = max(1 + self.offset, 0)
        table = np.array([i * factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
        image = cv2.LUT(image, table)
        return image


class AdjustContrast:
    """Adjusts contrast of an image.
    See Also `torchvision.transforms.functional.adjust_contrast` or `albumentations.adjust_contrast_torchvision`

    Args:
        offset (float):
            factor = 1 + offset
            factor in [0, inf], where 0 give solid gray, 1 give original, 2 give contrast doubled

    """

    def __init__(self, offset=.5):
        self.offset = offset

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        factor = max(1 + self.offset, 0)
        table = np.array([(i - 74) * factor + 74 for i in range(0, 256)]).clip(0, 255).astype('uint8')
        image = cv2.LUT(image, table)
        return image


class AdjustSaturation:
    """Adjusts color saturation of an image.
    See Also `torchvision.transforms.functional.adjust_saturation` or `albumentations.adjust_saturation_torchvision`

    Args:
        offset (float):
            factor = 1 + offset
            factor, where 0 give black and white, 1 give original, 2 give saturation doubled

    """

    def __init__(self, offset=.5):
        self.offset = offset

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        factor = 1 + self.offset
        dtype = image.dtype
        image = image.astype(np.float32)
        alpha = np.random.uniform(max(0, 1 - factor), 1 + factor)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image[..., np.newaxis]
        image = image * alpha + gray_image * (1 - alpha)
        image = image.clip(0, 255).astype(dtype)
        return image


class AdjustHue:
    """Adjusts hue of an image.
    See Also `torchvision.transforms.functional.adjust_hue` or `albumentations.adjust_hue_torchvision`

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        offset (float):
            factor = 1 + offset
            factor in [-0.5, 0.5], where
            -0.5 give complete reversal of hue channel in HSV space in positive direction respectively
            0 give no shift,
            0.5 give complete reversal of hue channel in HSV space in negative direction respectively

    Returns:
        np.array: Hue adjusted image.

    """

    def __init__(self, offset=.1):
        self.offset = offset

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        factor = min(max(1 + self.offset, -0.5), 0.5)

        dtype = image.dtype
        image = image.astype(np.uint8)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        h, s, v = cv2.split(hsv_image)

        alpha = np.random.uniform(factor, factor)
        h = h.astype(np.uint8)
        # uint8 addition take cares of rotation across boundaries
        with np.errstate(over="ignore"):
            h += np.uint8(alpha * 255)
        hsv_image = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR_FULL).astype(dtype)
        return image


class Jitter:
    """Randomly change the brightness, contrast, saturation and hue of an image.
    See Also `torchvision.transforms.ColorJitter` or `albumentations.ColorJitter`

    Args:
        apply_func: (brightness, contrast, saturation, hue)
        offsets (tuple): offset of apply_func

    """

    def __init__(
            self,
            offsets=(0.5, 0.5, 0.5, 0.1),
            apply_func=(AdjustBrightness, AdjustContrast, AdjustSaturation, AdjustHue)
    ):
        funcs = [func(offset) for offset, func in zip(offsets, apply_func)]
        self.apply = RandomChoice(funcs)

    def __call__(self, image, **kwargs):
        return self.apply(image=image)


class GaussianBlur:
    """see also `torchvision.transforms.GaussianBlur` or `albumentations.GaussianBlur`"""

    def __init__(self, ksize=None, sigma=(.5, .5)):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.ksize = ksize
        self.sigma = sigma

    def get_params(self, w, h):
        ksize = self.ksize
        if not ksize:
            ksize = min(w, h) // 8
            ksize = (ksize * 2) + 1

        return ksize

    def get_add_params(self, w, h):
        ksize = self.get_params(w, h)
        return {self.name: dict(ksize=ksize)}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['ksize']

    def __call__(self, image, **kwargs):
        h, w = image.shape[:2]
        add_params = self.get_add_params(w, h)

        return {
            'image': self.apply_image(image, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        ksize = self.parse_add_params(ret)
        return cv2.GaussianBlur(image, ksize, sigmaX=self.sigma[0], sigmaY=self.sigma[1])


class MotionBlur:
    """see also `albumentations.MotionBlur`"""

    def __init__(self, degree=12, angle=90):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.degree = degree
        self.angle = angle

    def get_params(self):
        return self.degree, self.angle

    def get_add_params(self):
        degree, angle = self.get_params()
        return {self.name: dict(
            degree=degree,
            angle=angle
        )}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['degree'], info['angle']

    def __call__(self, image, degree=None, angle=None, **kwargs):
        add_params = self.get_add_params()

        return {
            'image': self.apply_image(image, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        degree, angle = self.parse_add_params(ret)
        M = cv2.getRotationMatrix2D((degree // 2, degree // 2), angle, 1)
        motion_blur_kernel = np.zeros((degree, degree))
        motion_blur_kernel[degree // 2, :] = 1
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        image = cv2.filter2D(image, -1, motion_blur_kernel)
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image


class RandomMotionBlur(MotionBlur):
    def get_params(self):
        degree = int((np.random.beta(4, 4) - 0.5) * 2 * self.degree)

        if isinstance(self.angle, numbers.Number):
            angle = int(np.random.uniform(-self.angle, self.angle))
        else:
            angle = int(np.random.uniform(*self.angle))

        return degree, angle


class Erase:
    def __init__(self, scale=0.1, ratio=1., fill=None, max_iter=10):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.scale = scale
        self.ratio = ratio
        self.fill = fill if fill is not None else np.random.randint(100, 125, size=3)
        self.max_iter = max_iter

    def get_params(self):
        scales = [self.scale for _ in range(self.max_iter)]
        ratios = [self.ratio for _ in range(self.max_iter)]
        return scales, ratios

    def get_add_params(self):
        scales, ratios = self.get_params()
        return {self.name: dict(
            scales=scales,
            ratios=ratios
        )}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['scales'], info['ratios']

    def __call__(self, image, **kwargs):
        add_params = self.get_add_params()

        return {
            'image': self.apply_image(image, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        scales, ratios = self.parse_add_params(ret)
        h, w, c = image.shape
        area = h * w
        for scale, ratio in zip(scales, ratios):
            erase_area = area * scale

            _h = int(round(np.sqrt(erase_area * ratio)))
            _w = int(round(np.sqrt(erase_area / ratio)))

            if _w < w and _h < h:
                y1 = np.random.randint(0, h - _h)
                x1 = np.random.randint(0, w - _w)
                if c == 3:
                    image[y1:y1 + _h, x1:x1 + _w] = self.fill

                break

        return image


class RandomErasing(Erase):
    """https://arxiv.org/abs/1708.04896
    see also `torchvision.transforms.RandomErasing`"""

    def get_params(self):
        scales, ratios = [], []
        for _ in range(self.max_iter):
            if isinstance(self.scale, numbers.Number):
                scale = np.random.uniform(self.scale - 0.05, self.scale + 0.05)
            else:
                scale = np.random.uniform(*self.scale)

            if isinstance(self.ratio, numbers.Number):
                ratio = np.random.uniform(self.ratio - 0.5, self.ratio + 0.5)
            else:
                ratio = np.random.uniform(*self.ratio)
            scales.append(scale)
            ratios.append(ratio)

        return scales, ratios


class CutOut:
    """replace with color blocks
    https://arxiv.org/abs/1708.04552
    See Also `albumentations.Cutout`
    """

    def __init__(self, scales=None, iou_thres=0.6):
        self.scales = scales or [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # color blocks size fraction
        self.iou_thres = iou_thres

    def __call__(self, image, bboxes=None, classes=None, **kwargs):
        image, bboxes, classes = self.apply_image_bboxes_classes(image, bboxes, classes)

        return dict(
            image=image,
            bboxes=bboxes,
            classes=classes,
        )

    def apply_image_bboxes_classes(self, image, bboxes, classes):
        h, w = image.shape[:2]

        mask_bboxes = []

        for s in self.scales:
            mask_h = np.random.randint(1, int(h * s))  # create random masks
            mask_w = np.random.randint(1, int(w * s))

            # box
            xmin = max(0, np.random.randint(0, w) - mask_w // 2)
            ymin = max(0, np.random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            image[ymin:ymax, xmin:xmax] = [np.random.randint(64, 191) for _ in range(3)]

            if bboxes is not None and s > 0.03:
                mask_bboxes.append([xmin, ymin, xmax, ymax])

        if bboxes is not None:
            from metrics.object_detection import Iou

            iou = Iou.iou(bboxes, mask_bboxes)
            iou = np.max(iou, axis=1)
            idx = iou < self.iou_thres
            bboxes = bboxes[idx]

            if classes is not None:
                classes = classes[idx]

        return image, bboxes, classes


class AxisProjection:
    def __init__(self, axis=0):
        """

        Args:
            axis: (h, w, c),
                0 gives that all pixels are projected to x-axis
                1 gives that all pixels are projected to y-axis
                2 gives that all pixels are projected to channel-axis
        """
        self.axis = axis

    def __call__(self, image, **kwargs):
        return dict(
            mapping=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        x = image.mean(axis=self.axis)
        x = np.expand_dims(x, self.axis)
        x = np.repeat(x, image.shape[self.axis], axis=self.axis)
        return x


class BorderMap:
    """refer to:
    https://github.com/WenmuZhou/DBNet.pytorch/blob/master/data_loader/modules/make_border_map.py"""

    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7, join_type=1, epoch=None, total_epoch=0.):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        # JT_MITER = 2
        # JT_ROUND = 1
        # JT_SQUARE = 0
        self.join_type = join_type
        if epoch is not None and total_epoch != 0:
            self.shrink_ratio = self.shrink_ratio + 0.2 * epoch / total_epoch

    def __call__(self, image, segmentations, **kwargs):
        mapping, mask = self.apply_image_segmentations(image, segmentations)
        return dict(
            mapping=mapping,
            mask=mask
        )

    def apply_image_segmentations(self, image, segmentations):
        mapping = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        for points in segmentations:
            self.draw_border_map(points, mapping, mask)
        mapping = mapping * (self.thresh_max - self.thresh_min) + self.thresh_min

        return mapping, mask

    def draw_border_map(self, points, mapping, mask):
        import pyclipper
        from shapely.geometry import Polygon

        points = np.array(points)

        polygon_shape = Polygon(points)
        if polygon_shape.area <= 0:
            return
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in points]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, self.join_type, pyclipper.ET_CLOSEDPOLYGON)

        pad = padding.Execute(distance)
        if not pad:
            return

        padded_polygon = np.array(pad[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        points[:, 0] = points[:, 0] - xmin
        points[:, 1] = points[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width)
        )
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width)
        )

        distance_map = np.zeros((points.shape[0], height, width), dtype=np.float32)
        for i in range(points.shape[0]):
            j = (i + 1) % points.shape[0]
            absolute_distance = self._distance(xs, ys, points[i], points[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), mapping.shape[1] - 1)
        xmax_valid = min(max(0, xmax), mapping.shape[1] - 1)
        ymin_valid = min(max(0, ymin), mapping.shape[0] - 1)
        ymax_valid = min(max(0, ymax), mapping.shape[0] - 1)
        mapping[ymin_valid: ymax_valid + 1, xmin_valid: xmax_valid + 1] = np.fmax(
            1 - distance_map[ymin_valid - ymin: ymax_valid - ymax + height, xmin_valid - xmin: xmax_valid - xmax + width],
            mapping[ymin_valid: ymax_valid + 1, xmin_valid: xmax_valid + 1],
        )

    def _distance(self, xs, ys, point_1, point_2):
        """
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        """
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        return result


class ShrinkMap:
    """refer to:
    https://github.com/WenmuZhou/DBNet.pytorch/blob/master/data_loader/modules/make_shrink_map.py"""

    def __init__(self, min_size=8, min_area=64, shrink_ratio=0.4, join_type=1, epoch=None, total_epoch=0.):
        self.min_size = min_size
        self.min_area = min_area
        self.shrink_ratio = shrink_ratio
        # JT_MITER = 2
        # JT_ROUND = 1
        # JT_SQUARE = 0
        self.join_type = join_type
        if epoch is not None and total_epoch != 0:
            self.shrink_ratio = self.shrink_ratio + 0.2 * epoch / total_epoch

    def __call__(self, image, segmentations, **kwargs):
        mapping, mask = self.apply_image_segmentations(image, segmentations)
        return dict(
            mapping=mapping,
            mask=mask
        )

    def apply_image_segmentations(self, image, segmentations):
        import pyclipper
        from shapely.geometry import Polygon

        h, w = image.shape[:2]
        mapping = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for points in segmentations:
            height = max(points[:, 1]) - min(points[:, 1])
            width = max(points[:, 0]) - min(points[:, 0])
            polygon_shape = Polygon(points)
            if min(height, width) < self.min_size or polygon_shape.area < self.min_area:
                cv2.fillPoly(mask, points.astype(np.int32)[np.newaxis, :, :], 0)
            else:
                subject = [tuple(l) for l in points]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, self.join_type, pyclipper.ET_CLOSEDPOLYGON)
                shrunk = []

                # Increase the shrink ratio every time we get multiple polygon returned back
                possible_ratios = np.arange(self.shrink_ratio, 1, self.shrink_ratio)
                np.append(possible_ratios, 1)
                for ratio in possible_ratios:
                    distance = polygon_shape.area * (1 - np.power(ratio, 2)) / polygon_shape.length
                    shrunk = padding.Execute(-distance)
                    if len(shrunk) == 1:
                        break

                if not shrunk:
                    cv2.fillPoly(mask, points.astype(np.int32)[np.newaxis, :, :], 0)
                    continue

                for each_shrink in shrunk:
                    shrink = np.array(each_shrink).reshape(-1, 2)
                    cv2.fillPoly(mapping, [shrink.astype(np.int32)], 1)

        return mapping, mask

