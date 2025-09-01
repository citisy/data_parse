from data_parse.cv_data_parse.data_augmentation import crop


class Pad:
    def __init__(self, **pad_kwargs):
        self.pad = crop.Pad(**pad_kwargs)
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__

    def get_add_params(self, dst, w_list, h_list):
        if dst is None:
            dst_w = max(w_list)
            dst_h = max(h_list)
        elif isinstance(dst, int):
            dst_w = dst
            dst_h = dst
        else:
            dst_w, dst_h = dst

        add_params = [self.pad.get_params(dst_w, dst_h, w, h) for w, h in zip(w_list, h_list)]
        return {self.name: add_params}

    def parse_add_params(self, ret):
        return ret[self.name]

    def __call__(self, image_list, dst=None, bboxes_list=None, **kwargs):
        w_list, h_list = [], []
        for image in image_list:
            h, w = image.shape[:2]
            w_list.append(w)
            h_list.append(h)

        add_params = self.get_add_params(dst, w_list, h_list)

        return {
            'image_list': self.apply_image_list(image_list, add_params),
            **add_params
        }

    def apply_image_list(self, image_list, ret):
        pad_infos = self.parse_add_params(ret)
        new_image_list = []
        for image, pad_info in zip(image_list, pad_infos):
            image = self.pad.apply_image(image, {self.pad.name: pad_info})
            new_image_list.append(image)
        return new_image_list

