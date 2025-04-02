import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

from utils import os_lib, converter
from .base import DataLoader, DataSaver, DataRegister, get_image, save_image, DatasetGenerator


class Loader(DataLoader):
    """a simple loader for image classification, image generation and so on

    Data structure:
        .
        └── [task]
            ├── xxx.json
            └── xxx.png

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.SimpleImage import DataRegister, Loader, DataVisualizer

            data_dir = 'data/xxx'
            loader = Loader(data_dir)
            data = loader(generator=True, image_type=DataRegister.ARRAY)

            # visual train dataset
            DataVisualizer(f'{data_dir}/visuals', verbose=False)(data[0])

    """
    default_set_type = [DataRegister.MIX]
    label_suffix = '.json'
    loader = os_lib.Loader(verbose=False)

    def _call(self, task='original', **gen_kwargs):
        gen_func = os_lib.find_all_suffixes_files(f'{self.data_dir}/{task}', self.image_suffixes)
        return self.gen_data(gen_func, task=task, **gen_kwargs)

    def get_ret(self, fp: Path, image_type=DataRegister.PATH, return_label=False, **kwargs) -> dict:
        image_path = os.path.abspath(fp)
        image = get_image(image_path, image_type)

        label = {}
        if return_label:
            label_path = str(fp).replace(fp.suffix, self.label_suffix)
            label = self.loader.load_json(label_path)

        return dict(
            _id=fp.name,
            image=image,
            **label
        )


class ZipLoader(DataLoader):
    default_set_type = [DataRegister.MIX]
    label_suffix = '.json'

    def _call(self, task='original', **gen_kwargs):
        zip_file = ZipFile(f'{self.data_dir}/{task}.zip', 'r')
        gen_func = zip_file.namelist()
        return self.gen_data(gen_func, task=task, zip_file=zip_file, **gen_kwargs)

    def get_ret(self, obj, image_type=DataRegister.PATH, zip_file=None, return_label=False, **kwargs) -> dict:
        image = zip_file.open(obj).read()
        image = converter.DataConvert.bytes_to_image(image)

        label = {}
        if return_label:
            label_path = str(obj).replace(self.image_suffix, self.label_suffix)
            label = json.loads(str(zip_file.open(label_path).read()))

        return dict(
            _id=obj,
            image=image,
            **label
        )


class Saver(DataSaver):
    label_suffix = '.json'
    saver = os_lib.Saver(verbose=False)

    def mkdirs(self, task='original', **kwargs):
        os_lib.mk_dir(f'{self.data_dir}/{task}')

    def _call(self, iter_data, **gen_kwargs):
        return self.gen_data(iter_data, **gen_kwargs)

    def parse_ret(self, ret, image_type=DataRegister.PATH, task='original', save_label=False, **get_kwargs):
        image = ret['image']
        _id = ret['_id']
        image_path = f'{self.data_dir}/{task}/{_id}'
        save_image(image, image_path, image_type)

        if save_label:
            ret = {k: v for k, v in ret if k != 'image'}
            label_path = image_path.replace(Path(image_path).suffix, self.label_suffix)
            self.saver.save_json(ret, label_path)


class ZipSaver(DataSaver):
    label_suffix = '.json'

    def mkdirs(self, **kwargs):
        os_lib.mk_dir(self.data_dir)

    def _call(self, iter_data, task='original', **gen_kwargs):
        f = ZipFile(f'{self.data_dir}/{task}.zip', 'w')
        return self.gen_data(iter_data, f=f, **gen_kwargs)

    def parse_ret(self, ret, sub_task='.', f=None, save_label=False, **get_kwargs):
        image = ret['image']
        _id = ret['_id']
        image = converter.DataConvert.image_to_bytes(image)
        image_path = f'{sub_task}/{_id}'
        f.writestr(image_path, image)

        if save_label:
            ret = {k: v for k, v in ret if k != 'image'}
            label_path = image_path.replace(Path(image_path).suffix, self.label_suffix)
            f.writestr(label_path, json.dumps(ret))


class Generator(DatasetGenerator):
    label_suffix = '.json'

    def gen_sets(self, label_dirs=(), image_dirs=(), save_dir='', task='',
                 id_distinguish='', id_sort=False,
                 set_names=('train', 'val'), split_ratio=(0.8, 1), **kwargs):
        """

        Args:
            label_dirs: special dir or if image_dirs is set, use image_dirs, else use Generator.label_dir
            image_dirs: special dir or use Generator.image_dir, if label_dirs is set, ignore this param
            save_dir: special dir or use {Generator.data_dir}/image_sets/{task}
            task:
            id_distinguish:
                the image file name where having same id must not be split to different sets.
                e.g. id_distinguish = '_', the fmt of image file name would like '{id}_{sub_id}.png'
            id_sort: if id_distinguish is set, True for sorting by sub_ids and sub_ids must be int type
            set_names: save txt name
            split_ratio:
                split ratio for each set, the shape must apply for set_names
                if id_distinguish is set, the ration is num of ids not files

        Data structure:
            .
            └── [task]
                ├── xxx.json
                └── xxx.png

        Usage:
            .. code-block:: python

                from data_parse.cv_data_parse.YoloV5 import Generator

                # single data dir
                data_dir = 'data/yolov5'
                gen = Generator(
                    data_dir=data_dir,
                    image_dir=f'{data_dir}/images/1',
                    label_dir=f'{data_dir}/labels/1',
                )
                gen.gen_sets(set_task='1')

                # multi data dir
                gen = Generator()
                gen.gen_sets(
                    image_dir=('data/yolov5_0/images', 'data/yolov5_1/images'),
                    # label_dirs=('data/yolov5_0/labels', 'data/yolov5_1/labels'),
                    save_dir='data/yolov5_0_1/image_sets'
                )
        """
        save_dir = save_dir or f'{self.data_dir}/{task}'
        os_lib.mk_dir(save_dir)

        if not label_dirs and not image_dirs and self.label_dir:
            label_dirs = [self.label_dir]

        image_dirs = image_dirs or [self.image_dir]

        data = []
        idx = []

        if label_dirs:
            for label_dir in label_dirs:
                tmp = list(os_lib.find_all_suffixes_files(label_dir, [self.label_suffix]))
                tmp = [x for x in tmp if self.filter_func(x)]

                if id_distinguish:
                    idx += [i.stem for i in tmp]

                tmp = [os.path.abspath(str(i).replace(i.suffix, self.image_suffixes[0])) for i in tmp]
                data += tmp

        else:
            for image_dir in image_dirs:
                tmp = list(os_lib.find_all_suffixes_files(image_dir, self.image_suffixes))
                tmp = [x for x in tmp if self.filter_func(x)]

                if id_distinguish:
                    idx += [i.stem for i in tmp]

                tmp = [os.path.abspath(i) for i in tmp]
                data += tmp

        self._gen_sets(data, idx, id_distinguish, id_sort, set_names, split_ratio, save_dir=save_dir)

    def save_func(self, iter_data, candidate_ids, set_name, save_dir='', **kwargs):
        root = f'{save_dir}/{set_name}'
        os_lib.mk_dir(root)

        for candidate_id in candidate_ids:
            fp = iter_data[candidate_id]
            is_label_file = False
            if self.label_suffix in fp:
                is_label_file = True

            if is_label_file:
                image_fp = fp.replace(self.label_suffix, self.image_suffix)
                label_fp = fp
            else:
                image_fp = fp
                label_fp = fp.replace(Path(fp).suffix, self.label_suffix)

            new_image_fp = f'{root}/{Path(image_fp).name}'
            new_label_fp = f'{root}/{Path(label_fp).name}'
            shutil.copy(image_fp, new_image_fp)
            shutil.copy(label_fp, new_label_fp)
