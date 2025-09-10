import pickle
from collections import defaultdict
from enum import Enum
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import os_lib, visualize


class DataRegister(Enum):
    place_holder = None

    # set type
    MIX = 'mix'
    FULL = 'full'
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
    DEV = 'dev'
    TRAIN_VAL = 'trainval'

    # image type
    PATH = 1
    ARRAY = 2
    GRAY_ARRAY = 2.1
    NPY = 2.2
    BASE64 = 3
    ZIP = 4
    URL = 5


class DataLoader:
    """
    for implementation, usually override the following methods:
        _call(): called by `load()`, which preparing the data, and returning an iterable function warped by `gen_data()`
        get_ret(): called by `gen_data()`, which is logic of parsing the data, and return a dict of result
    """
    default_set_type = [DataRegister.TRAIN, DataRegister.TEST]
    default_data_type = DataRegister.FULL
    classes = []
    dataset_info: dict

    def __init__(self, data_dir, verbose=True, stdout_method=print, **kwargs):
        self.data_dir = data_dir
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()
        self.__dict__.update(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.load(*args, **kwargs)

    def load(self, set_type=None, generator=True, **load_kwargs):
        """
        Args:
            set_type(list or DataRegister): a DataRegister type or a list of them
                FULL -> DataLoader.default_set_type
                other set_type -> [set_type]
            generator(bool):
                return a generator if True else a list
            load_kwargs:
                see also `_call` function to get more details of load_kwargs
                see also 'gen_data' function to get more details of gen_kwargs
                see also 'get_ret' function to get more details of get_kwargs

        Returns:
            a list apply for set_type
            e.g.
                set_type=DataRegister.TRAIN, return a list of [DataRegister.TRAIN]
                set_type=[DataRegister.TRAIN, DataRegister.TEST], return a list of them
        """
        set_type = set_type or self.default_data_type

        if set_type == DataRegister.FULL:
            set_types = self.default_set_type
        elif isinstance(set_type, list):
            set_types = set_type
        elif isinstance(set_type, DataRegister):
            set_types = [set_type]
        else:
            raise ValueError(f'Unknown input {set_type = }')

        r = []
        for set_type in set_types:
            pbar = self._call(set_type=set_type, **load_kwargs)

            if self.verbose:
                pbar = tqdm(pbar, desc=visualize.TextVisualize.highlight_str(f'Load {set_type.value} dataset'))

            if generator:
                r.append(pbar)
            else:
                r.append(list(pbar))

        return r

    def _call(self, **gen_kwargs):
        """called by `load()`, which preparing the data, and returning an iterable function warped by `gen_data()`
        usually, defined where can find the data file, or how to read the file to memory, etc.
        for implementation, used like the following scripts:

            def _call(self, **gen_kwargs):
                gen_func = iter(...)
                return self.gen_data(gen_func, **gen_kwargs)

        Args:
            gen_kwargs:
                kwargs for `self.gen_data()`, to see the funtion to get more info

        Returns:
            an iterator which yield a dict of data

        """
        raise NotImplementedError

    def gen_data(self, gen_func, max_size=None, **get_kwargs):
        """

        Args:
            gen_func:
                an iterator
            max_size:
                num of loaded data
            **get_kwargs:
                kwargs for `self.get_ret()`, to see the funtion to get more info

        Yields
            a dict of result data

        """
        max_size = max_size or float('inf')
        i = 0
        for obj in gen_func:
            if i >= max_size:
                break

            obj = self.on_start_convert(obj)

            if not self.on_start_filter(obj):
                continue

            ret = self.get_ret(obj, **get_kwargs)
            if not ret:
                continue

            ret = self.on_end_convert(ret)

            if not self.on_end_filter(ret):
                continue

            i += 1
            yield ret

        if hasattr(gen_func, 'close'):
            gen_func.close()

    def get_ret(self, obj, **kwargs) -> dict:
        """called by `gen_data()`, which is logic of parsing the data, and return a dict of result"""
        raise NotImplementedError

    def load_cache(self, save_name):
        with open(f'{self.data_dir}/cache/{save_name}.pkl', 'rb') as f:
            data = pickle.load(f)

        return data

    def on_start_convert(self, obj):
        return obj

    def on_end_convert(self, ret):
        return ret

    def on_start_filter(self, obj):
        return True

    def on_end_filter(self, ret):
        return True


class DataSaver:
    def __init__(self, data_dir, verbose=True, stdout_method=print):
        self.data_dir = data_dir
        self.default_set_type = [DataRegister.TRAIN, DataRegister.TEST]
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()

    def __call__(self, *args, **kwargs):
        return self.save(*args, **kwargs)

    def save(self, data, set_type=DataRegister.FULL, **save_kwargs):
        """

        Args:
            data(list): a list apply for set_type
                See Also return of `DataLoader.__call__`
            set_type(list or DataRegister): a DataRegister type or a list of them
                ALL -> DataLoader.default_set_type
                other set_type -> [set_type]
        """
        if set_type == DataRegister.FULL:
            set_types = self.default_set_type
        elif isinstance(set_type, list):
            set_types = set_type
        elif isinstance(set_type, DataRegister):
            set_types = [set_type]
        else:
            raise ValueError(f'Unknown input {set_type = }')

        self.mkdirs(set_types=set_types, **save_kwargs)

        for iter_data, set_type in zip(data, set_types):
            if self.verbose:
                iter_data = tqdm(iter_data, desc=visualize.TextVisualize.highlight_str(f'Save {set_type.value} dataset'))

            self._call(iter_data, set_type=set_type, **save_kwargs)

    def mkdirs(self, **kwargs):
        os_lib.mk_dir(self.data_dir)

    def _call(self, iter_data, **gen_kwargs):
        raise NotImplementedError

    def gen_data(self, gen_func, max_size=float('inf'), **get_kwargs):
        """

        Args:
            gen_func:
            max_size: num of loaded data
            **get_kwargs:
                see also `parse_ret` function to get more details of get_kwargs

        Yields
            a dict of result data

        """
        i = 0
        for ret in gen_func:
            if i >= max_size:
                break

            ret = self.on_start_convert(ret)

            if not self.on_start_filter(ret):
                continue

            self.parse_ret(ret, **get_kwargs)
            i += 1

    def parse_ret(self, ret, image_type=DataRegister.PATH, **get_kwargs):
        raise NotImplementedError

    def save_cache(self, data, save_name):
        save_dir = f'{self.data_dir}/cache'
        os_lib.mk_dir(save_dir)

        with open(f'{save_dir}/{save_name}.pkl', 'wb') as f:
            pickle.dump(data, f)

    def on_start_convert(self, ret):
        return ret

    def on_start_filter(self, ret):
        return True


class DatasetGenerator:
    """generate datasets for training, testing and valuating"""
    image_suffix = '.jpg'
    image_suffixes = os_lib.suffixes_dict['img']

    def __init__(self, data_dir=None, image_dir=None, label_dir=None, verbose=True, stdout_method=print):
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()

    def gen_sets(self, *args, **kwargs):
        """please implement this function, and call _gen_sets finally"""
        return self._gen_sets(*args, **kwargs)

    def _gen_sets(
            self,
            iter_data, idx=None, id_distinguish='', id_sort=False,
            set_names=('train', 'test'), split_ratio=(0.8, 1.), **save_kwargs
    ):
        """

        Args:
            iter_data (List[Any]):
            idx (List[str]): be used for sorting
            id_distinguish:
                the image file name where having same id must not be split to different sets.
                e.g. id_distinguish = '_', the fmt of image file name would like '{id}_{sub_id}.png'
            id_sort: if id_distinguish is set, True for sorting by sub_ids and sub_ids must be int type
            save_dir: save_dir
            set_names: save txt name
            split_ratio:
                split ratio for each set, the shape must apply for set_names
                if id_distinguish is set, the ration is num of ids not files
            **save_kwargs:

        Returns:

        """
        if id_distinguish:
            tmp = defaultdict(list)
            for i, (d, _idx) in enumerate(zip(iter_data, idx)):
                stem = Path(_idx).stem
                _ = stem.split(id_distinguish)
                a, b = id_distinguish.join(_[:-1]), _[-1]
                tmp[a].append([i, b])

            if id_sort:
                for k, v in tmp.items():
                    # convert str to int
                    for vv in v:
                        try:
                            vv[1] = int(vv[1])
                        except ValueError:
                            pass

                    tmp[k] = sorted(v, key=lambda x: x[1])

            ids = list(tmp.keys())
            np.random.shuffle(ids)

            i = 0
            for j, set_name in zip(split_ratio, set_names):
                j = int(j * len(ids))
                candidate_ids = []
                for k in ids[i:j]:
                    candidate_ids += [vv[0] for vv in tmp[k]]

                if not id_sort:
                    np.random.shuffle(candidate_ids)

                self.save_func(iter_data, candidate_ids, set_name, **save_kwargs)

                i = j

        else:
            ids = list(range(len(iter_data)))
            np.random.shuffle(ids)

            i = 0
            for j, set_name in zip(split_ratio, set_names):
                j = int(j * len(iter_data))
                candidate_ids = ids[i: j]
                self.save_func(iter_data, candidate_ids, set_name, **save_kwargs)

                i = j

    def save_func(self, iter_data, candidate_ids, set_name, **kwargs):
        raise NotImplementedError

    def filter_func(self, x):
        return True
