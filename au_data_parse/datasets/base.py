import os
import pickle
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from utils import os_lib, visualize
from typing import List
from numbers import Number
from ... import DataRegister, DataLoader, DataSaver, DatasetGenerator


def get_audio(obj: str, audio_type):
    if audio_type == DataRegister.PATH:
        audio = obj
    elif audio_type == DataRegister.ARRAY:
        audio = os_lib.loader.load_audio(obj)
    elif audio_type == DataRegister.NPY:
        audio = np.load(obj)
    else:
        raise ValueError(f'Unknown input {audio_type = }')

    return audio


def save_audio(obj, save_path, audio_type):
    if audio_type == DataRegister.PATH:
        if os.path.abspath(obj) != os.path.abspath(save_path):
            shutil.copy(obj, save_path)
    else:
        raise ValueError(f'Unknown input {audio_type = }')


class audioClsDict:
    # audio file name, e.g. 'xxx.png'
    _id: str

    # if `audio_type = DataRegister.PATH`,, return a string
    # if `audio_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    audio: str or np.ndarray

    # usually an int number giving the pic class
    _class: Number


class ObjDetDict:
    # audio file name, e.g. 'xxx.png'
    _id: str

    # if `audio_type = DataRegister.PATH`,, return a string
    # if `audio_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    audio: str or np.ndarray

    # a np.ndarray with shape of (n_obj, 4), 4 gives [top_left_x, top_left_y, right_down_x, right_down_y] usually
    bboxes: np.ndarray

    # a np.ndarray with shape of (n_obj, )
    classes: np.ndarray


class audioSegDict:
    # audio file name, e.g. 'xxx.png'
    _id: str

    # if `audio_type = DataRegister.PATH`,, return a string
    # if `audio_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    audio: str or np.ndarray

    # if `audio_type = DataRegister.PATH`,, return a string
    # if `audio_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    label_mask: str or np.ndarray


class DataLoader(DataLoader):
    """
    for implementation, usually override the following methods:
        _call(): called by `load()`, which preparing the data, and returning an iterable function warped by `gen_data()`
        get_ret(): called by `gen_data()`, which is logic of parsing the data, and return a dict of result
    """
    default_audio_type = DataRegister.PATH
    audio_suffix = '.wav'
    audio_suffixes = os_lib.suffixes_dict['audio']

    def load(self, audio_type=None, **load_kwargs):
        """
        Args:
            audio_type(DataRegister): `DataRegister.PATH` or `DataRegister.ARRAY`
                PATH -> a str of audio abs path
                ARRAY -> a np.ndarray of audio, read from cv2, as (h, w, c)
            load_kwargs:
                see also `_call` function to get more details of load_kwargs
                see also 'gen_data' function to get more details of gen_kwargs
                see also 'get_ret' function to get more details of get_kwargs

        """
        audio_type = audio_type or self.default_audio_type
        return super().load(audio_type=audio_type, **load_kwargs)


class DatasetGenerator(DatasetGenerator):
    """generate datasets for training, testing and valuating"""
    audio_suffix = '.wav'
    audio_suffixes = os_lib.suffixes_dict['audio']
