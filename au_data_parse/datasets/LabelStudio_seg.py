import json
import os
from pathlib import Path

from utils import os_lib
from .base import DataLoader, DataRegister, DataSaver, DatasetGenerator, get_audio, save_audio


class Loader(DataLoader):
    """https://labelstud.io/

    Data structure:
        .
        ├── audios
        └── [set_task].json

    """

    default_set_type = [DataRegister.MIX]

    def _call(self, set_task='label_studio', **kwargs):
        with open(f'{self.data_dir}/{set_task}.json', 'r', encoding='utf8') as f:
            gen_func = json.load(f)
        return self.gen_data(gen_func, set_task=set_task, **kwargs)

    def get_ret(self, js, audio_type=DataRegister.PATH, set_task='label_studio', task='annotations', **kwargs) -> dict:
        audio_path = Path(js['data']['audio'])
        audio_root = str(audio_path.parent)
        audio_name = audio_path.name
        if '-' in audio_name:
            sub_id, _id = audio_name.split('-', 1)
        else:
            sub_id, _id = '', audio_name
        audio_path = f'{self.data_dir}/audios/{set_task}/{_id}'
        audio_path = os.path.abspath(audio_path)
        audio = get_audio(audio_path, audio_type)

        classes = []
        timestamps = []
        for a in js[task]:
            for r in a['result']:
                v = r['value']
                if 'labels' in v:
                    classes.append(v['labels'][0])
                    timestamps.append([v['start'], v['end']])

        return dict(
            _id=_id,
            sub_id=sub_id,
            audio_root=audio_root,
            audio=audio,
            classes=classes,
            timestamps=timestamps
        )


class Saver(DataSaver):
    def _call(self, iter_data, audio_type=DataRegister.PATH, set_task='label_studio', task='annotations', cls_alias=None, is_save_audio=True, **kwargs):
        rets = []
        for dic in iter_data:
            _id = dic['_id']
            sub_id = dic['sub_id']
            audio_root = dic['audio_root']
            audio = dic['audio']
            if is_save_audio:
                audio_path = f'{self.data_dir}/audios/{set_task}/{_id}'
                save_audio(audio, audio_path, audio_type)

            timestamps = dic['timestamps']
            classes = dic['classes']
            if cls_alias:
                classes = [cls_alias[i] for i in classes]

            result = []
            for cls, timestamp in zip(classes, timestamps):
                result.append(dict(
                    value=dict(
                        labels=[cls],
                        start=timestamp[0],
                        end=timestamp[1]
                    ),
                    type='labels'
                ))

            if sub_id:
                audio = f'{audio_root}/{sub_id}-{_id}'
            else:
                audio = f'{audio_root}/{_id}'
            rets.append({
                'data': dict(audio=audio),
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