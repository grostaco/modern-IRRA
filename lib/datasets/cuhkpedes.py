from pathlib import Path 

from typing import Union 
import pandas as pd 
from datasets import Dataset, DatasetDict, Features, Value, Image 
from PIL import Image as PImage

# class CUHKPEDES(GeneratorBasedBuilder):
#     def _info(self) -> DatasetInfo:
#         return DatasetInfo(
#             features=Features({
#                 'pids': Value('int64'),
#                 'image': Image,
#                 'caption': Value('string')
#             })
#         )
#     def _split_generators(self, dl_manager: DownloadManager):
#         root = Path('./data/CUHK-PEDES')
#         img_dir = root / 'imgs'
#         annotation_path = root / 'reid_raw.json'

#         df = pd.read_json(annotation_path).groupby('split')

#         return [
#             SplitGenerator('train', gen_kwargs={'df': df.get_group('train'),
#                                                 'img_dir': img_dir}),
#             SplitGenerator('test', gen_kwargs={'df': df.get_group('test'),
#                                                 'img_dir': img_dir}),
#             SplitGenerator('validation', gen_kwargs={'df': df.get_group('val'),
#                                                 'img_dir': img_dir}),
#         ]
    
#     def _generate_examples(self, df: pd.DataFrame, img_dir: str):
#         df = df.copy()
#         df['id'] = df['id'] - 1
#         df['file_path'] = df['file_path'].apply(lambda x: img_dir / x)
#         df = df.explode('captions').reset_index()

#         df = df.rename(columns={'index': 'image_id', 'id': 'pid', 'captions': 'caption', 'file_path': 'img_path'})
#         df = df[['image_id', 'pid', 'caption', 'img_path']]

#         for idx, row in df.iterrows():
#             yield idx, {
#                 'pids': row['pid'],
#                 'image': PImage.open(row['file_path']).convert('RGB'),
#                 'caption': row['caption']
#             } 

class CUHKPEDES:
    def __init__(self, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)

        self.dataset = path 
        self.img_dir = path / 'imgs'

        self.annotation_path = path / 'reid_raw.json'

        df = pd.read_json(self.annotation_path).groupby('split')
        self.dataset_train, self.train_id_container = self.process_annotations(df.get_group('train'))
        self.dataset_test, self.test_id_container = self.process_annotations(df.get_group('test'))
        self.dataset_val, self.val_id_container = self.process_annotations(df.get_group('val'))

        self.dataset = DatasetDict({
            'train': self.dataset_train,
            'test': self.dataset_test,
            'validation': self.dataset_val
        })
        

    def process_annotations(self, df: pd.DataFrame):
        df = df.copy()

        df['id'] = df['id'] - 1
        df['file_path'] = df['file_path'].apply(lambda x: (self.img_dir / x).as_posix())
        df = df.explode('captions').reset_index()

        df = df.rename(columns={'id': 'pid', 'captions': 'caption', 'file_path': 'image'})
        df = df[['pid', 'caption', 'image']]
        
        return Dataset.from_pandas(df, features=Features({
            'pid': Value('uint64'),
            'caption': Value('string'),
            'image': Image()
        })), set(df['pid'])



# class MLMDataset(GeneratorBasedBuilder):
#     def _info(self) -> DatasetInfo:
#         return DatasetInfo(
#             features=Features({
#                 'pids': Value('int64'),
#                 'image': Image,
#                 'caption': Value('str')
#             })
#         )
#     def _split_generators(self, dl_manager: DownloadManager):
#         df = 
    
#     def _generate_examples(self, **kwargs):
#         return super()._generate_examples(**kwargs)