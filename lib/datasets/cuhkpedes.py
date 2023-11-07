from pathlib import Path 

from typing import Union 
import pandas as pd 

class CUHKPEDES:
    def __init__(self, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)

        self.dataset = path 
        self.img_dir = path / 'imgs'

        self.annotation_path = path / 'reid_raw.json'

        df = pd.read_json(self.annotation_path).groupby('split')
        self.df_train, self.train_id_container = self.process_annotations(df.get_group('train'))
        self.df_test, self.test_id_container = self.process_annotations(df.get_group('test'))
        self.df_val, self.val_id_container = self.process_annotations(df.get_group('val'))
        

    def process_annotations(self, df: pd.DataFrame):
        df = df.copy()

        df['id'] = df['id'] - 1
        df['file_path'] = df['file_path'].apply(lambda x: self.img_dir / x)
        df = df.explode('captions').reset_index()

        df = df.rename(columns={'index': 'image_id', 'id': 'pid', 'captions': 'caption', 'file_path': 'img_path'})
        df = df[['image_id', 'pid', 'caption', 'img_path']]

        df = df[df['img_path'].map(lambda x: x.exists())]
        
        return df, set(df['pid'])

