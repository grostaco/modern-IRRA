from datasets.download.download_manager import DownloadManager
from datasets.info import DatasetInfo
import torch.utils.data 
import pandas as pd 

from typing import Any, Callable, Union, List, Dict
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from transformers.data.data_collator import DataCollatorForLanguageModeling
from PIL import Image 

from datasets import GeneratorBasedBuilder, Features, Value, Image


# class MLMDataset(torch.utils.data.Dataset):
#     def __init__(self, df: pd.DataFrame, tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer], transforms: Callable[[Any], torch.Tensor]):
#         self.df = df 
#         self.transforms = transforms 

#         self.tokenizer = tokenizer
#         self.mask_id = tokenizer.mask_token_id 
#         self.collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=.15)
#         self.processor = CLIPImageProcessor()

#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx: int):
#         row = self.df.iloc[idx]
#         pid, image_id, img_path, caption = row['pid'], row['image_id'], row['img_path'], row['caption']

#         img = Image.open(img_path).convert('RGB')
#         #img = self.transforms(img)
#         img = self.processor(img, return_tensors='pt')['pixel_values'][0]

#         caption_tokens = self.tokenizer(caption, max_length=77, truncation=True, padding='max_length', return_tensors='pt')['input_ids']
        
#         mlm_tokens, mlm_labels = self.collator.torch_mask_tokens(caption_tokens)

#         # Try to mask at least one
#         if (mlm_labels == -100).all() and len(mlm_tokens) > 1:
#             mlm_labels[1] = mlm_tokens[1]
#             mlm_tokens[1] = self.mask_id

#         return {
#             'pids': pid,
#             'image_ids': image_id,
#             'images': img,
#             'caption_ids': caption_tokens.squeeze(0), # type: ignore
#             'mlm_ids': mlm_tokens.squeeze(0),
#             'mlm_labels': mlm_labels.squeeze(0)     
#         }