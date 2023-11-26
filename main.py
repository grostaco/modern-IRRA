from lib.datasets.cuhkpedes import CUHKPEDES
from lib.datasets.utils import MLMDataset
from lib.datasets.builder import build_transforms
from lib.models.irra import IRRA

from transformers import AutoTokenizer 
from transformers.models.clip import CLIPModel

import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


dataset = CUHKPEDES('./data/CUHK-PEDES')
transforms = build_transforms(is_train=True)
transforms_val = build_transforms(is_train=False)
tokenizer = AutoTokenizer.from_pretrained('./masked_tokenizer')

train_dataset = MLMDataset(dataset.df_train, tokenizer=tokenizer, transforms=transforms)
val_dataset = MLMDataset(dataset.df_val, tokenizer=tokenizer, transforms=transforms_val)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, drop_last=True, shuffle=True, num_workers=8)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, drop_last=True, num_workers=8)
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')


model = IRRA(clip_model, 4, tokenizer.vocab_size, num_classes=len(dataset.train_id_container)) # type: ignore

logger = TensorBoardLogger('logs')
trainer = pl.Trainer(
    max_epochs=60,
    log_every_n_steps=10,
    logger=logger,
    precision=16)

trainer.fit(model, train_dataloaders=train_dataloader)