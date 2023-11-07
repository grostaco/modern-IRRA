from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch 
import torch.nn as nn

from typing import cast 
from transformers.models.clip.modeling_clip import CLIPModel, CLIPConfig, CLIPOutput

from .residual import ResidualEncoder, ResidualAttentionBlock
from ..losses import sdm_loss, id_loss, mlm_loss


class IRRA(pl.LightningModule):
    def __init__(self, 
                 vit: CLIPModel,
                 num_layers: int,
                 vocab_size: int,
                 eos_token_id: int,
                 temperature: float,
                 num_classes = 11003):
        super().__init__()
        
        self.logit_scale = torch.ones([]) * (1 / temperature)

        self.vit = vit 
        self.eos_token_id = eos_token_id
        
        vit_config = cast(CLIPConfig, vit.config)
        
        self.text_classification_head = nn.Linear(vit_config.text_config.hidden_size, num_classes)
        nn.init.normal_(self.text_classification_head.weight, std=.001)
        nn.init.constant_(self.text_classification_head.bias.data, val=.0)

        self.vision_classification_head = nn.Linear(vit_config.vision_config.hidden_size, num_classes)
        nn.init.normal_(self.vision_classification_head.weight, std=.001)
        nn.init.constant_(self.vision_classification_head.bias.data, val=.0)

        self.img_proj = nn.Linear(vit_config.vision_config.hidden_size, vit_config.text_config.hidden_size) 

        self.mlm = MLM(vit_config.projection_dim, num_layers, vocab_size)

    def encode_image(self, image: torch.FloatTensor):
        x = self.vit.get_image_features(image)

        return x
    
    def encode_text(self, text: torch.Tensor):
        x = self.vit.get_text_features(text).float()

        return x
    
    def forward(self, batch: dict[str, torch.Tensor]) -> CLIPOutput:
        images = batch['images']
        caption_ids = batch['caption_ids']
        clip_output = self.vit(caption_ids, images)

        return clip_output
    
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: ...):
        clip_output = self.forward(batch)

        vision_model_output, text_model_output = clip_output.vision_model_output, clip_output.text_model_output

        # TODO: use projection directly from CLIP
        image_cls = vision_model_output.pooler_output 
        text_cls = text_model_output.pooler_output

        image_logits = self.vision_classification_head(image_cls)
        text_logits = self.text_classification_head(text_cls)

        # MLM loss
        mlm_ids = batch['mlm_ids']
        mlm_embeds = self.vit.text_model(mlm_ids).last_hidden_state
        image_embeds = self.img_proj(vision_model_output.last_hidden_state)

        scores = self.mlm(mlm_embeds, image_embeds)
        mlm_labels = batch['mlm_labels'].reshape(-1)

        loss_mlm = mlm_loss(scores, mlm_labels)

        # SDM loss
        loss_sdm = sdm_loss(clip_output.image_embeds, clip_output.text_embeds, batch['pids'], self.logit_scale) 

        # ID loss
        loss_id =  id_loss(image_logits, text_logits, batch['pids']) 

        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_acc = (image_pred == batch['pids']).float().mean()
        text_acc = (text_pred == batch['pids']).float().mean()
        

        self.log_dict({'train/loss_sdm': loss_sdm,
                       'train/loss_id': loss_id,
                       'train/loss_mlm': loss_mlm,
                       'train/acc_text': text_acc,
                       'train/acc_image': image_acc},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)

        loss = loss_sdm + loss_id + loss_mlm 
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: ...):
        clip_output = self.forward(batch)

        vision_model_output, text_model_output = clip_output.vision_model_output, clip_output.text_model_output

        # TODO: use projection directly from CLIP
        image_cls = vision_model_output.pooler_output 
        text_cls = text_model_output.pooler_output

        image_logits = self.vision_classification_head(image_cls)
        text_logits = self.text_classification_head(text_cls)

        # MLM loss
        mlm_ids = batch['mlm_ids']
        mlm_embeds = self.vit.text_model(mlm_ids).last_hidden_state
        image_embeds = self.img_proj(vision_model_output.last_hidden_state)

        scores = self.mlm(mlm_embeds, image_embeds)
        mlm_labels = batch['mlm_labels'].reshape(-1)

        loss_mlm = mlm_loss(scores, mlm_labels)

        # SDM loss
        loss_sdm = sdm_loss(clip_output.image_embeds, clip_output.text_embeds, batch['pids'], self.logit_scale) 

        # ID loss
        loss_id =  id_loss(image_logits, text_logits, batch['pids']) 

        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_acc = (image_pred == batch['pids']).float().mean()
        text_acc = (text_pred == batch['pids']).float().mean()
        

        self.log_dict({'val/loss_sdm': loss_sdm,
                        'val/loss_id': loss_id,
                        'val/loss_mlm': loss_mlm,
                        'val/acc_text': text_acc,
                        'val/acc_image': image_acc},
                        prog_bar=True,
                        logger=True,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True)

        loss = loss_sdm + loss_id + loss_mlm 
        return loss
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=.001)
    

class MLM(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, vocab_size: int):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(embed_dim, embed_dim // 64, batch_first=True)
        self.encoder = ResidualEncoder(embed_dim, num_layers, embed_dim // 64)
        self.vocab_size = vocab_size 

        scale = self.encoder.d_model ** -.5

        self.ln_pre_t = nn.LayerNorm(embed_dim)
        self.ln_pre_i = nn.LayerNorm(embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)

        proj_std = scale * ((2 * self.encoder.num_layers) ** -.5)
        attn_std = scale 

        fc_std = (2 * self.encoder.d_model) ** -.5
        
        for block in self.encoder.residual_blocks:
            block = cast(ResidualAttentionBlock, block)
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)

            nn.init.normal_(block.dense1.weight, std=fc_std)
            nn.init.normal_(block.dense2.weight, std=proj_std)

        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        self.mlm_head = nn.Sequential(
            OrderedDict(
                [
                    ('dense', nn.Linear(embed_dim, embed_dim)),
                    ('gelu', nn.GELU()),
                    ('ln', nn.LayerNorm(embed_dim)),
                    ('fc', nn.Linear(embed_dim, vocab_size))
                ]
            )
        )

    def cross_former(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        x, _ = self.cross_attn(
            self.ln_pre_t(query),
            self.ln_pre_i(key),
            self.ln_pre_i(value)
        )

        x = self.encoder(x)
        x = self.ln_post(x)
        return x


    def forward(self, text_feats: torch.Tensor, image_feats: torch.Tensor):
        h = self.cross_former(text_feats, image_feats, image_feats)
        h = self.mlm_head(h)

        scores = h.float().reshape(-1, self.vocab_size)
        return scores 
