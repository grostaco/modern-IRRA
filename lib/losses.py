from typing import Optional
import torch
import torch.nn.functional as F

def sdm_loss(image_features: torch.Tensor, text_features: torch.Tensor, pid: torch.Tensor, logit_scale: torch.Tensor,
             image_id: Optional[torch.Tensor] = None,
             factor = .3,
             epsilon = 1e-8):
    batch_size = image_features.shape[0]
    pid = pid.reshape((batch_size, 1))

    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id is not None:
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()

        labels = (labels - image_id_mask) * factor + image_id_mask 
    
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss 

def mlm_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    return F.cross_entropy(y_true, y_pred, ignore_index=-100)

def id_loss(image_logits: torch.Tensor, text_logits: torch.Tensor, labels: torch.Tensor):
    return (F.cross_entropy(image_logits, labels) + F.cross_entropy(text_logits, labels)) / 2
