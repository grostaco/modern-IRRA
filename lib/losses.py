from typing import Optional
import torch
import torch.nn.functional as F

def sdm_loss(image_logits: torch.Tensor, text_logits: torch.Tensor, pid: torch.Tensor,
             image_id: Optional[torch.Tensor] = None,
             factor = .3,
             epsilon = 1e-8):
    batch_size = image_logits.size(0)
    pid = pid.reshape((batch_size, 1))

    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id is not None:
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()

        labels = (labels - image_id_mask) * factor + image_id_mask 

    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_logits, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_logits, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_logits, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_logits, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss 

def mlm_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    return F.cross_entropy(y_pred, y_true, ignore_index=-100)

def id_loss(image_logits: torch.Tensor, text_logits: torch.Tensor, labels: torch.Tensor):
    return (F.cross_entropy(image_logits, labels) + F.cross_entropy(text_logits, labels)) / 2
