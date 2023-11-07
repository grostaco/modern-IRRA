import torchvision.transforms as T


from .cuhkpedes import CUHKPEDES

def build_transforms(img_size: tuple[int, int] = (384, 128), augment: bool = False, is_train: bool = False):
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transforms = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        return transforms
    
    if augment:
        transforms = T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(),
            T.Pad(10),
            T.RandomCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(.02, .4), value=mean) # type: ignore
        ])
    else:
        transforms = T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    
    return transforms 

def build_dataloader(training: bool = True):
    dataset = CUHKPEDES('./data/CHUK-PEDES')
    num_classes = len(dataset.train_id_container)

    if training:
        train_transforms = build_transforms(is_train=True)
        val_transforms = build_transforms(is_train=False)
