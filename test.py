from lib.datasets.cuhkpedes import CUHKPEDES
from lib.models.irra import IRRA
from lib.datasets.builder import build_transforms
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling, AutoTokenizer, Trainer, TrainingArguments
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from datasets import load_dataset

cuhk = CUHKPEDES('./data/CUHK-PEDES')

tokenizer = AutoTokenizer.from_pretrained('./masked_tokenizer')
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch16')

def tokenize(examples):
    return tokenizer(examples['caption'], max_length=77, 
                     truncation=True,
                     padding='max_length',
                     return_tensors='pt')

def mlm(examples):
    return {f'mlm_{k}': v for k, v in collator.torch_call(examples['input_ids']).items()}

dataset = cuhk.dataset
dataset['train'] = dataset['train'].select(range(32))
dataset['test'] = dataset['test'].select(range(8))
dataset['validation'] = dataset['validation'].select(range(8))

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.map(mlm, batched=True)
dataset = dataset.map(processor, batched=True, input_columns='image')
dataset = dataset.remove_columns(['image', 'caption'])
dataset = dataset.rename_column('pid', 'pids')

model = IRRA.from_pretrained('openai/clip-vit-base-patch16', num_classes=len(set(dataset['train']['pids'])))


args = TrainingArguments(
    '.',
    fp16=True,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    remove_unused_columns=False,
    report_to=[],
    save_strategy='steps',
    logging_steps=4,
    seed=42,
    num_train_epochs=10,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=dataset['train'],
)

trainer.train()