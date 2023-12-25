from transformers import AutoImageProcessor, AutoTokenizer
from utils.misc import set_seed
from utils.dataloader import load_data, load_vg_dict
from utils.model import SGGModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn


def pre_process(example):
    example = processor(images=example['image'].convert('RGB'), return_tensors="pt")
    return example


set_seed(621)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
idx_to_label, idx_to_predicate, label_list, predicate_list = load_vg_dict('/home/wjw/data/VG/VG-SGG-dicts-with-attri.json')
processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = SGGModel(roberta_model_name="roberta-base", ddetr_model_name="SenseTime/deformable-detr",
                 embed_dim=768, hidden_dim=1024, num_heads=8, N_ALIF=2, num_queries=100).to(device)

tokenized_text = tokenizer(label_list + predicate_list, padding='longest')

dataset = load_data("vg", path='/home/wjw/data/')

# dataset = dataset.remove_columns(["target", 'bbox', 'predicate', 'relationship'])
dataset = dataset.map(pre_process, writer_batch_size=512)
dataset = dataset.remove_columns(["image"])
dataset = dataset.with_format("torch")
train_dataloader = DataLoader(dataset['train'], batch_size=8, shuffle=True)
test_dataloader = DataLoader(dataset['test'], batch_size=32)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8, last_epoch=-1)
celloss = nn.CrossEntropyLoss()
total_epoch = 20

for epoch in range(total_epoch):
    model.train()
    with tqdm(total=len(train_dataloader)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, total_epoch))
        for batch in train_dataloader:
            inputs = {'pixel_values': batch['pixel_values'].to(device), 'pixel_mask': batch['pixel_mask'].to(device), 
                      'input_ids': tokenized_text['input_ids'].to(device), 'attention_mask': tokenized_text['attention_mask'].to(device)}
            
            optimizer.zero_grad()
            outputs = model(**inputs)

