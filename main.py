from transformers import AutoImageProcessor, AutoTokenizer
from utils.dataloader import load_data, load_vg_dict
from utils.model import SGGModel, SetCriterion
from torch.utils.data import DataLoader
from utils.misc import collate_fn
from utils.matcher import HungarianMatcher
from tqdm import tqdm
import numpy as np
import random
import os
import torch
import torch.nn as nn
from dataset import build_dataset, get_coco_api_from_dataset


def set_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）


def pre_process(example):
    example = processor(images=example['image'].convert('RGB').resize((800,600)), return_tensors="pt")
    return example


dataset_name = 'vg'
num_classes = 151 if dataset_name != 'oi' else 289 # some entity categories in OIV6 are deactivated.
num_rel_classes = 51 if dataset_name != 'oi' else 31
weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_rel': 1}


set_seed(621)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

idx_to_label, idx_to_predicate, label_list, predicate_list = load_vg_dict('/home/wjw/data/VG/VG-SGG-dicts-with-attri.json')
processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = SGGModel(roberta_model_name="roberta-large", ddetr_model_name="SenseTime/deformable-detr",
                 embed_dim=256, hidden_dim=512, num_heads=8, N_ALIF=2, num_queries=100).to(device)

tokenized_text = tokenizer(label_list + predicate_list, padding='max_length', max_length=8)
tokenized_text = {k: torch.tensor(v).to(device) for k, v in tokenized_text.items()}

# dataset = load_data(dataset_name, path='/home/wjw/data/')
# dataset = dataset['train'].train_test_split(test_size=0.3)  ######## 

dataset_train = build_dataset('train', dataset_name, '/home/wjw/data/VG/', '/home/wjw/data/VG/VG_100K/')
dataset_val = build_dataset('val', dataset_name, '/home/wjw/data/VG/', '/home/wjw/data/VG/VG_100K/')

train_dataloader = DataLoader(dataset_train, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(dataset_val, batch_size=1, collate_fn=collate_fn)

matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2, iou_threshold=0.7)
criterion = SetCriterion(num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.1, losses=['labels', 'boxes', 'cardinality', "relations"]).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8, last_epoch=-1)
total_epoch = 50

for epoch in range(total_epoch):
    model.train()
    with tqdm(total=len(train_dataloader)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, total_epoch))
        for batch in train_dataloader:
            inputs = {'nested_pixel_values': batch[0].to(device), 
                      'input_ids': tokenized_text['input_ids'], 'attention_mask': tokenized_text['attention_mask']}
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            targets = targets = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(loss='{:.4f}'.format(loss.item()))
            _tqdm.update(1)

        
    if epoch % 5 == 0:
        model.eval()
        recall_all = [0] * 3
        print('########## START TEST #########')
        for batch in tqdm(val_dataloader):
            inputs = {'nested_pixel_values': batch[0].to(device), 
                    'input_ids': tokenized_text['input_ids'], 'attention_mask': tokenized_text['attention_mask']}
            outputs = model(**inputs)
            targets = targets = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
            recall_all = model.evaluate(outputs, targets, matcher, recall_all)

        recall_all = [float(recall/len(dataset_val)) for recall in recall_all]
        print(recall_all)


            

