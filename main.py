from transformers import AutoImageProcessor, AutoTokenizer
from utils.dataloader import load_data, load_vg_dict
from utils.model import SGGModel
from torch.utils.data import DataLoader
from utils.misc import collate_fn
from dataset import build_dataset, get_coco_api_from_dataset
from tqdm import tqdm
import numpy as np
import random
import os
import torch
import torch.nn as nn
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Transformer
    parser.add_argument('--roberta_model_name', default='roberta-base', type=str)
    parser.add_argument('--ddetr_model_name', default='SenseTime/deformable-detr', type=str)
    parser.add_argument('--ALIF_embed_dim', default=768, type=int)
    parser.add_argument('--N_ALIF', default=2, type=int)
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--ann_path', default='/home/wjw/data/VG/', type=str)
    parser.add_argument('--img_folder', default='/home/wjw/data/VG/VG_100K/', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser


def set_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）


def pre_process(example):
    example = processor(images=example['image'].convert('RGB'), return_tensors="pt")
    return example


parser = argparse.ArgumentParser('SGG training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
set_seed(args.seed)
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

idx_to_label, idx_to_predicate, label_list, predicate_list = load_vg_dict('/home/wjw/data/VG/VG-SGG-dicts-with-attri.json')
processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = SGGModel(args).to(args.device)

tokenized_text = tokenizer(label_list + predicate_list, padding='longest')

# dataset = load_data("vg", path='/home/wjw/data/')

# # dataset = dataset.remove_columns(["target", 'bbox', 'predicate', 'relationship'])
# dataset = dataset.map(pre_process, writer_batch_size=512)
# dataset = dataset.remove_columns(["image"])
# dataset = dataset.with_format("torch")
# train_dataloader = DataLoader(dataset['train'], batch_size=8, shuffle=True)
# test_dataloader = DataLoader(dataset['test'], batch_size=32)

dataset_train = build_dataset(image_set='train', args=args)
dataset_val = build_dataset(image_set='val', args=args)
sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers)
data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8, last_epoch=-1)
celloss = nn.CrossEntropyLoss()
total_epoch = 20

for epoch in range(total_epoch):
    model.train()
    with tqdm(total=len(data_loader_train)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, total_epoch))
        for batch in data_loader_train:
            print(1)
            inputs = {'pixel_values': batch['pixel_values'].to(args.device), 'pixel_mask': batch['pixel_mask'].to(args.device), 
                      'input_ids': tokenized_text['input_ids'].to(args.device), 'attention_mask': tokenized_text['attention_mask'].to(args.device)}
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            

