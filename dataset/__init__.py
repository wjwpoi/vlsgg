# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, dataset_name, ann_path, img_folder):
    if dataset_name == 'vg' or dataset_name == 'oi':
        return build_coco(image_set, ann_path, img_folder)
    raise ValueError(f'dataset {dataset_name} not supported')