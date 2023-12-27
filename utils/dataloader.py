from datasets import Dataset, Image, DatasetDict
from tqdm import tqdm
import h5py
import json
import numpy as np


def load_vg_dict(vg_dict_dir):
    with open(vg_dict_dir) as f:
        vg_dict = json.load(f)
    idx_to_label = vg_dict['idx_to_label']
    idx_to_label['0'] = 'background'
    idx_to_predicate = vg_dict['idx_to_predicate']
    idx_to_predicate['0'] = 'no relation'

    label_list = []
    predicate_list = []
    for i in range(len(idx_to_label)):
        label_list.append(idx_to_label[str(i)])
    for i in range(len(idx_to_predicate)):
        predicate_list.append(idx_to_predicate[str(i)])
    return idx_to_label, idx_to_predicate, label_list, predicate_list


def load_data(dataset="vg", path='/home/wjw/data/'):
    if dataset.lower() == "vrd":  # TO DO
        # data = load_dataset("imagefolder", data_dir="~/data/VRD")
        # data = data.filter(lambda example: len(example["anno"]) > 5)
        # data = data.map(lambda example: {"answer": encode_vrd(eval(example["anno"]))})
        pass
    elif dataset.lower() == "vg":
        path = path + 'VG/'
        with h5py.File(path + "VG-SGG-with-attri.h5", 'r') as f:
            corrupted_imgs = ['1592', '1722', '4616', '4617']

            with open(path + 'image_data.json', 'r') as f:
                image_data = json.load(f)

            img_list = []
            label_list = []
            split_list = []
            with h5py.File(path + "VG-SGG-with-attri.h5", 'r') as f:
                i = 0
                for content in tqdm(image_data):  # no enumerate, because of corrupted_imgs  
                    image_id =  str(content['image_id'])
                    width = content['width']
                    height = content['height']
                    if image_id in corrupted_imgs:
                        continue

                    image = path + 'VG_100K/' + image_id + '.jpg'
                    box = f['boxes_512'][f['img_to_first_box'][i]: f['img_to_last_box'][i] + 1]
                    scale = max(width, height) / 512
                    box = box * scale / np.array([width, height, width, height])  # normalize

                    label = f['labels'][f['img_to_first_box'][i]: f['img_to_last_box'][i] + 1]
                    predicates = f['predicates'][f['img_to_first_rel'][i]: f['img_to_last_rel'][i] + 1]
                    relationships = f['relationships'][f['img_to_first_rel'][i]: f['img_to_last_rel'][i] + 1] - f['img_to_first_box'][i]

                    i += 1
                    rel_annotations = np.concatenate((relationships, predicates), axis=-1)
                    if rel_annotations.size == 0:  # no relation, pass this picture
                        continue


                    img_list.append(image)
                    label_dict = {'labels': label.squeeze(), 'boxes': box, 'rel_annotations': rel_annotations}
                    label_list.append(label_dict)

                    split_list.append(f['split'][i])
                    
                    if i >= 256:
                        break

        list_dict = {'image': img_list, 'label': label_list, 'split': split_list}
        dataset = Dataset.from_dict(list_dict).cast_column('image', Image())

        train_dataset = dataset.filter(lambda example: example['split'] == 0).remove_columns(["split"])
        test_dataset = dataset.filter(lambda example: example['split'] == 2).remove_columns(["split"])
        dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
    return dataset
