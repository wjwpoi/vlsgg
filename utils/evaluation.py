import numpy as np
import torch
from dataset.coco_eval import CocoEvaluator
import utils.misc as utils
from utils.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list
from lib.openimages_evaluation import task_evaluation_sg
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, dataset_name, tokenized_text):
    model.eval()
    criterion.eval()

    # initilize evaluator
    # TODO merge evaluation programs
    if dataset_name == 'vg':
        evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
        evaluator_list = []
        for index, name in enumerate(data_loader.dataset.rel_categories):
            if index == 0:
                continue
            evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    else:
        all_results = []

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    result_dict = {'class_error': 0, 'sub_error': 0, 'obj_error': 0, 'rel_error': 0}

    with tqdm(total=len(data_loader)) as _tqdm:
        for i, (samples, targets) in enumerate(data_loader):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples, tokenized_text['input_ids'], tokenized_text['attention_mask'])
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            # loss_dict_reduced_scaled = {k: v * weight_dict[k]
            #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
            # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
            #                             for k, v in loss_dict_reduced.items()}
            result_dict = {k: float(loss_dict_reduced[k]) + v for k, v in result_dict.items()}
            ave_result_dict = {k: v / (i+1) for k, v in result_dict.items()}

            if dataset_name == 'vg':
                evaluate_rel_batch(outputs, targets, evaluator, evaluator_list)
            else:
                evaluate_rel_batch_oi(outputs, targets, all_results)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)

            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)
            
            if i % 20 == 0:
                _tqdm.set_postfix(**ave_result_dict)
            _tqdm.update(1)

    if dataset_name == 'vg':
        evaluator['sgdet'].print_stats()
    else:
        task_evaluation_sg.eval_rel_results(all_results, 100, do_val=True, do_vis=False)

    if dataset_name == 'vg':
        calculate_mR_from_evaluator_list(evaluator_list, 'sgdet')

    # gather the stats from all processes

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    return coco_evaluator

def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list):
    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)
        rel_scores = outputs['rel_logits'][batch][:,1:-1].softmax(-1)

        pred_entry = {'sub_boxes': sub_bboxes_scaled,
                      'sub_classes': pred_sub_classes.cpu().clone().numpy(),
                      'sub_scores': pred_sub_scores.cpu().clone().numpy(),
                      'obj_boxes': obj_bboxes_scaled,
                      'obj_classes': pred_obj_classes.cpu().clone().numpy(),
                      'obj_scores': pred_obj_scores.cpu().clone().numpy(),
                      'rel_scores': rel_scores.cpu().clone().numpy()}

        evaluator['sgdet'].evaluate_scene_graph_entry(gt_entry, pred_entry)

        if evaluator_list is not None:
            for pred_id, _, evaluator_rel in evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
                gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
                if gt_entry_rel['gt_relations'].shape[0] == 0:
                    continue
                evaluator_rel['sgdet'].evaluate_scene_graph_entry(gt_entry_rel, pred_entry)


def evaluate_rel_batch_oi(outputs, targets, all_results):

    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)

        rel_scores = outputs['rel_logits'][batch][:, :-1].softmax(-1)

        relation_idx = target['rel_annotations'].cpu().numpy()
        gt_sub_boxes = target_bboxes_scaled[relation_idx[:, 0]]
        gt_sub_labels = target['labels'][relation_idx[:, 0]].cpu().clone().numpy()
        gt_obj_boxes = target_bboxes_scaled[relation_idx[:, 1]]
        gt_obj_labels = target['labels'][relation_idx[:, 1]].cpu().clone().numpy()

        img_result_dict = {'sbj_boxes': sub_bboxes_scaled,
                           'sbj_labels': pred_sub_classes.cpu().clone().numpy(),
                           'sbj_scores': pred_sub_scores.cpu().clone().numpy(),
                           'obj_boxes': obj_bboxes_scaled,
                           'obj_labels': pred_obj_classes.cpu().clone().numpy(),
                           'obj_scores': pred_obj_scores.cpu().clone().numpy(),
                           'prd_scores': rel_scores.cpu().clone().numpy(),
                           'image': str(target['image_id'].item())+'.jpg',
                           'gt_sbj_boxes': gt_sub_boxes,
                           'gt_sbj_labels': gt_sub_labels,
                           'gt_obj_boxes': gt_obj_boxes,
                           'gt_obj_labels': gt_obj_labels,
                           'gt_prd_labels': relation_idx[:, 2]
                           }
        all_results.append(img_result_dict)

        