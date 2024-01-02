from transformers import DeformableDetrModel, RobertaModel, RobertaConfig, DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrEncoderLayer, DeformableDetrDecoderLayer
from transformers.models.roberta.modeling_roberta import RobertaLayer
from utils.misc import accuracy, get_world_size, is_dist_avail_and_initialized
import torch.nn as nn
import torch
import torch.nn.functional as F
import utils.box_ops as box_ops
from transformers.models.deformable_detr.configuration_deformable_detr import DeformableDetrConfig


class ALIF(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=512, num_heads=8):
        super().__init__()
        roberta_cfg = RobertaConfig(is_decoder=False, hidden_size=embed_dim, num_attention_heads=num_heads)
        ddetr_cfg = DeformableDetrConfig(d_model=embed_dim, encoder_ffn_dim=hidden_dim, encoder_attention_heads=num_heads)
        # self.ddetr1 = DeformableDetrEncoderLayer(ddetr_cfg)
        # self.ddetr2 = DeformableDetrEncoderLayer(ddetr_cfg)
        self.detr = nn.Linear(embed_dim, embed_dim)
        self.roberta = RobertaLayer(roberta_cfg)
        self.cross_attentions = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.b = nn.Parameter(torch.tensor(0.5))

    def forward(self, image_embeds, text_embeds):
        image_embeds, text_embeds = (image_embeds + self.b * self.cross_attentions(image_embeds, text_embeds, text_embeds, need_weights=False)[0],
                                     text_embeds + self.b * self.cross_attentions(text_embeds, image_embeds, image_embeds, need_weights=False)[0])
        # image_embeds = self.ddetr2(self.ddetr1(image_embeds))
        image_embeds = self.detr(image_embeds)
        text_embeds = self.roberta(text_embeds)[0]
        return image_embeds, text_embeds


class SGGModel(nn.Module):
    def __init__(self, roberta_model_name="roberta-large", ddetr_model_name="SenseTime/deformable-detr",
                 embed_dim=256, hidden_dim=512, num_heads=8, N_ALIF=2, num_queries=100):
        super().__init__()
        self.num_queries = num_queries
        self.num_labels = 151
        self.roberta_model = RobertaModel.from_pretrained(roberta_model_name)
        self.ddetr_model = DeformableDetrModel.from_pretrained(ddetr_model_name)
        self.ddetr_model.freeze_backbone()
        
        for _, param in self.roberta_model.named_parameters():
            param.requires_grad_(False)

        self.text_project = nn.Linear(self.roberta_model.config.hidden_size, embed_dim)
        self.image_project = nn.Linear(self.ddetr_model.config.hidden_size, embed_dim)

        # self.ALIF = ALIF(embed_dim, 512, 8)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, embed_dim))
        self.queries = nn.Parameter(torch.Tensor(num_queries*2, embed_dim))
        nn.init.xavier_uniform_(self.queries)

        self.bbox_decoder = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 4))
        self.label_decoder = nn.Linear(embed_dim, embed_dim)

        self.entity_class_embed = nn.Linear(embed_dim * 2, embed_dim)
        self.entity_bbox_embed = nn.Sequential(nn.Linear(embed_dim * 2, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 4))
        
        self.relation_decoder = nn.Sequential(nn.Linear(2 * embed_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, embed_dim))

    def forward(self, nested_pixel_values, input_ids, attention_mask):
        batch_size = nested_pixel_values.tensors.shape[0]
        image_embeds = self.ddetr_model(pixel_values=nested_pixel_values.tensors, pixel_mask=nested_pixel_values.mask).encoder_last_hidden_state
        text_embeds = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        text_embeds = self.text_project(text_embeds).repeat(batch_size, 1, 1)
        # image_embeds, text_embeds = self.ALIF(image_embeds, text_embeds)

        target_embeds = self.ffn(self.mha(self.queries.repeat(batch_size,1,1), image_embeds, image_embeds, need_weights=False)[0])
        target_bbox = self.bbox_decoder(target_embeds).sigmoid()
        target_label_logit = self.label_decoder(target_embeds) @ text_embeds[:, :self.num_labels, :].transpose(1, 2)  # 前半部分是实体

        couple_embeds = torch.concat((target_embeds[:, :self.num_queries, :], target_embeds[:, self.num_queries:, :]), dim=-1)  # half of targets are subjects while others are objects
        relation_logit = self.relation_decoder(couple_embeds) @ text_embeds[:, self.num_labels:, :].transpose(1, 2)  # 后半部分是关系

        pred_logits = self.entity_class_embed(couple_embeds) @ text_embeds[:, :self.num_labels, :].transpose(1, 2)
        pred_boxes = self.entity_bbox_embed(couple_embeds).sigmoid()

        out = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes,
               'sub_logits': target_label_logit[:, :self.num_queries, :], 'sub_boxes': target_bbox[:, :self.num_queries, :],
               'obj_logits': target_label_logit[:, self.num_queries:, :], 'obj_boxes': target_bbox[:, self.num_queries:, :],
               'rel_logits': relation_logit}
        # out = [{'pred_logits': pred_logits[i], 'pred_boxes': pred_boxes[i],
        #        'sub_logits': target_label_logit[:, :self.num_queries, :][i], 'sub_boxes': target_bbox[:, :self.num_queries, :][i],
        #        'obj_logits': target_label_logit[:, self.num_queries:, :][i], 'obj_boxes': target_bbox[:, self.num_queries:, :][i],
        #        'rel_logits': relation_logit[i]} for i in range(batch_size)]

        return out
    
    def evaluate(self, outputs, targets, matcher=None, recall_all=[0, 0, 0], topk=[20, 50, 100]):
        entity_indices, rel_indices, _, _= matcher(outputs, targets)
        pred_rel = outputs['rel_logits'].argmax(-1)
        pred_sub = outputs['sub_logits'].argmax(-1)
        pred_obj = outputs['obj_logits'].argmax(-1)
        pred_sub_box = outputs['sub_boxes']
        pred_obj_box = outputs['obj_boxes']

        for i, rel_index in enumerate(rel_indices):
            target = targets[i]
            pred_rel_index, gt_rel_index = rel_index
            gt_rel_annotations = target['rel_annotations'][gt_rel_index]
            sub_label = (target['labels'][gt_rel_annotations[:, 0]] == pred_sub[i][pred_rel_index]).to('cpu')
            obj_label = (target['labels'][gt_rel_annotations[:, 1]] == pred_obj[i][pred_rel_index]).to('cpu')
            # sub_box = (target['boxes'][gt_rel_annotations[:, 0]] == pred_sub_box[i][pred_rel_index])
            # obj_box = (target['boxes'][gt_rel_annotations[:, 1]] == pred_obj_box[i][pred_rel_index])
            sub_box, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(target['boxes'][gt_rel_annotations[:, 0]]),
                                          box_ops.box_cxcywh_to_xyxy(pred_sub_box[i][pred_rel_index]))
            obj_box, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(target['boxes'][gt_rel_annotations[:, 1]]),
                                          box_ops.box_cxcywh_to_xyxy(pred_obj_box[i][pred_rel_index]))
            sub_box = (sub_box >= 0.5).squeeze(0).to('cpu')
            obj_box = (obj_box >= 0.5).squeeze(0).to('cpu')
            rel_label = (gt_rel_annotations[:, 2] == pred_rel[i][pred_rel_index]).to('cpu')
            result = logical_ands((sub_label, obj_label, sub_box, obj_box, rel_label))
            for i in range(len(topk)):
                recall_all[i] += (result[: topk[i]] == True).sum() / len(gt_rel_annotations)
        return recall_all
    

def logical_ands(and_list):  # extend torch.logical_and to more than two inputs
    assert len(and_list) >= 2
    result = and_list[0]
    for i in range(1, len(and_list)):
        result = torch.logical_and(result, and_list[i])
    return result
    
    
    
class SetCriterion(nn.Module):
    """ This class computes the loss for model.
        COPY from RelTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_rel_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.num_rel_classes = num_rel_classes
        empty_weight_rel = torch.ones(self.num_rel_classes)
        empty_weight_rel[0] = self.eos_coef
        self.register_buffer('empty_weight_rel', empty_weight_rel)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Entity/subject/object Classification loss
        """
        assert 'pred_logits' in outputs

        pred_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices[0])
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[0])])
        target_classes = torch.full(pred_logits.shape[:2], 0, dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o

        sub_logits = outputs['sub_logits']
        obj_logits = outputs['obj_logits']

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 0]] for t, (_, J) in zip(targets, indices[1])])
        target_relo_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 1]] for t, (_, J) in zip(targets, indices[1])])

        target_sub_classes = torch.full(sub_logits.shape[:2], 0, dtype=torch.int64, device=sub_logits.device)
        target_obj_classes = torch.full(obj_logits.shape[:2], 0, dtype=torch.int64, device=obj_logits.device)

        target_sub_classes[rel_idx] = target_rels_classes_o
        target_obj_classes[rel_idx] = target_relo_classes_o

        target_classes = torch.cat((target_classes, target_sub_classes, target_obj_classes), dim=1)
        src_logits = torch.cat((pred_logits, sub_logits, obj_logits), dim=1)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')

        loss_weight = torch.cat((torch.ones(pred_logits.shape[:2]).to(pred_logits.device), indices[2]*0.5, indices[3]*0.5), dim=-1)
        losses = {'loss_ce': (loss_ce * loss_weight).sum()/self.empty_weight[target_classes].sum()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
            losses['sub_error'] = 100 - accuracy(sub_logits[rel_idx], target_rels_classes_o)[0]
            losses['obj_error'] = 100 - accuracy(obj_logits[rel_idx], target_relo_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['rel_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["rel_annotations"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the entity/subject/object bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices[0])
        pred_boxes = outputs['pred_boxes'][idx]
        target_entry_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices[0])], dim=0)

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 0]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        target_relo_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 1]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        rels_boxes = outputs['sub_boxes'][rel_idx]
        relo_boxes = outputs['obj_boxes'][rel_idx]

        src_boxes = torch.cat((pred_boxes, rels_boxes, relo_boxes), dim=0)
        target_boxes = torch.cat((target_entry_boxes, target_rels_boxes, target_relo_boxes), dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_relations(self, outputs, targets, indices, num_boxes, log=True):
        """Compute the predicate classification loss
        """
        assert 'rel_logits' in outputs

        src_logits = outputs['rel_logits']
        idx = self._get_src_permutation_idx(indices[1])
        target_classes_o = torch.cat([t["rel_annotations"][J,2] for t, (_, J) in zip(targets, indices[1])])
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)

        losses = {'loss_rel': loss_ce}
        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'relations': self.loss_relations
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        self.indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"])+len(t["rel_annotations"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels' or loss == 'relations':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results