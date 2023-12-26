from transformers import DeformableDetrModel, RobertaModel, RobertaConfig, DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrEncoderLayer
from transformers.models.roberta.modeling_roberta import RobertaLayer
import torch.nn as nn
import torch
from transformers.models.deformable_detr.configuration_deformable_detr import DeformableDetrConfig


class ALIF(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=1024, num_heads=8):
        super().__init__()
        roberta_cfg = RobertaConfig(is_decoder=False, hidden_size=embed_dim)
        ddetr_cfg = DeformableDetrConfig(d_model=embed_dim, encoder_ffn_dim=hidden_dim)
        self.ddetr1 = DeformableDetrEncoderLayer(ddetr_cfg)
        self.ddetr2 = DeformableDetrEncoderLayer(ddetr_cfg)
        self.roberta = RobertaLayer(roberta_cfg)
        self.cross_attentions = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.b = nn.Parameter(torch.tensor(0.5))

    def forward(self, image_embeds, text_embeds):
        image_embeds, text_embeds = (image_embeds + self.b * self.cross_attentions(image_embeds, text_embeds, text_embeds),
                                     text_embeds + self.b * self.cross_attentions(text_embeds, image_embeds, image_embeds))
        image_embeds = self.ddetr2(self.ddetr1(image_embeds, None), None)
        text_embeds = self.roberta(self.roberta(text_embeds))
        return image_embeds, text_embeds


class SGGModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_queries = args.num_queries
        self.num_labels = 151
        self.roberta_model = RobertaModel.from_pretrained(args.roberta_model_name)
        self.ddetr_model = DeformableDetrModel.from_pretrained(args.ddetr_model_name)
        self.ALIF = nn.ModuleList([ALIF(args.ALIF_embed_dim, 1024, 8) for _ in range(args.N_ALIF)])
        self.mha = nn.MultiheadAttention(args.embed_dim, args.num_heads, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(args.embed_dim, args.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.hidden_dim, args.embed_dim))
        self.queries = nn.Parameter(torch.Tensor(args.num_queries*2, args.embed_dim))

        self.bbox_decoder = nn.Sequential(nn.Linear(args.embed_dim, args.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.hidden_dim, args.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.hidden_dim, args.embed_dim))
        self.label_decoder = nn.Sequential(nn.Linear(args.embed_dim, args.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.hidden_dim, args.embed_dim))
        self.relation_decoder = nn.Sequential(nn.Linear(3 * args.embed_dim, args.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.hidden_dim, args.embed_dim))
        

    def forward(self, pixel_values, pixel_mask, input_ids, attention_mask):
        image_embeds = self.ddetr_model(pixel_values=pixel_values.squeeze(), pixel_mask=pixel_mask).encoder_last_hidden_state
        text_embeds = self.roberta_model(input_ids=input_ids.view(-1, input_ids.shape[-1]), attention_mask=attention_mask.view(-1, attention_mask.shape[-1])).last_hidden_state
        text_embeds = text_embeds.view(input_ids.shape[0], input_ids.shape[1], -1)

        target_embeds = self.ffn(self.mha(self.queries, image_embeds, image_embeds))
        target_bbox = self.bbox_decoder(target_embeds)
        target_label = self.bbox_decoder(target_embeds) * text_embeds[:self.num_labels, :]

        relation_embeds = torch.concat(target_embeds[:, :self.num_queries, :], target_embeds[:, self.num_queries:, :], dim=-1)
        relation = self.relation_decoder(relation_embeds) * text_embeds[self.num_labels:, :]

        return target_bbox, target_label, relation

    def match_triple(self, target_bbox, target_label, relation):
        
        return 0
        

        


