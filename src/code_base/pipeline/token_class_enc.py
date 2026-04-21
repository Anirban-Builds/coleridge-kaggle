import torch.nn as nn
import torch.nn.functional as F
import transformers as tfe
from transformers import AutoModelForTokenClassification, AutoConfig
from src.code_base.utils import CFG

class TokenClassEncoder(nn.Module):
    def __init__(self,
                 num_classes,
                 embed_size=1024,
                 max_length=128,
                 backbone=None,
                 device=CFG.device,
                 eval_model=False,
                 ):
        super().__init__()
        self.backbone_name = backbone
        if eval_model:
            self.config = AutoConfig.from_pretrained(self.backbone_name)
            self.config.num_labels = num_classes
            self.backbone = AutoModelForTokenClassification.from_config(self.config)
        else:
            self.backbone = AutoModelForTokenClassification.from_pretrained(self.backbone_name,
                                                            num_labels = num_classes,
                                                            ignore_mismatched_sizes=True)
        self.out_feat = num_classes
        self.embed_size = embed_size
        self.max_length = max_length
        self.device = device

    def forward(self, input_ids, att_mask):
        features = self.backbone(input_ids,
                                 attention_mask= att_mask).logits
        # features = features.view(-1, self.out_feat)
        return features


