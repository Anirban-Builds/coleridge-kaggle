import torch.nn as nn
import torch.nn.functional as F
import transformers as tfe
from transformers import AutoModel, AutoConfig
from code_base.utils import CFG

class TextEncoder(nn.Module):
    def __init__(self,
                 num_classes,
                 embed_size=1024,
                 max_length=35,
                 backbone=None,
                 dropout=0.5,
                 device=CFG.device,
                 eval_model=False,
                 ):
        super.__init__()
        self.backbone_name = backbone
        if eval_model:
            self.config = AutoConfig.from_pretrained(self.backbone_name)
            self.backbone = AutoModel.from_config(self.config)
        else:
            self.backbone = AutoModel.from_pretrained(self.backbone_name)
        self.out_feat = num_classes
        self.embed_size = embed_size
        self.dropout = dropout
        self.max_length = max_length
        self.device = device
        self.fc = nn.Linear(self.backbone.config.hidden_size, self.embed_size)
        self.pool = nn.AvgPool1d(kernel_size=self.max_length)
        self.bn = nn.BatchNorm1d(self.embed_size)

    def forward(self, input_ids, att_mask, labels=None):
        features = self.backbone(input_ids,
                                 attention_mask= att_mask).last_hidden_state
        features = self.fc(features)
        features = features.transpose(1, 2)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        features = self.bn(features)
        if labels is not None:
            return self.final(features, labels)
        return features


