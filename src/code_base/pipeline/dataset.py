import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class NERDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer=None,
                 padding = "max_length",
                 truncation = True,
                 max_length=35,
                 gen_feat_only = False):
        self.data = data # ner_list
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.only_feat = gen_feat_only
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index] # one sample in ner_list
        words=[]
        tags=[]
        for word, tag in sample:
            words.append(word)
            tags.append(tag)
        # words = " ".join(words)
        text = self.tokenizer(
                words,
                padding = self.padding,
                is_split_into_words = True,
                truncation = self.truncation,
                max_length = self.max_length,
                return_tensors = "pt"
            )
        input_ids = text["input_ids"][0]
        att_mask = text["attention_mask"][0]

        label2id = {'O': 0, 'B': 1, 'I': 2}
        labels = []
        for word_idx in text.word_ids(batch_index=0):
            if word_idx is not None:
                labels.append(label2id[tags[word_idx]])
            else:
                labels.append(0)

        if self.only_feat:
            return input_ids, att_mask
        return input_ids, att_mask, torch.tensor(labels).float()
