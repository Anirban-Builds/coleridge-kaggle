import torch
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
        self.data = data # dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.only_feat = gen_feat_only
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.loc[index] # one row = one sentence

        words = row.text.split()

        text = self.tokenizer(
                words,
                padding = self.padding,
                truncation = self.truncation,
                max_length = self.max_length,
                return_tensors = "pt",
                is_split_into_words=True
            )
        input_ids = text["input_ids"][0]
        att_mask = text["attention_mask"][0]
        word_ids = text.word_ids(batch_index=0)
        if not self.only_feat:
            label2id = {'O': 0, 'B': 1, 'I': 2}
            labels = []
            for word_idx in word_ids:
                if word_idx is not None:
                    labels.append(label2id[row.tags[word_idx]])
                else:
                    labels.append(0)

        if self.only_feat: # if in inference/eval mode
            word_ids = [w if w is not None else -1 for w in word_ids]
            return input_ids, att_mask, torch.tensor(word_ids)
        return input_ids, att_mask, torch.tensor(labels).float()
