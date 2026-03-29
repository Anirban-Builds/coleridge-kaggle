import re
import json
import numpy as np
# from torch.utils.data import Dataset
from code_base.utils.utils import clean_text

class Preprocess:
    def __init__(self,
                 df,
                 dir,
                 inference=False,
                 ):
        self.df = df
        self.dir = dir
        self.inference = inference

    def _sublist(self, list_, sublist_):
        '''
        function to find label subarray in sentence
        '''
        pos=[]
        for i in range(len(list_)-len(sublist_)+1):
            if sublist_ == list_[i:i+len(sublist_)]:
                pos.append(i)
        return pos

    def _ner(self, sentence, labels):
        '''
        per sentence BIO tagging to generate positve and negative samples
        '''
        words = sentence.split()

        if labels is not None and any(re.findall(f'\\b{label}\\b', sentence)
                                      for label in labels):
            n = ['O'] * len(words)
            for label in labels:
                label_words = label.split()
                pos = self._sublist(words, label_words)

                for p in pos:
                    n[p] = 'B'
                    for i in range(p+1, p+len(label_words)):
                        n[i]= 'I'

            return True, list(zip(words, n))
        else:
            n = ['O'] *len(words)
            return False, list(zip(words, n))

    def _has_kwords(self, text):
        if re.search(r'(?:[A-Z][a-z]{2,20} ){2,6}', text):
            return True
        if re.search(r'[A-Z]{3,10}', text):
            return True
        if re.search(r'(?: [Dd]ata| [Rr]egistry|[Gg]enome [Ss]equence| [Mm]odel| [Ss]tudy| [Ss]urvey)', text):
            return True
        return False

    def __getitem__(self, index):
        ner_lst=[]
        labels = None
        row = self.df.loc[index]
        if not self.inference:
            labels = row.dataset_label.split('|')
            labels = [clean_text(label) for label in labels]

        with open(f'{self.dir}{row.Id}.json', 'r') as f:
             text_list = json.load(f)
        sentences = [clean_text(sentence)
                     for section in text_list
                     for sentence in section['text'].split('.') # list of sentences
                    ]
        for sentence in sentences:
            ispositive, tags = self._ner(sentence, labels)
            if self.inference:
                ner_lst.append(tags)
            else:
                if ispositive:
                    ner_lst.append(tags)
                elif self._has_kwords(sentence):
                    ner_lst.append(tags)
        return ner_lst



