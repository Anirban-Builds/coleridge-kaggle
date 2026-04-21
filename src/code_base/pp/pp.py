import re
import json
import numpy as np
from src.code_base.utils import clean_text

class Preprocess:
    def __init__(self,
                 df = None,
                 dir = "",
                 inference=False,
                 tiny = False,
                 text = ""
                 ):
        self.df = df
        self.dir = dir
        self.inference = inference
        self.tiny = tiny
        self.text = text

    def _choose_chunk(self, sent, chunks):
        if 200<= len(sent) < 400:
            chunks.append(sent)
            return chunks, ""
        elif(len(sent)) >= 400:
            chunks.extend(self._split_long(sent))
            return chunks, ""
        else:
            return chunks, sent

    def _chunk_sentences(self, sentences):
        '''
        sentences must stay in chunk of 200 - 400 chars
        '''
        chunks=[]
        buf=""
        for sent in sentences:
            if not sent.strip():
                continue # empty sentence
            if not buf: # if buffer is empty
                chunks, buf = self._choose_chunk(sent, chunks)
            else: # if buffer is not empty
                combined = buf + ". " + sent
                if len(combined) >=400:
                    chunks.append(buf)
                    buf = ""
                    chunks, buf = self._choose_chunk(sent, chunks)
                else:
                    buf = combined

        if buf:
            chunks.append(buf) # if buf is not empty at last
        return chunks

    def _split_long(self, sentence):
        '''
        split long sentence into multiple overlapping candidates
        '''
        results = []
        results.extend(re.findall(
            r'(?:^| ).{0,150}[A-Z][a-z]{2,20} (?:(?:[A-Z][a-z]{2,20}|of|up|to|and|the|in|on|s|for)[- .,]){0,10}(?:[A-Z][a-z]{2,20})(?: data| survey| sample| study| [0-9]{2,4})*.{0,150}(?:[. ]|$)',
            sentence))
        results.extend(re.findall(
            r'(?:^| ).{0,200}(?: [Dd]ata| [Rr]egistry|[Gg]enome [Ss]equence| [Mm]odel| [Ss]tudy| [Ss]urvey).{0,200}(?:[. ]|$)',
            sentence))
        results.extend(re.findall(
            r'(?:^| ).{0,200}[A-Z]{4,10}.{0,200}(?:[. ]|$)',
            sentence))
        return results

    def _is_candidate(self, sentence):
        '''
        regex pattern to identify candidate sentence
        '''
        a = re.findall(r'(?:(?:[A-Z][a-z]{2,20}|of|in|s|for|and) ){3,6}', sentence)
        a.extend(re.findall(r'(?: [Dd]ata| [Rr]egistry|[Gg]enome [Ss]equence| [Mm]odel| [Ss]tudy| [Ss]urvey)', sentence))
        a.extend(re.findall(r'[A-Z]{4,10}', sentence))
        return len(a) > 0

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
        n = ['O'] * len(words)

        if labels is not None and any(re.findall(f'\\b{label}\\b', sentence)
                                      for label in labels):
            for label in labels:
                label_words = label.split()
                pos = self._sublist(words, label_words)

                for p in pos:
                    n[p] = 'B'
                    for i in range(p+1, p+len(label_words)):
                        n[i]= 'I'

            return True, list(zip(words, n))
        else:
            return False, list(zip(words, n))

    def __getitem__(self, index=None):
        pos_lst=[]
        neg_lst=[]
        labels = None
        if not self.tiny:
            row = self.df.loc[index]

        if not self.inference:
            labels = row.dataset_label.split('|')
            labels = [clean_text(label) for label in labels]

        if not self.tiny:
            with open(f'{self.dir}{row.Id}.json', 'r') as f:
                text_list = json.load(f)

        if not self.tiny:
            sentences = [clean_text(sentence)
                        for section in text_list
                        for sentence in section['text'].split('.') # list of sentences
                        ]
        if self.tiny:
             sentences = [clean_text(sentence)
                        for sentence in self.text.split('.') # list of sentences
                        ]

        print("number of sentences before chunking : ", len(sentences))
        print("\n")

        chunks = self._chunk_sentences(sentences)
        print("number of sentences after chunking : ", len(chunks))
        print("\n")
        cand_sents = [s for s in chunks if self._is_candidate(s)]
        print("number of sentences after filtering : ", len(cand_sents))
        print("\n")

        for sentence in cand_sents:
            ispositive, tags = self._ner(sentence, labels)
            if self.inference:
                    pos_lst.append(tags)
            else:
                if ispositive:
                    pos_lst.append(tags)
                else:
                    neg_lst.append(tags)
        return pos_lst if self.inference else (pos_lst, neg_lst)