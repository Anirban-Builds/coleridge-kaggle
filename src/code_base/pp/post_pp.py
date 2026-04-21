import numpy as np
from src.code_base.utils import clean_text, clean_spans, regex_extract
from src.code_base.pp import Automata as A

def post_process(preds, word_ids, text):
    preds_class1 = preds[:, :, 1].cpu().numpy()  # B logits
    preds_class2 = preds[:, :, 2].cpu().numpy()   # I logits
    preds_set= set()

    for i in range(len(text)):

        labels = set(v for _, v in A.iter(text[i].lower()))
        if labels :
            preds_set = preds_set.union(labels)

        print(preds_set)
        words = text[i].split()
        wid = word_ids[i].tolist()

        b_tokens, = np.where(preds_class1[i] > 1) #th=1
        i_tokens, = np.where(preds_class2[i] > 0) #th=0
        i_tokens = sorted(i_tokens) # i tokens sorted must

        spans = []
        span_end_idx = 0
        for b_token_idx in b_tokens:
            if span_end_idx > b_token_idx:
                continue
            nearest_i_token_idx = 0
            for i_token_idx in i_tokens:
                if i_token_idx > b_token_idx:
                    nearest_i_token_idx = i_token_idx
                    break

            if nearest_i_token_idx - b_token_idx > 10:
                continue

            if nearest_i_token_idx == 0 and preds_class1[i][b_token_idx] > 2 : # high confidence
                nearest_i_token_idx = b_token_idx
            if nearest_i_token_idx >= b_token_idx and b_token_idx > 0: #skip cls token
                span_word_indices=[]
                for token_idx in range(b_token_idx, nearest_i_token_idx+1):
                    word_idx = wid[token_idx]
                    if word_idx != -1 and word_idx < len(words):
                        span_word_indices.append(word_idx)
                if span_word_indices:
                    span_text = " ".join(words[w] for w in sorted(set(span_word_indices)))
                    spans.append(span_text)
                span_end_idx = nearest_i_token_idx
        if spans:
            li = list(set([clean_text(s, Lower=True) for s in spans]))
            if preds_set:
                li = clean_spans(li, list(preds_set), th=0.7)
            preds_set = preds_set.union(set(li))

        regex_text = list(regex_extract(text[i]))
        if regex_text:
            li = list(set([clean_text(s) for s in regex_text]))
            if preds_set:
                li = clean_spans(li, list(preds_set), th=0.6)
            preds_set = preds_set.union(set(li))

    return list(preds_set)