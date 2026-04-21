import ahocorasick

def jaccard(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def replace_label(s, cand, th=0.5):
    for known in cand:
        if jaccard(s, known) >= th:
            return known
    return s

def clean_spans(spans, labels, th=0.5):
    A_ = ahocorasick.Automaton()
    for s in spans:
        A_.add_word(s, s)
    A_.make_automaton()
    cand = set(v for _, v in A_.iter(". ".join(labels)))
    return [replace_label(s, cand, th) for s in spans]