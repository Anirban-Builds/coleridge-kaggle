from .pp import Preprocess

def preprocess(text, inference=True, tiny=True):
    pp = Preprocess(text, inference=inference, tiny=tiny) # initialization
    text_lst = []
    x = pp.__getitem__() # return pos_list
    for sent in x:
        if sent:
            sent, _ = zip(*sent)
            text_ = " ".join(sent)
            text_lst.append(text_)

    return text_lst