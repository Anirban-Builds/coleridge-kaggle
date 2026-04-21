import pandas as pd
from src.code_base.utils.utils import clean_text

url1 = "https://huggingface.co/datasets/Anirban0011/coleridge-data/resolve/main/data_set1.csv"
url2 = "https://huggingface.co/datasets/Anirban0011/coleridge-data/resolve/main/new_set.csv"


df1 = pd.read_csv(url1)
df2 = pd.read_csv(url2)

known_labels = [clean_text(i, Lower=True) for i in df1.title.tolist()]
known_labels += [clean_text(i, Lower=True) for i in df2.title.tolist() if len(i.split()) > 2]
known_labels = list(set(known_labels))