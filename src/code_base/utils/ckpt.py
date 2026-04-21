from huggingface_hub import hf_hub_download

REPO_ID = "Anirban0011/coleridge-bert-ner"

def get_path(filename, repo):
    path = hf_hub_download(repo_id=repo, filename=filename)
    return path

ckpt_path = get_path(repo=REPO_ID,
                            filename="coleridge_txt_model_bert-base-NER_full_128_25e.pth")
ckpt = [ckpt_path]