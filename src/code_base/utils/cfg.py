import torch

class CFG:
    r"""
    default config class
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    n_epochs = 5
    fold_id = 0
    init_lr = 3e-4
    num_classes = 3
    max_length = 128
    num_workers = 8
    embed_size = 512
    backbone = ["dslim/bert-base-NER"]
