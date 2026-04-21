from src.code_base.pipeline import NERDataset, TextEncoder
from torch.utils.data import DataLoader
from src.code_base.utils import CFG
import torch

def gen_data(data,
             max_length = CFG.max_length,
             tkn_pth=CFG.backbone[0],
             batch_size = CFG.batch_size,
             shuffle = False,
             num_workers = CFG.num_workers,
             gen_feat_only = True
             ):
    data_txt = NERDataset(data,
                          max_length=max_length,
                          gen_feat_only = gen_feat_only,
                          tokenizer = tkn_pth,
                          data_is_list=True
                          )
    dataloader_txt = DataLoader(data_txt, batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                )
    return dataloader_txt

def load_model(backbone, ckpt_path):
    model = TextEncoder(num_classes = CFG.num_classes,
                        backbone = backbone,
                        eval_model=True,
                        embed_size=CFG.embed_size)

    ckpt = torch.load(ckpt_path, weights_only=True, map_location=CFG.device)

    new_state_dict = {}
    for k, v in ckpt.items():
        new_key = k.replace("module.", "")  # remove module. prefix
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model = model.to(CFG.device)
    print(f"model {backbone} loaded successfully")
    return model

class gen_feas:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def gen_txt_feas(self):

        self.model.eval()
        bar = self.dataloader

        FEAS = []
        WORD_IDS = []

        with torch.no_grad():
            for batch_idx, (inp_ids, att_masks, word_ids) in enumerate(bar):
                inp_ids, att_masks, word_ids = inp_ids.to(CFG.device), att_masks.to(CFG.device), \
                                        word_ids.to(CFG.device),

                logits = self.model(inp_ids, att_masks)
                FEAS += [logits.detach().cpu()]
                WORD_IDS += [word_ids.cpu()]

        FEAS = torch.cat(FEAS).cpu().numpy()
        FEAS = FEAS.reshape(-1, CFG.max_length, 3)
        WORD_IDS = torch.cat(WORD_IDS).cpu().numpy()
        return FEAS, WORD_IDS

def return_feas(model, dataloader):
    feas, word_ids = gen_feas(model, dataloader).gen_txt_feas()
    feas = torch.tensor(feas)
    return feas, word_ids