import gc
from src.code_base.pipeline.gen_data \
import return_feas, gen_data, load_model
from src.code_base.pp.pp_head import preprocess
from src.code_base.utils.ckpt import ckpt
from src.code_base.utils.cfg import CFG
from src.code_base.pp.post_pp import post_process
def clean():
    gc.collect()

def inference(text : str):
    #will get text input as data
    data = preprocess(text) # return list of sentences
    if not data:
        return []
    dataloader = gen_data(data)
    model = load_model(CFG.backbone[0], ckpt[0])
    feas, word_ids = return_feas(model, dataloader)

    pred_lst = post_process(feas, word_ids, data)

    return pred_lst

