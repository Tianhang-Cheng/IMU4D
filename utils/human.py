
from human_body_prior.body_model.body_model import BodyModel
from dataset_process.custom_path import smplx_model_path

def load_smplx_model() -> BodyModel:
    smplx_model = BodyModel(bm_fname=smplx_model_path, num_betas=10, model_type='smplx')
    smplx_model.eval()
    for p in smplx_model.parameters():
        p.requires_grad = False
    smplx_model.cuda()
    return smplx_model
