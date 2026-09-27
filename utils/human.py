
import torch

from human_body_prior.body_model.body_model import BodyModel
from dataset_process.custom_path import smplx_model_path

def load_smplx_model(device=None) -> BodyModel:
    """Build the SMPL-X body model on ``device`` (default: this process's GPU).

    A bare ``.cuda()`` always lands on cuda:0, so under multi-GPU every rank put
    the body model on rank 0's device while training on its own -- mixing the two
    raises "CUDA error: an illegal memory access". ``torch.cuda.current_device()``
    follows the device accelerate assigns to each rank.
    """
    smplx_model = BodyModel(bm_fname=smplx_model_path, num_betas=10, model_type='smplx')
    smplx_model.eval()
    for p in smplx_model.parameters():
        p.requires_grad = False
    if device is None:
        device = (torch.device(f"cuda:{torch.cuda.current_device()}")
                  if torch.cuda.is_available() else torch.device("cpu"))
    smplx_model.to(device)
    return smplx_model
