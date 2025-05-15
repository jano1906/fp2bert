from mol2vec.features import mol2alt_sentence
from rdkit import Chem
import sys
import os
import torch
from typing import Literal, Optional
import numpy as np
from tqdm import tqdm

from transformers import ElectraModel
pre_trained_dir = os.path.join(os.path.dirname(__file__), "pre-trained")
sys.path.append(pre_trained_dir)
from my_tokenizers2 import SmilesTokenizer

MAX_SMI_LEN = 256

root = os.path.dirname(__file__)
model_path = os.path.join(root, "fingerprints_smile_output256")
MODEL_PATHS = {"FP-BERT": model_path}

class State:
    model: Optional[ElectraModel] = None
    tokenizer: Optional[SmilesTokenizer] = None

    model_name: Optional[str] = None
    device: Optional[str] = None
    batch_size: Optional[int] = None

    initialized: bool = False

def setup(model_name: str, device: Literal["cpu", "cuda"], batch_size: int):
    if not os.path.isdir(model_path):
        raise RuntimeError(f"Download checkpoints from `https://figshare.com/articles/software/fingerprints_smile_output256_tar_gz/19609440?file=34830750` and untar to `{model_path}`.")
    model = ElectraModel.from_pretrained(MODEL_PATHS[model_name])
    model = model.to(device)
    model.eval()
    
    State.tokenizer = SmilesTokenizer(vocab_file = os.path.join(pre_trained_dir, 'mol2vec_vocabs.txt'))
    State.model = model
    State.model_name = model_name
    State.device = device
    State.batch_size = batch_size

    State.initialized = True


def encode(input_file: str, output_file: str):
    with open(input_file, "r") as f:
        smiles = f.readlines()

    if not State.initialized:
        raise RuntimeError("Service is not setup, call 'setup' before 'encode'.")
    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles), State.batch_size), "Encoding FP-BERT:"):
            batch = smiles[i:i+State.batch_size]
            batch = [Chem.MolFromSmiles(x) for x in batch]
            batch = [" ".join(mol2alt_sentence(x,1)) for x in batch]
            inputs = State.tokenizer(batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(State.device)[:, :256] for k, v in inputs.items()}
            outs = State.model(**inputs)
            out = outs[0]
            mask = inputs["attention_mask"]
            counts = mask.sum(dim=-1, keepdim=True)
            embed = (out * mask.unsqueeze(-1)).sum(dim=-2) / counts
            outputs.append(embed.cpu())

    outputs = torch.concat(outputs)
    outputs = outputs.numpy()
    with open(output_file, "wb") as f:
        np.save(f, outputs)

