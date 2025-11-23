# CLIP
# Gemma3 4b
# LLaVa Architecture
# Hv is new vision dimension. Hq is text dimension
# Matching Dimension Hv = W * Zv where Zv = g(Xv) grid features
# Now Hv and Hq is the same dimension
# Xinstruction in t = random [Xv, Xq at i] or [Xq at i, Xv] if t = 1 else [Xq at t] where t is a turn
# Training Predict Response P(Xa|Xv, Xinstruction) = capital pi from i = 1 to L where L is sequence lenght * Ptheta(Xi|Xv, Xinstruction, < i, Xa, < i)
# frozen vision weight but train only LLM. trainable param is theta = { W, phi}
# batch 128, lr 2e-3, 1 epoch is enough

import torch
import torch.nn as nn
from models.clip_model import ClipModel
from models.gemma_model import GemmaModel
from torch.utils.data import Dataset
from PIL import Image
import requests

class LLaVaProjection(nn.Module):
    def __init__(self, in_dim : int, out_dim : int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        );
    
    def forward(self, x : torch.Tensor):
        return self.net(x)

class LLaVaBatchDataset(Dataset):
    def __init__(self, dataset, vision: ClipModel, llm: GemmaModel) -> None:
        super().__init__()
        self.data = dataset
        self.vision = vision
        self.llm = llm
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index : int):
        s = self.data[index]

        img = Image.open(requests.get(s["image_url"], stream=True).raw.data) #type: ignore
        img_input = self.vision.processor(images=img, return_tensors="pt") #type: ignore
        pixel : torch.Tensor = img_input['pixel_values'].squeeze(0)
        labels = "".join(s["labels"])
        cap : str = s["caption"] + " . " + labels
        text_token = self.llm.tokenizer(cap, return_tensors="pt", max_length=128, truncation=True)
        input_ids : torch.Tensor = text_token['input_ids'].squeeze(0)
        return {
            "pixel_values" : pixel,
            "input_ids" : input_ids
        }

    