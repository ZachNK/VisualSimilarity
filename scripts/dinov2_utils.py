# dinov2_utils.py — DINOv2 임베딩 추출 (GPU 자동, CPU 폴백)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np

MODEL_ID = "facebook/dinov2-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

@torch.inference_mode()
def get_embedding(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)

    # CLS 토큰 벡터
    emb = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
    return emb.squeeze().astype('float32')
