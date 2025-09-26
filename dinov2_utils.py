# Scripts/dinov2_utils.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from transformers import AutoModel, AutoImageProcessor, AutoConfig

MODEL_ID = "facebook/dinov2-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- robust processor loader ---
def _load_processor(model_id: str):
    # 1) 시도: AutoImageProcessor
    try:
        return AutoImageProcessor.from_pretrained(model_id)
    except Exception as e1:
        # 2) 시도: Dinov2 전용
        try:
            from transformers import Dinov2ImageProcessor
            return Dinov2ImageProcessor.from_pretrained(model_id)
        except Exception as e2:
            # 3) 최후: ViT 계열 제너릭(표준 mean/std, shortest_edge=518)
            from transformers import ViTImageProcessor
            print("[warn] Falling back to ViTImageProcessor. "
                  "Consider upgrading 'transformers' or clearing HF cache.")
            return ViTImageProcessor(
                size={"shortest_edge": 518},  # Dinov2 Base 추천 입력
                resample=Image.BICUBIC,
                do_rescale=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )

processor = _load_processor(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

@torch.inference_mode()
def get_embedding(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(**inputs)
    else:
        out = model(**inputs)

    emb = out.last_hidden_state[:, 0, :]        # CLS
    return emb.squeeze(0).float().cpu().numpy().astype("float32")

@torch.inference_mode()
def get_embeddings(paths, batch_size=64) -> np.ndarray:
    """여러 이미지를 배치로 임베딩 (인덱스 빌드 속도↑)."""
    vecs = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**inputs)
        else:
            out = model(**inputs)

        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()  # (B, D)
        vecs.append(cls.astype("float32"))
        for im in imgs:
            im.close()
    return np.concatenate(vecs, axis=0)
