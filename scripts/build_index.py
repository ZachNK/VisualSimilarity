# (build_index.py) 임베딩 추출 및 FAISS 인덱스 구축
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import numpy as np
from pathlib import Path
from dinov2_utils import get_embedding

# ==== 경로 상수 ====
BASE     = Path(r"D:/KNK/_KSNU/_Projects/dino_test")
DATA     = BASE / "data"
IMAGES   = DATA / "images"
INDEXDIR = DATA / "index"
INDEXDIR.mkdir(parents=True, exist_ok=True)

# 처리 대상 확장자
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 이미지 경로 수집
image_paths = sorted([p for p in IMAGES.iterdir() if p.suffix.lower() in EXTS])
if not image_paths:
    raise RuntimeError(f"No images found under: {IMAGES}")

# (선택) 배치 추출 예시
# BATCH = 128
# embs = []
# for i in range(0, len(image_paths), BATCH):
#     for p in image_paths[i:i+BATCH]:
#         embs.append(get_embedding(str(p)))
# embeddings = np.ascontiguousarray(np.stack(embs, axis=0).astype("float32"))

# 단순 일괄 추출
embeddings = np.ascontiguousarray(
    np.stack([get_embedding(str(p)) for p in image_paths], axis=0).astype("float32")
)
dim = embeddings.shape[1]

# L2 인덱스 (DB/쿼리 모두 비정규화 전제)
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# 저장(항상 CPU 인덱스를 디스크에 저장하는 게 호환성 좋음)
faiss.write_index(index, str(INDEXDIR / "faiss_index.bin"))
np.save(INDEXDIR / "image_paths.npy", np.array([str(p) for p in image_paths]))

print(f"Indexed {len(image_paths)} images @ dim={dim}")
print("Index and image paths saved to:", INDEXDIR)
