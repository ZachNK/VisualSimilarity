# (search_query.py) 쿼리 이미지 검색 및 결과 확인
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
INDEXDIR = DATA / "index"
QUERIES  = DATA / "queries"

# ==== 인덱스 로드 (+GPU 있을 때 래핑) ====
index = faiss.read_index(str(INDEXDIR / "faiss_index.bin"))
try:
    ngpu = faiss.get_num_gpus()
    # Flat/IVF 계열만 GPU 지원 (HNSW 등은 제외)
    if ngpu > 0 and "HNSW" not in type(index).__name__.upper():
        index = faiss.index_cpu_to_all_gpus(index)
        print(f"[FAISS] Using {ngpu} GPU(s) for search.")
    else:
        print("[FAISS] Using CPU.")
except Exception as e:
    print("[FAISS] GPU wrap skipped:", e)

# ==== 이미지 경로 목록 ====
image_paths = np.load(INDEXDIR / "image_paths.npy", allow_pickle=True)
# 안전 캐스팅(문자열/bytes/object 모두 커버)
image_paths = [str(p) for p in image_paths]

# ==== 쿼리 목록 ====
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
query_files = sorted([p for p in QUERIES.iterdir() if p.suffix.lower() in EXTS])
if not query_files:
    raise RuntimeError(f"No query images under: {QUERIES}")

K = min(5, index.ntotal)

for query_path in query_files:
    query_emb = get_embedding(str(query_path)).reshape(1, -1).astype("float32")
    distances, indices = index.search(query_emb, K)
    distances, indices = distances[0], indices[0]

    print(f"\nQuery: {query_path.name}")
    for rank, idx in enumerate(indices, start=1):
        fname = Path(image_paths[idx]).name
        print(f"  {rank}. {fname} (dist={distances[rank-1]:.4f})")
