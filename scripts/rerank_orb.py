# (rerank_orb.py) ORB+RANSAC 재정렬 스니펫
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import numpy as np
from pathlib import Path
from dinov2_utils import get_embedding  # 쿼리 임베딩 재사용
import cv2, csv
from collections import defaultdict

# ---------------- cfg ----------------
BASE = Path(r"D:/KNK/_KSNU/_Projects/dino_test")
DATA     = BASE / "data"
IMAGES   = DATA / "images"
QUERIES  = DATA / "queries"
INDEXDIR = DATA / "index"
GT_DIR   = DATA / "gt"
RESULTS  = BASE / "results"
FIGS     = RESULTS / "figs"
TABLES   = RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# 인덱스 로드 (+ GPU 있으면 래핑)
index = faiss.read_index(str(INDEXDIR / "faiss_index.bin"))
try:
    ngpu = faiss.get_num_gpus()
    if ngpu > 0 and "HNSW" not in type(index).__name__.upper():
        index = faiss.index_cpu_to_all_gpus(index)
        print(f"[FAISS] Using {ngpu} GPU(s) for search.")
    else:
        print("[FAISS] Using CPU.")
except Exception as e:
    print("[FAISS] GPU wrap skipped:", e)

# 인덱스 순서에 맞는 경로/이름 매핑
image_paths = np.load(INDEXDIR / "image_paths.npy", allow_pickle=True)
image_paths = [str(p) for p in image_paths]
fname_by_idx = {i: Path(p).name for i, p in enumerate(image_paths)}
fpath_by_idx = {i: Path(p)      for i, p in enumerate(image_paths)}

# GT / 쿼리
GT    = GT_DIR / "gt_queries.csv"
if not GT.exists():
    raise FileNotFoundError(f"GT file not found: {GT}\n(e.g., run eval_search.py once to generate it.)")

queries_dir = QUERIES
K = min(20, index.ntotal)
SAVE = TABLES / "search_results_rerank_orb.csv"

# --------- 유틸 -----------
def load_gray(path, long_side=960):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

orb = cv2.ORB_create(nfeatures=4000, fastThreshold=7, scoreType=cv2.ORB_HARRIS_SCORE)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def orb_ransac_score(q_img, d_img, ratio=0.75, ransac_thresh=5.0, min_match=8):
    kp1, des1 = orb.detectAndCompute(q_img, None)
    kp2, des2 = orb.detectAndCompute(d_img, None)
    if des1 is None or des2 is None:
        return 0.0, 0   # (score, inliers)
    
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < min_match:
        return 0.0, 0
    
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return 0.0, 0
    
    inliers = int(mask.sum())

    # 점수: 인라이어 비율(안정) * 인라이어 수(규모) 의 가중 합
    score = (inliers / max(len(good),1)) * 0.5 + (inliers / 200.0) * 0.5
    return float(score), inliers

# L2 거리 → 유사도(큰 값이 좋음)로 변환 (Top-K 내부 min-max 정규화)
def l2_to_sim(dists):
    d = np.array(dists, dtype=np.float32)
    d = np.nan_to_num(d, nan=(d.max() if d.size else 0))
    if d.size == 0: return d
    mx, mn = d.max(), d.min()
    if mx == mn: return np.ones_like(d)
    return (mx - d) / (mx - mn)  # 0~1

# 메인 루프: 쿼리마다 Top-K 후보 검색 → ORB 점수와 결합 → 재정렬
def main():
    out_rows = []
    # GT 로드 (transform/정답 매핑)
    with open(GT, newline="", encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))

    for row in gt_rows:
        qname = row["query"]; base_id = row["base_id"]; transform = row["transform"]
        qpath = queries_dir / qname
        qemb = get_embedding(str(qpath)).reshape(1,-1).astype("float32")

        # L2 인덱스 가정: 정규화하지 않은 쿼리
        dists, inds = index.search(qemb, K)
        dists, inds = dists[0].tolist(), inds[0].tolist()
        cand_names = [fname_by_idx[i] for i in inds]
        cand_paths = [fpath_by_idx[i] for i in inds]

        # ORB+RANSAC 점수
        qimg = load_gray(qpath)
        geom_scores, inliers = [], []
        for p in cand_paths:
            dimg = load_gray(p)
            s, inl = orb_ransac_score(qimg, dimg)
            geom_scores.append(s); inliers.append(inl)
        
        # 두 점수 결합 (0~1 정규화 후 가중합)
        sim_faiss = l2_to_sim(dists)              # L2 → [0,1] (큰 값이 좋음)
        sim_geom  = l2_to_sim(geom_scores)        # ORB 점수 → [0,1]
        alpha = 0.7                                # 가중치: FAISS 70%, ORB 30%
        combined = alpha*sim_faiss + (1-alpha)*sim_geom
        order = np.argsort(-combined)             # 내림차순(큰 값 → 상위)

        cand_names2 = [cand_names[i] for i in order]
        dists2      = [dists[i]      for i in order]
        geom2       = [geom_scores[i] for i in order]
        comb2       = [combined[i]   for i in order]

        gold = f"{base_id}.jpg"
        rank_pos = (cand_names2.index(gold)+1) if gold in cand_names2 else -1
        top1_name = cand_names2[0]
        top1_dist = dists2[0]   # 커버리지-리콜 비교용: 여전히 FAISS L2를 사용

        out_rows.append({
            "query": qname, "transform": transform, "gold": gold,
            "rank_pos": rank_pos,
            "top1_name": top1_name,
            "top1_dist": top1_dist,
            "top1_geom": geom2[0],
            "top1_combined": comb2[0],
            "pred@K": ";".join(cand_names2),
            "dist@K": ";".join(f"{d:.4f}" for d in dists2),
            "geom@K": ";".join(f"{g:.4f}" for g in geom2),
            "combined@K": ";".join(f"{c:.4f}" for c in comb2)
        })

    if not out_rows:
        raise RuntimeError("No rows generated. Check GT/queries/index.")
    with open(SAVE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print("Saved:", SAVE)

if __name__ == "__main__":
    main()
