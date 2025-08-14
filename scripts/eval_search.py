# (eval_search.py) 거리 지표 및 Recall@K 계산
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import numpy as np
from pathlib import Path
from dinov2_utils import get_embedding
import csv, re
from collections import defaultdict

# ==== 경로 상수 ====
BASE     = Path(r"D:/KNK/_KSNU/_Projects/dino_test")
DATA     = BASE / "data"
IMAGES   = DATA / "images"
QUERIES  = DATA / "queries"
INDEXDIR = DATA / "index"
GT_DIR   = DATA / "gt"
RESULTS  = BASE / "results"
FIGS     = RESULTS / "figs"
TABLES   = RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)

# ==== 인덱스 로드 (+GPU 있으면 래핑) ====
index = faiss.read_index(str(INDEXDIR / "faiss_index.bin"))
try:
    ngpu = faiss.get_num_gpus()
    # Flat/IVF 계열만 GPU 지원(HNSW 등은 제외)
    if ngpu > 0 and "HNSW" not in type(index).__name__.upper():
        index = faiss.index_cpu_to_all_gpus(index)
        print(f"[FAISS] Using {ngpu} GPU(s) for search.")
    else:
        print("[FAISS] Using CPU.")
except Exception as e:
    print("[FAISS] GPU wrap skipped:", e)

# 메트릭 감지
try:
    is_ip = (index.metric_type == faiss.METRIC_INNER_PRODUCT)
except Exception:
    is_ip = "IP" in type(index).__name__.upper()  # fallback
metric_name = "IP" if is_ip else "L2"
seq_key     = "score@K" if is_ip else "dist@K"

# 이미지 경로 매핑(인덱스 순서와 동일)
image_paths = np.load(INDEXDIR / "image_paths.npy", allow_pickle=True)
image_paths = [str(p) for p in image_paths]
fname_by_idx = {i: Path(p).name for i, p in enumerate(image_paths)}

# ---- 1) GT 테이블 생성 ----
gt_csv = GT_DIR / "gt_queries.csv"
if not gt_csv.exists():
    with open(gt_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query","base_id","transform"])
        for q in sorted(QUERIES.glob("*.jpg")):  # 필요시 확장자 세트로 교체 가능
            m = re.match(r"([0-9]{7})_(.+)\.jpg", q.name)
            if not m:
                base_id, transform = q.stem, "orig"
            else:
                base_id, transform = m.group(1), m.group(2)
            w.writerow([q.name, base_id, transform])

# ---- 2) 유틸 ----
def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

# ---- 3) 검색 & 로깅 ----
K = min(10, index.ntotal)
results_csv = TABLES / "search_results.csv"
metrics_csv = TABLES / "metrics_per_query.csv"

rows = []
with open(gt_csv, newline="", encoding="utf-8") as f:
    gt = list(csv.DictReader(f))

for row in gt:
    qname = row["query"]; base_id = row["base_id"]; transform = row["transform"]
    qpath = QUERIES / qname
    qemb = get_embedding(str(qpath)).reshape(1, -1).astype("float32")
    # 선택: 임베딩 정규화 (인덱스 metric에 맞춰 on/off)
    if is_ip:
        qemb = l2_normalize(qemb)

    # vals: L2면 거리(작을수록 좋음), IP면 유사도(클수록 좋음)
    vals, inds = index.search(qemb, K)   
    vals, inds = vals[0].tolist(), inds[0].tolist()
    pred_names = [fname_by_idx[i] for i in inds]

    # rank of correct
    gold = f"{base_id}.jpg"
    rank_pos = (pred_names.index(gold)+1) if gold in pred_names else -1
    v_pos = vals[pred_names.index(gold)] if gold in pred_names else np.nan
    v_top1 = vals[0]; name_top1 = pred_names[0]

    # 로깅(Top-K 전체)
    rows.append({
        "query": qname,
        "transform": transform,
        "gold": gold,
        "rank_pos": rank_pos,
        ("score_pos" if is_ip else "d_pos"): v_pos,
        "top1_name": name_top1,
        ("top1_score" if is_ip else "top1_dist"): v_top1,
        "metric": metric_name,
        "pred@K": ";".join(pred_names),
        seq_key: ";".join(f"{v:.4f}" for v in vals),
    })

# 비어있을 가능성 방지
if not rows:
    raise RuntimeError("No queries found under queries/*.jpg")


# 저장
with open(results_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ---- 4) 변형별 Recall@K & 분포 통계 ----
def recall_at_k(items, k):
    return sum(1 for r in items if int(r["rank_pos"]) != -1 and int(r["rank_pos"]) <= k) / max(len(items), 1)

from statistics import mean
by_tf = defaultdict(list)
for r in rows:
    by_tf[r["transform"]].append(r)

summary = []
for tf, items in by_tf.items():
    rec1 = recall_at_k(items, 1)
    rec5 = recall_at_k(items, 5)
    rec10 = recall_at_k(items, 10)
    # 분포 비교용 평균: IP면 'score'(높을수록 좋음), L2면 'distance'(낮을수록 좋음)
    if is_ip:
        pos_vals = [float(r["score_pos"]) for r in items if str(r.get("score_pos")) != "nan"]
        neg_vals = [float(r["top1_score"]) for r in items if int(r["rank_pos"]) != 1]
        summary.append({
            "transform": tf, "metric": metric_name, "N": len(items),
            "Recall@1": f"{rec1:.4f}", "Recall@5": f"{rec5:.4f}", "Recall@10": f"{rec10:.4f}",
            "pos_mean_score": f"{(mean(pos_vals) if pos_vals else float('nan')):.2f}",
            "neg_mean_score": f"{(mean(neg_vals) if neg_vals else float('nan')):.2f}",
        })
    else:
        pos_vals = [float(r["d_pos"]) for r in items if str(r.get("d_pos")) != "nan"]
        neg_vals = [float(r["top1_dist"]) for r in items if int(r["rank_pos"]) != 1]
        summary.append({
            "transform": tf, "metric": metric_name, "N": len(items),
            "Recall@1": f"{rec1:.4f}", "Recall@5": f"{rec5:.4f}", "Recall@10": f"{rec10:.4f}",
            "pos_mean_dist": f"{(mean(pos_vals) if pos_vals else float('nan')):.2f}",
            "neg_mean_dist": f"{(mean(neg_vals) if neg_vals else float('nan')):.2f}",
        })

with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)

print("Saved:", results_csv, metrics_csv)
