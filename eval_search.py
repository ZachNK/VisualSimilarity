# Scripts/eval_search.py — 거리 지표/Recall@K 계산 (GPU 검색, CSV 출력, per-env 경로, 타임스탬프 저장)

### ================================ 사용방법 ================================ 
#ex) orb2025 환경, visdrone
# (orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python eval_search.py --dataset visdrone --k 10

# 또는 경로 수동 지정
# (orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python eval_search.py --index-dir D:/.../data/index/visdrone --queries-dir D:/.../data/corpora/visdrone/images --k 10 --profile orb

### ================================ 사용방법 ================================ 

# 저장 경로 요약

# 결과 테이블(타임스탬프):
# results/<dataset>/<profile>/baseline/tables/search_results_<ts>.csv
# results/<dataset>/<profile>/baseline/tables/metrics_per_query_<ts>.csv
# results/<dataset>/<profile>/baseline/tables/search_perf_summary_<ts>.csv

# 최신 별칭(덮어쓰기):
# results/<dataset>/<profile>/baseline/tables/search_results.csv 등 3종

# 누적 로그(append):
# results/<dataset>/<profile>/baseline/tables/baseline_runs.csv

# GT: data/gt/<dataset>/gt_queries.csv

# 인덱스: data/index/<dataset>/{faiss_index.bin,image_paths.npy}


# Scripts/eval_search.py — 거리 지표/Recall@K 계산 (GPU 검색, CSV 출력, per-env 경로, 타임스탬프 저장)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, faiss, numpy as np, csv, re, time
from PIL import ImageFile
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from dinov2_utils import get_embedding

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------- base/profile ----------------
def resolve_base() -> Path:
    env = os.getenv("DINO_BASE")
    if env and Path(env).exists(): return Path(env)
    p_win = Path(r"D:/KNK/_KSNU/_Projects/dino_test")
    if p_win.exists(): return p_win
    p_wsl = Path("/mnt/d/KNK/_KSNU/_Projects/dino_test")
    if p_wsl.exists(): return p_wsl
    raise FileNotFoundError("Project base not found. Set DINO_BASE or check drive mount.")

def detect_profile(cli: str | None) -> str:
    if cli: return cli.strip()
    env = os.getenv("CONDA_DEFAULT_ENV") or (Path(os.getenv("VIRTUAL_ENV","")).name or "")
    low = env.lower()
    for k in ("orb","light","super"):
        if k in low: return k
    return "default"

BASE = resolve_base()
DATA = BASE / "data"
RESULTS = BASE / "results"
EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}  # +.tif

def default_queries_dir(dataset: str, queries_subdir: str | None) -> Path:
    """우선순위: --queries-subdir 지정 시 → corpora/<dataset>/<subdir> → corpora/<dataset>/images → corpora/<dataset>"""
    corp = DATA / "corpora" / dataset
    if queries_subdir:
        cand = corp / queries_subdir
        if cand.exists(): return cand
    c1 = corp / "images"
    return c1 if c1.exists() else corp

# ---------------- helpers ----------------
def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

_tf_pat = re.compile(r"(_rot\d{3}|_bright\d{3}|_scale\d{3})$", re.IGNORECASE)

def parse_base_transform(name: str) -> tuple[str, str]:
    """
    파일명에서 변환 토큰을 추출하고 base_id를 반환.
    우선순위: ..._<token>.ext  (token ∈ {rot###, bright###, scale###})
    매칭 없으면 transform='orig', base_id=stem
    """
    stem = Path(name).stem
    m = _tf_pat.search(stem)
    if not m:
        return stem, "orig"
    token = m.group(1).lstrip("_")
    base_id = stem[:m.start()]  # 토큰 제거
    return base_id, token

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate baseline FAISS retrieval (Recall@K etc.)")
    ap.add_argument("--dataset", type=str, help="데이터셋 이름(예: visdrone, sodaa, aihub, union)")
    ap.add_argument("--index-dir", type=str, default=None, help="인덱스 폴더(기본: data/index/<dataset>)")
    ap.add_argument("--queries-dir", type=str, default=None, help="쿼리 폴더(기본: data/corpora/<dataset>/[images|<queries-subdir>])")
    ap.add_argument("--queries-subdir", type=str, default=None, help="쿼리 하위폴더명(예: queries_scale). --queries-dir 보다 후순위")
    ap.add_argument("--k", type=int, default=None, help="top-K (기본: min(10, index.ntotal))")
    ap.add_argument("--profile", type=str, default=None, help="결과 하위 폴더명(orb/light/super). 미지정 시 자동 추론")
    ap.add_argument("--force-gt-regen", action="store_true", help="GT 캐시를 무시하고 강제로 재생성")
    args = ap.parse_args()

    if not args.dataset and (args.index_dir is None or args.queries_dir is None):
        raise SystemExit("Either --dataset OR both --index-dir and --queries-dir must be provided.")

    dataset = args.dataset or "custom"
    profile = detect_profile(args.profile)

    INDEXDIR = Path(args.index_dir) if args.index_dir else (DATA / "index" / dataset)
    QUERIES  = Path(args.queries_dir) if args.queries_dir else default_queries_dir(dataset, args.queries_subdir)
    GT_DIR   = DATA / "gt" / dataset
    GT_DIR.mkdir(parents=True, exist_ok=True)

    queries_tag = QUERIES.name  # ex) images / queries / queries_scale / custom
    OUTROOT  = RESULTS / dataset / profile / "baseline"
    TABLES   = OUTROOT / "tables"
    TABLES.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dataset={dataset} profile={profile}")
    print(f"[INFO] index_dir={INDEXDIR}")
    print(f"[INFO] queries_dir={QUERIES} (tag={queries_tag})")

    # ---- index & mapping
    idx_path = INDEXDIR / "faiss_index.bin"
    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    cpu_index = faiss.read_index(str(idx_path))
    index = cpu_index
    ngpu = 0
    try:
        ngpu = faiss.get_num_gpus()
        if ngpu > 0 and "HNSW" not in type(cpu_index).__name__.upper():
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions(); co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
            print("[FAISS] GPU search enabled (fp16).")
        else:
            print("[FAISS] CPU search.")
    except Exception as e:
        print("[FAISS] GPU wrap skipped:", e)

    # metric type
    is_ip = False
    try:
        is_ip = (index.metric_type == faiss.METRIC_INNER_PRODUCT)
    except Exception:
        pass
    metric_name = "IP" if is_ip else "L2"
    seq_key     = "score@K" if is_ip else "dist@K"

    # image paths
    paths_npy = INDEXDIR / "image_paths.npy"
    if not paths_npy.exists():
        raise FileNotFoundError(f"image_paths.npy not found: {paths_npy}")
    image_paths = [str(p) for p in np.load(paths_npy, allow_pickle=True)]
    fname_by_idx = {i: Path(p).name for i, p in enumerate(image_paths)}
    # stem→원본파일명(우선 .jpg 선호)
    stem_to_name = {}
    for name in fname_by_idx.values():
        st = Path(name).stem
        if st not in stem_to_name or name.lower().endswith(".jpg"):
            stem_to_name[st] = name

    # ---- GT 생성/로딩 (쿼리 파일 기준, 쿼리셋별로 분리)
    qfiles = sorted([p for p in QUERIES.rglob("*") if p.suffix.lower() in EXTS and p.is_file()])
    if not qfiles:
        raise RuntimeError(f"No query images under: {QUERIES}")

    gt_csv = GT_DIR / f"gt_queries_{queries_tag}.csv"  # 쿼리셋별 파일
    needs_regen = args.force_gt_regen or (not gt_csv.exists())
    if not needs_regen:
        try:
            with open(gt_csv, newline="", encoding="utf-8") as f:
                tmp = list(csv.DictReader(f))
            needs_regen = any(not (QUERIES / r["query"]).exists() for r in tmp)
        except Exception:
            needs_regen = True

    if needs_regen:
        with open(gt_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["query","base_id","transform"])
            for q in qfiles:
                base_id, transform = parse_base_transform(q.name)
                w.writerow([q.name, base_id, transform])

    # ---- 평가
    K = args.k if args.k is not None else max(1, min(10, index.ntotal))
    with open(gt_csv, newline="", encoding="utf-8") as f:
        gt = list(csv.DictReader(f))

    rows = []
    pbar = tqdm(gt, desc=f"Baseline search ({dataset}/{queries_tag})", unit="q", ncols=100)
    for r in pbar:
        qname = r["query"]; base_id = r["base_id"]; transform = r["transform"]
        qpath = QUERIES / qname
        try:
            t0 = time.perf_counter()
            qvec = get_embedding(str(qpath)).reshape(1, -1).astype("float32")
            if is_ip: qvec = l2_normalize(qvec)
            t1 = time.perf_counter()
            vals, inds = index.search(qvec, K)   # L2: dist(↓) / IP: score(↑)
            t2 = time.perf_counter()
        except Exception:
            pbar.set_postfix_str(f"error: {qname[:18]}")
            continue

        vals, inds = vals[0].tolist(), inds[0].tolist()
        pred_names = [fname_by_idx.get(i, f"IDX_{i}") for i in inds]

        gold = stem_to_name.get(base_id, f"{base_id}.jpg")
        rank_pos = (pred_names.index(gold)+1) if gold in pred_names else -1
        v_pos = vals[pred_names.index(gold)] if gold in pred_names else np.nan
        v_top1 = vals[0]; name_top1 = pred_names[0]

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
            "t_embed_ms": f"{(t1 - t0)*1000:.2f}",
            "t_search_ms": f"{(t2 - t1)*1000:.2f}",
        })

        pbar.set_postfix_str(f"rank@1={'OK' if rank_pos==1 else rank_pos} top1={name_top1[:16]}..")
    pbar.close()

    if not rows:
        raise RuntimeError("No queries evaluated.")

    # ---- 저장 (타임스탬프 + latest)
    ts = time.strftime("%Y%m%d-%H%M%S")
    OUTROOT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    results_csv_ts = TABLES / f"search_results_{ts}.csv"
    metrics_csv_ts = TABLES / f"metrics_per_query_{ts}.csv"
    perf_csv_ts    = TABLES / f"search_perf_summary_{ts}.csv"

    results_csv_latest = TABLES / "search_results.csv"
    metrics_csv_latest = TABLES / "metrics_per_query.csv"
    perf_csv_latest    = TABLES / "search_perf_summary.csv"

    write_csv(results_csv_ts, rows)

    # 3) 변형별 Recall@K & 분포 통계
    def recall_at_k(items, k):
        return sum(1 for r in items if int(r["rank_pos"]) != -1 and int(r["rank_pos"]) <= k) / max(len(items), 1)

    from statistics import mean
    by_tf = defaultdict(list)
    for r in rows: by_tf[r["transform"]].append(r)

    summary = []
    for tf, items in by_tf.items():
        rec1 = recall_at_k(items, 1)
        rec5 = recall_at_k(items, 5)
        rec10 = recall_at_k(items, 10)
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

    write_csv(metrics_csv_ts, summary)

    # perf 요약(+ 쿼리셋 정보 기록)
    N = len(rows)
    e_ms = [float(r["t_embed_ms"]) for r in rows]
    s_ms = [float(r["t_search_ms"]) for r in rows]
    total_embed_sec = sum(e_ms)/1000.0
    total_search_sec = sum(s_ms)/1000.0
    total_e2e_sec = total_embed_sec + total_search_sec
    device = "GPU" if (isinstance(ngpu, int) and ngpu > 0) else "CPU"

    perf_row = [{
        "dataset": dataset, "profile": profile, "queries_tag": queries_tag, "queries_dir": str(QUERIES),
        "K": K, "metric": metric_name, "device": device, "index_ntotal": getattr(index, "ntotal", 0),
        "avg_embed_ms": f"{(sum(e_ms)/max(N,1)):.2f}",
        "avg_search_ms": f"{(sum(s_ms)/max(N,1)):.2f}",
        "sum_embed_sec":  f"{total_embed_sec:.2f}",
        "sum_search_sec": f"{total_search_sec:.2f}",
        "sum_e2e_sec":    f"{total_e2e_sec:.2f}",
        "qps_embed":  f"{(N/total_embed_sec  if total_embed_sec  > 0 else 0):.2f}",
        "qps_search": f"{(N/total_search_sec if total_search_sec > 0 else 0):.2f}",
        "qps_e2e":    f"{(N/total_e2e_sec    if total_e2e_sec    > 0 else 0):.2f}",
        "run_ts": ts,
    }]
    write_csv(perf_csv_ts, perf_row)

    # latest 별칭(덮어쓰기)
    for src, dst in [(results_csv_ts, results_csv_latest),
                     (metrics_csv_ts, metrics_csv_latest),
                     (perf_csv_ts,    perf_csv_latest)]:
        try:
            dst.write_bytes(src.read_bytes())
        except Exception:
            pass

    # 누적 롤업(append)
    rollup = TABLES / "baseline_runs.csv"
    rollup_exists = rollup.exists()
    with open(rollup, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(perf_row[0].keys()))
        if not rollup_exists:
            w.writeheader()
        w.writerow(perf_row[0])

    print("Saved:")
    print(" ", results_csv_ts, metrics_csv_ts, perf_csv_ts)
    print(" ", results_csv_latest, metrics_csv_latest, perf_csv_latest)
    print(" ", rollup)

if __name__ == "__main__":
    main()
