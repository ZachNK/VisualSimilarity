# Scripts/rerank_orb.py — ORB+RANSAC 재정렬 (per-env 저장, tqdm, 타임스탬프/롤업)

### ================================ 사용방법 ================================ 
#ex) orb2025 환경, visdrone
# (orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python rerank_orb.py --dataset visdrone --k 20 --alpha 0.7

# 또는 경로 수동 지정 + 프로파일 강제
# python rerank_orb.py --index-dir D:/.../data/index/visdrone --queries-dir D:/.../data/corpora/visdrone/images --k 20 --alpha 0.7 --profile orb

### ================================ 사용방법 ================================ 

# 저장 규칙 (이제 통일)

# 출력 폴더: results/<dataset>/<profile>/orb/tables/

# 스냅샷(ts): search_results_rerank_orb_<YYYYmmdd-HHMMSS>.csv, rerank_perf_summary_orb_<YYYYmmdd-HHMMSS>.csv

# latest 별칭: search_results_rerank_orb.csv, rerank_perf_summary_orb.csv

# 누적 로그: rerank_orb_runs.csv

# 입력:

# 인덱스: data/index/<dataset>/{faiss_index.bin,image_paths.npy}

# 쿼리: data/corpora/<dataset>/images(없으면 data/corpora/<dataset>)

# GT: data/gt/<dataset>/gt_queries.csv (eval_search.py 실행 시 생성)

# Scripts/rerank_orb.py — ORB+RANSAC 재정렬 (게이트+α결합+Top-K, per-env 저장, tqdm, 타임스탬프/롤업)
# 사용예:
#   (orb2025) python rerank_orb.py --dataset visdrone --k 20 --alpha 0.2
#   python rerank_orb.py --index-dir D:/.../data/index/visdrone --queries-dir D:/.../data/corpora/visdrone/images --k 20 --alpha 0.2 --profile orb
# Scripts/rerank_orb.py — ORB+RANSAC 재정렬 (게이트+α결합+Top-K, variants 저장, 롤업/latest 유지)

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, csv, time
# --- robust imports: project root + 두 가지 경로 모두 지원 ---
import sys
from pathlib import Path
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# DINO 임베딩
try:
    from scripts.dinov2_utils import get_embedding
except Exception:
    # (직접 실행 대비)
    from dinov2_utils import get_embedding

# 게이트/재정렬 유틸
try:
    from scripts.rerank_utils import GateCfg, apply_rerank, combine_scores
except Exception:
    from rerank_utils import GateCfg, apply_rerank, combine_scores

import faiss, numpy as np, cv2
from PIL import ImageFile
from tqdm import tqdm



ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------- base/profile ----------------
def resolve_base() -> Path:
    env = os.getenv("DINO_BASE")
    if env and Path(env).exists(): return Path(env)
    p_win = Path(r"D:/KNK/_KSNU/_Projects/dino_test")
    if p_win.exists(): return p_win
    p_wsl = Path("/mnt/d/KNK/_KSNU/_Projects/dino_test")
    if p_wsl.exists(): return p_wsl
    raise FileNotFoundError("Project base not found. Set DINO_BASE.")

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
EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tiff"}

def default_queries_dir(dataset: str) -> Path:
    c1 = DATA / "corpora" / dataset / "images"
    return c1 if c1.exists() else (DATA / "corpora" / dataset)

# ---------------- helpers for variants ----------------
def _fmtf(x: float) -> str:
    s = f"{x:.3f}".rstrip('0').rstrip('.')
    return s.replace('.', 'p') if '.' in s else s

def make_variant_tag(stage: str, args) -> str:
    gateflag = "gate" if not args.gate_off else "nogate"
    return (
        f"{stage}-a{_fmtf(args.alpha)}-k{args.k}-"
        f"gm{_fmtf(args.gate_margin)}-gi{_fmtf(args.gate_inlier)}-"
        f"gs{_fmtf(getattr(args,'gate_scale',1.0))}-{gateflag}"
    )

# ---------------- ORB utils ----------------
def load_gray(path: Path, long_side=960):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

orb = cv2.ORB_create(nfeatures=4000, fastThreshold=7, scoreType=cv2.ORB_HARRIS_SCORE)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def orb_ransac_score(q_img, d_img, ratio=0.75, ransac_thresh=5.0, min_match=8):
    """
    반환:
      score: 휴리스틱 점수(로그용)
      inliers: RANSAC 인라이어 수
      inlier_ratio: 인라이어 비율(0..1) — 게이트/결합용
    """
    kp1, des1 = orb.detectAndCompute(q_img, None)
    kp2, des2 = orb.detectAndCompute(d_img, None)
    if des1 is None or des2 is None:
        return 0.0, 0, 0.0
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in knn for m, n in [pair] if (n is not None and m.distance < ratio * n.distance)]
    if len(good) < min_match:
        return 0.0, 0, 0.0
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return 0.0, 0, 0.0
    inliers = int(mask.sum())
    denom = max(1, min(len(kp1), len(kp2)))
    inlier_ratio = float(inliers) / float(denom)
    score = (inliers / max(len(good),1)) * 0.5 + (inliers / 200.0) * 0.5
    return float(score), inliers, float(inlier_ratio)

def l2_to_sim(d):
    d = np.asarray(d, np.float32)
    d = np.nan_to_num(d, nan=(d.max() if d.size else 0))
    if d.size == 0: return d
    mx, mn = d.max(), d.min()
    if mx == mn: return np.ones_like(d)
    return (mx - d) / (mx - mn)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Rerank with ORB+RANSAC on FAISS Top-K candidates.")
    ap.add_argument("--dataset", type=str, help="데이터셋(예: visdrone, sodaa, aihub, union)")
    ap.add_argument("--index-dir", type=str, default=None, help="인덱스 폴더(기본: data/index/<dataset>)")
    ap.add_argument("--queries-dir", type=str, default=None, help="쿼리 폴더(기본: data/corpora/<dataset>/[images])")
    # 공통 재정렬 파라미터(형평성 유지)
    ap.add_argument("--alpha", type=float, default=0.2)      # 임베딩:기하 결합 가중치
    ap.add_argument("--k",     type=int,   default=20)       # top-K 재정렬 폭
    ap.add_argument("--gate.margin", dest="gate_margin", type=float, default=0.03)
    ap.add_argument("--gate.inlier", dest="gate_inlier", type=float, default=0.08)
    ap.add_argument("--gate.off",     dest="gate_off", action="store_true")
    ap.add_argument("--gate.scale",   dest="gate_scale", type=float, default=1.0,
                    help="게이트 임계 스케일(0.7=완화, 1.0=기본, 1.3=강화)")
    # 메타
    ap.add_argument("--profile", type=str, default=None, help="결과 하위 폴더(orb/light/super). 미지정 시 자동")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"], help="임베딩 디바이스 힌트")
    # ORB 파라미터(선택)
    ap.add_argument("--long-side", type=int, default=960, help="이미지 리사이즈 최대 변")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio")
    ap.add_argument("--ransac", type=float, default=5.0, help="RANSAC reproj thresh")
    ap.add_argument("--min-match", type=int, default=8, help="최소 매칭 수")
    args = ap.parse_args()

    if not args.dataset and (args.index_dir is None or args.queries_dir is None):
        raise SystemExit("Either --dataset OR both --index-dir and --queries-dir must be provided.")

    dataset = args.dataset or "custom"
    profile = detect_profile(args.profile)
    stage   = "orb"  # ORB 리랭크 스테이지 고정

    INDEXDIR = Path(args.index_dir) if args.index_dir else (DATA / "index" / dataset)
    QUERIES  = Path(args.queries_dir) if args.queries_dir else default_queries_dir(dataset)

    queries_tag = Path(QUERIES).name
    print(f"[INFO] queries_dir={QUERIES} (tag={queries_tag})")

    GT_DIR   = DATA / "gt" / dataset
    GT_DIR.mkdir(parents=True, exist_ok=True)

    OUTROOT  = RESULTS / dataset / profile / stage
    TABLES   = OUTROOT / "tables"
    TABLES.mkdir(parents=True, exist_ok=True)

    # Variant 서브폴더
    TAG = make_variant_tag(stage, args)
    TABLES_VAR = TABLES / "variants" / TAG
    TABLES_VAR.mkdir(parents=True, exist_ok=True)

    if args.device in ("cuda","cpu"):
        os.environ["DINO_DEVICE"] = args.device

    # ---- index (+GPU)
    idx_path = INDEXDIR / "faiss_index.bin"
    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    cpu_index = faiss.read_index(str(idx_path))
    index = cpu_index
    ngpu = 0
    try:
        ngpu = faiss.get_num_gpus()
        if ngpu > 0 and "HNSW" not in type(cpu_index).__name__.upper():
            index = faiss.index_cpu_to_all_gpus(cpu_index)
            print(f"[FAISS] Using {ngpu} GPU(s) for search.")
        else:
            print("[FAISS] Using CPU.")
    except Exception as e:
        print("[FAISS] GPU wrap skipped:", e)

    # ---- image paths
    paths_npy = INDEXDIR / "image_paths.npy"
    if not paths_npy.exists():
        raise FileNotFoundError(f"image_paths.npy not found: {paths_npy}")
    image_paths = [str(p) for p in np.load(paths_npy, allow_pickle=True)]
    fname_by_idx = {i: Path(p).name for i, p in enumerate(image_paths)}
    fpath_by_idx = {i: Path(p)      for i, p in enumerate(image_paths)}

    # 대표 파일명(stem 기준) 매핑(.jpg 우선)
    stem_to_name = {}
    for name in fname_by_idx.values():
        st = Path(name).stem
        if st not in stem_to_name or name.lower().endswith(".jpg"):
            stem_to_name[st] = name

    # ---- GT
    qtag = (Path(args.queries_dir).name if args.queries_dir else default_queries_dir(dataset).name)
    cand1 = GT_DIR / f"gt_queries_{qtag}.csv"
    cand2 = GT_DIR / "gt_queries.csv"
    gt_csv = cand1 if cand1.exists() else cand2
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT not found: {cand1} or {cand2} (run eval_search.py first)")

    with open(gt_csv, newline="", encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))
    if not gt_rows:
        raise RuntimeError("Empty GT file.")

    K = args.k if args.k is not None else max(1, min(20, index.ntotal))
    alpha = float(args.alpha)

    print(f"[INFO] dataset={dataset}, profile={profile}, stage={stage}, index.ntotal={getattr(index,'ntotal',0)}, K={K}, alpha={alpha}, device={'GPU' if ngpu>0 else 'CPU'}")

    out_rows = []
    embed_ms_all, search_ms_all, geom_ms_all, total_ms_all = [], [], [], []
    rerank_applied_cnt, rerank_k_sum = 0, 0

    pbar_q = tqdm(gt_rows, desc=f"Rerank ORB ({dataset})", unit="q", ncols=100)
    for row in pbar_q:
        qname = row["query"]; base_id = row["base_id"]; transform = row["transform"]
        qpath = QUERIES / qname

        t_total0 = time.perf_counter()
        # 1) 임베딩
        try:
            t_e0 = time.perf_counter()
            qemb = get_embedding(str(qpath)).reshape(1,-1).astype("float32")
            t_e1 = time.perf_counter()
        except Exception:
            pbar_q.set_postfix_str(f"embed error: {qname[:18]}")
            continue

        # 2) FAISS 검색
        try:
            t_s0 = time.perf_counter()
            dists, inds = index.search(qemb, K)
            t_s1 = time.perf_counter()
        except Exception:
            pbar_q.set_postfix_str(f"search error: {qname[:18]}")
            continue

        dists = dists[0].tolist()
        inds  = inds[0].tolist()
        cand_names = [fname_by_idx[i] for i in inds]
        cand_paths = [fpath_by_idx[i] for i in inds]

        # 3) ORB+RANSAC → inlier_ratio(0..1)
        try:
            qimg = load_gray(qpath, long_side=args.long_side)
        except Exception:
            pbar_q.set_postfix_str(f"load qimg error: {qname[:18]}")
            continue

        geom_scores_raw, inlier_ratios = [], []
        t_g0 = time.perf_counter()
        pbar_c = tqdm(cand_paths, desc=f"  candidates({qname})", unit="img", leave=False, ncols=100)
        for p in pbar_c:
            try:
                dimg = load_gray(p, long_side=args.long_side)
                s, inl, ir = orb_ransac_score(
                    q_img=qimg, d_img=dimg,
                    ratio=args.ratio, ransac_thresh=args.ransac, min_match=args.min_match
                )
            except Exception:
                s, inl, ir = 0.0, 0, 0.0
            geom_scores_raw.append(s)
            inlier_ratios.append(ir)
            pbar_c.set_postfix_str(f"last={Path(p).name[:18]}.. inl={inl} ir={ir:.3f}")
        pbar_c.close()
        t_g1 = time.perf_counter()

        # 4) 게이트+α결합+Top-K 재정렬
        nn_dists = np.asarray(dists, dtype=float)
        cand_ids = np.asarray(inds,  dtype=int)
        geom_rat = np.asarray(inlier_ratios, dtype=float)

        gcfg = GateCfg(
            use_gate=(not args.gate_off),
            tau_margin=args.gate_margin * args.gate_scale,
            tau_inlier=args.gate_inlier * args.gate_scale
        )

        new_ids, applied, kk = apply_rerank(
            nn_dists=nn_dists,
            candidate_ids=cand_ids,
            geom_scores=geom_rat,
            alpha=alpha,
            k=K,
            gcfg=gcfg
        )
        rerank_applied_cnt += int(applied)
        rerank_k_sum += int(kk)

        # 새 순서 반영
        id_to_pos = {i: j for j, i in enumerate(cand_ids.tolist())}
        order_idx = [id_to_pos[i] for i in new_ids.tolist()]

        cand_names2 = [cand_names[j] for j in order_idx]
        dists2      = [dists[j]      for j in order_idx]
        geom_raw2   = [geom_scores_raw[j] for j in order_idx]
        inlier2     = [inlier_ratios[j]    for j in order_idx]

        # 결합 점수 로깅(Top-kk만)
        final_full = np.full(len(nn_dists), np.nan, dtype=float)
        if kk > 1:
            tmp = combine_scores(nn_dists[:kk], geom_rat[:kk], alpha)
            final_full[:kk] = tmp
        combined2 = [float(final_full[j]) for j in order_idx]

        # 5) 랭크/타임/로그
        gold = stem_to_name.get(base_id, f"{base_id}.jpg")
        rank_pos = (cand_names2.index(gold)+1) if gold in cand_names2 else -1
        top1_name = cand_names2[0]
        top1_dist = dists2[0]
        top1_inlier_ratio = inlier2[0]

        t_total1 = time.perf_counter()
        t_embed_ms = (t_e1 - t_e0) * 1000.0
        t_search_ms = (t_s1 - t_s0) * 1000.0
        t_geom_ms   = (t_g1 - t_g0) * 1000.0
        t_total_ms  = (t_total1 - t_total0) * 1000.0

        embed_ms_all.append(t_embed_ms)
        search_ms_all.append(t_search_ms)
        geom_ms_all.append(t_geom_ms)
        total_ms_all.append(t_total_ms)

        out_rows.append({
            "query": qname, "transform": transform, "gold": gold,
            "rank_pos": rank_pos,
            "top1_name": top1_name,
            "top1_dist": f"{top1_dist:.6f}",
            "top1_geom": f"{top1_inlier_ratio:.6f}",   # inlier_ratio(0..1)
            "top1_geom_raw": f"{geom_raw2[0]:.6f}",    # ORB 휴리스틱 점수(참고)
            "pred@K": ";".join(cand_names2),
            "dist@K": ";".join(f"{d:.6f}" for d in dists2),
            "inlier_ratio@K": ";".join(f"{g:.6f}" for g in inlier2),
            "geom_score@K": ";".join(f"{g:.6f}" for g in geom_raw2),
            "combined@K": ";".join("" if np.isnan(c) else f"{c:.6f}" for c in combined2),
            "K": K,
            "alpha": alpha,
            "gate_margin": args.gate_margin,
            "gate_inlier": args.gate_inlier,
            "gate_scale": args.gate_scale,
            "gate_on": int(not args.gate_off),
            "rerank_applied": int(applied),
            "rerank_k": int(kk),
            "t_embed_ms": f"{t_embed_ms:.2f}",
            "t_search_ms": f"{t_search_ms:.2f}",
            "t_geom_ms": f"{t_geom_ms:.2f}",
            "t_total_ms": f"{t_total_ms:.2f}",
        })

        pbar_q.set_postfix_str(
            f"rank@1={'OK' if rank_pos==1 else rank_pos}  top1={top1_name[:16]}..  "
            f"t(ms):E{t_embed_ms:.0f}/S{t_search_ms:.0f}/G{t_geom_ms:.0f}  "
            f"gate={'ON' if not args.gate_off else 'OFF'} a={alpha:.2f} K={K} "
            f"{'rerank' if applied else 'keep'}"
        )
    pbar_q.close()

    if not out_rows:
        raise RuntimeError("No rows generated.")

    # ---- 저장 (Variant ts + latest 별칭 + 롤업)
    ts = time.strftime("%Y%m%d-%H%M%S")

    # variant 전용 ts 파일
    results_csv_ts_var = TABLES_VAR / f"search_results_rerank_{stage}_{TAG}_{ts}.csv"
    perf_csv_ts_var    = TABLES_VAR / f"rerank_perf_summary_{stage}_{TAG}_{ts}.csv"

    with open(results_csv_ts_var, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    with open(perf_csv_ts_var, "w", newline="", encoding="utf-8") as f:
        # perf_row는 아래에서 준비
        pass

    # perf 요약
    N = len(out_rows)
    def _avg(xs): return (sum(xs)/max(len(xs),1)) if xs else 0.0
    sum_embed_sec  = sum(embed_ms_all)/1000.0
    sum_search_sec = sum(search_ms_all)/1000.0
    sum_geom_sec   = sum(geom_ms_all)/1000.0
    sum_total_sec  = sum(total_ms_all)/1000.0
    device_str = "GPU" if ngpu>0 else "CPU"

    perf_row = [{
        "dataset": dataset, "profile": profile, "stage": stage,
        "queries_tag": queries_tag, "queries_dir": str(QUERIES),
        "K": K, "alpha": alpha, "device": device_str, "index_ntotal": getattr(index,"ntotal",0),
        "gate_margin": args.gate_margin, "gate_inlier": args.gate_inlier, "gate_scale": args.gate_scale,
        "gate_on": int(not args.gate_off),
        "variant_tag": TAG,
        "rerank_applied_ratio": f"{(rerank_applied_cnt/max(N,1)):.4f}",
        "rerank_k_mean": f"{(rerank_k_sum/max(rerank_applied_cnt,1)):.2f}" if rerank_applied_cnt>0 else "0.00",
        "avg_embed_ms": f"{_avg(embed_ms_all):.2f}",
        "avg_search_ms": f"{_avg(search_ms_all):.2f}",
        "avg_geom_ms":   f"{_avg(geom_ms_all):.2f}",
        "avg_total_ms":  f"{_avg(total_ms_all):.2f}",
        "sum_embed_sec":  f"{sum_embed_sec:.2f}",
        "sum_search_sec": f"{sum_search_sec:.2f}",
        "sum_geom_sec":   f"{sum_geom_sec:.2f}",
        "sum_total_sec":  f"{sum_total_sec:.2f}",
        "qps_embed":  f"{(N/sum_embed_sec  if sum_embed_sec  > 0 else 0):.2f}",
        "qps_search": f"{(N/sum_search_sec if sum_search_sec > 0 else 0):.2f}",
        "qps_geom":   f"{(N/sum_geom_sec   if sum_geom_sec   > 0 else 0):.2f}",
        "qps_total":  f"{(N/sum_total_sec  if sum_total_sec  > 0 else 0):.2f}",
        "run_ts": ts,
    }]

    # perf CSV (variant ts)
    with open(perf_csv_ts_var, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(perf_row[0].keys()))
        w.writeheader(); w.writerows(perf_row)

    print("Saved (variant):", results_csv_ts_var)
    print("Saved (variant):", perf_csv_ts_var)

    # latest 별칭(루트 tables/)
    results_csv_latest = TABLES / f"search_results_rerank_{stage}.csv"
    perf_csv_latest    = TABLES / f"rerank_perf_summary_{stage}.csv"
    try: results_csv_latest.write_bytes(results_csv_ts_var.read_bytes())
    except Exception: pass
    try: perf_csv_latest.write_bytes(perf_csv_ts_var.read_bytes())
    except Exception: pass

    # 롤업(append)
    rollup_csv = TABLES / f"rerank_{stage}_runs.csv"
    rollup_exists = rollup_csv.exists()
    with open(rollup_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(perf_row[0].keys()))
        if not rollup_exists: w.writeheader()
        w.writerow(perf_row[0])

if __name__ == "__main__":
    main()
