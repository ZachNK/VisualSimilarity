# scripts/rerank_orb_fast.py
# ORB+RANSAC rerank (baseline reuse + sharding + perf logs)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, csv, time, sys
from pathlib import Path

import faiss, numpy as np, cv2
from PIL import ImageFile
from tqdm import tqdm

# ---- project root import path ----
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# emb/rerank utils (robust import)
try:
    from scripts.dinov2_utils import get_embedding
except Exception:
    from dinov2_utils import get_embedding

try:
    from scripts.rerank_utils import GateCfg, apply_rerank, combine_scores
except Exception:
    from rerank_utils import GateCfg, apply_rerank, combine_scores

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

# ---------------- helpers ----------------
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

def default_queries_dir(dataset: str) -> Path:
    c1 = DATA / "corpora" / dataset / "images"
    return c1 if c1.exists() else (DATA / "corpora" / dataset)

# ---------------- vision ----------------
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
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def orb_ransac_score(q_img, d_img, ratio=0.75, ransac_thresh=5.0, min_match=8):
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

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="FAST ORB Rerank: baseline reuse + sharding")
    ap.add_argument("--dataset", type=str, help="visdrone/sodaa/aihub/...")
    ap.add_argument("--index-dir", type=str, default=None, help="data/index/<dataset>")
    ap.add_argument("--queries-dir", type=str, default=None, help="data/corpora/<dataset>/images")
    # core params
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--k",     type=int,   default=20)
    ap.add_argument("--gate.margin", dest="gate_margin", type=float, default=0.03)
    ap.add_argument("--gate.inlier", dest="gate_inlier", type=float, default=0.08)
    ap.add_argument("--gate.off",     dest="gate_off", action="store_true")
    ap.add_argument("--gate.scale",   dest="gate_scale", type=float, default=1.0)
    # meta
    ap.add_argument("--profile", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--omp-threads", type=int, default=None, help="override OMP_NUM_THREADS inside script")
    # ORB params
    ap.add_argument("--long-side", type=int, default=960)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac", type=float, default=5.0)
    ap.add_argument("--min-match", type=int, default=8)
    # FAST mode
    ap.add_argument("--reuse-baseline", type=str, default=None,
                    help="path to baseline search_results.csv to reuse Top-K (skip embedding+search)")
    # sharding
    ap.add_argument("--split", type=int, default=1, help="num shards")
    ap.add_argument("--shard", type=int, default=0, help="this shard id [0..split-1]")
    args = ap.parse_args()

    # honor omp threads (still recommend exporting in shell)
    if args.omp_threads and args.omp_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    if not args.dataset and (args.index_dir is None or args.queries_dir is None):
        raise SystemExit("Either --dataset OR both --index-dir and --queries-dir must be provided.")

    if args.k != 20:
        print(f"[WARN] K is {args.k}, but you planned K=20. Proceeding anyway.")

    dataset = args.dataset or "custom"
    profile = detect_profile(args.profile)
    stage   = "orb"

    INDEXDIR = Path(args.index_dir) if args.index_dir else (DATA / "index" / dataset)
    QUERIES  = Path(args.queries_dir) if args.queries_dir else default_queries_dir(dataset)

    queries_tag = Path(QUERIES).name
    print(f"[INFO] dataset={dataset} profile={profile} stage={stage}")
    print(f"[INFO] queries_dir={QUERIES} (tag={queries_tag})")

    GT_DIR = DATA / "gt" / dataset
    GT_DIR.mkdir(parents=True, exist_ok=True)

    OUTROOT = RESULTS / dataset / profile / stage
    TABLES  = OUTROOT / "tables"
    TABLES.mkdir(parents=True, exist_ok=True)

    TAG = make_variant_tag(stage, args)
    TABLES_VAR = TABLES / "variants" / TAG
    TABLES_VAR.mkdir(parents=True, exist_ok=True)

    if args.device in ("cuda","cpu"):
        os.environ["DINO_DEVICE"] = args.device

    # image paths (for mapping name -> real path)
    paths_npy = INDEXDIR / "image_paths.npy"
    if not paths_npy.exists():
        raise FileNotFoundError(f"image_paths.npy not found: {paths_npy}")
    image_paths = [str(p) for p in np.load(paths_npy, allow_pickle=True)]
    name_to_path = {Path(p).name: Path(p) for p in image_paths}

    # stem->canonical name (.jpg pref)
    stem_to_name = {}
    for nm in name_to_path.keys():
        st = Path(nm).stem
        if st not in stem_to_name or nm.lower().endswith(".jpg"):
            stem_to_name[st] = nm

    # GT
    qtag = (Path(args.queries_dir).name if args.queries_dir else default_queries_dir(dataset).name)
    cand1 = GT_DIR / f"gt_queries_{qtag}.csv"
    cand2 = GT_DIR / "gt_queries.csv"
    gt_csv = cand1 if cand1.exists() else cand2
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT not found: {cand1} or {cand2} (run eval_search.py first)")
    with gt_csv.open(newline="", encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))
    if not gt_rows:
        raise RuntimeError("Empty GT file.")

    # sharding
    total = len(gt_rows)
    if args.split > 1:
        gt_rows = [r for i, r in enumerate(gt_rows) if (i % args.split) == args.shard]
        print(f"[INFO] shard filter: kept {len(gt_rows)}/{total} (split={args.split}, shard={args.shard})")

    # optional FAISS (only if baseline not reused)
    use_baseline = args.reuse_baseline is not None
    index = None
    ngpu = 0
    if not use_baseline:
        idx_path = INDEXDIR / "faiss_index.bin"
        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {idx_path}")
        cpu_index = faiss.read_index(str(idx_path))
        index = cpu_index
        try:
            ngpu = faiss.get_num_gpus()
            if ngpu > 0 and "HNSW" not in type(cpu_index).__name__.upper():
                index = faiss.index_cpu_to_all_gpus(cpu_index)
                print(f"[FAISS] Using {ngpu} GPU(s) for search.")
            else:
                print("[FAISS] Using CPU.")
        except Exception as e:
            print("[FAISS] GPU wrap skipped:", e)

    # load baseline map if requested
    baseline_map = None
    if use_baseline:
        bp = Path(args.reuse_baseline)
        if not bp.exists():
            raise FileNotFoundError(f"Baseline CSV not found: {bp}")
        baseline_map = {}
        with bp.open(encoding="utf-8") as f:
            rr = csv.DictReader(f)
            for row in rr:
                q = row.get("query","")
                preds = (row.get("pred@K","") or "").split(";")
                dists = [float(x) for x in (row.get("dist@K","") or "").split(";") if x!=""]
                baseline_map[q] = {"names": preds, "dists": dists}
        print(f"[INFO] baseline reuse: loaded {len(baseline_map)} rows from {bp}")

    # runtime accumulators
    K = args.k
    alpha = float(args.alpha)
    out_rows = []
    embed_ms_all, search_ms_all, geom_ms_all, total_ms_all = [], [], [], []
    rerank_applied_cnt, rerank_k_sum = 0, 0

    pbar_q = tqdm(gt_rows, desc=f"Rerank ORB FAST ({dataset})", unit="q", ncols=100)
    for row in pbar_q:
        qname = row["query"]; base_id = row["base_id"]; transform = row["transform"]
        qpath = QUERIES / qname

        t_total0 = time.perf_counter()

        # 1) embedding/search or baseline reuse
        if use_baseline and qname in baseline_map:
            # reuse baseline
            t_e0 = t_e1 = time.perf_counter()  # ~0ms
            t_s0 = time.perf_counter()
            cand_names = baseline_map[qname]["names"][:K]
            dists      = baseline_map[qname]["dists"][:K]
            # fabricate indices 0..K-1
            inds       = list(range(len(cand_names)))
            cand_paths = []
            for nm in cand_names:
                p = name_to_path.get(nm)
                if p is None:
                    # try stem-based fallback
                    st = Path(nm).stem
                    cnm = stem_to_name.get(st)
                    p = name_to_path.get(cnm, None)
                if p is None:
                    # skip missing
                    continue
                cand_paths.append(p)
            # align dists/inds with kept paths
            keep = [i for i,nm in enumerate(cand_names) if name_to_path.get(nm) or stem_to_name.get(Path(nm).stem) in name_to_path]
            cand_names = [cand_names[i] for i in keep]
            dists      = [dists[i]      for i in keep]
            inds       = list(range(len(cand_names)))
            t_s1 = time.perf_counter()
        else:
            # standard embedding + FAISS search
            try:
                t_e0 = time.perf_counter()
                qemb = get_embedding(str(qpath)).reshape(1,-1).astype("float32")
                t_e1 = time.perf_counter()
            except Exception:
                pbar_q.set_postfix_str(f"embed error: {qname[:18]}")
                continue
            try:
                t_s0 = time.perf_counter()
                dists_np, inds_np = index.search(qemb, K)
                t_s1 = time.perf_counter()
            except Exception:
                pbar_q.set_postfix_str(f"search error: {qname[:18]}")
                continue
            dists = dists_np[0].tolist()
            inds  = inds_np[0].tolist()
            # map to names/paths
            cand_names = [Path(image_paths[i]).name for i in inds]
            cand_paths = [Path(image_paths[i])      for i in inds]

        # 2) ORB+RANSAC on Top-K
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
            pbar_c.set_postfix_str(f"last={p.name[:18]}.. inl={inl} ir={ir:.3f}")
        pbar_c.close()
        t_g1 = time.perf_counter()

        # 3) gate + alpha combine + rerank
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
            alpha=float(args.alpha),
            k=K,
            gcfg=gcfg
        )
        rerank_applied_cnt += int(applied)
        rerank_k_sum += int(kk)

        # reorder
        id_to_pos = {i: j for j, i in enumerate(cand_ids.tolist())}
        order_idx = [id_to_pos[i] for i in new_ids.tolist()]
        cand_names2 = [cand_names[j] for j in order_idx]
        dists2      = [dists[j]      for j in order_idx]
        geom_raw2   = [geom_scores_raw[j] for j in order_idx]
        inlier2     = [inlier_ratios[j]    for j in order_idx]

        final_full = np.full(len(nn_dists), np.nan, dtype=float)
        if kk > 1:
            tmp = combine_scores(nn_dists[:kk], geom_rat[:kk], float(args.alpha))
            final_full[:kk] = tmp
        combined2 = [float(final_full[j]) for j in order_idx]

        # 4) metrics/logs
        gold = stem_to_name.get(base_id, f"{base_id}.jpg")
        rank_pos = (cand_names2.index(gold)+1) if gold in cand_names2 else -1
        top1_name = cand_names2[0] if cand_names2 else ""
        top1_dist = dists2[0]      if dists2      else float("nan")
        top1_ir   = inlier2[0]     if inlier2     else float("nan")

        t_total1 = time.perf_counter()
        t_embed_ms  = (t_e1 - t_e0) * 1000.0
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
            "top1_dist": f"{top1_dist:.6f}" if isinstance(top1_dist,(int,float)) else "",
            "top1_geom": f"{top1_ir:.6f}"   if isinstance(top1_ir,(int,float))   else "",
            "top1_geom_raw": f"{geom_raw2[0]:.6f}" if geom_raw2 else "",
            "pred@K": ";".join(cand_names2),
            "dist@K": ";".join(f"{d:.6f}" for d in dists2),
            "inlier_ratio@K": ";".join(f"{g:.6f}" for g in inlier2),
            "geom_score@K":   ";".join(f"{g:.6f}" for g in geom_raw2),
            "combined@K":     ";".join("" if np.isnan(c) else f"{c:.6f}" for c in combined2),
            "K": K,
            "alpha": float(args.alpha),
            "gate_margin": args.gate_margin,
            "gate_inlier": args.gate_inlier,
            "gate_scale": args.gate_scale,
            "gate_on": int(not args.gate_off),
            "rerank_applied": int(applied),
            "rerank_k": int(kk),
            "t_embed_ms": f"{t_embed_ms:.2f}",
            "t_search_ms": f"{t_search_ms:.2f}",
            "t_geom_ms":   f"{t_geom_ms:.2f}",
            "t_total_ms":  f"{t_total_ms:.2f}",
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

    # ---- save (variant ts + latest + rollup)
    ts = time.strftime("%Y%m%d-%H%M%S")
    results_csv_ts_var = TABLES_VAR / f"search_results_rerank_{stage}_{TAG}_{ts}.csv"
    perf_csv_ts_var    = TABLES_VAR / f"rerank_perf_summary_{stage}_{TAG}_{ts}.csv"

    with results_csv_ts_var.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    # perf summary
    N = len(out_rows)
    def _avg(xs): return (sum(xs)/max(len(xs),1)) if xs else 0.0
    sum_embed_sec  = sum(embed_ms_all)/1000.0
    sum_search_sec = sum(search_ms_all)/1000.0
    sum_geom_sec   = sum(geom_ms_all)/1000.0
    sum_total_sec  = sum(total_ms_all)/1000.0
    device_str = "GPU" if (not use_baseline and isinstance(faiss.get_num_gpus(), int) and faiss.get_num_gpus() > 0) else "CPU"

    perf_row = [{
        "dataset": dataset, "profile": profile, "stage": stage,
        "queries_tag": queries_tag, "queries_dir": str(QUERIES),
        "K": K, "alpha": float(args.alpha), "device": device_str,
        "index_ntotal": 0 if use_baseline else getattr(index,"ntotal",0),
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
        "used_baseline": int(use_baseline),
        "split": args.split,
        "shard": args.shard,
        "omp_threads": os.getenv("OMP_NUM_THREADS",""),
    }]

    with perf_csv_ts_var.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(perf_row[0].keys()))
        w.writeheader(); w.writerows(perf_row)

    print("Saved (variant):", results_csv_ts_var)
    print("Saved (variant):", perf_csv_ts_var)

    # latest alias
    results_csv_latest = TABLES / f"search_results_rerank_{stage}.csv"
    perf_csv_latest    = TABLES / f"rerank_perf_summary_{stage}.csv"
    try: results_csv_latest.write_bytes(results_csv_ts_var.read_bytes())
    except Exception: pass
    try: perf_csv_latest.write_bytes(perf_csv_ts_var.read_bytes())
    except Exception: pass

    # rollup
    rollup_csv = TABLES / f"rerank_{stage}_runs.csv"
    rollup_exists = rollup_csv.exists()
    with rollup_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(perf_row[0].keys()))
        if not rollup_exists: w.writeheader()
        w.writerow(perf_row[0])

if __name__ == "__main__":
    main()
