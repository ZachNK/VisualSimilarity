# Scripts/rerank_superglue_fast.py — SuperGlue+SuperPoint "FAST" rerank
# - Baseline의 Top-K 후보/거리 재사용 (임베딩/FAISS 재검색 없음)
# - N-way 샤딩 (--split/--shard) 병렬 실행
# - 스레드 수 지정 (--omp-threads)
# - 게이트/α결합/Top-K 재정렬 로직은 기존과 동일 (rerank_utils)

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, csv, time, sys, math
from pathlib import Path

import numpy as np, torch, cv2
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- project root import path (scripts.*, dinov2_utils not 사용) ---
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# === SuperGlue 레포 경로 자동 주입 ===
CANDIDATES = [
    Path(os.getenv("SUPERGLUE_DIR", "")),
    PROJ_ROOT / "third_party" / "SuperGluePretrainedNetwork",
    PROJ_ROOT / "SuperGluePretrainedNetwork",
]
picked = None
for p in CANDIDATES:
    if p and (p / "models" / "superpoint.py").exists() and (p / "models" / "superglue.py").exists():
        sys.path.insert(0, str(p))
        picked = p
        break
if not picked:
    raise ModuleNotFoundError(
        "SuperGluePretrainedNetwork의 'models' 폴더를 찾지 못했습니다.\n"
        "1) 레포를 clone해서 dino_test/third_party/SuperGluePretrainedNetwork 에 넣거나\n"
        "2) SUPERGLUE_DIR=/path/to/SuperGluePretrainedNetwork 를 설정하세요."
    )

from models.superpoint import SuperPoint
from models.superglue  import SuperGlue

# 공통 재정렬 유틸(게이트/결합/적용)
try:
    from scripts.rerank_utils import GateCfg, apply_rerank, combine_scores
except Exception:
    # 직접 실행 대비
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rerank_utils import GateCfg, apply_rerank, combine_scores

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
        f"{stage}-FAST-a{_fmtf(args.alpha)}-k{args.k}-"
        f"gm{_fmtf(args.gate_margin)}-gi{_fmtf(args.gate_inlier)}-"
        f"gs{_fmtf(getattr(args,'gate_scale',1.0))}-{gateflag}-"
        f"sp{_fmtf(args.sp_max_kp)}-w{args.sg_weights}"
    )

# ---------------- I/O utils ----------------
def load_gray(path: Path, long_side=960):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

def read_csv_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ---------------- SuperGlue scoring ----------------
@torch.inference_mode()
def sg_score(img0, img1, sp, sg, device, ransac_thresh=5.0):
    """
    반환:
      score         : 휴리스틱 점수(로그/모니터링용)
      inliers       : RANSAC 인라이어 수
      inlier_ratio  : 인라이어 비율(0..1) = inliers / min(#kpts0, #kpts1)
    """
    t0 = torch.from_numpy(img0)[None, None].float().to(device) / 255.0
    t1 = torch.from_numpy(img1)[None, None].float().to(device) / 255.0

    pred0 = sp({'image': t0})
    pred1 = sp({'image': t1})

    k0 = pred0['keypoints'][0]
    k1 = pred1['keypoints'][0]
    if k0.numel() == 0 or k1.numel() == 0:
        return 0.0, 0, 0.0

    data = {
        'keypoints0':   k0.unsqueeze(0),
        'keypoints1':   k1.unsqueeze(0),
        'descriptors0': pred0['descriptors'][0].unsqueeze(0),
        'descriptors1': pred1['descriptors'][0].unsqueeze(0),
        'scores0':      pred0['scores'][0].unsqueeze(0),
        'scores1':      pred1['scores'][0].unsqueeze(0),
    }
    out = sg(data)
    m0 = out['matches0'][0]     # (N0,) each idx into keypoints1 or -1
    valid = m0 > -1
    n_match = int(valid.sum().item())

    k0_np = k0.detach().cpu().numpy()
    k1_np = k1.detach().cpu().numpy()
    denom = max(1, min(k0_np.shape[0], k1_np.shape[0]))

    if n_match < 8:
        # inlier_ratio를 "매칭 수 / denom"으로 근사
        return 0.0, n_match, float(n_match) / float(denom)

    mk0 = k0[valid].detach().cpu().numpy()
    mk1 = k1[m0[valid].long()].detach().cpu().numpy()

    H, mask = cv2.findHomography(np.float32(mk0), np.float32(mk1), cv2.RANSAC, ransac_thresh)
    inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = float(inliers) / float(denom)

    score = (inliers / max(n_match, 1)) * 0.5 + (inliers / 200.0) * 0.5
    return float(score), inliers, float(inlier_ratio)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="FAST rerank with SuperGlue+SuperPoint using baseline Top-K.")
    # 데이터/경로
    ap.add_argument("--dataset", type=str, required=True, help="데이터셋 (visdrone, sodaa, aihub, union 등)")
    ap.add_argument("--profile", type=str, default=None, help="결과 하위 폴더(super 권장). 미지정 시 자동")
    ap.add_argument("--queries-dir", type=str, default=None, help="쿼리 폴더(기본: data/corpora/<dataset>/[images])")
    ap.add_argument("--index-dir", type=str, default=None, help="index 폴더 (image_paths.npy 찾기용)")
    ap.add_argument("--reuse-baseline", type=str, required=True,
                    help="baseline search_results.csv 경로 (Top-K 후보/거리 재사용)")
    # 재정렬 파라미터
    ap.add_argument("--k", type=int, default=20, help="Top-K (baseline K와 동일/이하 권장)")
    ap.add_argument("--alpha", type=float, default=0.2, help="임베딩:기하 결합 가중치")
    ap.add_argument("--gate.margin", dest="gate_margin", type=float, default=0.03)
    ap.add_argument("--gate.inlier", dest="gate_inlier", type=float, default=0.08)
    ap.add_argument("--gate.off", dest="gate_off", action="store_true")
    ap.add_argument("--gate.scale", dest="gate_scale", type=float, default=1.0)
    # 성능/장치
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--omp-threads", type=int, default=8, help="OpenMP/Torch 스레드 수")
    ap.add_argument("--split", type=int, default=1, help="쿼리 샤딩 개수(N)")
    ap.add_argument("--shard", type=int, default=0, help="현재 샤드 index (0..N-1)")
    # SuperPoint / SuperGlue
    ap.add_argument("--long-side", type=int, default=960, help="이미지 리사이즈 최대 변")
    ap.add_argument("--sg-weights", type=str, default="outdoor", choices=["outdoor","indoor"])
    ap.add_argument("--sp-max-kp", type=int, default=2048)
    ap.add_argument("--sp-kp-th", type=float, default=0.005)
    ap.add_argument("--sp-nms", type=int, default=4)
    ap.add_argument("--ransac", type=float, default=5.0)
    args = ap.parse_args()

    # 스레드 설정
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.omp_threads)))
    try:
        torch.set_num_threads(max(1, int(args.omp_threads)))
    except Exception:
        pass

    dataset = args.dataset
    profile = detect_profile(args.profile)  # 보통 'super'
    stage   = "super"                       # 고정
    OUTROOT = RESULTS / dataset / profile / stage
    TABLES  = OUTROOT / "tables"
    TABLES.mkdir(parents=True, exist_ok=True)

    # variants 폴더
    TAG = make_variant_tag(stage, args)
    TABLES_VAR = TABLES / "variants" / TAG
    TABLES_VAR.mkdir(parents=True, exist_ok=True)

    # 쿼리/인덱스 경로
    QUERIES  = Path(args.queries_dir) if args.queries_dir else default_queries_dir(dataset)
    INDEXDIR = Path(args.index_dir) if args.index_dir else (DATA / "index" / dataset)

    # image_paths.npy (filename -> fullpath) 매핑
    paths_npy = INDEXDIR / "image_paths.npy"
    if not paths_npy.exists():
        raise FileNotFoundError(f"image_paths.npy not found: {paths_npy}")
    image_paths = [str(p) for p in np.load(paths_npy, allow_pickle=True)]
    fpath_by_name = {Path(p).name: Path(p) for p in image_paths}

    # baseline CSV 읽기
    baseline_csv = Path(args.reuse_baseline)
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv}")
    base_rows = read_csv_rows(baseline_csv)

    # baseline rows -> query -> (cand_names, dists)
    base_by_query = {}
    stem_to_name = {}
    for r in base_rows:
        qname = r.get("query") or r.get("qname") or r.get("query_name")
        if not qname:
            continue
        pred = (r.get("pred@K") or "").split(";")
        dstr = (r.get("dist@K") or "").split(";")
        dists = []
        for x in dstr:
            try: dists.append(float(x))
            except: dists.append(np.nan)
        base_by_query[qname] = (pred, dists)

        # stem -> 대표 이름 (gold name 추정 시 사용)
        base_id = (r.get("base_id") or "").strip()
        if base_id:
            st = Path(base_id).stem
            # 후보군에서 같은 stem을 가진 첫 번째 파일명 선택
            for nm in pred:
                if Path(nm).stem == st:
                    stem_to_name.setdefault(st, nm)

    # GT 파일 (query, base_id, transform)
    GT_DIR = DATA / "gt" / dataset
    cand1 = GT_DIR / f"gt_queries_{QUERIES.name}.csv"
    cand2 = GT_DIR / "gt_queries.csv"
    gt_csv = cand1 if cand1.exists() else cand2
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT not found: {cand1} or {cand2}")
    gt_rows = read_csv_rows(gt_csv)
    if not gt_rows:
        raise RuntimeError("Empty GT file.")

    # 샤딩 적용
    split = max(1, int(args.split))
    shard = max(0, int(args.shard))
    if shard >= split:
        raise ValueError("--shard must be < --split.")
    gt_rows = [r for i, r in enumerate(gt_rows) if (i % split) == shard]
    if not gt_rows:
        print(f"[WARN] No GT rows for shard {shard}/{split}.")
        return

    # SuperPoint + SuperGlue 준비
    device = torch.device("cuda" if (args.device=="cuda" or (args.device=="auto" and torch.cuda.is_available())) else "cpu")
    sp_conf = {
        'descriptor_dim': 256,
        'nms_radius': int(args.sp_nms),
        'keypoint_threshold': float(args.sp_kp_th),
        'max_keypoints': int(args.sp_max_kp),
    }
    sg_conf = {
        'weights': args.sg_weights,
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
    sp = SuperPoint(sp_conf).eval().to(device)
    sg = SuperGlue(sg_conf).eval().to(device)

    # 파라미터
    K = int(args.k)
    alpha = float(args.alpha)
    gcfg = GateCfg(
        use_gate=(not args.gate_off),
        tau_margin=args.gate_margin * args.gate_scale,
        tau_inlier=args.gate_inlier * args.gate_scale
    )

    print(f"[INFO] dataset={dataset}, profile={profile}, stage={stage}, K={K}, alpha={alpha}, device={device.type}, shard={shard}/{split}")

    out_rows = []
    # 시간 통계(임베딩/검색은 baseline 재사용 → 0으로 기록)
    embed_ms_all, search_ms_all, geom_ms_all, total_ms_all = [], [], [], []
    rerank_applied_cnt, rerank_k_sum = 0, 0

    pbar_q = tqdm(gt_rows, desc=f"FAST Rerank SuperGlue+SP ({dataset})", unit="q", ncols=100)
    for row in pbar_q:
        qname = row["query"]; base_id = row["base_id"]; transform = row.get("transform","")
        if qname not in base_by_query:
            pbar_q.set_postfix_str(f"no baseline: {qname[:18]}")
            continue
        cand_names, dists = base_by_query[qname]
        if not cand_names:
            continue

        # baseline K와 fast K 정합
        kk = min(K, len(cand_names))
        cand_names = cand_names[:kk]
        dists      = dists[:kk]

        # 이미지 로딩
        qpath = QUERIES / qname
        try:
            qimg = load_gray(qpath, long_side=args.long_side)
        except Exception:
            pbar_q.set_postfix_str(f"load qimg error: {qname[:18]}")
            continue

        # SuperGlue 기하 매칭
        geom_scores_raw, inlier_ratios = [], []
        t_total0 = time.perf_counter()
        t_g0 = time.perf_counter()
        for nm in cand_names:
            p = fpath_by_name.get(nm)
            if p is None or (not p.exists()):
                # filename이 index에 없을 수도 있으므로 쿼리 폴더 근처도 시도
                cand_try = QUERIES.parent / nm
                p = cand_try if cand_try.exists() else None
            if p is None:
                s, inl, ir = 0.0, 0, 0.0
            else:
                try:
                    dimg = load_gray(p, long_side=args.long_side)
                    s, inl, ir = sg_score(qimg, dimg, sp, sg, device, ransac_thresh=args.ransac)
                except Exception:
                    s, inl, ir = 0.0, 0, 0.0
            geom_scores_raw.append(s)
            inlier_ratios.append(ir)
        t_g1 = time.perf_counter()

        # 게이트+α결합+재정렬
        nn_dists = np.asarray(dists, dtype=float)
        cand_ids = np.arange(len(cand_names), dtype=int)  # 로컬 인덱스
        geom_rat = np.asarray(inlier_ratios, dtype=float)

        new_ids, applied, used_k = apply_rerank(
            nn_dists=nn_dists,
            candidate_ids=cand_ids,
            geom_scores=geom_rat,
            alpha=alpha,
            k=kk,
            gcfg=gcfg
        )
        rerank_applied_cnt += int(applied)
        rerank_k_sum += int(used_k)

        # 새 순서 반영
        order_idx = new_ids.tolist()
        cand_names2 = [cand_names[j] for j in order_idx]
        dists2      = [dists[j]      for j in order_idx]
        geom_raw2   = [geom_scores_raw[j] for j in order_idx]
        inlier2     = [inlier_ratios[j]    for j in order_idx]

        # 결합 점수 기록(top-used_k만)
        final_full = np.full(len(nn_dists), np.nan, dtype=float)
        if used_k > 1:
            tmp = combine_scores(nn_dists[:used_k], geom_rat[:used_k], alpha)
            final_full[:used_k] = tmp
        combined2 = [float(final_full[j]) for j in order_idx]

        # gold 이름 추정
        gold = stem_to_name.get(Path(base_id).stem, f"{Path(base_id).stem}.jpg")
        rank_pos = (cand_names2.index(gold)+1) if gold in cand_names2 else -1
        top1_name = cand_names2[0]
        top1_dist = dists2[0]
        top1_inlier_ratio = inlier2[0]

        # 시간 기록 (embed/search=0, geom/total만 측정)
        t_total1 = time.perf_counter()
        t_embed_ms  = 0.0
        t_search_ms = 0.0
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
            "top1_geom": f"{top1_inlier_ratio:.6f}",
            "top1_geom_raw": f"{geom_raw2[0]:.6f}",
            "pred@K": ";".join(cand_names2),
            "dist@K": ";".join(f"{d:.6f}" for d in dists2),
            "inlier_ratio@K": ";".join(f"{g:.6f}" for g in inlier2),
            "geom_score@K": ";".join(f"{g:.6f}" for g in geom_raw2),
            "combined@K": ";".join("" if np.isnan(c) else f"{c:.6f}" for c in combined2),
            "K": kk,
            "alpha": alpha,
            "gate_margin": args.gate_margin,
            "gate_inlier": args.gate_inlier,
            "gate_scale": args.gate_scale,
            "gate_on": int(not args.gate_off),
            "rerank_applied": int(applied),
            "rerank_k": int(used_k),
            "t_embed_ms": f"{t_embed_ms:.2f}",
            "t_search_ms": f"{t_search_ms:.2f}",
            "t_geom_ms": f"{t_geom_ms:.2f}",
            "t_total_ms": f"{t_total_ms:.2f}",
        })

        pbar_q.set_postfix_str(
            f"rank@1={'OK' if rank_pos==1 else rank_pos} "
            f"t(ms):E{t_embed_ms:.0f}/S{t_search_ms:.0f}/G{t_geom_ms:.0f} "
            f"gate={'ON' if not args.gate_off else 'OFF'} a={alpha:.2f} K={kk} "
            f"{'rerank' if applied else 'keep'}"
        )
    pbar_q.close()

    if not out_rows:
        raise RuntimeError("No rows generated (check baseline/GT/shard filters).")

    # 저장
    ts = time.strftime("%Y%m%d-%H%M%S")
    results_csv_ts_var = TABLES_VAR / f"search_results_rerank_{stage}_{TAG}_{ts}.csv"
    perf_csv_ts_var    = TABLES_VAR / f"rerank_perf_summary_{stage}_{TAG}_{ts}.csv"

    with open(results_csv_ts_var, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    # perf 요약
    N = len(out_rows)
    def _avg(xs): return (sum(xs)/max(len(xs),1)) if xs else 0.0
    sum_embed_sec  = sum(embed_ms_all)/1000.0
    sum_search_sec = sum(search_ms_all)/1000.0
    sum_geom_sec   = sum(geom_ms_all)/1000.0
    sum_total_sec  = sum(total_ms_all)/1000.0

    perf_row = [{
        "dataset": dataset, "profile": profile, "stage": stage,
        "queries_tag": QUERIES.name, "queries_dir": str(QUERIES),
        "K": int(args.k), "alpha": alpha, "device": ("GPU" if device.type=="cuda" else "CPU"),
        "index_ntotal": 0,  # FAST 모드: 검색 생략
        "gate_margin": args.gate_margin, "gate_inlier": args.gate_inlier, "gate_scale": args.gate_scale,
        "gate_on": int(not args.gate_off),
        "variant_tag": TAG,
        "rerank_applied_ratio": f"{(rerank_applied_cnt/max(N,1)):.4f}",
        "rerank_k_mean": f"{(rerank_k_sum/max(rerank_applied_cnt,1)):.2f}" if rerank_applied_cnt>0 else "0.00",
        "avg_embed_ms": f"{_avg(embed_ms_all):.2f}",   # 0.00
        "avg_search_ms": f"{_avg(search_ms_all):.2f}", # 0.00
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

    with open(perf_csv_ts_var, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(perf_row[0].keys()))
        w.writeheader(); w.writerows(perf_row)

    # latest 별칭
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

    print("Saved (variant):", results_csv_ts_var)
    print("Saved (variant):", perf_csv_ts_var)
    print("Saved (latest) :", results_csv_latest)
    print("Saved (latest) :", perf_csv_latest)

if __name__ == "__main__":
    main()
