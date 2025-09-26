# scripts/rerank_lightglue_fast.py

# LightGlue + SuperPoint fast reranker
# - Reuse baseline Top-K (no FAISS / no embedding)
# - N-way sharding (--split/--shard) for parallel runs
# - Thread control (--omp-threads), tqdm, timestamped variants, latest, rollup

# 실행 전 준비
# 1. Baseline CSV 확인: baseline은 eval_search.py 실행 시 생성된 Top-K 검색 결과임.
# -results/<dataset>/<profile>/baseline/tables/search_results.csv

# 예: results/visdrone/light/baseline/tables/search_results.csv
# 예: results/sodaa/light/baseline/tables/search_results.csv

# 2. GT 파일 확인: 없으면 eval_search.py --dataset <dataset> --profile light로 생성해야함.
# data/gt/<dataset>/gt_queries.csv

# 3. 기본 실행 방법
# 예: VisDrone 데이터셋, LightGlue+SuperPoint, K=20, GPU(cuda) 사용

# > (light2025) BL="/mnt/d/KNK/_KSNU/_Projects/dino_test/results/visdrone/light/baseline/tables/search_results.csv"

# > (light2025) python -m scripts.rerank_lightglue_fast \
#   --dataset visdrone \
#   --profile light \
#   --reuse-baseline "$BL" \
#   --k 20 --alpha 0.7 \
#   --device cuda \
#   --omp-threads 8 \
#   --long-side 960 --ransac 5.0 --sp-max-kp 2048

# 4. 병렬 샤딩 실행 (N-way split)
# 예: --split 4: 전체 쿼리를 4등분
# --shard 0..3: 각각 1/4 쿼리만 처리

# > (light2025) BL="/mnt/d/KNK/_KSNU/_Projects/dino_test/results/visdrone/light/baseline/tables/search_results.csv"

# > (light2025) python
# > for S in 0 1 2 3; do
#...   python -m scripts.rerank_lightglue_fast \
#...   --dataset visdrone \
#...   --profile light \
#...   --reuse-baseline "$BL" \
#...   --k 20 --alpha 0.7 \
#...   --split 4 --shard $S \
#...   --omp-threads 8 \
#...   --device cuda \
#...   --long-side 960 --ransac 5.0 --sp-max-kp 2048 &
# > done

# 5. 주요 파라미터 설명

# --dataset: 데이터셋 이름 (visdrone, sodaa, aihub, union 등)
# --profile: 결과 저장 경로 하위 폴더 (light)
# --reuse-baseline: baseline 검색 결과 CSV 경로 (필수)
# --k: Top-K 후보 수 (논문 실험은 20 고정 권장)
# --alpha: 임베딩 점수 vs 기하 점수 가중치
# --gate.margin, --gate.inlier, --gate.scale: 게이트 설정
# --split, --shard: 병렬 샤딩
# --omp-threads: CPU/OpenMP/torch 스레드 수
# --device: cuda (GPU) / cpu
# --long-side: 이미지 resize 최대 변
# --ransac: RANSAC reprojection threshold
# --sp-max-kp: SuperPoint 최대 키포인트 수 (default 2048)

# 6. 출력 파일

# results/<dataset>/<profile>/light/tables/

#     search_results_rerank_light.csv (latest 별칭)
#     rerank_perf_summary_light.csv (성능 요약)
#     variants/<tag>/... (파라미터별 스냅샷)
#     rerank_light_runs.csv (실행 로그 누적)


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, csv, time, sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# (옵션) DINO 임베딩을 쓰지 않지만, 호환성 위해 import 가드
try:
    from scripts.dinov2_utils import get_embedding   # noqa: F401
except Exception:
    try:
        from dinov2_utils import get_embedding       # noqa: F401
    except Exception:
        get_embedding = None

# 게이트/재정렬 유틸
try:
    from scripts.rerank_utils import GateCfg, apply_rerank, combine_scores
except Exception:
    from rerank_utils import GateCfg, apply_rerank, combine_scores

import numpy as np, cv2, torch
from PIL import ImageFile
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint

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

def default_queries_dir(dataset: str) -> Path:
    c1 = DATA / "corpora" / dataset / "images"
    return c1 if c1.exists() else (DATA / "corpora" / dataset)

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

def load_gray(path: Path, long_side=960):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

@torch.inference_mode()
def lg_score(img0, img1, extractor, matcher, ransac_thresh=5.0):
    """
    반환:
      score        : 휴리스틱 점수(로그)
      inliers      : RANSAC 인라이어 수
      inlier_ratio : 0..1 (= inliers / min(#kpts0, #kpts1)) — 게이트/결합용
    """
    device = next(matcher.parameters()).device
    t0 = torch.from_numpy(img0)[None, None].float().to(device) / 255.0
    t1 = torch.from_numpy(img1)[None, None].float().to(device) / 255.0

    f0 = extractor({'image': t0})
    f1 = extractor({'image': t1})
    out = matcher({'image0': f0, 'image1': f1})

    m = out['matches'][0].detach().cpu()   # [M,2]
    k0 = f0['keypoints'][0].detach().cpu().numpy()
    k1 = f1['keypoints'][0].detach().cpu().numpy()

    if m.numel() == 0 or k0.shape[0] == 0 or k1.shape[0] == 0:
        return 0.0, 0, 0.0

    mk0 = k0[m[:, 0].long().numpy()]
    mk1 = k1[m[:, 1].long().numpy()]
    if mk0.shape[0] < 8:
        denom = max(1, min(k0.shape[0], k1.shape[0]))
        return 0.0, int(mk0.shape[0]), float(mk0.shape[0]) / float(denom)

    H, mask = cv2.findHomography(np.float32(mk0), np.float32(mk1), cv2.RANSAC, ransac_thresh)
    inliers = int(mask.sum()) if mask is not None else 0

    denom = max(1, min(k0.shape[0], k1.shape[0]))
    inlier_ratio = float(inliers) / float(denom)

    score = (inliers / max(len(mk0), 1)) * 0.5 + (inliers / 200.0) * 0.5
    return float(score), inliers, float(inlier_ratio)

def read_csv_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def build_name_maps(image_paths):
    fname_by_idx = {i: Path(p).name for i, p in enumerate(image_paths)}
    fpath_by_idx = {i: Path(p)      for i, p in enumerate(image_paths)}
    stem_to_name = {}
    for name in fname_by_idx.values():
        st = Path(name).stem
        if st not in stem_to_name or name.lower().endswith(".jpg"):
            stem_to_name[st] = name
    return fname_by_idx, fpath_by_idx, stem_to_name

def parse_semicolon(s):
    return [x for x in s.split(";") if x]

# ---------------- baseline reuse ----------------
def load_baseline_lookup(baseline_csv: Path):
    """query → (pred_names, dists) 룩업 생성.
       baseline은 search_results.csv (eval_search.py) 또는 search_results_rerank_*.csv 형태를 허용.
       pred 후보는 'pred@K', 거리 'dist@K' 컬럼을 우선, 없으면 top1만이라도 사용."""
    rows = read_csv_rows(baseline_csv)
    lut = {}
    for r in rows:
        q = r.get("query") or r.get("qname") or r.get("query_name")
        if not q: continue
        if "pred@K" in r:
            preds = parse_semicolon(r["pred@K"])
        else:
            preds = [r.get("top1_name","")] if r.get("top1_name") else []
        if "dist@K" in r:
            dists = [float(x) if x else np.nan for x in parse_semicolon(r["dist@K"])]
        else:
            dists = [float(r.get("top1_dist","nan"))] if r.get("top1_dist") else []
        lut[q] = (preds, dists)
    return lut

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="FAST rerank: LightGlue+SuperPoint using baseline Top-K.")
    # 데이터/경로
    ap.add_argument("--dataset", type=str, help="데이터셋(예: visdrone, sodaa, aihub)")
    ap.add_argument("--index-dir", type=str, default=None, help="인덱스 폴더(이미지 경로 로딩용)")
    ap.add_argument("--queries-dir", type=str, default=None, help="쿼리 폴더 (기본: data/corpora/<dataset>[/images])")
    ap.add_argument("--reuse-baseline", type=str, default=None, help="baseline 또는 이전 rerank CSV 경로(search_results*.csv)")
    # 재정렬 파라미터
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--k",     type=int,   default=20)
    ap.add_argument("--gate.margin", dest="gate_margin", type=float, default=0.03)
    ap.add_argument("--gate.inlier", dest="gate_inlier", type=float, default=0.08)
    ap.add_argument("--gate.off",     dest="gate_off", action="store_true")
    ap.add_argument("--gate.scale",   dest="gate_scale", type=float, default=1.0)
    # 실행 제어
    ap.add_argument("--profile", type=str, default=None, help="결과 하위 폴더(orb/light/super). 미지정 시 자동")
    ap.add_argument("--device",  type=str, default="auto", choices=["auto","cuda","cpu"], help="LightGlue 디바이스")
    ap.add_argument("--omp-threads", type=int, default=8, help="OpenMP/BLAS 스레드수")
    ap.add_argument("--split", type=int, default=1, help="N-way sharding: 총 분할 개수")
    ap.add_argument("--shard", type=int, default=0, help="현재 샤드 인덱스 [0..split-1]")
    # LG/SP & 이미지 파라미터
    ap.add_argument("--long-side", type=int, default=960)
    ap.add_argument("--ransac", type=float, default=5.0)
    ap.add_argument("--sp-max-kp", type=int, default=2048)
    args = ap.parse_args()

    # 스레드 설정
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.omp_threads)))
    try: cv2.setNumThreads(int(args.omp_threads))
    except Exception: pass
    try: torch.set_num_threads(int(args.omp_threads))
    except Exception: pass

    if args.split < 1 or not (0 <= args.shard < args.split):
        raise SystemExit(f"--split must be >=1 and 0 <= --shard < --split (got split={args.split}, shard={args.shard})")

    if not args.dataset and (args.index_dir is None or args.queries_dir is None):
        raise SystemExit("Either --dataset OR both --index-dir and --queries-dir must be provided.")

    dataset = args.dataset or "custom"
    profile = detect_profile(args.profile)
    stage   = "light"

    INDEXDIR = Path(args.index_dir) if args.index_dir else (DATA / "index" / dataset)
    QUERIES  = Path(args.queries_dir) if args.queries_dir else default_queries_dir(dataset)
    queries_tag = Path(QUERIES).name

    # 이미지 경로 로딩 (인덱스 생성 시 저장된 image_paths.npy 가정)
    paths_npy = INDEXDIR / "image_paths.npy"
    if not paths_npy.exists():
        raise FileNotFoundError(f"image_paths.npy not found: {paths_npy}")
    image_paths = [str(p) for p in np.load(paths_npy, allow_pickle=True)]
    fname_by_idx, fpath_by_idx, stem_to_name = build_name_maps(image_paths)

    # GT 로딩
    GT_DIR = DATA / "gt" / dataset
    qtag = (Path(args.queries_dir).name if args.queries_dir else default_queries_dir(dataset).name)
    gt_csv = (GT_DIR / f"gt_queries_{qtag}.csv") if (GT_DIR / f"gt_queries_{qtag}.csv").exists() else (GT_DIR / "gt_queries.csv")
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT not found: {gt_csv}")
    gt_rows_all = read_csv_rows(gt_csv)
    if not gt_rows_all:
        raise RuntimeError("Empty GT file.")

    # 샤딩 적용
    gt_rows = [r for i,r in enumerate(gt_rows_all) if (i % args.split) == args.shard]

    # baseline 재사용
    if not args.reuse_baseline:
        raise SystemExit("--reuse-baseline CSV must be provided for the *fast* script.")
    baseline_csv = Path(args.reuse_baseline)
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv}")
    base_lut = load_baseline_lookup(baseline_csv)

    # 출력 경로
    OUTROOT  = RESULTS / dataset / profile / stage
    TABLES   = OUTROOT / "tables"
    TABLES.mkdir(parents=True, exist_ok=True)
    TAG = make_variant_tag(stage, args)
    TABLES_VAR = TABLES / "variants" / TAG
    TABLES_VAR.mkdir(parents=True, exist_ok=True)

    # 디바이스
    lg_device = torch.device("cuda" if (args.device=="cuda" or (args.device=="auto" and torch.cuda.is_available())) else "cpu")

    # LightGlue / SuperPoint
    extractor = SuperPoint(max_num_keypoints=int(args.sp_max_kp)).eval().to(lg_device)
    matcher   = LightGlue(features='superpoint').eval().to(lg_device)

    K = int(args.k)
    alpha = float(args.alpha)

    print(f"[INFO] dataset={dataset}, profile={profile}, stage={stage}, device={lg_device.type}, "
          f"queries_dir={QUERIES} shard={args.shard+1}/{args.split}, K={K}, alpha={alpha}")
    print(f"[INFO] baseline reuse: {baseline_csv.name}")

    out_rows = []
    # (임베딩/검색은 baseline을 재사용하므로 0 처리)
    embed_ms_all, search_ms_all, geom_ms_all, total_ms_all = [], [], [], []
    rerank_applied_cnt, rerank_k_sum = 0, 0

    pbar_q = tqdm(gt_rows, desc=f"Rerank LightGlue+SP (shard {args.shard+1}/{args.split})", unit="q", ncols=100)
    for row in pbar_q:
        qname = row["query"]; base_id = row["base_id"]; transform = row.get("transform","")
        qpath = QUERIES / qname

        # baseline Top-K 로드
        preds, dists = base_lut.get(qname, ([], []))
        if not preds:
            # baseline에 없다면 skip
            pbar_q.set_postfix_str(f"no baseline: {qname[:18]}")
            continue

        # 제한 K
        cand_names = preds[:K]
        dists = (dists[:K] + [np.nan]*(K-len(dists)))[:K] if dists else [np.nan]*len(cand_names)
        cand_paths = []
        for nm in cand_names:
            # 인덱스의 원본 경로를 찾아준다 (파일명이 일치한다고 가정)
            # 경로가 절대경로일 수 있으므로 우선 그대로 사용하고, 아니면 인덱스에서 찾음
            p = Path(nm)
            if p.exists():
                cand_paths.append(p)
            else:
                # image_paths에서 파일명 매칭
                # 느릴 수 있지만 K가 작으니 허용
                # 더 빠르게 하려면 name->path dict를 사전 구축
                matches = [Path(ip) for ip in image_paths if Path(ip).name == nm]
                cand_paths.append(matches[0] if matches else Path(nm))

        # 기하 매칭
        try:
            qimg = load_gray(qpath, long_side=args.long_side)
        except Exception:
            pbar_q.set_postfix_str(f"load qimg error: {qname[:18]}")
            continue

        geom_scores_raw, inlier_ratios = [], []
        t_total0 = time.perf_counter()
        t_g0 = time.perf_counter()
        pbar_c = tqdm(cand_paths, desc=f"  candidates({qname})", unit="img", leave=False, ncols=100)
        for p in pbar_c:
            try:
                dimg = load_gray(p, long_side=args.long_side)
                s, inl, ir = lg_score(qimg, dimg, extractor, matcher, ransac_thresh=args.ransac)
            except Exception:
                s, inl, ir = 0.0, 0, 0.0
            geom_scores_raw.append(s)
            inlier_ratios.append(ir)
            pbar_c.set_postfix_str(f"last={p.name[:18]}.. ir={ir:.3f}")
        pbar_c.close()
        t_g1 = time.perf_counter()

        # 게이트 + α 결합 + 재정렬
        nn_dists = np.asarray(dists, dtype=float)
        cand_ids = np.arange(len(cand_names), dtype=int)  # 로컬 인덱스
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

        # 새 순서 적용
        order_idx = new_ids.tolist()
        cand_names2 = [cand_names[j] for j in order_idx]
        dists2      = [dists[j]      for j in order_idx]
        geom_raw2   = [geom_scores_raw[j] for j in order_idx]
        inlier2     = [inlier_ratios[j]    for j in order_idx]

        final_full = np.full(len(nn_dists), np.nan, dtype=float)
        if kk > 1:
            tmp = combine_scores(nn_dists[:kk], geom_rat[:kk], alpha)
            final_full[:kk] = tmp
        combined2 = [float(final_full[j]) for j in order_idx]

        # 랭크/타임 기록 (embed/search는 baseline 재사용 → 0)
        gold = stem_to_name.get(base_id, f"{base_id}.jpg")
        rank_pos = (cand_names2.index(gold)+1) if gold in cand_names2 else -1
        top1_name = cand_names2[0]
        top1_dist = dists2[0]
        top1_inlier_ratio = inlier2[0]

        t_total1 = time.perf_counter()
        t_embed_ms = 0.0
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
            "top1_dist": f"{(top1_dist if top1_dist==top1_dist else np.nan):.6f}" if not (top1_dist is None) else "",
            "top1_geom": f"{top1_inlier_ratio:.6f}",
            "top1_geom_raw": f"{geom_raw2[0]:.6f}",
            "pred@K": ";".join(cand_names2),
            "dist@K": ";".join("" if (d!=d) else f"{d:.6f}" for d in dists2),
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
        raise RuntimeError("No rows generated (this shard may be empty).")

    # 저장
    ts = time.strftime("%Y%m%d-%H%M%S")
    TAG = make_variant_tag(stage, args) + f"-s{args.split}-h{args.shard}"
    results_csv_ts_var = TABLES_VAR / f"search_results_rerank_{stage}_{TAG}_{ts}.csv"
    perf_csv_ts_var    = TABLES_VAR / f"rerank_perf_summary_{stage}_{TAG}_{ts}.csv"

    N = len(out_rows)
    def _avg(xs): return (sum(xs)/max(len(xs),1)) if xs else 0.0
    sum_embed_sec  = sum(embed_ms_all)/1000.0
    sum_search_sec = sum(search_ms_all)/1000.0
    sum_geom_sec   = sum(geom_ms_all)/1000.0
    sum_total_sec  = sum(total_ms_all)/1000.0

    perf_row = [{
        "dataset": dataset, "profile": profile, "stage": stage,
        "queries_tag": queries_tag, "queries_dir": str(QUERIES),
        "K": K, "alpha": alpha, "device": ("GPU" if lg_device.type=="cuda" else "CPU"),
        "index_ntotal": len(image_paths),
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
        "split": args.split, "shard": args.shard, "omp_threads": args.omp_threads,
        "baseline_csv": str(baseline_csv),
    }]

    with open(results_csv_ts_var, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    with open(perf_csv_ts_var, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(perf_row[0].keys()))
        w.writeheader(); w.writerows(perf_row)

    print("Saved (variant):", results_csv_ts_var)
    print("Saved (variant):", perf_csv_ts_var)

    # latest 별칭
    TABLES.mkdir(parents=True, exist_ok=True)
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
