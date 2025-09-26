# Scripts/analyze_rerank.py — ΔRecall@1, CMC, Coverage–Recall
# 저장 규칙 통일: results/<dataset>/<profile>/analysis/tables/
# - ts 스냅샷 + latest 별칭 + 롤업(delta_recall1_runs.csv)

### ================================ 사용방법 ================================

# # 1) 특정 CSV 한 개 분석
# python analyze_rerank.py --dataset visdrone --after-csv D:/KNK/_KSNU/_Projects/dino_test/results/visdrone/orb/light/tables/search_results_rerank_light_20250910-1530.csv

# # 2) 특정 스테이지(orb/light/super) 폴더의 모든 rerank CSV 분석
# python analyze_rerank.py --dataset visdrone --stage orb

# # 3) 해당 프로파일(자동 감지) 아래 모든 스테이지의 rerank CSV 일괄 분석
# python analyze_rerank.py --dataset visdrone --scan-all

# # 4) 프로파일 강제 지정
# python analyze_rerank.py --dataset visdrone --profile light --scan-all


### ================================ 사용방법 ================================

import os, argparse, csv, time
from pathlib import Path
from collections import defaultdict
import numpy as np

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

BASE     = resolve_base()
RESULTS  = BASE / "results"

# ---------------- utils ----------------
def load_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def recall_at_k(rows, K: int) -> float:
    ok = sum(1 for r in rows if int(r["rank_pos"]) != -1 and int(r["rank_pos"]) <= K)
    return ok / max(len(rows), 1)

def cmc_curve(rows, Kmax: int = 20):
    ranks = [int(r["rank_pos"]) for r in rows if int(r["rank_pos"]) != -1]
    return [sum(1 for rk in ranks if rk <= k) / max(len(rows), 1) for k in range(1, Kmax + 1)]

def pick_top1_col(rows):
    if not rows: raise RuntimeError("Empty rows.")
    keys = set(rows[0].keys())
    # 선호 순서: distance < combined/geom/score/value
    cand = ["top1_dist", "top1_combined", "top1_geom", "top1_score", "top1_value"]
    for c in cand:
        if c in keys: return c
    # fallback: 가장 'top1_' 접두 가진 수치형 컬럼 탐색
    for k in keys:
        if k.lower().startswith("top1_"):
            try:
                float(rows[0][k])
                return k
            except Exception:
                continue
    raise KeyError(f"Cannot find top1 distance/score column in: {sorted(keys)}")

def tofloat(x):
    try: return float(x)
    except Exception: return float("nan")

def is_ip_col(col_name: str) -> bool:
    # Higher-is-better for score/sim/geom/ratio/prob/combined
    n = col_name.lower()
    return any(t in n for t in ["score","sim","geom","ratio","prob","combined"])

def coverage_recall(rows, top1_col: str, n_grid: int = 200):
    vals = np.array([tofloat(r.get(top1_col, "nan")) for r in rows], np.float32)
    gold = np.array([r.get("top1_name","") == r.get("gold","") for r in rows], bool)
    ok = ~np.isnan(vals)
    vals = vals[ok]; gold = gold[ok]
    if vals.size == 0: return []
    taus = np.quantile(vals, np.linspace(0, 1, n_grid))
    use_ip = is_ip_col(top1_col)
    out = []
    for t in taus:
        acc = (vals >= t) if use_ip else (vals <= t)
        cov = acc.mean()
        rec = (gold & acc).mean()
        out.append((float(t), float(cov), float(rec)))
    return out

def group_by_tf(rows):
    d = defaultdict(list)
    for r in rows: d[r.get("transform","")].append(r)
    return d

def guess_stage_from_path(p: Path) -> str:
    # 우선 파일명에서, 없으면 상위 폴더명에서 추정
    name = p.name.lower()
    for k in ("orb","light","super"):
        if f"rerank_{k}" in name or f"_{k}_" in name or name.endswith(f"_{k}.csv"):
            return k
    parts = [s.lower() for s in p.parts]
    for k in ("orb","light","super"):
        if k in parts: return k
    return "unknown"

def extract_variant_tag(path: Path) -> str:
    parts = list(path.parts)
    if "variants" in parts:
        i = parts.index("variants")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""

def extract_params(rows):
    """rerank CSV 1행에서 파라미터 메타 추출 (없으면 빈 문자열)"""
    if not rows: return {}
    r0 = rows[0]
    keys = ["alpha","gate_margin","gate_inlier","gate_scale","gate_on","K"]
    out = {}
    for k in keys:
        if k in r0:
            out[k] = r0.get(k, "")
    return out

# ---------------- core ----------------
def find_baseline_csv(dataset: str, profile: str) -> Path:
    # 신규 규칙
    p1 = RESULTS / dataset / profile / "baseline" / "tables" / "search_results.csv"
    if p1.exists(): return p1
    # 구 규칙 폴백
    p2 = RESULTS / "tables" / "search_results.csv"
    if p2.exists(): return p2
    raise FileNotFoundError(
        f"Baseline CSV not found at:\n  {p1}\n  (fallback tried) {p2}\n"
        "Run eval_search.py with the unified layout first."
    )

def analysis_out_dirs(dataset: str, profile: str):
    root   = RESULTS / dataset / profile / "analysis"
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    return root, tables

def run_for_file(dataset: str, profile: str, rerank_csv: Path, stage_hint: str | None = None):
    BEFORE = find_baseline_csv(dataset, profile)
    if not rerank_csv.exists():
        raise FileNotFoundError(f"AFTER CSV not found: {rerank_csv}")

    stage = (stage_hint or guess_stage_from_path(rerank_csv)).lower()
    name_tag = stage if stage != "unknown" else rerank_csv.stem.replace("search_results_rerank_","")

    before = load_rows(BEFORE)
    after  = load_rows(rerank_csv)
    top1_before = pick_top1_col(before)
    top1_after  = pick_top1_col(after)

    gb_b = group_by_tf(before)
    gb_a = group_by_tf(after)

    # 메타(variants/파라미터)
    variant_tag = extract_variant_tag(rerank_csv)
    params = extract_params(after)

    # 출력 경로
    _, TABLES = analysis_out_dirs(dataset, profile)
    ts = time.strftime("%Y%m%d-%H%M%S")

    # ---- 1) ΔRecall@1
    summary = []
    for tf in sorted(set(gb_b) | set(gb_a)):
        rb = gb_b.get(tf, []); ra = gb_a.get(tf, [])
        rec1_b = recall_at_k(rb, 1)
        rec1_a = recall_at_k(ra, 1)
        row = {
            "dataset": dataset,
            "profile": profile,
            "stage":   stage,
            "transform": tf,
            "Recall@1_before": f"{rec1_b:.4f}",
            "Recall@1_after":  f"{rec1_a:.4f}",
            "Delta":           f"{(rec1_a-rec1_b):+.4f}",
            "N": len(ra) or len(rb),
            "run_ts": ts,
            "variant_tag": variant_tag,
        }
        # 파라미터 메타 병합
        for k, v in params.items():
            row[k] = v
        summary.append(row)

    delta_ts = TABLES / f"delta_recall1_{name_tag}_{ts}.csv"
    delta_latest = TABLES / f"delta_recall1_{name_tag}.csv"
    with open(delta_ts, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    try: delta_latest.write_bytes(delta_ts.read_bytes())
    except Exception: pass

    # 롤업(append)
    rollup = TABLES / "delta_recall1_runs.csv"
    rollup_exists = rollup.exists()
    with open(rollup, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        if not rollup_exists: w.writeheader()
        for row in summary: w.writerow(row)

    # ---- 2) CMC
    Kmax = 20
    def write_cmc(path: Path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["transform"] + [f"K{k}" for k in range(1, Kmax + 1)]
            w.writerow(header)
            for tf in sorted(rows):
                w.writerow([tf] + [f"{v:.4f}" for v in cmc_curve(rows[tf], Kmax)])

    cmc_before_ts = TABLES / f"cmc_before_{ts}.csv"
    cmc_after_ts  = TABLES / f"cmc_after_{name_tag}_{ts}.csv"
    cmc_before    = TABLES / "cmc_before.csv"
    cmc_after     = TABLES / f"cmc_after_{name_tag}.csv"

    write_cmc(cmc_before_ts, gb_b)
    write_cmc(cmc_after_ts, gb_a)
    for s,d in [(cmc_before_ts, cmc_before), (cmc_after_ts, cmc_after)]:
        try: d.write_bytes(s.read_bytes())
        except Exception: pass

    # ---- 3) Coverage–Recall
    def write_covrec(path: Path, grouped, top1_col):
        rows_out = []
        for tf, items in grouped.items():
            curve = coverage_recall(items, top1_col=top1_col, n_grid=150)
            for tau, cov, rec in curve:
                rows_out.append({
                    "transform": tf,
                    "tau": f"{tau:.4f}",
                    "coverage": f"{cov:.4f}",
                    "recall_at1": f"{rec:.4f}",
                })
        if rows_out:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
                w.writeheader(); w.writerows(rows_out)

    cov_b_ts = TABLES / f"coverage_recall_before_{ts}.csv"
    cov_a_ts = TABLES / f"coverage_recall_after_{name_tag}_{ts}.csv"
    cov_b    = TABLES / "coverage_recall_before.csv"
    cov_a    = TABLES / f"coverage_recall_after_{name_tag}.csv"

    write_covrec(cov_b_ts, gb_b, top1_col=top1_before)
    write_covrec(cov_a_ts, gb_a, top1_col=top1_after)
    for s,d in [(cov_b_ts, cov_b), (cov_a_ts, cov_a)]:
        try: d.write_bytes(s.read_bytes())
        except Exception: pass

    print(f"[OK] Saved analysis to {TABLES}  (dataset={dataset}, profile={profile}, stage={name_tag}, variant={variant_tag or '-'})")
    return {
        "delta": delta_ts,
        "cmc_before": cmc_before_ts,
        "cmc_after": cmc_after_ts,
        "cov_before": cov_b_ts,
        "cov_after": cov_a_ts,
    }

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Analyze rerank results vs baseline (ΔRecall@1, CMC, Coverage–Recall).")
    ap.add_argument("--dataset", type=str, required=True, help="데이터셋 이름(e.g., visdrone)")
    ap.add_argument("--profile", type=str, default=None, help="결과 하위 폴더(orb/light/super). 미지정 시 자동 감지")
    # 대상 지정: 1) 특정 rerank CSV 경로, 2) 특정 stage, 3) 전부 스캔
    ap.add_argument("--after-csv", type=str, default=None, help="분석할 rerank CSV 경로 (search_results_rerank_*.csv)")
    ap.add_argument("--stage", type=str, default=None, choices=["orb","light","super"], help="--after-csv에 대한 스테이지 수동 지정")
    ap.add_argument("--scan-all", action="store_true",
                    help="results/<dataset>/<profile>/**/tables/search_results_rerank_*.csv 전부 분석(variants 포함)")
    args = ap.parse_args()

    dataset = args.dataset
    profile = detect_profile(args.profile)

    if args.after_csv:
        run_for_file(dataset, profile, Path(args.after_csv), args.stage)
        return

    if args.scan_all:
        base_dir = RESULTS / dataset / profile
        # variants 하위까지 재귀 스캔
        candidates = list(base_dir.rglob("tables/search_results_rerank_*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No rerank CSVs found under {base_dir}")
        for f in sorted(candidates):
            run_for_file(dataset, profile, f, None)
        return

    if args.stage:
        # 해당 stage의 모든 rerank CSV 처리 (variants 포함)
        stage_dir = RESULTS / dataset / profile / args.stage / "tables"
        files = sorted(stage_dir.rglob("search_results_rerank_*.csv"))
        if not files:
            raise FileNotFoundError(f"No rerank CSVs in {stage_dir}")
        for f in files:
            run_for_file(dataset, profile, f, args.stage)
        return

    # 기본: profile 아래 모든 stage 스캔 (variants 포함)
    base_dir = RESULTS / dataset / profile
    candidates = list(base_dir.rglob("tables/search_results_rerank_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No rerank CSVs found under {base_dir}")
    for f in sorted(candidates):
        run_for_file(dataset, profile, f, None)

if __name__ == "__main__":
    main()
