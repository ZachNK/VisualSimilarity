# scripts/analyze_perquery.py
import os, csv, time, argparse, re
from pathlib import Path

def resolve_base() -> Path:
    env = os.getenv("DINO_BASE")
    if env and Path(env).exists(): return Path(env)
    p_win = Path(r"D:/KNK/_KSNU/_Projects/dino_test")
    if p_win.exists(): return p_win
    p_wsl = Path("/mnt/d/KNK/_KSNU/_Projects/dino_test")
    if p_wsl.exists(): return p_wsl
    raise FileNotFoundError("Set DINO_BASE.")

def detect_profile(cli: str | None) -> str:
    if cli: return cli.strip()
    env = os.getenv("CONDA_DEFAULT_ENV") or (Path(os.getenv("VIRTUAL_ENV","")).name or "")
    low = env.lower()
    for k in ("orb","light","super"):
        if k in low: return k
    return "default"

def resolve_results_root(cli_root: str | None, base: Path) -> Path:
    # 우선순위: --results-root > DINO_RESULTS/RESULTS_ROOT > <base>/results
    if cli_root:
        p = Path(cli_root)
        if not p.exists(): raise FileNotFoundError(f"--results-root not found: {p}")
        return p
    for envk in ("DINO_RESULTS","RESULTS_ROOT"):
        ev = os.getenv(envk, "")
        if ev and Path(ev).exists():
            return Path(ev)
    return base / "results"

# ---------- I/O utils ----------
def load_rows(p: Path):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def to_int(x, miss=10**9):
    try:
        return int(float(x))
    except:
        return miss

def pick_top1_col(rows):
    if not rows: return None
    keys = set(rows[0].keys())
    # after 쪽은 top1_combined(있으면) > top1_geom > top1_dist 순으로 선호
    for c in ("top1_combined","top1_geom","top1_dist","top1_score","top1_value"):
        if c in keys: return c
    return None

def find_baseline_csv(results_root: Path, dataset: str, profile: str) -> Path:
    p1 = results_root / dataset / profile / "baseline" / "tables" / "search_results.csv"
    if p1.exists(): return p1
    p2 = results_root / "tables" / "search_results.csv"    # 예전 폴더 구조 폴백
    if p2.exists(): return p2
    raise FileNotFoundError(f"Baseline CSV not found under {results_root}")

def guess_stage_from_path(p: Path) -> str:
    name = p.name.lower()
    for k in ("orb","light","super"):
        if f"rerank_{k}" in name or f"_{k}_" in name or name.endswith(f"_{k}.csv"):
            return k
    parts = [s.lower() for s in p.parts]
    for k in ("orb","light","super"):
        if k in parts: return k
    return "unknown"

# ---------- metrics ----------
def recall_at_k(rows, K: int) -> float:
    ok = sum(1 for r in rows if to_int(r.get("rank_pos",-1), -1) != -1 and to_int(r["rank_pos"]) <= K)
    return ok / max(len(rows), 1)

def cmc_curve(rows, Kmax: int = 20):
    ranks = [to_int(r.get("rank_pos",-1), -1) for r in rows if to_int(r.get("rank_pos",-1), -1) != -1]
    return [sum(1 for rk in ranks if rk <= k) / max(len(rows), 1) for k in range(1, Kmax + 1)]

def tofloat(x):
    try: return float(x)
    except: return float("nan")

def is_ip_col(col_name: str) -> bool:
    return "score" in (col_name or "").lower() and "dist" not in (col_name or "").lower()

def coverage_recall(rows, top1_col: str, n_grid: int = 200):
    import numpy as np
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

# ---------- transform parsing (옵션) ----------
import re
ROT_RE  = re.compile(r"(rot|rot_|angle|a_)?([+-]?\d{1,3})")
BRT_RE  = re.compile(r"(bright|illum|brt|b_)?(low|mid|high|dark|bright)")

def parse_transform(tf: str):
    tf = (tf or "").lower()
    ang = None; br = None
    m = ROT_RE.search(tf)
    if m:
        try: ang = int(m.group(2))
        except: pass
    m = BRT_RE.search(tf)
    if m:
        br = m.group(2)
        if br == "mid": br = "normal"
        if br == "dark": br = "low"
        if br == "bright": br = "high"
    return ang, br

# ---------- run ----------
def main():
    base = resolve_base()

    ap = argparse.ArgumentParser(description="Per-query comparison (baseline vs rerank).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--profile", default=None, help="orb/light/super (auto if unset)")
    ap.add_argument("--after-csv", default=None, help="path to search_results_rerank_*.csv (else pick latest under stage)")
    ap.add_argument("--stage", choices=["orb","light","super"], help="stage of after-csv")
    ap.add_argument("--results-root", default=None, help="RESULTS root override (e.g. D:/.../results_legacy_250917)")
    args = ap.parse_args()

    profile = detect_profile(args.profile)
    results_root = resolve_results_root(args.results_root, base)

    BEFORE = find_baseline_csv(results_root, args.dataset, profile)

    if args.after_csv:
        AFTER = Path(args.after_csv)
        stage = (args.stage or guess_stage_from_path(AFTER))
    else:
        stage = (args.stage or profile)
        tdir = results_root / args.dataset / profile / stage / "tables"
        cands = sorted(tdir.glob("search_results_rerank_*.csv"))
        if not cands:
            raise FileNotFoundError(f"No rerank CSVs under {tdir}")
        AFTER = cands[-1]

    print(f"[INFO] dataset={args.dataset} profile={profile} stage={stage}")
    print(f"[INFO] RESULTS_ROOT={results_root}")
    print(f"[INFO] BEFORE={BEFORE}")
    print(f"[INFO] AFTER ={AFTER}")

    rows_b = load_rows(BEFORE)
    rows_a = load_rows(AFTER)

    idx_b = {r["query"]: r for r in rows_b}
    idx_a = {r["query"]: r for r in rows_a}

    top1_b_col = pick_top1_col(rows_b)
    top1_a_col = pick_top1_col(rows_a)

    out = []
    for q, ra in idx_a.items():
        rb = idx_b.get(q)
        if not rb: 
            continue
        tf   = ra.get("transform","")
        rkb  = to_int(rb.get("rank_pos",-1), -1)
        rka  = to_int(ra.get("rank_pos",-1), -1)
        miss = 10**6
        rkb2 = miss if rkb < 1 else rkb
        rka2 = miss if rka < 1 else rka
        delta_rank = rkb2 - rka2  # +면 개선
        hit_b = 1 if rkb==1 else 0
        hit_a = 1 if rka==1 else 0
        ang, br = parse_transform(tf)

        out.append({
            "dataset": args.dataset, "profile": profile, "stage": stage,
            "query": q, "base_id": rb.get("base_id",""), "transform": tf,
            "angle_deg": ang if ang is not None else "",
            "brightness": br if br is not None else "",
            "rank_before": rkb if rkb>0 else -1,
            "rank_after":  rka if rka>0 else -1,
            "delta_rank":  (delta_rank if delta_rank<miss else ""),
            "hit_before": hit_b, "hit_after": hit_a, "delta_hit": hit_a - hit_b,
            "top1_before": rb.get("top1_name",""),
            "top1_after":  ra.get("top1_name",""),
            "top1_score_before": rb.get(top1_b_col,"") if top1_b_col else "",
            "top1_score_after":  ra.get(top1_a_col,"") if top1_a_col else "",
            "t_embed_ms": ra.get("t_embed_ms",""),
            "t_search_ms":ra.get("t_search_ms",""),
            "t_geom_ms":  ra.get("t_geom_ms",""),
            "t_total_ms": ra.get("t_total_ms",""),
            "alpha": ra.get("alpha",""),
            "K":     ra.get("K",""),
            "gate_on": ra.get("gate_on",""),
            "gate_margin": ra.get("gate_margin",""),
            "gate_inlier": ra.get("gate_inlier",""),
            "rerank_applied": ra.get("rerank_applied",""),
            "rerank_k": ra.get("rerank_k",""),
        })

    if not out:
        raise RuntimeError("No matched queries between BEFORE and AFTER.")

    OUTDIR = results_root / args.dataset / profile / "analysis" / "per_query"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")

    perq_ts = OUTDIR / f"per_query_{stage}_{ts}.csv"
    perq_latest = OUTDIR / f"per_query_{stage}.csv"

    with open(perq_ts, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    try: perq_latest.write_bytes(perq_ts.read_bytes())
    except: pass

    # 원본(21변형) 요약
    by_base = {}
    for r in out:
        by_base.setdefault(r["base_id"], []).append(r)

    agg = []
    for base_id, items in by_base.items():
        n = len(items)
        hit_b = sum(int(x["hit_before"]) for x in items)
        hit_a = sum(int(x["hit_after"])  for x in items)
        rk_b = [to_int(x["rank_before"], 10**6) for x in items]
        rk_a = [to_int(x["rank_after"],  10**6) for x in items]
        drs  = [x["delta_rank"] for x in items if isinstance(x["delta_rank"], int)]
        agg.append({
            "dataset": args.dataset, "profile": profile, "stage": stage, "base_id": base_id,
            "n_transforms": n,
            "hit1_before_rate": round(hit_b/max(n,1), 4),
            "hit1_after_rate":  round(hit_a/max(n,1), 4),
            "mean_rank_before": round(sum(rk_b)/max(n,1), 2),
            "mean_rank_after":  round(sum(rk_a)/max(n,1), 2),
            "worst_rank_after": max(rk_a) if rk_a else "",
            "improved_cnt": sum(1 for x in items if str(x["delta_hit"])=="1" or (isinstance(x["delta_rank"], int) and x["delta_rank"]>0)),
            "worsened_cnt": sum(1 for x in items if str(x["delta_hit"])=="-1" or (isinstance(x["delta_rank"], int) and x["delta_rank"]<0)),
            "mean_delta_rank": round(sum(drs)/max(len(drs),1),2) if drs else "",
        })

    if agg:
        base_ts = OUTDIR / f"per_base_{stage}_{ts}.csv"
        base_latest = OUTDIR / f"per_base_{stage}.csv"
        with open(base_ts, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            w.writeheader(); w.writerows(agg)
        try: base_latest.write_bytes(base_ts.read_bytes())
        except: pass

    # 각도×밝기 민감도 (옵션)
    grid = []
    gb = {}
    for r in out:
        key = (r["angle_deg"] or "NA", r["brightness"] or "NA")
        gb.setdefault(key, []).append(r)
    for (ang, br), items in gb.items():
        n = len(items)
        if n == 0: 
            continue
        hr_b = sum(int(x["hit_before"]) for x in items)/n
        hr_a = sum(int(x["hit_after"])  for x in items)/n
        dr   = [x["delta_rank"] for x in items if isinstance(x["delta_rank"], int)]
        grid.append({"dataset":args.dataset,"profile":profile,"stage":stage,
                     "angle_deg":ang,"brightness":br,"n":n,
                     "hit1_rate_before":round(hr_b,4),
                     "hit1_rate_after": round(hr_a,4),
                     "mean_delta_rank": round(sum(dr)/max(len(dr),1),2) if dr else ""})
    if grid:
        grid_ts = OUTDIR / f"sensitivity_angle_brightness_{stage}_{ts}.csv"
        with open(grid_ts, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
            w.writeheader(); w.writerows(grid)

    print("[OK]", perq_ts)
    if agg: print("[OK]", base_ts)
    print("[OK] angle x brightness grid:", grid_ts if grid else "(none)")

if __name__ == "__main__":
    main()
