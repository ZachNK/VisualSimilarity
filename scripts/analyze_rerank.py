# (analyze_rerank.py) 전/후 지표 산출: ΔRecall@1, CMC, Coverage–Recall
import numpy as np
import csv
from pathlib import Path
from collections import defaultdict

BASE     = Path(r"D:/KNK/_KSNU/_Projects/dino_test")
RESULTS  = BASE / "results"
TABLES   = RESULTS / "tables"

BEFORE = TABLES / "search_results.csv"
AFTER  = TABLES / "search_results_rerank_orb.csv"

def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def recall_at_k(rows, K):
    ok = sum(1 for r in rows if int(r["rank_pos"]) != -1 and int(r["rank_pos"]) <= K)
    return ok / max(len(rows), 1)

def cmc_curve(rows, Kmax=20):
    # 전체 쿼리 대비 누적 매칭율(CMC)
    ranks = [int(r["rank_pos"]) for r in rows if int(r["rank_pos"]) != -1]
    curve = []
    for k in range(1, Kmax+1):
        curve.append(sum(1 for rk in ranks if rk <= k) / max(len(rows), 1))
    return curve

# --- helpers: column picking & safe float ---
def pick_top1_col(rows):
    if not rows:
        raise RuntimeError("Empty rows.")
    keys = rows[0].keys()
    for c in ("top1_dist", "top1_score", "top1_value"):
        if c in keys:
            return c
    raise KeyError(f"Cannot find top1 distance/score column in: {list(keys)}")

def tofloat(x):
    try:
        return float(x)
    except:
        return float("nan")

def is_ip_col(col_name: str) -> bool:
    # top1_score면 IP일 확률 높음, top1_dist면 L2
    return "score" in col_name.lower()

# Coverage–Recall 커브 (L2): 'top1 <= tau' 이면 수락
def coverage_recall(rows, top1_col, n_grid=200):
    vals = np.array([tofloat(r[top1_col]) for r in rows], dtype=np.float32)
    gold_top1 = np.array([r["top1_name"] == r["gold"] for r in rows], dtype=bool)

    # NaN 제거
    ok = ~np.isnan(vals)
    vals = vals[ok]; gold_top1 = gold_top1[ok]
    if vals.size == 0:
        return []

    taus = np.quantile(vals, np.linspace(0, 1, n_grid))
    use_ip = is_ip_col(top1_col)

    out = []
    for t in taus:
        accepted = (vals >= t) if use_ip else (vals <= t)
        cov = accepted.mean()
        rec = (gold_top1 & accepted).mean()
        out.append((float(t), float(cov), float(rec)))
    return out

def group_by_tf(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["transform"]].append(r)
    return d

def main():
    before = load_rows(BEFORE)
    after  = load_rows(AFTER)

    # 파일별 top1 컬럼 자동 감지
    top1_before = pick_top1_col(before)   # ex) 'top1_dist' or 'top1_score'
    top1_after  = pick_top1_col(after)    # ex) 'top1_dist'

    gb_b = group_by_tf(before)
    gb_a = group_by_tf(after)

    # 1) ΔRecall@1 요약
    summary = []
    for tf in sorted(set(gb_b) | set(gb_a)):
        rb = gb_b.get(tf, [])
        ra = gb_a.get(tf, [])
        rec1_b = recall_at_k(rb, 1)
        rec1_a = recall_at_k(ra, 1)
        summary.append({
            "transform": tf,
            "Recall@1_before": f"{rec1_b:.4f}",
            "Recall@1_after":  f"{rec1_a:.4f}",
            "Delta":           f"{(rec1_a - rec1_b):+.4f}",
            "N": len(ra) or len(rb)
        })
    
    # ✅ 경로 변수 수정: TABLES 사용
    with open(TABLES / "delta_recall1_orb.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    # 2) CMC 전/후 (변형별)
    Kmax = 20
    with open(TABLES / "cmc_before.csv", "w", newline="", encoding="utf-8") as fb, \
         open(TABLES / "cmc_after.csv",  "w", newline="", encoding="utf-8") as fa:
        kb = csv.writer(fb); ka = csv.writer(fa)
        header = ["transform"] + [f"K{k}" for k in range(1, Kmax+1)]
        kb.writerow(header); ka.writerow(header)
        for tf in sorted(gb_b):
            kb.writerow([tf] + [f"{v:.4f}" for v in cmc_curve(gb_b[tf], Kmax)])
        for tf in sorted(gb_a):
            ka.writerow([tf] + [f"{v:.4f}" for v in cmc_curve(gb_a[tf], Kmax)])

    # 3) Coverage–Recall 전/후
    #    -> 동일 τ에서 Recall@1이 얼마나 개선되는지 비교 가능
    for gb, outname, top1_col in [
        (gb_b, "coverage_recall_before.csv", top1_before),
        (gb_a, "coverage_recall_after.csv",  top1_after),
    ]:
        rows_out = []
        for tf, items in gb.items():
            curve = coverage_recall(items, top1_col=top1_col, n_grid=150)
            for tau, cov, rec in curve:
                rows_out.append({
                    "transform": tf,
                    "tau": f"{tau:.4f}",
                    "coverage": f"{cov:.4f}",
                    "recall_at1": f"{rec:.4f}"
                })
        if rows_out:
            with open(TABLES / outname, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
                w.writeheader(); w.writerows(rows_out)

    print("Saved: delta_recall1_orb.csv, cmc_before/after.csv, coverage_recall_before/after.csv")

if __name__ == "__main__":
    main()
