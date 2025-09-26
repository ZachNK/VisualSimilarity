# scripts/rollup_all.py  — robust reader version
import os, time
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- base ----------
def resolve_base() -> Path:
    env = os.getenv("DINO_BASE")
    if env and Path(env).exists(): return Path(env)
    p_win = Path(r"D:/KNK/_KSNU/_Projects/dino_test")
    if p_win.exists(): return p_win
    p_wsl = Path("/mnt/d/KNK/_KSNU/_Projects/dino_test")
    if p_wsl.exists(): return p_wsl
    raise FileNotFoundError("Set DINO_BASE or check drive mount.")

BASE = resolve_base()
RESULTS = BASE / "results"

DATASETS = ["visdrone","sodaa","aihub"]
PROFILES = ["orb","light","super"]
STAGES   = ["orb","light","super"]

ROLLUP_DIR = RESULTS / "_rollup"
(ROLLUP_DIR / "tables").mkdir(parents=True, exist_ok=True)
TS = time.strftime("%Y%m%d-%H%M%S")

def read_csv_safe(path: Path, usecols=None):
    if not path.exists(): return None
    # 1차: 기본 엔진
    try:
        return pd.read_csv(path, usecols=usecols)
    except Exception:
        pass
    # 2차: 파이썬 엔진 + 불량 라인 스킵
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", usecols=usecols)
    except Exception:
        # 3차: 전부 문자열로 읽고, 나중에 필요한 컬럼만 취함
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str)
            if usecols is not None:
                keep = [c for c in usecols if c in df.columns]
                if not keep: return None
                df = df[keep].copy()
            return df
        except Exception:
            return None

# ---------- 1) ΔR@1 (최신 실행만) ----------
need_cols = ["dataset","profile","stage","transform","Recall@1_before","Recall@1_after","Delta","N","run_ts"]

delta_rows = []
for ds in DATASETS:
    for pf in PROFILES:
        tdir = RESULTS / ds / pf / "analysis" / "tables"
        roll = tdir / "delta_recall1_runs.csv"
        df = read_csv_safe(roll)
        if df is None or df.empty: continue
        # 안전장치: 필요한 컬럼만
        df = df[[c for c in need_cols if c in df.columns]].copy()
        if df.empty: continue
        # 최신 실행만: (stage, transform)별 run_ts 최대 선택
        if "run_ts" not in df.columns: continue
        df["key"] = list(zip(df.get("stage", ""), df.get("transform","")))
        idx = df.groupby("key")["run_ts"].transform(lambda s: s==s.max())
        df_latest = df[idx].drop(columns=["key"])
        delta_rows.append(df_latest)

delta_all_latest = pd.concat(delta_rows, ignore_index=True) if delta_rows else pd.DataFrame(columns=need_cols)
delta_all_latest.to_csv(ROLLUP_DIR / "tables" / f"rollup_delta_latest_{TS}.csv", index=False)
delta_all_latest.to_csv(ROLLUP_DIR / "tables" / "rollup_delta_latest.csv", index=False)

# dataset/profile/stage별 평균 ΔR@1 요약
if not delta_all_latest.empty:
    dd = delta_all_latest.copy()
    for c in ("Delta","Recall@1_before","Recall@1_after"):
        if c in dd.columns:
            dd[c] = pd.to_numeric(dd[c], errors="coerce")
    summary_delta = (dd
        .groupby(["dataset","profile","stage"], as_index=False)
        .agg(mean_delta=("Delta","mean"),
             median_delta=("Delta","median"),
             n_transforms=("Delta","count")))
else:
    summary_delta = pd.DataFrame(columns=["dataset","profile","stage","mean_delta","median_delta","n_transforms"])

summary_delta.to_csv(ROLLUP_DIR / "tables" / f"summary_delta_by_dps_{TS}.csv", index=False)
summary_delta.to_csv(ROLLUP_DIR / "tables" / "summary_delta_by_dps.csv", index=False)

# ---------- 2) CMC R@{1,5,10,20} ----------
def melt_cmc(df, tag):
    if df is None or df.empty: return None
    ks = [c for c in df.columns if c.startswith("K")]
    if "transform" not in df.columns or not ks: return None
    out = df[["transform"]+ks].copy()
    ren = { "K1":f"R1_{tag}", "K5":f"R5_{tag}", "K10":f"R10_{tag}", "K20":f"R20_{tag}" }
    out = out.rename(columns={k:v for k,v in ren.items() if k in out.columns})
    keep = ["transform"]+[v for v in ren.values() if v in out.columns]
    return out[keep]

cmc_rows = []
for ds in DATASETS:
    for pf in PROFILES:
        tdir = RESULTS / ds / pf / "analysis" / "tables"
        cmc_b = read_csv_safe(tdir / "cmc_before.csv")
        mf_b = melt_cmc(cmc_b, "before")
        for st in STAGES:
            cmc_a = read_csv_safe(tdir / f"cmc_after_{st}.csv")
            mf_a = melt_cmc(cmc_a, "after")
            if mf_b is None or mf_a is None: continue
            merged = pd.merge(mf_b, mf_a, on="transform", how="outer")
            merged["dataset"]=ds; merged["profile"]=pf; merged["stage"]=st
            cmc_rows.append(merged)

cmc_all = pd.concat(cmc_rows, ignore_index=True) if cmc_rows else pd.DataFrame()
cmc_all.to_csv(ROLLUP_DIR / "tables" / f"rollup_cmc_r1_5_10_20_{TS}.csv", index=False)
cmc_all.to_csv(ROLLUP_DIR / "tables" / "rollup_cmc_r1_5_10_20.csv", index=False)

# ---------- 3) Coverage–Recall AUC ----------
def auc_covrec(df):
    if df is None or df.empty: return None
    x = pd.to_numeric(df.get("coverage"), errors="coerce").values
    y = pd.to_numeric(df.get("recall_at1"), errors="coerce").values
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x)==0 or len(y)==0: return None
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    return float(np.trapz(y, x))

cov_auc_rows = []
for ds in DATASETS:
    for pf in PROFILES:
        tdir = RESULTS / ds / pf / "analysis" / "tables"
        cov_b = read_csv_safe(tdir / "coverage_recall_before.csv")
        if cov_b is not None and not cov_b.empty:
            auc_b = cov_b.groupby("transform").apply(auc_covrec).reset_index(name="AUC_before")
        else:
            auc_b = pd.DataFrame(columns=["transform","AUC_before"])
        for st in STAGES:
            cov_a = read_csv_safe(tdir / f"coverage_recall_after_{st}.csv")
            if cov_a is None or cov_a.empty: continue
            ga = cov_a.groupby("transform").apply(auc_covrec).reset_index(name="AUC_after")
            merged = pd.merge(auc_b, ga, on="transform", how="outer")
            merged["dataset"]=ds; merged["profile"]=pf; merged["stage"]=st
            cov_auc_rows.append(merged)

cov_auc_all = pd.concat(cov_auc_rows, ignore_index=True) if cov_auc_rows else pd.DataFrame()
cov_auc_all.to_csv(ROLLUP_DIR / "tables" / f"rollup_covrec_auc_{TS}.csv", index=False)
cov_auc_all.to_csv(ROLLUP_DIR / "tables" / "rollup_covrec_auc.csv", index=False)

# ---------- 4) Best stage per dataset/transform ----------
best_rows = []
if not delta_all_latest.empty:
    d = delta_all_latest.copy()
    d["Delta"] = pd.to_numeric(d["Delta"], errors="coerce")
    if "dataset" in d.columns and "transform" in d.columns:
        idx = d.groupby(["dataset","transform"])["Delta"].transform(lambda s: s==s.max())
        best = d[idx].copy()
        best_rows.append(best)
best_all = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
best_all.to_csv(ROLLUP_DIR / "tables" / f"best_stage_per_dataset_transform_{TS}.csv", index=False)
best_all.to_csv(ROLLUP_DIR / "tables" / "best_stage_per_dataset_transform.csv", index=False)

print("[OK] Wrote rollups under", ROLLUP_DIR / "tables")
