# Scripts/search_query.py — 쿼리 검색 (CLI, FAISS GPU FP16, tqdm, per-env 저장)

### ================================ 사용방법 ================================ 

#ex) orb2025 가상환경에서 visdrone, 진행바 + per-query 시간 CSV
# (orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python search_query.py --dataset visdrone --device cuda --k 5 --save-csv --tag smoke
# 저장 -> D:\...\results\visdrone\orb\smoke\tables\search_query_smoke.csv

# 프로파일을 수동 지정하고 싶을 때
# python search_query.py --dataset visdrone --device cuda --k 5 --save-csv --tag smoke --profile orb

### ================================ 사용방법 ================================ 

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, csv, time
from pathlib import Path
import faiss, numpy as np
from dinov2_utils import get_embedding
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------- base resolve ----------------
def resolve_base() -> Path:
    env = os.getenv("DINO_BASE")
    if env and Path(env).exists(): return Path(env)
    p_win = Path(r"D:/KNK/_KSNU/_Projects/dino_test")
    if p_win.exists(): return p_win
    p_wsl = Path("/mnt/d/KNK/_KSNU/_Projects/dino_test")
    if p_wsl.exists(): return p_wsl
    raise FileNotFoundError("Project base not found. Set DINO_BASE or check drive mount.")

BASE = resolve_base()
DATA = BASE / "data"
RESULTS = BASE / "results"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ---------------- profile detect (env → orb/light/super) ----------------
def detect_profile(cli_profile: str | None) -> str:
    if cli_profile:
        return cli_profile.strip()
    env_name = os.getenv("CONDA_DEFAULT_ENV") or (Path(os.getenv("VIRTUAL_ENV", "")).name or "")
    env_lower = env_name.lower()
    for key in ("orb", "light", "super"):
        if key in env_lower:
            return key
    return "default"

def default_queries_dir(dataset: str) -> Path:
    c1 = DATA / "corpora" / dataset / "images"
    return c1 if c1.exists() else (DATA / "corpora" / dataset)

def load_index(index_dir: Path):
    idx_path = index_dir / "faiss_index.bin"
    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    cpu_index = faiss.read_index(str(idx_path))
    index = cpu_index
    try:
        ngpu = faiss.get_num_gpus()
        if ngpu > 0 and "HNSW" not in type(cpu_index).__name__.upper():
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions(); co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
            print(f"[FAISS] GPU search enabled (fp16).")
        else:
            print("[FAISS] CPU search.")
    except Exception as e:
        print("[FAISS] GPU wrap skipped:", e)
    return index

def main():
    ap = argparse.ArgumentParser(description="Search top-K images for queries using FAISS.")
    ap.add_argument("--dataset", type=str, help="데이터셋 이름(예: visdrone, sodaa, aihub, union)")
    ap.add_argument("--index-dir", type=str, default=None, help="인덱스 폴더(기본: data/index/<dataset>)")
    ap.add_argument("--queries-dir", type=str, default=None, help="쿼리 폴더(기본: data/corpora/<dataset>/[images])")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"],
                    help="임베딩 디바이스 힌트(get_embedding 내부 구현에 따라 무시될 수 있음)")
    ap.add_argument("--k", type=int, default=None, help="top-K (기본: min(5, index.ntotal))")
    # 저장 옵션
    ap.add_argument("--save-csv", action="store_true", help="per-query 결과/시간을 CSV로 저장")
    ap.add_argument("--tag", type=str, default="smoke", help="결과 저장 하위 태그(스테이지) (기본: smoke)")
    ap.add_argument("--profile", type=str, default=None, help="결과 하위 프로파일(orb/light/super). 미지정 시 자동 추론")
    args = ap.parse_args()

    if not args.dataset and (args.index_dir is None or args.queries_dir is None):
        raise SystemExit("Either --dataset OR both --index-dir and --queries-dir must be provided.")

    dataset = args.dataset or "custom"
    index_dir = Path(args.index_dir) if args.index_dir else (DATA / "index" / dataset)
    queries_dir = Path(args.queries_dir) if args.queries_dir else default_queries_dir(dataset)
    profile = detect_profile(args.profile)

    if args.device in ("cuda", "cpu"):
        os.environ["DINO_DEVICE"] = args.device

    index = load_index(index_dir)
    paths_npy = index_dir / "image_paths.npy"
    if not paths_npy.exists():
        raise FileNotFoundError(f"image_paths.npy not found: {paths_npy}")
    image_paths = [str(p) for p in np.load(paths_npy, allow_pickle=True)]

    if not queries_dir.exists():
        raise FileNotFoundError(f"Queries dir not found: {queries_dir}")
    qfiles = sorted([p for p in queries_dir.rglob("*") if p.suffix.lower() in EXTS and p.is_file()])
    if not qfiles:
        raise RuntimeError(f"No query images under: {queries_dir}")

    K = args.k if args.k is not None else max(1, min(5, index.ntotal))

    # 저장 경로: results/<dataset>/<profile>/<tag>/tables/
    rows = []
    out_csv = None
    if args.save_csv:
        OUTROOT = RESULTS / dataset / profile / args.tag
        TABLES = OUTROOT / "tables"
        TABLES.mkdir(parents=True, exist_ok=True)
        # 태그가 smoke면 파일명은 search_query_smoke.csv, 아니면 태그 반영
        fname = f"search_query_{args.tag}.csv" if args.tag else "search_query.csv"
        out_csv = TABLES / fname

    print("="*60)
    print(f"Dataset   : {dataset}")
    print(f"Profile   : {profile}")
    print(f"Index Dir : {index_dir}")
    print(f"Queries   : {queries_dir}  (N={len(qfiles)})")
    print(f"Top-K     : {K}")
    print("="*60)

    pbar = tqdm(qfiles, desc=f"Search ({dataset})", unit="q", ncols=100)
    for q in pbar:
        try:
            t0 = time.perf_counter()
            qvec = get_embedding(str(q)).reshape(1, -1).astype("float32")
            t1 = time.perf_counter()
            dists, inds = index.search(qvec, K)
            t2 = time.perf_counter()
        except Exception as e:
            pbar.set_postfix_str(f"err:{q.name[:18]}")
            tqdm.write(f"[ERROR] {q}: {e}")
            continue

        dists, inds = dists[0], inds[0]
        names = []
        for r, i in enumerate(inds, 1):
            name = Path(image_paths[i]).name if 0 <= i < len(image_paths) else f"UNKNOWN({i})"
            names.append(name)

        # 진행바 요약
        embed_ms = (t1 - t0) * 1000.0
        search_ms = (t2 - t1) * 1000.0
        pbar.set_postfix_str(f"{q.name[:18]} → {names[0][:18]}  E{embed_ms:.0f}ms S{search_ms:.0f}ms")

        # 상세 결과는 tqdm.write로
        tqdm.write(f"\nQuery: {q.name}")
        for r, (nm, dist) in enumerate(zip(names, dists.tolist()), 1):
            tqdm.write(f"  {r}. {nm} (dist={dist:.4f})")

        if args.save_csv:
            rows.append({
                "query": q.name,
                "topk_names": ";".join(names),
                "dist@K": ";".join(f"{v:.4f}" for v in dists.tolist()),
                "K": K,
                "t_embed_ms": f"{embed_ms:.2f}",
                "t_search_ms": f"{search_ms:.2f}",
            })
    pbar.close()

    if args.save_csv and rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n[Saved] {out_csv}")

if __name__ == "__main__":
    main()
