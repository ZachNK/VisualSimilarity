# Scripts/build_index.py — 데이터셋별 인덱스 구축 (CLI) + 진행바 + 성능 로그 CSV

### ================================ 사용방법 ================================ 
# img2query_gpu.py로 쿼리 이미지 생성 한 후 다음 순서.

#ex) orb2025환경에서 visdrone 데이터셋
# (orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python build_index.py --dataset visdrone --device cuda --overwrite --profile orb
# -> 이제 기본 루트는: /mnt/d/KNK/_KSNU/_Projects/dino_test/data/corpora/visdrone/images  (없으면 datasets/visdrone...)

# 저장: results/visdrone/orb/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

### ================================ 사용방법 ================================ 

# 결과 구조: results/<dataset>/<profile>/tables/...
#   - build_index_runs.csv (append)
#   - build_index_perf_<YYYYmmdd-HHMMSS>.csv (1회 실행 스냅샷)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import time
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

# -----------------------------------
# 프로젝트 베이스 경로 해석
# -----------------------------------
def resolve_base():
    env = os.getenv("DINO_BASE")
    if env:
        p = Path(env)
        if p.exists():
            return p

    p_win = Path(r"D:/KNK/_KSNU/_Projects/dino_test")
    if p_win.exists():
        return p_win

    p_wsl = Path("/mnt/d/KNK/_KSNU/_Projects/dino_test")
    if p_wsl.exists():
        return p_wsl

    raise FileNotFoundError(
        "Project base not found in Windows/WSL. "
        "Set DINO_BASE env var or check drive mount."
    )

BASE = resolve_base()
DATA = BASE / "data"
RESULTS = BASE / "results"

# -----------------------------------
# 프로파일(가상환경) 감지/정규화
# -----------------------------------
def detect_profile(cli_profile: str | None) -> str:
    if cli_profile:
        return cli_profile.strip()
    # conda/venv 이름에서 유추
    env_name = os.getenv("CONDA_DEFAULT_ENV") or (Path(os.getenv("VIRTUAL_ENV", "")).name or "")
    env_lower = env_name.lower()
    for key in ("orb", "light", "super"):
        if key in env_lower:
            return key
    return "default"

# -----------------------------------
# 유틸: 임베딩
# -----------------------------------
def get_embeddings_with_device(batch_paths, batch_size=64, device=None):
    """
    dinov2_utils.get_embeddings의 인자 호환성을 고려해 안전 호출.
    device 인자를 지원하지 않으면 자동으로 제외하여 호출.
    """
    from dinov2_utils import get_embeddings as _ge

    if device is not None:
        try:
            return _ge(batch_paths, batch_size=batch_size, device=device)
        except TypeError:
            pass
    return _ge(batch_paths, batch_size=batch_size)

# -----------------------------------
# 유틸: 이미지 루트 선택 유틸 추가
# -----------------------------------
def pick_images_root(dataset: str | None, images_dir: str | None) -> tuple[Path, str]:
    """
    우선순위:
      1) --images-dir (명시)
      2) data/corpora/<dataset>/images
      3) data/datasets/<dataset>         (legacy)
      4) data/images                     (최후 폴백)
    반환: (선택된 경로, 설명 문자열)
    """
    if images_dir:
        p = Path(images_dir)
        if not p.exists():
            raise RuntimeError(f"--images-dir not found: {p}")
        return p, "--images-dir"

    cand = []
    if dataset:
        cand = [
            DATA / "corpora"  / dataset / "images",
            DATA / "datasets" / dataset,             # legacy
        ]
    else:
        cand = [DATA / "images"]

    for p in cand:
        if p.exists():
            return p, f"auto:{p}"

    # 최후 폴백
    p = DATA / "images"
    if p.exists():
        return p, f"auto:{p}"

    raise RuntimeError(
        "No valid image root found.\n"
        + "\n".join(f"  tried: {c}" for c in cand + [DATA / 'images'])
    )

# -----------------------------------
# 이미지 수집
# -----------------------------------
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def collect_images(root: Path):
    if not root.exists():
        raise RuntimeError(f"Image root not found: {root}")
    paths = [p for p in root.rglob("*") if p.suffix.lower() in EXTS and p.is_file()]
    paths.sort()
    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    return paths

# -----------------------------------
# 메인
# -----------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from dataset images (DINOv2 embeddings)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dataset",
        type=str,
        help="데이터셋 이름 (예: visdrone, sodaa, aihub, union). "
             "이미지 루트는 data/datasets/<dataset> 으로 간주하여 재귀 탐색."
    )
    group.add_argument(
        "--images-dir",
        type=str,
        help="임의의 이미지 최상위 디렉터리 (재귀 탐색). 지정 시 --dataset 무시."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto", None],
        help="임베딩 디바이스 힌트 (dinov2_utils 구현에 따라 무시될 수 있음)."
    )
    parser.add_argument("--batch", type=int, default=64, help="임베딩 배치 크기")
    parser.add_argument("--tag", type=str, default=None,
                        help="결과 CSV/폴더 구분용 태그(미지정 시 dataset 또는 'custom').")
    parser.add_argument(
        "--index-out",
        type=str,
        default=None,
        help="인덱스 저장 폴더(기본: data/index/<tag>)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 인덱스 파일이 있어도 덮어쓰기"
    )
    # ▼ 추가: 결과 폴더의 <profile> 결정
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="결과 하위 폴더명(orb/light/super 등). 미지정 시 가상환경명에서 자동 추론."
    )
    # ▼ 추가: 실행 스냅샷 파일도 남길지 여부
    parser.add_argument(
        "--no-run-snapshot",
        action="store_true",
        help="실행별 스냅샷 CSV(build_index_perf_<ts>.csv) 생성을 생략하고 누적 CSV만 갱신"
    )

    args = parser.parse_args()

    # ---------------- 경로/태그/프로파일 결정
    if args.images_dir:
        images_root, picked_from = pick_images_root(None, args.images_dir)
        dataset_tag = args.tag or "custom"
    elif args.dataset:
        images_root, picked_from = pick_images_root(args.dataset, None)
        dataset_tag = args.tag or args.dataset
    else:
        images_root, picked_from = pick_images_root(None, None)
        dataset_tag = args.tag or "default"

    profile = detect_profile(args.profile)  # orb/light/super/default
    # 결과 저장 루트: results/<dataset>/<profile>
    OUTROOT = RESULTS / dataset_tag / profile
    OUT_TABLES = OUTROOT / "tables"
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    # 인덱스 저장 루트(변경 없음)
    index_dir = Path(args.index_out) if args.index_out else (DATA / "index" / dataset_tag)
    index_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 이미지 수집
    image_paths = collect_images(images_root)
    all_paths = [str(p) for p in image_paths]

    # ---------------- 디바이스 추론(표시용)
    try:
        import torch
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch_device = "unknown"

    faiss_ngpu = 0
    try:
        faiss_ngpu = faiss.get_num_gpus()
    except Exception:
        pass

    # ---------------- 임베딩
    t_total0 = time.perf_counter()
    emb_list = []
    embed_ms = 0.0

    device_hint = None
    if args.device == "auto":
        device_hint = None
    elif args.device in ("cuda", "cpu"):
        device_hint = args.device
        os.environ["DINO_DEVICE"] = args.device

    pbar = tqdm(range(0, len(all_paths), args.batch),
                desc=f"Embedding ({dataset_tag})", unit="batch", ncols=100)

    for i in pbar:
        batch = all_paths[i:i + args.batch]
        t0 = time.perf_counter()
        embs = get_embeddings_with_device(batch, batch_size=args.batch, device=device_hint)
        t1 = time.perf_counter()

        emb_list.append(embs)
        dt_ms = (t1 - t0) * 1000.0
        embed_ms += dt_ms
        pbar.set_postfix_str(f"last={Path(batch[-1]).name[:24]}..  {dt_ms:.0f} ms")
    pbar.close()

    embeddings = np.ascontiguousarray(np.vstack(emb_list), dtype="float32")
    dim = embeddings.shape[1]

    # ---------------- 인덱스 구축
    t_add0 = time.perf_counter()
    index = faiss.IndexFlatL2(dim)  # CPU 인덱스
    index.add(embeddings)
    t_add1 = time.perf_counter()

    # ---------------- 파일 저장 (index artifacts)
    idx_path = index_dir / "faiss_index.bin"
    paths_npy = index_dir / "image_paths.npy"

    if (idx_path.exists() or paths_npy.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Index already exists under {index_dir}. Use --overwrite to replace."
        )

    t_w0 = time.perf_counter()
    faiss.write_index(index, str(idx_path))
    np.save(paths_npy, np.array(all_paths, dtype=object))
    t_w1 = time.perf_counter()
    t_total1 = time.perf_counter()

    # ---------------- 요약 출력
    N = len(image_paths)
    print("=" * 60)
    print(f"Dataset Tag : {dataset_tag}")
    print(f"Profile     : {profile}")
    print(f"Images Root : {images_root}")
    print(f"Indexed N   : {N} images @ dim={dim}")
    print(f"Index Dir   : {index_dir}")
    print(f"Index File  : {idx_path.name}")
    print(f"Paths File  : {paths_npy.name}")
    print(f"Images Root : {images_root}  [source={picked_from}]")
    print("-" * 60)
    print(f"Embed device (torch avail): {torch_device} | FAISS nGPU: {faiss_ngpu}")
    print("=" * 60)

    # ---------------- 성능 로그 CSV (폴더 구조 정리)
    run_ts = time.strftime("%Y%m%d-%H%M%S")  # 실행 타임스탬프
    total_ms = (t_total1 - t_total0) * 1000.0
    add_ms = (t_add1 - t_add0) * 1000.0
    write_ms = (t_w1 - t_w0) * 1000.0
    img_per_s = (N / (embed_ms / 1000.0)) if embed_ms > 0 else 0.0

    row = {
        "run_ts": run_ts,
        "dataset_tag": dataset_tag,
        "profile": profile,
        "n_images": N,
        "dim": dim,
        "batch_size": args.batch,
        "embed_device_hint": args.device or "",
        "torch_device": torch_device,
        "faiss_ngpu": faiss_ngpu,
        "embed_ms": f"{embed_ms:.2f}",
        "index_add_ms": f"{add_ms:.2f}",
        "write_ms": f"{write_ms:.2f}",
        "total_ms": f"{total_ms:.2f}",
        "embed_img_per_sec": f"{img_per_s:.2f}",
        "images_root": str(images_root),
        "index_dir": str(index_dir),
        "index_path": str(idx_path),
        "paths_path": str(paths_npy),
    }

    # 1) 누적(Roll-up) CSV — append
    rollup_csv = OUT_TABLES / "build_index_runs.csv"
    write_header = not rollup_csv.exists()
    with open(rollup_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    # 2) 실행 스냅샷 CSV — 옵션 (기본 생성)
    if not args.no_run_snapshot:
        snap_csv = OUT_TABLES / f"build_index_perf_{run_ts}.csv"
        with open(snap_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"Saved perf snapshot : {snap_csv}")

    print(f"Saved perf roll-up  : {rollup_csv}")

if __name__ == "__main__":
    main()

