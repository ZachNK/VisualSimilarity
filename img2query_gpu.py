
# img2query_gpu.py (WSL/Windows 겸용, GPU 가속 + 설정파일 분리 최종판)
# ---------------------------------------------------------------------------
# 기능 요약
# - data/corpora/<dataset>/images 아래의 모든 이미지를 기반으로
#   회전/밝기/크기(스케일) 증강 쿼리 이미지를 data/corpora/<dataset>/<out-subdir>에 생성
# - "크기변형": 원점(0,0) 기준으로 s∈{0.25,0.50,0.75} 배 축소 → 원본 크기의 캔버스에 좌상단(0,0)으로 붙여넣기
# - 기본 파라미터(angles/brights/scales/quality)는 설정파일로 분리 (YAML 또는 JSON)
# - 우선순위: CLI 인자 > 설정파일(데이터셋별 override) > 설정파일(global) > 내장 기본값
# - 진행상황은 tqdm로 콘솔 시각화
# - 이미 생성된 출력이 있으면 스킵, --overwrite로 강제 재생성
# - GPU 가속: --device auto(default)/cuda/cpu, auto는 CUDA 사용 가능 시 자동 선택
#
# 권장 설정파일 경로(자동 탐색 순서):
#   1) --config로 지정한 파일
#   2) BASE/config/images2queries.yaml
#   3) BASE/config/images2queries.yml
#   4) BASE/config/images2queries.json
#
# 샘플 설정파일 (YAML):
# ---
# global:
#   angles:  [30,45,60,90,120,135,150,180,210,225,240,270,300,315,330]
#   brights: [0.25,0.5,0.75,1.25,1.5,1.75]
#   scales:  [0.25,0.5,0.75]     # ★ 추가
#   quality: 95
# datasets:
#   visdrone:
#     angles: [0,90,180,270]
#   union:
#     brights: [0.8,1.0,1.2]
# ---------------------------------------------------------------------------

### ================================ 사용방법 ================================

# corpora에 데이터셋이 준비된 후 실행하는 것

# 1) 전체 corpora 일괄 처리 (설정파일 자동 탐색)
# python img2query_gpu.py --all --device cuda

# 2) 특정 데이터셋만 (예: union)
# python img2query_gpu.py --dataset union --device cuda --overwrite

# 3) 설정파일 직접 지정
# python img2query_gpu.py --all --config "/mnt/d/KNK/_KSNU/_Projects/dino_test/config/images2queries.yaml"

# 4) 일회성으로 설정값 덮어쓰기(설정파일보다 우선)
# python img2query_gpu.py --dataset aihub --angles 0 90 180 270 --brights 0.8 1.0 1.2 --scales 0.25 0.5 0.75 --quality 92

# 5) 출력 하위폴더를 쿼리 전용으로 분리
# python img2query_gpu.py --dataset visdrone --out-subdir queries_scale

# 6) 기존 출력 무시하고 재생성
# python img2query_gpu.py --dataset visdrone --overwrite
### ================================ 사용방법 ================================

import os
import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

# 선택: YAML 지원 (없으면 JSON만 사용)
try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    yaml = None
    _HAVE_YAML = False

# ---------------- 경로 해석 ----------------

def resolve_base() -> Path:
    """DINO_BASE 환경변수 > Windows 경로 > WSL 경로 순으로 프로젝트 루트 탐색."""
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
        "Project base not found. Set DINO_BASE env var or check drive mount."
    )

BASE = resolve_base()
DATA = BASE / "data"
CORPORA = DATA / "corpora"

# ---------------- 기본값 ----------------

DEFAULT_ANGLES  = [30,45,60,90,120,135,150,180,210,225,240,270,300,315,330]
DEFAULT_BRIGHTS = [0.25,0.50,0.75,1.25,1.50,1.75]
DEFAULT_SCALES  = [0.25,0.50,0.75]   # ★ 추가
DEFAULT_QUALITY = 95

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------------- 설정 로더 ----------------

def _candidate_config_paths(explicit: str | None) -> List[Path]:
    cands: List[Path] = []
    if explicit:
        cands.append(Path(explicit))
    cands.extend([
        BASE / "config" / "images2queries.yaml",
        BASE / "config" / "images2queries.yml",
        BASE / "config" / "images2queries.json",
    ])
    # 중복 제거 + 존재하는 경로만
    uniq: List[Path] = []
    seen = set()
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            uniq.append(p)
    return uniq


def load_config(explicit_path: str | None) -> Dict[str, Any]:
    """설정파일을 로드(YAML/JSON). 없으면 빈 dict 반환."""
    for p in _candidate_config_paths(explicit_path):
        try:
            if p.suffix.lower() in {".yaml", ".yml"}:
                if not _HAVE_YAML:
                    print(f"[WARN] YAML not available. Install pyyaml or use JSON. Skipping {p}\n")
                    continue
                with p.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)  # type: ignore
                if isinstance(cfg, dict):
                    print(f"[INFO] Loaded config: {p}\n")
                    return cfg
            elif p.suffix.lower() == ".json":
                with p.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict):
                    print(f"[INFO] Loaded config: {p}\n")
                    return cfg
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}\n")
    # not found or failed
    print("[INFO] No valid config found. Using built-in defaults.\n")
    return {}


def _list_or_default(cfg: Dict[str, Any], key: str, default: List[Any]) -> List[Any]:
    v = cfg.get(key)
    if isinstance(v, list) and v:
        return v
    return default


def _quality_or_default(cfg: Dict[str, Any], key: str, default: int) -> int:
    v = cfg.get(key)
    if isinstance(v, int) and 1 <= v <= 100:
        return v
    return default


def resolve_params(dataset_name: str, args, cfg: Dict[str, Any]) -> Tuple[List[int], List[float], List[float], int]:
    """데이터셋별 유효 파라미터(angles, brights, scales, quality)를 결정."""
    # 1) 글로벌 설정
    gconf = cfg.get("global", {}) if isinstance(cfg.get("global"), dict) else {}
    angles  = _list_or_default(gconf, "angles",  DEFAULT_ANGLES)
    brights = _list_or_default(gconf, "brights", DEFAULT_BRIGHTS)
    scales  = _list_or_default(gconf, "scales",  DEFAULT_SCALES)   # ★ 추가
    quality = _quality_or_default(gconf, "quality", DEFAULT_QUALITY)

    # 2) 데이터셋별 override
    dconfs = cfg.get("datasets", {}) if isinstance(cfg.get("datasets"), dict) else {}
    dconf = dconfs.get(dataset_name, {}) if isinstance(dconfs.get(dataset_name), dict) else {}
    if dconf:
        angles  = _list_or_default(dconf, "angles",  angles)
        brights = _list_or_default(dconf, "brights", brights)
        scales  = _list_or_default(dconf, "scales",  scales)       # ★ 추가
        quality = _quality_or_default(dconf, "quality", quality)

    # 3) CLI override
    if args.angles is not None and len(args.angles) > 0:
        angles = args.angles
    if args.brights is not None and len(args.brights) > 0:
        brights = args.brights
    if args.scales is not None and len(args.scales) > 0:          # ★ 추가
        scales = args.scales
    if args.quality is not None:
        quality = args.quality

    # 정수/실수 캐스팅 보정
    angles  = [int(x) for x in angles]
    brights = [float(x) for x in brights]
    scales  = [float(x) for x in scales]

    return angles, brights, scales, int(quality)

# ---------------- 유틸 ----------------

def list_image_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in EXTS:
        files.extend(root.rglob(f"*{ext}"))
    files = sorted(set(files))
    return files


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def targets_for(img_path: Path, images_root: Path, queries_root: Path,
                angles: List[int], brights: List[float], scales: List[float]) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    출력 대상 파일 경로를 (rot_outs, bright_outs, scale_outs)로 반환.
    파일명 규칙:
      - 회전  : *_rot{deg:03d}.jpg
      - 밝기  : *_bright{pct:03d}.jpg  (pct=밝기*100 반올림)
      - 크기  : *_scale{pct:03d}.jpg   (pct=스케일*100 반올림)  ex) 25%→025
    """
    rel = img_path.relative_to(images_root)
    stem = rel.stem
    parent_rel = rel.parent
    out_dir = queries_root / parent_rel

    rot_outs, bright_outs, scale_outs = [], [], []

    for a in angles:
        rot_outs.append(out_dir / f"{stem}_rot{a:03d}.jpg")

    for b in brights:
        tag = f"{int(round(b*100)):03d}"
        bright_outs.append(out_dir / f"{stem}_bright{tag}.jpg")

    for s in scales:
        tag = f"{int(round(s*100)):03d}"
        scale_outs.append(out_dir / f"{stem}_scale{tag}.jpg")

    return rot_outs, bright_outs, scale_outs


# ---------------- GPU 유틸 (PyTorch + Kornia) ----------------

def _try_import_gpu():
    try:
        import torch
        import kornia
        import torch.nn.functional as F  # noqa
        return torch, kornia
    except Exception as e:
        raise RuntimeError(
            "GPU 경로를 사용하려면 'torch'와 'kornia'가 필요합니다. "
            "pip install torch torchvision kornia 로 설치하세요."
        ) from e

def _compute_expanded_size(w: int, h: int, angle_deg: float) -> Tuple[int, int]:
    th = math.radians(angle_deg % 360)
    cos_t, sin_t = abs(math.cos(th)), abs(math.sin(th))
    new_w = int(math.ceil(w * cos_t + h * sin_t))
    new_h = int(math.ceil(w * sin_t + h * cos_t))
    return new_w, new_h

def _pil_to_tensor(img: Image.Image, torch):
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # C,H,W in [0,1]
    return t

def _tensor_to_pil(t, torch) -> Image.Image:
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()  # H,W,C
    return Image.fromarray(arr, mode="RGB")

def _rotate_gpu_tensor(img_t, angle_deg: float, torch, kornia, mode: str = "bilinear"):
    # img_t: (B,3,H,W)
    B, C, H, W = img_t.shape
    device = img_t.device
    dtype = img_t.dtype

    # (B,2) 중심
    center = torch.tensor([[W/2.0, H/2.0]], dtype=dtype, device=device).repeat(B, 1)

    # (B,) 각도
    angle = torch.tensor([float(angle_deg)], dtype=dtype, device=device).repeat(B)

    # (B,2) 스케일
    scale = torch.ones(B, 2, dtype=dtype, device=device)

    # 회전 행렬
    M = kornia.geometry.transform.get_rotation_matrix2d(center, angle, scale)  # (B,2,3)

    # expand=True 효과: 새 캔버스 크기와 translation 보정
    new_w, new_h = _compute_expanded_size(W, H, angle_deg)
    M[:, 0, 2] += (new_w - W) / 2.0
    M[:, 1, 2] += (new_h - H) / 2.0

    out = kornia.geometry.transform.warp_affine(
        img_t, M, dsize=(new_h, new_w), mode=mode, padding_mode="zeros", align_corners=False
    )
    return out

def _scale_gpu_canvas(img_t, scale: float, torch):
    """
    원점(0,0) 기준 스케일 축소 후, 원본 크기의 검은 캔버스에 좌상단(0,0)으로 붙여넣기.
    img_t: (B,3,H,W) in [0,1]
    반환: (B,3,H,W)
    """
    B, C, H, W = img_t.shape
    new_h = max(1, int(round(H * float(scale))))
    new_w = max(1, int(round(W * float(scale))))
    # bilinear resize
    scaled = torch.nn.functional.interpolate(img_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    # paste into blank canvas
    canvas = torch.zeros((B, C, H, W), dtype=img_t.dtype, device=img_t.device)
    canvas[:, :, :new_h, :new_w] = scaled
    return canvas

def process_one_image_gpu(img_path: Path, images_root: Path, queries_root: Path,
                          angles: List[int], brights: List[float], scales: List[float],
                          quality: int, overwrite: bool,
                          device: str = "cuda") -> int:
    torch, kornia = _try_import_gpu()
    rot_outs, bright_outs, scale_outs = targets_for(img_path, images_root, queries_root, angles, brights, scales)
    outs_all = rot_outs + bright_outs + scale_outs

    if not overwrite and all(o.exists() for o in outs_all):
        return 0

    if outs_all:
        ensure_dir(outs_all[0].parent)

    # 로드 → 텐서 → 디바이스
    with Image.open(img_path) as img_raw:
        img = ImageOps.exif_transpose(img_raw).convert("RGB")
    t = _pil_to_tensor(img, torch).unsqueeze(0)  # (1,3,H,W)
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    t = t.to(dev, non_blocking=True)

    saved = 0

    # 1) 회전 저장 (GPU)
    for a, outp in zip(angles, rot_outs):
        if not overwrite and outp.exists():
            continue
        with torch.no_grad():
            rot = _rotate_gpu_tensor(t, float(a), torch, kornia, mode="bilinear").squeeze(0)  # (3,H,W)
        _tensor_to_pil(rot, torch).save(outp, quality=quality)
        saved += 1

    # 2) 밝기 저장 (GPU: 배율 곱)
    for b, outp in zip(brights, bright_outs):
        if not overwrite and outp.exists():
            continue
        with torch.no_grad():
            bt = torch.clamp(t * float(b), 0.0, 1.0).squeeze(0)  # (3,H,W)
        _tensor_to_pil(bt, torch).save(outp, quality=quality)
        saved += 1

    # 3) 크기(스케일) 저장 (GPU)
    for s, outp in zip(scales, scale_outs):
        if not overwrite and outp.exists():
            continue
        with torch.no_grad():
            st = _scale_gpu_canvas(t, float(s), torch).squeeze(0)  # (3,H,W)
        _tensor_to_pil(st, torch).save(outp, quality=quality)
        saved += 1

    return saved


# ---------------- 기존 CPU 경로 (PIL) ----------------

def process_one_image_cpu(img_path: Path, images_root: Path, queries_root: Path,
                          angles: List[int], brights: List[float], scales: List[float],
                          quality: int = 95, overwrite: bool = False) -> int:
    rot_outs, bright_outs, scale_outs = targets_for(img_path, images_root, queries_root, angles, brights, scales)
    outs_all = rot_outs + bright_outs + scale_outs

    if not overwrite and all(o.exists() for o in outs_all):
        return 0

    if outs_all:
        ensure_dir(outs_all[0].parent)

    with Image.open(img_path) as img_raw:
        img = ImageOps.exif_transpose(img_raw).convert("RGB")
        W, H = img.size

        # 회전 저장
        for a, outp in zip(angles, rot_outs):
            if overwrite or not outp.exists():
                img.rotate(a, resample=Image.BICUBIC, expand=True).save(outp, quality=quality)

        # 밝기 저장
        enh = ImageEnhance.Brightness(img)
        for b, outp in zip(brights, bright_outs):
            if overwrite or not outp.exists():
                enh.enhance(b).save(outp, quality=quality)

        # 크기(스케일) 저장: 원점 기준 축소 후 원본 크기 캔버스에 (0,0) 붙여넣기
        for s, outp in zip(scales, scale_outs):
            if overwrite or not outp.exists():
                new_w = max(1, int(round(W * float(s))))
                new_h = max(1, int(round(H * float(s))))
                scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (W, H))
                canvas.paste(scaled, (0, 0))
                canvas.save(outp, quality=quality)

    return sum(1 for o in outs_all if o.exists())


# ---------------- 데이터셋 실행 ----------------

def run_dataset(dataset_dir: Path, angles: List[int], brights: List[float], scales: List[float],
                quality: int, overwrite: bool, device: str, out_subdir: str) -> Dict[str, Any]:
    images_dir  = dataset_dir / "images"
    queries_dir = dataset_dir / out_subdir
    ensure_dir(queries_dir)

    if not images_dir.exists():
        return {"dataset": dataset_dir.name, "ok": False, "reason": f"not found: {images_dir}"}

    files = list_image_files(images_dir)
    if not files:
        return {"dataset": dataset_dir.name, "ok": False, "reason": f"no images under {images_dir}"}

    # 디바이스 결정
    dev = device
    if dev == "auto":
        try:
            import torch  # noqa
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    created_total = 0
    errors = 0
    desc = (
        f"[{dataset_dir.name}] images→{out_subdir} | "
        f"ang={len(angles)} bri={len(brights)} scl={len(scales)} q={quality} | {dev.upper()}\n"
    )
    with tqdm(files, desc=desc, unit="img") as bar:
        for f in bar:
            try:
                if dev == "cuda":
                    created = process_one_image_gpu(
                        f, images_dir, queries_dir,
                        angles=angles, brights=brights, scales=scales,
                        quality=quality, overwrite=overwrite,
                        device="cuda"
                    )
                else:
                    created = process_one_image_cpu(
                        f, images_dir, queries_dir,
                        angles=angles, brights=brights, scales=scales,
                        quality=quality, overwrite=overwrite
                    )
                created_total += created
            except Exception as e:
                errors += 1
                bar.write(f"[WARN] {f}: {e}\n")

    return {
        "dataset": dataset_dir.name,
        "ok": True,
        "images": len(files),
        "created": created_total,
        "errors": errors,
        "queries_dir": str(queries_dir),
        "angles": angles,
        "brights": brights,
        "scales": scales,
        "quality": quality,
        "device": dev,
    }

# -------------- 단일 이미지 실행 --------------

def find_dataset_by_image(img_path: Path):
    """img_path가 어느 data/corpora/<ds>/images 아래에 있는지 찾아서 (ds_dir, images_dir, queries_dir) 반환."""
    ip = img_path.resolve()
    for ds in CORPORA.iterdir():
        if not ds.is_dir():
            continue
        images_dir = (ds / "images").resolve()
        try:
            ip.relative_to(images_dir)
            return ds, images_dir, (ds / "queries")
        except Exception:
            continue
    return None

def run_single_file(img_path: Path, angles: List[int], brights: List[float], scales: List[float],
                    quality: int, overwrite: bool, device: str, out_subdir: str) -> Dict[str, Any]:
    found = find_dataset_by_image(img_path)
    if not found:
        raise ValueError(
            f"Given file is not under any 'data/corpora/<dataset>/images': {img_path}"
        )
    ds_dir, images_dir, _legacy_queries = found
    queries_dir = ds_dir / out_subdir
    ensure_dir(queries_dir)

    # 디바이스 결정
    dev = device
    if dev == "auto":
        try:
            import torch  # noqa
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    # 실제 처리
    if dev == "cuda":
        created = process_one_image_gpu(
            img_path, images_dir, queries_dir,
            angles=angles, brights=brights, scales=scales, quality=quality,
            overwrite=overwrite, device="cuda"
        )
    else:
        created = process_one_image_cpu(
            img_path, images_dir, queries_dir,
            angles=angles, brights=brights, scales=scales, quality=quality,
            overwrite=overwrite
        )

    return {
        "dataset": ds_dir.name, "ok": True, "images": 1, "created": created, "errors": 0,
        "queries_dir": str(queries_dir),
        "angles": angles, "brights": brights, "scales": scales,
        "quality": quality, "device": dev, "file": str(img_path)
    }


# ---------------- 엔트리포인트 ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate query images (rotation/brightness/scale) with config file support (GPU-accelerated)."
    )

    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--dataset", type=str, help="Single dataset name under data/corpora (e.g., visdrone, aihub, soda-a, union)")
    g.add_argument("--all", action="store_true", help="Process all datasets under data/corpora")

    ap.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config file (overrides auto-discovery)")

    # CLI override (있으면 설정파일보다 우선)
    ap.add_argument("--angles",  type=int,   nargs="*", default=None, help="Rotation angles in degrees (override)")
    ap.add_argument("--brights", type=float, nargs="*", default=None, help="Brightness factors (override)")
    ap.add_argument("--scales",  type=float, nargs="*", default=None, help="Scale factors (override), e.g., 0.25 0.5 0.75")  # ★ 추가
    ap.add_argument("--quality", type=int, default=None, help="JPEG quality 1~100 (override)")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # 디바이스 선택: auto/cuda/cpu
    ap.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto",
                    help="Processing device. 'auto' uses CUDA if available.")

    # 출력 하위 폴더 선택 (기본: queries) — 예: queries_scale
    ap.add_argument("--out-subdir", type=str, default="queries",
                    help="Subfolder under each dataset to write queries (default: 'queries')")

    # 단일 이미지 처리
    ap.add_argument("--file", type=str,
                    help="Process a single image file under data/corpora/<dataset>/images")

    args = ap.parse_args()
    cfg = load_config(args.config)

    # ===== 옵션 충돌/부족 검증 =====
    if not args.file and not (args.all or args.dataset):
        ap.error("Specify --all or --dataset, or use --file to process a single image.")
    if args.all and args.file:
        ap.error("--all cannot be used with --file (use either all/dataset or file).")

    # ===== ✅ 단일 이미지 모드 =====
    if args.file:
        p = Path(args.file)
        if not p.is_absolute() and args.dataset:
            p = (CORPORA / args.dataset / "images" / p).resolve()

        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            sys.exit(2)

        found = find_dataset_by_image(p)
        if not found:
            print(f"[ERROR] File must reside under data/corpora/<dataset>/images: {p}")
            sys.exit(2)
        ds_dir, _, _ = found

        angles, brights, scales, quality = resolve_params(ds_dir.name, args, cfg)
        res = run_single_file(p, angles, brights, scales, quality, args.overwrite, args.device, args.out_subdir)

        print("\n=== Summary (single file) ===")
        print(
            f"- {res['dataset']}: file={res['file']}\n"
            f"  created(files)={res['created']}, errors={res['errors']}\n"
            f"  → {args.out_subdir}: {res['queries_dir']} | "
            f"angles={len(res['angles'])} brights={len(res['brights'])} scales={len(res['scales'])} "
            f"quality={res['quality']} | device={res['device']}"
        )
        return  # 단일 파일 모드 종료

    # ===== 여러 데이터셋 처리 경로 =====
    if args.all:
        datasets = [p for p in CORPORA.iterdir() if p.is_dir()]
    else:
        datasets = [CORPORA / args.dataset]

    start = datetime.now()

    results = []
    for ds in sorted(datasets, key=lambda p: p.name.lower()):
        angles, brights, scales, quality = resolve_params(ds.name, args, cfg)
        res = run_dataset(ds, angles, brights, scales, quality, args.overwrite, args.device, args.out_subdir)
        results.append(res)

    # 요약 출력
    print("\n=== Summary ===")
    for r in results:
        if not r.get("ok"):
            print(f"- {r['dataset']}: FAILED ({r.get('reason')})")
        else:
            print(
                f"- {r['dataset']}: images={r['images']}, created(files)={r['created']}, errors={r['errors']}\n"
                f"  → {args.out_subdir}: {r['queries_dir']} | "
                f"angles={len(r['angles'])} brights={len(r['brights'])} scales={len(r['scales'])} "
                f"quality={r['quality']} | device={r['device']}"
            )

    dur = (datetime.now() - start).total_seconds()
    print(f"\nDone in {dur:.1f}s.")


if __name__ == "__main__":
    main()
