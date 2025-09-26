# images2queries.py (WSL/Windows 겸용, corpora 전용 + 설정파일 분리 최종판)
# ---------------------------------------------------------------------------
# 기능 요약
# - data/corpora/<dataset>/images 아래의 모든 이미지를 기반으로
#   회전/밝기 증강 쿼리 이미지를 data/corpora/<dataset>/queries에 생성
# - 기본 파라미터(angles/brights/quality)는 설정파일로 분리 (YAML 또는 JSON)
# - 우선순위: CLI 인자 > 설정파일(데이터셋별 override) > 설정파일(global) > 내장 기본값
# - 진행상황은 tqdm로 WSL/Windows 콘솔에 시각화
# - 이미 생성된 출력이 있으면 기본적으로 스킵, --overwrite로 강제 재생성
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
#   angles: [30,45,60,90,120,135,150,180,210,225,240,270,300,315,330]
#   brights: [0.25,0.5,0.75,1.25,1.5,1.75]
#   quality: 95
# datasets:
#   visdrone:
#     angles: [0,90,180,270]
#   union:
#     brights: [0.8,1.0,1.2]
#
# 샘플 설정파일 (JSON):
# {
#   "global": {
#     "angles": [30,45,60,90,120,135,150,180,210,225,240,270,300,315,330],
#     "brights": [0.25,0.5,0.75,1.25,1.5,1.75],
#     "quality": 95
#   },
#   "datasets": {
#     "visdrone": { "angles": [0,90,180,270] },
#     "union": { "brights": [0.8,1.0,1.2] }
#   }
# }
# ---------------------------------------------------------------------------

### ================================ 사용방법 ================================ 
# 1) 전체 corpora 일괄 처리 (설정파일 자동 탐색)
# python images2queries_with_config.py --all

# 2) 특정 데이터셋만 (예: union)
# python images2queries_with_config.py --dataset union

# 3) 설정파일 직접 지정
# python images2queries_with_config.py --all --config "/mnt/d/KNK/_KSNU/_Projects/dino_test/config/images2queries.yaml"

# 4) 일회성으로 설정값 덮어쓰기(설정파일보다 우선)
# python images2queries_with_config.py --dataset aihub --angles 0 90 180 270 --brights 0.8 1.0 1.2 --quality 92

# 5) 기존 출력 무시하고 재생성
# python images2queries_with_config.py --dataset visdrone --overwrite
### ================================ 사용방법 ================================ 


import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

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
DEFAULT_QUALITY = 95

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

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
                    print(f"[WARN] YAML not available. Install pyyaml or use JSON. Skipping {p}")
                    continue
                with p.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)  # type: ignore
                if isinstance(cfg, dict):
                    print(f"[INFO] Loaded config: {p}")
                    return cfg
            elif p.suffix.lower() == ".json":
                with p.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict):
                    print(f"[INFO] Loaded config: {p}")
                    return cfg
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    # not found or failed
    print("[INFO] No valid config found. Using built-in defaults.")
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


def resolve_params(dataset_name: str, args, cfg: Dict[str, Any]) -> Tuple[List[int], List[float], int]:
    """데이터셋별 유효 파라미터(angles, brights, quality)를 결정."""
    # 1) 글로벌 설정
    gconf = cfg.get("global", {}) if isinstance(cfg.get("global"), dict) else {}
    angles = _list_or_default(gconf, "angles", DEFAULT_ANGLES)
    brights = _list_or_default(gconf, "brights", DEFAULT_BRIGHTS)
    quality = _quality_or_default(gconf, "quality", DEFAULT_QUALITY)

    # 2) 데이터셋별 override
    dconfs = cfg.get("datasets", {}) if isinstance(cfg.get("datasets"), dict) else {}
    dconf = dconfs.get(dataset_name, {}) if isinstance(dconfs.get(dataset_name), dict) else {}
    if dconf:
        angles = _list_or_default(dconf, "angles", angles)
        brights = _list_or_default(dconf, "brights", brights)
        quality = _quality_or_default(dconf, "quality", quality)

    # 3) CLI override
    if args.angles is not None and len(args.angles) > 0:
        angles = args.angles
    if args.brights is not None and len(args.brights) > 0:
        brights = args.brights
    if args.quality is not None:
        quality = args.quality

    # 정수/실수 캐스팅 보정
    angles = [int(x) for x in angles]
    brights = [float(x) for x in brights]

    return angles, brights, int(quality)

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
                angles: List[int], brights: List[float]) -> List[Path]:
    # images_root 하위 상대경로 유지 ⇒ queries_root에 동일 구조
    rel = img_path.relative_to(images_root)
    stem = rel.stem
    parent_rel = rel.parent
    out_dir = queries_root / parent_rel

    outs: List[Path] = []
    for a in angles:
        outs.append(out_dir / f"{stem}_rot{a:03d}.jpg")
    for b in brights:
        tag = f"{int(round(b*100)):03d}"
        outs.append(out_dir / f"{stem}_bright{tag}.jpg")
    return outs


def process_one_image(img_path: Path, images_root: Path, queries_root: Path,
                      angles: List[int], brights: List[float],
                      quality: int = 95, overwrite: bool = False) -> int:
    outs = targets_for(img_path, images_root, queries_root, angles, brights)

    if not overwrite and all(o.exists() for o in outs):
        return 0

    ensure_dir(outs[0].parent)

    with Image.open(img_path) as img_raw:
        img = ImageOps.exif_transpose(img_raw).convert("RGB")

        # 회전 저장
        for a, outp in zip(angles, outs[:len(angles)]):
            if overwrite or not outp.exists():
                img.rotate(a, resample=Image.BICUBIC, expand=True).save(outp, quality=quality)

        # 밝기 저장
        enh = ImageEnhance.Brightness(img)
        for b, outp in zip(brights, outs[len(angles):]):
            if overwrite or not outp.exists():
                enh.enhance(b).save(outp, quality=quality)

    return sum(1 for o in outs if o.exists())


def run_dataset(dataset_dir: Path, angles: List[int], brights: List[float],
                quality: int, overwrite: bool) -> Dict[str, Any]:
    images_dir = dataset_dir / "images"
    queries_dir = dataset_dir / "queries"
    ensure_dir(queries_dir)

    if not images_dir.exists():
        return {"dataset": dataset_dir.name, "ok": False, "reason": f"not found: {images_dir}"}

    files = list_image_files(images_dir)
    if not files:
        return {"dataset": dataset_dir.name, "ok": False, "reason": f"no images under {images_dir}"}

    created_total = 0
    errors = 0
    desc = f"[{dataset_dir.name}] images→queries | ang={len(angles)} bri={len(brights)} q={quality}"
    with tqdm(files, desc=desc, unit="img") as bar:
        for f in bar:
            try:
                created = process_one_image(
                    f, images_dir, queries_dir,
                    angles=angles, brights=brights, quality=quality, overwrite=overwrite
                )
                created_total += created
            except Exception as e:
                errors += 1
                bar.write(f"[WARN] {f}: {e}")

    return {
        "dataset": dataset_dir.name,
        "ok": True,
        "images": len(files),
        "created": created_total,
        "errors": errors,
        "queries_dir": str(queries_dir),
        "angles": angles,
        "brights": brights,
        "quality": quality,
    }

# ---------------- 엔트리포인트 ----------------

def main():
    ap = argparse.ArgumentParser(description="Generate query images (rotations/brightness) for corpora datasets with config file support.")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", type=str, help="Single dataset name under data/corpora (e.g., visdrone, aihub, soda-a, union)")
    g.add_argument("--all", action="store_true", help="Process all datasets under data/corpora")

    ap.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config file (overrides auto-discovery)")

    # CLI override (있으면 설정파일보다 우선)
    ap.add_argument("--angles", type=int, nargs="*", default=None, help="Rotation angles in degrees (override)")
    ap.add_argument("--brights", type=float, nargs="*", default=None, help="Brightness factors (override)")
    ap.add_argument("--quality", type=int, default=None, help="JPEG quality 1~100 (override)")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.all:
        datasets = [p for p in CORPORA.iterdir() if p.is_dir()]
    else:
        datasets = [CORPORA / args.dataset]

    start = datetime.now()

    results = []
    for ds in sorted(datasets, key=lambda p: p.name.lower()):
        angles, brights, quality = resolve_params(ds.name, args, cfg)
        res = run_dataset(ds, angles, brights, quality, args.overwrite)
        results.append(res)

    # 요약 출력
    print("\n=== Summary ===")
    for r in results:
        if not r.get("ok"):
            print(f"- {r['dataset']}: FAILED ({r.get('reason')})")
        else:
            print(
                f"- {r['dataset']}: images={r['images']}, created(files)={r['created']}, errors={r['errors']}\n"
                f"  → queries: {r['queries_dir']} | angles={len(r['angles'])} brights={len(r['brights'])} quality={r['quality']}"
            )

    dur = (datetime.now() - start).total_seconds()
    print(f"\nDone in {dur:.1f}s.")


if __name__ == "__main__":
    main()
