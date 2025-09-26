# scripts/normalize_dataset.py
### ================================ 사용방법 ================================ 
### 1. WSL에서, DINO_BASE 먼저:
# export DINO_BASE=/mnt/d/KNK/_KSNU/_Projects/dino_test
# cd "$DINO_BASE/scripts"

### 2-1. VisDrone 정규화
# python normalize_dataset.py --dataset visdrone --mode link --verify
## (필요시) 소스 직접 지정
## python normalize_dataset.py --dataset visdrone --src "/mnt/d/KNK/_Datasets/VisDrone" --mode link --verify

### 2-2. SODA-A 정규화
# python normalize_dataset.py --dataset sodaa --mode link --verify

### 2-3. AI-Hub 정규화 (TIF-JPG 변환 권장)
## 권장: tif → jpg 변환 저장(파이프라인 호환)
## python normalize_dataset.py --dataset aihub --convert tiff2jpg --jpeg-quality 92 --verify
## 원본 확장자 유지(정말 필요한 경우만)
## python normalize_dataset.py --dataset aihub --convert keep --mode link --verify

### 2-4. 통합 정규화
# python normalize_dataset.py --dataset union --sources visdrone,sodaa,aihub --mode link --verify --skip-existing

### (3. 무결성 체크)
# 개수 세기
# find "$DINO_BASE/data/corpora/visdrone/images" -maxdepth 1 -type f | wc -l
# find "$DINO_BASE/data/corpora/sodaa/images"    -maxdepth 1 -type f | wc -l
# find "$DINO_BASE/data/corpora/aihub/images"    -maxdepth 1 -type f | wc -l
# find "$DINO_BASE/data/corpora/union/images"    -maxdepth 1 -type f | wc -l
# 
# # 무결성 샘플 점검(랜덤 50장)
# python - <<'PY'
# import random,cv2,glob,sys
# from pathlib import Path
# d = Path("/mnt/d/KNK/_KSNU/_Projects/dino_test/data/corpora/union/images")
# fs = glob.glob(str(d/"*.jpg"))
# bad = []
# for p in random.sample(fs, min(50,len(fs))):
#     im = cv2.imread(p, 0)
#     if im is None or im.size==0: bad.append(p)
# print("sample_bad:", len(bad), "\n", "\n".join(bad[:5]))
# PY

### ================================ 사용방법 ================================ 

# 한 스크립트로 visdrone / sodaa / aihub / union 모두 정규화 (하드링크/복사/변환/검증/진행바/매니페스트)
import os, sys, cv2, csv, shutil, hashlib, argparse
from pathlib import Path
from tqdm import tqdm

# ------------------------- 공통 유틸 -------------------------
def resolve_base() -> Path:
    env = os.getenv("DINO_BASE")
    if env and Path(env).exists():
        return Path(env)
    for p in [Path(r"D:/KNK/_KSNU/_Projects/dino_test"),
              Path("/mnt/d/KNK/_KSNU/_Projects/dino_test")]:
        if p.exists():
            return p
    raise FileNotFoundError("DINO_BASE를 찾지 못했습니다. 환경변수 DINO_BASE를 설정하세요.")

def safe_name(*parts: str) -> str:
    s = "_".join(parts)
    return s.replace("/", "_").replace("\\", "_").replace(" ", "")

def link_or_copy(src: Path, dst: Path, mode: str = "link"):
    """mode: link(하드링크, 권장) | copy | symlink"""
    if mode == "link":
        try:
            os.link(src, dst)
            return "link"
        except Exception:
            shutil.copy2(src, dst)  # 폴백
            return "copy"
    elif mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    else:
        dst.symlink_to(src)
        return "symlink"

def verify_image(p: Path) -> bool:
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (im is not None) and (im.size > 0)

def write_manifest(rows, out_csv: Path):
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ------------------------- 데이터셋별 소스 탐색 -------------------------
def find_visdrone_src() -> Path:
    cands = [Path("/mnt/d/KNK/_Datasets/VisDrone"), Path(r"D:/KNK/_Datasets/VisDrone")]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("VisDrone 원본 경로를 찾지 못했습니다. --src 로 지정하세요.")

def find_sodaa_src() -> Path:
    cands = [Path("/mnt/d/KNK/_Datasets/SODA_A"), Path(r"D:/KNK/_Datasets/SODA_A")]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("SODA_A 원본 경로를 찾지 못했습니다. --src 로 지정하세요.")

def find_aihub_src() -> Path:
    # 둘 다 케이스 지원: 바로 안쪽 or 한 단계 바깥
    cands = [
        Path("/mnt/d/KNK/_Datasets/AiHub/01-1.정식개방데이터"),
        Path(r"D:/KNK/_Datasets/AiHub/01-1.정식개방데이터"),
        Path("/mnt/d/KNK/_Datasets/AiHub"),
        Path(r"D:/KNK/_Datasets/AiHub"),
    ]
    for c in cands:
        if c.exists():
            # 바깥 루트를 준 경우 자동으로 안쪽으로 진입
            inner = c / "01-1.정식개방데이터"
            return inner if inner.exists() else c
    raise FileNotFoundError("AiHub 원본 경로를 찾지 못했습니다. --src 로 지정하세요.")

# ------------------------- VisDrone 정규화 -------------------------
def normalize_visdrone(dst_root: Path, src: Path | None, mode: str, verify: bool, max_files: int | None):
    SRC = src if src else find_visdrone_src()
    DST = dst_root / "corpora" / "visdrone" / "images"
    IDX = dst_root / "corpora" / "visdrone" / "index"
    DST.mkdir(parents=True, exist_ok=True)
    IDX.mkdir(parents=True, exist_ok=True)

    files = list(SRC.rglob("sequences/*/*.jpg"))
    if not files:
        print(f"[ERR] VisDrone sequences/*.jpg 를 찾지 못했습니다: {SRC}")
        sys.exit(1)
    if max_files:
        files = files[:max_files]
    print(f"[VisDrone] found {len(files)} frames")

    manifest = []
    ok = dup = bad = 0
    for srcf in tqdm(files, desc="VisDrone → corpora/visdrone/images", unit="img", ncols=100):
        parts = srcf.parts
        try:
            i = parts.index("sequences")
            split = parts[i-1] if i-1 >= 0 else "vis"
            seq   = parts[i+1]
            frame = parts[-1]
        except ValueError:
            split = parts[-4] if len(parts) > 3 else "vis"
            seq   = parts[-2] if len(parts) > 1 else "seq"
            frame = parts[-1]
        name = safe_name(split.lower(), seq, frame)
        dst = DST / name
        if dst.exists():
            h = hashlib.md5(str(srcf).encode()).hexdigest()[:6]
            dst = DST / safe_name(split.lower(), seq, h, frame)
            dup += 1
        try:
            how = link_or_copy(srcf, dst, mode)
        except Exception as e:
            bad += 1
            continue
        if verify and not verify_image(dst):
            try: dst.unlink()
            except: pass
            bad += 1
            continue
        ok += 1
        manifest.append({"dataset": "visdrone", "src": str(srcf), "dst": str(dst), "how": how, "bytes": srcf.stat().st_size})

    write_manifest(manifest, IDX / "files_manifest.csv")
    print(f"[VisDrone] ok={ok}, dup_renamed={dup}, bad={bad}, dst={DST}")

# ------------------------- SODA-A 정규화 -------------------------
def normalize_sodaa(dst_root: Path, src: Path | None, mode: str, verify: bool, max_files: int | None):
    SRC = src if src else find_sodaa_src()
    IMG = SRC / "Images"
    if not IMG.exists():
        raise FileNotFoundError(f"SODA_A Images 폴더가 없습니다: {IMG}")

    DST = dst_root / "corpora" / "sodaa" / "images"
    IDX = dst_root / "corpora" / "sodaa" / "index"
    DST.mkdir(parents=True, exist_ok=True)
    IDX.mkdir(parents=True, exist_ok=True)

    files = sorted(IMG.glob("*.jpg"))
    if not files:
        print(f"[ERR] {IMG}에 *.jpg가 없습니다.")
        sys.exit(1)
    if max_files:
        files = files[:max_files]
    print(f"[SODA-A] found {len(files)} images")

    manifest = []
    ok = dup = bad = 0
    for srcf in tqdm(files, desc="SODA-A → corpora/sodaa/images", unit="img", ncols=100):
        name = safe_name("sodaa", srcf.name)  # 충돌 거의 없지만 prefix 부여
        dst = DST / name
        if dst.exists():
            h = hashlib.md5(str(srcf).encode()).hexdigest()[:6]
            dst = DST / safe_name("sodaa", h, srcf.name)
            dup += 1
        try:
            how = link_or_copy(srcf, dst, mode)
        except Exception:
            bad += 1
            continue
        if verify and not verify_image(dst):
            try: dst.unlink()
            except: pass
            bad += 1
            continue
        ok += 1
        manifest.append({"dataset": "sodaa", "src": str(srcf), "dst": str(dst), "how": how, "bytes": srcf.stat().st_size})

    write_manifest(manifest, IDX / "files_manifest.csv")
    print(f"[SODA-A] ok={ok}, dup_renamed={dup}, bad={bad}, dst={DST}")

# ------------------------- AI-Hub 정규화 -------------------------
def iter_aihub_tifs(root: Path):
    # root / (["01-1.정식개방데이터"]?) / Training|Validation / 01.원천데이터 / subset / *.tif(f)
    bases = []
    for cand in (root, root / "01-1.정식개방데이터"):
        t = cand / "Training" / "01.원천데이터"
        v = cand / "Validation" / "01.원천데이터"
        if t.exists(): bases.append(t)
        if v.exists(): bases.append(v)

    exts = ("*.tif", "*.TIF", "*.tiff", "*.TIFF")
    for base in bases:
        for subset in base.iterdir():
            if not subset.is_dir():
                continue
            for ext in exts:
                for f in subset.rglob(ext):
                    yield f

def normalize_aihub(dst_root: Path, src: Path | None, mode: str, verify: bool, max_files: int | None,
                    convert: bool, jpeg_quality: int):
    SRC = src if src else find_aihub_src()
    DST = dst_root / "corpora" / "aihub" / "images"
    IDX = dst_root / "corpora" / "aihub" / "index"
    DST.mkdir(parents=True, exist_ok=True)
    IDX.mkdir(parents=True, exist_ok=True)

    tifs = list(iter_aihub_tifs(SRC))
    if not tifs:
        print(f"[ERR] AiHub *.tif 원천데이터를 찾지 못했습니다: {SRC}")
        sys.exit(1)
    if max_files:
        tifs = tifs[:max_files]
    print(f"[AI-Hub] found {len(tifs)} tif frames (원천데이터)")

    manifest = []
    ok = dup = bad = 0
    for srcf in tqdm(tifs, desc="AI-Hub → corpora/aihub/images", unit="img", ncols=100):
        # 예) Training/01.원천데이터/TS_AP25_512픽셀/XXX.tif
        parts = srcf.parts
        # split = Training/Validation, subset = TS_* 폴더명
        split = next((p for p in parts if p in ("Training", "Validation")), "train")
        subset = srcf.parent.name
        stem = srcf.stem

        if convert:
            # tif → jpg 변환 저장
            name = safe_name("aihub", split.lower(), subset, stem) + ".jpg"
            dst = DST / name
            if dst.exists():
                h = hashlib.md5(str(srcf).encode()).hexdigest()[:6]
                dst = DST / (safe_name("aihub", split.lower(), subset, h, stem) + ".jpg")
                dup += 1
            try:
                img = cv2.imread(str(srcf), cv2.IMREAD_UNCHANGED)
                if img is None:
                    bad += 1
                    continue
                # 16bit/단채널도 JPG로 저장 가능하도록 변환
                if len(img.shape) == 2:  # gray
                    pass
                elif img.shape[2] == 4:  # RGBA → BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                # JPG 저장
                cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                how = "convert"
            except Exception:
                bad += 1
                continue
        else:
            # 확장자를 유지하면서 링크/복사 (파이프라인이 .jpg를 선호한다면 convert=True 권장)
            name = safe_name("aihub", split.lower(), subset, srcf.name)
            dst = DST / name
            if dst.exists():
                h = hashlib.md5(str(srcf).encode()).hexdigest()[:6]
                dst = DST / safe_name("aihub", split.lower(), subset, h, srcf.name)
                dup += 1
            try:
                how = link_or_copy(srcf, dst, mode)
            except Exception:
                bad += 1
                continue

        if verify and not verify_image(dst):
            try: dst.unlink()
            except: pass
            bad += 1
            continue
        ok += 1
        manifest.append({"dataset": "aihub", "src": str(srcf), "dst": str(dst), "how": how, "bytes": srcf.stat().st_size})

    write_manifest(manifest, IDX / "files_manifest.csv")
    print(f"[AI-Hub] ok={ok}, dup_renamed={dup}, bad={bad}, dst={DST}")

# ------------------------- UNION 정규화 -------------------------
def normalize_union(dst_root: Path, sources: list[str], mode: str, verify: bool, skip_existing: bool = True):
    DST = dst_root / "corpora" / "union" / "images"
    IDX = dst_root / "corpora" / "union" / "index"
    DST.mkdir(parents=True, exist_ok=True)
    IDX.mkdir(parents=True, exist_ok=True)

    # 1) 각 소스에서 이미지 확장자만 수집
    src_img_lists = []
    for tag in sources:
        root = dst_root / "corpora" / tag / "images"
        if not root.exists():
            raise FileNotFoundError(f"[union] 소스가 먼저 정규화되어 있어야 합니다: {root}")
        imgs = [f for f in root.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTS]
        src_img_lists.append((tag, sorted(imgs)))

    total = sum(len(v) for _, v in src_img_lists)
    manifest = []
    ok = dup = bad = 0

    from tqdm import tqdm
    with tqdm(total=total, desc="UNION ← corpora/*/images", unit="img", ncols=100) as pbar:
        for tag, img_list in src_img_lists:
            for srcf in img_list:
                # 이미 한 번 visdrone/sodaa/aihub 쪽에서 접두사가 들어간 경우가 많아
                # 중복 접두사를 피하려면 파일명이 이미 tag로 시작하면 그대로 씁니다.
                base = srcf.name
                if not base.lower().startswith(f"{tag.lower()}_"):
                    base = f"{tag}_{base}"

                dst = DST / base

                # (new) 재실행 시 그대로 "스킵"하여 멱등성 확보
                if dst.exists() and skip_existing:
                    ok += 1
                    manifest.append({"dataset": "union", "src": str(srcf), "dst": str(dst), "how": "skip", "bytes": srcf.stat().st_size})
                    pbar.update(1)
                    continue

                # 이름 충돌이면서 skip_existing=False인 경우에만 리네임
                if dst.exists():
                    import hashlib
                    h = hashlib.md5(str(srcf).encode()).hexdigest()[:6]
                    dst = DST / f"{tag}_{h}_{srcf.name}"
                    dup += 1

                try:
                    how = link_or_copy(srcf, dst, mode)
                except Exception:
                    bad += 1
                    pbar.update(1)
                    continue

                if verify and not verify_image(dst):
                    try:
                        dst.unlink()
                    except:
                        pass
                    bad += 1
                    pbar.update(1)
                    continue

                ok += 1
                manifest.append({"dataset": "union", "src": str(srcf), "dst": str(dst), "how": how, "bytes": srcf.stat().st_size})
                pbar.update(1)

    write_manifest(manifest, IDX / "files_manifest.csv")
    print(f"[UNION] ok={ok}, dup_renamed={dup}, bad={bad}, dst={DST}")

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Dataset normalizer → dino_test/data/corpora/<name>/images")
    ap.add_argument("--dataset", required=True, choices=["visdrone", "sodaa", "aihub", "union"],
                    help="정규화할 대상")
    ap.add_argument("--src", type=str, default=None, help="원본 루트(미지정 시 자동 탐색)")
    ap.add_argument("--mode", choices=["link","copy","symlink"], default="link", help="링크 방식")
    ap.add_argument("--verify", action="store_true", help="cv2.imread로 무결성 검사")
    ap.add_argument("--max-files", type=int, default=None, help="테스트용 상한(앞에서부터 N개만)")
    # aihub 전용
    ap.add_argument("--convert", choices=["tiff2jpg","keep"], default="tiff2jpg",
                    help="AI-Hub TIF 처리: tiff2jpg(권장) | keep(원본 확장자 유지)")
    ap.add_argument("--jpeg-quality", type=int, default=92, help="AI-Hub JPG 저장 품질(1~100)")
    # union 전용
    ap.add_argument("--sources", type=str, default="visdrone,sodaa,aihub",
                    help="union에 포함할 corpora 소스 목록(콤마 구분)")
    ap.add_argument("--skip-existing", action="store_true",
                help="union 생성 시 이미 있는 파일은 건너뜀(멱등 실행).")

    args = ap.parse_args()
    BASE = resolve_base()

    if args.dataset == "visdrone":
        normalize_visdrone(BASE / "data", Path(args.src) if args.src else None,
                           args.mode, args.verify, args.max_files)
    elif args.dataset == "sodaa":
        normalize_sodaa(BASE / "data", Path(args.src) if args.src else None,
                        args.mode, args.verify, args.max_files)
    elif args.dataset == "aihub":
        normalize_aihub(BASE / "data", Path(args.src) if args.src else None,
                        args.mode, args.verify, args.max_files,
                        convert=(args.convert=="tiff2jpg"),
                        jpeg_quality=args.jpeg_quality)
    elif args.dataset == "union":
        srcs = [s.strip() for s in args.sources.split(",") if s.strip()]
        normalize_union(BASE / "data", srcs, args.mode, args.verify, skip_existing=args.skip_existing)
    else:  # union
        srcs = [s.strip() for s in args.sources.split(",") if s.strip()]
        normalize_union(BASE / "data", srcs, args.mode, args.verify)

if __name__ == "__main__":
    main()
