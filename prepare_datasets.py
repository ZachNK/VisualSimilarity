# scripts/prepare_datasets.py
#!/usr/bin/env python3
import os, sys, argparse, shutil
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm

# --------- 경로 해석 (Windows 경로 → WSL 경로 자동 변환 포함) ----------
def to_path(p: str) -> Path:
    """
    Windows 경로("D:\\foo\\bar")를 WSL 경로("/mnt/d/foo/bar")로 안전 변환.
    이미 리눅스 경로면 그대로 사용.
    """
    p = p.strip().rstrip("\\/")
    if len(p) >= 3 and p[1] == ":" and p[2] in ("\\", "/"):  # "D:\..."
        drive = p[0].lower()
        rest = p[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(p)

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def count_files(root: Path) -> int:
    n = 0
    for _, _, files in os.walk(root):
        n += len(files)
    return n

def copy_tree(src: Path, dst: Path, overwrite: bool = False) -> Tuple[int, int]:
    """
    src 전체 트리를 dst로 그대로 복사. 파일 개수와 제로바이트 개수 반환.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    ensure_dir(dst)
    total_files = count_files(src)
    zero_bytes = 0
    copied = 0

    with tqdm(total=total_files, unit="file", desc=f"Copying {src.name} -> {dst.name}") as pbar:
        for root, _, files in os.walk(src):
            root_p = Path(root)
            rel = root_p.relative_to(src)
            out_dir = dst / rel
            ensure_dir(out_dir)

            for fn in files:
                s = root_p / fn
                d = out_dir / fn

                if not overwrite and d.exists():
                    pbar.update(1)
                    continue

                # 원본 0바이트면 복사하지 않고 경고
                try:
                    if s.stat().st_size == 0:
                        zero_bytes += 1
                        pbar.set_postfix_str("WARN: zero-byte src")
                        # 그래도 동일 구조 유지를 위해 빈 파일로 둘 수도 있으나,
                        # 여기서는 스킵하지 않고 그대로 copy2 시도(메타 포함)
                except FileNotFoundError:
                    pbar.set_postfix_str("WARN: src missing")
                    pbar.update(1)
                    continue

                # 안전 복사
                shutil.copy2(s, d)

                # 사후 검증: 대상이 0바이트인지 확인
                try:
                    if d.stat().st_size == 0:
                        zero_bytes += 1
                        # 즉시 중단해 문제 파악 쉽게
                        raise RuntimeError(
                            f"Zero-byte written: {d}\n"
                            f"→ 경로 변환/권한/안티바이러스/네트워크 경로 문제 확인 필요"
                        )
                except FileNotFoundError:
                    raise RuntimeError(f"Destination vanished while copying: {d}")

                copied += 1
                pbar.update(1)

    return copied, zero_bytes

# --------- 데이터셋 사전 ----------
def dataset_paths(dataset: str) -> Tuple[Path, Path]:
    """
    질문에서 주신 원본/대상 경로를 그대로 매핑.
    """
    if dataset.lower() == "aihub":
        src = to_path(r"D:\KNK\_Datasets\AiHub")
        dst = to_path(r"D:\KNK\_KSNU\_Projects\dino_test\data\datasets\aihub")
    elif dataset.lower() == "visdrone":
        src = to_path(r"D:\KNK\_Datasets\VisDrone")
        dst = to_path(r"D:\KNK\_KSNU\_Projects\dino_test\data\datasets\visdrone")
    elif dataset.lower() in ("sodaa", "soda-a", "soda_a"):
        src = to_path(r"D:\KNK\_Datasets\SODA_A")
        dst = to_path(r"D:\KNK\_KSNU\_Projects\dino_test\data\datasets\sodaa")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return src, dst

# --------- 메인 ----------
def main(argv: List[str] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["aihub", "visdrone", "sodaa", "soda-a", "soda_a", "all"],
                    help="복사할 데이터셋")
    ap.add_argument("--overwrite", action="store_true", help="기존 파일이 있으면 덮어쓰기")
    args = ap.parse_args(argv)

    tasks = []
    if args.dataset == "all":
        tasks = ["aihub", "visdrone", "sodaa"]
    else:
        # soda-a/soda_a → sodaa 표준화
        tasks = ["sodaa" if args.dataset in ("soda-a","soda_a") else args.dataset]

    total_copied = 0
    total_zero = 0

    for ds in tasks:
        src, dst = dataset_paths(ds)
        print(f"[{ds}] SRC={src}")
        print(f"[{ds}] DST={dst}")
        ensure_dir(dst)
        copied, zero = copy_tree(src, dst, overwrite=args.overwrite)
        print(f"[{ds}] copied={copied}, zero_byte={zero}")
        total_copied += copied
        total_zero += zero

    print(f"[DONE] total_copied={total_copied}, total_zero={total_zero}")
    if total_zero > 0:
        print("※ 0바이트 파일이 감지되었습니다. 다음을 점검하세요:")
        print(" - 원본 파일 크기 (원본도 0바이트인지)")
        print(" - 경로 번역이 올바른지 (/mnt/d/... 이 실제 존재하는지)")
        print(" - 안티바이러스/랜섬웨어 보호가 Python의 쓰기/읽기를 막지 않는지")
        print(" - 네트워크 드라이브/권한 문제")

if __name__ == "__main__":
    main()
