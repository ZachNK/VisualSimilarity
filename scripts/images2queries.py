# (images2queries.py) 원본 이미지 rot90, flip, bright 쿼리 이미지로 변환 
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps


BASE = Path(r"D:/KNK/_KSNU/_Projects/dino_test")
DATA     = BASE / "data"
IMAGES   = DATA / "images"
QUERIES  = DATA / "queries"
QUERIES.mkdir(parents=True, exist_ok=True)

# 처리 대상 확장자
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

angles = [30,45,60,90,120,135,150,180,210,225,240,270,300,315,330]
brights = [0.25,0.50,0.75,1.25,1.50,1.75]  # 1.00은 원본

cnt = 0
for img_path in IMAGES.iterdir():
    if img_path.suffix.lower() not in EXTS:
        continue

    with Image.open(img_path) as img_raw:
        # EXIF 회전 보정
        img = ImageOps.exif_transpose(img_raw).convert("RGB")

        # 회전
        for a in angles:
            img.rotate(a, expand=True).save(QUERIES/f"{img_path.stem}_rot{a:03d}.jpg", quality=95)

        # 명도
        enh = ImageEnhance.Brightness(img)
        for b in brights:
            tag = f"{int(round(b*100)):03d}"
            enh.enhance(b).save(QUERIES/f"{img_path.stem}_bright{tag}.jpg", quality=95)
    cnt += 1

print(f"Done. Generated {len(angles)+len(brights)} queries per image for {cnt} images → {QUERIES}")

    
