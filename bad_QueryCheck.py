# bad_queryCheck.py
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 일부 깨진 JPEG도 강제로 열기 시도

prompt = "Choose dataset (0~3)\n0: VisDrone\n1: SODA-A\n2: AI-Hub\n3: Union\n"
datasetName = ["visdrone", "sodaa", "aihub", "union"]
chk = 0
while True:
    user_input = int(input(prompt))
    if user_input <= 3 & user_input >= 0:
        print("OK\n")
        chk = user_input
        break
    else:
        print(f"You wrote '{prompt}'. Please try again\n")

print(f"You choose {datasetName[chk]}, continue counting...")
base = Path(f"/mnt/d/KNK/_KSNU/_Projects/dino_test/data/corpora/{datasetName[chk]}/images")
bad = []
for p in base.rglob("*.jpg"):
    try:
        with Image.open(p) as im:
            im.verify()  # 파일 무결성 체크
    except Exception as e:
        bad.append((p, e))
for p, e in bad:
    print("BAD:", p, "|", e)
print("bad count:", len(bad))
