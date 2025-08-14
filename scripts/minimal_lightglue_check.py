# minimal_lightglue_check_fix.py
import torch
from lightglue import LightGlue, DISK
from lightglue.utils import load_image, rbd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

extractor = DISK(max_num_keypoints=2048).eval().to(device)
matcher   = LightGlue(features='disk').eval().to(device)

im0 = load_image("/mnt/d/KNK/_KSNU/_Projects/dino_test/data/images/0000003.jpg").to(device)
im1 = load_image("/mnt/d/KNK/_KSNU/_Projects/dino_test/data/queries/0000003_rot90.jpg").to(device)

with torch.inference_mode():
    f0 = extractor.extract(im0)   # 배치 포함(B, ...), 여기서 rbd() 쓰지 말기
    f1 = extractor.extract(im1)

    # (가드) DISK의 descriptor가 [B,128,N]이면 [B,N,128]로 전치
    for f in (f0, f1):
        d = f['descriptors']
        if d.ndim == 3 and d.shape[-1] not in (128, 256) and d.shape[1] in (128, 256):
            f['descriptors'] = d.transpose(1, 2).contiguous()

    # 매칭 (배치 유지)
    out = matcher({'image0': f0, 'image1': f1})

    # 매칭 후에 배치 제거
    f0, f1, out = [rbd(x) for x in (f0, f1, out)]

print("desc dims:", f0['descriptors'].shape, f1['descriptors'].shape)  # -> [N,128]
print("matches:", out['matches'].shape)  # -> [M,2], M>0 기대
