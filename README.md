
# Scene Similarity Retrieval 
DINOv2 + FAISS + {ORB+RANSAC, LightGlue+SuperPoint, SuperGlue+SuperPoint} <br />


# 개요
- 현재 코드는 가장 단순한 FAISS L2(flat) 설정을 함. <br />
- IVF/HNSW/PQ 등 고급 파라미터는 쓰지 않음. <br />

## Datasets 경로

<img width="540" height="445" alt="Datasets_Root_2025-09-29 132050" src="https://github.com/user-attachments/assets/954f7dc2-4cd5-42ac-b82d-2b8d369414b8" />


## FAISS 인덱스(구축 시)

임베딩 차원: <br />
- dinov2_utils.get_embeddings(...)의 출력 차원(= DINOv2 Base CLS 토큰 벡터 차원). <br />
- 코드에서 하드코딩 없이 자동 추출(로그에 dim=으로 기록). <br />

거리 지표 / 인덱스 타입: 
- faiss.IndexFlatL2(dim) → L2(유클리드), Flat. <br />

배치/가속:
- 임베딩 추출 배치 크기 --batch 64(기본). GPU 사용 여부는 Torch/CUDA 가용성에 따름(임베딩 단계). <br />

저장 산출물: <br />
- 인덱스 파일: data/index/<dataset_tag>/faiss_index.bin
- 경로 맵: data/index/<dataset_tag>/image_paths.npy

## 검색 단계(eval/baseline)

탐색 Top-K: 
- --k. <br />

GPU 사용 방식:
- GPU가 있으면 CPU 인덱스를 GPU(0)로 클론하고 useFloat16=True로 FP16 검색 활성화. (HNSW 인덱스가 아니어야 GPU 래핑) <br />

정규화:
- 인덱스 metric이 IP일 때만 L2 정규화하는 코드가 있으나, 실제 인덱스는 L2라 정규화는 적용되지 않음. <br />

결과 기록:
- baseline 결과/성능 요약 CSV(Recall@K, 임베딩/검색 시간 ms) 저장. <br />

## 재정렬(rerank) 단계와 FAISS

FAST 스크립트(LightGlue/SuperGlue): 
- --reuse-baseline을 요구 → FAISS 재검색 없음, baseline의 Top-K 후보·거리 그대로 사용 후 기하 매칭으로 재정렬. <br />

ORB FAST 스크립트: 
- --reuse-baseline 없을 때만 인덱스를 불러 그 자리에서 FAISS 검색 수행. 이 경우: <br />

인덱스 파일 로드: 
- faiss_index.bin <br />

GPU 래핑: 
- GPU가 있으면 index_cpu_to_all_gpus(...)로 전 GPU에 분산(옵션 기본). HNSW면 CPU 유지. <br />

Top-K:
---k(= 1, 5, 10, 20). <br />

## 안 쓰는(또는 고정된) FAISS 파라미터

nlist, nprobe, efSearch/efConstruction, PQ(압축), OPQ, IVF, HNSW:
- 미사용 (코드는 전부 IndexFlatL2 기반). <br />

Metric 전환(IP): 
- 미사용 (인덱스가 L2이므로 항상 거리=작을수록 유사). <br />





# 0. 가상환경 준비

## 0.1 공통:
- \<Project File\> : 사용자 임의 로컬 경로 <br />

- Miniforge/Conda가 깔려 있다는 전제
conda config --add channels conda-forge
conda config --set channel_priority flexible

- (선택) mamba 설치 — 설치속도↑, 의존성 충돌↓
conda install -n base -y mamba

- 프로젝트 루트 환경변수(WSL)
echo 'export DINO_BASE="/mnt/d/\<Project File\>/_Projects/dino_test"' >> ~/.bashrc
source ~/.bashrc

## 0.2 orb2025 (ORB+RANSAC 알고리즘 패키지)
- 핵심 패키지: torch+cuda, torchvision, timm, transformers, faiss(권장: GPU), opencv, pandas, numpy, pyyaml
mamba create -y -n queryGPU python=3.10
conda activate queryGPU

# PyTorch (CUDA 12.x 전용 채널; 실패 시 "대안" 참고)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# 비전/임베딩/유틸
mamba install -y opencv numpy pandas scipy pillow tqdm pyyaml rich
pip install timm transformers einops huggingface-hub

# FAISS (선택 1: GPU 권장)
- cu12 계열: pip 패키지명이 시기에 따라 "faiss-gpu-cu12", "faiss-cuda12x" 등으로 다를 수 있음
- 먼저 아래를 시도하고, 실패 시 선택 2로 (CPU) 진행
pip install faiss-gpu-cu12 || true

- FAISS (선택 2: CPU 대안 — GPU가 안 맞을 때 안정적)
pip install faiss-cpu

0.3 light2025 (LightGlue+SuperPoint 알고리즘 패키지)
- 핵심 패키지: opencv-contrib(xfeatures 포함), numpy/scipy, pandas, pyyaml

mamba create -y -n orb2025 python=3.10
conda activate orb2025

- OpenCV는 contrib 포함 버전이 필요 (SIFT/SURF 등 확장 기능 포함)
mamba install -y opencv numpy scipy pandas pillow tqdm pyyaml rich

0.4 super2025 (SuperGlue+SuperPoint 알고리즘 패키지)
- 핵심 패키지: torch+cuda, kornia, lightglue, opencv, numpy, pandas, pyyaml

mamba create -y -n light2025 python=3.10
conda activate light2025

# PyTorch(CUDA)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

- 필수 라이브러리
mamba install -y opencv numpy pandas pillow tqdm pyyaml rich
pip install kornia

- LightGlue (공식 PyPI 패키지명: lightglue)
pip install lightglue



# 1. 이미지 원본(_Datasets → \data\datasets 으로 이동)
D:\<Project File>\_Datasets\{VisDrone, SODA_A, AiHub}의 데이터들이 D:\\<Project File\>\_Projects\dino_test\data\datasets\{visdrone, sodaa, aihub, union}로 이동

## 1.1 queryGPU (데이터셋 정리용) 가상환경 이동
(base) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ conda activate queryGPU
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$

## 1.2 데이터셋별 이동
1.2.1 VisDrone 데이터셋 이동:
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python prepare_datasets.py --dataset visdrone --overwrite

## 1.2.2 SODA-A 데이터셋 이동:
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python prepare_datasets.py --dataset sodaa --overwrite

## 1.2.3 AI-Hub 데이터셋 이동:
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python prepare_datasets.py --dataset aihub --overwrite

## 1.3 (옵션) 모든 데이터셋 이동:
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python prepare_datasets.py --dataset all 



# 2. 데이터셋 정규화 작업 (\data\datasets → \data\corpora 으로 이동)
D:\\<Project File\>\_Projects\dino_test\data\datasets\{visdrone, sodaa, aihub, union}의 원본 데이터셋을 D:\\<Project File\>\_Projects\dino_test\data\corpora\{visdrone, sodaa, aihub, union}로 이미지와 어노테이션을 공통으로 정규화 하여 이동 및 저장

## 2.1 queryGPU 가상환경에서 시작
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$

## 2.2 데이터셋별 정규화
###2.2.1 VisDrone 데이터셋 정규화
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python normalize_dataset.py --dataset visdrone --mode link --verify

### 2.2.2 SODA-A 데이터셋 정규화 
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python normalize_dataset.py --dataset sodaa --mode link --verify

### 2.2.3 AI-Hub 데이터셋 정규화
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python normalize_dataset.py --dataset aihub --convert tiff2jpg --jpeg-quality 92 --verify



# 3. 검색용 벡터 index 생성기 (파이프라인 3단계)
D:\\<Project File\>\_Projects\dino_test\data\corpora\{visdrone, sodaa, aihub, union}\images의 이미지들을 가지고 DINOv2 임베딩 추출하고, 얻은 임베딩을 수직 결합 후 L2 거리 값의 IndexFlatL2로 인덱싱하는 FAISS 인덱스 구축을 한다. 
- 동작 흐름
1) 경로/프로파일/태그 결정 → 결과 폴더 생성
2) 이미지 재귀 수집 (필터/정렬)
3) 배치 임베딩 계산 (진행바/시간 누적)
4) FAISS 인덱스 구축 (IndexFlatL2)
5) 인덱스/경로 저장 (+overwrite 체크)
6) 성능 매트릭 CSV기록 (누적/스냅샷)

## 3.1 각 가상환경에 접근
(base) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ conda activate orb2025
(orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$

(base) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ conda activate light2025
(light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$

(base) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ conda activate super2025
(super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$

## 3.2 각 가상환경별로, 데이터셋별로 index구축
### 3.2.1 orb2025에서, VisDrone의 index 생성
(orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python build_index.py --dataset visdrone --device cuda --overwrite --profile orb
저장: results/visdrone/orb/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

### 3.2.2 orb2025에서, SODA-A의 index 생성
(orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python build_index.py --dataset sodaa --device cuda --overwrite --profile orb
저장: results/sodaa/orb/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

### 3.2.3 orb2025에서, AI-Hub의 index 생성
(orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python build_index.py --dataset aihub --device cuda --overwrite --profile orb
저장: results/aihub/orb/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

### 3.2.1 light2025에서, VisDrone의 index 생성
(light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python build_index.py --dataset visdrone --device cuda --overwrite --profile light
저장: results/visdrone/light/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

### 3.2.2 light2025에서, SODA-A의 index 생성
(light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python build_index.py --dataset sodaa --device cuda --overwrite --profile light
저장: results/sodaa/light/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

### 3.2.3 light2025에서, AI-Hub의 index 생성
(light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python build_index.py --dataset aihub --device cuda --overwrite --profile light
저장: results/aihub/light/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

### 3.2.1 super2025에서, VisDrone의 index 생성
(super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python build_index.py --dataset visdrone --device cuda --overwrite --profile super
저장: results/visdrone/super/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

### 3.2.2 super2025에서, SODA-A의 index 생성
(super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python build_index.py --dataset sodaa --device cuda --overwrite --profile super
저장: results/sodaa/super/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

### 3.2.3 super2025에서, AI-Hub의 index 생성
(super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python build_index.py --dataset aihub --device cuda --overwrite --profile super
저장: results/aihub/super/tables/{build_index_runs.csv, build_index_perf_날짜.csv}



# 4. 쿼리 검색
- 각 가상환경과 데이터셋별로 쿼리 검색하는 알고리즘

## 4.1 orb2025 + visdrone
(orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python eval_search.py --dataset visdrone --k 10

## 4.2 orb2025 + sodaa
(orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python eval_search.py --dataset sodaa --k 10

## 4.3 orb2025 + aihub
(orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python eval_search.py --dataset aihub --k 10

## 4.4 light2025 + visdrone
(light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python eval_search.py --dataset visdrone --k 10

## 4.5 light2025 + sodaa
(light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python eval_search.py --dataset sodaa --k 10

## 4.6 light2025 + aihub
(light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python eval_search.py --dataset aihub --k 10

## 4.7 super2025 + visdrone
(super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python eval_search.py --dataset visdrone --k 10

## 4.8 super2025 + sodaa
(super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python eval_search.py --dataset sodaa --k 10

## 4.9 super2025 + aihub
(super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/\<Project File\>/_Projects/dino_test/scripts$ python eval_search.py --dataset aihub --k 10

























