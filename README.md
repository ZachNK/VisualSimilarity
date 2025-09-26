<!-- 
Scene Similarity Retrieval 
DINOv2 + FAISS + {ORB+RANSAC, LightGlue+SuperPoint, SuperGlue+SuperPoint}

1. 이미지 원본(_Datasets → \data\datasets 으로 이동)
"D:\KNK\_Datasets\{VisDrone, SODA_A, AiHub}"의 데이터들이 "D:\KNK\_KSNU\_Projects\dino_test\data\datasets\{visdrone, sodaa, aihub, union}"로 이동

1.1 queryGPU (데이터셋 정리용) 가상환경 이동
(base) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ conda activate queryGPU
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$

1.2 데이터셋별 이동
1.2.1 VisDrone 데이터셋 이동:
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python prepare_datasets.py --dataset visdrone --overwrite

1.2.2 SODA-A 데이터셋 이동:
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python prepare_datasets.py --dataset sodaa --overwrite

1.2.3 AI-Hub 데이터셋 이동:
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python prepare_datasets.py --dataset aihub --overwrite

1.3 (옵션) 모든 데이터셋 이동:
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python prepare_datasets.py --dataset all 



2. 데이터셋 정규화 작업 (\data\datasets → \data\corpora 으로 이동)
"D:\KNK\_KSNU\_Projects\dino_test\data\datasets\{visdrone, sodaa, aihub, union}"의 원본 데이터셋을 "D:\KNK\_KSNU\_Projects\dino_test\data\corpora\{visdrone, sodaa, aihub, union}"로 이미지와 어노테이션을 공통으로 정규화 하여 이동 및 저장

2.1 queryGPU 가상환경에서 시작
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$

2.2 데이터셋별 정규화
2.2.1 VisDrone 데이터셋 정규화
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python normalize_dataset.py --dataset visdrone --mode link --verify

2.2.2 SODA-A 데이터셋 정규화 
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python normalize_dataset.py --dataset sodaa --mode link --verify

2.2.3 AI-Hub 데이터셋 정규화
(queryGPU) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python normalize_dataset.py --dataset aihub --convert tiff2jpg --jpeg-quality 92 --verify



3. 검색용 벡터 index 생성기 (파이프라인 3단계)
"D:\KNK\_KSNU\_Projects\dino_test\data\corpora\{visdrone, sodaa, aihub, union}\images"의 이미지들을 가지고 DINOv2 임베딩 추출하고, 얻은 임베딩을 수직 결합 후 L2 거리갑늬 IndexFlatL2로 인덱싱하는 FAISS 인덱스 구축을 한다. 
# 동작 흐름
1) 경로/프로파일/태그 결정 → 결과 폴더 생성
2) 이미지 재귀 수집 (필터/정렬)
3) 배치 임베딩 계산 (진행바/시간 누적)
4) FAISS 인덱스 구축 (IndexFlatL2)
5) 인덱스/경로 저장 (+overwrite 체크)
6) 성능 매트릭 CSV기록 (누적/스냅샷)

3.1 각 가상환경에 접근
(base) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ conda activate orb2025
(orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$

(base) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ conda activate light2025
(light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$

(base) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ conda activate super2025
(super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$

3.2 각 가상환경별로, 데이터셋별로 index구축
3.2.1 orb2025에서, VisDrone의 index 생성
(orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python build_index.py --dataset visdrone --device cuda --overwrite --profile orb
# 저장: results/visdrone/orb/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

3.2.2 orb2025에서, SODA-A의 index 생성
(orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python build_index.py --dataset sodaa --device cuda --overwrite --profile orb
# 저장: results/sodaa/orb/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

3.2.3 orb2025에서, AI-Hub의 index 생성
(orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python build_index.py --dataset aihub --device cuda --overwrite --profile orb
# 저장: results/aihub/orb/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

3.2.1 light2025에서, VisDrone의 index 생성
(light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python build_index.py --dataset visdrone --device cuda --overwrite --profile light
# 저장: results/visdrone/light/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

3.2.2 light2025에서, SODA-A의 index 생성
(light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python build_index.py --dataset sodaa --device cuda --overwrite --profile light
# 저장: results/sodaa/light/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

3.2.3 light2025에서, AI-Hub의 index 생성
(light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python build_index.py --dataset aihub --device cuda --overwrite --profile light
# 저장: results/aihub/light/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

3.2.1 super2025에서, VisDrone의 index 생성
(super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python build_index.py --dataset visdrone --device cuda --overwrite --profile super
# 저장: results/visdrone/super/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

3.2.2 super2025에서, SODA-A의 index 생성
(super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python build_index.py --dataset sodaa --device cuda --overwrite --profile super
# 저장: results/sodaa/super/tables/{build_index_runs.csv, build_index_perf_날짜.csv}

3.2.3 super2025에서, AI-Hub의 index 생성
(super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python build_index.py --dataset aihub --device cuda --overwrite --profile super
# 저장: results/aihub/super/tables/{build_index_runs.csv, build_index_perf_날짜.csv}



4. 쿼리 검색
각 가상환경과 데이터셋별로 쿼리 검색하는 알고리즘

4.1 orb2025 + visdrone
# (orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python eval_search.py --dataset visdrone --k 10

4.2 orb2025 + sodaa
# (orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python eval_search.py --dataset sodaa --k 10

4.3 orb2025 + aihub
# (orb2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python eval_search.py --dataset aihub --k 10

4.4 light2025 + visdrone
# (light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python eval_search.py --dataset visdrone --k 10

4.5 light2025 + sodaa
# (light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python eval_search.py --dataset sodaa --k 10

4.6 light2025 + aihub
# (light2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python eval_search.py --dataset aihub --k 10

4.7 super2025 + visdrone
# (super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python eval_search.py --dataset visdrone --k 10

4.8 super2025 + sodaa
# (super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python eval_search.py --dataset sodaa --k 10

4.9 super2025 + aihub
# (super2025) knk2025@DESKTOP-59ULDOH:/mnt/d/KNK/_KSNU/_Projects/dino_test/scripts$ python eval_search.py --dataset aihub --k 10

-->

