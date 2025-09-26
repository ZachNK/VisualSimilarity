# scripts/rerank_utils.py
from dataclasses import dataclass
import numpy as np

@dataclass
class GateCfg:
    use_gate: bool = True
    tau_margin: float = 0.03   # 임베딩 1위/2위 거리차(d2-d1) 임계
    tau_inlier: float = 0.08   # 인라이어 비율(0~1) 임계

def should_rerank(d1, d2, inlier_ratio, g: GateCfg) -> bool:
    if not g.use_gate:
        return True
    # 1등이 충분히 확실하면 굳이 재정렬 X
    if (d2 - d1) >= g.tau_margin:
        return False
    # 기하 신뢰도가 낮으면(인라이어 비율 낮음) 재정렬 X
    if inlier_ratio < g.tau_inlier:
        return False
    return True

def combine_scores(embed_dists, geom_scores, alpha: float) -> np.ndarray:
    """
    embed_dists: (K,) 작을수록 좋음(L2 등)
    geom_scores: (K,) 클수록 좋음(inlier_ratio 등)
    반환: 작을수록 좋은 최종 스코어(0~1 정규화 후 결합)
    """
    e = np.asarray(embed_dists, dtype=float)
    g = np.asarray(geom_scores, dtype=float)
    # [0,1] 정규화
    e = (e - e.min()) / (e.ptp() + 1e-9)            # 작은 값이 좋음
    g = (g - g.min()) / (g.ptp() + 1e-9)            # 큰 값이 좋음
    g = 1.0 - g                                      # 작은 값이 좋음으로 변환
    return (1.0 - alpha) * e + alpha * g

def apply_rerank(nn_dists, candidate_ids, geom_scores,
                 alpha: float, k: int, gcfg: GateCfg):
    """
    nn_dists: (N,) 임베딩 거리(오름차순)
    candidate_ids: (N,) 후보 인덱스
    geom_scores: (N,) 후보별 기하 신뢰도(예: inlier_ratio, 0~1)
    반환: new_ids, applied(bool), kk(실제 재정렬한 K)
    """
    d1 = float(nn_dists[0]) if len(nn_dists) else 0.0
    d2 = float(nn_dists[1]) if len(nn_dists) > 1 else (d1 + 1e9)
    inlier_ratio_top1 = float(geom_scores[0]) if len(geom_scores) else 0.0

    if not should_rerank(d1, d2, inlier_ratio_top1, gcfg):
        return candidate_ids, False, 0

    kk = int(min(k, len(nn_dists), len(candidate_ids), len(geom_scores)))
    if kk <= 1:
        return candidate_ids, False, 0

    final = combine_scores(nn_dists[:kk], geom_scores[:kk], alpha)  # 작을수록 좋음
    order = np.argsort(final)
    new_ids = np.concatenate([candidate_ids[:kk][order], candidate_ids[kk:]])
    return new_ids, True, kk
