"""
11_VCI_계산.py
─────────────────────────────────────────────
Day 4: Shannon Entropy 기반 VCI(Visual Complexity Index) 계산

각 이미지의 클래스 픽셀 비율 → Shannon entropy
교차로별 4방향 평균 = 그 교차로의 VCI

두 가지 VCI 계산:
  · VCI_basic    : 모든 클래스 동일 가중치
  · VCI_weighted : 도로교통 시각 부담 큰 클래스에 가중치 부여
                   (traffic sign/light, 보행자, 이륜차 등 높게)

입력:
  · 999_pilot/10_segmentation_stats.csv
  · 999_pilot/08_표본_30개.csv (메타데이터)
출력:
  · 999_pilot/11_VCI_교차로별.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── 경로 설정 ───────────────────────────────
BASE_DIR  = Path.home() / "dev/졸업논문/999_pilot"
STATS_CSV = BASE_DIR / "03_data/03_segmentation_stats.csv"
SAMPLE_CSV = BASE_DIR / "03_data/01_표본_30개.csv"
OUTPUT_CSV = BASE_DIR / "03_data/04_VCI_교차로별.csv"

# ─── 클래스별 가중치 (도로교통 시각 부담) ────
# 큰 값 = 운전자 주목/위험 인지 부담 큼
WEIGHTS = {
    "road":          1.0,
    "sidewalk":      1.0,
    "building":      1.0,
    "wall":          0.5,
    "fence":         1.0,
    "pole":          1.5,   # 작지만 시각적 clutter
    "traffic light": 2.0,   # 운전자 핵심 정보
    "traffic sign":  2.0,   # 운전자 핵심 정보
    "vegetation":    0.7,
    "terrain":       0.5,
    "sky":           0.3,   # 단조로움
    "person":        2.5,   # 보행자, 사고 위험 최고
    "rider":         2.5,
    "car":           1.5,
    "truck":         1.5,
    "bus":           1.5,
    "train":         1.0,
    "motorcycle":    2.0,   # 사각지대 위험
    "bicycle":       2.0,
}

# ─── Entropy 계산 함수 ───────────────────────
def shannon_entropy(p):
    """기본 Shannon entropy. p: array of probabilities summing to 1."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    p = p / p.sum()   # 정규화 (혹시 합이 1이 아닐 때)
    return float(-np.sum(p * np.log2(p)))


def weighted_shannon_entropy(p_dict, weights):
    """
    가중치 Shannon entropy.
    1) 각 클래스 확률에 가중치 곱하기: q_i = w_i * p_i
    2) 합이 1이 되도록 정규화
    3) 표준 Shannon entropy 계산
    """
    weighted = np.array([
        weights.get(cls, 1.0) * p_dict.get(cls, 0.0)
        for cls in weights
    ])
    return shannon_entropy(weighted)


# ─── 데이터 로드 ─────────────────────────────
df = pd.read_csv(STATS_CSV)
print(f"이미지 수    : {len(df)}")
print(f"고유 교차로  : {df['intersection_id'].nunique()}")

# 클래스 컬럼 (intersection_id, direction 제외)
class_cols = [c for c in df.columns if c not in ["intersection_id", "direction"]]
print(f"클래스 컬럼  : {class_cols}")

# ─── 이미지별 entropy 계산 ───────────────────
def row_to_entropies(row):
    p_dict = {c: row[c] for c in class_cols}
    p_array = np.array([row[c] for c in class_cols])
    return pd.Series({
        "H_basic":    shannon_entropy(p_array),
        "H_weighted": weighted_shannon_entropy(p_dict, WEIGHTS),
    })

df[["H_basic", "H_weighted"]] = df.apply(row_to_entropies, axis=1)

print("\n[이미지별 entropy 분포]")
print(df[["H_basic", "H_weighted"]].describe().to_string())

# ─── 교차로별 평균 (4방향) ───────────────────
vci = df.groupby("intersection_id").agg(
    VCI_basic     = ("H_basic", "mean"),
    VCI_weighted  = ("H_weighted", "mean"),
    H_basic_std   = ("H_basic", "std"),     # 방향별 변동성
    n_directions  = ("direction", "count"),
).reset_index()

print(f"\n교차로 수: {len(vci)}")

# ─── 메타데이터 결합 ─────────────────────────
sample = pd.read_csv(SAMPLE_CSV)
vci = sample.merge(vci, on="intersection_id", how="left")

# Street View 없는 교차로는 VCI=NaN 됨 (1254, 3804 예상)
missing = vci[vci["VCI_basic"].isna()]["intersection_id"].tolist()
if missing:
    print(f"⚠ VCI 계산 불가 (Street View 없음): {missing}")

# ─── 저장 ────────────────────────────────────
vci.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n저장: {OUTPUT_CSV}")

# ─── 요약 출력 ───────────────────────────────
print("\n" + "=" * 70)
print("VCI 통계 요약")
print("=" * 70)

valid = vci[vci["VCI_basic"].notna()].copy()
print(f"\n[VCI_basic]    min={valid['VCI_basic'].min():.3f}  max={valid['VCI_basic'].max():.3f}  mean={valid['VCI_basic'].mean():.3f}")
print(f"[VCI_weighted] min={valid['VCI_weighted'].min():.3f}  max={valid['VCI_weighted'].max():.3f}  mean={valid['VCI_weighted'].mean():.3f}")
print(f"  (최대 가능값 ≈ log₂(19) = {np.log2(19):.3f} bits)")

print("\n[VCI_basic 상위 5개 (복잡한 교차로)]")
top5 = valid.nlargest(5, "VCI_basic")[
    ["intersection_id", "구_이름", "MOCT_NODE_NAME", "VCI_basic", "VCI_weighted"]
]
print(top5.to_string(index=False))

print("\n[VCI_basic 하위 5개 (단조로운 교차로)]")
bot5 = valid.nsmallest(5, "VCI_basic")[
    ["intersection_id", "구_이름", "MOCT_NODE_NAME", "VCI_basic", "VCI_weighted"]
]
print(bot5.to_string(index=False))

print("\n[도로등급 그룹별 평균 VCI]")
by_road = valid.groupby("road_group")[["VCI_basic", "VCI_weighted"]].mean().round(3)
print(by_road.to_string())

print("\n[신호화 여부별 평균 VCI]")
by_sig = valid.groupby("is_signalized")[["VCI_basic", "VCI_weighted"]].mean().round(3)
print(by_sig.to_string())
