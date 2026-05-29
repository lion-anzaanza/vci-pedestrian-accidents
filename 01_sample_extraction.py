"""
08_표본_30개_추출.py
─────────────────────────────────────────────
30개 층화 표본 추출 (Day 1)

층화 변수:
  · 도로등급 그룹: A_국도급(101~103) / B_광역시도(104) / C_기타(105~108)
  · 신호화 여부: TRUE / FALSE

→ 6개 셀 × 5개씩 = 30개 (셀별 부족하면 가용한 만큼)

입력: ~/dev/졸업논문/003_표준노드링크_매칭/결과/07_결과_교차로_표준노드링크.csv
출력: ~/dev/졸업논문/999_pilot/08_표본_30개.csv
"""

import pandas as pd
from pathlib import Path

CSV_PATH    = Path.home() / "dev/졸업논문/003_표준노드링크_매칭/결과/07_결과_교차로_표준노드링크.csv"
OUTPUT_PATH = Path.home() / "dev/졸업논문/999_pilot/08_표본_30개.csv"

# ─── 1. 데이터 로드 ───────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"전체 교차로     : {len(df):,}")

# ─── 2. 매칭된 것만 (속성 결측 없음) ──────────
matched = df[df["MOCT_NODE_ID"].notna()].copy()
print(f"매칭된 교차로   : {len(matched):,}")

# ─── 3. 도로등급 그룹화 ──────────────────────
def road_group(code):
    if code in [101, 102, 103]:  return "A_국도급"
    if code == 104:               return "B_광역시도"
    return "C_기타"

matched["road_group"] = matched["도로등급_코드"].map(road_group)

# ─── 4. 모집단 층별 분포 ─────────────────────
print("\n[모집단 층별 분포]")
print(matched.groupby(["road_group", "is_signalized"]).size().to_string())

# ─── 5. 층화 무작위 추출 (셀당 5개) ──────────
# pandas 3.x 호환성을 위해 명시적 루프 사용
sampled_dfs = []
for (rg, sig), group in matched.groupby(["road_group", "is_signalized"]):
    n = min(5, len(group))
    if n > 0:
        sampled_dfs.append(group.sample(n=n, random_state=42))

sample = pd.concat(sampled_dfs, ignore_index=True)

print(f"\n[표본 크기]: {len(sample)}")
print("\n[표본 층별 분포]")
print(sample.groupby(["road_group", "is_signalized"]).size().to_string())

# ─── 6. Street View 필요 컬럼만 저장 ──────────
cols = [
    "intersection_id", "lon", "lat",
    "구_이름", "type", "is_signalized",
    "MOCT_NODE_NAME", "도로등급", "차선수", "제한속도",
    "road_group",
]
available = [c for c in cols if c in sample.columns]
missing   = [c for c in cols if c not in sample.columns]
if missing:
    print(f"\n⚠ 누락된 컬럼: {missing}")
    print(f"   사용 가능한 컬럼: {list(sample.columns)}")

sample_out = sample[available].sort_values("intersection_id").reset_index(drop=True)
sample_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n저장 완료: {OUTPUT_PATH}")
print(f"\n[표본 미리보기]")
print(sample_out.to_string())
