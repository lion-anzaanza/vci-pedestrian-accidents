"""
14_회귀분석.py
─────────────────────────────────────────────
Day 6: VCI ↔ 사고 hotspot 회귀 분석

분석:
  1. 단순 상관 (Pearson, Spearman)
  2. 다중회귀 (VCI + 통제변수: 신호화, 차선수, 도로등급)
  3. 사고유형별 분리 (보행노인이 main, 다른 유형은 N 부족)
  4. 산점도 + 회귀선 시각화 (PNG 저장)

⚠ N=28의 한계 명시. 데모로 방법론 작동 입증이 목적.

입력: 999_pilot/13_분석테이블_최종.csv
출력:
  · 999_pilot/14_회귀결과_요약.txt
  · 999_pilot/14_산점도_VCI_사고.png

사전 설치:
  pip install statsmodels matplotlib
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats as sps

# ─── 경로 ────────────────────────────────────
BASE_DIR  = Path.home() / "dev/졸업논문/999_pilot"
DATA_CSV  = BASE_DIR / "03_data/06_분석테이블_최종.csv"
SUMMARY   = BASE_DIR / "03_data/07_회귀결과_요약.txt"
PLOT_PNG  = BASE_DIR / "03_data/07_산점도_VCI_사고.png"

# 한글 폰트 (Mac)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ─── 데이터 로드 + 정제 ──────────────────────
df = pd.read_csv(DATA_CSV)
print(f"전체 교차로: {len(df)}")

# 분석 대상: VCI + 사고 매칭 모두 OK
ana = df[df["VCI_basic"].notna() & df["n_hotspots"].notna()].copy()
print(f"분석 가능: {len(ana)}개")

# 카테고리 → dummy
ana["is_signalized_num"] = ana["is_signalized"].astype(int)
ana["road_A"] = (ana["road_group"] == "A_국도급").astype(int)
ana["road_C"] = (ana["road_group"] == "C_기타").astype(int)
# B_광역시도가 reference

# 로그 변환 (사고건수 right-skewed)
ana["log_occrrnc"] = np.log1p(ana["sum_occrrnc"])

# ─── 결과 저장용 ─────────────────────────────
out_lines = []
def log(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    out_lines.append(line)

log("=" * 70)
log(" Day 6 회귀 분석 결과")
log("=" * 70)
log(f"\n분석 표본: N = {len(ana)} (전체 30개 중 VCI + 사고 매칭 모두 OK)")
log(f"hotspot 매칭된 교차로: {(ana['n_hotspots'] > 0).sum()}개")

# ─── 1) 기초 통계 ────────────────────────────
log("\n" + "=" * 70)
log(" 1. 기초 통계")
log("=" * 70)
for col in ["VCI_basic", "VCI_weighted", "n_hotspots", "sum_occrrnc"]:
    s = ana[col]
    log(f"  {col:15s}  mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.2f}  max={s.max():.2f}")

# ─── 2) 단순 상관 (Pearson + Spearman) ───────
log("\n" + "=" * 70)
log(" 2. 단순 상관 (Pearson · Spearman)")
log("=" * 70)
log(f"\n{'변수쌍':<40s} {'Pearson r':>10s} {'p':>8s} {'Spearman ρ':>12s} {'p':>8s}")
log("-" * 90)
pairs = [
    ("VCI_basic",    "n_hotspots"),
    ("VCI_basic",    "sum_occrrnc"),
    ("VCI_basic",    "sum_caslt"),
    ("VCI_basic",    "n_보행노인"),
    ("VCI_weighted", "n_hotspots"),
    ("VCI_weighted", "sum_occrrnc"),
    ("VCI_weighted", "sum_caslt"),
    ("VCI_weighted", "n_보행노인"),
]
for v, a in pairs:
    if a not in ana.columns:
        continue
    pr, pp = sps.pearsonr(ana[v], ana[a])
    sr, sp = sps.spearmanr(ana[v], ana[a])
    log(f"{v} ↔ {a:<20s} {pr:+10.3f} {pp:>8.3f} {sr:+12.3f} {sp:>8.3f}")

# ─── 3) 다중회귀 (통제변수 추가) ─────────────
log("\n" + "=" * 70)
log(" 3. 다중회귀: 종속 = log(1 + 사고건수)")
log(" 통제: 신호화, 차선수, 제한속도, 도로등급(A/C; B는 reference)")
log("=" * 70)

# 결측 제거
reg_df = ana.dropna(subset=[
    "log_occrrnc", "VCI_basic", "is_signalized_num",
    "차선수", "제한속도", "road_A", "road_C",
]).copy()
log(f"\n회귀 표본 N = {len(reg_df)}")

# 모델 1: VCI_basic만
log("\n--- 모델 1: VCI_basic 단독 ---")
m1 = ols("log_occrrnc ~ VCI_basic", data=reg_df).fit()
log(m1.summary().as_text())

# 모델 2: VCI_basic + 통제변수
log("\n--- 모델 2: VCI_basic + 통제변수 ---")
m2 = ols(
    "log_occrrnc ~ VCI_basic + is_signalized_num + 차선수 + 제한속도 + road_A + road_C",
    data=reg_df,
).fit()
log(m2.summary().as_text())

# 모델 3: VCI_weighted 사용
log("\n--- 모델 3: VCI_weighted + 통제변수 ---")
m3 = ols(
    "log_occrrnc ~ VCI_weighted + is_signalized_num + 차선수 + 제한속도 + road_A + road_C",
    data=reg_df,
).fit()
log(m3.summary().as_text())

# ─── 4) 모델 비교 ────────────────────────────
log("\n" + "=" * 70)
log(" 4. 모델 비교")
log("=" * 70)
log(f"\n  {'모델':<35s} {'R²':>8s} {'Adj.R²':>10s} {'VCI coef':>12s} {'VCI p':>10s}")
log("-" * 80)
for name, m, vci_var in [
    ("M1: VCI_basic 단독",          m1, "VCI_basic"),
    ("M2: VCI_basic + 통제",        m2, "VCI_basic"),
    ("M3: VCI_weighted + 통제",     m3, "VCI_weighted"),
]:
    r2 = m.rsquared
    ar2 = m.rsquared_adj
    if vci_var in m.params.index:
        coef = m.params[vci_var]
        pval = m.pvalues[vci_var]
        log(f"  {name:<35s} {r2:>8.3f} {ar2:>10.3f} {coef:>+12.3f} {pval:>10.4f}")

# ─── 5) 시각화 ───────────────────────────────
log("\n" + "=" * 70)
log(" 5. 산점도 + 회귀선 저장")
log("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

panels = [
    ("VCI_basic",    "n_hotspots",  "VCI (basic) ↔ hotspot 수"),
    ("VCI_basic",    "sum_occrrnc", "VCI (basic) ↔ 사고건수 합"),
    ("VCI_weighted", "n_hotspots",  "VCI (weighted) ↔ hotspot 수"),
    ("VCI_weighted", "sum_occrrnc", "VCI (weighted) ↔ 사고건수 합"),
]

for ax, (x, y, title) in zip(axes.flat, panels):
    ax.scatter(ana[x], ana[y], s=60, alpha=0.7, edgecolor="k")
    # 회귀선
    if ana[x].nunique() > 1:
        slope, intercept, r, p, se = sps.linregress(ana[x], ana[y])
        xx = np.linspace(ana[x].min(), ana[x].max(), 50)
        ax.plot(xx, slope * xx + intercept, "r-", alpha=0.6,
                label=f"r={r:.3f}, p={p:.3f}")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(alpha=0.3)

plt.suptitle(f"VCI ↔ 사고 hotspot (N={len(ana)})", fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(PLOT_PNG, dpi=130, bbox_inches="tight")
log(f"  저장: {PLOT_PNG}")

# ─── 저장 ────────────────────────────────────
SUMMARY.write_text("\n".join(out_lines), encoding="utf-8")
print(f"\n최종 요약 저장: {SUMMARY}")
print(f"산점도 저장   : {PLOT_PNG}")
