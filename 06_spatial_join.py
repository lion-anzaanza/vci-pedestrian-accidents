"""
13_사고데이터_spatial_join.py (수정판)
─────────────────────────────────────────────
변경점: acc_df에서 pano_year 제거 (sample과 중복돼 merge 시 _x/_y로 split되는 문제)
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

# ─── 경로 ────────────────────────────────────
BASE_DIR    = Path.home() / "dev/졸업논문/999_pilot"
SAMPLE_CSV  = BASE_DIR / "03_data/01_표본_30개.csv"
SV_STATUS   = BASE_DIR / "03_data/02_streetview_status.csv"
HOTSPOT_CSV = BASE_DIR / "03_data/05_사고다발지역_부산.csv"
VCI_CSV     = BASE_DIR / "03_data/04_VCI_교차로별.csv"
OUTPUT_CSV  = BASE_DIR / "03_data/06_분석테이블_최종.csv"

WGS84      = "EPSG:4326"
KOREA_CRS  = "EPSG:5179"
BUFFER_M   = 150
TIME_WIN   = 2

# ─── 1) 데이터 로드 ──────────────────────────
print("[1] 데이터 로드")
sample = pd.read_csv(SAMPLE_CSV)
sv     = pd.read_csv(SV_STATUS)
hot    = pd.read_csv(HOTSPOT_CSV)
vci    = pd.read_csv(VCI_CSV)

print(f"   · 표본 교차로     : {len(sample)}")
print(f"   · SV 메타데이터   : {len(sv)}")
print(f"   · 다발지 hotspot  : {len(hot)}")
print(f"   · VCI 결과        : {len(vci)}")

# ─── 2) SV 촬영연도 추출 + sample에 merge ────
sv["pano_year"] = sv["pano_date"].astype(str).str[:4]
sv["pano_year"] = pd.to_numeric(sv["pano_year"], errors="coerce")
sample = sample.merge(
    sv[["intersection_id", "pano_year"]], on="intersection_id", how="left"
)
print(f"\n[2] SV 촬영연도 분포")
print(sample["pano_year"].value_counts().sort_index().to_string())

# ─── 3) 교차로 / 다발지 GeoDataFrame ─────────
print(f"\n[3] GeoDataFrame 변환 (CRS={KOREA_CRS})")
inter_gdf = gpd.GeoDataFrame(
    sample,
    geometry=gpd.points_from_xy(sample["lon"], sample["lat"]),
    crs=WGS84,
).to_crs(KOREA_CRS)

hot_gdf = gpd.GeoDataFrame(
    hot,
    geometry=gpd.points_from_xy(hot["longitude"], hot["latitude"]),
    crs=WGS84,
).to_crs(KOREA_CRS)

# ─── 4) 교차로별 매칭 ────────────────────────
print(f"\n[4] Spatial + Temporal 매칭 (buffer={BUFFER_M}m, ±{TIME_WIN}년)")

results = []
for _, inter in inter_gdf.iterrows():
    iid = int(inter["intersection_id"])
    pano_year = inter["pano_year"]

    if pd.isna(pano_year):
        results.append({"intersection_id": iid, "match_status": "no_sv"})
        continue

    pano_year = int(pano_year)
    y_min = pano_year - TIME_WIN
    y_max = pano_year + TIME_WIN

    hot_time = hot_gdf[
        (hot_gdf["acdntYear"] >= y_min) & (hot_gdf["acdntYear"] <= y_max)
    ]

    distances = hot_time.geometry.distance(inter.geometry)
    nearby = hot_time[distances <= BUFFER_M].copy()
    nearby["dist_m"] = distances[distances <= BUFFER_M]

    # ⚠ pano_year는 sample에 이미 있어서 여기서 제외
    row = {
        "intersection_id": iid,
        "time_window":     f"{y_min}-{y_max}",
        "match_status":    "ok",
        "n_hotspots":      len(nearby),
        "n_보행노인":      int((nearby["acdntTypeSe"] == "보행노인").sum()),
        "n_자전거":        int((nearby["acdntTypeSe"] == "자전거").sum()),
        "n_보행어린이":    int((nearby["acdntTypeSe"] == "보행어린이").sum()),
        "n_스쿨존어린이":  int((nearby["acdntTypeSe"] == "스쿨존어린이").sum()),
        "sum_occrrnc":     int(nearby["occrrncCo"].sum()) if len(nearby) else 0,
        "sum_caslt":       int(nearby["casltCo"].sum()) if len(nearby) else 0,
        "sum_death":       int(nearby["deathCo"].sum()) if len(nearby) else 0,
        "sum_swpsn":       int(nearby["swpsnCo"].sum()) if len(nearby) else 0,
        "sum_sinjpsn":     int(nearby["sinjpsnCo"].sum()) if len(nearby) else 0,
        "min_dist_m":      float(nearby["dist_m"].min()) if len(nearby) else None,
    }
    results.append(row)

acc_df = pd.DataFrame(results)
print(f"   · 매칭 완료: {len(acc_df)}개 교차로")
print(f"   · SV 없는 교차로: {(acc_df['match_status']=='no_sv').sum()}개")
print(f"   · hotspot >0 교차로: {(acc_df.get('n_hotspots', 0) > 0).sum()}개")

# ─── 5) VCI + 메타데이터 + 사고 결합 ─────────
print(f"\n[5] 최종 분석 테이블 생성")
final = sample.merge(
    vci[["intersection_id", "VCI_basic", "VCI_weighted", "H_basic_std"]],
    on="intersection_id", how="left",
).merge(
    acc_df, on="intersection_id", how="left",
)

front_cols = [
    "intersection_id",
    "구_이름", "MOCT_NODE_NAME", "road_group", "도로등급",
    "type", "is_signalized", "차선수", "제한속도",
    "lon", "lat", "pano_year", "time_window",
    "VCI_basic", "VCI_weighted", "H_basic_std",
    "n_hotspots",
    "n_보행노인", "n_자전거", "n_보행어린이", "n_스쿨존어린이",
    "sum_occrrnc", "sum_caslt", "sum_death", "sum_swpsn", "sum_sinjpsn",
    "min_dist_m", "match_status",
]
final = final[[c for c in front_cols if c in final.columns]]

final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"   · 저장: {OUTPUT_CSV}")

# ─── 6) 요약 ─────────────────────────────────
print("\n" + "=" * 70)
print(" 최종 요약")
print("=" * 70)

analyzable = final[final["VCI_basic"].notna() & final["n_hotspots"].notna()]
print(f"\n분석 가능 교차로: {len(analyzable)}개 (VCI + 사고 매칭 모두 OK)")

print(f"\n[hotspot 매칭 분포]")
print(final["n_hotspots"].value_counts().sort_index().to_string())

print(f"\n[사고유형별 매칭 총합]")
for col in ["n_보행노인", "n_자전거", "n_보행어린이", "n_스쿨존어린이"]:
    print(f"   {col:15s}: {int(final[col].fillna(0).sum())}건")

print(f"\n[VCI vs 사고건수 — 단순 상관 미리보기]")
if len(analyzable) > 0:
    for vci_col in ["VCI_basic", "VCI_weighted"]:
        for acc_col in ["n_hotspots", "sum_occrrnc", "sum_caslt"]:
            corr = analyzable[[vci_col, acc_col]].corr().iloc[0, 1]
            print(f"   {vci_col:15s} ↔ {acc_col:15s}: r = {corr:+.3f}")

print(f"\n[샘플 10개 교차로 — 매칭 결과]")
preview_cols = ["intersection_id", "구_이름", "MOCT_NODE_NAME",
                "pano_year", "VCI_basic", "n_hotspots", "sum_occrrnc"]
preview_avail = [c for c in preview_cols if c in final.columns]
print(final[preview_avail].head(10).to_string(index=False))
