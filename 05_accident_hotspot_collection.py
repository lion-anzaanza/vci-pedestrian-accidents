"""
12_사고다발지역_API_수집.py (수정판)
─────────────────────────────────────────────
변경점:
  1) 응답 키 camelCase로 통일 (latitude, acdntYear, ctprvnSignguNm ...)
  2) timeout 30→90초로 늘리고 retry 3회 추가
  3) NODATA 연도는 건너뛰고 계속
"""

import os
import sys
import time
from pathlib import Path

import requests
import pandas as pd


# ─── 설정 ────────────────────────────────────
SERVICE_KEY = os.environ.get("DATA_GO_KR_KEY")
if not SERVICE_KEY:
    raise SystemExit('❌ export DATA_GO_KR_KEY="..." 먼저 설정하세요.')

BASE_URL    = "https://api.data.go.kr/openapi/tn_pubr_public_acdnt_area_api"
BASE_DIR    = Path.home() / "dev/졸업논문/999_pilot"
OUTPUT_CSV  = BASE_DIR / "03_data/05_사고다발지역_부산.csv"
DEBUG_JSON  = BASE_DIR / "03_data/05_API_응답샘플.json"

YEARS     = list(range(2010, 2024))   # 2010~2023, NODATA 연도는 자동 스킵
PAGE_SIZE = 1000
TIMEOUT   = 90    # 30 → 90초
MAX_RETRY = 3


# ─── HTTP with retry ─────────────────────────
def get_with_retry(params, max_retry=MAX_RETRY):
    last_err = None
    for attempt in range(1, max_retry + 1):
        try:
            r = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
            return r.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            wait = 2 * attempt
            print(f"      ⚠ timeout/connection error (시도 {attempt}/{max_retry}), {wait}초 후 재시도")
            time.sleep(wait)
        except Exception as e:
            last_err = e
            break
    raise last_err


# ─── Phase 1: 테스트 ──────────────────────────
def test_one_call(year=2022):
    print(f"\n[테스트] year={year} 한 번 호출")
    params = {
        "serviceKey": SERVICE_KEY,
        "pageNo": 1,
        "numOfRows": 10,
        "type": "json",
        "ACDNT_YEAR": year,
    }
    try:
        data = get_with_retry(params)
    except Exception as e:
        print(f"  ❌ 요청 실패: {e}")
        return False

    import json
    DEBUG_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    resp = data.get("response", {})
    header = resp.get("header", {})
    body = resp.get("body", {})
    code = str(header.get("resultCode", ""))
    msg  = header.get("resultMsg", "")
    print(f"  HTTP 200  resultCode={code}  resultMsg={msg}")

    if code not in ("00", "0", ""):
        print(f"  ❌ API 오류")
        return False

    items = body.get("items", [])
    if isinstance(items, dict):
        items = items.get("item", [])
    if not isinstance(items, list):
        items = []

    print(f"  이번 페이지: {len(items)}건")
    if items:
        print(f"  키: {list(items[0].keys())}")
        sample = {k: items[0].get(k) for k in [
            "ctprvnSignguNm", "acdntAreaLcNm", "acdntYear",
            "acdntTypeSe", "latitude", "longitude",
            "occrrncCo", "casltCo", "deathCo",
        ] if k in items[0]}
        print(f"\n  [첫 row 미리보기]")
        for k, v in sample.items():
            print(f"    {k}: {v}")

    return True


# ─── Phase 2: 연도별 수집 + 부산 필터 ─────────
def fetch_year(year):
    busan_items = []
    page = 1
    while True:
        params = {
            "serviceKey": SERVICE_KEY,
            "pageNo": page,
            "numOfRows": PAGE_SIZE,
            "type": "json",
            "ACDNT_YEAR": year,
        }
        try:
            data = get_with_retry(params)
        except Exception as e:
            print(f"  ✗ page {page} 최종 실패: {e}")
            break

        resp = data.get("response", {})
        header = resp.get("header", {})
        body = resp.get("body", {})
        code = str(header.get("resultCode", ""))
        if code not in ("00", "0", ""):
            if code == "03":
                print(f"  · 데이터 없음 (NODATA)")
            else:
                print(f"  ✗ API 오류: code={code}, msg={header.get('resultMsg', '')}")
            break

        items = body.get("items", [])
        if isinstance(items, dict):
            items = items.get("item", [])
        if not isinstance(items, list):
            items = []
        if not items:
            break

        # 부산 필터 (camelCase)
        page_busan = [
            it for it in items
            if "부산" in str(it.get("ctprvnSignguNm", ""))
        ]
        busan_items.extend(page_busan)
        print(f"  page {page}: 전국 {len(items):>4d}건 / 부산 +{len(page_busan):>3d}")

        if len(items) < PAGE_SIZE:
            break
        page += 1
        time.sleep(0.5)

    return busan_items


# ─── 메인 ────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(" Phase 1: API 테스트")
    print("=" * 60)
    if not test_one_call(year=2022):
        sys.exit(1)

    print("\n✓ 테스트 통과. 전체 수집 시작...\n")
    time.sleep(1)

    print("=" * 60)
    print(f" Phase 2: 부산 {YEARS[0]}~{YEARS[-1]} 수집")
    print("=" * 60)

    all_rows = []
    available_years = []
    for year in YEARS:
        print(f"\n[{year}]")
        rows = fetch_year(year)
        all_rows.extend(rows)
        if rows:
            available_years.append(year)
        print(f"  → {year} 부산 {len(rows)}건 (누적 {len(all_rows)})")

    if not all_rows:
        print("\n⚠ 수집된 부산 데이터 0건. 디버그 JSON 확인:")
        print(f"  {DEBUG_JSON}")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"수집 완료: 총 {len(df)}건")
    print(f"데이터 있는 연도: {available_years}")
    print(f"저장: {OUTPUT_CSV}")
    print("=" * 60)

    print(f"\n[연도별 분포]")
    print(df["acdntYear"].value_counts().sort_index().to_string())

    print(f"\n[시군구 분포]")
    print(df["ctprvnSignguNm"].value_counts().to_string())

    print(f"\n[샘플 3건]")
    cols = ["acdntYear", "ctprvnSignguNm", "acdntAreaLcNm",
            "latitude", "longitude", "occrrncCo", "deathCo"]
    show = [c for c in cols if c in df.columns]
    print(df[show].head(3).to_string())
