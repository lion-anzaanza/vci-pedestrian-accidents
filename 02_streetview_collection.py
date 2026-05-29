"""
09_streetview_수집.py
─────────────────────────────────────────────
Day 2: Street View 이미지 수집 (30개 교차로 × 4방향 = 120장)

각 교차로마다:
  1. metadata 체크 (free) - Street View 존재하는지
  2. 존재하면 4방향(N/E/S/W) 이미지 다운로드
  3. 저장: streetview/{intersection_id}/{N|E|S|W}.jpg
  4. 중복 다운로드 방지 (이미 있으면 스킵)

API 키 설정:
  export GOOGLE_API_KEY="..."  ← 셸에서 먼저 설정
  python 09_streetview_수집.py

입력: 999_pilot/08_표본_30개.csv
출력:
  · 999_pilot/streetview/{intersection_id}/{N|E|S|W}.jpg
  · 999_pilot/09_streetview_status.csv
"""

import os
import time
import pandas as pd
import requests
from pathlib import Path

# ─── 설정 ────────────────────────────────────
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit(
        "❌ GOOGLE_API_KEY 환경변수가 없습니다.\n"
        '   터미널에서: export GOOGLE_API_KEY="YOUR_KEY_HERE"\n'
        "   그 다음 다시 실행하세요."
    )

BASE_DIR   = Path.home() / "dev/졸업논문/999_pilot"
SAMPLE_CSV = BASE_DIR / "03_data/01_표본_30개.csv"
OUTPUT_DIR = BASE_DIR / "01_raw/streetview"
STATUS_CSV = BASE_DIR / "03_data/02_streetview_status.csv"

HEADINGS   = {"N": 0, "E": 90, "S": 180, "W": 270}
IMAGE_SIZE = "640x640"   # free tier 최대
FOV        = 90          # 일반 시야각
PITCH      = 0           # 수평

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL    = "https://maps.googleapis.com/maps/api/streetview"

# ─── 표본 로드 ───────────────────────────────
sample = pd.read_csv(SAMPLE_CSV)
print(f"표본 교차로: {len(sample)}개")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 다운로드 루프 ───────────────────────────
status_records = []

for idx, row in sample.iterrows():
    iid = int(row["intersection_id"])
    lat = row["lat"]
    lon = row["lon"]
    name = row.get("MOCT_NODE_NAME", "")
    gu   = row.get("구_이름", "")

    print(f"\n[{idx+1}/{len(sample)}] id={iid} {gu} {name}")

    # 1. metadata 체크 (무료, Street View 존재 확인)
    try:
        meta_resp = requests.get(METADATA_URL, params={
            "location": f"{lat},{lon}",
            "key": API_KEY,
        }, timeout=10)
        meta = meta_resp.json()
        status = meta.get("status", "UNKNOWN")
        pano_date = meta.get("date", "")
    except Exception as e:
        print(f"  ⚠ metadata 요청 실패: {e}")
        status, pano_date = "ERROR", ""

    if status != "OK":
        print(f"  ⚠ Street View 없음 (status={status}), 건너뜀")
        status_records.append({
            "intersection_id": iid,
            "metadata_status": status,
            "pano_date": pano_date,
            "N": False, "E": False, "S": False, "W": False,
        })
        continue

    print(f"  ✓ metadata OK (촬영일: {pano_date})")

    # 2. 4방향 이미지 다운로드
    inter_dir = OUTPUT_DIR / str(iid)
    inter_dir.mkdir(exist_ok=True)

    record = {
        "intersection_id": iid,
        "metadata_status": status,
        "pano_date": pano_date,
    }

    for direction, heading in HEADINGS.items():
        img_path = inter_dir / f"{direction}.jpg"

        # 중복 다운로드 방지
        if img_path.exists() and img_path.stat().st_size > 5000:
            print(f"    · {direction} 이미 있음, 스킵")
            record[direction] = True
            continue

        try:
            resp = requests.get(IMAGE_URL, params={
                "size": IMAGE_SIZE,
                "location": f"{lat},{lon}",
                "heading": heading,
                "fov": FOV,
                "pitch": PITCH,
                "key": API_KEY,
            }, timeout=15)

            if resp.status_code == 200 and len(resp.content) > 5000:
                img_path.write_bytes(resp.content)
                print(f"    ✓ {direction} ({heading:>3}°) → {len(resp.content)//1024:>4} KB")
                record[direction] = True
            else:
                print(f"    ✗ {direction} 실패 (HTTP {resp.status_code}, {len(resp.content)} bytes)")
                record[direction] = False
        except Exception as e:
            print(f"    ✗ {direction} 예외: {e}")
            record[direction] = False

        time.sleep(0.1)   # rate limit 회피

    status_records.append(record)

# ─── 상태 저장 ───────────────────────────────
status_df = pd.DataFrame(status_records)
status_df.to_csv(STATUS_CSV, index=False, encoding="utf-8-sig")

# ─── 요약 ────────────────────────────────────
total_attempted = len(sample) * 4
total_downloaded = sum(
    sum(r.get(d, False) for d in HEADINGS.keys())
    for r in status_records
)
sv_available = sum(1 for r in status_records if r["metadata_status"] == "OK")

print("\n" + "=" * 60)
print(f"교차로 중 Street View 가능: {sv_available}/{len(sample)}")
print(f"이미지 다운로드           : {total_downloaded}/{total_attempted}장")
print(f"상태 CSV : {STATUS_CSV}")
print(f"이미지   : {OUTPUT_DIR}")
print("=" * 60)
