"""
10_segmentation.py
─────────────────────────────────────────────
Day 3: Mask2Former Cityscapes panoptic segmentation

112장 Street View 이미지 → semantic segmentation map
각 이미지마다 19개 Cityscapes 클래스의 픽셀 비율 계산

입력: 999_pilot/streetview/{iid}/{N|E|S|W}.jpg
출력:
  · 999_pilot/segmentation/{iid}/{direction}_seg.npy   ← raw seg map
  · 999_pilot/segmentation/{iid}/{direction}_viz.png   ← 시각화
  · 999_pilot/10_segmentation_stats.csv                 ← 클래스별 픽셀 비율

사전 설치:
  pip install transformers torch torchvision pillow
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

# ─── 설정 ────────────────────────────────────
BASE_DIR  = Path.home() / "dev/졸업논문/999_pilot"
SV_DIR    = BASE_DIR / "01_raw/streetview"
SEG_DIR   = BASE_DIR / "01_raw/segmentation"
STATS_CSV = BASE_DIR / "03_data/03_segmentation_stats.csv"

# Cityscapes panoptic - 도로 장면 특화 모델
MODEL_NAME = "facebook/mask2former-swin-base-IN21k-cityscapes-panoptic"

# 디바이스 자동 선택
if torch.backends.mps.is_available():
    DEVICE = "mps"   # Apple Silicon Mac (M1/M2/M3)
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"디바이스: {DEVICE}")

# ─── 모델 로드 ───────────────────────────────
print(f"모델 로딩: {MODEL_NAME}")
print("(첫 실행 시 모델 다운로드 ~250MB)")
processor = Mask2FormerImageProcessor.from_pretrained(MODEL_NAME)
model = (
    Mask2FormerForUniversalSegmentation
    .from_pretrained(MODEL_NAME)
    .to(DEVICE)
    .eval()
)
id2label = model.config.id2label
print(f"클래스 수: {len(id2label)}")
print(f"클래스: {list(id2label.values())}")

# ─── Cityscapes 시각화 컬러 ──────────────────
CITYSCAPES_COLORS = np.array([
    [128, 64, 128],   # 0: road
    [244, 35, 232],   # 1: sidewalk
    [70, 70, 70],     # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170, 30],   # 6: traffic light
    [220, 220, 0],    # 7: traffic sign
    [107, 142, 35],   # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [70, 130, 180],   # 10: sky
    [220, 20, 60],    # 11: person
    [255, 0, 0],      # 12: rider
    [0, 0, 142],      # 13: car
    [0, 0, 70],       # 14: truck
    [0, 60, 100],     # 15: bus
    [0, 80, 100],     # 16: train
    [0, 0, 230],      # 17: motorcycle
    [119, 11, 32],    # 18: bicycle
], dtype=np.uint8)


# ─── 함수 정의 ───────────────────────────────
@torch.no_grad()
def process_image(img_path):
    """이미지를 semantic segmentation map (HxW with class IDs)으로 변환"""
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    sem_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0].cpu().numpy()
    return sem_map


def class_pixel_ratios(sem_map):
    """클래스별 픽셀 비율 dict 반환 (라벨 이름 키)"""
    total = sem_map.size
    unique, counts = np.unique(sem_map, return_counts=True)
    return {id2label[int(c)]: cnt / total for c, cnt in zip(unique, counts)}


def visualize(sem_map, save_path):
    """semantic map을 컬러 PNG로 저장"""
    viz = CITYSCAPES_COLORS[sem_map % len(CITYSCAPES_COLORS)]
    Image.fromarray(viz.astype(np.uint8)).save(save_path)


# ─── 메인 루프 ───────────────────────────────
SEG_DIR.mkdir(exist_ok=True)
stats_rows = []

intersection_dirs = sorted([d for d in SV_DIR.iterdir() if d.is_dir()])
print(f"\n교차로 폴더 수: {len(intersection_dirs)}")

n_processed = 0
for inter_dir in intersection_dirs:
    iid = inter_dir.name
    out_dir = SEG_DIR / iid
    out_dir.mkdir(exist_ok=True)

    print(f"\n[{iid}]", end=" ")

    for img_path in sorted(inter_dir.glob("*.jpg")):
        direction = img_path.stem
        npy_path = out_dir / f"{direction}_seg.npy"
        viz_path = out_dir / f"{direction}_viz.png"

        if npy_path.exists():
            sem_map = np.load(npy_path)
            print(f"{direction}(캐시)", end=" ")
        else:
            sem_map = process_image(img_path)
            np.save(npy_path, sem_map)
            visualize(sem_map, viz_path)
            print(f"{direction}✓", end=" ")

        ratios = class_pixel_ratios(sem_map)
        row = {"intersection_id": int(iid), "direction": direction}
        row.update(ratios)
        stats_rows.append(row)
        n_processed += 1

# ─── 통계 저장 ───────────────────────────────
stats_df = pd.DataFrame(stats_rows).fillna(0)
class_cols = [c for c in stats_df.columns if c not in ["intersection_id", "direction"]]
stats_df = stats_df[["intersection_id", "direction"] + sorted(class_cols)]
stats_df.to_csv(STATS_CSV, index=False, encoding="utf-8-sig")

print(f"\n\n{'=' * 60}")
print(f"완료: {n_processed}장 처리")
print(f"통계 CSV : {STATS_CSV}")
print(f"세그멘테이션 폴더: {SEG_DIR}")
print(f"\n[전체 클래스별 평균 픽셀 비율 (모든 이미지)]")
mean_ratios = stats_df[class_cols].mean().sort_values(ascending=False)
for label, ratio in mean_ratios.head(10).items():
    print(f"  {label:20s}  {ratio*100:5.1f}%")
print(f"{'=' * 60}")
