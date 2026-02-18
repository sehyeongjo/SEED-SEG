import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import argparse
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm


# -------------------------------------------------
# 1. 유틸: BGR uint8 변환
# -------------------------------------------------
def to_bgr_u8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        if img.shape[2] == 4:
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            bgr = img.copy()

    if bgr.dtype != np.uint8:
        bgr = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return bgr


# -------------------------------------------------
# 2. 회전 (잘림 방지)
# -------------------------------------------------
def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


# -------------------------------------------------
# 3. 각도 계산 (공통 로직)
# -------------------------------------------------
def calculate_angle(gray_img: np.ndarray) -> float:
    gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    angle_rad = 0.5 * np.arctan2(
        2 * np.mean(gx * gy),
        np.mean(gx**2 - gy**2)
    )

    return np.degrees(angle_rad)


# -------------------------------------------------
# 4. 시각화
# -------------------------------------------------
def draw_visualization(img_bgr, angle_deg, save_path, note=""):
    h, w = img_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    length = min(w, h) // 3

    angle = np.radians(angle_deg)

    # 기준 축
    cv2.line(img_bgr, (cx - length, cy), (cx + length, cy), (200, 200, 200), 1)
    cv2.line(img_bgr, (cx, cy - length), (cx, cy + length), (200, 200, 200), 1)

    # 기울기 축
    x1 = int(cx + length * np.cos(angle))
    y1 = int(cy + length * np.sin(angle))
    x2 = int(cx - length * np.cos(angle))
    y2 = int(cy - length * np.sin(angle))
    cv2.line(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # 수직 축
    perp = angle + np.pi/2
    x3 = int(cx + length * np.cos(perp))
    y3 = int(cy + length * np.sin(perp))
    x4 = int(cx - length * np.cos(perp))
    y4 = int(cy - length * np.sin(perp))
    cv2.line(img_bgr, (x3, y3), (x4, y4), (255, 0, 0), 3)

    cv2.circle(img_bgr, (cx, cy), 6, (0, 255, 0), -1)

    text = f"Angle: {angle_deg:.2f} deg"
    if note:
        text += f" ({note})"

    cv2.putText(img_bgr, text, (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 0, 0), 5)
    cv2.putText(img_bgr, text, (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (255, 255, 255), 2)

    save_path = save_path.with_suffix(".png")
    cv2.imwrite(str(save_path), img_bgr)


# -------------------------------------------------
# 5. 메인 처리
# -------------------------------------------------
def process_tray(tray_num, manual_angle=None, sample_visual=5):
    tray_str = str(tray_num)
    base_input_dir = Path(f"../project/tray/{tray_str}")

    if not base_input_dir.exists():
        print("입력 폴더 없음")
        return

    # 출력 구조
    json_dir = Path("slope_output/json")
    # vis_dir = Path(f"slope_output/visual/tray/{tray_str}")
    base_vis_dir = Path(f"slope_output/visual/tray/{tray_str}")

    if manual_angle is None:
        vis_dir = base_vis_dir / "auto"
    else:
        vis_dir = base_vis_dir / "manual"


    if vis_dir.exists():
        shutil.rmtree(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    images = sorted([p for p in base_input_dir.glob("*") if p.suffix.lower() in valid_exts])

    if not images:
        print("이미지 없음")
        return

    print(f"총 {len(images)}장 처리 중...")

    if manual_angle is not None:
        print(f"[Manual Mode] tray={tray_str}, angle={manual_angle}")
        print("JSON 저장 안함. 샘플만 생성.")

        sample_count = min(sample_visual, len(images))
        selected = random.sample(images, sample_count)

        for img_path in selected:
            raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue

            img = to_bgr_u8(raw)
            img = rotate_image(img, manual_angle)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            angle = calculate_angle(gray)

            save_path = vis_dir / img_path.name
            draw_visualization(img, angle, save_path,
                            note=f"Manual Rot {manual_angle}")

        print("=== Manual 완료 ===")
        return



    angle_data = []

    for img_path in tqdm(images, desc="Calculating"):
        raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue

        img = to_bgr_u8(raw)

        if manual_angle is not None:
            img = rotate_image(img, manual_angle)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = calculate_angle(gray)

        angle_data.append({
            "filename": img_path.name,
            "angle": float(f"{angle:.4f}")
        })

    # 평균
    # avg_angle = sum(d["angle"] for d in angle_data) / len(angle_data)
    angles = np.array([d["angle"] for d in angle_data])

    median = np.median(angles)
    mad = np.median(np.abs(angles - median))

    # 너무 작은 MAD 방지
    if mad < 1:
        mad = 1

    threshold = 3 * mad

    filtered = angles[np.abs(angles - median) <= threshold]

    print(f"Median: {median:.2f}")
    print(f"MAD: {mad:.2f}")
    print(f"Threshold: ±{threshold:.2f}")
    print(f"{len(filtered)}/{len(angles)} used")

    avg_angle = np.mean(filtered)

    if avg_angle < 0:
        avg_angle += 90

    vertical_ccw = 90 - avg_angle
    vertical_cw  = -avg_angle

    json_output = {
        "tray_id": tray_str,
        "total_images": len(angle_data),
        "average_angle": float(f"{avg_angle:.4f}"),
        "align_vertical_ccw": float(f"{vertical_ccw:.4f}"),
        "align_vertical_cw": float(f"{vertical_cw:.4f}"),
        "manual_rotation": manual_angle,
        "files": angle_data
    }

    with open(json_dir / f"{tray_str}.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4)

    print(f"평균 각도: {avg_angle:.4f}")

    # -------------------------
    # 샘플 시각화
    # -------------------------
    sample_count = min(sample_visual, len(images))
    selected = random.sample(images, sample_count)

    for img_path in selected:
        raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img = to_bgr_u8(raw)

        note = "Original"
        if manual_angle is not None:
            img = rotate_image(img, manual_angle)
            note = f"Rotated {manual_angle}"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = calculate_angle(gray)

        save_path = vis_dir / img_path.name
        draw_visualization(img, angle, save_path, note)

    print("=== 완료 ===")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tray", type=str, required=True)
    parser.add_argument("--angle", type=float, default=None)
    parser.add_argument("--sample", type=int, default=3)

    args = parser.parse_args()

    process_tray(
        tray_num=args.tray,
        manual_angle=args.angle,
        sample_visual=args.sample
    )