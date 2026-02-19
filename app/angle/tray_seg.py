import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------
# For Orientation Check (Status : 0)
# 
# For Build Template (Status : 1)
# python axis_seg_test_v22.py \
#   --mode build_template \
#   --tray_num {} \
#   --manual_angle {*}
# 
# For Process (Status : 2)
# python axis_seg_test_v22.py \
#   --mode process \
#   --tray_num {} \
#   --manual_angle {*}
# ---------------------------

# ---------------------------
# Utils
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_bgr_u8(img: np.ndarray) -> np.ndarray:
    """시각화/저장용: BGR uint8로 맞춤."""
    if img is None:
        raise ValueError("img is None")
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


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """검출용: GRAY uint8로 맞춤."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

# ---------------------------
# Orientation
# ---------------------------
def rotate_keep_all(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def estimate_rotation_to_axis(gray_u8: np.ndarray) -> float:
    """
    희미한 격자에서도 잘 잡히도록:
    CLAHE + Top-hat + Canny + HoughLines -> 수평/수직 축으로 정렬.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray_u8)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    g = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)

    p99 = float(np.percentile(g, 99))
    if p99 < 5:
        return 0.0

    low = int(max(1, 0.30 * p99))
    high = int(min(255, 0.90 * p99))
    edges = cv2.Canny(g, low, high, L2gradient=True)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)
    if lines is None or len(lines) < 10:
        return 0.0

    angles = []
    for rho, theta in lines[:, 0, :]:
        line_ang = (np.degrees(theta) - 90) % 180
        angles.append(line_ang)
    angles = np.array(angles, dtype=np.float32)

    hist, bin_edges = np.histogram(angles, bins=180, range=(0, 180))
    peak_idx = int(np.argmax(hist))
    peak_deg = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0

    peak_mod90 = peak_deg % 90.0
    if peak_mod90 <= 45:
        angle_deg = -peak_mod90
    else:
        angle_deg = (90.0 - peak_mod90)

    angle_deg = ((angle_deg + 45.0) % 90.0) - 45.0
    return float(angle_deg)


# ---------------------------
# Grid ROI detection
# ---------------------------
def find_grid_roi_bbox(rot_gray_u8: np.ndarray) -> tuple[int, int, int, int]:
    """
    orientation된 그레이 이미지에서 '가운데 사각 그리드(큰 사각형)' bbox 찾기.
    """
    H, W = rot_gray_u8.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(rot_gray_u8)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    enh = cv2.max(tophat, blackhat)

    th = cv2.adaptiveThreshold(
        enh, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, -2
    )

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_k, iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Grid ROI detection failed: no contours.")

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.03 * (H * W):
            continue
        if area > 0.90 * (H * W):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / (h + 1e-6)
        ar_ok = (0.8 <= ar <= 6.5)

        extent = area / (w * h + 1e-6)
        if not ar_ok:
            continue

        cx, cy = x + w / 2, y + h / 2
        dist_center = np.hypot(cx - W / 2, cy - H / 2) / (np.hypot(W, H) + 1e-6)
        center_bonus = 1.0 - dist_center

        score = (area / (H * W)) * 2.0 + extent + center_bonus
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        raise RuntimeError("Grid ROI detection failed: no suitable rectangle found.")

    return best


# ---------------------------
# Labeling: ALWAYS top-left=A_1, bottom-right=C_10
# ---------------------------
def assign_labels_by_sorting(temp_results, num_rows=10, num_cols=3):
    """
    30개 박스 기준으로:
    - center_y로 정렬 -> 위에서부터 3개씩 끊어 10개 row
    - 각 row에서 center_x로 정렬 -> A,B,C
    """
    assert len(temp_results) == num_rows * num_cols, "temp_results must be exactly 30 boxes"

    centers = []
    for i, item in enumerate(temp_results):
        x, y, w, h = item["box"]
        centers.append((i, x + w / 2.0, y + h / 2.0))  # (idx, cx, cy)

    centers.sort(key=lambda t: t[2])  # cy asc (top->bottom)

    col_names = ["A", "B", "C"][:num_cols]
    labels = [""] * len(temp_results)

    for r in range(num_rows):
        chunk = centers[r * num_cols:(r + 1) * num_cols]
        chunk.sort(key=lambda t: t[1])  # cx asc (left->right)

        for c, (idx, cx, cy) in enumerate(chunk):
            labels[idx] = f"{col_names[c]}_{r+1}"

    return labels


def all_labels(num_rows=10, num_cols=3):
    col_names = ["A", "B", "C"][:num_cols]
    out = []
    for c in range(num_cols):
        for r in range(1, num_rows + 1):
            out.append(f"{col_names[c]}_{r}")
    return out


# ---------------------------
# Segmentation inside ROI (connected components)
# ---------------------------
def segment_boxes_in_roi(
    roi_bgr_u8: np.ndarray,
    roi_gray_u8: np.ndarray,
    expected_boxes: int = 30,
    min_area: int = 300,
    extent_threshold: float = 0.15,
):
    """
    ROI 내에서만 threshold->CC로 박스 후보 잡기.
    반환:
      - temp_results: list[{"mask","box","area","extent"}] (최대 expected_boxes)
      - debug_bw: binary image
    """
    _, bw = cv2.threshold(roi_gray_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    candidates = []
    for i in range(1, n):
        bx, by, bw_, bh_, area = stats[i]
        if area < min_area:
            continue

        extent = float(area) / (bw_ * bh_ + 1e-6)
        if extent < extent_threshold:
            continue

        temp_m = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(temp_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        filled_mask = np.zeros_like(temp_m)
        cv2.drawContours(filled_mask, [cnt], -1, 255, -1)

        candidates.append({
            "mask": filled_mask,              # ROI size
            "box": (bx, by, bw_, bh_),        # ROI coords
            "area": int(area),
            "extent": float(extent)
        })

    if len(candidates) == 0:
        return [], bw

    # robust filter to choose expected_boxes
    areas = np.array([c["area"] for c in candidates], dtype=np.float32)
    med_area = float(np.median(areas))
    mad_area = float(np.median(np.abs(areas - med_area))) + 1e-6

    k = 2.5
    kept_idx = None
    while k <= 10.0:
        idx = np.where(np.abs(areas - med_area) <= k * mad_area)[0]
        if len(idx) >= expected_boxes:
            kept_idx = idx
            break
        k += 0.5
    if kept_idx is None:
        kept_idx = np.arange(len(candidates))

    kept = [candidates[i] for i in kept_idx]
    kept.sort(key=lambda c: abs(c["area"] - med_area))

    temp_results = kept[:min(expected_boxes, len(kept))]
    return temp_results, bw


# ---------------------------
# Template (build + apply)
# ---------------------------
def init_template_acc():
    return defaultdict(lambda: {"sum": None, "count": 0})


def add_to_template_acc(template_acc, label: str, mask01: np.ndarray):
    """
    mask01: uint8 0/1, ROI fixed size
    """
    if template_acc[label]["sum"] is None:
        template_acc[label]["sum"] = mask01.astype(np.float32)
    else:
        template_acc[label]["sum"] += mask01.astype(np.float32)
    template_acc[label]["count"] += 1


def finalize_and_save_template(template_acc, template_dir: Path, roi_w: int, roi_h: int, thresh: float = 0.5):
    ensure_dir(template_dir)

    masks = {}
    meta = {
        "roi_w": roi_w,
        "roi_h": roi_h,
        "thresh": thresh,
        "counts": {}
    }

    for label in all_labels():
        entry = template_acc.get(label, None)
        if entry is None or entry["sum"] is None or entry["count"] == 0:
            # 없는 건 빈 mask로
            final = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cnt = 0
        else:
            mean = entry["sum"] / float(entry["count"])
            final = (mean >= thresh).astype(np.uint8) * 255
            cnt = int(entry["count"])

        masks[label] = final
        meta["counts"][label] = cnt
        cv2.imwrite(str(template_dir / f"{label}.png"), final)

    # npz로 저장 (필수)
    np.savez_compressed(str(template_dir / "template_masks.npz"), **masks)

    # meta 저장
    with open(template_dir / "template_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] template saved: {template_dir}")
    return masks, meta


def load_template(template_dir: Path):
    npz_path = template_dir / "template_masks.npz"
    meta_path = template_dir / "template_meta.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing template: {npz_path}")

    data = np.load(str(npz_path))
    masks = {k: data[k].astype(np.uint8) for k in data.files}

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # roi size는 meta 우선, 없으면 mask에서 추정
    if "roi_w" in meta and "roi_h" in meta:
        roi_w = int(meta["roi_w"])
        roi_h = int(meta["roi_h"])
    else:
        any_mask = next(iter(masks.values()))
        roi_h, roi_w = any_mask.shape[:2]

    return masks, roi_w, roi_h, meta


def overlay_and_save_from_masks(
    roi_bgr_rs: np.ndarray,
    label_to_mask: dict,
    out_cells_dir: Path,
    overlay_alpha: float = 0.45,
    seed: int = 0,
    *,
    # --- saving in original ratio (aspect) ---
    roi_bgr_orig: np.ndarray | None = None,
    roi_size_rs: tuple[int, int] | None = None,   # (w_rs, h_rs)
):
    """
    label_to_mask: masks aligned to resized ROI (roi_bgr_rs size).
    저장 시 원본 비율을 유지하려면 roi_bgr_orig를 같이 넘겨줘야 함.

    반환:
      - overlay_vis_rs: resized-ROI overlay (same size as roi_bgr_rs)
      - overlay_vis_orig: original-ROI overlay (same size as roi_bgr_orig) or None
    """
    ensure_dir(out_cells_dir)
    rng = np.random.default_rng(seed)

    h_rs, w_rs = roi_bgr_rs.shape[:2]
    if roi_size_rs is None:
        roi_size_rs = (w_rs, h_rs)
    w_rs, h_rs = roi_size_rs

    overlay_layer_rs = np.zeros_like(roi_bgr_rs, dtype=np.uint8)
    overlay_vis_rs = roi_bgr_rs.copy()

    # If original ROI is provided, prepare orig overlay as well
    overlay_vis_orig = None
    overlay_layer_orig = None
    if roi_bgr_orig is not None:
        overlay_vis_orig = roi_bgr_orig.copy()
        overlay_layer_orig = np.zeros_like(roi_bgr_orig, dtype=np.uint8)
        h0, w0 = roi_bgr_orig.shape[:2]
        sx = w0 / float(w_rs)
        sy = h0 / float(h_rs)

    for label in sorted(label_to_mask.keys()):
        m = label_to_mask[label]
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)

        mask01 = (m > 0).astype(np.uint8)
        if mask01.sum() == 0:
            continue

        color = rng.integers(50, 256, size=3, dtype=np.uint8)

        # ---- resized overlay ----
        overlay_layer_rs[mask01 > 0] = color

        ys, xs = np.where(mask01 > 0)
        x0_rs, x1_rs = int(xs.min()), int(xs.max())
        y0_rs, y1_rs = int(ys.min()), int(ys.max())

        # ---- crop saving (ORIGINAL ratio) ----
        if roi_bgr_orig is not None:
            # map bbox back to original ROI coordinates
            x0_o = int(np.floor(x0_rs * sx))
            x1_o = int(np.ceil((x1_rs + 1) * sx)) - 1
            y0_o = int(np.floor(y0_rs * sy))
            y1_o = int(np.ceil((y1_rs + 1) * sy)) - 1

            expand_ratio = 0.05

            box_w = x1_o - x0_o
            box_h = y1_o - y0_o

            dx = int(box_w * expand_ratio)
            dy = int(box_h * expand_ratio)

            x0_o -= dx
            x1_o += dx
            y0_o -= dy
            y1_o += dy

            h0, w0 = roi_bgr_orig.shape[:2]
            x0_o = max(0, min(w0 - 1, x0_o))
            x1_o = max(0, min(w0 - 1, x1_o))
            y0_o = max(0, min(h0 - 1, y0_o))
            y1_o = max(0, min(h0 - 1, y1_o))

            crop = roi_bgr_orig[y0_o:y1_o + 1, x0_o:x1_o + 1]
            cv2.imwrite(str(out_cells_dir / f"{label}.png"), crop)

            # orig overlay + label
            # (mask -> orig by nearest resize)
            m_orig = cv2.resize(mask01 * 255, (w0, h0), interpolation=cv2.INTER_NEAREST)
            overlay_layer_orig[m_orig > 0] = color

            tx, ty = int(x0_o + 5), int(y0_o + 30)
            cv2.putText(overlay_vis_orig, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(overlay_vis_orig, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 3, cv2.LINE_AA)
        else:
            # fallback: save crop from resized ROI (old behavior)
            crop = roi_bgr_rs[y0_rs:y1_rs + 1, x0_rs:x1_rs + 1]
            cv2.imwrite(str(out_cells_dir / f"{label}.png"), crop)

        # resized label text (for debug)
        tx, ty = int(x0_rs + 5), int(y0_rs + 30)
        cv2.putText(overlay_vis_rs, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(overlay_vis_rs, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3, cv2.LINE_AA)

    overlay_vis_rs = cv2.addWeighted(overlay_vis_rs, 1.0, overlay_layer_rs, overlay_alpha, 0)

    if roi_bgr_orig is not None:
        overlay_vis_orig = cv2.addWeighted(overlay_vis_orig, 1.0, overlay_layer_orig, overlay_alpha, 0)

    return overlay_vis_rs, overlay_vis_orig


# ---------------------------
# Core per-image pipeline
# ---------------------------
def process_one_image(
    img_path: Path,
    output_root: Path,
    expected_boxes: int = 30,
    num_rows: int = 10,
    num_cols: int = 3,
    roi_margin: int = 6,
    roi_w: int = 900,
    roi_h: int = 2400,
    min_area: int = 300,
    extent_threshold: float = 0.15,
    overlay_alpha: float = 0.45,
    seed: int = 42,
    template_masks: dict | None = None,
    use_template_if_missing: bool = True,
    template_only: bool = False,
    manual_angle: float = 0.0,
    out_dir_suffix: str = "_output",
    should_stop=None,
):
    """
    요구사항 반영:
      1) 내부 process(ROI resize 등)는 그대로 두되, 저장되는 이미지는 원본 ratio로 저장
         - cells/*.png는 원본 ROI에서 crop해서 저장
         - overlay도 원본 ROI 버전으로 저장(추가로 resized 디버그도 저장)
      2) template_only=True인 경우에만 template만 사용하고 segmentation을 건너뜀

    반환:
      - success_30: bool (30개 detection 성공)
      - labels/temp_results/roi_resized_gray/roi_resized_bgr (template build에 필요)
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    if should_stop is not None and should_stop():
        raise RuntimeError("Cancelled by user")

    bgr_u8 = to_bgr_u8(img)
    gray_u8 = to_gray_u8(img)

    # # 1) orientation
    # angle = estimate_rotation_to_axis(gray_u8)
    # rot_bgr = rotate_keep_all(bgr_u8, angle)
    # rot_gray = to_gray_u8(rot_bgr)

    # 1) MANUAL orientation only
    angle = manual_angle
    rot_bgr = rotate_keep_all(bgr_u8, angle)
    rot_gray = to_gray_u8(rot_bgr)
    if should_stop is not None and should_stop():
        raise RuntimeError("Cancelled by user")

    # 2) grid ROI bbox
    x, y, w, h = find_grid_roi_bbox(rot_gray)

    x0 = max(0, x + roi_margin)
    y0 = max(0, y + roi_margin)
    x1 = min(rot_gray.shape[1], x + w - roi_margin)
    y1 = min(rot_gray.shape[0], y + h - roi_margin)

    roi_bgr = rot_bgr[y0:y1, x0:x1]   # <-- ORIGINAL ratio ROI
    roi_gray = rot_gray[y0:y1, x0:x1]

    # 3) ROI resize to fixed size (template alignment / 내부 process 유지)
    roi_bgr_rs = cv2.resize(roi_bgr, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    roi_gray_rs = cv2.resize(roi_gray, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    stem = img_path.stem
    out_dir = output_root / f"{stem}{out_dir_suffix}"
    out_cells = out_dir / "cells"
    out_debug = out_dir / "debug"
    ensure_dir(out_cells)
    ensure_dir(out_debug)

    # debug saves
    cv2.imwrite(str(out_debug / f"{stem}_rotated.png"), rot_bgr)

    rot_dbg = rot_bgr.copy()
    cv2.rectangle(rot_dbg, (x0, y0), (x1, y1), (0, 255, 255), 3)
    cv2.imwrite(str(out_debug / f"{stem}_rotated_with_roi.png"), rot_dbg)

    # 원본 ratio ROI도 저장 (요구사항)
    cv2.imwrite(str(out_debug / f"{stem}_roi_original.png"), roi_bgr)

    # 기존 디버그(내부 process 확인용)도 남겨둠
    cv2.imwrite(str(out_debug / f"{stem}_roi_resized.png"), roi_bgr_rs)

    # -------------------------
    # 4) segmentation OR template-only
    # -------------------------
    temp_results = []
    bw = None

    if template_only:
        # process 단계: template만 사용 (segmentation 수행 안 함)
        if template_masks is None:
            raise RuntimeError("template_only=True but template_masks is None. Provide --template_dir.")
    else:
        # build_template 단계 or 옵션상 segmentation 수행
        scale = (roi_w * roi_h) / float(900 * 2400)
        min_area_scaled = int(max(50, min_area * scale))

        temp_results, bw = segment_boxes_in_roi(
            roi_bgr_rs,
            roi_gray_rs,
            expected_boxes=expected_boxes,
            min_area=min_area_scaled,
            extent_threshold=extent_threshold,
        )
        if should_stop is not None and should_stop():
            raise RuntimeError("Cancelled by user")
        cv2.imwrite(str(out_debug / f"{stem}_roi_bw.png"), bw)

    # -------------------------
    # 5) If 30 detected -> label & overlay
    # -------------------------
    if (not template_only) and (len(temp_results) == expected_boxes):
        labels = assign_labels_by_sorting(temp_results, num_rows=num_rows, num_cols=num_cols)

        label_to_mask = {label: item["mask"] for label, item in zip(labels, temp_results)}

        overlay_rs, overlay_orig = overlay_and_save_from_masks(
            roi_bgr_rs,
            label_to_mask=label_to_mask,
            out_cells_dir=out_cells,
            overlay_alpha=overlay_alpha,
            seed=seed,
            roi_bgr_orig=roi_bgr,                 # <-- original ratio crop/save
            roi_size_rs=(roi_w, roi_h),
        )
        cv2.imwrite(str(out_debug / f"{stem}_overlay_resized.png"), overlay_rs)
        if overlay_orig is not None:
            cv2.imwrite(str(out_debug / f"{stem}_overlay_original.png"), overlay_orig)

        return True, labels, temp_results, roi_gray_rs, roi_bgr_rs, out_dir

    # -------------------------
    # 6) template 적용 (process 단계에서는 항상 여기로 옴)
    # -------------------------
    if template_masks is not None and (use_template_if_missing or template_only):
        overlay_rs, overlay_orig = overlay_and_save_from_masks(
            roi_bgr_rs,
            label_to_mask=template_masks,
            out_cells_dir=out_cells,
            overlay_alpha=overlay_alpha,
            seed=seed,
            roi_bgr_orig=roi_bgr,                 # <-- original ratio crop/save
            roi_size_rs=(roi_w, roi_h),
        )
        cv2.imwrite(str(out_debug / f"{stem}_overlay_TEMPLATE_resized.png"), overlay_rs)
        if overlay_orig is not None:
            cv2.imwrite(str(out_debug / f"{stem}_overlay_TEMPLATE_original.png"), overlay_orig)

        with open(out_debug / f"{stem}_note.txt", "w", encoding="utf-8") as f:
            if template_only:
                f.write("Mode: TEMPLATE ONLY (no segmentation)\n")
            else:
                f.write(f"Detected boxes: {len(temp_results)} / {expected_boxes}\n")
            f.write("Used TEMPLATE.\n")

        return False, None, None, roi_gray_rs, roi_bgr_rs, out_dir

    # -------------------------
    # 7) template도 없고 부족하면, partial overlay만 저장
    # -------------------------
    if (bw is not None) and len(temp_results) > 0:
        rng = np.random.default_rng(seed)
        overlay_layer = np.zeros_like(roi_bgr_rs, dtype=np.uint8)
        for item in temp_results:
            color = rng.integers(50, 256, size=3, dtype=np.uint8)
            overlay_layer[item["mask"] > 0] = color
        overlay = cv2.addWeighted(roi_bgr_rs, 1.0, overlay_layer, overlay_alpha, 0)
        cv2.imwrite(str(out_debug / f"{stem}_overlay_PARTIAL_resized.png"), overlay)

    with open(out_debug / f"{stem}_note.txt", "w", encoding="utf-8") as f:
        f.write(f"Detected boxes: {len(temp_results)} / {expected_boxes}\n")
        f.write("No template applied.\n")

    return False, None, None, roi_gray_rs, roi_bgr_rs, out_dir

    # 7) template도 없고 부족하면, 그래도 현재 검출된 것만 overlay로 남김
    #    (나중에 템플릿 만든 뒤 다시 돌리면 됨)
    if len(temp_results) > 0:
        # 임시 라벨 없이 색만 덮기
        rng = np.random.default_rng(seed)
        overlay_layer = np.zeros_like(roi_bgr_rs, dtype=np.uint8)
        for item in temp_results:
            color = rng.integers(50, 256, size=3, dtype=np.uint8)
            overlay_layer[item["mask"] > 0] = color
        overlay = cv2.addWeighted(roi_bgr_rs, 1.0, overlay_layer, overlay_alpha, 0)
        cv2.imwrite(str(out_debug / f"{stem}_overlay_PARTIAL.png"), overlay)

    with open(out_debug / f"{stem}_note.txt", "w", encoding="utf-8") as f:
        f.write(f"Detected boxes: {len(temp_results)} / {expected_boxes}\n")
        f.write("No template applied.\n")

    return False, None, None, roi_gray_rs, roi_bgr_rs, out_dir


# ---------------------------
# Batch: build template + process all
# ---------------------------
def list_images(input_dir: Path, exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg")):
    files = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def build_template_from_dataset(
    input_dir: Path,
    template_dir: Path,
    output_root: Path,
    roi_w: int = 900,
    roi_h: int = 2400,
    expected_boxes: int = 30,
    num_rows: int = 10,
    num_cols: int = 3,
    roi_margin: int = 6,
    min_area: int = 300,
    extent_threshold: float = 0.15,
    overlay_alpha: float = 0.45,
    seed: int = 42,
    thresh: float = 0.5,
    manual_angle: float = 0.0,
    progress_callback=None,
    out_dir_suffix: str = "_output",
    should_stop=None,
):
    """
    1) dataset에서 30개 검출 성공한 이미지들만 모아서 template mask 생성
    2) template_dir에 저장
    """
    files = list_images(input_dir)
    if len(files) == 0:
        raise RuntimeError(f"No images found in {input_dir}")

    template_acc = init_template_acc()
    good = 0
    bad = 0

    total = len(files)
    if progress_callback is not None:
        progress_callback(done=0, total=total, good=0, bad=0, current="")

    for idx, img_path in enumerate(tqdm(files, desc="Building template (good=30 only)"), start=1):
        if should_stop is not None and should_stop():
            raise RuntimeError("Cancelled by user")
        try:
            ok, labels, temp_results, roi_gray_rs, roi_bgr_rs, out_dir = process_one_image(
                img_path=img_path,
                output_root=output_root,
                expected_boxes=expected_boxes,
                num_rows=num_rows,
                num_cols=num_cols,
                roi_margin=roi_margin,
                roi_w=roi_w,
                roi_h=roi_h,
                min_area=min_area,
                extent_threshold=extent_threshold,
                overlay_alpha=overlay_alpha,
                seed=seed,
                template_masks=None,
                use_template_if_missing=False,
                manual_angle=manual_angle,
                out_dir_suffix=out_dir_suffix,
                should_stop=should_stop,
            )
        except Exception as e:
            bad += 1
            if progress_callback is not None:
                progress_callback(done=idx, total=total, good=good, bad=bad, current=img_path.name)
            continue

        if not ok:
            bad += 1
            if progress_callback is not None:
                progress_callback(done=idx, total=total, good=good, bad=bad, current=img_path.name)
            continue

        # accumulate per label
        for label, item in zip(labels, temp_results):
            mask01 = (item["mask"] > 0).astype(np.uint8)  # 0/1
            add_to_template_acc(template_acc, label, mask01)

        good += 1
        if progress_callback is not None:
            progress_callback(done=idx, total=total, good=good, bad=bad, current=img_path.name)

    print(f"[OK] template build candidates: good={good}, bad/skip={bad}")

    masks, meta = finalize_and_save_template(
        template_acc=template_acc,
        template_dir=template_dir,
        roi_w=roi_w,
        roi_h=roi_h,
        thresh=thresh,
    )
    return masks, meta

def process_dataset_with_optional_template(
    input_dir: Path,
    output_root: Path,
    template_dir: Path | None = None,
    roi_w: int = 900,
    roi_h: int = 2400,
    expected_boxes: int = 30,
    num_rows: int = 10,
    num_cols: int = 3,
    roi_margin: int = 6,
    min_area: int = 300,
    extent_threshold: float = 0.15,
    overlay_alpha: float = 0.45,
    seed: int = 42,
    manual_angle: float = 0.0,
    progress_callback=None,
    out_dir_suffix: str = "_output",
    should_stop=None,
):
    files = list_images(input_dir)
    if len(files) == 0:
        raise RuntimeError(f"No images found in {input_dir}")

    template_masks = None
    if template_dir is None:
        raise RuntimeError("Process mode requires template_dir.")
    npz_path = template_dir / "template_masks.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing template: {npz_path}")
    template_masks, tw, th, meta = load_template(template_dir)
    if (tw != roi_w) or (th != roi_h):
        raise RuntimeError(
            f"Template ROI size mismatch. "
            f"Template={tw}x{th}, current={roi_w}x{roi_h}. "
            f"Use same roi_w/roi_h."
        )

    # process_one_image에 넣을 공통 kwargs (img_path만 job에서 따로 넣음)
    process_kwargs = dict(
        output_root=output_root,
        expected_boxes=expected_boxes,
        num_rows=num_rows,
        num_cols=num_cols,
        roi_margin=roi_margin,
        roi_w=roi_w,
        roi_h=roi_h,
        min_area=min_area,
        extent_threshold=extent_threshold,
        overlay_alpha=overlay_alpha,
        seed=seed,
        template_masks=template_masks,
        use_template_if_missing=True,
        template_only=True,
        manual_angle=manual_angle,
        out_dir_suffix=out_dir_suffix,
        should_stop=should_stop,
    )

    ok_count = 0
    tpl_count = 0
    fail_count = 0
    total = len(files)
    done = 0
    if progress_callback is not None:
        progress_callback(done=0, total=total, ok=0, template=0, fail=0)

    for img_path in tqdm(files, desc="Processing dataset"):
        if should_stop is not None and should_stop():
            raise RuntimeError("Cancelled by user")
        try:
            ok, _, _, _, _, _ = process_one_image(img_path=img_path, **process_kwargs)
            if ok:
                ok_count += 1
            else:
                if template_masks is not None:
                    tpl_count += 1
                else:
                    fail_count += 1
        except Exception:
            fail_count += 1
        finally:
            done += 1
            if progress_callback is not None:
                progress_callback(done=done, total=total, ok=ok_count, template=tpl_count, fail=fail_count)

    print(f"[DONE] ok(30 detected)={ok_count}, template_used={tpl_count}, fail(no output)={fail_count}")

def orient_preview(input_dir: Path, output_root: Path, manual_angle: float, k: int = 3, seed: int = 42):
    files = list_images(input_dir)
    if len(files) == 0:
        raise RuntimeError(f"No images found in {input_dir}")

    k = min(k, len(files))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(files, size=k, replace=False)

    out_orient = output_root / "orient"
    ensure_dir(out_orient)

    for p in tqdm(chosen, desc=f"Orient preview (k={k}, angle={manual_angle})"):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        bgr_u8 = to_bgr_u8(img)
        rot = rotate_keep_all(bgr_u8, manual_angle)
        cv2.imwrite(str(out_orient / f"{p.stem}_rot_{manual_angle:+.2f}deg.png"), rot)

# ---------------------------
# CLI
# ---------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser("Tray grid segmentation with template fallback")
    parser.add_argument("--mode", type=str, required=True, choices=["build_template", "process", "orient", "all"])

    parser.add_argument("--template_dir", type=str, default="templates")
    parser.add_argument("--roi_w", type=int, default=900)
    parser.add_argument("--roi_h", type=int, default=2400)

    parser.add_argument("--expected_boxes", type=int, default=30)
    parser.add_argument("--num_rows", type=int, default=10)
    parser.add_argument("--num_cols", type=int, default=3)

    parser.add_argument("--roi_margin", type=int, default=6)
    parser.add_argument("--min_area", type=int, default=300)
    parser.add_argument("--extent_threshold", type=float, default=0.15)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--template_thresh", type=float, default=0.5)
    
    parser.add_argument("--manual_angle", type=float, default=0.0)
    parser.add_argument("--orient_k", type=int, default=3)

    parser.add_argument("--tray_num", type=int, required=True)
    parser.add_argument("--tray_root", type=str, default="/data/tray")
    parser.add_argument("--template_root", type=str, default="/data/trayseg_output")
    parser.add_argument("--template_result_root", type=str, default="/data/trayseg_output")
    parser.add_argument("--process_result_root", type=str, default="/data/trayseg_output")
    parser.add_argument("--orient_result_root", type=str, default="seg_output/orient/tray")

    args = parser.parse_args()
    input_dir = Path(args.tray_root)
    template_dir = Path(args.template_root)

    input_full_path = input_dir / str(args.tray_num)
    template_full_path = template_dir / str(args.tray_num) / "template"

    def _run_build_template():
        output_root = Path(args.template_result_root)
        output_full_path = output_root / str(args.tray_num)
        ensure_dir(output_full_path)
        build_template_from_dataset(
            input_dir=input_full_path,
            template_dir=template_full_path,
            output_root=output_full_path,
            roi_w=args.roi_w,
            roi_h=args.roi_h,
            expected_boxes=args.expected_boxes,
            num_rows=args.num_rows,
            num_cols=args.num_cols,
            roi_margin=args.roi_margin,
            min_area=args.min_area,
            extent_threshold=args.extent_threshold,
            overlay_alpha=args.overlay_alpha,
            seed=args.seed,
            thresh=args.template_thresh,
            manual_angle=args.manual_angle,
            out_dir_suffix="",
        )

    def _run_process():
        output_root = Path(args.process_result_root)
        output_full_path = output_root / str(args.tray_num)
        ensure_dir(output_full_path)
        process_dataset_with_optional_template(
            input_dir=input_full_path,
            output_root=output_full_path,
            template_dir=template_full_path,
            roi_w=args.roi_w,
            roi_h=args.roi_h,
            expected_boxes=args.expected_boxes,
            num_rows=args.num_rows,
            num_cols=args.num_cols,
            roi_margin=args.roi_margin,
            min_area=args.min_area,
            extent_threshold=args.extent_threshold,
            overlay_alpha=args.overlay_alpha,
            seed=args.seed,
            manual_angle=args.manual_angle,
            out_dir_suffix="",
        )

    def _run_orient():
        output_root = Path(args.orient_result_root)
        output_full_path = output_root / str(args.tray_num)
        ensure_dir(output_full_path)
        orient_preview(
            input_dir=input_full_path,
            output_root=output_full_path,
            manual_angle=args.manual_angle,
            k=args.orient_k,
            seed=args.seed,
        )

    if args.mode == "build_template":
        _run_build_template()
    elif args.mode == "process":
        _run_process()
    elif args.mode == "orient":
        _run_orient()
    elif args.mode == "all":
        _run_build_template()
        _run_process()


if __name__ == "__main__":
    main()
