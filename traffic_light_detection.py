import cv2, os, json, math
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

#===HELPER FUNCTIONS FOR TRAFFIC LIGHT DETECTION===#
def expand_box(box, img_shape, orientation, scale_y=3.0, scale_x=1.8):
    """
    Mở rộng bounding box của bóng đang sáng để bao trùm cả cụm đèn giao thông.
    box: [x, y, w, h] trên ảnh resized.
    img_shape: (H, W, C)
    orientation: "vertical" hoặc "horizontal"
    """
    if box is None:
        return None

    H, W = img_shape[:2]
    x, y, w, h = box

    cx = x + w // 2
    cy = y + h // 2

    if orientation == "vertical":
        # cao gấp ~3 lần, rộng hơn chút
        new_h = int(h * scale_y)
        new_w = int(w * scale_x)
    else:
        # horizontal: rộng gấp ~3 lần, cao hơn chút
        new_w = int(w * scale_y)
        new_h = int(h * scale_x)

    x0 = max(0, cx - new_w // 2)
    y0 = max(0, cy - new_h // 2)
    x1 = min(W, cx + new_w // 2)
    y1 = min(H, cy + new_h // 2)

    return [x0, y0, x1 - x0, y1 - y0]

def make_lamp_roi_from_best_box(best_box, img_shape, orientation):
    """
    Tạo ROI bao quanh cụm đèn giao thông dựa trên best_box (1 bóng sáng).
    best_box: [x, y, w, h] trên ảnh resized (bgr_norm).
    img_shape: bgr_norm.shape
    orientation: "vertical" hoặc "horizontal"
    """
    if best_box is None:
        return None

    H, W = img_shape[:2]
    x, y, w, h = best_box

    if orientation == "vertical":
        # mở rộng ngang vừa phải, dọc: trên ít, dưới nhiều
        x0 = max(0, x - int(0.4 * w))
        x1 = min(W, x + w + int(0.4 * w))

        y0 = max(0, y - int(0.8 * h))
        y1 = min(H, y + int(2.2 * h))
    else:
        # horizontal: mở rộng dọc vừa phải, ngang: trái/phải nhiều
        y0 = max(0, y - int(0.4 * h))
        y1 = min(H, y + h + int(0.4 * h))

        x0 = max(0, x - int(0.8 * w))
        x1 = min(W, x + int(2.2 * w))

    return [x0, y0, x1 - x0, y1 - y0]

# ========== A. Resize giữ tỉ lệ (letterbox) ==========
def letterbox_resize(img: np.ndarray, target: Tuple[int, int] = (512, 512),
                     color=(0, 0, 0)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    th, tw = target
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, (left, top)

# ========== B. Tiền xử lý sáng màu ==========
def gray_world_wb(img: np.ndarray) -> np.ndarray:
    eps = 1e-6
    b, g, r = cv2.split(img.astype(np.float32))
    mb, mg, mr = b.mean()+eps, g.mean()+eps, r.mean()+eps
    m = (mb + mg + mr) / 3.0
    b = np.clip(b * (m / mb), 0, 255)
    g = np.clip(g * (m / mg), 0, 255)
    r = np.clip(r * (m / mr), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

def auto_gamma_bgr(img: np.ndarray, target_mean_v: float = 0.55) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2] / 255.0
    mean_v = float(max(v.mean(), 1e-3))
    gamma = np.log(target_mean_v) / np.log(mean_v)
    x = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip((x ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)

# ========== C. Mask màu HSV ==========
def color_masks(hsv: np.ndarray) -> Dict[str, np.ndarray]:
    s_min, v_min = 60, 60
    # red (2 dải H)
    m1 = cv2.inRange(hsv, np.array([0, s_min, v_min]),   np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, s_min, v_min]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(m1, m2)
    # yellow
    yellow = cv2.inRange(hsv, np.array([15, s_min, v_min]), np.array([35, 255, 255]))
    # green
    green = cv2.inRange(hsv, np.array([40, s_min, v_min]), np.array([90, 255, 255]))
    return {"red": red, "yellow": yellow, "green": green}

def clean_mask(mask: np.ndarray, k_open=3, k_close=5) -> np.ndarray:
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc, iterations=1)
    return m

# ========== D. Chấm điểm contour ==========
def contour_score(mask: np.ndarray, hsv: np.ndarray) -> Tuple[Optional[Tuple[int,int,int,int]], float, Optional[Tuple[int,int]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box, best_score, best_center = None, 0.0, None
    v = hsv[:, :, 2]
    h_img, w_img = mask.shape[:2]
    min_area = max(50, int(0.0002 * h_img * w_img))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area: continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0: continue
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < 0.4: continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi_mask = mask[y:y+h, x:x+w]
        roi_v = v[y:y+h, x:x+w]
        if roi_mask.size == 0: continue
        mean_v = float(roi_v[roi_mask > 0].mean()) if np.any(roi_mask > 0) else 0.0
        score = area * (mean_v/255.0) * circ
        if score > best_score:
            best_score = score
            best_box = (int(x), int(y), int(w), int(h))
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
            else:
                cx = int(x + w/2); cy = int(y + h/2)
            best_center = (cx, cy)
    return best_box, best_score, best_center

# ========== E. Hough circles ==========
def hough_candidates(bgr: np.ndarray) -> List[Tuple[int,int,int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=32,
                               param1=120, param2=20, minRadius=6, maxRadius=90)
    if circles is None: return []
    return [tuple(map(int, c)) for c in np.round(circles[0, :]).astype(int)]

def hough_best(bgr: np.ndarray) -> Optional[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]:
    cand = hough_candidates(bgr)
    if not cand: return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    best, best_v = None, -1.0
    for (x, y, r) in cand:
        if x-r < 0 or y-r < 0 or x+r >= bgr.shape[1] or y+r >= bgr.shape[0]: continue
        mask = np.zeros(hsv.shape[:2], np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        pix = hsv[mask > 0]
        if pix.size == 0: continue
        hmean, smean, vmean = np.mean(pix, axis=0).astype(int).tolist()
        if vmean > best_v:
            best_v = vmean
            best = ((int(x), int(y), int(r)), (int(hmean), int(smean), int(vmean)))
    return best

# ========== F. HSV → label ==========
def hsv_to_label(hsv_triplet: Tuple[int,int,int]) -> str:
    h, s, v = hsv_triplet
    if s < 60 or v < 60: return "unknown"
    if (0 <= h <= 10) or (170 <= h <= 180): return "red"
    if 15 <= h <= 35: return "yellow"
    if 40 <= h <= 90: return "green"
    return "unknown"

# ========== G. Orientation v3 (dải dọc quanh bóng sáng nhất) ==========
def _lines_stat(img_gray: np.ndarray) -> Tuple[int,int]:
    edges = cv2.Canny(img_gray, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=28, minLineLength=18, maxLineGap=6)
    v_cnt = h_cnt = 0
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            dx, dy = abs(x2-x1), abs(y2-y1)
            if dy >= dx*1.5: v_cnt += 1
            elif dx >= dy*1.5: h_cnt += 1
    return v_cnt, h_cnt

def determine_orientation(bgr_norm: np.ndarray,
                          contour_centers: List[Tuple[int,int]],
                          masks_union: Optional[np.ndarray] = None,
                          debug: bool=False) -> str:
    h, w = bgr_norm.shape[:2]

    hb = hough_best(bgr_norm)
    if hb is not None:
        (cx, cy, r), _ = hb
        strip_w = int(max(24, 2.2*r))
        x1 = max(0, cx - strip_w//2); x2 = min(w-1, cx + strip_w//2)
        roi = bgr_norm[:, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        v_cnt, h_cnt = _lines_stat(gray)
        if debug:
            print(f"[ORI] ROI around brightest circle x=[{x1},{x2}]  r={r}  lines V/H={v_cnt}/{h_cnt}")

        if v_cnt > h_cnt * 1.15:
            return "vertical"
        elif h_cnt > v_cnt * 1.15:
            return "horizontal"
        else:
            # hoà nhau → ưu tiên vertical (an toàn hơn cho đa số đèn)
            return "vertical"

    centers = contour_centers or []
    if len(centers) >= 2:
        xs = [c[0] for c in centers]; ys = [c[1] for c in centers]
        spread_x = float(max(xs)-min(xs)+1e-6); spread_y = float(max(ys)-min(ys)+1e-6)
        if debug: print(f"[ORI] fallback centers spreadX={spread_x:.1f} spreadY={spread_y:.1f}")
        if spread_y > spread_x * 1.2: return "vertical"
        if spread_x > spread_y * 1.2: return "horizontal"
        return "vertical"

    return "vertical" if h >= w*0.9 else "horizontal"

# ========== H. Multi-lamp (phát hiện cả 3 bóng) ==========
def nms_circles(circles, iou_thresh=0.35):
    # circles: list[(x,y,r,score,(h,s,v))]
    circles = sorted(circles, key=lambda c: c[3], reverse=True)
    kept = []
    def iou(c1, c2):
        x1,y1,r1,_,_ = c1; x2,y2,r2,_,_ = c2
        b1 = (x1-r1, y1-r1, x1+r1, y1+r1)
        b2 = (x2-r2, y2-r2, x2+r2, y2+r2)
        xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
        xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xb-xa) * max(0, yb-ya)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        union = a1 + a2 - inter + 1e-6
        return inter/union
    for c in circles:
        if all(iou(c, k) < iou_thresh for k in kept):
            kept.append(c)
    return kept

def detect_all_lamps(bgr_norm, orientation, v_threshold=110, roi_box=None):
    """
    Trả về danh sách 0–3 đèn, đã lọc theo cột đèn và gom cụm theo trục chính.
    Mỗi phần tử:
      {'center':[x,y], 'r':r, 'box':[x,y,w,h], 'hsv_avg':[h,s,v], 'label':..., 'slot':..., 'brightness':...}
    """
    h, w = bgr_norm.shape[:2]
    hsv = cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2HSV)

    # --- 1) Ứng viên từ Hough + brightness ---
    # raw = hough_candidates(bgr_norm)  # [(x,y,r), ...]
    if roi_box is not None:
        rx, ry, rw, rh = roi_box
        # cắt ROI đúng từ bgr_norm
        roi = bgr_norm[ry:ry+rh, rx:rx+rw]
        raw_local = hough_candidates(roi)

        raw = []
        if raw_local is not None:
            for (cx, cy, r) in raw_local:
                gx = cx + rx
                gy = cy + ry
                # bỏ circle ngoài biên ảnh
                if gx - r < 0 or gx + r >= w or gy - r < 0 or gy + r >= h:
                    continue
                raw.append((gx, gy, r))
    else:
        raw = hough_candidates(bgr_norm)

    
    cands = []
    for (x,y,r) in raw:
        if x-r < 0 or y-r < 0 or x+r >= w or y+r >= h:
            continue
        if r < 8:  # loại đốm nhỏ
            continue
        mask = np.zeros(hsv.shape[:2], np.uint8)
        cv2.circle(mask, (x,y), r, 255, -1)
        pix = hsv[mask > 0]
        if pix.size == 0:
            continue
        hmean, smean, vmean = np.mean(pix, axis=0).astype(int).tolist()
        score = float(vmean)
        cands.append((int(x), int(y), int(r), score, (int(hmean), int(smean), int(vmean))))

    if not cands:
        return []

    # --- 2) Xác định "cột đèn": dải quanh bóng sáng nhất ---
    brightest = max(cands, key=lambda c: c[3])
    cx, cy, r_best, _, _ = brightest
    strip_scale = 2.4  # dải rộng quanh tâm theo phương vuông góc trục chính
    if orientation == "vertical":
        x1 = max(0, int(cx - strip_scale * r_best // 2))
        x2 = min(w-1, int(cx + strip_scale * r_best // 2))
        cands = [c for c in cands if x1 <= c[0] <= x2]
    else:
        y1 = max(0, int(cy - strip_scale * r_best // 2))
        y2 = min(h-1, int(cy + strip_scale * r_best // 2))
        cands = [c for c in cands if y1 <= c[1] <= y2]

    # Lọc theo độ sáng
    cands = [c for c in cands if c[3] >= v_threshold]
    if not cands:
        return []

    # --- 3) NMS để tránh trùng ---
    cands = nms_circles(cands, iou_thresh=0.35)

    # --- 4) Gom cụm 1D bằng "gap clustering" (ổn định, không dùng k-means) ---
    # Sắp theo trục chính
    if orientation == "vertical":
        cands.sort(key=lambda c: c[1])  # theo y
        axis_vals = [c[1] for c in cands]
        gap_unit = max(int(0.9 * r_best), 10)  # ngưỡng tách cụm
    else:
        cands.sort(key=lambda c: c[0])  # theo x
        axis_vals = [c[0] for c in cands]
        gap_unit = max(int(0.9 * r_best), 10)

    clusters = [[cands[0]]]
    for i in range(1, len(cands)):
        if abs(axis_vals[i] - axis_vals[i-1]) > gap_unit and len(clusters) < 3:
            clusters.append([cands[i]])
        else:
            clusters[-1].append(cands[i])

    # --- 5) Mỗi cụm lấy ứng viên sáng nhất ---
    reps = []
    for cl in clusters:
        reps.append(max(cl, key=lambda c: c[3]))

    # --- 6) Gán slot & đóng gói kết quả (tối đa 3 bóng) ---
    if orientation == "vertical":
        reps.sort(key=lambda c: c[1])  # y
        slots = ["top", "mid", "bot"]
    else:
        reps.sort(key=lambda c: c[0])  # x
        slots = ["left", "center", "right"]

    lamps = []
    for i, (x,y,r,score,hsv_avg) in enumerate(reps[:3]):
        label = hsv_to_label(hsv_avg)
        bx, by = max(0, x-r), max(0, y-r)
        bw = min(2*r, bgr_norm.shape[1]-bx-1)
        bh = min(2*r, bgr_norm.shape[0]-by-1)
        lamps.append({
            "center": [int(x), int(y)],
            "r": int(r),
            "box": [int(bx), int(by), int(bw), int(bh)],
            "hsv_avg": [int(hsv_avg[0]), int(hsv_avg[1]), int(hsv_avg[2])],
            "label": label,
            "slot": slots[i] if i < len(slots) else "unknown",
            "brightness": float(score)
        })
    return lamps

# ========== I. Suy luận màu theo vị trí ==========
def infer_by_position(x: int, y: int, img_w: int, img_h: int, orientation: str) -> str:
    cx = x / max(1, img_w); cy = y / max(1, img_h)
    if orientation == "vertical":
        if cy < 0.33: return "red"
        if cy < 0.66: return "yellow"
        return "green"
    else:
        if cx < 0.33: return "red"
        if cx < 0.66: return "yellow"
        return "green"

# ========== J. Hàm chính ==========
def detect_traffic_light_color(image_path: str,
                               denoise: str = "bilateral",
                               force_orientation: Optional[str] = None,
                               debug: bool=False) -> Dict[str, Any]:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if denoise == "nlmeans":
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    else:
        bgr = cv2.bilateralFilter(bgr, d=7, sigmaColor=60, sigmaSpace=60)

    bgr_resized, scale, offset = letterbox_resize(bgr, (512, 512))
    bgr_wb = gray_world_wb(bgr_resized)
    bgr_gamma = auto_gamma_bgr(bgr_wb, target_mean_v=0.55)
    hsv = cv2.cvtColor(bgr_gamma, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    bgr_norm = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    hsv2 = cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2HSV)
    raw_masks = color_masks(hsv2)
    masks = {k: clean_mask(m) for k, m in raw_masks.items()}
    masks_union = cv2.bitwise_or(cv2.bitwise_or(masks["red"], masks["yellow"]), masks["green"])

    boxes: Dict[str, Optional[Tuple[int,int,int,int]]] = {}
    scores: Dict[str, float] = {}
    centers: List[Tuple[int,int]] = []
    for color, m in masks.items():
        box, score, center = contour_score(m, hsv2)
        boxes[color] = box
        scores[color] = float(score)
        if center is not None:
            centers.append(center)

    if force_orientation in ("vertical", "horizontal"):
        orientation = force_orientation
        if debug: print(f"[ORI] forced={orientation}")
    else:
        orientation = determine_orientation(bgr_norm, centers, masks_union=masks_union, debug=debug)

    if debug:
        xs = [c[0] for c in centers] or [0,0]
        ys = [c[1] for c in centers] or [0,0]
        spread_x = float(max(xs)-min(xs)) if centers else 0.0
        spread_y = float(max(ys)-min(ys)) if centers else 0.0
        print(f"[ORI] spreadX={spread_x:.1f} spreadY={spread_y:.1f}  -> {orientation}")

    colors = ["red","yellow","green"]
    best_color = max(colors, key=lambda c: scores[c])
    best_score = scores[best_color]
    best_box = boxes[best_color]
    label = best_color if best_score >= 1500.0 else "unknown"
    note = ""

    if label == "unknown":
        hb = hough_best(bgr_norm)
        if hb is not None:
            (x, y, r), mean_hsv = hb
            label2 = hsv_to_label(mean_hsv)
            best_box = (max(x-r, 0), max(y-r, 0),
                        min(2*r, bgr_norm.shape[1]-1), min(2*r, bgr_norm.shape[0]-1))
            best_score = float(mean_hsv[2])
            if label2 != "unknown":
                label = label2
            else:
                label = infer_by_position(x, y, bgr_norm.shape[1], bgr_norm.shape[0], orientation)
                note = "inferred_by_position"
        else:
            if best_box is not None:
                bx, by, bw, bh = best_box
                cx, cy = int(bx + bw/2), int(by + bh/2)
                label = infer_by_position(cx, cy, bgr_norm.shape[1], bgr_norm.shape[0], orientation)
                note = "inferred_by_position"
            else:
                label = "unknown"

    # Multi-lamp (phát hiện tất cả bóng đang sáng)
    # all_lamps = detect_all_lamps(bgr_norm, orientation, v_threshold=110)
    # best_box chính là box tương ứng màu có score cao nhất
    roi_box = None
    
    if best_box is not None and orientation in ("vertical", "horizontal"):
        roi_box = expand_box(
            best_box,
            bgr_norm.shape,
            orientation,
            scale_y=3.0,   # có thể chỉnh sau
            scale_x=1.8
        )

    roi_box = make_lamp_roi_from_best_box(best_box, bgr_norm.shape, orientation)

    all_lamps = detect_all_lamps(
        bgr_norm,
        orientation,
        v_threshold=160,   # xem mục 2 bên dưới
        roi_box=roi_box
    )

    # --- CHỈ GIỮ 1 BOX "ĐÚNG NHẤT" ---
    best_lamp = None
    if all_lamps:
        same_label = [lp for lp in all_lamps if lp.get("label") == label]
        pick_from = same_label if same_label else all_lamps
        best_lamp = max(pick_from, key=lambda lp: float(lp.get("brightness", 0.0)))

    lamps_keep = [best_lamp] if best_lamp is not None else []

    return {
        "label": str(label),
        "box": [int(v) for v in best_box] if best_box else None,
        "score": float(best_score),
        "orientation": orientation,
        # "lamps": all_lamps,   # <— danh sách 0–3 bóng
        "lamps": lamps_keep,   # <— danh sách 0–3 bóng
        "debug": {
            "scores": {k: float(v) for k, v in scores.items()},
            "offset": [int(offset[0]), int(offset[1])],
            "scale": float(scale),
            "note": note
        }
    }

# ========== K. CLI ==========
def _np_convert(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Traffic light detection (robust orientation + multi-lamp)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--denoise", default="bilateral", choices=["bilateral","nlmeans"])
    parser.add_argument("--force-orientation", default=None, choices=["vertical","horizontal"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    res = detect_traffic_light_color(args.image, denoise=args.denoise,
                                     force_orientation=args.force_orientation,
                                     debug=args.debug)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=_np_convert))

    # visualize
    bgr = cv2.imread(args.image)
    vis, _, _ = letterbox_resize(bgr, (512, 512))
   # Nếu có lamp thì CHỈ vẽ lamp (1 box). Nếu không có lamp thì vẽ res["box"].
    if res.get("lamps"):
        lp = res["lamps"][0]  # bạn đã lọc còn 1 phần tử rồi
        x, y, w, h = lp["box"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(
        #     vis,
        #     f"{lp['slot']}:{lp['label']}",
        #     (x, max(0, y - 6)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 255, 0),
        #     1,
        #     cv2.LINE_AA,
        # )
    else:
        if res.get("box") is not None:
            x, y, w, h = res["box"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)


    txt = f"{res['label']} | {res['orientation']} ({int(res['score'])})"
    # cv2.putText(vis, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
    out_path = os.path.splitext(args.image)[0] + "_vis.jpg"
    cv2.imwrite(out_path, vis)
    print(f"Saved visualization: {out_path}")
