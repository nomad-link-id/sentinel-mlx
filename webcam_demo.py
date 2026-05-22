import sys
import time
from collections import deque

import cv2
import numpy as np
from yolo26mlx import YOLO

MODEL_PATH = "models/yolo26n.npz"
CONF_THRESHOLD = 0.40
PERSON_CLASS = 0

# BGR colors
WHITE = (255, 255, 255)
GRAY = (140, 140, 140)
DARK_GRAY = (60, 60, 60)
BLACK_BG = (15, 15, 15)
SIDEBAR_BG = (20, 20, 20)
GREEN = (94, 197, 34)
ORANGE = (11, 158, 245)
RED = (68, 68, 239)

TRIAGE_COLORS = {"T1": RED, "T2": ORANGE, "T3": GREEN}

SIDEBAR_W = 380
HEADER_H = 70
FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_DUPLEX


def classify_posture(x1, y1, x2, y2):
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    ratio = h / w
    if ratio > 1.6:
        return "T3"
    elif ratio >= 1.0:
        return "T2"
    else:
        return "T1"


def draw_badge(img, text, x, y, color, font_scale=0.42):
    (tw, th), _ = cv2.getTextSize(text, MONO, font_scale, 1)
    pad = 10
    cv2.rectangle(img, (x, y), (x + tw + pad * 2, y + th + pad), color, 1)
    cv2.putText(img, text, (x + pad, y + th + pad // 2), MONO, font_scale, color, 1, cv2.LINE_AA)
    return tw + pad * 2


def draw_triage_box(img, label, full_name, count, color, x, y, w, h):
    # Tinted background fill (9% alpha — subtle card fill)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.09, img, 0.91, 0, img)
    # Border
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    # Left accent bar (thicker)
    cv2.rectangle(img, (x, y), (x + 4, y + h), color, -1)
    # Label + name
    cv2.putText(img, label, (x + 18, y + 30), MONO, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(img, full_name, (x + 18, y + 55), FONT, 0.42, GRAY, 1, cv2.LINE_AA)
    # Count (large, right-aligned)
    cnt_str = str(count)
    (tw, th), _ = cv2.getTextSize(cnt_str, MONO, 1.5, 3)
    cv2.putText(img, cnt_str, (x + w - tw - 18, y + h - 16), MONO, 1.5, color, 3, cv2.LINE_AA)


def render_sidebar(width, height, fps, counts, total_dets, frame_count):
    sb = np.full((height, width, 3), SIDEBAR_BG, dtype=np.uint8)
    cv2.putText(sb, "TRIAGE STATUS", (20, 32), MONO, 0.6, WHITE, 1, cv2.LINE_AA)
    box_y = 50
    box_h = 80
    gap = 14
    for label, name, color in [
        ("T1", "IMMEDIATE", RED),
        ("T2", "DELAYED", ORANGE),
        ("T3", "AMBULATORY", GREEN),
    ]:
        draw_triage_box(sb, label, name, counts[label], color, 20, box_y, width - 40, box_h)
        box_y += box_h + gap

    perf_y = box_y + 30
    cv2.putText(sb, "PERFORMANCE", (20, perf_y), MONO, 0.6, WHITE, 1, cv2.LINE_AA)
    perf_y += 30
    rows = [
        ("FPS", f"{fps:.1f} FPS", GREEN),
        ("Detections", str(total_dets), WHITE),
        ("Frames", str(frame_count), WHITE),
        ("Model", "yolo26n MLX", WHITE),
    ]
    for k, v, col in rows:
        cv2.putText(sb, k, (20, perf_y), FONT, 0.5, GRAY, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(v, MONO, 0.5, 1)
        cv2.putText(sb, v, (width - tw - 20, perf_y), MONO, 0.5, col, 1, cv2.LINE_AA)
        perf_y += 30

    sys_y = perf_y + 30
    cv2.putText(sb, "SYSTEM", (20, sys_y), MONO, 0.6, WHITE, 1, cv2.LINE_AA)
    sys_y += 28
    for txt in ["Connected", "On-device inference", "Zero cloud dependency"]:
        cv2.circle(sb, (24, sys_y - 5), 5, GREEN, -1)
        cv2.putText(sb, txt, (38, sys_y), MONO, 0.45, WHITE, 1, cv2.LINE_AA)
        sys_y += 24
    return sb


def render_header(width):
    h = np.full((HEADER_H, width, 3), BLACK_BG, dtype=np.uint8)
    senti_text = "SENTI"
    (senti_w, _), _ = cv2.getTextSize(senti_text, MONO, 1.0, 2)
    cv2.putText(h, senti_text, (20, 45), MONO, 1.0, WHITE, 2, cv2.LINE_AA)
    cv2.putText(h, "NEL", (20 + senti_w + 2, 45), MONO, 1.0, RED, 2, cv2.LINE_AA)
    x = width - 20
    for txt in ["LIVE", "YOLO26 MLX", "ZERO NETWORK EGRESS"]:
        (tw, _), _ = cv2.getTextSize(txt, MONO, 0.42, 1)
        bw = tw + 20
        x -= bw + 10
        draw_badge(h, txt, x, 22, GREEN, 0.42)
    cv2.line(h, (0, HEADER_H - 1), (width, HEADER_H - 1), GREEN, 1)
    return h


def main():
    print(f"Loading {MODEL_PATH}...", flush=True)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera 0", flush=True)
        sys.exit(1)

    fps_window = deque(maxlen=15)
    last_t = time.time()
    frame_count = 0

    cv2.namedWindow("SENTINEL", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_count += 1
            results = model.predict(frame, conf=CONF_THRESHOLD)

            counts = {"T1": 0, "T2": 0, "T3": 0}
            total = 0

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls_v = boxes.cls[i]
                    cls = int(cls_v.item()) if hasattr(cls_v, "item") else int(cls_v)
                    if cls != PERSON_CLASS:
                        continue
                    conf_v = boxes.conf[i]
                    conf = float(conf_v.item()) if hasattr(conf_v, "item") else float(conf_v)
                    if conf < CONF_THRESHOLD:
                        continue
                    xyxy_v = boxes.xyxy[i]
                    xyxy = xyxy_v.tolist() if hasattr(xyxy_v, "tolist") else list(xyxy_v)
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    triage = classify_posture(x1, y1, x2, y2)
                    counts[triage] += 1
                    total += 1
                    color = TRIAGE_COLORS[triage]

                    # Body fill (28% alpha — visible but not overpowering)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.38, frame, 0.62, 0, frame)

                    # Thin full-rectangle border (1px) for box shape continuity
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                    # Radar-style corner brackets (thick, scale with box size)
                    box_w = max(x2 - x1, 40)
                    box_h = max(y2 - y1, 40)
                    bracket_len = max(min(box_w, box_h) // 6, 16)
                    bt = 4  # bracket thickness
                    # top-left
                    cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, bt)
                    cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, bt)
                    # top-right
                    cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, bt)
                    cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, bt)
                    # bottom-left
                    cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, bt)
                    cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, bt)
                    # bottom-right
                    cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, bt)
                    cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, bt)

                    # Label pill — solid color background, white text (high contrast)
                    label = f"{triage}  |  {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, MONO, 0.85, 2)
                    pad_x, pad_y = 16, 11
                    pill_w = tw + pad_x * 2
                    pill_h = th + pad_y * 2
                    pill_x1 = x1
                    pill_y1 = max(y1 - pill_h - 6, 0)
                    pill_x2 = pill_x1 + pill_w
                    pill_y2 = pill_y1 + pill_h
                    # Solid color background
                    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)
                    # White text on top
                    cv2.putText(frame, label, (pill_x1 + pad_x, pill_y2 - pad_y),
                                MONO, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            now = time.time()
            dt = max(now - last_t, 1e-6)
            fps_window.append(1.0 / dt)
            last_t = now
            fps = sum(fps_window) / len(fps_window)

            h_cam, w_cam = frame.shape[:2]
            total_w = w_cam + SIDEBAR_W
            total_h = HEADER_H + h_cam
            canvas = np.full((total_h, total_w, 3), BLACK_BG, dtype=np.uint8)
            canvas[:HEADER_H, :] = render_header(total_w)
            canvas[HEADER_H:, :w_cam] = frame
            canvas[HEADER_H:, w_cam:] = render_sidebar(
                SIDEBAR_W, h_cam, fps, counts, total, frame_count
            )
            cv2.line(canvas, (w_cam, HEADER_H), (w_cam, total_h), GREEN, 1)

            cv2.imshow("SENTINEL", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
