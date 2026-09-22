"""Render RT-DETR fire/smoke predictions onto a video, frame by frame.

Draws one box per detection with a class-specific colour, the class name and
the confidence. Frames are batched (the model is dispatch-bound at batch 1, so
batching is most of the speed) and run in fp16, which is accuracy-neutral here.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / "repos/fire_detection"
CKPT = ROOT / "runs/seedvar_fasdd_s2/weights/best.pt"
SRC = Path("/mnt/c/Users/ms115/source/repos/dfire-models/Original.mp4")
DST = Path("/mnt/c/Users/ms115/source/repos/dfire-models/Processed.mp4")

CONF = 0.40
BATCH = 16

# BGR. Fire warm/red, smoke cool/blue -- readable against both flame and haze.
COLORS = {
    "fire":  (36, 60, 255),
    "smoke": (255, 190, 90),
}
FALLBACK = (0, 255, 0)


def draw(frame, boxes, names):
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        cls = names[int(b.cls[0])]
        conf = float(b.conf[0])
        color = COLORS.get(cls.lower(), FALLBACK)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"{cls} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # keep the label on-screen when the box is near the top edge
        ty = y1 - 8 if y1 - th - 12 > 0 else y2 + th + 12
        cv2.rectangle(frame, (x1, ty - th - base - 4), (x1 + tw + 8, ty + base - 2),
                      color, -1)
        cv2.putText(frame, label, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    from ultralytics import RTDETR

    cap = cv2.VideoCapture(str(SRC))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {SRC}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"in : {SRC.name}  {w}x{h}  {fps:.2f} fps  {total} frames", flush=True)

    out = cv2.VideoWriter(str(DST), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        raise SystemExit("VideoWriter failed to open")

    model = RTDETR(str(CKPT))
    names = model.model.names

    n_frames = 0
    n_det = {"fire": 0, "smoke": 0}
    t0 = time.time()
    batch = []

    while True:
        ok, frame = cap.read()
        if ok:
            batch.append(frame)
        if batch and (not ok or len(batch) == BATCH):
            results = model.predict(batch, imgsz=640, conf=CONF, half=True,
                                    device="0", verbose=False)
            for f, r in zip(batch, results):
                for b in r.boxes:
                    c = names[int(b.cls[0])].lower()
                    if c in n_det:
                        n_det[c] += 1
                out.write(draw(f, r.boxes, names))
                n_frames += 1
            batch = []
            if n_frames % 480 < BATCH:
                el = time.time() - t0
                print(f"  {n_frames}/{total}  {n_frames/el:.1f} fps  "
                      f"eta {(total-n_frames)/max(n_frames/el,1e-9):.0f}s", flush=True)
        if not ok:
            break

    cap.release()
    out.release()
    el = time.time() - t0
    print(f"\ndone: {n_frames} frames in {el:.1f}s ({n_frames/el:.1f} fps)")
    print(f"detections drawn -- fire: {n_det['fire']:,}  smoke: {n_det['smoke']:,}")
    print(f"out: {DST}  ({DST.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
