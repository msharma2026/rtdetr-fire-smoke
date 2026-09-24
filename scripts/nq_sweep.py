"""Query-count sweep: can RT-DETR run fewer object queries at inference?

RTDETRDecoder._get_decoder_input selects self.num_queries top-scoring encoder
proposals to seed the decoder (head.py:1809). Decoder self-attention is
O(nq^2) and cross-attention O(nq x features), so on a 2-class dataset
averaging ~1.2 instances per image, 300 queries is mostly spent on empty
slots.

This is DETR-exclusive: YOLO's output count is fixed by grid resolution, not
selectable at inference.

Accuracy first -- if cutting queries costs mAP50, nothing else matters.
"""
from pathlib import Path

import torch
from ultralytics import RTDETR

ROOT = Path.home() / "repos/fire_detection"
CKPT = ROOT / "runs/joint_r16_explore/weights/best.pt"
DATA = ROOT / "data/dfire.yaml"
QUERIES = [300, 200, 100, 50, 30]


def main() -> None:
    print(f"ckpt {CKPT.name}\n", flush=True)
    rows = []
    for nq in QUERIES:
        model = RTDETR(str(CKPT))
        head = model.model.model[-1]
        head.num_queries = nq
        assert model.model.model[-1].num_queries == nq

        res = model.val(data=str(DATA), split="test", batch=16, imgsz=640,
                        device="0", half=True, plots=False, verbose=False,
                        project=str(ROOT / "runs"), name=f"nq_{nq}",
                        exist_ok=True)
        rows.append((nq, float(res.box.map50), float(res.box.map)))
        print(f"__ROW__ nq={nq:>3}  mAP50={res.box.map50:.4f}  "
              f"mAP50-95={res.box.map:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    b50, b95 = rows[0][1], rows[0][2]
    print("\n__SUMMARY__")
    print(f"{'queries':>8} {'mAP50':>8} {'d mAP50':>9} {'mAP50-95':>9} {'d 50-95':>9}")
    for nq, m50, m95 in rows:
        print(f"{nq:>8} {m50:>8.4f} {m50 - b50:>+9.4f} {m95:>9.4f} {m95 - b95:>+9.4f}")


if __name__ == "__main__":
    main()
