"""Pre/post-processing for a YOLOv8 ONNX model. Uses only numpy and Pillow."""
import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {"damaged": (211, 47, 47), "undamaged": (46, 125, 50)}


def letterbox(image, size):
    """Resize to fit size x size, pad with grey. Returns (tensor, scale, pad_x, pad_y)."""
    w, h = image.size
    scale = min(size / w, size / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
    return tensor, scale, pad_x, pad_y


def nms(boxes, scores, class_ids, iou_thr):
    """Greedy non-maximum suppression, applied separately per class."""
    keep = []
    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        order = idx[np.argsort(-scores[idx])]
        while order.size:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-9)
            order = rest[iou <= iou_thr]
    return keep


def postprocess(output, scale, pad_x, pad_y, orig_w, orig_h, class_names, conf_thr=0.25, iou_thr=0.7):
    """Turn raw YOLOv8 output into [{class_id, class_name, confidence, box}] in original-image pixels."""
    preds = output[0]
    if preds.shape[0] < preds.shape[1]:  # (4+nc, N) -> (N, 4+nc)
        preds = preds.T
    scores = preds[:, 4:]
    class_ids = scores.argmax(axis=1)
    confs = scores[np.arange(len(scores)), class_ids]
    keep = confs >= conf_thr
    preds, class_ids, confs = preds[keep], class_ids[keep], confs[keep]
    if len(preds) == 0:
        return []

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / scale).clip(0, orig_w)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / scale).clip(0, orig_h)

    detections = []
    for i in nms(boxes, confs, class_ids, iou_thr):
        cid = int(class_ids[i])
        name = class_names[cid] if cid < len(class_names) else str(cid)
        detections.append(
            {
                "class_id": cid,
                "class_name": name,
                "confidence": round(float(confs[i]), 4),
                "box": [round(float(v), 2) for v in boxes[i]],
            }
        )
    detections.sort(key=lambda d: -d["confidence"])
    return detections


def annotate(image, detections, max_side=1280):
    """Draw boxes on a (possibly downscaled) copy and return it as a Base64 JPEG string."""
    view = image.copy()
    view.thumbnail((max_side, max_side))
    f = view.width / image.width
    draw = ImageDraw.Draw(view)
    font_size = max(14, view.width // 40)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    line = max(2, view.width // 300)

    for d in detections:
        x1, y1, x2, y2 = [v * f for v in d["box"]]
        color = COLORS.get(d["class_name"], (25, 118, 210))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line)
        text = f'{d["class_name"]} {d["confidence"]:.2f}'
        left, top_, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top_
        top = max(0, y1 - th - 6)
        draw.rectangle([x1, top, x1 + tw + 8, top + th + 6], fill=color)
        draw.text((x1 + 4, top + 2), text, fill=(255, 255, 255), font=font)

    out = BytesIO()
    view.save(out, format="JPEG", quality=85)
    return base64.b64encode(out.getvalue()).decode("utf-8")
