import os
import tempfile
import urllib.request
from io import BytesIO
from pathlib import Path

import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from detect import annotate, letterbox, postprocess

MAX_BYTES = 10 * 1024 * 1024
BASE_DIR = Path(__file__).resolve().parent
CONF = float(os.getenv("CONF", "0.25"))
# Order must match training: class 0 = undamaged, class 1 = damaged
CLASS_NAMES = [n.strip() for n in os.getenv("CLASS_NAMES", "undamaged,damaged").split(",")]

app = FastAPI(title="YOLO Prediction API (ONNX)")

# Set FRONTEND_URL in Render to your Vercel URL, with NO trailing slash.
# Several URLs are allowed, separated by commas.
origins = [
    o.strip().rstrip("/")
    for o in os.getenv("FRONTEND_URL", "http://localhost:5173").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


MODEL_URL = os.getenv("MODEL_URL")  # direct link to best.onnx, used if the file isn't in the repo


def find_model_file():
    candidates = [os.getenv("MODEL_PATH"), "best.onnx"]
    for name in candidates:
        if name and (BASE_DIR / name).is_file():
            return str(BASE_DIR / name)

    if MODEL_URL:
        target = Path(tempfile.gettempdir()) / "best.onnx"
        if not target.is_file():
            print(f"Downloading model from {MODEL_URL} ...", flush=True)
            part = target.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL, part)  # follows redirects
            if part.stat().st_size < 1_000_000:
                raise RuntimeError("Downloaded model is too small; check MODEL_URL points to the file itself.")
            part.replace(target)
            print("Model downloaded.", flush=True)
        return str(target)

    raise RuntimeError(
        f"Model file not found. Looked for {[c for c in candidates if c]} "
        f"in {BASE_DIR} and MODEL_URL is not set. Files there: {sorted(os.listdir(BASE_DIR))}"
    )


# Settings that keep memory low on small instances.
opts = ort.SessionOptions()
opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
opts.enable_cpu_mem_arena = False
opts.enable_mem_pattern = False

session = ort.InferenceSession(find_model_file(), sess_options=opts, providers=["CPUExecutionProvider"])
input_meta = session.get_inputs()[0]
INPUT_SIZE = input_meta.shape[2] if isinstance(input_meta.shape[2], int) else 640


@app.get("/")
def home():
    return {"status": "running", "message": "YOLO API is ready"}


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASS_NAMES, "input_size": INPUT_SIZE}


# Plain "def": FastAPI runs it in a worker thread, so /health stays responsive.
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, "Upload a JPG, PNG, or WEBP image.")

    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image limit is 10 MB.")

    try:
        # exif_transpose fixes sideways phone photos
        image = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Invalid image.")

    tensor, scale, pad_x, pad_y = letterbox(image, INPUT_SIZE)
    output = session.run(None, {input_meta.name: tensor})[0]
    detections = postprocess(output, scale, pad_x, pad_y, image.width, image.height, CLASS_NAMES, conf_thr=CONF)

    if any(d["class_name"] == "damaged" for d in detections):
        verdict = "damaged"
    elif detections:
        verdict = "undamaged"
    else:
        verdict = "no_package_detected"

    return {
        "success": True,
        "verdict": verdict,
        "count": len(detections),
        "detections": detections,
        "annotated_image": annotate(image, detections),
    }
