
import os
import base64
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

app = FastAPI(title="YOLO Prediction API")

# Set FRONTEND_URL in Render to your actual Vercel URL.
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Model loads once when the server starts.
model_path = os.getenv("MODEL_PATH", "best (1).pt")
model = YOLO(model_path)


@app.get("/")
def home():
    return {"status": "running", "message": "YOLO API is ready"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in (
        "image/jpeg", "image/png", "image/webp"
    ):
        raise HTTPException(400, "Upload a JPG, PNG, or WEBP image.")

    data = await file.read(10 * 1024 * 1024 + 1)

    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image limit is 10 MB.")

    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Invalid image.")

    result = model.predict(image, conf=0.25, verbose=False)[0]
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        detections.append({
            "class_id": class_id,
            "class_name": model.names[class_id],
            "confidence": round(float(box.conf[0].item()), 4),
            "box": [
                round(float(v), 2)
                for v in box.xyxy[0].tolist()
            ]
        })

    # Convert annotated prediction to a Base64 JPEG.
    annotated = result.plot()[:, :, ::-1]
    output = BytesIO()
    Image.fromarray(annotated).save(
        output, format="JPEG", quality=85
    )

    return {
        "success": True,
        "count": len(detections),
        "detections": detections,
        "annotated_image": base64.b64encode(
            output.getvalue()
        ).decode("utf-8")
    }

