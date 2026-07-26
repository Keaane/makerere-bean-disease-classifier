"""
main.py
-------
FastAPI service for the Bean Leaf Disease Classifier.

Endpoints:
  GET  /status                      - uptime, model version, dataset & prediction stats
  POST /predict                     - predict the class of a single uploaded image
  POST /upload-data/{class_name}    - bulk-upload images for a class, for retraining
  POST /retrain                     - trigger retraining on newly uploaded data
  GET  /visualizations/class-distribution   - class counts (train data + uploads)

Run locally with:
    uvicorn main:app --reload --port 8000
"""

import os
import sys
import time
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import CLASS_NAMES, load_dataset_from_directory, build_pipeline  # noqa: E402
from model import retrain_model  # noqa: E402
from prediction import predict_image  # noqa: E402

import database  # noqa: E402

APP_START_TIME = time.time()

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_TRAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "train")
MODEL_PATH = os.path.join(MODEL_DIR, "bean_model_latest.h5")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def _count_images(directory: str) -> int:
    """Count image files in a directory, ignoring placeholders like .gitkeep."""
    if not os.path.isdir(directory):
        return 0
    return sum(
        1
        for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    )

app = FastAPI(
    title="Bean Leaf Disease Classifier API",
    description="Predict bean leaf disease from an image, and manage retraining.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relaxed for the demo UI; tighten for real production use
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    database.init_db()
    os.makedirs(MODEL_DIR, exist_ok=True)
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(DATA_TRAIN_DIR, class_name), exist_ok=True)


UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")


@app.get("/")
def root():
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"service": "Bean Leaf Disease Classifier API", "docs": "/docs"}


@app.get("/api")
def api_info():
    return {
        "service": "Bean Leaf Disease Classifier API",
        "endpoints": ["/status", "/predict", "/upload-data/{class_name}", "/retrain",
                      "/visualizations/class-distribution"],
    }


@app.get("/status")
def status():
    """
    Model uptime, current model version, and prediction/upload stats -
    powers the UI's monitoring dashboard.
    """
    uptime_seconds = time.time() - APP_START_TIME

    model_version = "no model found"
    if os.path.exists(MODEL_PATH):
        model_version = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(MODEL_PATH))
        )

    pred_stats = database.get_prediction_stats()
    upload_dist = database.get_upload_class_distribution()
    retrain_history = database.get_retrain_history()

    return {
        "status": "up",
        "uptime_seconds": round(uptime_seconds, 1),
        "model_last_updated": model_version,
        "total_predictions_served": pred_stats["total"],
        "predictions_by_class": pred_stats["by_class"],
        "recent_predictions": pred_stats["recent"],
        "uploaded_images_by_class": upload_dist,
        "retrain_history": retrain_history,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the disease class of a single uploaded bean leaf image.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    image_bytes = await file.read()
    try:
        result = predict_image(image_bytes=image_bytes, model_path=MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    database.log_prediction(result["predicted_class"], result["confidence"])
    return result


@app.post("/upload-data/{class_name}")
async def upload_data(class_name: str, files: list[UploadFile] = File(...)):
    """
    Bulk-uploads images for a given class, to be used in the next
    retraining run. Saves files to data/train/<class_name>/ and logs
    each upload to the database.
    """
    if class_name not in CLASS_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid class_name. Must be one of {CLASS_NAMES}",
        )

    class_dir = os.path.join(DATA_TRAIN_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    saved = []
    for f in files:
        timestamp = int(time.time() * 1000)
        safe_name = f"uploaded_{timestamp}_{f.filename}"
        dest_path = os.path.join(class_dir, safe_name)
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(f.file, out_file)
        database.log_upload(class_name, safe_name)
        saved.append(safe_name)

    return {
        "class_name": class_name,
        "num_uploaded": len(saved),
        "filenames": saved,
    }


@app.post("/retrain")
def retrain(epochs: int = 5):
    """
    Triggers retraining on the current contents of data/train (original
    data + any images uploaded via /upload-data). Loads the current
    production model and fine-tunes it further, then promotes the
    result to bean_model_latest.h5.
    """
    total_images = sum(
        _count_images(os.path.join(DATA_TRAIN_DIR, c)) for c in CLASS_NAMES
    )
    if total_images < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough training data to retrain ({total_images} images found, need >= 20).",
        )

    try:
        train_raw, val_raw = load_dataset_from_directory(DATA_TRAIN_DIR)
        train_ds = build_pipeline(train_raw, batch_size=16, augment=True, shuffle=True)
        val_ds = build_pipeline(val_raw, batch_size=16, augment=False, shuffle=False)

        history, versioned_path, latest_path = retrain_model(
            train_ds, val_ds, model_dir=MODEL_DIR, epochs=epochs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

    final_val_accuracy = None
    val_accs = history.history.get("val_accuracy") or []
    if val_accs:
        final_val_accuracy = float(val_accs[-1])

    database.log_retrain_event(
        num_new_images=total_images,
        epochs=epochs,
        final_val_accuracy=final_val_accuracy,
        model_version=os.path.basename(versioned_path),
    )

    return {
        "status": "retrained",
        "model_version": os.path.basename(versioned_path),
        "final_val_accuracy": final_val_accuracy,
        "trained_on_images": total_images,
    }


@app.get("/visualizations/class-distribution")
def class_distribution():
    """
    Current class counts in data/train (or