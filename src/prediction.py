"""
prediction.py
-------------
Single-datapoint inference. This is what the API's /predict endpoint
and the notebook's evaluation cells both call, so predictions are
computed identically in every context.
"""

import os
import numpy as np
import tensorflow as tf

from preprocessing import (
    CLASS_NAMES,
    preprocess_image_from_path,
    preprocess_uploaded_bytes,
)

_MODEL_CACHE = {}


def clear_model_cache():
    """Drop cached models so retrain can reclaim RAM on small hosts."""
    _MODEL_CACHE.clear()


def load_model(model_path="../models/bean_model_latest.h5"):
    """
    Loads (and caches) the current production model. Cache is keyed by
    path + last-modified time, so a retrain that overwrites
    bean_model_latest.h5 is picked up automatically on the next call
    without restarting the process.
    """
    key = (model_path, os.path.getmtime(model_path))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()  # drop stale versions
        _MODEL_CACHE[key] = tf.keras.models.load_model(model_path)
    return _MODEL_CACHE[key]


def predict_image(image_path=None, image_bytes=None, model_path="../models/bean_model_latest.h5"):
    """
    Predicts the class of a single bean leaf image.
    Provide exactly one of image_path or image_bytes.

    Returns a dict:
        {
          "predicted_class": "healthy",
          "confidence": 0.94,
          "class_probabilities": {"angular_leaf_spot": 0.02, "bean_rust": 0.04, "healthy": 0.94}
        }
    """
    if (image_path is None) == (image_bytes is None):
        raise ValueError("Provide exactly one of image_path or image_bytes")

    model = load_model(model_path)

    batch = (
        preprocess_image_from_path(image_path)
        if image_path is not None
        else preprocess_uploaded_bytes(image_bytes)
    )

    probs = model.predict(batch, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "class_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }
