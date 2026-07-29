"""
model.py
--------
Model architecture, training, and retraining logic for the Bean Leaf
Disease classifier. Uses transfer learning (MobileNetV2) since the
iBean dataset is small (~1,300 images) and this gives strong accuracy
without long training times or a GPU requirement.
"""

import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models

from preprocessing import IMG_SIZE, NUM_CLASSES


def build_model(num_classes=NUM_CLASSES, fine_tune_base=False):
    """
    Builds a MobileNetV2-based transfer learning model.

    fine_tune_base=False -> base is frozen (fast, good default for the
        first training run and for lightweight retraining jobs).
    fine_tune_base=True  -> unfreezes the last ~30 layers of the base
        for a small accuracy boost once the classifier head is stable.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = fine_tune_base
    if fine_tune_base:
        for layer in base.layers[:-30]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    # MobileNetV2 expects [-1, 1] inputs; our pipeline produces [0, 1],
    # so rescale here to keep preprocessing.py the single source of truth.
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)
    x = base(x, training=fine_tune_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_ds, val_ds, epochs=15, model_dir="../models", lightweight=False):
    """
    Trains the model with early stopping + checkpointing on val_accuracy,
    and saves a timestamped version alongside a stable 'latest' pointer
    so the API can always load the current production model.

    lightweight=True skips mid-training ModelCheckpoint (saves a second
    full model copy) so API retrain fits on small Render instances.
    """
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(model_dir, f"bean_model_{timestamp}.h5")

    patience = 1 if lightweight else 4
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True
        ),
    ]
    if not lightweight:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                versioned_path, monitor="val_accuracy", save_best_only=True
            )
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Stable path the API always loads from.
    latest_path = os.path.join(model_dir, "bean_model_latest.h5")
    model.save(latest_path)
    if lightweight:
        model.save(versioned_path)

    return history, versioned_path, latest_path


def retrain_model(new_train_ds, new_val_ds, model_dir="../models", epochs=8, lightweight=False):
    """
    Retraining entry point used by the API's /retrain endpoint.

    Loads the current production model and continues training on newly
    uploaded data (fine-tuning) rather than training from scratch, then
    versions + saves the result. This is the function the retraining
    trigger in the UI ultimately calls.
    """
    import gc

    tf.keras.backend.clear_session()
    gc.collect()

    latest_path = os.path.join(model_dir, "bean_model_latest.h5")
    if os.path.exists(latest_path):
        model = tf.keras.models.load_model(latest_path)
        # Re-compile with a fresh optimizer instance. Loading a saved
        # model and continuing training directly on it can leave the
        # optimizer holding references to stale variables (Keras 3),
        # raising "This optimizer can only be called for the variables
        # it was originally built with." Recompiling avoids that.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model = build_model()

    history, versioned_path, latest_path = train_model(
        model,
        new_train_ds,
        new_val_ds,
        epochs=epochs,
        model_dir=model_dir,
        lightweight=lightweight,
    )
    return history, versioned_path, latest_path
