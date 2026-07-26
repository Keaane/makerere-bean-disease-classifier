"""
preprocessing.py
-----------------
Data acquisition + preprocessing utilities for the Bean Leaf Disease
classifier (iBean dataset, Makerere AI Lab).

Used both by the training notebook and by the API's retraining endpoint,
so preprocessing logic is defined exactly once and stays consistent
between offline training and production retraining.
"""

import os
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------
# 1. Data acquisition
# ---------------------------------------------------------------------
def load_raw_datasets():
    """
    Downloads (if needed) and loads the iBean dataset via the Hugging
    Face `datasets` library (preinstalled in Colab, far fewer moving
    parts than tensorflow_datasets). Returns the three official splits
    as tf.data.Dataset objects of (image, label) pairs, matching the
    same interface the rest of the pipeline expects.
    """
    from datasets import load_dataset

    # HF deprecated the old un-namespaced "beans" path; use the
    # namespaced repo id instead.
    hf_ds = load_dataset("AI-Lab-Makerere/beans")  # -> DatasetDict w/ train/validation/test

    def to_tf_dataset(split):
        images = []
        labels = []
        for example in split:
            img = np.array(example["image"].convert("RGB"))
            images.append(img)
            labels.append(example["labels"])

        def gen():
            for img, lbl in zip(images, labels):
                yield img, lbl

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                tf.TensorSpec(shape=(), dtype=tf.int64),
            ),
        )

    train_ds = to_tf_dataset(hf_ds["train"])
    val_ds = to_tf_dataset(hf_ds["validation"])
    test_ds = to_tf_dataset(hf_ds["test"])
    return train_ds, val_ds, test_ds


def export_split_to_disk(ds, split_name, out_dir="../data"):
    """
    Materializes a tf.data.Dataset of (image, label) pairs to
    data/<split_name>/<class_name>/img_XXXX.jpg on disk, matching the
    required repo directory structure (data/train, data/test) and
    making the data directly usable by ImageDataGenerator-style
    pipelines, the retraining endpoint, and manual inspection.
    """
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(out_dir, split_name, class_name), exist_ok=True)

    counters = {c: 0 for c in CLASS_NAMES}
    for image, label in ds.as_numpy_iterator():
        class_name = CLASS_NAMES[int(label)]
        path = os.path.join(
            out_dir, split_name, class_name,
            f"{class_name}_{counters[class_name]:04d}.jpg"
        )
        tf.keras.utils.save_img(path, image)
        counters[class_name] += 1
    return counters


def load_dataset_from_directory(data_dir, validation_split=0.2, seed=42):
    """
    Loads images from data_dir/<class_name>/*.jpg into train/val
    tf.data.Datasets. Used by the API /retrain endpoint so newly
    uploaded images under data/train are included in the next training run.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=None,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=None,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
    )
    # Cast labels to int64 so they match the rest of the pipeline.
    train_ds = train_ds.map(
        lambda img, lbl: (img, tf.cast(lbl, tf.int64)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda img, lbl: (img, tf.cast(lbl, tf.int64)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------
def preprocess_image(image, label=None, augment=False):
    """
    Resize + normalize a single image to [0, 1] floats at IMG_SIZE.
    Optionally applies light augmentation (used only for the training
    split) to improve generalization on a relatively small dataset.
    """
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)

    if label is None:
        return image
    return image, label


def build_pipeline(ds, batch_size=32, augment=False, shuffle=False):
    """
    Turns a raw tf.data.Dataset of (image, label) pairs into a batched,
    preprocessed, prefetching pipeline ready for model.fit/evaluate.
    """
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(
        lambda img, lbl: preprocess_image(img, lbl, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def preprocess_image_from_path(path):
    """
    Load + preprocess a single image file from disk (e.g. an image a
    user uploads through the API/UI). Returns a (1, H, W, 3) batch
    ready for model.predict().
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = preprocess_image(image, augment=False)
    return tf.expand_dims(image, axis=0).numpy()


def preprocess_uploaded_bytes(file_bytes):
    """
    Same as above, but for raw bytes coming from an API upload
    (UploadFile.read() in FastAPI) instead of a file path.
    """
    image = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
    image = preprocess_image(image, augment=False)
    return tf.expand_dims(image, axis=0).numpy()


if __name__ == "__main__":
    train_raw, val_raw, test_raw = load_raw_datasets()
    print("Exporting train split...")
    print(export_split_to_disk(train_raw, "train"))
    print("Exporting test split...")
    print(export_split_to_disk(test_raw, "test"))
