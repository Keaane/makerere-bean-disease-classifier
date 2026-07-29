#!/bin/sh
set -e

PORT="${PORT:-8000}"

# If Render mounted a persistent disk at /var/data, use it for models + training
# images so uploads/retrains survive redeploys. Seed from the image on first boot.
if [ -d /var/data ]; then
  mkdir -p /var/data/models /var/data/data/train /var/data/data/test

  if [ ! -f /var/data/models/bean_model_latest.h5 ] && [ -f /app/seed/models/bean_model_latest.h5 ]; then
    echo "Seeding model onto persistent disk..."
    cp -a /app/seed/models/. /var/data/models/
  fi

  # Seed train/test class folders if empty (ignore .gitkeep-only dirs).
  for split in train test; do
    count=$(find /var/data/data/$split -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l | tr -d ' ')
    if [ "${count:-0}" = "0" ] && [ -d /app/seed/data/$split ]; then
      echo "Seeding data/$split onto persistent disk..."
      cp -a /app/seed/data/$split/. /var/data/data/$split/
    fi
  done

  export BEAN_DATA_ROOT=/var/data
  export BEAN_MODEL_DIR=/var/data/models
  export BEAN_TRAIN_DIR=/var/data/data/train
fi

echo "Starting API on 0.0.0.0:${PORT} (model_dir=${BEAN_MODEL_DIR:-/app/models})"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export RETRAIN_MAX_PER_CLASS="${RETRAIN_MAX_PER_CLASS:-12}"
export RETRAIN_BATCH_SIZE="${RETRAIN_BATCH_SIZE:-2}"
export RETRAIN_MAX_EPOCHS="${RETRAIN_MAX_EPOCHS:-2}"
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --timeout-keep-alive 300
