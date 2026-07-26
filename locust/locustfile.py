"""
locustfile.py
-------------
Load test for the Bean Leaf Disease Classifier API.

Simulates a flood of users hitting the /predict endpoint (the expensive,
model-inference path) and lighter traffic on /status (the monitoring
path), to measure latency and throughput under load — and to compare
behavior across different numbers of Docker container replicas.

Usage (headless, e.g. against the docker-compose nginx entrypoint):
    locust -f locustfile.py --host=http://localhost:8080 \
        --headless -u 50 -r 10 -t 60s \
        --csv=results/run_1container

Run once per container-count scenario (1, 2, 4 replicas), each time
saving a differently-named --csv prefix, then compare the *_stats.csv
summaries.
"""

import os

from locust import HttpUser, task, between

# Prefer a real leaf JPEG next to this file (no Pillow required).
_SAMPLE = os.path.join(os.path.dirname(__file__), "sample_leaf.jpg")
with open(_SAMPLE, "rb") as _f:
    _LEAF_BYTES = _f.read()


class BeanClassifierUser(HttpUser):
    # Simulated users wait a random short interval between actions,
    # like a real person looking at a result before the next upload.
    wait_time = between(0.5, 2.0)

    @task(5)
    def predict(self):
        files = {"file": ("leaf.jpg", _LEAF_BYTES, "image/jpeg")}
        self.client.post("/predict", files=files, name="/predict")

    @task(2)
    def status(self):
        self.client.get("/status", name="/status")

    @task(1)
    def class_distribution(self):
        self.client.get("/visualizations/class-distribution", name="/visualizations/class-distribution")
