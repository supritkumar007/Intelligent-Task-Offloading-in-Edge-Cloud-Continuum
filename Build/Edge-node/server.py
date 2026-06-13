import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from io import BytesIO
from PIL import Image
import requests
import psutil
import os
import threading
import time
from collections import deque


# ===============================
# CONFIG
# ===============================
REMOTE_SERVER_URL = os.getenv("REMOTE_SERVER_URL", "http://default-server:8080/data")

model = tf.keras.models.load_model('./cnn_model_converted.h5')

labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ===============================
# MODEL METADATA
# (single source of truth for this pod)
# ===============================
MODEL_METADATA = {
    "model_name": "cnn_model_converted",
    "task_flops": 120000000,
    "input_size": [32, 32, 3],
    "application": "image_classification",
    "deadline_ms": 1000
}

app = FastAPI()

current_requests = 0
lock = threading.Lock()

# Zero snapshot
zero_snapshot = {
    "current_requests": 0,
    "cpu_percent": 0.0,
    "memory_percent": 0.0,
    "net_bytes_sent": 0,
    "net_bytes_recv": 0,
    "num_threads": 0
}

# Store last 10 metric samples
metrics_window = deque([zero_snapshot.copy() for _ in range(10)], maxlen=10)


# ===============================
# SEND IMAGE TO CLOUD
# ===============================
def send_image_to_remote_server(image_bytes: bytes, filename: str):
    try:
        files = {'file': (filename, image_bytes, 'image/jpeg')}
        response = requests.post(REMOTE_SERVER_URL, files=files, timeout=5)
        response.raise_for_status()
    except Exception as e:
        print(f"Error sending image to remote server: {e}")


# ===============================
# BACKGROUND METRIC COLLECTION
# ===============================
def capture_system_metrics():
    while True:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        net_io = psutil.net_io_counters()
        p = psutil.Process()

        snapshot = {
            "current_requests": current_requests,
            "cpu_percent": cpu,
            "memory_percent": mem,
            "net_bytes_sent": net_io.bytes_sent,
            "net_bytes_recv": net_io.bytes_recv,
            "num_threads": p.num_threads()
        }

        with lock:
            metrics_window.append(snapshot)

        time.sleep(1)


@app.on_event("startup")
def startup_event():
    threading.Thread(target=capture_system_metrics, daemon=True).start()
    print("Background system metrics capturing started.")


# ===============================
# STATUS
# Extended: now includes model_flops, deadline_ms, queue_length
# Gateway reads ONLY this endpoint — no need for /model_info separately
# ===============================
@app.get("/status")
def status():
    cpu = psutil.cpu_percent(interval=0.1) / 100
    mem = psutil.virtual_memory().percent / 100

    queue = min(current_requests / 10, 1.0)
    task_load = np.clip(current_requests * 0.1, 0, 1)

    net = psutil.net_io_counters()
    p = psutil.Process()

    return {
        # ── Runtime metrics ──────────────────────
        "cpu_percent": cpu,
        "memory_percent": mem,
        "queue": queue,
        "task": task_load,
        "net_bytes_sent": net.bytes_sent,
        "net_bytes_recv": net.bytes_recv,
        "threads": p.num_threads(),
        "current_requests": current_requests,

        # ── Model metadata (Phase 2 addition) ────
        "model_flops": MODEL_METADATA["task_flops"],
        "deadline_ms": MODEL_METADATA["deadline_ms"],
        "queue_length": current_requests
    }


# ===============================
# MODEL INFO
# Kept for direct inspection / curl testing
# ===============================
@app.get("/model_info")
def model_info():
    return MODEL_METADATA


# ===============================
# PREDICT
# ===============================
@app.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    global current_requests

    with lock:
        current_requests += 1

    try:
        image_bytes = await file.read()

        img = Image.open(BytesIO(image_bytes)).resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = labels[predicted_index]

        if background_tasks:
            background_tasks.add_task(
                send_image_to_remote_server,
                image_bytes,
                file.filename
            )

        return {
            "class": predicted_label,
            "current_requests": current_requests
        }

    finally:
        with lock:
            current_requests -= 1


# ===============================
# METRICS
# ===============================
@app.get("/metrics")
def get_metrics():
    with lock:
        metrics_list = list(metrics_window)
    return {"instances": metrics_list}
