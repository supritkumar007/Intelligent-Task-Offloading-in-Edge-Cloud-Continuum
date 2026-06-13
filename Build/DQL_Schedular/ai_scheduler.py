from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import requests
import os
import threading
import time
import uvicorn
import warnings
from sklearn.preprocessing import MinMaxScaler
import pickle
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ================= CONFIG =================
BACKEND_URLS = os.getenv(
    "BACKEND_URLS",
    "http://172.16.23.129:30081,"
    "http://172.16.23.129:30082,"
    "http://172.16.23.129:30083,"
    "http://172.16.23.129:30085,"
    "http://172.16.23.129:30086,"
    "http://172.16.23.129:30087,"
    "http://172.16.23.129:30088,"
    "http://172.16.23.129:30089"
    "http://172.16.23.129:30090"
).split(",")

BACKEND_URLS = [b.strip() for b in BACKEND_URLS if b.strip()]

CLOUD_SERVER = "http://172.16.23.128:8000"
CHECK_INTERVAL = 3

print("\n===== AI SCHEDULER STARTED =====")
print("Loaded BACKENDS:", BACKEND_URLS)
print("Cloud Inference:", CLOUD_SERVER)
print("================================\n")

# ================= MODEL =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

MODEL_PATH = "hybrid_dqn_9pods_balanced.pt"
SCALER_PATH = "state_scaler.pkl"

STATE_SIZE = 4
NUM_ACTIONS = len(BACKEND_URLS) + 1   # edges + cloud

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS)
        )

    def forward(self, x):
        return self.net(x)

RL_ENABLED = False
model = None
scaler = None

try:
    model = DQN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler: MinMaxScaler = pickle.load(f)

    RL_ENABLED = True
    print("Model & Scaler Loaded Successfully")

except Exception as e:
    print("❌ MODEL LOAD ERROR =====")
    print(str(e))
    print("Running PURE HEURISTIC Mode")
    print("================================")

# ================= LIMITS =================
CPU_LIMIT = 0.80
MEM_LIMIT = 0.85
QUEUE_LIMIT = 0.60

# ================= FASTAPI =================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

lock = threading.Lock()
latest_state = {}

# ================= METRICS =================
def fetch_metrics(url):
    try:
        r = requests.get(url + "/metrics", timeout=4)
        data = r.json()

        cpu = data.get("cpu_percent", 60) / 100.0
        mem = data.get("memory_percent", 60) / 100.0
        queue = data.get("queue", 0.3)
        active = data.get("current_requests", 0)

        if "instances" in data and data["instances"]:
            inst = data["instances"][-1]
            cpu = inst.get("cpu_percent", 60) / 100.0
            mem = inst.get("memory_percent", 60) / 100.0
            active = inst.get("current_requests", 0)

        task = np.random.uniform(0.2, 0.9)

        return [cpu, mem, queue, task, active]

    except:
        return None


def monitor():
    global latest_state
    while True:
        tmp = {}
        for url in BACKEND_URLS:
            s = fetch_metrics(url)
            if s:
                tmp[url] = s

        with lock:
            latest_state = tmp

        time.sleep(CHECK_INTERVAL)


@app.on_event("startup")
def startup():
    threading.Thread(target=monitor, daemon=True).start()

# ================= HEURISTIC DECISION =================
def choose_backend():
    with lock:
        if not latest_state:
            print("[FAILSAFE] No metrics → CLOUD")
            return CLOUD_SERVER

        nodes = list(latest_state.items())

    # Sort based on CPU, MEM, QUEUE, ACTIVE
    ranked = sorted(nodes, key=lambda x: (x[1][0], x[1][1], x[1][2], x[1][4]))
    url, state = ranked[0]
    cpu, mem, queue, task, active = state

    if cpu < CPU_LIMIT and mem < MEM_LIMIT and queue < QUEUE_LIMIT:
        print(f"[EDGE OK] Using EDGE => {url}")
        return url

    print("[OVERLOAD] All Edge Busy → CLOUD")
    return CLOUD_SERVER


# ================= PREDICT =================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()

    target = choose_backend()
    final_url = f"{target}/predict"
    print(f"[FORWARD] => {final_url}")

    try:
        r = requests.post(
            final_url,
            files={"file": (file.filename, image)},
            timeout=30
        )

        try:
            backend = r.json()
        except:
            backend = {"raw_response": r.text}

        return {
            "scheduler": "AI",
            "handled_by": target,
            "status": r.status_code,
            "response": backend
        }

    except Exception as e:
        return {
            "scheduler": "AI",
            "decision": "FAILED",
            "fallback": CLOUD_SERVER,
            "error": str(e)
        }


@app.get("/health")
def health():
    with lock:
        return {
            "status": "AI Scheduler Running",
            "rl_enabled": RL_ENABLED,
            "known_nodes": len(latest_state),
            "pods": list(latest_state.keys())
        }


if __name__ == "__main__":
    uvicorn.run("ai_scheduler:app", host="0.0.0.0", port=5001)

