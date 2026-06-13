# ==========================
# milp_gateway.py
# ==========================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import requests
import pandas as pd
import uvicorn
import logging

from milp_core import milp_scheduler, DEBUG_LAST

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

EDGE_SERVICES = [
    ("edge1", "http://172.16.23.129:30081"),
    ("edge2", "http://172.16.23.129:30082"),
    ("edge3", "http://172.16.23.129:30083"),
    ("edge4", "http://172.16.23.129:30085"),
    ("edge5", "http://172.16.23.129:30086"),
    ("edge6", "http://172.16.23.129:30087"),
    ("edge7", "http://172.16.23.129:30088"),
    ("edge8", "http://172.16.23.129:30089"),
    ("edge9", "http://172.16.23.129:30090")
]


CLOUD_URL = "http://172.16.23.128:8000/predict"

PRIMARY_EDGE_URL = EDGE_SERVICES[0][1]


# ===============================
# DYNAMIC MODEL PROFILE
def get_model_profile():

    try:

        r = requests.get(
            f"{PRIMARY_EDGE_URL}/status",
            timeout=3
        ).json()

        task_flops  = r.get("model_flops",  120000000)
        deadline_ms = r.get("deadline_ms",  1000)

        logging.info(
            f"Model profile fetched from pod | "
            f"FLOPS={task_flops} | "
            f"Deadline={deadline_ms} ms"
        )

        return {
            "task_flops":  task_flops,
            "deadline_ms": deadline_ms
        }

    except Exception as e:

        logging.warning(
            f"Could not fetch model profile: {e} | "
            f"Using fallback defaults."
        )

        return {
            "task_flops":  120000000,
            "deadline_ms": 1000
        }


# ===============================
# REALTIME EDGE UTIL FETCH
# ===============================
def get_edge_realtime_util():

    edge_nodes = []

    for name, base in EDGE_SERVICES:

        try:

            r = requests.get(
                f"{base}/metrics",
                timeout=2
            ).json()

            util_vals = [
                x["cpu_percent"]
                for x in r["instances"]
            ]

            avg_util = sum(util_vals) / len(util_vals)

            edge_nodes.append({
                "id":   name,
                "f_j":  2.5e9,
                "util": avg_util / 100.0
            })

        except:

            edge_nodes.append({
                "id":   name,
                "f_j":  2.5e9,
                "util": 0.95    # assume busy if unreachable
            })

    return edge_nodes


# ===============================
# MAIN API
# ===============================
@app.post("/milp_predict")
async def milp_entry(
    file: UploadFile = File(...)
):

    file_bytes = await file.read()

    # ===================================
    # DYNAMIC TASK PROFILE
    # ===================================
    profile = get_model_profile()

    task_flops  = profile["task_flops"]
    deadline_ms = profile["deadline_ms"]

    data_mb = round(
        len(file_bytes) / (1024 * 1024),
        4
    )

    # ===================================
    # REAL UE THROUGHPUT FROM CORE
    # ===================================
    try:

        bw_resp = requests.get(
            "http://172.16.23.129:7000/latest",
            timeout=2
        ).json()

        bandwidth_mbps = float(
            bw_resp["bandwidth_mbps"]
        )

        logging.info(
            f"UE Throughput API: "
            f"{bandwidth_mbps:.2f} Mbps"
        )

    except Exception as e:

        logging.warning(
            f"Throughput API unavailable: {e}"
        )

        bandwidth_mbps = 20

    logging.info(
        f"Task Profile | "
        f"Size={data_mb} MB | "
        f"FLOPS={task_flops:.0f} | "
        f"Deadline={deadline_ms} ms | "
        f"BW={bandwidth_mbps} Mbps"
    )

    df = pd.DataFrame([{
        "task_flops":     task_flops,
        "deadline_ms":    deadline_ms,
        "data_mb":        data_mb,
        "bandwidth_mbps": bandwidth_mbps
    }])

    EDGE_LIST  = get_edge_realtime_util()
    CLOUD_LIST = [{"id": "cloud1", "f_k": 10e9, "util": 0.10}]

    decision = milp_scheduler(df, EDGE_LIST, CLOUD_LIST)[0]

    logging.info(f"MILP DECISION: {decision}")

    # ===================================
    # EDGE SELECTED
    # ===================================
    if decision.startswith("Edge"):

        edge_id = decision.split(":")[1]

        url = dict(EDGE_SERVICES)[edge_id] + "/predict"

        try:

            r = requests.post(
                url,
                files={
                    "file": (
                        file.filename,
                        file_bytes,
                        file.content_type
                    )
                },
                timeout=5
            )

            try:
                resp = r.json()
            except:
                resp = {"error": "Invalid edge response"}

        except Exception as e:
            resp = {"error": str(e)}

        # ==========================
        # Per-node metrics
        # Built from DEBUG_LAST which
        # milp_core populates after solve
        # ==========================
        edge_matrix = DEBUG_LAST.get("edge_matrix", [])

        node_metrics = {}

        for idx, e in enumerate(EDGE_LIST):

            node_id = e["id"]

            latency = (
                edge_matrix[0][idx]
                if edge_matrix and idx < len(edge_matrix[0])
                else None
            )

            node_metrics[node_id] = {
                "utilization": round(e["util"], 4),
                "latency":     round(latency, 6) if latency is not None else "N/A"
            }

        return JSONResponse({

            "decision":      decision,

            "mode":          "MILP_ONLY",

            "selected_node": edge_id,

            "task": {
                "task_flops":     task_flops,
                "deadline_ms":    deadline_ms,
                "data_mb":        data_mb,
                "bandwidth_mbps": bandwidth_mbps
            },

            "nodes": node_metrics,

            "scheduler_metrics": {
                "reason":           DEBUG_LAST.get("reason"),
                "deadline_seconds": DEBUG_LAST.get("deadline_seconds"),
                "final_decision":   DEBUG_LAST.get("final_decision"),
                "cloud_latency":    DEBUG_LAST.get("cloud_matrix"),
                "debug_steps":      DEBUG_LAST.get("debug_latency_steps")
            },

            "response": resp
        })

    # ===================================
    # CLOUD SELECTED
    # ===================================
    elif decision.startswith("Cloud"):

        logging.info(f"Forwarding to CLOUD -> {CLOUD_URL}")

        try:

            r = requests.post(
                CLOUD_URL,
                files={
                    "file": (
                        file.filename,
                        file_bytes,
                        file.content_type
                    )
                },
                timeout=10
            )

            resp = r.json()

        except Exception as e:
            resp = {"error": str(e)}

        return JSONResponse({
            "decision":  decision,
            "mode":      "MILP_ONLY",
            "final":     "CLOUD",
            "task": {
                "task_flops":     task_flops,
                "deadline_ms":    deadline_ms,
                "data_mb":        data_mb,
                "bandwidth_mbps": bandwidth_mbps
            },
            "scheduler_metrics": {
                "reason":           DEBUG_LAST.get("reason"),
                "deadline_seconds": DEBUG_LAST.get("deadline_seconds"),
                "final_decision":   DEBUG_LAST.get("final_decision")
            },
            "response": resp
        })

    # ===================================
    # LOCAL SELECTED
    # ===================================
    else:

        return JSONResponse({
            "decision": "Local",
            "mode":     "MILP_ONLY",
            "final":    "LOCAL",
            "task": {
                "task_flops":     task_flops,
                "deadline_ms":    deadline_ms,
                "data_mb":        data_mb,
                "bandwidth_mbps": bandwidth_mbps
            },
            "scheduler_metrics": {
                "reason":           DEBUG_LAST.get("reason"),
                "deadline_seconds": DEBUG_LAST.get("deadline_seconds"),
                "final_decision":   DEBUG_LAST.get("final_decision")
            },
            "output": "Processed locally (logical — no execution)"
        })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6001)
