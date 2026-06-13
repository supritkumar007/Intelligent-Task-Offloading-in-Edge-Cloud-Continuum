# Intelligent Task Offloading in Edge-Cloud Continuum Environment

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Kubernetes](https://img.shields.io/badge/MicroK8s-Kubernetes-326CE5?logo=kubernetes)
![OpenStack](https://img.shields.io/badge/Cloud-OpenStack-ED1944?logo=openstack)
![Open5GS](https://img.shields.io/badge/5G_Core-Open5GS-00ADEF)
![srsRAN](https://img.shields.io/badge/RAN-srsRAN-orange)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)

</div>

---

## Abstract

This project presents a **real-time intelligent task offloading framework** for a private 5G standalone (SA) network integrated with an edge-cloud computing continuum. The system enables mobile users (UEs) connected to the 5G network to submit image classification requests, which are dynamically routed to the optimal processing node—whether an edge server, cloud backup, or local execution—based on real-time network conditions and resource availability.

## Architecture Overview

<img width="378" height="640" alt="system" src="https://github.com/user-attachments/assets/42f2405e-b18d-4347-a126-ef8d08f264a5" />

The framework employs a hierarchical decision-making approach:

### Scheduling Strategies

| Strategy | Description |
|----------|-------------|
| **MILP-based Task Scheduling** | Mixed Integer Linear Programming optimization that minimizes end-to-end latency while respecting deadline and resource constraints. Uses real-time UE bandwidth and edge node utilization metrics. |
| **DQN-based AI Scheduler** | A Deep Q-Network trained on historical scheduling data that intelligently selects the optimal edge pod based on predicted reward functions. |
| **Heuristic Fallback** | Threshold-based routing using CPU, memory, and queue metrics when ML models are unavailable or unresponsive. |

Processed inference results are persisted to cloud storage for future model retraining and analytics.

---

## Testbed Configuration

| Component | Technology |
|-----------|------------|
| **Operating System** | Ubuntu LTS 22.04 |
| **SIM Card** | Programmable SIM (LTE/5G compatible) |
| **Radio Hardware** | USRP B210 Software Defined Radio |
| **UE Stack** | srsRAN (latest) |
| **5G Core** | Open5GS 2.7.x |
| **Edge Orchestration** | MicroK8s (Kubernetes) |
| **Container Runtime** | Docker |
| **Optimization Solver** | CBC / HiGHS |
| **Programming Language** | Python 3.10 |
| **API Framework** | FastAPI |
| **Cloud Infrastructure** | VMware Workstation (OpenStack DevStack) |

---

## Repository Structure

```
Offloading/
├── Build/
│   ├── DQL_Scheduler/              # Deep Q-Network AI Scheduler
│   │   ├── ai_scheduler.py         # FastAPI-based DQN scheduler with inference endpoint
│   │   ├── Dockerfile
│   │   ├── DQL_Training.ipynb      # Model training notebook
│   │   ├── hybrid_dqn.pt           # Trained DQN model weights
│   │   ├── state_scaler.pkl        # MinMaxScaler for state normalization
│   │   ├── scheduler_dataset.csv     # Training dataset
│   │   └── requirements.txt
│   │
│   ├── Edge-node/                  # CIFAR-10 Edge Inference Service
│   │   ├── server.py               # FastAPI inference server
│   │   ├── cnn_model_converted.h5  # Pre-trained CIFAR-10 CNN model
│   │   ├── edge-deployments.yaml    # K8s Deployment and Service manifests (9 nodes)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── steps.md
│   │
│   └── MILP_Selection/             # MILP-based Task Offloading Gateway
│       ├── milp_core.py            # Latency model and Pyomo MILP solver
│       ├── milp_gateway.py         # FastAPI gateway with routing logic
│       └── throughput_server.py      # Real-time UE bandwidth monitor via ogstun
│
└── Setup/                          # Comprehensive setup documentation
    ├── img/                        # Architecture diagrams and screenshots
    ├── 1. 5g-Setup-Guide.md
    ├── 2. Microk8s_edge_deployment.md
    ├── 3. scheduler-deployment.md
    ├── 4. milp-gateway-setup.md
    ├── 5. DevStack-Single Node Setup Guide.md
    ├── 5.1 VM launch guide.md
    ├── 6. cloud-storage-vm-setup.md
    └── 7. cloud-backup-docker-setup.md
```

---

## System Layers

### 1. UE Layer (User Equipment)
Mobile devices connected to the private 5G SA network transmit image classification requests through the data plane to the Task Offloading Scheduler.

### 2. Radio Access Network (RAN)
- **srsRAN Project gNB** running on the host machine
- **USRP B210** serving as the SDR front-end
- Operating on: Band n78, ARFCN 632628, 20 MHz bandwidth, 30 kHz SCS, PLMN 99970

### 3. 5G Core Network
- **Open5GS** deployed on Ubuntu VM (`172.16.23.129`)
- **UPF** manages GTP-U tunnelling; `ogstun` interface transports UE traffic
- **Throughput Server** monitors `ogstun` interface and exposes real-time bandwidth at `:7000/latest`

### 4. Edge Computing Layer (MicroK8s)
- **9 Edge Pods** (`edge-node1` through `edge-node9`): CIFAR-10 CNN inference services on NodePorts `30081–30090`
- **AI Scheduler** (DQN): NodePort `30084` — routes requests to the least-loaded edge pod
- **MILP Gateway**: Port `6001` — executes per-request optimization using real-time metrics

### 5. Cloud Layer (OpenStack DevStack)
- **Cloud Backup Node**: Docker container (`amogh0709/fastapi-application:v5`) on port `8000` — handles overflow requests
- **Cloud Storage VM**: Node.js Express server (`172.16.23.128:8080`) — accumulates processed results for retraining

---

## Network Configuration

| Component | IP Address | Port(s) |
|-----------|------------|---------|
| gNB (srsRAN) | `172.16.23.1` | — |
| Open5GS Core VM | `172.16.23.129` | — |
| Edge Nodes | `172.16.23.129` | `30081–30083`, `30085–30090` |
| AI Scheduler | `172.16.23.129` | `30084` |
| MILP Gateway | `172.16.23.129` | `6001` |
| Throughput Server | `172.16.23.129` | `7000` |
| Cloud Backup | `172.16.23.100` | `8000` |
| Cloud Storage VM | `172.16.23.128` | `8080` |
| UE IP Pool | `10.45.0.0/16` | — |

---

## Setup Guide

Follow the setup guides in numerical order:

| Step | Document | Purpose |
|------|----------|---------|
| 1 | `1. 5g-Setup-Guide.md` | Configure srsRAN gNB, Open5GS Core, and SIM programming |
| 2 | `2. Microk8s_edge_deployment.md` | Install MicroK8s and deploy 9 edge inference nodes |
| 3 | `3. scheduler-deployment.md` | Deploy the AI Scheduler (DQN) to Kubernetes |
| 4 | `4. milp-gateway-setup.md` | Set up MILP Gateway and Throughput Server |
| 5 | `5. DevStack-Single Node Setup Guide.md` | Install OpenStack DevStack (single-node) |
| 5.1 | `5.1 VM launch guide.md` | Launch Ubuntu VM instances on OpenStack |
| 6 | `6. cloud-storage-vm-setup.md` | Configure Node.js Cloud Storage server |
| 7 | `7. cloud-backup-docker-setup.md` | Deploy Docker-based Cloud Backup inference node |

---

## Quick Test

After completing the setup, test the inference pipeline:

```bash
# Test via AI Scheduler (DQN):
curl -X POST http://10.45.0.1:30084/predict \
  -F "file=@image.jpg"

# Test via MILP Gateway:
curl -X POST http://172.16.23.129:6001/milp_predict \
  -F "file=@image.jpg"
```

---

## Technology Stack

| Category | Technology |
|----------|------------|
| **5G RAN** | srsRAN Project, USRP B210 |
| **5G Core** | Open5GS, UPF |
| **Edge Orchestration** | MicroK8s (Kubernetes) |
| **Cloud Infrastructure** | OpenStack DevStack |
| **ML Scheduler** | PyTorch DQN, scikit-learn |
| **MILP Solver** | Pyomo, HiGHS / CBC |
| **Inference Server** | FastAPI, Uvicorn |
| **Cloud Storage** | Node.js, Express, Multer |
| **Containerisation** | Docker |
| **Model** | CIFAR-10 CNN (TensorFlow / Keras) |

---

## Project Screenshots

| System Architecture | Horizon Dashboard |
|---|---|
| <img src="Setup/img/system.png" width="300"> | <img src="Setup/img/horizon.png" width="300"> |

| MicroK8s Pod Status | MILP Decision Output |
|---|---|
| <img src="Setup/img/pods.png" width="300"> | <img src="Setup/img/milp.png" width="300"> |

| Throughput Results |
|---|
| <img src="Setup/img/throughput.png" width="300"> |

---

## Authors

**Supritkumar RP**  
B.E. Computer Science and Engineering  
KLE Technological University, Hubballi

> If you use, reference, or build upon this work, you **must** credit the original authors and link back to this repository. See the [License](#license) section.

---

## License

This project is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

### You are free to:
- **Share** — Copy and redistribute the material in any medium or format
- **Adapt** — Remix, transform, and build upon the material

### Under the following terms:
- **Attribution** — Credit must be given to the original authors, with a link to this repository. Indicate if changes were made.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

<div align="center">
<sub>© 2025 Supritkumar RP · KLE Technological University · All rights reserved under CC BY-NC 4.0</sub>
</div>