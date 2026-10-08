# POLARIS: Research Exemplars & Benchmark Replication Suite

This directory contains research exemplars, academic baseline algorithms, workload traces, and scientific replication tooling for the **POLARIS** framework (*Proactive Optimization & Learning Architecture for Resilient Intelligent Systems*, arXiv:2512.04702v2 / ICSE / TAAS submission).

---

## 1. Architectural Segregation

POLARIS is designed as a **production-grade autonomous MAPE-K control plane** for cloud-native infrastructure, Kubernetes clusters, and microservices.

To maintain clean separation of concerns:
- **Core Production Tooling (`polaris/`)**: General-purpose runtime orchestration, neuro-symbolic runtime verifiers, calibrated world models, LLM multi-agent strategies, and Kubernetes/HTTP connectors.
- **Benchmark & Research Suite (`benchmarks/`)**: Benchmark exemplars, academic baselines (e.g. AdaMLS, Reactive rules), simulation adapters, and automated statistical replication scripts.

```
benchmarks/
├── exemplars/                          # Managed system benchmark exemplars
│   ├── swim/                           # SWIM web infrastructure (TCP socket)
│   ├── switch/                         # SWITCH YOLOv5 vision model switching (REST / Synthetic)
│   ├── wildfire/                       # Wildfire UAV search & rescue (Mesa simulation REST)
│   └── suave/                          # SUAVE underwater robotics (ROS bridge)
├── baselines/                          # Academic baseline strategies
│   ├── adamls.py                       # AdaMLS baseline strategy (ASE 2023)
│   └── suave_threshold.py              # SUAVE reactive threshold baseline
├── results/                            # Generated publication LaTeX tables & markdown reports
├── reproduce_paper.py                  # Master scientific reproduction harness
└── run_benchmark.sh                    # Automated end-to-end experiment executor
```

---

## 2. Quick Start: Reproducing Paper Results (Tables 2, 3, 4 & Statistical Tests)

To reproduce the benchmark comparison tables and rigorous non-parametric statistical significance tests (Mann-Whitney U, Vargha-Delaney $\hat{A}_{12}$ effect sizes) reported in the paper:

```bash
# Run fast deterministic replication across 5 experimental seeds
python benchmarks/reproduce_paper.py --mode fast --seeds 5
```

This compiles:
- `benchmarks/results/table2_swim.tex`: Table 2 (SWIM Baselines vs. POLARIS)
- `benchmarks/results/table3_switch.tex`: Table 3 (SWITCH / AdaMLS vs. POLARIS)
- `benchmarks/results/statistical_significance.md`: Formal empirical software engineering report

---

## 3. Supported Exemplars

### A. SWIM (Simulated Web Infrastructure Manager)
- **Domain**: Cloud web service auto-scaling and optional-content admission control.
- **Metrics**: `average_response_time`, `average_utilization`, `server_count`, `active_servers`, `dimmer`.
- **Actions**: `scale_up`, `scale_down`, `set_dimmer`.
- **Directory**: `benchmarks/exemplars/swim/`
- **Execution**:
  ```bash
  # Launch container and run POLARIS experiment
  ./scripts/run_experiment.sh --exemplar swim --config benchmarks/exemplars/swim/config.yaml
  ```

### B. SWITCH (Adaptive Machine Learning-Enabled System)
- **Domain**: Machine learning object detection with dynamic model switching (YOLOv5n/s/m/l/x on COCO 2017).
- **Paper**: Kulkarni et al., *"Towards Self-Adaptive Machine Learning-Enabled Systems Through QoS-Aware Model Switching"*, IEEE/ACM ASE 2023.
- **Metrics**: `confidence_mean`, `response_time`, `cpu_usage`, `inference_rate`, `switch_count`.
- **Actions**: `switch_model` (e.g. `{"model_name": "yolov5s"}`).
- **Directory**: `benchmarks/exemplars/switch/`
- **Execution**:
  ```bash
  # Run in high-fidelity zero-dependency synthetic simulation mode
  polaris --config benchmarks/exemplars/switch/config.yaml
  ```

### C. Wildfire-UAVSim
- **Domain**: Multi-agent UAV fleet coordinated search-and-rescue over burning forest grid.
- **Metrics**: `fire_cells_burning`, `fire_cells_total`, `num_agents`, `mr1_values`, `mr2_value`.
- **Actions**: `wildfire_move` (directions: north, south, east, west, hold), `wildfire_reset`.
- **Directory**: `benchmarks/exemplars/wildfire/`
- **Execution**:
  ```bash
  # 1. Start Mesa simulation adapter
  python benchmarks/exemplars/wildfire/adapter.py

  # 2. Start Polaris controller
  polaris --config benchmarks/exemplars/wildfire/config.yaml
  ```

### D. SUAVE (Underwater Autonomous Vehicle)
- **Domain**: Autonomous underwater vehicle thruster recovery and visibility adaptation.
- **Metrics**: `water_visibility`, `thruster_failure`, `motion_status`.
- **Actions**: `change_mode` (`fd_spiral_low`, `fd_spiral_medium`, `fd_recover_thrusters`).
- **Directory**: `benchmarks/exemplars/suave/`
- **Execution**:
  ```bash
  polaris --config benchmarks/exemplars/suave/config.yaml
  ```

---

## 4. Academic Baseline Strategies

The `benchmarks/baselines/` directory houses reference adaptive strategies:
- `adamls.py`: QoS-Aware Model Switching baseline implementing sliding-window SLA enforcement.
- `suave_threshold.py`: Reactive threshold baseline reacting to underwater visibility and thruster failures.

These strategies can be bound to any POLARIS pipeline via YAML configuration:
```yaml
strategy:
  type: "adamls"
  params:
    latency_sla: 0.15
    cpu_sla: 70.0
    confidence_target: 0.65
```
