# Autonomous Drone Navigation Using Deep Reinforcement Learning

A full-stack prototype for continuous 3D autonomous drone navigation and obstacle avoidance in procedural polyhedral environments using Deep Reinforcement Learning (PPO), Three.js, and FastAPI.

---

## 🚁 Architecture Overview

```
                          ┌────────────────────────────────────────────────────────┐
                          │               Three.js 3D Web Simulation               │
                          │   Solid Slate Sky (#8992a1) + Charcoal Ground (#1a1a24) │
                          │   Aerodynamic Pod Model (4 Vector Thruster Pads)       │
                          │   16-Ray Synthetic LIDAR + Polyhedral Obstacle Field   │
                          └───────────────────────────┬────────────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
            [Option A: Standalone In-Browser]                           [Option B: Full-Stack WebSocket]
            • Pure JavaScript MLP Forward-Pass                          • FastAPI Real-Time Streaming
            • Evaluates exported weights JSON (60 FPS)                  • Bidirectional ws://localhost:8000/ws/drone
            • 0.1ms inference latency                                   • PyTorch / SB3 PPO Policy Server
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                   ┌──────────────────┴──────────────────┐
                                   │  Gymnasium DroneNavEnv-v1 (Python)  │
                                   │  • State Space: 26 Continuous Floats│
                                   │  • Action Space: 4 Continuous Torques│
                                   │  • PPO (Proximal Policy Optimization)│
                                   └─────────────────────────────────────┘
```

---

## 📐 Reinforcement Learning Specification

### 1. Environment: `DroneNavEnv-v1` ([backend/drone_env.py](file:///Users/rayquaza/Documents/autonomus%20drone/backend/drone_env.py))
- Custom `gymnasium.Env` simulating 3D drone kinematics with aerodynamic drag, bank angle roll damping, and collision volumes.
- Procedural obstacle field featuring convex polyhedra (hexagonal prisms, dodecahedrons, low-poly monoliths, and pillars).

### 2. State / Observation Space ($S \in \mathbb{R}^{26}$)
| Index | Feature | Description | Range |
|---|---|---|---|
| `[0:3]` | Relative Target Vector | $(dx, dy, dz)$ relative to goal, normalized | $[-1.0, 1.0]$ |
| `[3:6]` | Linear Velocity | $(v_x, v_y, v_z)$ normalized by max speed ($7.5\text{ m/s}$) | $[-1.0, 1.0]$ |
| `[6:10]` | Attitude | $[\sin(\psi), \cos(\psi), \theta / \theta_{\max}, \phi / \phi_{\max}]$ (Yaw trig, Pitch, Roll) | $[-1.0, 1.0]$ |
| `[10:25]` | 16 LIDAR Ray Distances | 12 horizontal rays (30° intervals) + 2 diagonal elevation rays + 2 vertical | $[0.0, 1.0]$ |

### 3. Action Space ($A \in [-1.0, 1.0]^4$)
- `a[0]`: Pitch command (forward / backward thrust)
- `a[1]`: Roll command (lateral strafe thrust)
- `a[2]`: Yaw rate command (heading rotation around vertical axis)
- `a[3]`: Vertical thrust command (altitude up / down)

### 4. Reward Function ($R$)
$$R = 2.8 \times (d_{t-1} - d_t) - 0.01 - 0.005 \|a\|_2^2 + R_{\text{terminal}}$$
- **Progress reward:** $+2.8 \times \Delta d$ for closing Euclidean distance to the waypoint.
- **Step penalty:** $-0.01$ per step to encourage the shortest path trajectory.
- **Action regularizer:** $-0.005 \|a\|_2^2$ to encourage smooth actuation.
- **Goal reached:** $+100.0$ when within $1.25\text{ m}$ of waypoint polygon.
- **Collision:** $-50.0$ and episode termination for hitting obstacles or boundary.

---

## 🎨 Visual & Frontend Features

- **Minimalist Low-Poly Aesthetic:**
  - Flat slate/cream sky: `#8992a1`.
  - Solid deep charcoal ground: `#1a1a24`.
  - Solid matte convex polyhedra obstacles with flat shading (`flatShading: true`).
- **Futuristic Pod Drone:**
  - Aerodynamic capsule lozenge fuselage with dark carbon finish.
  - Sensor visor strip glowing cyan in Manual mode and purple in Autonomous mode.
  - 4 vector thruster pads with dynamic glowing exhaust rings that scale with engine power.
- **Telemetry HUD:**
  - Dual Mode toggle: `[MANUAL WASD]` $\leftrightarrow$ `[AUTONOMOUS RL]`.
  - Real-time coordinates $(X,Y,Z)$, pitch/roll/yaw, velocity $(m/s)$, and distance to goal.
  - Live collision counter, waypoints reached, and episodic reward accumulator.
  - LIDAR laser beam visualizer toggle with red hit-point indicators.
  - Multi-camera mode switcher: **Chase Cam**, **Orbit Cam**, **Top-Down Tactical**, and **FPV Visor**.

---

## 🚀 Quickstart Guide (Using Conda `tf2.18`)

### Prerequisites & Environment Setup
Ensure you activate your conda environment named `tf2.18`:
```bash
conda activate tf2.18
```

Verify installed dependencies:
```bash
python -m pip install gymnasium stable-baselines3 fastapi "uvicorn[standard]" websockets onnx onnxscript
```

---

### Step 1: Run Full-Stack Simulation Server
Start the FastAPI server, which hosts both the REST/WebSocket backend and the 3D frontend simulation:
```bash
conda activate tf2.18
python backend/server.py
```
Open your browser at **[http://localhost:8000](http://localhost:8000)**.

---

### Step 2: Train the PPO Reinforcement Learning Policy
To train a fresh PPO agent and automatically export weights to ONNX (`.onnx`), PyTorch (`.zip`), and JSON (`.json`):
```bash
conda activate tf2.18
python backend/train.py --timesteps 50000
```
- Model checkpoint is saved to: `backend/models/ppo_drone_final.zip`
- ONNX model is exported to: `backend/models/drone_policy.onnx`
- Browser neural weights are exported to: `frontend/js/drone_policy_weights.json`

---

### Step 3: Evaluate & Generate Publication-Quality Evaluation Graphs
Run the benchmark suite and generate the comprehensive 4-panel evaluation dashboard:
```bash
conda activate tf2.18
python backend/evaluate.py --plot --episodes 10
# or run directly:
python backend/plot_eval.py --episodes 10
```
- Saved plot image: `frontend/assets/evaluation_report.png` and `backend/models/evaluation_report.png`
- Generates 4 comprehensive panels:
  1. **2D Overhead Trajectory & Obstacle Field Map**
  2. **3D Flight Trajectories (Altitude vs Position)**
  3. **Distance to Waypoint Convergence Curves over Time**
  4. **Episodic Returns & Success/Collision Breakdown Bar Chart**
- Can also be viewed directly in the web app by clicking the **`[📊 EVAL REPORT]`** button in the top navigation bar!


---

## 🕹️ Flight Controls

| Key | Action |
|---|---|
| <kbd>W</kbd> / <kbd>S</kbd> | Pitch forward / backward |
| <kbd>A</kbd> / <kbd>D</kbd> | Yaw rotate left / right |
| <kbd>Q</kbd> / <kbd>E</kbd> | Strafe lateral left / right |
| <kbd>SPACE</kbd> | Vertical thrust up |
| <kbd>SHIFT</kbd> | Vertical thrust down |
| <kbd>M</kbd> | Toggle between Manual WASD and Autonomous RL |
| <kbd>R</kbd> | Reset drone pod to spawn point |

---

## 📁 Project Structure

```
.
├── backend/
│   ├── drone_env.py             # Custom Gymnasium Environment (DroneNavEnv-v1)
│   ├── train.py                 # PPO training & ONNX/JSON weight exporter
│   ├── evaluate.py              # Multi-episode evaluation script
│   ├── server.py                # FastAPI WebSocket & REST server (/ws/drone)
│   └── models/
│       ├── ppo_drone_final.zip  # Trained Stable-Baselines3 model
│       ├── drone_policy.onnx    # Standard ONNX exported policy graph
│       └── drone_policy_weights.json # MLP weights for zero-dependency inference
├── frontend/
│   ├── index.html               # 3D Simulation Web App
│   ├── css/
│   │   └── style.css            # Futuristic cybernetic glassmorphism HUD
│   └── js/
│       ├── main.js              # Three.js app orchestrator
│       ├── world.js             # Solid flat sky, charcoal ground & polyhedra
│       ├── drone.js             # Futuristic pod drone model & kinematics
│       ├── sensors.js           # 16-ray synthetic LIDAR & visualizer
│       ├── agent.js             # Dual Option A (Browser) & Option B (WebSocket)
│       ├── controls.js          # WASD flight controls & dynamic camera
│       ├── telemetry.js         # HUD telemetry & episodic reward metrics
│       └── drone_policy_weights.json # In-browser neural policy weights
├── package.json                 # Optional Vite / npm scripts
├── requirements.txt             # Python dependencies
└── README.md                    # System documentation
```
