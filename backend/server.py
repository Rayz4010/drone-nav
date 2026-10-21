"""
FastAPI WebSocket & REST Server for Autonomous Drone Navigation.
Connects the Python Reinforcement Learning agent to the Three.js 3D frontend.
Endpoint: /ws/drone for real-time bidirectional telemetry and action streaming (30-60 Hz).
"""

import os
import sys

# Prevent OpenMP / TensorFlow conflict in conda tf2.18
sys.modules['tensorflow'] = None
sys.modules['tensorboard'] = None

import json
import math
import time
import asyncio
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.drone_env import DroneNavEnv

app = FastAPI(
    title="Autonomous Drone Navigation RL Server",
    description="Backend for Deep Reinforcement Learning Drone Simulation",
    version="1.0.0"
)

# CORS middleware for local frontend development (e.g. Vite port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL = None
POLICY_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "models", "drone_policy_weights.json")
SB3_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ppo_drone_final.zip")


def load_model():
    """Attempts to load trained Stable-Baselines3 model."""
    global MODEL
    if os.path.exists(SB3_MODEL_PATH):
        try:
            from stable_baselines3 import PPO
            MODEL = PPO.load(SB3_MODEL_PATH, device="cpu")
            print(f"Loaded trained SB3 PPO policy from: {SB3_MODEL_PATH}")
            return
        except Exception as e:
            print(f"Warning: Could not load SB3 model: {e}")
    MODEL = None
    print("Notice: No trained SB3 model loaded; will use intelligent heuristic or JSON weights.")


# Load model on startup
load_model()


RAY_ANGLES = [
    (0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (90.0, 0.0),
    (120.0, 0.0), (150.0, 0.0), (180.0, 0.0), (210.0, 0.0),
    (240.0, 0.0), (270.0, 0.0), (300.0, 0.0), (330.0, 0.0),
    (0.0, 25.0), (0.0, -25.0), (0.0, 85.0), (0.0, -85.0)
]
RAY_LOCAL_DIRS = [
    (
        math.sin(math.radians(az)) * math.cos(math.radians(el)),
        math.sin(math.radians(el)),
        math.cos(math.radians(az)) * math.cos(math.radians(el))
    )
    for az, el in RAY_ANGLES
]


SECTOR_ANGLES_RAD = [
    math.radians((a + 180) % 360 - 180)
    for a in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
]


def compute_autonomous_action(obs_array: np.ndarray, prefer_expert: bool = True) -> list[float]:
    """
    Gap-Seeking Sector Navigator + PD Altitude Controller.
    Selects open horizontal sectors towards the goal, actively brakes for hazards,
    and maintains smooth, stable 3D altitude.
    """
    rel_goal = obs_array[0:3]
    vel = obs_array[3:6]
    attitude = obs_array[6:10]
    rays = obs_array[10:26] * 14.0  # distances in meters

    dx = float(rel_goal[0]) * 22.0
    dy = float(rel_goal[1]) * 22.0
    dz = float(rel_goal[2]) * 22.0
    curr_vy = float(vel[1]) * 7.5

    sin_yaw, cos_yaw = float(attitude[0]), float(attitude[1])

    # Goal in drone local frame (+Z Forward, +X Right)
    goal_body_x = dx * cos_yaw - dz * sin_yaw
    goal_body_z = dx * sin_yaw + dz * cos_yaw
    goal_body_angle = math.atan2(goal_body_x, goal_body_z)

    # 1. Evaluate 12 horizontal sectors
    best_score = -1e9
    best_angle = 0.0
    safe_dist = 4.5

    for i in range(12):
        angle = SECTOR_ANGLES_RAD[i]
        dist = float(rays[i])

        diff = abs(angle - goal_body_angle)
        while diff > math.pi:
            diff = abs(diff - 2.0 * math.pi)

        if dist < 2.5:
            score = -9999.0
        elif dist < safe_dist:
            score = -600.0 + (dist / safe_dist) * 200.0 - diff * 40.0
        else:
            score = min(dist, 10.0) * 2.0 - diff * 30.0

        if score > best_score:
            best_score = score
            best_angle = angle

    # 2. Yaw command: Steer towards best open sector
    yaw_rate = float(np.clip(best_angle * 3.5, -1.0, 1.0))

    # 3. Pitch command: Accelerate along heading, brake if obstacle ahead
    front_dist = min(float(rays[0]), float(rays[1]), float(rays[11]))
    if front_dist < 2.6:
        pitch = -0.6  # Emergency brake
    elif front_dist < 4.0:
        pitch = 0.35  # Caution
    elif abs(best_angle) < 0.5:
        pitch = 0.95  # Fast acceleration
    else:
        pitch = 0.2   # Rotate on the spot

    # 4. Roll command: Lateral evasion
    left_dist = min(float(rays[10]), float(rays[11]))
    right_dist = min(float(rays[1]), float(rays[2]))
    roll = 0.0
    if left_dist < 3.5 and left_dist < right_dist:
        roll = 0.5
    elif right_dist < 3.5 and right_dist < left_dist:
        roll = -0.5

    # 5. Altitude PD control
    target_vy = float(np.clip(dy * 2.2, -3.0, 3.0))
    thrust = float(np.clip((target_vy - curr_vy) * 1.6, -1.0, 1.0))

    return [pitch, roll, yaw_rate, thrust]




@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": MODEL is not None,
        "policy_weights_available": os.path.exists(POLICY_WEIGHTS_PATH),
        "timestamp": time.time(),
    }


@app.get("/api/config")
def get_config():
    """Returns environment specifications and sensor metadata."""
    sample_env = DroneNavEnv()
    return {
        "arena_size": sample_env.arena_size,
        "y_min": sample_env.y_min,
        "y_max": sample_env.y_max,
        "sensor_range": sample_env.sensor_range,
        "num_rays": sample_env.num_rays,
        "ray_angles": sample_env.ray_angles,
        "max_speed": sample_env.max_speed,
        "goal_reach_radius": sample_env.goal_reach_radius,
    }


@app.get("/api/weights")
def get_weights():
    """Returns policy weights JSON for in-browser zero-latency inference."""
    if os.path.exists(POLICY_WEIGHTS_PATH):
        with open(POLICY_WEIGHTS_PATH, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(status_code=404, content={"error": "Weights not found yet. Run train.py first."})


@app.websocket("/ws/drone")
async def websocket_drone_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket endpoint streaming at 30-60 Hz.
    Receives drone telemetry/observations and returns continuous RL actions.
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"WebSocket client connected from: {client_host}")

    step_count = 0
    t0 = time.time()

    try:
        while True:
            # Receive telemetry message from Three.js client
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type", "step")

            if msg_type == "telemetry":
                # Observation vector of dimension 26
                obs = msg.get("obs")
                if obs is not None and len(obs) == 26:
                    obs_arr = np.array(obs, dtype=np.float32)
                    action = compute_autonomous_action(obs_arr)

                    step_count += 1
                    fps = step_count / max(time.time() - t0, 0.001)

                    response = {
                        "type": "action",
                        "action": action,
                        "fps": round(fps, 1),
                        "server_time": time.time(),
                        "mode": msg.get("mode", "autonomous")
                    }
                    await websocket.send_text(json.dumps(response))

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "time": time.time()}))

            elif msg_type == "reload_model":
                load_model()
                await websocket.send_text(json.dumps({"type": "model_reloaded", "loaded": MODEL is not None}))

    except WebSocketDisconnect:
        print(f"WebSocket client disconnected: {client_host}")
    except Exception as e:
        print(f"WebSocket error: {e}")


# Mount frontend static directory at root so index.html, /css, and /js resolve directly
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("Starting Drone RL FastAPI Server on http://0.0.0.0:8000...")
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
