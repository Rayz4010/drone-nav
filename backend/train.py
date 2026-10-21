"""
Reinforcement Learning Training Script for Autonomous Drone Navigation.
Uses Stable-Baselines3 (PPO) to train an obstacle avoidance and waypoint navigation agent.
Exports the trained policy to PyTorch (.zip), ONNX (.onnx), and JSON weights for browser inference.
"""

import os
import sys

# Prevent conflicting C++ OpenMP runtime clash between PyTorch and TensorFlow in conda tf2.18
sys.modules['tensorflow'] = None
sys.modules['tensorboard'] = None

import json
import argparse
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Ensure workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.drone_env import DroneNavEnv


class PyTorchPolicyExport(nn.Module):
    """
    Standalone PyTorch policy module for ONNX and direct forward-pass export.
    Takes observation vector (batch, 26) and outputs continuous actions (batch, 4).
    """
    def __init__(self, sb3_policy):
        super().__init__()
        # SB3 MlpPolicy uses policy_net (MLP extractor) and action_net (Linear)
        self.policy_extractor = sb3_policy.mlp_extractor.policy_net
        self.action_head = sb3_policy.action_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.policy_extractor(x)
        action = self.action_head(latent)
        # Apply tanh to guarantee action in [-1.0, 1.0] range
        return torch.tanh(action)


class MetricsCallback(BaseCallback):
    """Callback for monitoring episodic training rewards and success events."""
    def __init__(self, check_freq=2000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            mean_r = np.mean(self.episode_rewards[-20:])
            mean_l = np.mean(self.episode_lengths[-20:])
            if self.verbose > 0:
                print(f"[Step {self.n_calls:06d}] Mean Ep Reward (last 20): {mean_r:.2f} | Mean Ep Length: {mean_l:.1f}")
        return True


def export_weights_to_json(model: PPO, output_paths: list[str]):
    """
    Extracts MLP weights and biases from SB3 policy and exports to portable JSON.
    This enables pure JavaScript neural inference directly in Three.js without any runtime dependencies.
    """
    policy = model.policy
    extractor_layers = policy.mlp_extractor.policy_net
    action_layer = policy.action_net

    layers_data = []

    # Extract sequential layers in policy extractor
    for layer in extractor_layers:
        if isinstance(layer, nn.Linear):
            w = layer.weight.detach().cpu().numpy().tolist()
            b = layer.bias.detach().cpu().numpy().tolist()
            layers_data.append({
                "type": "linear",
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "weights": w,
                "bias": b,
                "activation": "tanh"
            })

    # Extract final action head
    layers_data.append({
        "type": "linear",
        "in_features": action_layer.in_features,
        "out_features": action_layer.out_features,
        "weights": action_layer.weight.detach().cpu().numpy().tolist(),
        "bias": action_layer.bias.detach().cpu().numpy().tolist(),
        "activation": "tanh"  # Bound continuous actions to [-1, 1]
    })

    export_obj = {
        "architecture": "MLP",
        "input_dim": 26,
        "output_dim": 4,
        "layers": layers_data
    }

    for path in output_paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_obj, f, indent=2)
        print(f"Exported policy weights to: {path}")


def export_policy_to_onnx(model: PPO, onnx_path: str):
    """Exports trained SB3 policy to standard ONNX graph."""
    export_module = PyTorchPolicyExport(model.policy)
    export_module.eval()

    dummy_input = torch.zeros(1, 26, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    try:
        torch.onnx.export(
            export_module,
            dummy_input,
            onnx_path,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
            opset_version=14,
        )
        print(f"Exported ONNX policy model to: {onnx_path}")
    except Exception as e:
        print(f"Notice: ONNX export encountered error: {e}")


def evaluate_policy(model: PPO, num_episodes: int = 5):
    """Evaluates the trained policy on a test environment."""
    print(f"\n--- Evaluating Policy across {num_episodes} Test Episodes ---")
    env = DroneNavEnv()
    successes = 0
    collisions = 0
    total_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            if done:
                if info.get("goal_reached", False):
                    successes += 1
                elif info.get("collision", False):
                    collisions += 1

        total_rewards.append(ep_reward)
        print(f"  Episode {ep+1}: Steps={steps}, Reward={ep_reward:.1f}, "
              f"GoalReached={info.get('goal_reached', False)}, Collision={info.get('collision', False)}")

    print(f"Results: Mean Reward = {np.mean(total_rewards):.1f} | Success Rate = {successes}/{num_episodes} "
          f"({successes/num_episodes*100:.0f}%) | Collisions = {collisions}/{num_episodes}\n")


def train(timesteps: int = 40000, save_dir: str = "backend/models"):
    """Main training routine using SB3 PPO."""
    os.makedirs(save_dir, exist_ok=True)

    # Wrap environment with Monitor for episodic metrics
    raw_env = DroneNavEnv()
    env = Monitor(raw_env)

    # PPO Policy with 2-layer MLP [128, 128] with Tanh activations
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    print(f"Initializing PPO agent with 26-dim observation and 4-dim continuous action space...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
    )

    print(f"Beginning PPO training for {timesteps} timesteps...")
    callback = MetricsCallback(check_freq=2048, verbose=1)
    model.learn(total_timesteps=timesteps, callback=callback)

    # Save SB3 zip model
    zip_path = os.path.join(save_dir, "ppo_drone_final.zip")
    model.save(zip_path)
    print(f"Saved trained SB3 model to: {zip_path}")

    # Export weights for standalone browser inference
    frontend_json = "frontend/js/drone_policy_weights.json"
    backend_json = os.path.join(save_dir, "drone_policy_weights.json")
    export_weights_to_json(model, [backend_json, frontend_json])

    # Export ONNX
    onnx_path = os.path.join(save_dir, "drone_policy.onnx")
    export_policy_to_onnx(model, onnx_path)

    # Quick evaluation
    evaluate_policy(model, num_episodes=5)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Drone Navigation Agent")
    parser.add_argument("--timesteps", type=int, default=30000, help="Total training timesteps")
    parser.add_argument("--save-dir", type=str, default="backend/models", help="Model output directory")
    args = parser.parse_args()

    train(timesteps=args.timesteps, save_dir=args.save_dir)
