"""
Policy Evaluation Script for Autonomous Drone Navigation.
Evaluates the trained PPO model across multiple procedural episodes,
printing trajectory efficiency, goal success rates, and collision statistics.
"""

import os
import sys

# Prevent OpenMP / TensorFlow conflict in conda tf2.18
sys.modules['tensorflow'] = None
sys.modules['tensorboard'] = None

import argparse
import numpy as np
from stable_baselines3 import PPO

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.drone_env import DroneNavEnv


def evaluate(model_path: str = "backend/models/ppo_drone_final.zip", num_episodes: int = 10):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train.py first.")
        return

    print(f"Loading trained PPO policy from: {model_path}")
    model = PPO.load(model_path, device="cpu")

    env = DroneNavEnv()
    successes = 0
    collisions = 0
    total_rewards = []
    episode_lengths = []

    print(f"\nRunning {num_episodes} evaluation episodes in procedural 3D obstacle fields...\n" + "-" * 60)

    for ep in range(num_episodes):
        obs, info = env.reset(seed=2000 + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, step_info = env.step(action)
            ep_reward += reward
            steps += 1

            if done:
                if step_info.get("goal_reached", False):
                    successes += 1
                elif step_info.get("collision", False):
                    collisions += 1

        total_rewards.append(ep_reward)
        episode_lengths.append(steps)
        outcome = "SUCCESS (Goal Reached)" if step_info.get("goal_reached") else ("COLLISION" if step_info.get("collision") else "MAX_STEPS")
        print(f"  Episode {ep+1:02d}: Outcome={outcome:<22} Steps={steps:3d} | Reward={ep_reward:6.1f} | Final Dist={step_info.get('distance_to_goal', 0):.2f}m")

    print("-" * 60)
    print(f"Evaluation Summary:")
    print(f"  Total Episodes     : {num_episodes}")
    print(f"  Success Rate       : {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"  Collision Rate     : {collisions}/{num_episodes} ({collisions/num_episodes*100:.1f}%)")
    print(f"  Mean Episodic Return: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.1f} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Drone RL Policy")
    parser.add_argument("--model", type=str, default="backend/models/ppo_drone_final.zip", help="Path to SB3 .zip model")
    parser.add_argument("--episodes", type=int, default=8, help="Number of evaluation episodes")
    parser.add_argument("--plot", action="store_true", help="Generate and save publication-quality 4-panel evaluation graphs")
    args = parser.parse_args()

    if args.plot:
        from backend.plot_eval import run_benchmark_and_plot
        run_benchmark_and_plot(num_episodes=args.episodes)
    else:
        evaluate(model_path=args.model, num_episodes=args.episodes)

