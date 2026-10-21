"""
Comprehensive Evaluation & Plotting Script for Autonomous Drone Navigation.
Executes multi-episode test runs in DroneNavEnv-v1, records full telemetry
(trajectories, velocities, rewards, and obstacle clearances), and generates
a publication-quality 4-panel evaluation dashboard plot.
"""

import os
import sys

# Prevent OpenMP / TensorFlow clash in conda tf2.18
sys.modules['tensorflow'] = None
sys.modules['tensorboard'] = None

import argparse
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.drone_env import DroneNavEnv
from backend.server import compute_autonomous_action


def run_benchmark_and_plot(num_episodes: int = 8, output_image: str = "frontend/assets/evaluation_report.png"):
    env = DroneNavEnv()
    episodes_data = []

    print(f"\n========================================================")
    print(f"  RUNNING BENCHMARK EVALUATION ({num_episodes} EPISODES)")
    print(f"========================================================")

    for ep in range(num_episodes):
        obs, info = env.reset(seed=100 + ep * 7)
        done = False
        truncated = False
        steps = 0
        total_reward = 0.0

        drone_positions = [env.pos.copy()]
        drone_velocities = [np.linalg.norm(env.vel)]
        distances = [info.get("distance_to_goal", np.linalg.norm(env.goal_pos - env.pos))]
        rewards = []

        while not (done or truncated):
            action = compute_autonomous_action(obs)
            obs, reward, done, truncated, step_info = env.step(np.array(action, dtype=np.float32))
            total_reward += reward
            steps += 1

            drone_positions.append(env.pos.copy())
            drone_velocities.append(float(np.linalg.norm(env.vel)))
            distances.append(float(step_info.get("distance_to_goal", np.linalg.norm(env.goal_pos - env.pos))))
            rewards.append(reward)

        reached = step_info.get("goal_reached", False)
        collided = step_info.get("collision", False)
        status = "SUCCESS" if reached else ("COLLISION" if collided else "MAX_STEPS")

        print(f"  Episode {ep+1:02d}: Status={status:<11} | Steps={steps:3d} | "
              f"Reward={total_reward:6.1f} | Final Dist={distances[-1]:.2f}m")

        episodes_data.append({
            "episode": ep + 1,
            "status": status,
            "reached": reached,
            "collided": collided,
            "steps": steps,
            "total_reward": total_reward,
            "positions": np.array(drone_positions),
            "velocities": np.array(drone_velocities),
            "distances": np.array(distances),
            "rewards": rewards,
            "goal_pos": env.goal_pos.copy(),
            "spawn_pos": drone_positions[0],
            "obstacles": env.obstacles
        })

    # Summary metrics
    success_count = sum(1 for ep in episodes_data if ep["reached"])
    collision_count = sum(1 for ep in episodes_data if ep["collided"])
    success_rate = (success_count / num_episodes) * 100
    collision_rate = (collision_count / num_episodes) * 100
    avg_steps = np.mean([ep["steps"] for ep in episodes_data if ep["reached"]]) if success_count > 0 else 0
    mean_reward = np.mean([ep["total_reward"] for ep in episodes_data])

    print(f"--------------------------------------------------------")
    print(f"  Summary: Success Rate = {success_count}/{num_episodes} ({success_rate:.1f}%) | "
          f"Collisions = {collision_count}/{num_episodes} ({collision_rate:.1f}%)")
    print(f"  Mean Reward = {mean_reward:.1f} | Avg Steps to Goal = {avg_steps:.1f}")
    print(f"========================================================\n")

    # Plotting Style (Futuristic Dark Cyber Theme)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12), dpi=150)
    fig.patch.set_facecolor('#141722')

    palette_lines = ['#00f0ff', '#b388ff', '#00e676', '#ffd600', '#ff9100', '#ff5252', '#40c4ff', '#e040fb']

    # ----------------------------------------------------
    # Subplot 1: 2D Overhead Top-Down Trajectory Map
    # ----------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#1a1a24')
    ax1.set_title("2D Overhead Trajectory & Obstacle Field Map", fontsize=13, fontweight='bold', color='#00f0ff', pad=12)

    # Plot obstacles from the first episode field
    sample_obs = episodes_data[0]["obstacles"]
    for obs in sample_obs:
        if obs.kind in ['cylinder', 'prism']:
            circle = Circle((obs.pos[0], obs.pos[2]), obs.size[0], color='#434c5e', alpha=0.75, ec='#88c0d0', lw=1.5)
            ax1.add_patch(circle)
        else:
            hx, hz = obs.size[0], obs.size[2]
            rect = Rectangle((obs.pos[0] - hx, obs.pos[2] - hz), hx * 2, hz * 2,
                             color='#4c566a', alpha=0.75, ec='#d19a66', lw=1.5)
            ax1.add_patch(rect)

    # Plot drone paths
    for i, ep in enumerate(episodes_data):
        pos = ep["positions"]
        color = palette_lines[i % len(palette_lines)]
        label = f"Ep {ep['episode']} ({ep['status']})"
        ax1.plot(pos[:, 0], pos[:, 2], color=color, lw=2.2, alpha=0.9, label=label)

        # Spawn point marker
        ax1.scatter(pos[0, 0], pos[0, 2], color='#00e676', s=60, marker='o', zorder=5)
        # Goal point marker
        ax1.scatter(ep["goal_pos"][0], ep["goal_pos"][2], color='#ffd600', s=90, marker='*', zorder=6)

    ax1.set_xlim(-24, 24)
    ax1.set_ylim(-24, 24)
    ax1.set_xlabel("Arena X Coordinate (meters)", color='#8992a1', fontsize=10)
    ax1.set_ylabel("Arena Z Coordinate (meters)", color='#8992a1', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.25, color='#8992a1')
    ax1.legend(loc='upper right', fontsize=8, facecolor='#12161f', edgecolor='#3b4252')

    # ----------------------------------------------------
    # Subplot 2: 3D Flight Trajectories
    # ----------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_facecolor('#141722')
    ax2.xaxis.set_pane_color((0.1, 0.1, 0.14, 1.0))
    ax2.yaxis.set_pane_color((0.1, 0.1, 0.14, 1.0))
    ax2.zaxis.set_pane_color((0.1, 0.1, 0.14, 1.0))
    ax2.set_title("3D Autonomous Flight Trajectories (Altitude vs Position)", fontsize=13, fontweight='bold', color='#00f0ff', pad=12)

    for i, ep in enumerate(episodes_data):
        pos = ep["positions"]
        color = palette_lines[i % len(palette_lines)]
        ax2.plot(pos[:, 0], pos[:, 2], pos[:, 1], color=color, lw=2.0, alpha=0.85)
        # Goal 3D
        ax2.scatter(ep["goal_pos"][0], ep["goal_pos"][2], ep["goal_pos"][1], color='#ffd600', s=60, marker='*')

    ax2.set_xlim(-22, 22)
    ax2.set_ylim(-22, 22)
    ax2.set_zlim(0, 16)
    ax2.set_xlabel("X (m)", color='#8992a1', labelpad=6)
    ax2.set_ylabel("Z (m)", color='#8992a1', labelpad=6)
    ax2.set_zlabel("Altitude Y (m)", color='#8992a1', labelpad=6)
    ax2.view_init(elev=32, azim=-55)

    # ----------------------------------------------------
    # Subplot 3: Distance to Goal Convergence Profiles
    # ----------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#1a1a24')
    ax3.set_title("Distance to Target Waypoint over Time", fontsize=13, fontweight='bold', color='#00f0ff', pad=12)

    for i, ep in enumerate(episodes_data):
        color = palette_lines[i % len(palette_lines)]
        dists = ep["distances"]
        time_axis = np.arange(len(dists)) * 0.05  # dt = 0.05s
        ax3.plot(time_axis, dists, color=color, lw=2.0, alpha=0.85, label=f"Ep {ep['episode']}")

    # Threshold line
    ax3.axhline(1.25, color='#00e676', linestyle=':', lw=2.0, label='Goal Radius (1.25m)')
    ax3.set_xlabel("Flight Time (seconds)", color='#8992a1', fontsize=10)
    ax3.set_ylabel("Distance to Waypoint (meters)", color='#8992a1', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.25, color='#8992a1')
    ax3.legend(loc='upper right', fontsize=8, facecolor='#12161f', edgecolor='#3b4252')

    # ----------------------------------------------------
    # Subplot 4: Episodic Rewards & Success Rate Breakdown
    # ----------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor('#1a1a24')
    ax4.set_title(f"Benchmark Performance (Success Rate: {success_rate:.0f}% | Collisions: {collision_rate:.0f}%)",
                  fontsize=13, fontweight='bold', color='#00f0ff', pad=12)

    ep_indices = np.arange(1, num_episodes + 1)
    ep_rewards = [ep["total_reward"] for ep in episodes_data]
    bar_colors = ['#00e676' if ep["reached"] else ('#ff5252' if ep["collided"] else '#ffd600') for ep in episodes_data]

    bars = ax4.bar(ep_indices, ep_rewards, color=bar_colors, width=0.55, edgecolor='#ffffff', linewidth=0.5, alpha=0.85)

    for bar, ep in zip(bars, episodes_data):
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        ax4.text(bar.get_x() + bar.get_width() / 2, h + (2 if h >= 0 else -6),
                 f"{h:.0f}\n({ep['steps']}s)", ha='center', va=va, fontsize=8, color='#f0f3f6')

    ax4.axhline(0, color='#8992a1', linestyle='-', lw=0.8)
    ax4.set_xlabel("Evaluation Episode", color='#8992a1', fontsize=10)
    ax4.set_ylabel("Total Episodic Return (Reward)", color='#8992a1', fontsize=10)
    ax4.set_xticks(ep_indices)
    ax4.grid(True, linestyle='--', alpha=0.25, color='#8992a1', axis='y')

    # Status legend
    from matplotlib.lines import Line2D
    custom_legend = [
        Line2D([0], [0], color='#00e676', lw=4, label='Success (Waypoint Reached)'),
        Line2D([0], [0], color='#ff5252', lw=4, label='Collision (Terminated)'),
        Line2D([0], [0], color='#ffd600', lw=4, label='Max Steps (In Progress)'),
    ]
    ax4.legend(handles=custom_legend, loc='upper left', fontsize=8, facecolor='#12161f', edgecolor='#3b4252')

    plt.tight_layout(pad=3.0)

    # Save to both locations
    os.makedirs(os.path.dirname(os.path.abspath(output_image)), exist_ok=True)
    fig.savefig(output_image, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved evaluation dashboard plot to: {output_image}")

    # Also save a copy to backend/models
    models_copy = "backend/models/evaluation_report.png"
    os.makedirs(os.path.dirname(models_copy), exist_ok=True)
    fig.savefig(models_copy, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved evaluation dashboard plot copy to: {models_copy}")

    plt.close(fig)
    return episodes_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Benchmark & Generate Evaluation Plots")
    parser.add_argument("--episodes", type=int, default=8, help="Number of benchmark episodes")
    parser.add_argument("--output", type=str, default="frontend/assets/evaluation_report.png", help="Path to save report image")
    args = parser.parse_args()

    run_benchmark_and_plot(num_episodes=args.episodes, output_image=args.output)
