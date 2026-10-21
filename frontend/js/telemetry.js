/**
 * Telemetry Module: Real-Time HUD Metric Streaming, Live Telemetry Canvas Charts,
 * 360° LIDAR Polar Radar Scope, and Evaluation Dashboard Modal.
 */

import * as THREE from 'three';

export class TelemetryHUD {
  constructor() {
    // DOM Elements
    this.elPos = document.getElementById('tel-pos');
    this.elAttitude = document.getElementById('tel-attitude');
    this.elVel = document.getElementById('tel-vel');
    this.elDist = document.getElementById('tel-dist');
    this.elReward = document.getElementById('tel-reward');
    this.elCollisions = document.getElementById('tel-collisions');
    this.elGoals = document.getElementById('tel-goals');
    this.elFps = document.getElementById('tel-fps');
    this.elBadge = document.getElementById('flight-mode-badge');
    this.elToast = document.getElementById('event-toast');
    this.elWsIndicator = document.getElementById('ws-indicator');

    // Canvases for Live Graphs
    this.canvasDynamics = document.getElementById('dynamics-canvas');
    this.ctxDynamics = this.canvasDynamics ? this.canvasDynamics.getContext('2d') : null;

    this.canvasRadar = document.getElementById('radar-canvas');
    this.ctxRadar = this.canvasRadar ? this.canvasRadar.getContext('2d') : null;

    // Evaluation Modal DOM
    this.modal = document.getElementById('eval-modal');
    this.btnOpenEval = document.getElementById('btn-open-eval');
    this.btnCloseModal = document.getElementById('btn-close-modal');

    // Counters & History Buffers
    this.episodicReward = 0.0;
    this.collisionCount = 0;
    this.goalsReachedCount = 0;
    this.prevDistance = 0.0;

    this.maxHistory = 60;
    this.distHistory = new Array(this.maxHistory).fill(15.0);
    this.altHistory = new Array(this.maxHistory).fill(3.5);
    this.chartTimer = 0;

    this.toastTimeout = null;

    this._setupModalListeners();
  }

  _setupModalListeners() {
    if (this.btnOpenEval && this.modal) {
      this.btnOpenEval.addEventListener('click', () => {
        this.modal.classList.remove('hidden');
      });
    }

    if (this.btnCloseModal && this.modal) {
      this.btnCloseModal.addEventListener('click', () => {
        this.modal.classList.add('hidden');
      });
    }

    if (this.modal) {
      this.modal.addEventListener('click', (e) => {
        if (e.target === this.modal) {
          this.modal.classList.add('hidden');
        }
      });
    }
  }

  setMode(isAutonomous) {
    if (this.elBadge) {
      if (isAutonomous) {
        this.elBadge.textContent = 'AUTONOMOUS [RL]';
        this.elBadge.className = 'badge badge-auto';
      } else {
        this.elBadge.textContent = 'MANUAL [WASD]';
        this.elBadge.className = 'badge badge-manual';
      }
    }
  }

  setWsStatus(connected) {
    if (this.elWsIndicator) {
      if (connected) {
        this.elWsIndicator.className = 'status-indicator online';
        this.elWsIndicator.title = 'FastAPI WebSocket connected (30-60 Hz)';
      } else {
        this.elWsIndicator.className = 'status-indicator offline';
        this.elWsIndicator.title = 'FastAPI WebSocket offline (using browser engine)';
      }
    }
  }

  showToast(message, isSuccess = false, durationMs = 2000) {
    if (!this.elToast) return;
    this.elToast.textContent = message;
    this.elToast.className = isSuccess ? 'toast success' : 'toast';

    if (this.toastTimeout) clearTimeout(this.toastTimeout);
    this.toastTimeout = setTimeout(() => {
      this.elToast.className = 'toast hidden';
    }, durationMs);
  }

  onGoalReached() {
    this.goalsReachedCount += 1;
    this.episodicReward += 100.0;
    if (this.elGoals) this.elGoals.textContent = this.goalsReachedCount;
    this.showToast(`🎯 WAYPOINT REACHED! (+100 REWARD)`, true, 2200);
  }

  onCollision(obstacleKind = 'Obstacle') {
    this.collisionCount += 1;
    this.episodicReward -= 50.0;
    if (this.elCollisions) this.elCollisions.textContent = this.collisionCount;
    this.showToast(`⚠️ COLLISION DETECTED: ${obstacleKind.toUpperCase()} (-50)`, false, 2000);
  }

  resetEpisode() {
    this.episodicReward = 0.0;
  }

  _renderDynamicsChart() {
    if (!this.ctxDynamics || !this.canvasDynamics) return;
    const ctx = this.ctxDynamics;
    const w = this.canvasDynamics.width;
    const h = this.canvasDynamics.height;

    ctx.clearRect(0, 0, w, h);

    // Subtle background grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let y = 15; y < h; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // 1. Distance to goal curve (Cyan line)
    ctx.beginPath();
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 1.8;
    const maxDist = 28.0;
    for (let i = 0; i < this.distHistory.length; i++) {
      const x = (i / (this.distHistory.length - 1)) * w;
      const normalizedD = THREE.MathUtils.clamp(this.distHistory[i] / maxDist, 0, 1);
      const y = h - normalizedD * (h - 8) - 4;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 2. Altitude curve (Green line)
    ctx.beginPath();
    ctx.strokeStyle = '#00e676';
    ctx.lineWidth = 1.8;
    const maxAlt = 16.0;
    for (let i = 0; i < this.altHistory.length; i++) {
      const x = (i / (this.altHistory.length - 1)) * w;
      const normalizedA = THREE.MathUtils.clamp(this.altHistory[i] / maxAlt, 0, 1);
      const y = h - normalizedA * (h - 8) - 4;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  _renderRadarScope(drone, goalPos, sensorDistances) {
    if (!this.ctxRadar || !this.canvasRadar) return;
    const ctx = this.ctxRadar;
    const w = this.canvasRadar.width;
    const h = this.canvasRadar.height;
    const cx = w / 2;
    const cy = h / 2;
    const rMax = Math.min(cx, cy) - 8;

    ctx.clearRect(0, 0, w, h);

    // Range rings (5m, 10m, 14m)
    ctx.strokeStyle = 'rgba(137, 146, 161, 0.2)';
    ctx.lineWidth = 1;
    [0.35, 0.7, 1.0].forEach(factor => {
      ctx.beginPath();
      ctx.arc(cx, cy, rMax * factor, 0, Math.PI * 2);
      ctx.stroke();
    });

    // Crosshairs
    ctx.beginPath();
    ctx.moveTo(cx - rMax, cy);
    ctx.lineTo(cx + rMax, cy);
    ctx.moveTo(cx, cy - rMax);
    ctx.lineTo(cx, cy + rMax);
    ctx.stroke();

    // 16 LIDAR Ray Vectors (Top-Down horizontal rays)
    if (sensorDistances && sensorDistances.length >= 12) {
      for (let i = 0; i < 12; i++) {
        const azDeg = i * 30.0;
        const angle = THREE.MathUtils.degToRad(azDeg - 90); // 0 deg is UP on radar
        const distNorm = THREE.MathUtils.clamp(sensorDistances[i], 0.05, 1.0);
        const rayLen = rMax * distNorm;

        const x = cx + Math.cos(angle) * rayLen;
        const y = cy + Math.sin(angle) * rayLen;

        ctx.beginPath();
        ctx.strokeStyle = distNorm < 0.25 ? 'rgba(255, 82, 82, 0.6)' : 'rgba(0, 240, 255, 0.35)';
        ctx.lineWidth = 1;
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.stroke();

        // Hit point dot
        if (distNorm < 0.95) {
          ctx.beginPath();
          ctx.arc(x, y, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = distNorm < 0.25 ? '#ff5252' : '#ffd600';
          ctx.fill();
        }
      }
    }

    // Direction to Target Goal (Yellow glowing needle/pointer)
    if (goalPos && drone) {
      const dx = goalPos.x - drone.pos.x;
      const dz = goalPos.z - drone.pos.z;
      const sinYaw = Math.sin(drone.yaw);
      const cosYaw = Math.cos(drone.yaw);

      // Goal in drone local frame (+Z Forward, +X Right)
      const goalBodyX = dx * cosYaw - dz * sinYaw;
      const goalBodyZ = dx * sinYaw + dz * cosYaw;
      const goalAngle = Math.atan2(goalBodyX, goalBodyZ) - Math.PI / 2;

      const gx = cx + Math.cos(goalAngle) * (rMax - 2);
      const gy = cy + Math.sin(goalAngle) * (rMax - 2);

      ctx.beginPath();
      ctx.arc(gx, gy, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#ffd600';
      ctx.shadowColor = '#ffd600';
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Center Drone Pod Symbol
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#00f0ff';
    ctx.fill();
  }

  update(drone, goalPos, latencyMs = 0.2, fps = 60, sensorDistances = null) {
    // 1. Coordinates (X, Y, Z)
    if (this.elPos) {
      this.elPos.textContent = `${drone.pos.x.toFixed(1)}, ${drone.pos.y.toFixed(1)}, ${drone.pos.z.toFixed(1)}`;
    }

    // 2. Attitude (Pitch / Roll / Yaw in degrees)
    if (this.elAttitude) {
      const pDeg = THREE.MathUtils.radToDeg(drone.pitch).toFixed(0);
      const rDeg = THREE.MathUtils.radToDeg(drone.roll).toFixed(0);
      const yDeg = THREE.MathUtils.radToDeg(drone.yaw).toFixed(0);
      this.elAttitude.textContent = `P:${pDeg}° R:${rDeg}° Y:${yDeg}°`;
    }

    // 3. Linear Velocity (m/s)
    const speed = drone.vel.length();
    if (this.elVel) {
      this.elVel.textContent = `${speed.toFixed(2)} m/s`;
    }

    // 4. Distance to Goal
    const currDist = drone.pos.distanceTo(goalPos);
    if (this.elDist) {
      this.elDist.textContent = `${currDist.toFixed(1)} m`;
    }

    // Progress reward step update
    if (this.prevDistance > 0) {
      const delta = (this.prevDistance - currDist) * 3.5 - 0.01;
      this.episodicReward += delta;
    }
    this.prevDistance = currDist;

    // 5. Reward
    if (this.elReward) {
      const prefix = this.episodicReward >= 0 ? '+' : '';
      this.elReward.textContent = `${prefix}${this.episodicReward.toFixed(2)}`;
      this.elReward.className = this.episodicReward >= 0 ? 'metric-val text-green' : 'metric-val text-red';
    }

    // 6. FPS / Latency
    if (this.elFps) {
      this.elFps.textContent = `${Math.round(fps)} FPS (${latencyMs.toFixed(1)}ms)`;
    }

    // 7. Push to Live Graph history buffers (~10 Hz update)
    this.chartTimer++;
    if (this.chartTimer >= 6) {
      this.chartTimer = 0;
      this.distHistory.shift();
      this.distHistory.push(currDist);
      this.altHistory.shift();
      this.altHistory.push(drone.pos.y);
      this._renderDynamicsChart();
    }

    // 8. Render 360° LIDAR Radar Scope
    this._renderRadarScope(drone, goalPos, sensorDistances);
  }
}
