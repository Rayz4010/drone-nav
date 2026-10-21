/**
 * Main Application Module: Autonomous Drone Navigation 3D Simulation.
 * Coordinates Three.js render loop, physics step, sensor raycasts,
 * DRL autonomous inference, and user interaction.
 */

import * as THREE from 'three';
import { World } from './world.js';
import { DronePod } from './drone.js';
import { LidarSensor } from './sensors.js';
import { RLAgent } from './agent.js';
import { FlightControls } from './controls.js';
import { TelemetryHUD } from './telemetry.js';

class SimulationApp {
  constructor() {
    this.container = document.getElementById('canvas-container');

    // 1. Core Three.js setup
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      200
    );
    this.camera.position.set(0, 5, -8);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // 2. Subsystems
    this.world = new World(this.scene);
    this.drone = new DronePod(this.scene);
    this.sensors = new LidarSensor(this.scene, this.drone);
    this.agent = new RLAgent();
    this.controls = new FlightControls(this.camera, this.renderer.domElement, this.drone);
    this.hud = new TelemetryHUD();

    // Simulation loop timing
    this.clock = new THREE.Clock();
    this.fpsCounter = 60;
    this.frameCount = 0;
    this.lastFpsUpdate = 0;

    // Connect subsystems & DOM event listeners
    this._bindEvents();

    // Start render loop
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  _bindEvents() {
    window.addEventListener('resize', () => this._onWindowResize());

    // Mode Toggle Buttons
    const btnManual = document.getElementById('btn-mode-manual');
    const btnAuto = document.getElementById('btn-mode-autonomous');

    btnManual.addEventListener('click', () => {
      this.controls.setAutonomousMode(false);
      btnManual.classList.add('active');
      btnAuto.classList.remove('active');
      this.hud.setMode(false);
    });

    btnAuto.addEventListener('click', () => {
      this.controls.setAutonomousMode(true);
      btnAuto.classList.add('active');
      btnManual.classList.remove('active');
      this.hud.setMode(true);
    });

    this.controls.onModeChange = (isAuto) => {
      if (isAuto) {
        btnAuto.classList.add('active');
        btnManual.classList.remove('active');
      } else {
        btnManual.classList.add('active');
        btnAuto.classList.remove('active');
      }
      this.hud.setMode(isAuto);
    };

    this.controls.onResetDrone = () => this._resetDrone();

    // Engine Selector (Option A vs Option B)
    const engineSelect = document.getElementById('engine-select');
    engineSelect.addEventListener('change', (e) => {
      this.agent.setEngineMode(e.target.value);
    });

    this.agent.onWsStatusChange = (online) => {
      this.hud.setWsStatus(online);
    };

    // Sensor LIDAR Toggle
    const btnLidar = document.getElementById('btn-toggle-lidar');
    btnLidar.addEventListener('click', () => {
      const visible = this.sensors.toggleVisibility();
      btnLidar.textContent = visible ? 'VISIBLE' : 'HIDDEN';
      btnLidar.classList.toggle('active', visible);
    });

    // Camera Selector
    const cameraSelect = document.getElementById('camera-select');
    cameraSelect.addEventListener('change', (e) => {
      this.controls.setCameraMode(e.target.value);
    });

    // Procedural World Randomize
    const btnRegen = document.getElementById('btn-regen-world');
    btnRegen.addEventListener('click', () => {
      this.world.generateProceduralWorld();
      this.world.spawnRandomGoal(8.0, this.drone.pos);
      this.hud.showToast('Generated New Procedural Obstacle Field', true, 1800);
    });

    // New Goal Waypoint
    const btnNewGoal = document.getElementById('btn-new-goal');
    btnNewGoal.addEventListener('click', () => {
      this.world.spawnRandomGoal(8.0, this.drone.pos);
      this.hud.showToast('Target Waypoint Relocated', true, 1500);
    });
  }

  _onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  _resetDrone() {
    const spawnPos = this.world.getSafeSpawnPosition();
    this.drone.reset(spawnPos.x, spawnPos.y, spawnPos.z);
    this.collisionCooldown = 1.5;
    this.hud.resetEpisode();
    this.hud.showToast('Drone Pod Reset to Safe Zone', true, 1200);
  }

  _checkCollisions() {
    // Ignore collisions during post-spawn cooldown
    if (this.collisionCooldown > 0) {
      return false;
    }

    const dronePos = this.drone.pos;
    const droneRadius = this.drone.radius;
    const arenaSize = this.world.arenaSize;

    // 1. Boundary check
    if (Math.abs(dronePos.x) >= arenaSize - 0.2 ||
        Math.abs(dronePos.z) >= arenaSize - 0.2 ||
        dronePos.y <= this.world.yMin ||
        dronePos.y >= this.world.yMax) {
      this.hud.onCollision('Perimeter Boundary');
      this._resetDrone();
      return true;
    }

    // 2. Obstacle check
    for (const obs of this.world.obstacles) {
      if (obs.kind === 'cylinder') {
        const dx = dronePos.x - obs.pos.x;
        const dz = dronePos.z - obs.pos.z;
        const distSq = dx * dx + dz * dz;
        const rTotal = obs.radius + droneRadius;
        const halfH = obs.height / 2 + droneRadius;
        const dy = Math.abs(dronePos.y - obs.pos.y);

        if (distSq <= rTotal * rTotal && dy <= halfH) {
          this.hud.onCollision('Low-Poly Pillar');
          this._resetDrone();
          return true;
        }
      } else {
        const d = dronePos.distanceTo(obs.pos);
        if (d <= obs.radius + droneRadius) {
          this.hud.onCollision('Polyhedron Block');
          this._resetDrone();
          return true;
        }
      }
    }

    // 3. Goal waypoint check (reach radius = 1.35m)
    const distToGoal = dronePos.distanceTo(this.world.goalPos);
    if (distToGoal <= 1.35) {
      this.hud.onGoalReached();
      this.world.spawnRandomGoal(9.0, this.drone.pos);
      return false;
    }

    return false;
  }

  animate() {
    requestAnimationFrame(this.animate);

    const delta = Math.min(this.clock.getDelta(), 0.1);
    if (this.collisionCooldown > 0) {
      this.collisionCooldown -= delta;
    }

    // 1. Update 16 LIDAR sensor rays against world obstacles
    const sensorDistances = this.sensors.update(this.world.obstacleMeshes, {
      arenaSize: this.world.arenaSize,
      yMin: this.world.yMin,
      yMax: this.world.yMax
    });

    // 2. Compute flight actions: Manual vs Autonomous DRL
    if (this.controls.isAutonomous) {
      // Build 26-dim observation vector
      const obs = this.drone.getObservation(this.world.goalPos, this.world.arenaSize, sensorDistances);
      const [pitch, roll, yawRate, thrust] = this.agent.getAction(obs, this.drone.pos, this.drone.vel, this.world.goalPos);
      this.drone.setActions(pitch, roll, yawRate, thrust);

    } else {
      // Manual WASD flight controls
      const [pitch, roll, yawRate, thrust] = this.controls.getManualActions();
      this.drone.setActions(pitch, roll, yawRate, thrust);
    }

    // 3. Step drone kinematics & physics
    this.drone.updatePhysics(delta);

    // 4. Check collisions and waypoints
    this._checkCollisions();

    // 5. Update World animations (rotating waypoint polyhedron)
    this.world.update(delta);

    // 6. Update Camera follow
    this.controls.updateCamera(delta);

    // 7. FPS calculation & HUD Telemetry
    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsUpdate >= 250) {
      this.fpsCounter = (this.frameCount * 1000) / (now - this.lastFpsUpdate);
      this.frameCount = 0;
      this.lastFpsUpdate = now;
    }

    this.hud.update(
      this.drone,
      this.world.goalPos,
      this.agent.inferenceLatencyMs,
      this.fpsCounter,
      sensorDistances
    );


    // 8. Render Three.js Scene
    this.renderer.render(this.scene, this.camera);
  }
}

// Initialize Application once DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
  window.droneApp = new SimulationApp();
});
