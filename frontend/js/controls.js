/**
 * Controls Module: Dual-Mode Keyboard Input & Dynamic Camera Modes.
 * Handles WASD / Space / Shift / QE flight inputs, mode switching, and camera follow.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class FlightControls {
  constructor(camera, domElement, drone) {
    this.camera = camera;
    this.domElement = domElement;
    this.drone = drone;

    this.isAutonomous = false;
    this.cameraMode = 'chase'; // 'chase' | 'orbit' | 'top' | 'fpv'

    // Active key states
    this.keys = {
      KeyW: false, KeyS: false,
      KeyA: false, KeyD: false,
      KeyQ: false, KeyE: false,
      KeyC: false,
      Space: false, ShiftLeft: false, ShiftRight: false
    };


    // Orbit controls for free view
    this.orbitControls = new OrbitControls(this.camera, this.domElement);
    this.orbitControls.enableDamping = true;
    this.orbitControls.dampingFactor = 0.05;
    this.orbitControls.maxPolarAngle = Math.PI / 2 - 0.05; // don't go below ground
    this.orbitControls.enabled = false;

    // Callbacks
    this.onModeChange = null;
    this.onResetDrone = null;

    this._setupListeners();
  }

  _setupListeners() {
    window.addEventListener('keydown', (e) => this._onKeyDown(e));
    window.addEventListener('keyup', (e) => this._onKeyUp(e));
  }

  _onKeyDown(e) {
    if (e.code in this.keys) {
      this.keys[e.code] = true;
      e.preventDefault();
    } else if (e.code === 'KeyM') {
      this.toggleMode();
    } else if (e.code === 'KeyR') {
      if (this.onResetDrone) this.onResetDrone();
    }
  }

  _onKeyUp(e) {
    if (e.code in this.keys) {
      this.keys[e.code] = false;
      e.preventDefault();
    }
  }

  setAutonomousMode(enabled) {
    this.isAutonomous = enabled;
    this.drone.setModeVisuals(this.isAutonomous);
    if (this.onModeChange) this.onModeChange(this.isAutonomous);
  }

  toggleMode() {
    this.setAutonomousMode(!this.isAutonomous);
  }

  setCameraMode(mode) {
    this.cameraMode = mode;
    if (mode === 'orbit') {
      this.orbitControls.enabled = true;
      this.orbitControls.target.copy(this.drone.pos);
    } else {
      this.orbitControls.enabled = false;
    }
  }

  getManualActions() {
    let pitch = 0.0;
    let roll = 0.0;
    let yawRate = 0.0;
    let thrust = 0.0;

    // Pitch: Forward (W) / Backward (S)
    if (this.keys.KeyW) pitch -= 1.0;
    if (this.keys.KeyS) pitch += 1.0;

    // Yaw: Rotate Left (A) / Right (D)
    if (this.keys.KeyA) yawRate += 1.0;
    if (this.keys.KeyD) yawRate -= 1.0;

    // Strafe: Left (Q) / Right (E)
    if (this.keys.KeyQ) roll += 1.0;
    if (this.keys.KeyE) roll -= 1.0;

    // Vertical Thrust: Up (Space) / Down (Shift or C)
    if (this.keys.Space) thrust += 1.0;
    if (this.keys.ShiftLeft || this.keys.ShiftRight || this.keys.KeyC) thrust -= 1.0;


    return [pitch, roll, yawRate, thrust];
  }

  updateCamera(delta) {
    const dronePos = this.drone.pos;
    const yaw = this.drone.yaw;

    if (this.cameraMode === 'chase') {
      // Third-person dynamic follow behind pod
      const chaseDistance = 4.2;
      const chaseHeight = 1.6;

      const cosYaw = Math.cos(yaw);
      const sinYaw = Math.sin(yaw);

      // Desired position behind the pod
      const targetCamPos = new THREE.Vector3(
        dronePos.x - sinYaw * chaseDistance,
        dronePos.y + chaseHeight,
        dronePos.z - cosYaw * chaseDistance
      );

      // Smooth lag dampening
      this.camera.position.lerp(targetCamPos, 8.0 * delta);

      // Look at slightly ahead of the drone
      const lookTarget = new THREE.Vector3(
        dronePos.x + sinYaw * 2.0,
        dronePos.y + 0.3,
        dronePos.z + cosYaw * 2.0
      );
      this.camera.lookAt(lookTarget);

    } else if (this.cameraMode === 'top') {
      // Top-down tactical overhead
      this.camera.position.set(dronePos.x, dronePos.y + 24.0, dronePos.z + 0.1);
      this.camera.lookAt(dronePos);

    } else if (this.cameraMode === 'fpv') {
      // Nose camera from front sensor visor
      const cosYaw = Math.cos(yaw);
      const sinYaw = Math.sin(yaw);

      this.camera.position.set(
        dronePos.x + sinYaw * 0.5,
        dronePos.y + 0.12,
        dronePos.z + cosYaw * 0.5
      );
      this.camera.lookAt(
        dronePos.x + sinYaw * 10.0,
        dronePos.y + 0.12,
        dronePos.z + cosYaw * 10.0
      );

    } else if (this.cameraMode === 'orbit') {
      this.orbitControls.target.lerp(dronePos, 5.0 * delta);
      this.orbitControls.update();
    }
  }
}
