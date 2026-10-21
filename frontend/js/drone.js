/**
 * Drone Module: Futuristic Aerodynamic Pod Model & Flight Kinematics.
 * Features:
 *  - Sleek capsule/lozenge body with matte composite armor (#232834).
 *  - Front sensor visor with emissive glow (Cyan for Manual, Purple for Autonomous).
 *  - 4 angled vector pylons with 4 thruster pads.
 *  - 4 glowing thruster exhaust indicator rings with dynamic thrust pulse.
 *  - Kinematics: 3D velocity damping, bank angles on yaw/strafe, collision sphere.
 */

import * as THREE from 'three';

export class DronePod {
  constructor(scene) {
    this.scene = scene;

    // Kinematic states
    this.pos = new THREE.Vector3(0, 3.5, 0);
    this.vel = new THREE.Vector3(0, 0, 0);
    this.yaw = 0.0;
    this.pitch = 0.0;
    this.roll = 0.0;

    // Physical parameters matching Python Gymnasium DroneNavEnv
    this.maxSpeed = 7.5; // m/s
    this.maxYawRate = 2.4; // rad/s
    this.maxTilt = THREE.MathUtils.degToRad(25.0); // max bank tilt
    this.radius = 0.55; // collision sphere
    this.dragCoeff = 1.35;

    // Active action commands [-1.0, 1.0]
    this.actions = [0.0, 0.0, 0.0, 0.0]; // [pitch, roll, yawRate, thrust]

    // Visual meshes
    this.mesh = new THREE.Group();
    this.thrusterRings = [];
    this.visorMat = null;
    this.thrusterMat = null;

    this._buildPodMesh();
    this.scene.add(this.mesh);
  }

  _buildPodMesh() {
    // 1. Central Aerodynamic Pod Fuselage (Capsule / Lozenge)
    // Capsule: radius 0.32, length 0.75, along Z axis
    const podGeo = new THREE.CapsuleGeometry(0.32, 0.75, 8, 16);
    // Align capsule forward along +Z axis
    podGeo.rotateX(Math.PI / 2);

    const podMat = new THREE.MeshStandardMaterial({
      color: 0x232834,
      roughness: 0.55,
      metalness: 0.45,
      flatShading: true,
    });
    const podBody = new THREE.Mesh(podGeo, podMat);
    podBody.castShadow = true;
    podBody.receiveShadow = true;
    this.mesh.add(podBody);

    // 2. Sleek Front Sensor Canopy Visor Strip
    const visorGeo = new THREE.BoxGeometry(0.42, 0.12, 0.28);
    this.visorMat = new THREE.MeshStandardMaterial({
      color: 0x00f0ff,
      emissive: 0x00f0ff,
      emissiveIntensity: 0.8,
      roughness: 0.1,
      metalness: 0.9,
    });
    const visor = new THREE.Mesh(visorGeo, this.visorMat);
    visor.position.set(0, 0.1, 0.45);
    this.mesh.add(visor);

    // Front sensor lens dot
    const lensGeo = new THREE.SphereGeometry(0.06, 8, 8);
    const lensMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const lens = new THREE.Mesh(lensGeo, lensMat);
    lens.position.set(0, 0.1, 0.6);
    this.mesh.add(lens);

    // 3. Four Angled Vector Pylons & Thruster Pads
    // Positions: Front-Left, Front-Right, Rear-Left, Rear-Right
    const thrusterOffsets = [
      { x: -0.58, y: -0.05, z:  0.42 }, // Front-Left
      { x:  0.58, y: -0.05, z:  0.42 }, // Front-Right
      { x: -0.58, y: -0.05, z: -0.42 }, // Rear-Left
      { x:  0.58, y: -0.05, z: -0.42 }, // Rear-Right
    ];

    this.thrusterMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      wireframe: false,
    });

    for (const offset of thrusterOffsets) {
      // Pylon arm connecting pod body to thruster pad
      const armGeo = new THREE.BoxGeometry(0.08, 0.05, 0.45);
      const armMat = new THREE.MeshStandardMaterial({
        color: 0x1a1e29,
        roughness: 0.6,
        metalness: 0.5,
        flatShading: true,
      });
      const arm = new THREE.Mesh(armGeo, armMat);
      // Orient arm diagonally from center body
      const angle = Math.atan2(offset.z, offset.x);
      arm.rotation.y = angle + Math.PI / 2;
      arm.position.set(offset.x * 0.5, offset.y, offset.z * 0.5);
      this.mesh.add(arm);

      // Thruster Pad Housing (Minimalist aerodynamic cylinder/beveled pad)
      const padGeo = new THREE.CylinderGeometry(0.16, 0.14, 0.12, 8);
      const pad = new THREE.Mesh(padGeo, podMat);
      pad.position.set(offset.x, offset.y, offset.z);
      pad.castShadow = true;
      this.mesh.add(pad);

      // Glowing Indicator Ring at base of thruster pad
      const ringGeo = new THREE.TorusGeometry(0.13, 0.025, 6, 12);
      ringGeo.rotateX(Math.PI / 2);
      const ring = new THREE.Mesh(ringGeo, this.thrusterMat.clone());
      ring.position.set(offset.x, offset.y - 0.07, offset.z);
      this.mesh.add(ring);
      this.thrusterRings.push(ring);
    }
  }

  setModeVisuals(isAutonomous) {
    const color = isAutonomous ? 0xb388ff : 0x00f0ff;
    if (this.visorMat) {
      this.visorMat.color.setHex(color);
      this.visorMat.emissive.setHex(color);
    }
    for (const ring of this.thrusterRings) {
      if (ring.material) {
        ring.material.color.setHex(color);
      }
    }
  }

  setActions(pitch, roll, yawRate, thrust) {
    this.actions[0] = THREE.MathUtils.clamp(pitch, -1.0, 1.0);
    this.actions[1] = THREE.MathUtils.clamp(roll, -1.0, 1.0);
    this.actions[2] = THREE.MathUtils.clamp(yawRate, -1.0, 1.0);
    this.actions[3] = THREE.MathUtils.clamp(thrust, -1.0, 1.0);
  }

  reset(x = 0, y = 3.5, z = 0) {
    this.pos.set(x, y, z);
    this.vel.set(0, 0, 0);
    this.yaw = 0.0;
    this.pitch = 0.0;
    this.roll = 0.0;
    this.actions = [0.0, 0.0, 0.0, 0.0];
    this.mesh.position.copy(this.pos);
    this.mesh.rotation.set(0, 0, 0);
  }

  updatePhysics(delta) {
    const [pitchCmd, rollCmd, yawRateCmd, thrustCmd] = this.actions;

    // 1. Heading (Yaw) update
    this.yaw = (this.yaw + yawRateCmd * this.maxYawRate * delta) % (Math.PI * 2);

    // 2. Smooth attitude banking (Pitch & Roll)
    const targetPitch = pitchCmd * this.maxTilt;
    const targetRoll = -rollCmd * this.maxTilt; // bank into strafe
    this.pitch = THREE.MathUtils.lerp(this.pitch, targetPitch, 12 * delta);
    this.roll = THREE.MathUtils.lerp(this.roll, targetRoll, 12 * delta);

    // 3. Accelerations along drone heading
    const cosYaw = Math.cos(this.yaw);
    const sinYaw = Math.sin(this.yaw);

    const forwardAccel = pitchCmd * 8.5;
    const strafeAccel = rollCmd * 6.5;
    const verticalAccel = thrustCmd * 7.5;

    // World-space acceleration vectors (+Z is forward when yaw=0)
    const ax = sinYaw * forwardAccel + cosYaw * strafeAccel;
    const ay = verticalAccel;
    const az = cosYaw * forwardAccel - sinYaw * strafeAccel;

    // Apply acceleration and linear drag damping
    this.vel.x += (ax - this.dragCoeff * this.vel.x) * delta;
    this.vel.y += (ay - this.dragCoeff * this.vel.y) * delta;
    this.vel.z += (az - this.dragCoeff * this.vel.z) * delta;

    // Clamp to maximum physical speed
    const currentSpeed = this.vel.length();
    if (currentSpeed > this.maxSpeed) {
      this.vel.multiplyScalar(this.maxSpeed / currentSpeed);
    }

    // Integrate position
    this.pos.addScaledVector(this.vel, delta);

    // Update Three.js Group position & orientation
    this.mesh.position.copy(this.pos);

    // Euler rotation order: YXZ for clean yaw-pitch-roll
    this.mesh.rotation.order = 'YXZ';
    this.mesh.rotation.y = this.yaw;
    this.mesh.rotation.x = this.pitch;
    this.mesh.rotation.z = this.roll;

    // Dynamic thruster ring glow pulsing based on active thrust
    const totalThrustMagnitude = Math.abs(pitchCmd) + Math.abs(rollCmd) + Math.abs(thrustCmd) * 0.8 + 0.2;
    for (const ring of this.thrusterRings) {
      if (ring.scale) {
        const s = 1.0 + Math.min(totalThrustMagnitude * 0.15, 0.4);
        ring.scale.set(s, s, s);
      }
    }
  }

  getObservation(goalPos, arenaSize, sensorDistances) {
    // Relative vector to goal
    const dx = (goalPos.x - this.pos.x) / arenaSize;
    const dy = (goalPos.y - this.pos.y) / arenaSize;
    const dz = (goalPos.z - this.pos.z) / arenaSize;

    // Normalized velocity
    const vx = this.vel.x / this.maxSpeed;
    const vy = this.vel.y / this.maxSpeed;
    const vz = this.vel.z / this.maxSpeed;

    // Attitude
    const sinYaw = Math.sin(this.yaw);
    const cosYaw = Math.cos(this.yaw);
    const normPitch = this.pitch / this.maxTilt;
    const normRoll = this.roll / this.maxTilt;

    // Sensor distances are already normalized to [0, 1]
    const obs = [
      dx, dy, dz,
      vx, vy, vz,
      sinYaw, cosYaw, normPitch, normRoll,
      ...sensorDistances
    ];

    return obs;
  }
}
