/**
 * Sensors Module: 16-Ray Synthetic LIDAR & Rangefinder Array.
 * Matches Python Gymnasium DroneNavEnv sensor angles:
 *  - 12 horizontal rays (30 deg intervals around yaw heading)
 *  - 2 diagonal elevation rays (+25°, -25° forward)
 *  - 1 straight up ray (+85°)
 *  - 1 straight down ray (-85°)
 *
 * Includes dynamic 3D raycast visualizer with hit dots and toggle button.
 */

import * as THREE from 'three';

export class LidarSensor {
  constructor(scene, drone) {
    this.scene = scene;
    this.drone = drone;
    this.sensorRange = 14.0;
    this.numRays = 16;
    this.visible = true;

    // Ray angular offsets [azimuthDeg, elevationDeg] in drone body frame
    this.rayAngles = [
      [0.0, 0.0], [30.0, 0.0], [60.0, 0.0], [90.0, 0.0],
      [120.0, 0.0], [150.0, 0.0], [180.0, 0.0], [210.0, 0.0],
      [240.0, 0.0], [270.0, 0.0], [300.0, 0.0], [330.0, 0.0],
      [0.0, 25.0],   // Forward-up
      [0.0, -25.0],  // Forward-down
      [0.0, 85.0],   // Upward
      [0.0, -85.0],  // Downward
    ];

    this.raycaster = new THREE.Raycaster();
    this.raycaster.far = this.sensorRange;

    // Normalized distances [0, 1]
    this.distances = new Array(this.numRays).fill(1.0);
    this.hitPoints = [];

    // Visualizer line geometry & instanced hit points
    this._initVisualizer();
  }

  _initVisualizer() {
    this.visualizerGroup = new THREE.Group();

    // 1. Ray Lines (LineSegments: 2 vertices per ray)
    const linePositions = new Float32Array(this.numRays * 2 * 3);
    const lineColors = new Float32Array(this.numRays * 2 * 3);

    for (let i = 0; i < this.numRays * 2; i++) {
      lineColors[i * 3 + 0] = 0.0;
      lineColors[i * 3 + 1] = 0.94; // cyan / green laser
      lineColors[i * 3 + 2] = 1.0;
    }

    this.lineGeometry = new THREE.BufferAttribute(linePositions, 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', this.lineGeometry);
    geo.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));

    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.55,
      blending: THREE.AdditiveBlending,
    });
    this.lineMesh = new THREE.LineSegments(geo, mat);
    this.visualizerGroup.add(this.lineMesh);

    // 2. Hit Dots (InstancedMesh for low draw calls)
    const dotGeo = new THREE.SphereGeometry(0.08, 6, 6);
    const dotMat = new THREE.MeshBasicMaterial({
      color: 0xff5252,
      wireframe: false,
    });
    this.dotInstancedMesh = new THREE.InstancedMesh(dotGeo, dotMat, this.numRays);
    this.visualizerGroup.add(this.dotInstancedMesh);

    this.scene.add(this.visualizerGroup);
  }

  toggleVisibility(forced = null) {
    if (forced !== null) {
      this.visible = forced;
    } else {
      this.visible = !this.visible;
    }
    this.visualizerGroup.visible = this.visible;
    return this.visible;
  }

  update(obstacleMeshes, worldBounds = { arenaSize: 22, yMin: 0.5, yMax: 16 }) {
    const origin = this.drone.pos;
    const yaw = this.drone.yaw;
    const cosYaw = Math.cos(yaw);
    const sinYaw = Math.sin(yaw);

    const posArray = this.lineGeometry.array;
    const dummy = new THREE.Object3D();
    dummy.scale.set(0, 0, 0);

    for (let i = 0; i < this.numRays; i++) {
      const [azDeg, elDeg] = this.rayAngles[i];
      const az = THREE.MathUtils.degToRad(azDeg);
      const el = THREE.MathUtils.degToRad(elDeg);

      // Local direction in body frame
      const lx = Math.sin(az) * Math.cos(el);
      const ly = Math.sin(el);
      const lz = Math.cos(az) * Math.cos(el);

      // Rotate around yaw
      const wx = lx * cosYaw + lz * sinYaw;
      const wy = ly;
      const wz = -lx * sinYaw + lz * cosYaw;

      const rayDir = new THREE.Vector3(wx, wy, wz).normalize();
      this.raycaster.set(origin, rayDir);

      let minHitDist = this.sensorRange;

      // 1. Raycast against Three.js obstacle meshes
      const intersects = this.raycaster.intersectObjects(obstacleMeshes, false);
      if (intersects.length > 0 && intersects[0].distance < minHitDist) {
        minHitDist = intersects[0].distance;
      }

      // 2. Raycast against ground (Y=0) and ceiling
      if (rayDir.y < -1e-5) {
        const tGround = -origin.y / rayDir.y;
        if (tGround > 0 && tGround < minHitDist) {
          minHitDist = tGround;
        }
      } else if (rayDir.y > 1e-5) {
        const tCeil = (worldBounds.yMax - origin.y) / rayDir.y;
        if (tCeil > 0 && tCeil < minHitDist) {
          minHitDist = tCeil;
        }
      }

      // 3. Raycast against outer perimeter walls
      const aSize = worldBounds.arenaSize;
      const walls = [
        { axis: 'x', val: -aSize }, { axis: 'x', val: aSize },
        { axis: 'z', val: -aSize }, { axis: 'z', val: aSize },
      ];
      for (const wall of walls) {
        const dComp = rayDir[wall.axis];
        if (Math.abs(dComp) > 1e-5) {
          const tWall = (wall.val - origin[wall.axis]) / dComp;
          if (tWall > 0 && tWall < minHitDist) {
            const hitY = origin.y + tWall * rayDir.y;
            if (hitY >= 0 && hitY <= worldBounds.yMax) {
              minHitDist = tWall;
            }
          }
        }
      }

      // Normalized distance in [0, 1]
      this.distances[i] = THREE.MathUtils.clamp(minHitDist / this.sensorRange, 0.0, 1.0);

      // Update visual lines
      const idx = i * 6;
      posArray[idx + 0] = origin.x;
      posArray[idx + 1] = origin.y;
      posArray[idx + 2] = origin.z;

      const hitX = origin.x + rayDir.x * minHitDist;
      const hitY = origin.y + rayDir.y * minHitDist;
      const hitZ = origin.z + rayDir.z * minHitDist;

      posArray[idx + 3] = hitX;
      posArray[idx + 4] = hitY;
      posArray[idx + 5] = hitZ;

      // Update hit dot if obstacle detected within range
      if (minHitDist < this.sensorRange - 0.1) {
        dummy.position.set(hitX, hitY, hitZ);
        dummy.scale.set(1, 1, 1);
      } else {
        dummy.scale.set(0, 0, 0);
      }
      dummy.updateMatrix();
      this.dotInstancedMesh.setMatrixAt(i, dummy.matrix);
    }

    this.lineGeometry.needsUpdate = true;
    this.dotInstancedMesh.instanceMatrix.needsUpdate = true;

    return this.distances;
  }
}
