/**
 * World Module: Procedural Low-Poly 3D Environment.
 * Implements solid slate/cream sky (#8992a1), deep charcoal ground (#1a1a24),
 * procedural convex polyhedra obstacles, and glowing target waypoint polyhedron.
 */

import * as THREE from 'three';

export class World {
  constructor(scene) {
    this.scene = scene;
    this.obstacles = [];
    this.obstacleMeshes = [];
    this.goalMesh = null;
    this.goalRing = null;
    this.goalPos = new THREE.Vector3(0, 4, 12);
    this.arenaSize = 22.0;
    this.yMin = 0.5;
    this.yMax = 16.0;

    this.palette = [
      0x3b4252, 0x434c5e, 0x4c566a, 0xe06c75,
      0x4fa6a1, 0xd19a66, 0x88c0d0, 0x5e81ac
    ];

    this._initEnvironment();
    this.generateProceduralWorld();
    this._createGoalWaypoint();
  }

  _initEnvironment() {
    // 1. Sky & Background: Solid flat slate/cream sky #8992a1 (no gradient textures)
    this.scene.background = new THREE.Color(0x8992a1);
    this.scene.fog = new THREE.FogExp2(0x8992a1, 0.015);

    // 2. Lighting: Directional sun + Hemisphere for crisp flat-shaded low-poly facets
    const hemiLight = new THREE.HemisphereLight(0xced7e0, 0x1a1a24, 0.75);
    this.scene.add(hemiLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 1.1);
    sunLight.position.set(25, 40, 20);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.near = 0.5;
    sunLight.shadow.camera.far = 100;
    sunLight.shadow.camera.left = -26;
    sunLight.shadow.camera.right = 26;
    sunLight.shadow.camera.top = 26;
    sunLight.shadow.camera.bottom = -26;
    this.scene.add(sunLight);

    // Subtle ambient fill
    const ambientLight = new THREE.AmbientLight(0x2e3440, 0.4);
    this.scene.add(ambientLight);

    // 3. Ground: Solid flat deep charcoal #1a1a24
    const groundGeo = new THREE.PlaneGeometry(120, 120);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x1a1a24,
      roughness: 0.9,
      metalness: 0.1,
      flatShading: true,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    // Minimalist arena boundary grid accents
    const grid = new THREE.GridHelper(this.arenaSize * 2, 22, 0x3b4252, 0x222633);
    grid.position.y = 0.02;
    this.scene.add(grid);

    // Bounding perimeter boundary markers
    const perimeterGeo = new THREE.RingGeometry(this.arenaSize - 0.2, this.arenaSize, 4);
    perimeterGeo.rotateX(-Math.PI / 2);
    perimeterGeo.rotateY(Math.PI / 4);
    const perimeterMat = new THREE.MeshBasicMaterial({
      color: 0x3b4252,
      side: THREE.DoubleSide
    });
    const perimeter = new THREE.Mesh(perimeterGeo, perimeterMat);
    perimeter.position.y = 0.03;
    this.scene.add(perimeter);
  }

  generateProceduralWorld(obstacleData = null) {
    // Clear previous obstacles
    for (const mesh of this.obstacleMeshes) {
      this.scene.remove(mesh);
      if (mesh.geometry) mesh.geometry.dispose();
      if (mesh.material) mesh.material.dispose();
    }
    this.obstacles = [];
    this.obstacleMeshes = [];

    if (obstacleData && obstacleData.length > 0) {
      // Build from Python backend specifications
      for (const d of obstacleData) {
        this._addObstacleFromData(d);
      }
    } else {
      // Generate randomized procedural convex polyhedra
      this._generateRandomObstacles();
    }
  }

  _generateRandomObstacles() {
    // 1. Tall low-poly hexagonal/pentagonal pillars & prisms
    const numPillars = 14;
    for (let i = 0; i < numPillars; i++) {
      const radius = THREE.MathUtils.randFloat(1.1, 2.0);
      const height = THREE.MathUtils.randFloat(8.0, 15.0);
      const segments = THREE.MathUtils.randInt(5, 7); // low-poly prism

      // Sample position ensuring clearance from origin spawn (0, 0, 0)
      let x = 0, z = 0;
      for (let attempt = 0; attempt < 50; attempt++) {
        x = THREE.MathUtils.randFloatSpread(this.arenaSize * 1.5);
        z = THREE.MathUtils.randFloatSpread(this.arenaSize * 1.5);
        if (Math.hypot(x, z) >= 5.5 + radius) break;
      }

      const geo = new THREE.CylinderGeometry(radius * 0.9, radius, height, segments);
      const color = this.palette[Math.floor(Math.random() * this.palette.length)];
      const mat = new THREE.MeshStandardMaterial({
        color: color,
        roughness: 0.85,
        metalness: 0.15,
        flatShading: true,
      });

      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x, height / 2, z);
      mesh.rotation.y = Math.random() * Math.PI;
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      this.scene.add(mesh);
      this.obstacleMeshes.push(mesh);
      this.obstacles.push({
        kind: 'cylinder',
        pos: mesh.position,
        radius: radius,
        height: height
      });
    }

    // 2. Procedural Convex Polyhedra (Dodecahedrons, Beveled Blocks, Low-Poly Monoliths)
    const numPolyhedra = 16;
    for (let i = 0; i < numPolyhedra; i++) {
      const type = Math.floor(Math.random() * 3);
      let geo;
      const r = THREE.MathUtils.randFloat(1.2, 2.2);

      // Sample position ensuring clearance from origin spawn (0, 0, 0)
      let x = 0, y = 3.5, z = 0;
      for (let attempt = 0; attempt < 50; attempt++) {
        x = THREE.MathUtils.randFloatSpread(this.arenaSize * 1.5);
        z = THREE.MathUtils.randFloatSpread(this.arenaSize * 1.5);
        y = THREE.MathUtils.randFloat(2.5, 9.5);
        if (Math.hypot(x, z) >= 5.5 + r) break;
      }

      if (type === 0) {
        // Dodecahedron (12-sided convex regular polyhedron)
        geo = new THREE.DodecahedronGeometry(r, 0);
      } else if (type === 1) {
        // Low-poly monolith block
        const hx = THREE.MathUtils.randFloat(1.6, 3.0);
        const hy = THREE.MathUtils.randFloat(2.5, 5.5);
        const hz = THREE.MathUtils.randFloat(1.6, 3.0);
        geo = new THREE.BoxGeometry(hx, hy, hz);
      } else {
        // Octahedron
        geo = new THREE.OctahedronGeometry(r * 1.1, 0);
      }

      const color = this.palette[Math.floor(Math.random() * this.palette.length)];
      const mat = new THREE.MeshStandardMaterial({
        color: color,
        roughness: 0.8,
        metalness: 0.2,
        flatShading: true,
      });

      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x, y, z);
      mesh.rotation.set(Math.random() * 0.4, Math.random() * Math.PI, Math.random() * 0.4);
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      this.scene.add(mesh);
      this.obstacleMeshes.push(mesh);
      this.obstacles.push({
        kind: 'polyhedron',
        pos: mesh.position,
        radius: r + 0.5
      });
    }
  }

  getSafeSpawnPosition() {
    // Check if default (0, 3.5, 0) is clean
    const defaultPos = new THREE.Vector3(0, 3.5, 0);
    let safe = true;
    for (const obs of this.obstacles) {
      if (obs.kind === 'cylinder') {
        const dx = defaultPos.x - obs.pos.x;
        const dz = defaultPos.z - obs.pos.z;
        if (Math.hypot(dx, dz) < obs.radius + 1.5) {
          safe = false;
          break;
        }
      } else {
        if (defaultPos.distanceTo(obs.pos) < obs.radius + 1.5) {
          safe = false;
          break;
        }
      }
    }
    if (safe) return defaultPos;

    // Search for a guaranteed safe coordinate
    for (let attempts = 0; attempts < 100; attempts++) {
      const cand = new THREE.Vector3(
        THREE.MathUtils.randFloatSpread(12),
        THREE.MathUtils.randFloat(3.0, 6.0),
        THREE.MathUtils.randFloatSpread(12)
      );
      let clear = true;
      for (const obs of this.obstacles) {
        if (cand.distanceTo(obs.pos) < (obs.radius || 2.0) + 2.0) {
          clear = false;
          break;
        }
      }
      if (clear) return cand;
    }
    return defaultPos;
  }

  _addObstacleFromData(d) {
    const color = d.color ? parseInt(d.color.replace('#', '0x')) : 0x4c566a;
    const mat = new THREE.MeshStandardMaterial({
      color: color,
      roughness: 0.85,
      metalness: 0.15,
      flatShading: true,
    });

    let mesh;
    if (d.kind === 'cylinder' || d.kind === 'prism') {
      const radius = d.size[0];
      const height = d.size[1];
      const geo = new THREE.CylinderGeometry(radius, radius, height, 6);
      mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(d.pos[0], d.pos[1], d.pos[2]);
    } else {
      const hx = d.size[0] * 2;
      const hy = d.size[1] * 2;
      const hz = d.size[2] * 2;
      const geo = new THREE.BoxGeometry(hx, hy, hz);
      mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(d.pos[0], d.pos[1], d.pos[2]);
    }

    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this.scene.add(mesh);
    this.obstacleMeshes.push(mesh);
  }

  _createGoalWaypoint() {
    // Target Waypoint: Glowing geometric low-poly polyhedron (Octahedron) + Pulsing ring
    const goalGroup = new THREE.Group();

    // Central geometric octahedron
    const octGeo = new THREE.OctahedronGeometry(0.85, 0);
    const octMat = new THREE.MeshStandardMaterial({
      color: 0xffd600,
      emissive: 0xffaa00,
      emissiveIntensity: 0.6,
      roughness: 0.2,
      metalness: 0.8,
      flatShading: true,
    });
    this.goalMesh = new THREE.Mesh(octGeo, octMat);
    goalGroup.add(this.goalMesh);

    // Glowing orbiting energy ring
    const ringGeo = new THREE.TorusGeometry(1.3, 0.05, 6, 24);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xffea00,
      wireframe: true,
    });
    this.goalRing = new THREE.Mesh(ringGeo, ringMat);
    this.goalRing.rotation.x = Math.PI / 2;
    goalGroup.add(this.goalRing);

    // Vertical beacon line
    const beaconGeo = new THREE.CylinderGeometry(0.04, 0.04, 25, 4);
    const beaconMat = new THREE.MeshBasicMaterial({
      color: 0xffd600,
      transparent: true,
      opacity: 0.25,
    });
    const beacon = new THREE.Mesh(beaconGeo, beaconMat);
    beacon.position.y = 12.5;
    goalGroup.add(beacon);

    this.goalGroup = goalGroup;
    this.goalGroup.position.copy(this.goalPos);
    this.scene.add(this.goalGroup);
  }

  setGoalPosition(pos) {
    this.goalPos.copy(pos);
    if (this.goalGroup) {
      this.goalGroup.position.copy(pos);
    }
  }

  spawnRandomGoal(minDistFromDrone = 8.0, dronePos = new THREE.Vector3(0, 3, 0)) {
    for (let attempts = 0; attempts < 100; attempts++) {
      const x = THREE.MathUtils.randFloatSpread(this.arenaSize * 1.4);
      const y = THREE.MathUtils.randFloat(2.0, 11.0);
      const z = THREE.MathUtils.randFloatSpread(this.arenaSize * 1.4);
      const candidate = new THREE.Vector3(x, y, z);

      if (candidate.distanceTo(dronePos) < minDistFromDrone) continue;

      // Check clearance from obstacles
      let clear = true;
      for (const obs of this.obstacles) {
        const d = candidate.distanceTo(obs.pos);
        if (d < (obs.radius || 2.0) + 1.8) {
          clear = false;
          break;
        }
      }

      if (clear) {
        this.setGoalPosition(candidate);
        return candidate;
      }
    }
    const fallback = new THREE.Vector3(8, 4, 8);
    this.setGoalPosition(fallback);
    return fallback;
  }

  update(delta) {
    if (this.goalMesh) {
      this.goalMesh.rotation.y += 1.8 * delta;
      this.goalMesh.rotation.x += 0.9 * delta;
    }
    if (this.goalRing) {
      this.goalRing.rotation.z += 1.2 * delta;
    }
  }
}
