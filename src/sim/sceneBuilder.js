// Build a three.js scene graph from the compiled MuJoCo model, then sync geom
// world transforms every frame straight out of d.geom_xpos / d.geom_xmat.
//
// Deliberately flat: one Object3D per geom parented to the root, no attempt to
// mirror MuJoCo's body tree. MuJoCo already gives world transforms, so a
// hierarchy would only add matrix work per frame.

import * as THREE from 'three';

const GEOM = {
  PLANE: 0, HFIELD: 1, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4,
  CYLINDER: 5, BOX: 6, MESH: 7,
};

// MuJoCo is Z-up, three.js/WebXR is Y-up. Rotating the whole robot root by
// -90deg about X converts once, instead of per-object every frame.
export const MJ_TO_XR = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0), -Math.PI / 2,
);

function meshGeometry(model, meshId) {
  const vadr = model.mesh_vertadr[meshId];
  const vnum = model.mesh_vertnum[meshId];
  const fadr = model.mesh_faceadr[meshId];
  const fnum = model.mesh_facenum[meshId];

  const verts = new Float32Array(vnum * 3);
  for (let i = 0; i < vnum * 3; i++) verts[i] = model.mesh_vert[vadr * 3 + i];

  const idx = new Uint32Array(fnum * 3);
  for (let i = 0; i < fnum * 3; i++) idx[i] = model.mesh_face[fadr * 3 + i];

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));

  const nadr = model.mesh_normaladr[meshId];
  if (nadr >= 0 && model.mesh_normal) {
    const norms = new Float32Array(vnum * 3);
    for (let i = 0; i < vnum * 3; i++) norms[i] = model.mesh_normal[nadr * 3 + i];
    g.setAttribute('normal', new THREE.BufferAttribute(norms, 3));
  } else {
    g.computeVertexNormals();
  }
  return g;
}

function primitiveGeometry(type, size) {
  const [a, b] = size;
  switch (type) {
    case GEOM.PLANE:
      return new THREE.PlaneGeometry(a > 0 ? a * 2 : 8, b > 0 ? b * 2 : 8);
    case GEOM.SPHERE:
      return new THREE.SphereGeometry(a, 20, 14);
    case GEOM.CAPSULE:
      return new THREE.CapsuleGeometry(a, b * 2, 4, 12);
    case GEOM.CYLINDER:
      return new THREE.CylinderGeometry(a, a, b * 2, 18);
    case GEOM.ELLIPSOID: {
      const g = new THREE.SphereGeometry(1, 18, 12);
      g.scale(size[0], size[1], size[2]);
      return g;
    }
    case GEOM.BOX:
      return new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
    default:
      return null;
  }
}

/**
 * @param {Sim} sim
 * @param {object} opts
 * @param {number[]} opts.hiddenGroups  MuJoCo geom groups to skip (3 = the
 *        capsule collision proxies; you want those invisible in normal use but
 *        toggleable, because seeing them is how you debug a bad grasp).
 */
export function buildSceneGraph(sim, { hiddenGroups = [3] } = {}) {
  const { model } = sim;
  const root = new THREE.Group();
  root.quaternion.copy(MJ_TO_XR);

  const meshCache = new Map();
  const entries = [];

  for (let g = 0; g < model.ngeom; g++) {
    const type = model.geom_type[g];
    const size = [model.geom_size[g * 3], model.geom_size[g * 3 + 1], model.geom_size[g * 3 + 2]];

    let geometry;
    if (type === GEOM.MESH) {
      const meshId = model.geom_dataid[g];
      if (!meshCache.has(meshId)) meshCache.set(meshId, meshGeometry(model, meshId));
      geometry = meshCache.get(meshId);
    } else {
      geometry = primitiveGeometry(type, size);
    }
    if (!geometry) continue;

    // Three's capsule/cylinder are Y-aligned; MuJoCo's are Z-aligned.
    const needsAxisFix = type === GEOM.CAPSULE || type === GEOM.CYLINDER;

    const rgba = model.geom_rgba;
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[g * 4], rgba[g * 4 + 1], rgba[g * 4 + 2]),
      opacity: rgba[g * 4 + 3],
      transparent: rgba[g * 4 + 3] < 1,
      roughness: 0.7,
      metalness: 0.1,
    });

    const holder = new THREE.Group();       // carries the MuJoCo world transform
    const mesh = new THREE.Mesh(geometry, material);
    if (needsAxisFix) mesh.rotateX(Math.PI / 2);
    mesh.castShadow = mesh.receiveShadow = type !== GEOM.PLANE;
    holder.add(mesh);
    holder.visible = !hiddenGroups.includes(model.geom_group[g]);
    holder.userData.mjGroup = model.geom_group[g];
    root.add(holder);
    entries.push({ geomId: g, holder });
  }

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();

  function sync(data) {
    const xpos = data.geom_xpos;
    const xmat = data.geom_xmat;
    for (const { geomId, holder } of entries) {
      const p = geomId * 3;
      const r = geomId * 9;
      // MuJoCo xmat is row-major 3x3; three.js Matrix4.set takes row-major.
      m.set(
        xmat[r], xmat[r + 1], xmat[r + 2], 0,
        xmat[r + 3], xmat[r + 4], xmat[r + 5], 0,
        xmat[r + 6], xmat[r + 7], xmat[r + 8], 0,
        0, 0, 0, 1,
      );
      q.setFromRotationMatrix(m);
      holder.position.set(xpos[p], xpos[p + 1], xpos[p + 2]);
      holder.quaternion.copy(q);
    }
  }

  function setGroupVisible(group, visible) {
    for (const { holder } of entries) {
      if (holder.userData.mjGroup === group) holder.visible = visible;
    }
  }

  const byGeomId = new Map(entries.map((e) => [e.geomId, e]));

  // Levels use this to signal per-target state (e.g. "already touched")
  // without recompiling the model -- opacity is a material property, so it's
  // cheap to flip every frame from tick().
  function setOpacity(geomId, opacity) {
    const entry = byGeomId.get(geomId);
    if (!entry) return;
    const material = entry.holder.children[0].material;
    material.opacity = opacity;
    material.transparent = opacity < 1;
  }

  return { root, sync, setGroupVisible, setOpacity, entries };
}
