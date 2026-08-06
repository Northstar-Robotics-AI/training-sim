// Headless check of the JS control path: Jacobian, quaternion math, clutch
// retargeting, and closed-loop IK convergence against the real YAM model.
//
// ik_bench.py covers the *algorithm*; this covers the WASM binding layer the
// browser actually runs, which is where an all-zero Jacobian can hide (the
// out-parameter wrappers in src/sim/mjmath.js exist for that reason).
//
// Run: node tools/ik_smoke.mjs

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import load_mujoco from '@mujoco/mujoco';

import { Sim } from '../src/sim/sim.js';
import { ArmIK } from '../src/control/ik.js';
import { ClutchRetargeter, xrPosToMj, xrQuatToMj } from '../src/control/retarget.js';
import { jacSite, mulQuat } from '../src/sim/mjmath.js';

const ASSETS = 'public/assets/yam';
const HOME_Q = [0, 0.9, 1.2, 0, -0.5, 0];

let failures = 0;
function check(name, ok, detail = '') {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? `  ${detail}` : ''}`);
  if (!ok) failures++;
}

/** Sim.init() fetches over HTTP; node reads from disk. Same MEMFS layout. */
async function loadSim() {
  const sim = new Sim();
  sim.mj = await load_mujoco();
  const FS = sim.mj.FS;
  FS.mkdir('/work');
  FS.mount(sim.mj.MEMFS, { root: '.' }, '/work');
  FS.mkdir('/work/meshes');
  for (const f of readdirSync(join(ASSETS, 'meshes'))) {
    FS.writeFile(`/work/meshes/${f}`, readFileSync(join(ASSETS, 'meshes', f)));
  }
  FS.writeFile('/work/scene.xml', readFileSync(join(ASSETS, 'bimanual_yam.xml'), 'utf8'));

  sim.model = sim.mj.MjModel.from_xml_path('/work/scene.xml');
  sim.data = new sim.mj.MjData(sim.model);
  sim.timestep = sim.model.opt.timestep;
  sim.mj.mj_forward(sim.model, sim.data);
  sim._cacheNames();
  return sim;
}

const sim = await loadSim();
const ik = new ArmIK(sim, { prefix: 'left_', siteName: 'left_grasp_site', homeQ: HOME_Q });

for (let i = 0; i < 6; i++) {
  sim.data.qpos[ik.qposAdr[i]] = HOME_Q[i];
  sim.data.ctrl[ik.actIds[i]] = HOME_Q[i];
}
ik.syncToCurrent();   // command, solver goal and integral term together
sim.mj.mj_forward(sim.model, sim.data);

// 1. The site pose quaternion must be a real unit quaternion, not zeros.
const pose = sim.sitePose('left_grasp_site');
const qNorm = Math.hypot(...pose.quat);
check('sitePose quat is unit', Math.abs(qNorm - 1) < 1e-9, `|q|=${qNorm.toFixed(6)}`);

// 2. The Jacobian must have nonzero entries in this arm's columns. An all-zero
//    Jacobian is the specific silent failure that froze the arms.
const jacp = new Float64Array(3 * sim.model.nv);
const jacr = new Float64Array(3 * sim.model.nv);
jacSite(sim.mj, sim.model, sim.data, jacp, jacr, ik.siteId);
let jMax = 0;
for (let r = 0; r < 3; r++) {
  for (let c = 0; c < 6; c++) {
    jMax = Math.max(jMax, Math.abs(jacp[r * sim.model.nv + ik.dofAdr[c]]));
  }
}
check('site Jacobian is nonzero', jMax > 1e-6, `max|Jp|=${jMax.toFixed(4)}`);

// 3. XR -> MuJoCo quaternion conversion. A 45 deg yaw about XR's up axis (+Y)
//    must come out as a 45 deg yaw about MuJoCo's up axis (+Z), same sense.
const qXr = xrQuatToMj(sim.mj, { x: 0, y: 0.3826834, z: 0, w: 0.9238795 });
const qWant = [0.9238795, 0, 0, 0.3826834];
const qErr = Math.max(...qWant.map((v, i) => Math.abs(qXr[i] - v)));
check('xrQuatToMj maps XR yaw to MuJoCo yaw', qErr < 1e-6,
  `err=${qErr.toExponential(2)} q=[${Array.from(qXr).map((v) => v.toFixed(6))}]`);

// 4. Clutch delta: hand moves 0.1 m, scaled target moves 0.06 m from the anchor.
//    The input EMA means one call only gets posAlpha of the way there, so hold
//    the hand still and let the filter settle before asserting the mapping --
//    the point of the check is the scale factor, not the absence of a filter.
const rt = new ClutchRetargeter(sim.mj, { posScale: 0.6 });
const handQuat = xrQuatToMj(sim.mj, { x: 0, y: 0, z: 0, w: 1 });
const handAt = (x) => xrPosToMj({ x, y: 1.2, z: -0.3 });
rt.engage(handAt(0), handQuat, pose);
const t0 = rt.target(handAt(0), handQuat);
let t1;
for (let i = 0; i < 200; i++) t1 = rt.target(handAt(0.1), handQuat);
const anchorErr = Math.hypot(...[0, 1, 2].map((i) => t0.pos[i] - pose.pos[i]));
check('target at engage equals current EE', anchorErr < 1e-12, `d=${anchorErr.toExponential(1)}`);
check('hand delta scales into target', Math.abs((t1.pos[0] - t0.pos[0]) - 0.06) < 1e-9,
  `dx=${(t1.pos[0] - t0.pos[0]).toFixed(4)}`);

// 4b. Reach clamp: a hand delta far past posReach must truncate to it, so the
//     solver is never handed a target the arm cannot follow.
const rtFar = new ClutchRetargeter(sim.mj, { posScale: 1.0, posReach: 0.2 });
rtFar.engage(handAt(0), handQuat, pose);
let tFar;
for (let i = 0; i < 400; i++) tFar = rtFar.target(handAt(1.5), handQuat);
const reach = Math.hypot(...[0, 1, 2].map((i) => tFar.pos[i] - pose.pos[i]));
check('target is clamped to posReach', Math.abs(reach - 0.2) < 1e-6,
  `reach=${reach.toFixed(4)}m`);

// 5. Closed loop: hold a target 8 cm away and confirm the gripper site actually
//    gets there. This is the end-to-end assertion -- it fails on a zero
//    Jacobian, a frame-mismatched rotation error, or a stuck servo alike.
const start = Float64Array.from(pose.pos);
const goal = Float64Array.from([start[0] + 0.05, start[1] - 0.04, start[2] + 0.04]);
const d0 = Math.hypot(goal[0] - start[0], goal[1] - start[1], goal[2] - start[2]);
ik.setGripper(1);
let moved = 0;
for (let i = 0; i < 400; i++) {
  ik.step(goal, pose.quat);
  sim.advance(0.01);
  const p = sim.data.site('left_grasp_site').xpos;
  moved = Math.max(moved, Math.hypot(p[0] - start[0], p[1] - start[1], p[2] - start[2]));
}
const now = sim.data.site('left_grasp_site').xpos;
const dEnd = Math.hypot(goal[0] - now[0], goal[1] - now[1], goal[2] - now[2]);
check('arm moves toward target', moved > 0.01, `moved=${(moved * 1000).toFixed(1)}mm`);
check('IK converges within 1 cm', dEnd < 0.01,
  `start=${(d0 * 1000).toFixed(1)}mm -> end=${(dEnd * 1000).toFixed(1)}mm`);

// 6. Orientation tracking. The rotation error is computed in the site frame and
//    used against a world-frame Jacobian, so a missing frame change shows up
//    here as a large permanent residual rather than as no motion at all.
const held = Float64Array.from(sim.data.site('left_grasp_site').xpos);
const yaw20 = [Math.cos(Math.PI / 18), 0, 0, Math.sin(Math.PI / 18)];
const goalQuat = mulQuat(sim.mj, new Float64Array(4), yaw20,
  sim.sitePose('left_grasp_site').quat);
const angTo = (q) => {
  const dot = Math.abs(q[0] * goalQuat[0] + q[1] * goalQuat[1]
    + q[2] * goalQuat[2] + q[3] * goalQuat[3]);
  return 2 * Math.acos(Math.min(dot, 1)) * 180 / Math.PI;
};
const rot0 = angTo(sim.sitePose('left_grasp_site').quat);
for (let i = 0; i < 600; i++) {
  ik.step(held, goalQuat);
  sim.advance(0.01);
}
const rotEnd = angTo(sim.sitePose('left_grasp_site').quat);
check('IK converges in orientation', rotEnd < 3,
  `start=${rot0.toFixed(1)}deg -> end=${rotEnd.toFixed(1)}deg`);

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
