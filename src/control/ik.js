// Per-arm differential IK.
//
// Damped least squares on the 6x6 arm Jacobian, solved fresh every control tick
// against the current qpos. Runs at the control rate (~100 Hz), not the render
// rate, so headset framerate never changes the arm's behaviour.
//
// Why DLS and not pseudo-inverse: the YAM's wrist passes through a singularity
// when joint5 crosses zero, and an unregularised inverse produces a violent
// flick there. The damping term trades a little tracking accuracy for a
// bounded joint velocity, which is what an operator actually wants to feel.

import { jacSite, mat2Quat, subQuat } from '../sim/mjmath.js';

const DOF = 6;

export class ArmIK {
  /**
   * @param {Sim} sim
   * @param {object} cfg
   * @param {string} cfg.prefix       'left_' or 'right_'
   * @param {string} cfg.siteName     site tracked to the target (e.g. left_grasp_site)
   * @param {number[]} cfg.homeQ      nullspace posture bias, radians
   */
  constructor(sim, { prefix, siteName, homeQ }) {
    this.sim = sim;
    this.mj = sim.mj;
    this.siteName = siteName;
    this.siteId = sim.id('site', siteName);

    this.jointIds = [];
    this.qposAdr = [];
    this.dofAdr = [];
    this.limits = [];
    for (let i = 1; i <= DOF; i++) {
      const jid = sim.id('joint', `${prefix}joint${i}`);
      this.jointIds.push(jid);
      this.qposAdr.push(sim.model.jnt_qposadr[jid]);
      this.dofAdr.push(sim.model.jnt_dofadr[jid]);
      this.limits.push([sim.model.jnt_range[jid * 2], sim.model.jnt_range[jid * 2 + 1]]);
    }
    this.actIds = [];
    for (let i = 1; i <= DOF; i++) this.actIds.push(sim.id('actuator', `${prefix}act${i}`));
    this.gripperAct = sim.id('actuator', `${prefix}gripper`);

    this.homeQ = homeQ ?? new Array(DOF).fill(0);
    this.qTarget = Float64Array.from(this.homeQ);

    // Tuning. These are measured, not guessed -- see tools/ik_bench.py.
    // lambda is the DLS damping; below ~0.05 the wrist snaps through its
    // singularity and the worst-case error jumps from 9 mm to 560 mm.
    this.lambda = 0.08;
    // maxStep 0.01 rad/tick at 100 Hz = 1 rad/s joint limit, ~230 mm/s at the
    // gripper. Larger values let the servo target outrun the arm.
    this.maxStep = 0.01;
    // posGain below 1 is what removes the steady-state limit cycle: at unity
    // the DLS output is always above maxStep, so the controller never stops
    // slewing and oscillates by ~9 mm around the target forever.
    this.posGain = 0.35;
    this.rotGain = 0.21;
    // nullGain trades tracking for posture. At 0.02 the bias exceeds maxStep
    // and dominates the task; 0.002 costs 0.25 mm and still unfolds the elbow.
    this.nullGain = 0.002;
    // Anti-windup. The target is integrated open-loop, so when the arm is
    // blocked (contact, joint limit) qTarget runs away -- measured up to
    // 1.8 rad before this was added, which diverges and never recovers.
    this.maxLag = 0.08;
    this.errCap = 0.02;

    this._jacp = new Float64Array(3 * sim.model.nv);
    this._jacr = new Float64Array(3 * sim.model.nv);
    this._J = new Float64Array(6 * DOF);
    this._err = new Float64Array(6);
  }

  /** Snap the internal target to wherever the arm currently is. Call on reset
   *  and whenever the clutch re-engages, or the arm lurches to a stale target. */
  syncToCurrent() {
    for (let i = 0; i < DOF; i++) this.qTarget[i] = this.sim.data.qpos[this.qposAdr[i]];
  }

  currentQ() {
    const q = new Float64Array(DOF);
    for (let i = 0; i < DOF; i++) q[i] = this.sim.data.qpos[this.qposAdr[i]];
    return q;
  }

  /**
   * One IK iteration toward a world-frame target, in MuJoCo coordinates.
   * @param {Float64Array} targetPos  xyz
   * @param {Float64Array} targetQuat wxyz
   * @returns {{posErr:number, rotErr:number, damped:boolean}}
   */
  step(targetPos, targetQuat) {
    const { mj, sim } = this;
    const { model, data } = sim;

    const site = data.site(this.siteName);
    const curQuat = mat2Quat(mj, new Float64Array(4), site.xmat);

    // Positional error, clipped so a big clutch jump doesn't demand an
    // impossible one-tick motion.
    for (let i = 0; i < 3; i++) this._err[i] = (targetPos[i] - site.xpos[i]) * this.posGain;

    // Orientation error as a rotation vector. mju_subQuat(res, qa, qb) solves
    // qb * quat(res) = qa, so res is expressed in the *current site's* frame.
    // mj_jacSite's rotational block is world-frame. Feeding one into the other
    // is a silent, plausible-looking bug: the arm still moves, it just never
    // converges (measured ~60 deg of permanent residual). Rotate into world.
    const dqLocal = subQuat(mj, new Float64Array(3), targetQuat, curQuat);
    const R = site.xmat;
    const dq = new Float64Array([
      R[0] * dqLocal[0] + R[1] * dqLocal[1] + R[2] * dqLocal[2],
      R[3] * dqLocal[0] + R[4] * dqLocal[1] + R[5] * dqLocal[2],
      R[6] * dqLocal[0] + R[7] * dqLocal[1] + R[8] * dqLocal[2],
    ]);
    for (let i = 0; i < 3; i++) this._err[3 + i] = dq[i] * this.rotGain;

    const posErr = Math.hypot(this._err[0], this._err[1], this._err[2]);
    const rotErr = Math.hypot(dqLocal[0], dqLocal[1], dqLocal[2]);

    if (posErr > this.errCap) {
      for (let i = 0; i < 3; i++) this._err[i] *= this.errCap / posErr;
    }

    jacSite(mj, model, data, this._jacp, this._jacr, this.siteId);

    // Slice out this arm's 6 columns from the full nv-wide Jacobian.
    const J = this._J;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < DOF; c++) {
        J[r * DOF + c] = this._jacp[r * model.nv + this.dofAdr[c]];
        J[(r + 3) * DOF + c] = this._jacr[r * model.nv + this.dofAdr[c]];
      }
    }

    // dq = J^T (J J^T + lambda^2 I)^-1 err
    const A = new Float64Array(36);
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        let s = 0;
        for (let k = 0; k < DOF; k++) s += J[i * DOF + k] * J[j * DOF + k];
        A[i * 6 + j] = s + (i === j ? this.lambda * this.lambda : 0);
      }
    }
    const y = solve6(A, this._err);
    const dqJoint = new Float64Array(DOF);
    for (let c = 0; c < DOF; c++) {
      let s = 0;
      for (let r = 0; r < 6; r++) s += J[r * DOF + c] * y[r];
      dqJoint[c] = s;
    }

    // Nullspace posture bias keeps elbows from drifting into awkward folds
    // over a long episode. Projected so it never fights the task.
    if (this.nullGain > 0) {
      const bias = new Float64Array(DOF);
      for (let c = 0; c < DOF; c++) bias[c] = (this.homeQ[c] - this.qTarget[c]) * this.nullGain;
      const Jb = new Float64Array(6);
      for (let r = 0; r < 6; r++) {
        let s = 0;
        for (let c = 0; c < DOF; c++) s += J[r * DOF + c] * bias[c];
        Jb[r] = s;
      }
      const yb = solve6(A, Jb);
      for (let c = 0; c < DOF; c++) {
        let s = 0;
        for (let r = 0; r < 6; r++) s += J[r * DOF + c] * yb[r];
        dqJoint[c] += bias[c] - s;
      }
    }

    // Scale the whole step vector rather than clipping each joint. Per-joint
    // clipping changes the step's *direction*, which sends the gripper off the
    // straight line the operator is drawing with their hand.
    let maxAbs = 0;
    for (let c = 0; c < DOF; c++) maxAbs = Math.max(maxAbs, Math.abs(dqJoint[c]));
    const damped = maxAbs > this.maxStep;
    if (damped) for (let c = 0; c < DOF; c++) dqJoint[c] *= this.maxStep / maxAbs;

    let leashed = false;
    for (let c = 0; c < DOF; c++) {
      const [lo, hi] = this.limits[c];
      let q = Math.min(Math.max(this.qTarget[c] + dqJoint[c], lo), hi);
      const actual = data.qpos[this.qposAdr[c]];
      if (q > actual + this.maxLag) { q = actual + this.maxLag; leashed = true; }
      else if (q < actual - this.maxLag) { q = actual - this.maxLag; leashed = true; }
      this.qTarget[c] = q;
    }

    for (let c = 0; c < DOF; c++) data.ctrl[this.actIds[c]] = this.qTarget[c];
    return { posErr, rotErr, damped, leashed };
  }

  /** @param {number} open 0 = closed, 1 = fully open */
  setGripper(open) {
    const lo = this.sim.model.actuator_ctrlrange[this.gripperAct * 2];
    const hi = this.sim.model.actuator_ctrlrange[this.gripperAct * 2 + 1];
    this.sim.data.ctrl[this.gripperAct] = lo + (hi - lo) * Math.min(Math.max(open, 0), 1);
  }
}

// Gaussian elimination with partial pivoting on a fixed 6x6. Small enough that
// a general solver would cost more in indirection than it saves in code.
function solve6(A, b) {
  const M = Float64Array.from(A);
  const x = Float64Array.from(b);
  for (let col = 0; col < 6; col++) {
    let piv = col;
    for (let r = col + 1; r < 6; r++) {
      if (Math.abs(M[r * 6 + col]) > Math.abs(M[piv * 6 + col])) piv = r;
    }
    if (piv !== col) {
      for (let c = 0; c < 6; c++) {
        const t = M[col * 6 + c]; M[col * 6 + c] = M[piv * 6 + c]; M[piv * 6 + c] = t;
      }
      const t = x[col]; x[col] = x[piv]; x[piv] = t;
    }
    const d = M[col * 6 + col] || 1e-12;
    for (let r = col + 1; r < 6; r++) {
      const f = M[r * 6 + col] / d;
      if (f === 0) continue;
      for (let c = col; c < 6; c++) M[r * 6 + c] -= f * M[col * 6 + c];
      x[r] -= f * x[col];
    }
  }
  for (let r = 5; r >= 0; r--) {
    let s = x[r];
    for (let c = r + 1; c < 6; c++) s -= M[r * 6 + c] * x[c];
    x[r] = s / (M[r * 6 + r] || 1e-12);
  }
  return x;
}
