// Per-arm differential IK.
//
// Solve the arm's IK to convergence against a scratch kinematics state, then
// rate-limit the command toward that answer. Runs at the control rate
// (~100 Hz), not the render rate, so headset framerate never changes the arm's
// behaviour.
//
// Why converge instead of taking one gradient step per tick: with a single
// step the rate limit *is* the convergence dynamics, so the controller has to
// decelerate exponentially into the goal to avoid a limit cycle, and the last
// few millimetres take longer than the first twenty centimetres. Converging
// first separates the two jobs -- the solve decides *where* to go, the rate
// limit decides *how fast* -- which is what removes the tail. Measured on
// tools/ik_bench.py: time to 20 mm 0.89 s -> 0.36 s, to 5 mm 1.10 s -> 0.61 s,
// against the incremental controller's own best tuning.
//
// Why damped least squares and not a plain pseudo-inverse: the YAM's wrist
// passes through a singularity when joint5 crosses zero, and an unregularised
// inverse produces a violent flick there. The damping term trades a little
// tracking accuracy for a bounded joint velocity, which is what an operator
// actually wants to feel.
//
// The three-level structure below matters, and collapsing any two of them
// reintroduces a specific failure:
//
//   qGoal    the solver's configuration. Free-running: it tracks the target
//            whether or not the arm can keep up, which is what keeps the warm
//            start good and the typical tick down to one or two iterations.
//   qTarget  the commanded configuration. Chases qGoal under the rate limit.
//   qpos     where the arm actually is. Lags qTarget by servo droop.

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
    this.scratch = sim.scratch();

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

    // Gravity-compensation feedforward actuators, summed with the position
    // actuator at each joint. gravityCompFactor is i2rt's own per-joint scale
    // on the inverse-dynamics gravity term (vendor/i2rt/i2rt/robots/config/yam.yml);
    // it exists there because the raw RNE torque runs slightly hot or cold per
    // joint against the real motors, so this is not something to re-derive.
    this.gcompActIds = [];
    for (let i = 1; i <= DOF; i++) this.gcompActIds.push(sim.id('actuator', `${prefix}gcomp${i}`));
    this.gravityCompFactor = [1.0, 1.1, 1.1, 1.2, 1.0, 1.0];

    this.homeQ = homeQ ?? new Array(DOF).fill(0);
    this.qTarget = Float64Array.from(this.homeQ);
    this.qGoal = Float64Array.from(this.homeQ);
    this.iTerm = new Float64Array(DOF);

    // Tuning. These are measured, not guessed -- see tools/ik_bench.py, which
    // runs this same algorithm against the same scene and reports settling
    // time, steady-state error and tracking error for both controllers.

    // DLS damping. Below ~0.05 the wrist snaps through its singularity and the
    // worst-case error jumps from 9 mm to 560 mm.
    this.lambda = 0.08;
    // Orientation weight relative to position in the least-squares. Only bites
    // when the target is over-constrained (a position plus an orientation the
    // arm cannot reach there), where it picks the compromise.
    this.rotWeight = 1.0;
    // Nullspace posture bias, projected so it never fights the task. Keeps
    // elbows from drifting into awkward folds over a long episode. At 0.02 the
    // bias dominates the task; 0.002 costs 0.25 mm and still unfolds the elbow.
    this.nullGain = 0.002;

    // --- inner solve ----------------------------------------------------
    // Warm starting keeps the typical tick at one or two iterations, so this
    // budget is only spent on the first tick after a clutch or on a target the
    // arm cannot reach. Raising it past 8 measured no difference to any
    // outcome; the solve is not what limits accuracy.
    this.maxIters = 8;
    this.posTol = 1e-3;      // 1 mm
    this.rotTol = 1e-2;      // ~0.6 deg
    // Per-iteration joint cap. Newton on a 6R arm will take a multi-radian
    // step toward a far target and land on a different IK branch (a wrist
    // flip); this keeps every iterate near the seed so the loop converges to
    // the branch it started on.
    this.iterStep = 0.20;
    // Give up when the iterate stops moving. An over-constrained target has no
    // solution, so the tolerance test never passes and the loop would burn its
    // whole budget every tick forever -- the damped least-squares iterate
    // still settles on the best compromise within a few steps.
    this.stepTol = 1e-4;

    // --- outer rate limit -----------------------------------------------
    // rad per tick; at 100 Hz that is 3 rad/s at the base and 8 at the wrist.
    // Split so a wrist correction is not rate-limited like a shoulder.
    this.maxDq = Float64Array.from([0.03, 0.03, 0.03, 0.08, 0.08, 0.08]);
    // Keep the command off the mechanical stop, so the position actuator is
    // not fighting the joint-limit constraint solver (which buzzes).
    this.limitMargin = 0.02;
    // Bound the free-running goal to the command. Without this a blocked arm
    // lets the goal chase the target indefinitely and the stored error
    // releases as a lurch when the block clears.
    this.maxGoalLag = 0.35;

    // --- branch-switch guard --------------------------------------------
    // Reject a solve that jumps this far ... but only when the goal was
    // already near the target, because a large joint move is also the correct
    // answer when the target is genuinely far away. Rejecting unconditionally
    // (what the reference implementation this was ported from does, because it
    // re-anchors the target afterwards) deadlocks without that re-anchor:
    // measured, the arm froze 200 mm short on 3 of 6 cases and never recovered.
    this.maxJump = 0.7;
    this.jumpNearPos = 0.05;
    this.jumpNearRot = 0.35;

    // --- integral droop correction ---------------------------------------
    // The converged solve is pure feedforward: it finds q with FK(q) = target
    // and commands it, but the position actuators are finite-stiffness
    // (kp 10-80, i2rt's official values -- soft, because most of the gravity
    // load is cancelled separately by the gcomp actuators below) so the arm
    // still settles a small residual droop short. This term mops up whatever
    // the feedforward gravity comp doesn't get exactly right (model error,
    // Coulomb friction, payload). Driving this off (goal - measured) rather than
    // (command - measured) is what makes it converge: the latter stays equal to
    // the droop no matter how large the correction gets, and winds up forever.
    //
    // Re-measured after switching to i2rt's official (softer) kp/kd: the old
    // 0.08 was tuned against the stiff hand-picked gains and now overdrives
    // the softer, laggier plant into a limit cycle that never settles (dEnd
    // oscillating 3-45 mm instead of converging). 0.08 was the ceiling before
    // it visibly degrades; 0.03 is comfortably inside it -- converges to
    // <1 mm residual with no oscillation in the same test.
    this.iGain = 0.03;
    this.iMax = 0.15;

    // Anti-windup leash against the measured position. The target is commanded
    // open-loop, so when the arm is blocked (contact, joint limit) it would
    // otherwise run away -- measured up to 1.8 rad before this existed, which
    // diverges and never recovers.
    //
    // Deliberately much wider than the incremental controller's 0.08. There it
    // bounded an integrator; here the command is feedforward, so the leash
    // instead caps tracking accuracy directly: at 0.12 it held the command
    // 2.4-9.6 deg behind a goal that was itself within 0.01 deg of the target.
    // Its anti-windup job survives intact at 0.25.
    this.maxLag = 0.25;

    this._jacp = new Float64Array(3 * sim.model.nv);
    this._jacr = new Float64Array(3 * sim.model.nv);
    this._J = new Float64Array(6 * DOF);
    this._err = new Float64Array(6);
    this._A = new Float64Array(36);
    this._q = new Float64Array(DOF);
    this._prevGoal = new Float64Array(DOF);
    this._qGoal = new Float64Array(DOF);
    this._qMeasured = new Float64Array(DOF);
    this._d = new Float64Array(DOF);
    this._dq = new Float64Array(DOF);
    this._bias = new Float64Array(DOF);
    this._Jb = new Float64Array(6);
    this._curQuat = new Float64Array(4);
    this._dqLocal = new Float64Array(3);
  }

  /** Snap every internal state to wherever the arm currently is. Call on reset
   *  and whenever the clutch re-engages, or the arm lurches to a stale target. */
  syncToCurrent() {
    for (let i = 0; i < DOF; i++) {
      const q = this.sim.data.qpos[this.qposAdr[i]];
      this.qTarget[i] = q;
      this.qGoal[i] = q;
      this.iTerm[i] = 0;
    }
  }

  currentQ() {
    const q = new Float64Array(DOF);
    for (let i = 0; i < DOF; i++) q[i] = this.sim.data.qpos[this.qposAdr[i]];
    return q;
  }

  /**
   * Task-space error of `data`'s current configuration, written into this._err
   * (weighted, for the solve).
   * @returns {{posErr:number, rotErr:number}} unweighted magnitudes, for tests
   */
  _poseError(data, targetPos, targetQuat) {
    const { mj } = this;
    const site = data.site(this.siteName);
    const curQuat = mat2Quat(mj, this._curQuat, site.xmat);

    for (let i = 0; i < 3; i++) this._err[i] = targetPos[i] - site.xpos[i];

    // mju_subQuat(res, qa, qb) solves qb * quat(res) = qa, so res is expressed
    // in the *current site's* frame. mj_jacSite's rotational block is
    // world-frame. Feeding one into the other is a silent, plausible-looking
    // bug: the arm still moves, it just never converges (measured ~60 deg of
    // permanent residual). Rotate into world.
    const dqLocal = subQuat(mj, this._dqLocal, targetQuat, curQuat);
    const R = site.xmat;
    for (let r = 0; r < 3; r++) {
      this._err[3 + r] = (R[r * 3] * dqLocal[0] + R[r * 3 + 1] * dqLocal[1]
        + R[r * 3 + 2] * dqLocal[2]) * this.rotWeight;
    }

    return {
      posErr: Math.hypot(this._err[0], this._err[1], this._err[2]),
      rotErr: Math.hypot(dqLocal[0], dqLocal[1], dqLocal[2]),
    };
  }

  /** Slice this arm's 6 columns out of the full nv-wide Jacobian of `data`. */
  _jacobian(data) {
    const { mj, sim } = this;
    jacSite(mj, sim.model, data, this._jacp, this._jacr, this.siteId);
    const J = this._J;
    const nv = sim.model.nv;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < DOF; c++) {
        J[r * DOF + c] = this._jacp[r * nv + this.dofAdr[c]];
        J[(r + 3) * DOF + c] = this._jacr[r * nv + this.dofAdr[c]];
      }
    }
    return J;
  }

  /**
   * Solve for a joint configuration reaching the target, seeded at `q0`.
   * Runs entirely on the scratch state; the live simulation is untouched.
   * @returns {{q:Float64Array, iters:number, converged:boolean,
   *            posErr:number, rotErr:number}} errors are those at the seed
   */
  _solve(q0, targetPos, targetQuat) {
    const { mj, sim, scratch } = this;
    const q = this._q;
    for (let i = 0; i < DOF; i++) {
      const [lo, hi] = this.limits[i];
      q[i] = Math.min(Math.max(q0[i], lo), hi);
    }

    let seedPos = 0;
    let seedRot = 0;
    for (let iter = 1; iter <= this.maxIters; iter++) {
      for (let i = 0; i < DOF; i++) scratch.qpos[this.qposAdr[i]] = q[i];
      // mj_kinematics gives xpos/xmat/xanchor/xaxis; mj_comPos gives the cdof
      // that mj_jacSite also needs. Together they are the complete set for a
      // position Jacobian, at a fraction of the cost of mj_forward.
      mj.mj_kinematics(sim.model, scratch);
      mj.mj_comPos(sim.model, scratch);

      const { posErr, rotErr } = this._poseError(scratch, targetPos, targetQuat);
      if (iter === 1) { seedPos = posErr; seedRot = rotErr; }
      if (posErr < this.posTol && rotErr < this.rotTol) {
        return { q, iters: iter, converged: true, posErr: seedPos, rotErr: seedRot };
      }

      const J = this._jacobian(scratch);

      // dq = J^T (J J^T + lambda^2 I)^-1 err
      const A = this._A;
      for (let i = 0; i < 6; i++) {
        for (let j = 0; j < 6; j++) {
          let s = 0;
          for (let k = 0; k < DOF; k++) s += J[i * DOF + k] * J[j * DOF + k];
          A[i * 6 + j] = s + (i === j ? this.lambda * this.lambda : 0);
        }
      }
      const y = solve6(A, this._err);
      const dq = this._dq;
      for (let c = 0; c < DOF; c++) {
        let s = 0;
        for (let r = 0; r < 6; r++) s += J[r * DOF + c] * y[r];
        dq[c] = s;
      }

      if (this.nullGain > 0) {
        const bias = this._bias;
        for (let c = 0; c < DOF; c++) bias[c] = (this.homeQ[c] - q[c]) * this.nullGain;
        const Jb = this._Jb;
        for (let r = 0; r < 6; r++) {
          let s = 0;
          for (let c = 0; c < DOF; c++) s += J[r * DOF + c] * bias[c];
          Jb[r] = s;
        }
        const yb = solve6(A, Jb);
        for (let c = 0; c < DOF; c++) {
          let s = 0;
          for (let r = 0; r < 6; r++) s += J[r * DOF + c] * yb[r];
          dq[c] += bias[c] - s;
        }
      }

      let maxAbs = 0;
      for (let c = 0; c < DOF; c++) maxAbs = Math.max(maxAbs, Math.abs(dq[c]));
      if (maxAbs < this.stepTol) {
        return { q, iters: iter, converged: false, posErr: seedPos, rotErr: seedRot };
      }
      const scale = maxAbs > this.iterStep ? this.iterStep / maxAbs : 1;
      for (let c = 0; c < DOF; c++) {
        const [lo, hi] = this.limits[c];
        q[c] = Math.min(Math.max(q[c] + dq[c] * scale, lo), hi);
      }
    }
    return { q, iters: this.maxIters, converged: false, posErr: seedPos, rotErr: seedRot };
  }

  /**
   * One control tick toward a world-frame target, in MuJoCo coordinates.
   * @param {Float64Array} targetPos  xyz
   * @param {Float64Array} targetQuat wxyz
   */
  step(targetPos, targetQuat) {
    const { data } = this.sim;

    this._prevGoal.set(this.qGoal);
    const sol = this._solve(this._prevGoal, targetPos, targetQuat);
    const qGoal = this._qGoal;
    qGoal.set(sol.q);

    // Branch-switch guard. sol.posErr/rotErr are the errors at the seed, i.e.
    // at the previous goal -- so this asks "the goal was already on target, so
    // why does the solver now want to move 0.7 rad?", which only a different
    // IK branch explains.
    let jump = 0;
    for (let c = 0; c < DOF; c++) {
      jump = Math.max(jump, Math.abs(qGoal[c] - this._prevGoal[c]));
    }
    const rejected = jump > this.maxJump
      && sol.posErr < this.jumpNearPos && sol.rotErr < this.jumpNearRot;
    if (rejected) qGoal.set(this._prevGoal);

    for (let c = 0; c < DOF; c++) {
      qGoal[c] = Math.min(Math.max(qGoal[c], this.qTarget[c] - this.maxGoalLag),
        this.qTarget[c] + this.maxGoalLag);
    }
    this.qGoal.set(qGoal);

    // Integral droop correction, gated on the feedforward command having
    // caught up to the goal. Mid-slew the arm is legitimately far behind, and
    // integrating that gap saturates iTerm in the direction of travel,
    // overshoots, then has to unwind -- measured, 0.7 s of extra settling on
    // every case. Measure the gap against the feedforward part of the command
    // only: comparing against qTarget (which already carries iTerm) makes the
    // integral switch itself off the moment it grows past one tick's motion,
    // which left 12 mm on the table.
    const qMeasured = this._qMeasured;
    for (let c = 0; c < DOF; c++) qMeasured[c] = data.qpos[this.qposAdr[c]];

    let settling = true;
    for (let c = 0; c < DOF; c++) {
      if (Math.abs(qGoal[c] - (this.qTarget[c] - this.iTerm[c])) > this.maxDq[c]) {
        settling = false;
        break;
      }
    }
    if (settling) {
      for (let c = 0; c < DOF; c++) {
        this.iTerm[c] = Math.min(Math.max(
          this.iTerm[c] + (qGoal[c] - qMeasured[c]) * this.iGain, -this.iMax), this.iMax);
      }
    }

    // Rate limit, scaled per group rather than per joint or per vector.
    //
    // Clipping each joint independently changes the step's *direction*, which
    // bends the path away from the line the operator is drawing with their
    // hand. Scaling the whole vector by its most-binding joint preserves
    // direction but throws away the point of having split caps at all: a
    // saturated shoulder then throttles the wrist too. Splitting into base
    // (1-3, which own position) and wrist (4-6, which own orientation) keeps
    // direction within each group while letting the wrist run at its own cap
    // while the base is saturated.
    const d = this._d;
    for (let c = 0; c < DOF; c++) d[c] = qGoal[c] + this.iTerm[c] - this.qTarget[c];
    let damped = false;
    for (const [from, to] of GROUPS) {
      let s = 1;
      for (let c = from; c < to; c++) {
        const mag = Math.abs(d[c]);
        if (mag > 1e-12) s = Math.min(s, this.maxDq[c] / mag);
      }
      if (s < 1) {
        damped = true;
        for (let c = from; c < to; c++) d[c] *= s;
      }
    }

    let leashed = false;
    for (let c = 0; c < DOF; c++) {
      const [lo, hi] = this.limits[c];
      const loM = Math.min(lo + this.limitMargin, hi);
      const hiM = Math.max(hi - this.limitMargin, lo);
      let q = Math.min(Math.max(this.qTarget[c] + d[c], loM), hiM);
      const actual = qMeasured[c];
      if (q > actual + this.maxLag) { q = actual + this.maxLag; leashed = true; }
      else if (q < actual - this.maxLag) { q = actual - this.maxLag; leashed = true; }
      this.qTarget[c] = q;
    }

    for (let c = 0; c < DOF; c++) data.ctrl[this.actIds[c]] = this.qTarget[c];

    // Gravity-comp feedforward. qfrc_bias is MuJoCo's own Coriolis+gravity
    // generalised force, computed each mj_step -- so this reads last tick's
    // value, a one-tick zero-order hold at 100+ Hz. Mirrors i2rt's
    // motor_torques = pd_torque + g * gravity_comp_factor (motor_chain_robot.py),
    // where g there is compute_inverse_dynamics(q, 0, 0); using qfrc_bias
    // instead of a separate zero-velocity RNE call picks up the (usually
    // negligible, at teleop speeds) Coriolis term too.
    for (let c = 0; c < DOF; c++) {
      data.ctrl[this.gcompActIds[c]] = data.qfrc_bias[this.dofAdr[c]] * this.gravityCompFactor[c];
    }

    // Report the error the operator actually sees -- at the measured pose, not
    // the solver's.
    const live = this._poseError(data, targetPos, targetQuat);
    return {
      posErr: live.posErr,
      rotErr: live.rotErr,
      damped,
      leashed,
      rejected,
      iters: sol.iters,
      converged: sol.converged,
    };
  }

  /** @param {number} open 0 = closed, 1 = fully open */
  setGripper(open) {
    const lo = this.sim.model.actuator_ctrlrange[this.gripperAct * 2];
    const hi = this.sim.model.actuator_ctrlrange[this.gripperAct * 2 + 1];
    this.sim.data.ctrl[this.gripperAct] = lo + (hi - lo) * Math.min(Math.max(open, 0), 1);
  }
}

// Base joints own position, wrist joints own orientation. See the rate limit.
const GROUPS = [[0, 3], [3, 6]];

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
