#!/usr/bin/env python3
"""
Reference implementation and benchmark for the arm controller in
src/control/ik.js. Line-for-line the same algorithm, so a change here can be
validated in seconds and then transcribed, instead of being debugged inside a
headset where the only diagnostic is "the arm feels wrong".

Run:  python tools/ik_bench.py --scene public/assets/yam/bimanual_yam.xml
      python tools/ik_bench.py --sweep     # pick constants from measurement

Two controllers live here so a change can be argued against numbers:

  Incremental  (--mode old)
    One damped-least-squares gradient step per control tick, evaluated at the
    *measured* qpos and integrated into qTarget. Convergence is a first-order
    lag: posGain sets the time constant and maxStep clips the rate. Three bugs
    were found this way, all of which look fine in a screenshot:
      1. mju_subQuat returns the rotation error in the *site's* frame, while
         mj_jacSite's rotational rows are world-frame. Mixing them leaves
         ~60 deg of permanent residual.
      2. The joint target is integrated open-loop. When the arm is blocked, the
         target runs away from the true position (measured 1.8 rad) and never
         recovers. Fixed by leashing the target to the measured qpos.
      3. At posGain=1 the damped-least-squares step is always above the
         velocity limit, so the controller never decelerates near the goal and
         settles into a ~9 mm limit cycle.

  Converged    (--mode new)
    Solve the IK to convergence against a scratch MjData, warm-started from
    last tick's answer, then rate-limit the move toward that answer. The rate
    limit stops being part of the convergence dynamics and becomes a plain
    velocity governor, which is what removes the exponential deceleration tail
    (bug 3 above) rather than tuning around it.
"""
import argparse
import numpy as np
import mujoco

HOME = [0.0, 0.9, 1.2, 0.0, -0.5, 0.0]

CONTROL_HZ = 100

# ── Incremental controller (current src/control/ik.js) ────────────────────────
OLD_GAINS = dict(lam=0.08, max_step=0.01, pos_gain=0.35, rot_gain=0.21,
                 null_gain=0.002, max_lag=0.08, err_cap=0.02)

# ── Converged controller (proposed) ───────────────────────────────────────────
# Defaults here are what --sweep settled on; see the transcribed values in
# src/control/ik.js.
NEW_GAINS = dict(
    lam=0.08,            # DLS damping, unchanged -- still guards the wrist singularity
    rot_weight=1.0,      # orientation vs position weight in the least-squares
    null_gain=0.002,     # nullspace posture bias, unchanged
    # Anti-windup leash vs measured qpos. Raised from the incremental
    # controller's 0.08 because the two controllers use it for different
    # things. There it bounded an open-loop integrator; here the command is
    # feedforward (q such that FK(q) = target), so the leash instead caps
    # tracking accuracy directly at max_lag -- measured, 0.12 rad held the
    # command 2.4-9.6 deg behind a goal that was itself within 0.01 deg. Its
    # anti-windup job survives at 0.25; 0.40 tests no better on tracking and
    # stores 23 deg of release when a blocked arm frees.
    max_lag=0.25,
    # inner solve
    max_iters=8,
    pos_tol=1e-3,        # 1 mm
    rot_tol=1e-2,        # ~0.6 deg
    iter_step=0.20,      # per-iteration joint cap, rad -- bounds branch leaps
    step_tol=1e-4,       # give up when the iterate stops moving (unreachable)
    # outer rate limit, rad per control tick (x100 Hz = 3 and 8 rad/s)
    max_dq=np.array([0.03, 0.03, 0.03, 0.08, 0.08, 0.08]),
    limit_margin=0.02,   # keep the command off the mechanical stop
    max_goal_lag=0.35,   # bound the free-running goal to the command
    # branch-switch guard
    max_jump=0.7,        # reject a converged q this far from the previous goal ...
    jump_near_pos=0.05,  # ... but only when already this close in position ...
    jump_near_rot=0.35,  # ... and this close in orientation.
    # integral droop correction -- re-measured for the official i2rt kp/kd
    # plus gravity comp, see the note in ConvergedIK.step.
    i_gain=0.03,
    i_max=0.15,
    i_gate=1.0,          # integrate only within this multiple of one tick's move
)


def site_pose(model, data, sid):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, data.site_xmat[sid])
    return data.site_xpos[sid].copy(), q


def pose_error(model, data, sid, target_pos, target_quat, rot_weight=1.0):
    """6-vector task error at the current configuration of `data`.

    The rotational half is rotated out of the site frame and into world, to
    match mj_jacSite's rotational rows. Getting this wrong is silent: the arm
    still moves, it just never converges.
    """
    pos, cq = site_pose(model, data, sid)
    dq_local = np.zeros(3)
    mujoco.mju_subQuat(dq_local, target_quat, cq)
    dq_world = data.site_xmat[sid].reshape(3, 3) @ dq_local
    err = np.concatenate([target_pos - pos, dq_world * rot_weight])
    return err, float(np.linalg.norm(target_pos - pos)), float(np.linalg.norm(dq_local))


class ArmBase:
    def __init__(self, model, data, prefix='left_', site='left_grasp_site'):
        self.m, self.d = model, data
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{prefix}joint{i}')
               for i in range(1, 7)]
        self.qadr = [model.jnt_qposadr[j] for j in jid]
        self.dof = [model.jnt_dofadr[j] for j in jid]
        self.lo, self.hi = model.jnt_range[jid, 0], model.jnt_range[jid, 1]
        self.act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'{prefix}act{i}')
                    for i in range(1, 7)]
        # Gravity-comp feedforward, applied through qfrc_applied rather than an
        # actuator -- mirrors src/control/ik.js. Without it the official (soft)
        # i2rt kp/kd below let the arm droop noticeably under its own weight;
        # through an actuator it would sit inside actuatorfrcrange and get
        # clipped at full extension. See the note in src/control/ik.js.
        #
        # Unity, not yam.yml's [1.0, 1.1, 1.1, 1.2, 1.0, 1.0] -- that is a
        # hardware trim for the real motors, and here the gravity model is the
        # plant.
        self.grav_factor = np.ones(6)
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self.home = np.array(HOME, float)
        self.qt = self.home.copy()
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def sync_to_current(self):
        self.qt = np.array([self.d.qpos[a] for a in self.qadr])

    def measured_q(self):
        return np.array([self.d.qpos[a] for a in self.qadr])

    def jac(self, data):
        mujoco.mj_jacSite(self.m, data, self._jp, self._jr, self.sid)
        return np.vstack([self._jp[:, self.dof], self._jr[:, self.dof]])

    def emit(self):
        for i, a in enumerate(self.act):
            self.d.ctrl[a] = self.qt[i]
        for i, dof in enumerate(self.dof):
            self.d.qfrc_applied[dof] = self.d.qfrc_bias[dof] * self.grav_factor[i]


class IncrementalIK(ArmBase):
    """One DLS gradient step per tick, integrated into qTarget."""

    def __init__(self, *a, **gains):
        super().__init__(*a)
        self.g = dict(OLD_GAINS, **gains)

    def step(self, target_pos, target_quat):
        g = self.g
        err, pe, re = pose_error(self.m, self.d, self.sid, target_pos, target_quat)
        err[:3] *= g['pos_gain']
        err[3:] *= g['rot_gain']
        if np.linalg.norm(err[:3]) > g['err_cap']:
            err[:3] *= g['err_cap'] / np.linalg.norm(err[:3])

        J = self.jac(self.d)
        A = J @ J.T + g['lam'] ** 2 * np.eye(6)
        step = J.T @ np.linalg.solve(A, err)
        if g['null_gain'] > 0:
            bias = (self.home - self.qt) * g['null_gain']
            step += bias - J.T @ np.linalg.solve(A, J @ bias)

        # Scale, do not clip -- clipping per joint rotates the step.
        mx = np.abs(step).max()
        damped = mx > g['max_step']
        if damped:
            step *= g['max_step'] / mx

        qt = np.clip(self.qt + step, self.lo, self.hi)
        qa = self.measured_q()
        leashed = bool(np.any(np.abs(qt - qa) > g['max_lag']))
        self.qt = np.clip(qt, qa - g['max_lag'], qa + g['max_lag'])
        self.emit()
        return dict(pos_err=pe, rot_err=re, damped=damped, leashed=leashed, iters=1)


class ConvergedIK(ArmBase):
    """Solve to convergence on a scratch MjData, then rate-limit toward it."""

    def __init__(self, *a, **gains):
        super().__init__(*a)
        self.g = dict(NEW_GAINS, **gains)
        # Scratch state so the solve never disturbs the live simulation. The
        # inner loop needs FK and a Jacobian at hypothetical configurations;
        # doing that in place would corrupt whatever reads site poses between
        # the control tick and the render.
        self.sd = mujoco.MjData(self.m)
        # The solver's configuration, kept separate from the command. Seeding
        # the solve from the rate-limited command instead means that whenever
        # the arm saturates, the seed falls behind the target and every solve
        # restarts from far away: measured on the tracking case, that burned
        # the full 8-iteration budget every tick and left 29 deg RMS of
        # orientation error. mink keeps the same separation (its Configuration
        # is integrated freely); VRTeleop collapses the two by passing its
        # rate-limited current_q as init_q, which is the likelier reason their
        # wrist poses burn the iteration budget than the budget being small.
        self.q_goal = self.qt.copy()
        self.i_term = np.zeros(6)
        self.jumps = 0
        self.fails = 0

    def sync_to_current(self):
        super().sync_to_current()
        self.q_goal = self.qt.copy()
        self.i_term[:] = 0.0

    def solve(self, q0, target_pos, target_quat):
        """@return (q, iters, converged, pos_err_at_seed, rot_err_at_seed)"""
        g = self.g
        q = np.clip(q0.copy(), self.lo, self.hi)
        pe0 = re0 = None
        for it in range(1, g['max_iters'] + 1):
            for i, a in enumerate(self.qadr):
                self.sd.qpos[a] = q[i]
            # mj_kinematics gives xpos/xmat/xanchor/xaxis; mj_comPos gives the
            # cdof that mj_jacSite also needs. Together they are the complete
            # set for a position-only Jacobian, at a fraction of mj_forward.
            mujoco.mj_kinematics(self.m, self.sd)
            mujoco.mj_comPos(self.m, self.sd)

            err, pe, re = pose_error(self.m, self.sd, self.sid,
                                     target_pos, target_quat, g['rot_weight'])
            if pe0 is None:
                pe0, re0 = pe, re
            if pe < g['pos_tol'] and re < g['rot_tol']:
                return q, it, True, pe0, re0

            J = self.jac(self.sd)
            A = J @ J.T + g['lam'] ** 2 * np.eye(6)
            step = J.T @ np.linalg.solve(A, err)
            if g['null_gain'] > 0:
                bias = (self.home - q) * g['null_gain']
                step += bias - J.T @ np.linalg.solve(A, J @ bias)

            # An over-constrained target (position plus an orientation the arm
            # cannot reach there) has no solution, so the tolerance test never
            # passes and the loop burns its whole budget every tick, forever --
            # this is the case VRTeleop cut max_iters 25 -> 10 to survive. The
            # damped least-squares iterate still settles on the best compromise
            # within a few steps, so stop when it stops moving and spend the
            # budget only where it can still buy something.
            mx = np.abs(step).max()
            if mx < g['step_tol']:
                return q, it, False, pe0, re0

            # Cap the per-iteration move. Newton on a 6R arm will happily take
            # a multi-radian step toward a far target and land on a different
            # IK branch (a wrist flip); this keeps every iterate near the seed
            # so the loop converges to the branch we started on.
            if mx > g['iter_step']:
                step *= g['iter_step'] / mx
            q = np.clip(q + step, self.lo, self.hi)
        return q, g['max_iters'], False, pe0, re0

    def step(self, target_pos, target_quat):
        g = self.g
        # Warm start from the previous *goal*, not the measured qpos: the
        # measured arm lags the command under load, and seeding from it drags
        # the solution backwards every tick. Warm starting is also what keeps
        # the typical tick at one or two iterations -- the target only moves a
        # couple of millimetres between ticks.
        prev_goal = self.q_goal.copy()
        q_goal, iters, converged, pe_now, re_now = self.solve(
            prev_goal, target_pos, target_quat)
        if not converged:
            self.fails += 1

        # Branch-switch guard. A converged solver can hop to a different IK
        # branch (typically a wrist unwind near a limit), which reads as a
        # violent flick. But a large joint move is also the *correct* answer
        # when the target is genuinely far away, so gate the rejection on the
        # goal already being near the target: only then is a 0.7 rad move
        # necessarily a different solution rather than a longer trip to the
        # same one. VRTeleop rejects unconditionally and re-anchors the target
        # afterwards; without a re-anchor that same test deadlocks -- measured
        # here, the arm froze 200 mm short on 3 of 6 cases and never recovered.
        # pe_now/re_now are the errors at the seed, i.e. at the previous goal.
        jump = float(np.abs(q_goal - prev_goal).max())
        rejected = (jump > g['max_jump']
                    and pe_now < g['jump_near_pos'] and re_now < g['jump_near_rot'])
        if rejected:
            self.jumps += 1
            q_goal = prev_goal

        # Keep the free-running goal within reach of the command. Without this
        # a blocked arm lets the goal chase the target indefinitely and the
        # stored error releases as a lurch when the block clears -- the same
        # open-loop windup the command leash below exists to prevent, one level
        # up. Wide enough that ordinary saturation still gets a good warm start.
        q_goal = np.clip(q_goal, self.qt - g['max_goal_lag'], self.qt + g['max_goal_lag'])
        self.q_goal = q_goal.copy()

        qa = self.measured_q()

        # Integral droop correction. The converged solve is pure feedforward:
        # it finds q with FK(q) = target and commands it, but the position
        # actuators are finite-stiffness (kp 10-80, i2rt's official values) so
        # the arm still settles a small residual droop short even with gravity
        # comp cancelling most of the static load. Driving this off
        # (q_goal - measured) rather than (command - measured) is what makes it
        # converge: the latter stays equal to the droop no matter how large the
        # correction gets, and winds up forever.
        #
        # i_gain was re-measured after switching from hand-tuned to official
        # kp/kd (2.5-4x softer) plus adding gravity comp: 0.08 now overdrives
        # the softer plant into a limit cycle that never settles (--sweep shows
        # worst-case tracking getting *worse* from 0.02 to 0.08). 0.03 is
        # comfortably inside the stable range.
        #
        # Integrate only once the command has caught up to the goal. Mid-slew
        # the arm is legitimately far behind, and integrating that gap saturates
        # i_term at i_max in the direction of travel, overshoots, and then has to
        # unwind -- measured, that cost 0.7 s of settling time on every case.
        # Measure the gap against the *feedforward* part of the command only.
        # Comparing against self.qt (which already carries i_term) makes the
        # integral switch itself off the moment it grows past one tick's worth
        # of motion, which is why it froze at ~0.02 rad and left 12 mm on the
        # table.
        d_ff = q_goal - (self.qt - self.i_term)
        settling = bool(np.all(np.abs(d_ff) <= g['max_dq'] * g['i_gate']))
        if settling:
            self.i_term = np.clip(self.i_term + (q_goal - qa) * g['i_gain'],
                                  -g['i_max'], g['i_max'])
        q_cmd = q_goal + self.i_term

        # Rate limit, scaled per group rather than per joint or per vector.
        #
        # Clipping each joint independently (what VRTeleop does) changes the
        # step's *direction*, which bends the path away from the line the
        # operator is drawing with their hand. Scaling the whole vector by its
        # most-binding joint preserves direction but throws away the point of
        # having per-joint caps at all: a saturated shoulder then throttles the
        # wrist too, and orientation tracking degrades to 7.6 deg RMS against
        # 2.5 for the incremental controller.
        #
        # Splitting into base (1-3, which own position) and wrist (4-6, which
        # own orientation) and scaling each by its own binding ratio keeps
        # direction within each group while letting the wrist run at its own
        # cap while the base is saturated -- which is the behaviour the split
        # caps were for.
        d = q_cmd - self.qt
        damped = False
        for grp in (slice(0, 3), slice(3, 6)):
            mag = np.abs(d[grp])
            ratios = np.where(mag > 1e-12, g['max_dq'][grp] / np.maximum(mag, 1e-12), np.inf)
            s = float(min(1.0, ratios.min()))
            if s < 1.0:
                damped = True
                d[grp] *= s
        qt = self.qt + d

        lo = self.lo + g['limit_margin']
        hi = self.hi - g['limit_margin']
        qt = np.clip(qt, np.minimum(lo, hi), np.maximum(lo, hi))

        # Anti-windup leash against the measured position. VRTeleop has no
        # equivalent -- their arm is compliant and they re-anchor on failure.
        # This one has hard contacts and no re-anchor, so without the leash a
        # blocked arm's target still runs away.
        leashed = bool(np.any(np.abs(qt - qa) > g['max_lag']))
        self.qt = np.clip(qt, qa - g['max_lag'], qa + g['max_lag'])
        self.emit()

        # Report error at the *measured* pose so old and new are comparable.
        _, pe, re = pose_error(self.m, self.d, self.sid, target_pos, target_quat)
        return dict(pos_err=pe, rot_err=re, damped=damped, leashed=leashed,
                    iters=iters, converged=converged, rejected=rejected)


CASES = [
    ('down 15cm',          [0.00, 0.00, -0.15], [0, 0, 1], 0.00),
    ('fwd+left',           [0.12, 0.10, -0.08], [0, 0, 1], 0.00),
    ('down + 45deg wrist', [0.05, -0.05, -0.20], [1, 0, 0], 0.785),
    ('up + yaw 60deg',     [-0.05, 0.08, 0.10], [0, 0, 1], 1.05),
    ('lateral 20cm',       [0.00, -0.20, 0.00], [0, 1, 0], 0.30),
    ('approach pose',      [0.10, 0.05, -0.25], [0, 1, 0], 0.90),
]


def rotq(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array(axis, float), angle)
    return q


def run_case(model, data, cls, gains, dp, axis, ang, n_ticks, substeps):
    mujoco.mj_resetData(model, data)
    ik = cls(model, data, **gains)
    for i, a in enumerate(ik.qadr):
        data.qpos[a] = HOME[i]
    for i, a in enumerate(ik.act):
        data.ctrl[a] = HOME[i]
    mujoco.mj_forward(model, data)

    p0, q0 = site_pose(model, data, ik.sid)
    tq = np.zeros(4)
    mujoco.mju_mulQuat(tq, rotq(axis, ang), q0)
    tp = p0 + np.array(dp)

    t20 = t5 = None
    tail, leash_ticks, iters = [], 0, []
    for k in range(n_ticks):
        r = ik.step(tp, tq)
        iters.append(r['iters'])
        if t20 is None and r['pos_err'] < 0.020:
            t20 = k / CONTROL_HZ
        if t5 is None and r['pos_err'] < 0.005:
            t5 = k / CONTROL_HZ
        if k > n_ticks - 150:
            tail.append(r['pos_err'])
        leash_ticks += r['leashed']
        for _ in range(substeps):
            mujoco.mj_step(model, data)

    return dict(steady_mm=float(np.mean(tail) * 1000),
                rot_deg=float(np.degrees(r['rot_err'])),
                t20=t20, t5=t5,
                leash=leash_ticks / n_ticks,
                mean_iters=float(np.mean(iters)),
                max_iters=int(np.max(iters)),
                jumps=getattr(ik, 'jumps', 0),
                fails=getattr(ik, 'fails', 0))


def run_track(model, data, cls, gains, substeps, seconds=6.0,
              speed=0.15, radius=0.05, rot_amp=0.15, rot_hz=0.25):
    """Follow a moving target, which is what teleop actually is.

    The step-response cases above over-weight the slew regime -- an operator
    almost never asks for a 20 cm jump, they draw a continuous path. What they
    do feel is the lag between hand and gripper while drawing it, so that is
    what this measures: a horizontal circle at a plausible hand speed with the
    wrist rocking, reporting steady-state tracking error rather than time to
    settle.

    Keep the path reachable or this measures nothing. A 0.10 m circle with the
    wrist rocking +/-0.5 rad leaves 87 mm / 32 deg of residual under a
    300-iteration from-scratch solve -- no controller can track that, and both
    of these duly scored ~90 mm RMS, which looks like a result and is not one.
    The defaults below are verified against that same solve at under 1 mm.
    """
    mujoco.mj_resetData(model, data)
    ik = cls(model, data, **gains)
    for i, a in enumerate(ik.qadr):
        data.qpos[a] = HOME[i]
    for i, a in enumerate(ik.act):
        data.ctrl[a] = HOME[i]
    mujoco.mj_forward(model, data)

    p0, q0 = site_pose(model, data, ik.sid)
    omega = speed / radius
    n_ticks = int(seconds * CONTROL_HZ)
    errs, rerrs, iters = [], [], []
    for k in range(n_ticks):
        t = k / CONTROL_HZ
        tp = p0 + np.array([radius * (np.cos(omega * t) - 1.0),
                            radius * np.sin(omega * t),
                            0.0])
        tq = np.zeros(4)
        mujoco.mju_mulQuat(tq, rotq([1, 0, 0], rot_amp * np.sin(2 * np.pi * rot_hz * t)), q0)
        r = ik.step(tp, tq)
        iters.append(r['iters'])
        # Skip the first second: that is the arm acquiring the path, not
        # tracking it.
        if t > 1.0:
            errs.append(r['pos_err'])
            rerrs.append(r['rot_err'])
        for _ in range(substeps):
            mujoco.mj_step(model, data)

    return dict(rms_mm=float(np.sqrt(np.mean(np.square(errs))) * 1000),
                max_mm=float(np.max(errs) * 1000),
                rms_deg=float(np.degrees(np.sqrt(np.mean(np.square(rerrs))))),
                mean_iters=float(np.mean(iters)),
                max_iters=int(np.max(iters)))


def report(model, data, cls, gains, n_ticks, substeps, title):
    print(f'\n{title}')
    print(f'  {"case":<22}{"steady":>10}{"rot":>10}{"t<20mm":>9}{"t<5mm":>9}'
          f'{"leash":>8}{"iters":>13}')
    print('  ' + '-' * 81)
    worst, t5s, iters = 0.0, [], []
    for name, dp, axis, ang in CASES:
        r = run_case(model, data, cls, gains, dp, axis, ang, n_ticks, substeps)
        worst = max(worst, r['steady_mm'])
        if r['t5'] is not None:
            t5s.append(r['t5'])
        iters.append(r['mean_iters'])
        it = f"{r['mean_iters']:.2f}/{r['max_iters']}"
        s20 = f"{r['t20']:.2f}s" if r['t20'] is not None else '--'
        s5 = f"{r['t5']:.2f}s" if r['t5'] is not None else '--'
        print(f"  {name:<22}{r['steady_mm']:>7.3f} mm{r['rot_deg']:>7.2f} deg"
              f"{s20:>9}{s5:>9}{r['leash']:>7.0%}{it:>13}")
    print(f'  worst steady-state: {worst:.3f} mm    '
          f'mean t<5mm: {np.mean(t5s):.2f}s ({len(t5s)}/{len(CASES)} reached)    '
          f'mean iters/tick: {np.mean(iters):.2f}')
    return worst, (np.mean(t5s) if t5s else float("nan")), len(t5s)


def sweep(model, data, n_ticks, substeps):
    """Pick the converged controller's constants from measurement.

    Ordered by how much each one actually moved the numbers. max_lag and
    i_gain dominate; the inner-solve knobs barely register, because warm
    starting keeps the typical tick at one iteration no matter what budget it
    is given.
    """
    def row(label, gains):
        g = dict(NEW_GAINS, **gains)
        w, t5, n = quiet(model, data, g, n_ticks, substeps)
        trk = run_track(model, data, ConvergedIK, g, substeps)
        print(f'  {label:>14}{w:>10.3f}{t5:>10.2f}{n:>4}/{len(CASES)}'
              f'{trk["rms_mm"]:>10.2f}{trk["rms_deg"]:>10.2f}{g["_iters"]:>9.2f}')

    hdr = (f'  {"":>14}{"worst mm":>10}{"t<5mm":>10}{"hit":>7}'
           f'{"trk mm":>10}{"trk deg":>10}{"iters":>9}')

    print('\n=== max_lag (anti-windup leash, rad) ===')
    print('  With a feedforward command this caps tracking accuracy directly,')
    print('  which is not the role it played in the incremental controller.')
    print(hdr)
    for v in (0.08, 0.12, 0.25, 0.40):
        row(f'{v:.2f}', dict(max_lag=v))

    print('\n=== i_gain (droop correction) ===')
    print(hdr)
    for v in (0.0, 0.02, 0.08, 0.15):
        row(f'{v:.2f}', dict(i_gain=v))

    print('\n=== max_dq (rate limit, rad/tick @100Hz -> rad/s) ===')
    print(hdr)
    for base, wrist in ((0.01, 0.01), (0.02, 0.05), (0.03, 0.08), (0.05, 0.14)):
        row(f'{base * CONTROL_HZ:.0f}/{wrist * CONTROL_HZ:.0f} rad/s',
            dict(max_dq=np.array([base] * 3 + [wrist] * 3)))

    print('\n=== max_iters (inner budget) ===')
    print(hdr)
    for v in (3, 4, 8, 12, 20):
        row(str(v), dict(max_iters=v))

    print('\n=== iter_step (per-iteration joint cap, rad) ===')
    print(hdr)
    for v in (0.05, 0.10, 0.20, 0.50):
        row(f'{v:.2f}', dict(iter_step=v))


def quiet(model, data, gains, n_ticks, substeps):
    worst, t5s, iters, leash = 0.0, [], [], []
    for name, dp, axis, ang in CASES:
        r = run_case(model, data, ConvergedIK, gains, dp, axis, ang, n_ticks, substeps)
        worst = max(worst, r['steady_mm'])
        if r['t5'] is not None:
            t5s.append(r['t5'])
        iters.append(r['mean_iters'])
        leash.append(r['leash'])
    gains['_iters'] = float(np.mean(iters))
    gains['_leash'] = float(np.mean(leash))
    return worst, (float(np.mean(t5s)) if t5s else float('nan')), len(t5s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='public/assets/yam/bimanual_yam.xml')
    ap.add_argument('--seconds', type=float, default=7.0)
    ap.add_argument('--mode', choices=('old', 'new', 'both'), default='both')
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    substeps = int(round((1.0 / CONTROL_HZ) / model.opt.timestep))
    n_ticks = int(args.seconds * CONTROL_HZ)

    if args.sweep:
        sweep(model, data, n_ticks, substeps)
        return

    if args.mode in ('old', 'both'):
        report(model, data, IncrementalIK, {}, n_ticks, substeps,
               'INCREMENTAL — one DLS step per tick (current ik.js)')
    if args.mode in ('new', 'both'):
        report(model, data, ConvergedIK, {}, n_ticks, substeps,
               'CONVERGED — solve to convergence, then rate-limit')

    print('\nnote: "up + yaw 60deg" does not reach 5 mm under either controller. '
          'Position plus a 60 deg yaw is over-constrained for a 6-DoF arm in '
          'that region, so the least-squares settles on a compromise. That is '
          'correct behaviour, not a tuning failure.')


if __name__ == '__main__':
    main()
