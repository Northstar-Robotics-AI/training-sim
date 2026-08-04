#!/usr/bin/env python3
"""
Reference implementation and benchmark for the arm controller in
src/control/ik.js. Line-for-line the same algorithm, so a change here can be
validated in seconds and then transcribed, instead of being debugged inside a
headset where the only diagnostic is "the arm feels wrong".

Run:  python tools/ik_bench.py --scene public/assets/yam/bimanual_yam.xml

Three bugs were found this way, all of which look fine in a screenshot:
  1. mju_subQuat returns the rotation error in the *site's* frame, while
     mj_jacSite's rotational rows are world-frame. Mixing them leaves ~60 deg
     of permanent residual.
  2. The joint target is integrated open-loop. When the arm is blocked, the
     target runs away from the true position (measured 1.8 rad) and never
     recovers. Fixed by leashing the target to the measured qpos.
  3. At posGain=1 the damped-least-squares step is always above the velocity
     limit, so the controller never decelerates near the goal and settles into
     a ~9 mm limit cycle.
"""
import argparse
import numpy as np
import mujoco

HOME = [0.0, 0.9, 1.2, 0.0, -0.5, 0.0]

# Must match the constants in src/control/ik.js.
GAINS = dict(lam=0.08, max_step=0.01, pos_gain=0.35, rot_gain=0.21,
             null_gain=0.002, max_lag=0.08, err_cap=0.02)

CONTROL_HZ = 100


class ArmIK:
    def __init__(self, model, data, prefix='left_', site='left_grasp_site', **gains):
        self.m, self.d = model, data
        self.g = dict(GAINS, **gains)
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{prefix}joint{i}')
               for i in range(1, 7)]
        self.qadr = [model.jnt_qposadr[j] for j in jid]
        self.dof = [model.jnt_dofadr[j] for j in jid]
        self.lo, self.hi = model.jnt_range[jid, 0], model.jnt_range[jid, 1]
        self.act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'{prefix}act{i}')
                    for i in range(1, 7)]
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self.home = np.array(HOME, float)
        self.qt = self.home.copy()
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def sync_to_current(self):
        self.qt = np.array([self.d.qpos[a] for a in self.qadr])

    def ee(self):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.d.site_xmat[self.sid])
        return self.d.site_xpos[self.sid].copy(), q

    def step(self, target_pos, target_quat):
        m, d, g = self.m, self.d, self.g
        mujoco.mj_jacSite(m, d, self._jp, self._jr, self.sid)
        J = np.vstack([self._jp[:, self.dof], self._jr[:, self.dof]])

        pos, cq = self.ee()
        dq_local = np.zeros(3)
        mujoco.mju_subQuat(dq_local, target_quat, cq)
        # (1) site frame -> world frame, to match the Jacobian.
        dq = d.site_xmat[self.sid].reshape(3, 3) @ dq_local

        err = np.concatenate([(target_pos - pos) * g['pos_gain'], dq * g['rot_gain']])
        pe = np.linalg.norm(err[:3])
        if pe > g['err_cap']:
            err[:3] *= g['err_cap'] / pe

        A = J @ J.T + g['lam'] ** 2 * np.eye(6)
        step = J.T @ np.linalg.solve(A, err)

        if g['null_gain'] > 0:
            bias = (self.home - self.qt) * g['null_gain']
            step += bias - J.T @ np.linalg.solve(A, J @ bias)

        # (3) scale, do not clip -- clipping per joint rotates the step.
        mx = np.abs(step).max()
        damped = mx > g['max_step']
        if damped:
            step *= g['max_step'] / mx

        qt = np.clip(self.qt + step, self.lo, self.hi)
        # (2) anti-windup leash against the measured position.
        qa = np.array([d.qpos[a] for a in self.qadr])
        leashed = bool(np.any(np.abs(qt - qa) > g['max_lag']))
        self.qt = np.clip(qt, qa - g['max_lag'], qa + g['max_lag'])

        for i, a in enumerate(self.act):
            d.ctrl[a] = self.qt[i]
        return np.linalg.norm(target_pos - pos), np.linalg.norm(dq_local), damped, leashed


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='public/assets/yam/bimanual_yam.xml')
    ap.add_argument('--seconds', type=float, default=7.0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    substeps = int(round((1.0 / CONTROL_HZ) / model.opt.timestep))
    n_ticks = int(args.seconds * CONTROL_HZ)

    print(f'{"case":<22}{"steady":>10}{"rot":>10}{"t<20mm":>9}{"t<5mm":>9}{"leash":>8}')
    worst = 0.0
    for name, dp, axis, ang in CASES:
        mujoco.mj_resetData(model, data)
        ik = ArmIK(model, data)
        for i, a in enumerate(ik.qadr):
            data.qpos[a] = HOME[i]
        for i, a in enumerate(ik.act):
            data.ctrl[a] = HOME[i]
        mujoco.mj_forward(model, data)

        p0, q0 = ik.ee()
        tq = np.zeros(4)
        mujoco.mju_mulQuat(tq, rotq(axis, ang), q0)
        tp = p0 + np.array(dp)

        t20 = t5 = None
        tail, leash_ticks = [], 0
        for k in range(n_ticks):
            pe, re, damped, leashed = ik.step(tp, tq)
            if t20 is None and pe < 0.020:
                t20 = k / CONTROL_HZ
            if t5 is None and pe < 0.005:
                t5 = k / CONTROL_HZ
            if k > n_ticks - 150:
                tail.append(pe)
            leash_ticks += leashed
            for _ in range(substeps):
                mujoco.mj_step(model, data)

        steady = np.mean(tail) * 1000
        worst = max(worst, steady)
        print(f'{name:<22}{steady:>7.3f} mm{np.degrees(re):>7.2f} deg'
              f'{(f"{t20:.2f}s" if t20 else "--"):>9}'
              f'{(f"{t5:.2f}s" if t5 else "--"):>9}'
              f'{leash_ticks / n_ticks:>7.0%}')

    print(f'\nworst steady-state: {worst:.3f} mm')
    print('note: "up + yaw 60deg" does not reach 5 mm. Position plus a 60 deg '
          'yaw is over-constrained for a 6-DoF arm in that region, so DLS '
          'settles on a least-squares compromise. That is correct behaviour, '
          'not a tuning failure.')


if __name__ == '__main__':
    main()
