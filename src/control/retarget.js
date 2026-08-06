// Map a WebXR controller pose onto an end-effector target.
//
// Absolute mapping (controller pose -> EE pose directly) is unusable: the
// operator's hand and the YAM's wrist have different workspaces, and the arm
// slams into a limit the moment the hand leaves the reachable shell. So this
// uses the standard clutch: hold the grip button, and only the *delta* since
// engagement is applied on top of the EE pose frozen at engagement.
//
// Releasing and re-centering the hand is how the operator "ratchets" across a
// workspace larger than their arm span.
//
// This stage also owns input conditioning, and that is not cosmetic. The IK
// downstream now solves to convergence, so it tracks whatever it is handed --
// including hand tremor. The controller it replaced ran a 0.35 proportional
// gain, which was accidentally acting as a low-pass filter on the input; take
// that away without putting a real filter here and the jitter arrives at the
// wrist instead.

import { mulQuat, negQuat } from '../sim/mjmath.js';

const XR_TO_MJ_QUAT = [Math.SQRT1_2, Math.SQRT1_2, 0, 0]; // wxyz, +90deg about X

/** WebXR (Y-up, -Z forward) position -> MuJoCo (Z-up) position. */
export function xrPosToMj(p) {
  return new Float64Array([p.x, -p.z, p.y]);
}

/** WebXR quaternion (xyzw) -> MuJoCo quaternion (wxyz). */
export function xrQuatToMj(mj, q) {
  const qXr = [q.w, q.x, q.y, q.z];
  const inv = negQuat(mj, new Float64Array(4), XR_TO_MJ_QUAT);
  const t = mulQuat(mj, new Float64Array(4), XR_TO_MJ_QUAT, qXr);
  return mulQuat(mj, new Float64Array(4), t, inv);
}

export class ClutchRetargeter {
  /**
   * @param {object} mj  the MuJoCo WASM module (for quaternion math)
   * @param {object} cfg
   * @param {number} cfg.posScale  <1 scales hand motion down for fine work.
   *        0.5 roughly halves tremor at the cost of more re-clutching.
   * @param {boolean} cfg.lockRoll  drop controller roll; helps novices whose
   *        wrist rolls unconsciously while translating.
   * @param {number} cfg.posAlpha  position EMA, 1 = off.
   * @param {number} cfg.rotAlpha  orientation EMA, 1 = off. Deliberately lower
   *        than posAlpha: the wrist solution of a 6-DoF arm is uniquely
   *        determined, so a noisy orientation target is the one thing that
   *        makes it dither frame to frame, and the motors ratchet on that.
   *        Smoothing the target fixes it at the source, where an output
   *        low-pass would only shrink the wobble. Costs rotation lag only,
   *        never position lag.
   * @param {number} cfg.ffGain  velocity feed-forward, to buy back some of the
   *        lag the EMA above introduces. Small on purpose -- prediction
   *        amplifies jitter, which is what we are here to remove.
   * @param {number} cfg.posReach  metres the target may sit from the anchor.
   * @param {number} cfg.rotReach  radians the target may sit from the anchor.
   */
  constructor(mj, {
    posScale = 1.0, rotScale = 1.0, lockRoll = false,
    posAlpha = 0.7, rotAlpha = 0.4, ffGain = 0.10,
    posReach = 0.5, rotReach = 1.2,
  } = {}) {
    this.mj = mj;
    this.posScale = posScale;
    this.rotScale = rotScale;
    this.lockRoll = lockRoll;
    this.posAlpha = posAlpha;
    this.rotAlpha = rotAlpha;
    this.ffGain = ffGain;
    this.posReach = posReach;
    this.rotReach = rotReach;
    this.engaged = false;
    this.clutchCount = 0;
    this._anchorHandPos = null;
    this._anchorHandQuatInv = null;
    this._anchorEePos = null;
    this._anchorEeQuat = null;
    this._smPos = null;
    this._smQuat = null;
    this._prevPos = null;
  }

  /** @param {{pos:Float64Array, quat:Float64Array}} eePose current EE pose, MuJoCo frame */
  engage(handPosMj, handQuatMj, eePose) {
    this.engaged = true;
    this.clutchCount++;
    this._anchorHandPos = Float64Array.from(handPosMj);
    this._anchorHandQuatInv = negQuat(this.mj, new Float64Array(4), handQuatMj);
    this._anchorEePos = Float64Array.from(eePose.pos);
    this._anchorEeQuat = Float64Array.from(eePose.quat);
    // Seed the filters at the current hand pose, so engaging does not drag the
    // arm through a stale smoothed value.
    this._smPos = Float64Array.from(handPosMj);
    this._smQuat = Float64Array.from(handQuatMj);
    this._prevPos = Float64Array.from(handPosMj);
  }

  release() {
    this.engaged = false;
  }

  /**
   * @param {Float64Array} handPosMj
   * @param {Float64Array} handQuatMj
   * @returns {{pos:Float64Array, quat:Float64Array}|null} target EE pose, or
   *          null when the clutch is open (caller should hold position).
   */
  target(handPosMj, handQuatMj) {
    if (!this.engaged) return null;

    // --- input conditioning -------------------------------------------
    // Feed-forward first, then smooth the prediction: smoothing the raw pose
    // and predicting afterwards would extrapolate from an already-lagged
    // value and give back the lag it was meant to cancel.
    const pos = new Float64Array(3);
    for (let i = 0; i < 3; i++) {
      const vel = handPosMj[i] - this._prevPos[i];
      const pred = handPosMj[i] + vel * this.ffGain;
      this._smPos[i] += this.posAlpha * (pred - this._smPos[i]);
      pos[i] = this._smPos[i];
    }
    this._prevPos.set(handPosMj);
    nlerpToward(this._smQuat, handQuatMj, this.rotAlpha);
    const quat = this._smQuat;

    // --- clutch delta --------------------------------------------------
    const outPos = new Float64Array(3);
    for (let i = 0; i < 3; i++) {
      outPos[i] = this._anchorEePos[i] + (pos[i] - this._anchorHandPos[i]) * this.posScale;
    }

    // Relative rotation since engagement, applied to the anchored EE rotation.
    const dRot = mulQuat(this.mj, new Float64Array(4), quat, this._anchorHandQuatInv);
    if (this.rotScale !== 1.0) slerpFromIdentity(dRot, this.rotScale);
    if (this.lockRoll) zeroRollAboutZ(dRot);

    // --- reach clamp ---------------------------------------------------
    // Truncate a target that sits further from the anchor than the arm can
    // usefully follow, rather than handing the solver something unreachable
    // and letting it return a least-squares compromise that reads as the arm
    // fighting the hand. Position and orientation clamp independently, so
    // over-reaching in one does not corrupt the other.
    let over = 0;
    for (let i = 0; i < 3; i++) over += (outPos[i] - this._anchorEePos[i]) ** 2;
    over = Math.sqrt(over);
    if (over > this.posReach) {
      const k = this.posReach / over;
      for (let i = 0; i < 3; i++) {
        outPos[i] = this._anchorEePos[i] + (outPos[i] - this._anchorEePos[i]) * k;
      }
    }
    const angle = 2 * Math.acos(Math.min(Math.max(Math.abs(dRot[0]), -1), 1));
    if (angle > this.rotReach) slerpFromIdentity(dRot, this.rotReach / angle);

    const outQuat = mulQuat(this.mj, new Float64Array(4), dRot, this._anchorEeQuat);
    return { pos: outPos, quat: outQuat };
  }
}

/** In-place EMA on a unit quaternion: q <- normalize(q + t*(target - q)).
 *  Exact slerp is not worth it here -- per-tick deltas are small, and the
 *  normalize keeps the result a valid rotation either way. */
function nlerpToward(q, target, t) {
  // Quaternion double cover: q and -q are the same rotation, but lerping
  // between opposite signs takes the long way round and flips the wrist.
  let dot = q[0] * target[0] + q[1] * target[1] + q[2] * target[2] + q[3] * target[3];
  const s = dot < 0 ? -1 : 1;
  let n = 0;
  for (let i = 0; i < 4; i++) {
    q[i] += t * (s * target[i] - q[i]);
    n += q[i] * q[i];
  }
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < 4; i++) q[i] /= n;
}

function slerpFromIdentity(q, t) {
  const w = Math.min(Math.max(q[0], -1), 1);
  const angle = 2 * Math.acos(w);
  if (angle < 1e-6) return;
  const s = Math.sin(angle / 2);
  const ax = [q[1] / s, q[2] / s, q[3] / s];
  const a2 = (angle * t) / 2;
  const s2 = Math.sin(a2);
  q[0] = Math.cos(a2);
  q[1] = ax[0] * s2; q[2] = ax[1] * s2; q[3] = ax[2] * s2;
}

function zeroRollAboutZ(q) {
  // Project out the component about the world Z axis.
  const dot = q[3];
  const w = q[0];
  const n = Math.hypot(w, dot) || 1;
  const rollW = w / n;
  const rollZ = dot / n;
  const inv = [rollW, 0, 0, -rollZ];
  const out = [
    q[0] * inv[0] - q[1] * inv[1] - q[2] * inv[2] - q[3] * inv[3],
    q[0] * inv[1] + q[1] * inv[0] + q[2] * inv[3] - q[3] * inv[2],
    q[0] * inv[2] - q[1] * inv[3] + q[2] * inv[0] + q[3] * inv[1],
    q[0] * inv[3] + q[1] * inv[2] - q[2] * inv[1] + q[3] * inv[0],
  ];
  q[0] = out[0]; q[1] = out[1]; q[2] = out[2]; q[3] = out[3];
}
