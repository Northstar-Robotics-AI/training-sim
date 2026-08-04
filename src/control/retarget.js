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

const XR_TO_MJ_QUAT = [Math.SQRT1_2, Math.SQRT1_2, 0, 0]; // wxyz, +90deg about X

/** WebXR (Y-up, -Z forward) position -> MuJoCo (Z-up) position. */
export function xrPosToMj(p) {
  return new Float64Array([p.x, -p.z, p.y]);
}

/** WebXR quaternion (xyzw) -> MuJoCo quaternion (wxyz). */
export function xrQuatToMj(mj, q) {
  const qXr = [q.w, q.x, q.y, q.z];
  const t = new Float64Array(4);
  const out = new Float64Array(4);
  const inv = new Float64Array(4);
  mj.mju_negQuat(inv, XR_TO_MJ_QUAT);
  mj.mju_mulQuat(t, XR_TO_MJ_QUAT, qXr);
  mj.mju_mulQuat(out, Array.from(t), Array.from(inv));
  return out;
}

export class ClutchRetargeter {
  /**
   * @param {object} mj  the MuJoCo WASM module (for quaternion math)
   * @param {object} cfg
   * @param {number} cfg.posScale  <1 scales hand motion down for fine work.
   *        0.5 roughly halves tremor at the cost of more re-clutching.
   * @param {boolean} cfg.lockRoll  drop controller roll; helps novices whose
   *        wrist rolls unconsciously while translating.
   */
  constructor(mj, { posScale = 1.0, rotScale = 1.0, lockRoll = false } = {}) {
    this.mj = mj;
    this.posScale = posScale;
    this.rotScale = rotScale;
    this.lockRoll = lockRoll;
    this.engaged = false;
    this.clutchCount = 0;
    this._anchorHandPos = null;
    this._anchorHandQuatInv = null;
    this._anchorEePos = null;
    this._anchorEeQuat = null;
  }

  /** @param {{pos:Float64Array, quat:Float64Array}} eePose current EE pose, MuJoCo frame */
  engage(handPosMj, handQuatMj, eePose) {
    this.engaged = true;
    this.clutchCount++;
    this._anchorHandPos = Float64Array.from(handPosMj);
    this._anchorHandQuatInv = new Float64Array(4);
    this.mj.mju_negQuat(this._anchorHandQuatInv, Array.from(handQuatMj));
    this._anchorEePos = Float64Array.from(eePose.pos);
    this._anchorEeQuat = Float64Array.from(eePose.quat);
  }

  release() {
    this.engaged = false;
  }

  /**
   * @returns {{pos:Float64Array, quat:Float64Array}|null} target EE pose, or
   *          null when the clutch is open (caller should hold position).
   */
  target(handPosMj, handQuatMj) {
    if (!this.engaged) return null;
    const pos = new Float64Array(3);
    for (let i = 0; i < 3; i++) {
      pos[i] = this._anchorEePos[i] + (handPosMj[i] - this._anchorHandPos[i]) * this.posScale;
    }

    // Relative rotation since engagement, applied to the anchored EE rotation.
    const dRot = new Float64Array(4);
    this.mj.mju_mulQuat(dRot, Array.from(handQuatMj), Array.from(this._anchorHandQuatInv));
    if (this.rotScale !== 1.0) slerpFromIdentity(dRot, this.rotScale);
    if (this.lockRoll) zeroRollAboutZ(dRot);

    const quat = new Float64Array(4);
    this.mj.mju_mulQuat(quat, Array.from(dRot), Array.from(this._anchorEeQuat));
    return { pos, quat };
  }
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
