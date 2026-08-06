// WebXR input plumbing.
//
// Button mapping (Quest Touch / standard xr-standard gamepad):
//   grip (button 1)     hold 2s -- reset episode, same as the 'r' key
//   trigger (button 0)  gripper -- analog, so the operator can feel a soft close
//   Y (button 5, left)  toggle freeze/unfreeze, left arm
//   B (button 5, right) toggle freeze/unfreeze, right arm
//   X (button 4, left)  previous level, same as the 'p' key
//   A (button 4, right) next level, same as the 'n' key
//
// Edges here are recomputed every update() rather than latched, so they must be
// consumed at render rate. The freeze toggle and the reset hold are the
// exception and say why.
//
// Poses are read in the *local-floor* reference space so the table sits at a
// believable height relative to the operator's real floor.

import * as THREE from 'three';

const BTN = { TRIGGER: 0, GRIP: 1, PRIMARY: 4, SECONDARY: 5 };
const RESET_HOLD_S = 2.0;

export class XRInput {
  constructor(renderer) {
    this.renderer = renderer;
    this.hands = { left: null, right: null };
    this.state = {
      left: emptyHand(),
      right: emptyHand(),
      head: { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), valid: false },
    };
    this._prevButtons = { left: {}, right: {} };

    for (let i = 0; i < 2; i++) {
      const space = renderer.xr.getController(i);
      space.addEventListener('connected', (e) => {
        const side = e.data.handedness === 'left' ? 'left' : 'right';
        this.hands[side] = { space, gamepad: e.data.gamepad, inputSource: e.data };
      });
      space.addEventListener('disconnected', () => {
        for (const s of ['left', 'right']) {
          if (this.hands[s] && this.hands[s].space === space) this.hands[s] = null;
        }
      });
    }
  }

  /** Call once per rendered frame, before control. @param {number} dt seconds since the last call */
  update(frame, refSpace, dt = 0) {
    for (const side of ['left', 'right']) {
      const h = this.hands[side];
      const st = this.state[side];
      if (!h) { st.valid = false; continue; }

      h.space.getWorldPosition(st.pos);
      h.space.getWorldQuaternion(st.quat);
      st.valid = true;

      const gp = h.gamepad;
      const prev = this._prevButtons[side];
      const read = (i) => (gp && gp.buttons[i] ? gp.buttons[i] : { pressed: false, value: 0 });

      st.trigger = read(BTN.TRIGGER).value;

      // Grip is a hold-to-reset button, not a per-frame move gate, so it is
      // timed against wall-clock dt rather than edge-detected. resetEdge
      // latches once when the hold crosses the threshold, and is consumed at
      // render rate for the same reason the freeze toggle is below.
      const gripPressed = read(BTN.GRIP).pressed || read(BTN.GRIP).value > 0.5;
      st.grip = gripPressed;
      if (gripPressed) {
        const wasBelow = st.gripHeld < RESET_HOLD_S;
        st.gripHeld += dt;
        if (wasBelow && st.gripHeld >= RESET_HOLD_S) st.resetEdge = true;
      } else {
        st.gripHeld = 0;
      }

      const prim = read(BTN.PRIMARY).pressed;
      const sec = read(BTN.SECONDARY).pressed;
      st.primaryEdge = prim && !prev.primary;
      st.secondaryEdge = sec && !prev.secondary;

      // The freeze toggle flips on the button edge and latches until
      // consumed, not cleared each frame. Control runs on its own fixed
      // clock and can execute zero times during a render frame (any time the
      // display is faster than the control rate), so an edge cleared here is
      // a toggle the arm never sees -- which looks exactly like a dead
      // controller.
      if (st.secondaryEdge) {
        st.frozen = !st.frozen;
        st.freezeEdge = st.frozen ? 'freeze' : 'unfreeze';
      }

      st.thumbstick = gp && gp.axes.length >= 4 ? [gp.axes[2], gp.axes[3]] : [0, 0];

      prev.primary = prim;
      prev.secondary = sec;
    }

    if (frame && refSpace) {
      const viewer = frame.getViewerPose(refSpace);
      if (viewer) {
        const t = viewer.transform;
        this.state.head.pos.set(t.position.x, t.position.y, t.position.z);
        this.state.head.quat.set(t.orientation.x, t.orientation.y, t.orientation.z, t.orientation.w);
        this.state.head.valid = true;
      }
    }
  }

  /** Read and clear a latched freeze-toggle transition. @returns {'freeze'|'unfreeze'|null} */
  consumeFreezeEdge(side) {
    const st = this.state[side];
    const edge = st.freezeEdge;
    st.freezeEdge = null;
    return edge;
  }

  /** Read and clear a latched 2s grip-hold reset trigger. @returns {boolean} */
  consumeResetEdge(side) {
    const st = this.state[side];
    const edge = st.resetEdge;
    st.resetEdge = false;
    return edge;
  }

  /** Short haptic tick. Used for grasp confirmation and level events — the
   *  single biggest usability win for operators who can't feel contact. */
  pulse(side, intensity = 0.4, ms = 40) {
    const h = this.hands[side];
    const act = h && h.gamepad && h.gamepad.hapticActuators && h.gamepad.hapticActuators[0];
    if (act && act.pulse) act.pulse(intensity, ms);
  }
}

function emptyHand() {
  return {
    valid: false,
    pos: new THREE.Vector3(),
    quat: new THREE.Quaternion(),
    trigger: 0,
    grip: false,
    gripHeld: 0,
    resetEdge: false,
    // Arms start frozen: the operator must explicitly unfreeze (Y/B) before
    // the retargeter engages, same as the old clutch's default-released state.
    frozen: true,
    freezeEdge: null,
    primaryEdge: false,
    secondaryEdge: false,
    thumbstick: [0, 0],
  };
}
