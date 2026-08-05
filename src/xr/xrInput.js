// WebXR input plumbing.
//
// Button mapping (Quest Touch / standard xr-standard gamepad):
//   grip (button 1)     clutch  -- hold to move the arm
//   trigger (button 0)  gripper -- analog, so the operator can feel a soft close
//   Y (button 5, left)  reset episode, same as the 'r' key
//   A / X (button 4)    read as primaryEdge, currently unbound
//   B (button 5, right) read as secondaryEdge, currently unbound
//
// Edges here are recomputed every update() rather than latched, so they must be
// consumed at render rate. The clutch is the exception and says why.
//
// Poses are read in the *local-floor* reference space so the table sits at a
// believable height relative to the operator's real floor.

import * as THREE from 'three';

const BTN = { TRIGGER: 0, GRIP: 1, PRIMARY: 4, SECONDARY: 5 };

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

  /** Call once per rendered frame, before control. */
  update(frame, refSpace) {
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
      st.clutch = read(BTN.GRIP).pressed || read(BTN.GRIP).value > 0.5;

      // Clutch edges are latched until consumed, not cleared each frame.
      // Control runs on its own fixed clock and can execute zero times during a
      // render frame (any time the display is faster than the control rate), so
      // an edge cleared here is a clutch press the arm never sees -- which
      // looks exactly like a dead controller.
      const edge = st.clutch && !prev.clutch ? 'down'
        : (!st.clutch && prev.clutch ? 'up' : null);
      if (edge) st.clutchEdge = edge;

      const prim = read(BTN.PRIMARY).pressed;
      const sec = read(BTN.SECONDARY).pressed;
      st.primaryEdge = prim && !prev.primary;
      st.secondaryEdge = sec && !prev.secondary;

      st.thumbstick = gp && gp.axes.length >= 4 ? [gp.axes[2], gp.axes[3]] : [0, 0];

      prev.clutch = st.clutch;
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

  /** Read and clear a latched clutch transition. @returns {'down'|'up'|null} */
  consumeClutchEdge(side) {
    const st = this.state[side];
    const edge = st.clutchEdge;
    st.clutchEdge = null;
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
    clutch: false,
    clutchEdge: null,
    primaryEdge: false,
    secondaryEdge: false,
    thumbstick: [0, 0],
  };
}
