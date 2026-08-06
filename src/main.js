import * as THREE from 'three';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { Sim } from './sim/sim.js';
import { buildSceneGraph } from './sim/sceneBuilder.js';
import { ArmIK } from './control/ik.js';
import { ClutchRetargeter, xrPosToMj, xrQuatToMj } from './control/retarget.js';
import { XRInput } from './xr/xrInput.js';
import { CURRICULUM } from './levels/curriculum.js';
import { composeSceneXML, makeRng, EpisodeMetrics } from './levels/Level.js';
import { Recorder, makeSampler } from './record/recorder.js';
import { HUD } from './ui/hud.js';

const BASE_XML_URL = '/assets/yam/bimanual_yam.xml';

// The mesh list used to be hand-maintained here, which drifted the moment
// the generator (tools/build_yam_scene.py) started emitting collision hulls
// alongside the visual STLs -- the WASM loader would 404 on any <mesh file>
// this list didn't happen to mention. The XML's own <asset> block is the
// actual source of truth, so read it from there instead.
function meshUrlsFromXml(xml) {
  const urls = [];
  for (const m of xml.matchAll(/<mesh\b[^>]*\bfile="([^"]+)"/g)) urls.push(`/assets/yam/meshes/${m[1]}`);
  return urls;
}

// Nullspace posture bias: elbow up, wrist level. A mid-range working posture,
// deliberately *not* the rest pose below -- biasing the solver toward a folded
// stow pose would pull the arm back toward its own joint stops.
const HOME_Q = [0, 0.9, 1.2, 0, -0.5, 0];

// Rest posture, per side: arms folded back over their own shoulders with the
// gripper level and pointing straight down +x, so an episode starts clear of
// the table with both tools presented the same way.
//
// joint1 is the shoulder yaw about world +z, and the two bases toe in by
// 0.3 rad each (see the base quats in bimanual_yam.xml), so the ±0.3 cancels
// the toe-in -- with joint1 at 0 the two grippers would splay 34 deg apart.
// The remaining joints are the fold: joint2/joint3 swing the upper arm back
// and the forearm forward again over the shoulder, and joint4 levels the
// wrist. Every one keeps >0.4 rad of clearance from its stop, so the first IK
// solve of an episode is not starting against a limit.
const REST_Q = {
  left: [0.3, 0.4, 0.55, -0.15, 0, 0],
  right: [-0.3, 0.4, 0.55, -0.15, 0, 0],
};

// Control runs on its own fixed clock. 100 Hz is well above what an operator
// can perceive and well below what the WASM build struggles with.
const CONTROL_HZ = 100;

// Where the VR user's play-space origin lands in the scene, in three.js world
// coordinates (Y-up). The arm bases sit at world x=-0.05, centred on z=0 (see
// bimanual_yam.xml's left/right_base pos, rotated by sceneBuilder's MJ_TO_XR).
// Base spot is a stride behind them with both arms and the table in frame --
// third person, not standing inside the robot -- then nudged 30cm forward and
// 50cm up from there for a better vantage over the table. y=0 would be the
// physical floor ('local-floor' puts the reference space origin there and the
// headset supplies actual eye height on top), so +0.5 raises the origin itself
// to roughly chest height.
const XR_SPAWN_POS = new THREE.Vector3(-1.1 + 0.3, 0.5, 0);
// Faces +X (toward the arms/table) from the default -Z forward, then tilts
// 20 degrees down to look more at the table than the horizon. The down-tilt
// is a rotation about world +Z, which is where +X-facing's local right axis
// lands after the yaw -- so it has to be composed on the left (applied after
// the yaw), not folded into one axis-angle.
const XR_SPAWN_YAW = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2);
const XR_SPAWN_TILT = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -20 * Math.PI / 180);
const XR_SPAWN_QUAT = XR_SPAWN_TILT.clone().multiply(XR_SPAWN_YAW);

class App {
  constructor() {
    this.levelIndex = 0;
    this.recorder = new Recorder();
    this.metrics = new EpisodeMetrics();
    this.results = [];      // recent outcomes for gating
    this.seed = 1;
    this.physHz = 0;
    this._ctlAcc = 0;
    this._physCount = 0;
    this._physWindow = 0;
  }

  async boot() {
    this.baseXml = await fetch(BASE_XML_URL).then((r) => r.text());
    this.meshUrls = meshUrlsFromXml(this.baseXml);
    this.initRenderer();
    await this.loadLevel(0);
    this.renderer.setAnimationLoop((t, frame) => this.frame(t, frame));
  }

  initRenderer() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0d12);
    this.scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x202028, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(1.5, 3, 1.2);
    key.castShadow = true;
    this.scene.add(key);

    this.camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 40);
    this.camera.position.set(-0.9, 1.5, 0.9);

    // The XR device pose overwrites camera.matrix directly every frame while
    // presenting, so repositioning the camera itself for VR would also have
    // to un-happen the moment the session ends. Instead the camera (and the
    // controllers, so tracked hands stay consistent with where the head
    // lands) sit under a dolly that's identity on the desktop path and only
    // gets moved on 'sessionstart' below.
    this.xrRig = new THREE.Group();
    this.scene.add(this.xrRig);
    this.xrRig.add(this.camera);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(innerWidth, innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.xr.enabled = true;
    // Foveation trades edge sharpness for fill rate. On Quest the arms are
    // centre-screen anyway, so this is close to free.
    this.renderer.xr.setFoveation(0.6);
    document.body.appendChild(this.renderer.domElement);
    document.body.appendChild(VRButton.createButton(this.renderer));

    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.target.set(0.35, 0.95, 0);
    this.orbit.update();

    this.xr = new XRInput(this.renderer);
    for (let i = 0; i < 2; i++) this.xrRig.add(this.renderer.xr.getController(i));

    // Placed fresh on every 'sessionstart' -- including the first one after a
    // page reload -- so the operator always steps into VR standing behind the
    // arms, never wherever their headset happened to be tracking from.
    this.renderer.xr.addEventListener('sessionstart', () => {
      this.xrRig.position.copy(XR_SPAWN_POS);
      this.xrRig.quaternion.copy(XR_SPAWN_QUAT);
    });
    // The XR pose leaves camera.matrix in rig-local coordinates from wherever
    // the headset last was; with the rig back at identity that would render
    // from a bogus spot on the desktop path until the user next drags to
    // orbit. Reset both so exiting VR lands back on the desktop framing.
    this.renderer.xr.addEventListener('sessionend', () => {
      this.xrRig.position.set(0, 0, 0);
      this.xrRig.quaternion.identity();
      this.camera.position.set(-0.9, 1.5, 0.9);
      this.camera.quaternion.identity();
      this.orbit.update();
    });

    addEventListener('resize', () => {
      this.camera.aspect = innerWidth / innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(innerWidth, innerHeight);
    });

    addEventListener('keydown', (e) => {
      if (e.key === 'r') this.resetEpisode();
      if (e.key === 'n') this.loadLevel(Math.min(this.levelIndex + 1, CURRICULUM.length - 1));
      if (e.key === 'p') this.loadLevel(Math.max(this.levelIndex - 1, 0));
      if (e.key === 'c') this.gfx.setGroupVisible(3, this._proxies = !this._proxies);
      if (e.key === 'd') this.recorder.download();
    });
  }

  async loadLevel(index) {
    // `level` stays local until everything it depends on is ready. The render
    // loop keeps ticking during the awaits below (evaluate() runs every frame
    // whenever status is still 'running', e.g. a manual skip mid-episode), and
    // if this.level flipped early it would run the *new* level's tick/success
    // against the *old*, still-current this.ctx/this.sim -- reading a site
    // name (like target_site) that only exists in the model being compiled,
    // not the one still loaded. That throw is uncaught inside
    // setAnimationLoop's callback, which silently kills the whole render loop
    // -- exactly what looked like a freeze switching levels.
    const level = CURRICULUM[index];

    // Compiling per level rather than hiding unused props keeps each level's
    // contact set minimal, which is the difference between 90 Hz and 45 Hz
    // once a level has a dozen loose objects in it.
    const xml = composeSceneXML(this.baseXml, level);
    if (this.gfx) this.scene.remove(this.gfx.root);
    const sim = await new Sim().init({
      xmlUrl: URL.createObjectURL(new Blob([xml], { type: 'text/xml' })),
      meshUrls: this.meshUrls,
    });
    const gfx = buildSceneGraph(sim, { hiddenGroups: [3] });
    this.scene.add(gfx.root);

    const arms = {
      left: new ArmIK(sim, { prefix: 'left_', siteName: 'left_grasp_site', homeQ: HOME_Q }),
      right: new ArmIK(sim, { prefix: 'right_', siteName: 'right_grasp_site', homeQ: HOME_Q }),
    };
    // posReach is generous because posScale already shrinks hand travel: at
    // 0.6, half a metre of arm travel is 0.83 m of hand travel, which is past
    // where an operator would re-clutch anyway.
    const retargetCfg = { posScale: 0.6, posReach: 0.5, rotReach: 1.2 };
    const retarget = {
      left: new ClutchRetargeter(sim.mj, retargetCfg),
      right: new ClutchRetargeter(sim.mj, retargetCfg),
    };

    if (!this.hud) this.hud = new HUD(this.scene);

    // Commit the new level's state in one synchronous block -- level, sim,
    // gfx, arms, retarget, and ctx (built inside resetEpisode) all flip
    // together, so evaluate() can never see a mix of old and new.
    this.levelIndex = index;
    this.level = level;
    this.results = [];
    this.sim = sim;
    this.gfx = gfx;
    this.arms = arms;
    this.retarget = retarget;
    this.sampler = makeSampler(sim, arms, this.xr, level);
    this.resetEpisode();
  }

  resetEpisode() {
    // Cancel any pending end-of-episode advance. Without this, resetting
    // manually during the 1.8 s results pause lets that timer fire on top of
    // the fresh episode -- resetting it a second time, or skipping a level.
    clearTimeout(this._endTimer);
    this._endTimer = null;

    const sim = this.sim;
    sim.reset();
    for (const side of ['left', 'right']) {
      const ik = this.arms[side];
      const rest = REST_Q[side];
      for (let i = 0; i < 6; i++) sim.data.qpos[ik.qposAdr[i]] = rest[i];
      // Resets the solver goal and the integral term too, not just the
      // command -- leaving either behind carries the previous episode's
      // droop correction into the new one.
      ik.syncToCurrent();
      for (let i = 0; i < 6; i++) sim.data.ctrl[ik.actIds[i]] = rest[i];
      ik.setGripper(1);
      this.retarget[side].release();
    }
    sim.mj.mj_forward(sim.model, sim.data);

    this.ctx = {
      sim,
      rng: makeRng(this.seed++),
      state: {},
      progress: 0,
      readout: '',
      eePos: { left: [0, 0, 0], right: [0, 0, 0] },
      setBodyPose: (name, pos, quat) => this.setBodyPose(name, pos, quat),
      setJoint: (name, v) => {
        sim.data.qpos[sim.model.jnt_qposadr[sim.id('joint', name)]] = v;
      },
      getJoint: (name) => sim.data.qpos[sim.model.jnt_qposadr[sim.id('joint', name)]],
    };
    // Not 'running' while the level's reset() runs -- some levels call
    // setBodyPose from here, which throws if it thinks it's mid-episode.
    // Leaving `this.status` at whatever the *previous* episode ended on
    // (still 'running', if this reset was a manual R-key/Y-button reset
    // rather than a post-episode one) would make that guard false-positive
    // and throw inside the render loop, which looks like a freeze.
    this.status = 'resetting';
    this.level.reset(this.ctx);
    sim.mj.mj_forward(sim.model, sim.data);

    this.metrics.reset();
    this._prevEE = null;
    this._ikRej = { left: 0, right: 0 };
    this.status = 'running';
    this.tRemaining = this.level.timeLimit;
    this.recorder.start({ levelId: this.level.id, seed: this.seed - 1 });
  }

  /**
   * Move a body that has no freejoint (level targets, fixtures).
   *
   * This writes model.body_pos, which mutates the *compiled model*, not the
   * state -- so it is a reset-time operation only. Calling it mid-episode
   * would silently change the model the recorded demo was collected against,
   * which makes the episode unreplayable. Guarded rather than commented.
   */
  setBodyPose(name, pos, quat) {
    if (this.status === 'running') {
      throw new Error(`setBodyPose("${name}") called mid-episode; `
        + 'give the body a freejoint and use placeFree instead');
    }
    const sim = this.sim;
    const bid = sim.id('body', name);
    sim.model.body_pos[bid * 3] = pos[0];
    sim.model.body_pos[bid * 3 + 1] = pos[1];
    sim.model.body_pos[bid * 3 + 2] = pos[2];
    if (quat) for (let i = 0; i < 4; i++) sim.model.body_quat[bid * 4 + i] = quat[i];
  }

  /** Contacts where both geoms belong to an arm. The model already excludes
   *  the legitimate gripper-internal pairs, so anything left is the operator
   *  folding the arm into itself or driving one arm into the other. */
  selfCollisionCount() {
    const { model, data } = this.sim;
    if (!this._armGeoms) {
      this._armGeoms = new Set();
      for (let g = 0; g < model.ngeom; g++) {
        const name = this.sim.mj.mj_id2name(
          model, this.sim.mj.mjtObj.mjOBJ_BODY.value, model.geom_bodyid[g]) || '';
        if (name.startsWith('left_') || name.startsWith('right_')) this._armGeoms.add(g);
      }
    }
    let n = 0;
    for (let i = 0; i < data.ncon; i++) {
      const c = data.contact.get(i);
      if (this._armGeoms.has(c.geom1) && this._armGeoms.has(c.geom2)) n++;
    }
    return n;
  }

  control(dt) {
    const { sim, arms, retarget } = this;
    for (const side of ['left', 'right']) {
      const s = this.xr.state[side];
      const ik = arms[side];
      const rt = retarget[side];

      // No controller data: hold still. Gravity comp still needs to run --
      // see the holdGravity() call below the freeze handling for why this
      // can't just be ik.step() with the site's own current pose as target.
      if (!s.valid) { ik.holdGravity(); continue; }

      const freezeEdge = this.xr.consumeFreezeEdge(side);
      if (freezeEdge === 'unfreeze') {
        ik.syncToCurrent();
        rt.engage(xrPosToMj(s.pos), xrQuatToMj(sim.mj, s.quat), sim.sitePose(`${side}_grasp_site`));
        this.xr.pulse(side, 0.3, 25);
        this.metrics.clutchCount[side]++;
        this.recorder.event('unfreeze', { side });
      } else if (freezeEdge === 'freeze') {
        rt.release();
        this.recorder.event('freeze', { side });
      }

      const target = rt.target(xrPosToMj(s.pos), xrQuatToMj(sim.mj, s.quat));
      if (target) {
        const r = ik.step(target.pos, target.quat);
        this._ikDbg = this._ikDbg || {};
        this._ikDbg[side] = { ...r, tgt: target.pos };
        if (r.rejected) this._ikRej[side]++;
        // Buzz only when the leash engages -- the arm is genuinely blocked by
        // contact or a joint limit. `damped` fires during any normal fast slew,
        // so buzzing on that would vibrate continuously and mean nothing.
        if (r.leashed) this.xr.pulse(side, 0.15, 15);
      } else {
        // Frozen: hold the last commanded qTarget (still being applied to
        // the position actuators every tick regardless) but keep the gravity
        // feedforward fresh. Must not be ik.step() here -- see holdGravity()'s
        // docstring for the 227 mm drift that produces over a sustained idle
        // period, which is exactly what "frozen" is.
        ik.holdGravity();
      }
      ik.setGripper(1 - s.trigger);
    }
    sim.advance(dt);
  }

  frame(timeMs, xrFrame) {
    const now = timeMs / 1000;
    const dt = Math.min(now - (this._last ?? now), 0.1);
    this._last = now;

    this.xr.update(xrFrame, this.renderer.xr.getReferenceSpace(), dt);

    // Either grip held 2s resets the episode, mirroring the 'r' key -- the
    // only way out of a wedged attempt without taking the headset off.
    //
    // Consumed here at render rate rather than in control(): that is where the
    // edge is produced, and the fixed-step control loop below can run zero
    // times in a frame whenever the display is faster than the control rate,
    // which would silently drop presses.
    const resetHeld = this.xr.consumeResetEdge('left') | this.xr.consumeResetEdge('right');
    if (resetHeld) {
      this.xr.pulse('left', 0.4, 40);
      this.xr.pulse('right', 0.4, 40);
      this.resetEpisode();
    }

    // X/A (left/right primary) step the curriculum without leaving the
    // headset, mirroring the desktop 'p'/'n' keys -- those never reach the
    // page once an immersive session has the headset's focus.
    if (this.xr.state.left.primaryEdge) {
      this.xr.pulse('left', 0.4, 40);
      this.loadLevel(Math.max(this.levelIndex - 1, 0));
    }
    if (this.xr.state.right.primaryEdge) {
      this.xr.pulse('right', 0.4, 40);
      this.loadLevel(Math.min(this.levelIndex + 1, CURRICULUM.length - 1));
    }

    // Fixed-rate control, independent of render rate.
    this._ctlAcc += dt;
    const step = 1 / CONTROL_HZ;
    let n = 0;
    while (this._ctlAcc >= step && n < 6) {
      this.control(step);
      this._ctlAcc -= step;
      n++;
      this._physCount++;
    }
    this._physWindow += dt;
    if (this._physWindow > 0.5) {
      this.physHz = this._physCount / this._physWindow;
      this._physCount = 0;
      this._physWindow = 0;
    }

    if (this.status === 'running') this.evaluate(dt);

    this.gfx.sync(this.sim.data);
    const fmt = (side) => {
      const s = this.xr.state[side];
      const rt = this.retarget[side];
      const d = this._ikDbg && this._ikDbg[side];
      // it= is the inner solve's iteration count. Steady 1-2 is healthy; if it
      // sits at the 8 budget the target is unreachable, not the solver slow.
      // rej= counts branch-switch rejections, which should be rare.
      const ikStr = d
        ? `err=${d.posErr.toFixed(4)} leash=${d.leashed ? 1 : 0}`
          + ` it=${d.iters}${d.converged ? '' : '!'} rej=${this._ikRej[side]}`
        : 'no-ik';
      return `${side[0].toUpperCase()} frz=${s.frozen ? 1 : 0} eng=${rt.engaged ? 1 : 0}`
        + ` cnt=${rt.clutchCount} ${ikStr}`;
    };
    this.hud.update({
      levelId: this.level.id,
      levelTitle: this.level.title,
      hint: this.ctx.readout || this.level.hint,
      progress: this.ctx.progress,
      timeLeft: this.tRemaining,
      readout: this.ctx.readout,
      status: this.status,
      gate: {
        passed: this.results.filter(Boolean).length,
        window: this.level.gate.window,
        needed: this.level.gate.needed,
      },
      physHz: this.physHz,
      debug: [fmt('left'), fmt('right')],
    });

    this.renderer.render(this.scene, this.camera);
  }

  evaluate(dt) {
    const { sim, ctx } = this;
    for (const side of ['left', 'right']) {
      ctx.eePos[side] = Array.from(sim.data.site(`${side}_grasp_site`).xpos);
    }
    this.tRemaining -= dt;
    if (this.level.tick) this.level.tick(ctx, dt);

    // Finite-difference EE velocity. Without this the jerk terms in
    // EpisodeMetrics are identically zero and the smoothness quality gates on
    // L1/L5 silently pass everything.
    const eeVel = {};
    for (const side of ['left', 'right']) {
      const prev = this._prevEE?.[side];
      eeVel[side] = prev
        ? [(ctx.eePos[side][0] - prev[0]) / dt,
           (ctx.eePos[side][1] - prev[1]) / dt,
           (ctx.eePos[side][2] - prev[2]) / dt]
        : [0, 0, 0];
    }
    this._prevEE = { left: [...ctx.eePos.left], right: [...ctx.eePos.right] };

    this.metrics.update(dt, {
      eePos: ctx.eePos,
      eeVel,
      contacts: { selfCollision: this.selfCollisionCount() > 0 },
      gripperForce: {
        left: Math.abs(sim.data.actuator_force[this.arms.left.gripperAct]),
        right: Math.abs(sim.data.actuator_force[this.arms.right.gripperAct]),
      },
    });
    this.recorder.tick(dt, this.sampler);

    if (this.level.success(ctx)) return this.endEpisode('success');
    if (this.level.failure && this.level.failure(ctx)) return this.endEpisode('failure');
    if (this.tRemaining <= 0) return this.endEpisode('timeout');
  }

  endEpisode(outcome) {
    this.status = outcome;
    const passedQuality = this.metrics.passesGate(this.level.qualityGate);
    this.recorder.finish(outcome, this.metrics);
    // An episode only counts toward the gate if it was clean. This is the whole
    // point of the curriculum -- a sloppy success is not evidence of skill and
    // its demo is not worth training on.
    this.results.push(outcome === 'success' && passedQuality);
    if (this.results.length > this.level.gate.window) this.results.shift();

    this.xr.pulse('left', outcome === 'success' ? 0.6 : 0.2, 90);
    this.xr.pulse('right', outcome === 'success' ? 0.6 : 0.2, 90);

    const passes = this.results.filter(Boolean).length;
    const unlocked = passes >= this.level.gate.needed
      && this.results.length >= this.level.gate.window;

    this._endTimer = setTimeout(() => {
      this._endTimer = null;
      if (unlocked && this.levelIndex < CURRICULUM.length - 1) {
        this.loadLevel(this.levelIndex + 1);
      } else {
        this.resetEpisode();
      }
    }, 1800);
  }
}

new App().boot().catch((e) => {
  document.getElementById('overlay').textContent = `boot failed: ${e.message}`;
  throw e;
});
