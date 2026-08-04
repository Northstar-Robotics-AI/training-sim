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

const MESHES = ['base', 'link1', 'link2', 'link3', 'link4', 'link5',
  'gripper', 'tip_left', 'tip_right'].map((m) => `/assets/yam/meshes/${m}.stl`);
const BASE_XML_URL = '/assets/yam/bimanual_yam.xml';

// Home posture: elbow up, wrist level. Doubles as the IK nullspace bias.
const HOME_Q = [0, 0.9, 1.2, 0, -0.5, 0];

// Control runs on its own fixed clock. 100 Hz is well above what an operator
// can perceive and well below what the WASM build struggles with.
const CONTROL_HZ = 100;

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
    for (let i = 0; i < 2; i++) this.scene.add(this.renderer.xr.getController(i));

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
    this.levelIndex = index;
    this.level = CURRICULUM[index];
    this.results = [];

    // Compiling per level rather than hiding unused props keeps each level's
    // contact set minimal, which is the difference between 90 Hz and 45 Hz
    // once a level has a dozen loose objects in it.
    const xml = composeSceneXML(this.baseXml, this.level);
    if (this.gfx) this.scene.remove(this.gfx.root);
    this.sim = await new Sim().init({
      xmlUrl: URL.createObjectURL(new Blob([xml], { type: 'text/xml' })),
      meshUrls: MESHES,
    });
    this.gfx = buildSceneGraph(this.sim, { hiddenGroups: [3] });
    this.scene.add(this.gfx.root);

    this.arms = {
      left: new ArmIK(this.sim, { prefix: 'left_', siteName: 'left_grasp_site', homeQ: HOME_Q }),
      right: new ArmIK(this.sim, { prefix: 'right_', siteName: 'right_grasp_site', homeQ: HOME_Q }),
    };
    this.retarget = {
      left: new ClutchRetargeter(this.sim.mj, { posScale: 0.6 }),
      right: new ClutchRetargeter(this.sim.mj, { posScale: 0.6 }),
    };

    if (!this.hud) this.hud = new HUD(this.scene);
    this.sampler = makeSampler(this.sim, this.arms, this.xr, this.level);
    this.resetEpisode();
  }

  resetEpisode() {
    const sim = this.sim;
    sim.reset();
    for (const side of ['left', 'right']) {
      const ik = this.arms[side];
      for (let i = 0; i < 6; i++) sim.data.qpos[ik.qposAdr[i]] = HOME_Q[i];
      ik.qTarget = Float64Array.from(HOME_Q);
      for (let i = 0; i < 6; i++) sim.data.ctrl[ik.actIds[i]] = HOME_Q[i];
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
    this.level.reset(this.ctx);
    sim.mj.mj_forward(sim.model, sim.data);

    this.metrics.reset();
    this._prevEE = null;
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

      if (!s.valid) { ik.step(...currentTarget(sim, side)); continue; }

      const clutchEdge = this.xr.consumeClutchEdge(side);
      if (clutchEdge === 'down') {
        ik.syncToCurrent();
        rt.engage(xrPosToMj(s.pos), xrQuatToMj(sim.mj, s.quat), sim.sitePose(`${side}_grasp_site`));
        this.xr.pulse(side, 0.3, 25);
        this.metrics.clutchCount[side]++;
        this.recorder.event('clutch_down', { side });
      } else if (clutchEdge === 'up') {
        rt.release();
        this.recorder.event('clutch_up', { side });
      }

      const target = rt.target(xrPosToMj(s.pos), xrQuatToMj(sim.mj, s.quat));
      if (target) {
        const r = ik.step(target.pos, target.quat);
        this._ikDbg = this._ikDbg || {};
        this._ikDbg[side] = { ...r, tgt: target.pos };
        // Buzz only when the leash engages -- the arm is genuinely blocked by
        // contact or a joint limit. `damped` fires during any normal fast slew,
        // so buzzing on that would vibrate continuously and mean nothing.
        if (r.leashed) this.xr.pulse(side, 0.15, 15);
      }
      ik.setGripper(1 - s.trigger);
    }
    sim.advance(dt);
  }

  frame(timeMs, xrFrame) {
    const now = timeMs / 1000;
    const dt = Math.min(now - (this._last ?? now), 0.1);
    this._last = now;

    this.xr.update(xrFrame, this.renderer.xr.getReferenceSpace());

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
      const ikStr = d ? `err=${d.posErr.toFixed(4)} leash=${d.leashed ? 1 : 0}` : 'no-ik';
      return `${side[0].toUpperCase()} c=${s.clutch ? 1 : 0} eng=${rt.engaged ? 1 : 0}`
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

    setTimeout(() => {
      if (unlocked && this.levelIndex < CURRICULUM.length - 1) {
        this.loadLevel(this.levelIndex + 1);
      } else {
        this.resetEpisode();
      }
    }, 1800);
  }
}

function currentTarget(sim, side) {
  const p = sim.sitePose(`${side}_grasp_site`);
  return [p.pos, p.quat];
}

new App().boot().catch((e) => {
  document.getElementById('overlay').textContent = `boot failed: ${e.message}`;
  throw e;
});
