// MuJoCo WASM lifecycle: load assets into MEMFS, compile the model, step it.
//
// The WASM module has no filesystem of its own, so every mesh referenced by the
// MJCF must be written into Emscripten's in-memory FS *before* from_xml_path.
// Paths inside MEMFS must match `meshdir` in the XML exactly.

import load_mujoco from '@mujoco/mujoco';

export class Sim {
  constructor() {
    this.mj = null;
    this.model = null;
    this.data = null;
    this.timestep = 0.002;
    this._accum = 0;
    this._names = {};
  }

  async init({ xmlUrl, meshUrls, workDir = '/work' }) {
    this.mj = await load_mujoco();
    const FS = this.mj.FS;
    try { FS.mkdir(workDir); } catch (e) { /* already exists */ }
    FS.mount(this.mj.MEMFS, { root: '.' }, workDir);
    FS.mkdir(`${workDir}/meshes`);

    // Meshes are binary; XML is text. Fetch in parallel — on a Quest over wifi
    // the 5 MB of STL is the dominant part of cold start.
    const [xml, meshes] = await Promise.all([
      fetch(xmlUrl).then((r) => r.text()),
      Promise.all(meshUrls.map(async (u) => [
        u.split('/').pop(),
        new Uint8Array(await (await fetch(u)).arrayBuffer()),
      ])),
    ]);
    for (const [name, bytes] of meshes) {
      FS.writeFile(`${workDir}/meshes/${name}`, bytes);
    }
    FS.writeFile(`${workDir}/scene.xml`, xml);

    this.model = this.mj.MjModel.from_xml_path(`${workDir}/scene.xml`);
    this.data = new this.mj.MjData(this.model);
    this.timestep = this.model.opt.timestep;
    this.mj.mj_forward(this.model, this.data);
    this._cacheNames();
    return this;
  }

  _cacheNames() {
    const { mj, model } = this;
    const kinds = {
      joint: mj.mjtObj.mjOBJ_JOINT,
      site: mj.mjtObj.mjOBJ_SITE,
      body: mj.mjtObj.mjOBJ_BODY,
      actuator: mj.mjtObj.mjOBJ_ACTUATOR,
      geom: mj.mjtObj.mjOBJ_GEOM,
    };
    for (const [kind, obj] of Object.entries(kinds)) {
      this._names[kind] = (name) => mj.mj_name2id(model, obj.value, name);
    }
  }

  id(kind, name) {
    const i = this._names[kind](name);
    if (i < 0) throw new Error(`no ${kind} named "${name}" in model`);
    return i;
  }

  /**
   * Advance physics by `dt` seconds of wall time using a fixed-step
   * accumulator. Render framerate and physics rate must stay decoupled or the
   * contact solver becomes framerate-dependent and demos stop being reproducible.
   * maxSteps caps the catch-up so a GC pause can't spiral into a freeze.
   */
  advance(dt, maxSteps = 12) {
    this._accum = Math.min(this._accum + dt, this.timestep * maxSteps);
    let n = 0;
    while (this._accum >= this.timestep) {
      this.mj.mj_step(this.model, this.data);
      this._accum -= this.timestep;
      n++;
    }
    return n;
  }

  reset() {
    this.mj.mj_resetData(this.model, this.data);
    this.mj.mj_forward(this.model, this.data);
    this._accum = 0;
  }

  sitePose(name) {
    const s = this.data.site(name);
    const q = new Float64Array(4);
    this.mj.mju_mat2Quat(q, Array.from(s.xmat));
    return { pos: Float64Array.from(s.xpos), quat: q };
  }

  bodyPose(name) {
    const b = this.data.body(name);
    return { pos: Float64Array.from(b.xpos), quat: Float64Array.from(b.xquat) };
  }
}
