// Episode recording.
//
// Logs at a fixed 30 Hz regardless of render framerate. This matters: if you
// log per rendered frame you get 72 Hz on a Quest 3, 90 on a Quest Pro and
// whatever the desktop happens to hit, and then your dataset has three
// different implicit control frequencies baked into the action deltas.
//
// The schema mirrors what LeRobot expects for a bimanual arm so that
// `tools/to_lerobot.py` is a rename rather than a rewrite:
//   observation.state   14  [6 joints + gripper] x 2 arms
//   action              14  commanded joint targets (what the servo was told)
// Everything else goes in a sidecar so it doesn't pollute the training tensors
// but is still there when you need to explain why an episode looks strange.

const HZ = 30;

export class Recorder {
  constructor({ hz = HZ } = {}) {
    this.period = 1 / hz;
    this.hz = hz;
    this._acc = 0;
    this.episodes = [];
    this.current = null;
  }

  start(meta) {
    this.current = {
      meta: { ...meta, hz: this.hz, started: new Date().toISOString() },
      frames: [],
      events: [],
    };
    this._acc = 0;
  }

  event(name, payload = {}) {
    if (this.current) {
      this.current.events.push({ t: this.current.frames.length / this.hz, name, ...payload });
    }
  }

  /** @param {number} dt seconds since last render frame */
  tick(dt, sample) {
    if (!this.current) return;
    this._acc += dt;
    while (this._acc >= this.period) {
      this.current.frames.push(sample());
      this._acc -= this.period;
    }
  }

  /**
   * @param {'success'|'failure'|'timeout'|'aborted'} outcome
   * @param {object} metrics EpisodeMetrics snapshot
   */
  finish(outcome, metrics) {
    if (!this.current) return null;
    this.current.meta.outcome = outcome;
    this.current.meta.durationS = this.current.frames.length / this.hz;
    this.current.meta.metrics = JSON.parse(JSON.stringify(metrics));
    const ep = this.current;
    this.episodes.push(ep);
    this.current = null;
    return ep;
  }

  discardLast() { return this.episodes.pop(); }

  /** JSONL, one episode per line. Small enough to keep in memory for a
   *  session; flush to the server every N episodes if you're crowdsourcing. */
  toJSONL() {
    return this.episodes.map((e) => JSON.stringify(e)).join('\n');
  }

  download(filename = `yam-xr-${Date.now()}.jsonl`) {
    const blob = new Blob([this.toJSONL()], { type: 'application/x-ndjson' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async upload(url) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-ndjson' },
      body: this.toJSONL(),
    });
    if (!res.ok) throw new Error(`upload failed: ${res.status}`);
    return res.json().catch(() => ({}));
  }
}

/**
 * Build the per-frame sample closure. Kept separate from Recorder so the
 * schema lives in one readable place.
 */
export function makeSampler(sim, arms, xrInput, level) {
  const qposAdr = {};
  const gripAdr = {};
  for (const side of ['left', 'right']) {
    qposAdr[side] = arms[side].qposAdr;
    gripAdr[side] = sim.model.jnt_qposadr[sim.id('joint', `${side}_joint7`)];
  }

  return () => {
    const state = [];
    const action = [];
    for (const side of ['left', 'right']) {
      for (const adr of qposAdr[side]) state.push(sim.data.qpos[adr]);
      state.push(sim.data.qpos[gripAdr[side]]);
      for (const q of arms[side].qTarget) action.push(q);
      action.push(sim.data.ctrl[arms[side].gripperAct]);
    }

    const ee = {};
    for (const side of ['left', 'right']) {
      const p = sim.sitePose(`${side}_grasp_site`);
      ee[side] = { pos: Array.from(p.pos), quat: Array.from(p.quat) };
    }

    const ctl = {};
    for (const side of ['left', 'right']) {
      const s = xrInput.state[side];
      ctl[side] = {
        valid: s.valid,
        pos: [s.pos.x, s.pos.y, s.pos.z],
        quat: [s.quat.x, s.quat.y, s.quat.z, s.quat.w],
        trigger: s.trigger,
        frozen: s.frozen,
      };
    }

    const head = xrInput.state.head;
    return {
      'observation.state': state,
      action,
      ee,
      // Object poses are what let you re-simulate an episode later without the
      // original operator. Cheap to store, impossible to recover afterwards.
      objects: level.trackedBodies
        ? Object.fromEntries(level.trackedBodies.map((b) => {
          const p = sim.bodyPose(b);
          return [b, [...p.pos, ...p.quat]];
        }))
        : {},
      xr: {
        controllers: ctl,
        head: head.valid
          ? { pos: [head.pos.x, head.pos.y, head.pos.z],
              quat: [head.quat.x, head.quat.y, head.quat.z, head.quat.w] }
          : null,
      },
    };
  };
}
