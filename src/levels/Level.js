// Level framework.
//
// A level is a chunk of MJCF injected into the base bimanual scene plus three
// predicates: reset, success, failure. Props are declared as XML so that a
// level author never touches JS geometry code; the model is compiled once per
// level, and per-episode variation happens by writing freejoint qpos, which is
// free.
//
// Levels are ordered as an operator curriculum, not a difficulty ladder for its
// own sake. Each one isolates exactly one new skill, and gating on the previous
// level's *quality* metrics (not just success) is what keeps bad habits from
// compounding into the dataset.

export class Level {
  constructor(spec) {
    Object.assign(this, {
      id: 'unnamed',
      title: '',
      teaches: '',
      timeLimit: 60,
      // Gate: pass `needed` of the last `window` episodes, each under
      // `qualityGate`, before the next level unlocks.
      gate: { window: 5, needed: 4 },
      qualityGate: {},
      props: () => '',
      reset: () => {},
      success: () => false,
      failure: () => false,
      // Shown on the HUD when `failure` fires. A single generic fallback
      // ('objective failed') covers levels that never define `failure`.
      failureReason: 'objective failed',
      hint: '',
    }, spec);
  }
}

/** Splice a level's props into the base scene XML. */
export function composeSceneXML(baseXml, level) {
  const props = level.props() || '';
  const { worldbody = '', asset = '', extra = '' } =
    typeof props === 'string' ? { worldbody: props } : props;
  if (!worldbody.trim() && !asset.trim() && !extra.trim()) return baseXml;

  let xml = baseXml;
  if (asset) xml = injectInto(xml, 'asset', asset);
  if (worldbody) xml = injectInto(xml, 'worldbody', worldbody);
  if (extra) xml = xml.replace(/<\/mujoco>\s*$/, `${extra}\n</mujoco>`);
  return xml;
}

function injectInto(xml, tag, fragment) {
  const close = new RegExp(`</${tag}>`);
  if (close.test(xml)) return xml.replace(close, `${fragment}\n  </${tag}>`);
  return xml.replace(/<\/mujoco>\s*$/, `  <${tag}>\n${fragment}\n  </${tag}>\n</mujoco>`);
}

/** Deterministic PRNG so a failed episode can be replayed exactly. */
export function makeRng(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

/** Write a freejoint body's pose. qpos layout is [x y z qw qx qy qz]. */
export function placeFree(sim, bodyName, pos, quat = [1, 0, 0, 0]) {
  const jid = sim.id('joint', `${bodyName}_free`);
  const adr = sim.model.jnt_qposadr[jid];
  const q = sim.data.qpos;
  q[adr] = pos[0]; q[adr + 1] = pos[1]; q[adr + 2] = pos[2];
  q[adr + 3] = quat[0]; q[adr + 4] = quat[1]; q[adr + 5] = quat[2]; q[adr + 6] = quat[3];
  // Zero velocity too, or a re-placed object inherits the old episode's motion.
  const dadr = sim.model.jnt_dofadr[jid];
  for (let i = 0; i < 6; i++) sim.data.qvel[dadr + i] = 0;
}

export function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/**
 * Running quality metrics. These are the numbers that decide whether an
 * episode is worth keeping — success alone is a weak filter, because a
 * jerky 55-second success teaches a policy the wrong thing.
 */
export class EpisodeMetrics {
  constructor() { this.reset(); }

  reset() {
    this.t = 0;
    this.pathLength = { left: 0, right: 0 };
    this.jerkIntegral = { left: 0, right: 0 };
    this.clutchCount = { left: 0, right: 0 };
    this.selfCollisionTime = 0;
    this.gripperOverforceTime = 0;
    this.idleTime = 0;
    this._prev = {};
  }

  update(dt, { eePos, eeVel, contacts, gripperForce }) {
    this.t += dt;
    for (const side of ['left', 'right']) {
      const p = eePos[side];
      const prev = this._prev[side];
      if (prev) {
        const d = dist(p, prev.pos);
        this.pathLength[side] += d;
        if (d < 1e-4) this.idleTime += dt / 2;
        const acc = [
          (eeVel[side][0] - prev.vel[0]) / dt,
          (eeVel[side][1] - prev.vel[1]) / dt,
          (eeVel[side][2] - prev.vel[2]) / dt,
        ];
        if (prev.acc) {
          this.jerkIntegral[side] += Math.hypot(
            (acc[0] - prev.acc[0]) / dt,
            (acc[1] - prev.acc[1]) / dt,
            (acc[2] - prev.acc[2]) / dt,
          ) * dt;
        }
        this._prev[side] = { pos: [...p], vel: [...eeVel[side]], acc };
      } else {
        this._prev[side] = { pos: [...p], vel: [...eeVel[side]], acc: null };
      }
    }
    if (contacts && contacts.selfCollision) this.selfCollisionTime += dt;
    if (gripperForce && (gripperForce.left > 40 || gripperForce.right > 40)) {
      this.gripperOverforceTime += dt;
    }
  }

  /** Straight-line-optimality proxy. 1.0 = perfectly direct. */
  efficiency(side, netDisplacement) {
    return this.pathLength[side] > 1e-6
      ? Math.min(netDisplacement / this.pathLength[side], 1) : 0;
  }

  /** Dotted-path lookup, e.g. get('jerkIntegral.left'). */
  get(key) {
    return key.includes('.')
      ? key.split('.').reduce((o, k) => o?.[k], this)
      : this[key];
  }

  passesGate(gate) {
    for (const [key, limit] of Object.entries(gate)) {
      const v = this.get(key);
      if (typeof v === 'number' && v > limit) return false;
    }
    return true;
  }

  /** Which gate entries are currently over their limit, for HUD/failure text. */
  failingEntries(gate) {
    const out = [];
    for (const [key, limit] of Object.entries(gate)) {
      const v = this.get(key);
      if (typeof v === 'number' && v > limit) out.push({ key, value: v, limit });
    }
    return out;
  }
}

const METRIC_LABELS = {
  'jerkIntegral.left': 'left jerk',
  'jerkIntegral.right': 'right jerk',
  gripperOverforceTime: 'gripper overforce',
  selfCollisionTime: 'self-collision',
};

export function metricLabel(key) {
  return METRIC_LABELS[key] || key;
}

/** Time-accumulator metrics read as seconds; everything else as a raw count. */
export function formatMetricValue(key, v) {
  return key.toLowerCase().endsWith('time') ? `${v.toFixed(2)}s` : v.toFixed(0);
}
