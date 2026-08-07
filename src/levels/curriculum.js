// The curriculum.
//
// Ordering principle: each level adds exactly one new demand, and nothing is
// introduced before the skill it depends on is gated. Novice teleoperators fail
// in a predictable sequence -- they lose the workspace, then they lose
// orientation, then they mistime the grasp, then they can't coordinate two arms
// under a shared constraint -- so the levels attack those in that order.
//
// Levels 0-2 are single-arm on purpose. Handing someone two arms before they
// can reliably drive one produces operators who fixate on one hand and let the
// other drift, which is a very hard habit to unlearn later.

import { Level, placeFree, dist } from './Level.js';

const TABLE_Z = 0.74;      // table top surface, world Z
const L = { x: 0.30, y: 0.22 };   // nominal left workspace centre
const R = { x: 0.30, y: -0.22 };

// left_grasp_site's orientation at REST_Q (main.js), i.e. what the gripper
// is actually pointing when an episode starts -- measured via FK. Now that
// REST_Q is the YAM's true zero position, this is a ~91 deg rotation with a
// slight tilt off pure Y, not the clean 90 deg-about-Y the old folded rest
// gave. L1's reset() perturbs yaw/pitch around *this*, not around world
// identity: identity is far from anything the arm can reach, so targets
// built as raw eulerToQuat(0, pitch, yaw) from identity landed way off rest
// and were routinely unreachable (the ghost gripper made this obvious -- it
// rendered backwards, since matching it would have required an impossible
// wrist twist). Recompute via FK (see tools/ik_smoke.mjs for the loadSim
// pattern) if REST_Q ever changes again.
const LEFT_GRASP_REST_QUAT = [0.699168143687048, 0.1056652698645116, 0.6991657381646165, -0.1056694287596678];

const freeBody = (name, geom, pos) => `
    <body name="${name}" pos="${pos.join(' ')}">
      <freejoint name="${name}_free"/>
      ${geom}
    </body>`;

const box = (size, rgba, mass = 0.15) =>
  `<geom type="box" size="${size}" rgba="${rgba}" mass="${mass}" friction="1.2 0.02 0.001"
             solref="0.008 1" condim="4"/>`;

// solref stiffer than default: soft contacts make a light cube feel like jelly
// through a teleop loop and operators over-correct.

// A round bore, approximated as a ring of flat box segments (MuJoCo has no
// boolean subtraction, so an actual cylindrical cavity isn't a primitive).
// Each segment's inner face sits exactly on the innerR circle at its own
// center angle -- the true constriction a peg has to pass through -- and
// bows slightly outward at the segment edges, so the polygon is always at
// or outside innerR and never pinches the peg tighter than intended.
const boreRing = (innerR, outerR, zCenter, halfHeight, rgba, n = 16) => {
  const segs = [];
  const radialHalf = (outerR - innerR) / 2;
  const midR = innerR + radialHalf;
  const tangentialHalf = outerR * Math.tan(Math.PI / n) * 1.05;
  for (let i = 0; i < n; i++) {
    const angle = (2 * Math.PI * i) / n;
    const x = midR * Math.cos(angle);
    const y = midR * Math.sin(angle);
    const qw = Math.cos(angle / 2);
    const qz = Math.sin(angle / 2);
    segs.push(`<geom type="box" size="${radialHalf.toFixed(5)} ${tangentialHalf.toFixed(5)} ${halfHeight}" `
      + `pos="${x.toFixed(5)} ${y.toFixed(5)} ${zCenter}" quat="${qw.toFixed(6)} 0 0 ${qz.toFixed(6)}" rgba="${rgba}"/>`);
  }
  return segs.join('\n      ');
};

export const CURRICULUM = [

  new Level({
    id: 'L0_calibration',
    title: 'Find your workspace',
    teaches: 'Clutch, motion scale, reach envelope. No objects, no failure.',
    timeLimit: 120,
    gate: { window: 1, needed: 1 },
    hint: 'Hold GRIP to move. Release, recenter your hand, grip again to ratchet further.',
    props: () => ({
      // Paired geom per site, same reason as L1's target_frame: sceneBuilder
      // only renders <geom>, so a bare <site> is invisible even though its
      // pose is exactly what tick() below reads.
      worldbody: [0, 1, 2, 3, 4, 5].map((i) => {
        const side = i < 3 ? L : R;
        const dx = [-0.12, 0.0, 0.14][i % 3];
        const dz = [0.06, 0.22, 0.34][i % 3];
        const pos = `${side.x + dx} ${side.y} ${TABLE_Z + dz}`;
        return `    <site name="wp${i}" pos="${pos}" size="0.035" rgba="0.2 0.8 1 0.8"/>
    <geom name="wp${i}_vis" type="sphere" pos="${pos}" size="0.035" rgba="0.2 0.8 1 0.8"
          contype="0" conaffinity="0"/>`;
      }).join('\n'),
    }),
    reset(ctx) {
      ctx.state.hit = new Set();
      for (let i = 0; i < 6; i++) ctx.setGeomOpacity(`wp${i}_vis`, 0.8);
    },
    tick(ctx) {
      for (let i = 0; i < 6; i++) {
        const wp = ctx.sim.data.site(`wp${i}`).xpos;
        const side = i < 3 ? 'left' : 'right';
        if (!ctx.state.hit.has(i) && dist(ctx.eePos[side], wp) < 0.05) {
          ctx.state.hit.add(i);
          ctx.setGeomOpacity(`wp${i}_vis`, 0.15);
        }
      }
      ctx.progress = ctx.state.hit.size / 6;
    },
    success: (ctx) => ctx.state.hit.size === 6,
  }),

  new Level({
    id: 'L1_pose_match',
    title: 'Match the frame',
    teaches: 'Orientation control. Position alone is easy; wrist alignment is what novices lose.',
    timeLimit: 45,
    gate: { window: 5, needed: 4 },
    qualityGate: { 'jerkIntegral.left': 900 },
    hint: 'Move your gripper into the ghost gripper. Position AND tilt must match. Hold it for one second.',
    props: () => ({
      // A translucent ghost of the *actual* gripper mesh (body + both
      // fingertips), not an abstract marker -- a thin box gave no sense of
      // scale or which way the jaw opens. The three geoms below reuse the
      // real gripper/tip_left/tip_right meshes (already in <asset> for the
      // arms themselves), placed so the ghost exactly overlaps the real
      // gripper once target_site's pose matches left_grasp_site's.
      //
      // The pos/quat below are NOT the raw world-relative offset -- MuJoCo
      // composes a mesh geom's declared pos/quat with the mesh asset's own
      // internal reference frame (its centering offset from STL loading) to
      // get the geom's actual compiled transform. Reusing the same offset
      // that FK reports for the real gripper parts (relative to
      // left_grasp_site) as a naive pos/quat here double-applies that
      // per-mesh offset and misplaces the ghost by several cm. The values
      // here already have that composition solved out (worked out once via
      // FK at the gripper's rest-open state, verified against the real
      // meshes' compiled geom_pos/geom_quat until the two overlapped to
      // float noise) so they land correctly; don't hand-derive replacements
      // from the site offsets alone.
      // contype/conaffinity 0: visual only, must not collide or get grasped.
      worldbody: `
    <body name="target_frame" pos="${L.x} ${L.y} ${TABLE_Z + 0.18}">
      <site name="target_site" size="0.006" rgba="0 0 0 0"/>
      <geom type="mesh" mesh="gripper" rgba="1 0.55 0.1 0.5" contype="0" conaffinity="0"
            pos="-0.0140000 0.0463995 -0.2078000" quat="0 1 0 0"/>
      <geom type="mesh" mesh="tip_left" rgba="1 0.55 0.1 0.5" contype="0" conaffinity="0"
            pos="-0.0139060 0.0938995 -0.2099231" quat="0 1 0 0"/>
      <geom type="mesh" mesh="tip_right" rgba="1 0.55 0.1 0.5" contype="0" conaffinity="0"
            pos="-0.0140951 -0.0011001 -0.2099231" quat="0 1 0 0"/>
    </body>`,
    }),
    reset(ctx) {
      const r = ctx.rng;
      ctx.state.holdT = 0;
      // Range picked by actually running the arm's IK solver (not just an
      // angle heuristic) against thousands of sampled targets: the previous
      // range -- yaw +-0.7, pitch +-0.5, position +-0.08, height 12-28 cm
      // above the table -- put ~20% of episodes outside what left_grasp_site
      // can reach, even composed around the correct rest orientation (see
      // LEFT_GRASP_REST_QUAT above). Position mattered as much as
      // orientation: low height (close to the 12 cm floor) alone was
      // unreachable at some x/y even with zero tilt. This narrower range
      // (yaw +-0.25, pitch +-0.15, position +-0.05, height 18-30 cm) measured
      // <0.5% unreachable across 5 seeds x 80 episodes -- tighten further
      // before widening it back up.
      const yaw = (r() - 0.5) * 0.5;
      const pitch = (r() - 0.5) * 0.3;
      ctx.setBodyPose('target_frame',
        [L.x + (r() - 0.5) * 0.10, L.y + (r() - 0.5) * 0.10, TABLE_Z + 0.18 + r() * 0.12],
        quatMul(LEFT_GRASP_REST_QUAT, eulerToQuat(0, pitch, yaw)));
    },
    tick(ctx, dt) {
      const t = ctx.sim.sitePose('target_site');
      const e = ctx.sim.sitePose('left_grasp_site');
      const dp = dist(t.pos, e.pos);
      const dr = quatAngle(t.quat, e.quat);
      ctx.state.holdT = (dp < 0.025 && dr < 0.20) ? ctx.state.holdT + dt : 0;
      ctx.progress = Math.min(ctx.state.holdT / 1.0, 1);
      ctx.readout = `dp ${(dp * 1000).toFixed(0)}mm  dr ${(dr * 57.3).toFixed(0)}deg`;
    },
    success: (ctx) => ctx.state.holdT >= 1.0,
  }),

  new Level({
    id: 'L2_pick_place',
    title: 'Pick and place',
    teaches: 'Grasp timing and approach direction. First level where contact matters.',
    timeLimit: 60,
    gate: { window: 6, needed: 5 },
    qualityGate: { gripperOverforceTime: 3.0, selfCollisionTime: 0.5 },
    hint: 'Approach from above with the gripper open. Close slowly — squeezing hard is scored against you.',
    props: () => ({
      worldbody: `${freeBody('cube', box('0.022 0.022 0.022', '0.9 0.3 0.25 1'),
        [L.x, L.y, TABLE_Z + 0.03])}
    <body name="bin" pos="${L.x + 0.18} ${L.y - 0.16} ${TABLE_Z}">
      <geom type="box" size="0.07 0.07 0.004" pos="0 0 0.004" rgba="0.2 0.6 0.3 1"/>
      <geom type="box" size="0.004 0.07 0.03" pos="0.07 0 0.03" rgba="0.2 0.6 0.3 1"/>
      <geom type="box" size="0.004 0.07 0.03" pos="-0.07 0 0.03" rgba="0.2 0.6 0.3 1"/>
      <geom type="box" size="0.07 0.004 0.03" pos="0 0.07 0.03" rgba="0.2 0.6 0.3 1"/>
      <geom type="box" size="0.07 0.004 0.03" pos="0 -0.07 0.03" rgba="0.2 0.6 0.3 1"/>
      <site name="bin_site" pos="0 0 0.02" size="0.01" rgba="0 0 0 0"/>
    </body>`,
    }),
    reset(ctx) {
      const r = ctx.rng;
      placeFree(ctx.sim, 'cube',
        [L.x + (r() - 0.5) * 0.16, L.y + (r() - 0.5) * 0.14, TABLE_Z + 0.03],
        eulerToQuat(0, 0, r() * Math.PI));
      ctx.state.settled = 0;
    },
    tick(ctx, dt) {
      const c = ctx.sim.bodyPose('cube').pos;
      const b = ctx.sim.sitePose('bin_site').pos;
      const inBin = Math.abs(c[0] - b[0]) < 0.06 && Math.abs(c[1] - b[1]) < 0.06
        && c[2] < b[2] + 0.05 && c[2] > b[2] - 0.02;
      ctx.state.settled = inBin ? ctx.state.settled + dt : 0;
      ctx.progress = Math.min(ctx.state.settled / 0.5, 1);
    },
    success: (ctx) => ctx.state.settled >= 0.5,
    failure: (ctx) => ctx.sim.bodyPose('cube').pos[2] < TABLE_Z - 0.15,
  }),

  new Level({
    id: 'L3_handover',
    title: 'Handover',
    teaches: 'Two arms, sequential. The receiving gripper must close before the giving one opens.',
    timeLimit: 75,
    gate: { window: 6, needed: 4 },
    qualityGate: { selfCollisionTime: 1.0 },
    hint: 'The bin is out of the left arm\'s reach. Pass the cube in mid-air near the centre.',
    props: () => ({
      worldbody: `${freeBody('cube', box('0.022 0.022 0.022', '0.85 0.7 0.2 1'),
        [L.x, L.y + 0.14, TABLE_Z + 0.03])}
    <body name="bin" pos="${R.x + 0.05} ${R.y - 0.24} ${TABLE_Z}">
      <geom type="box" size="0.06 0.06 0.004" pos="0 0 0.004" rgba="0.2 0.6 0.3 1"/>
      <geom type="box" size="0.004 0.06 0.03" pos="0.06 0 0.03" rgba="0.2 0.6 0.3 1"/>
      <geom type="box" size="0.004 0.06 0.03" pos="-0.06 0 0.03" rgba="0.2 0.6 0.3 1"/>
      <site name="bin_site" pos="0 0 0.02" size="0.01" rgba="0 0 0 0"/>
    </body>`,
    }),
    reset(ctx) {
      const r = ctx.rng;
      placeFree(ctx.sim, 'cube',
        [L.x + (r() - 0.5) * 0.1, L.y + 0.12 + (r() - 0.5) * 0.08, TABLE_Z + 0.03]);
      ctx.state.settled = 0;
      ctx.state.everHeldRight = false;
    },
    tick(ctx, dt) {
      const c = ctx.sim.bodyPose('cube').pos;
      if (dist(c, ctx.eePos.right) < 0.07 && c[2] > TABLE_Z + 0.10) {
        ctx.state.everHeldRight = true;
      }
      const b = ctx.sim.sitePose('bin_site').pos;
      const inBin = Math.abs(c[0] - b[0]) < 0.05 && Math.abs(c[1] - b[1]) < 0.05
        && c[2] < b[2] + 0.05;
      ctx.state.settled = inBin ? ctx.state.settled + dt : 0;
      ctx.progress = ctx.state.everHeldRight ? 0.5 + Math.min(ctx.state.settled, 0.5) : 0.2;
    },
    // Requiring everHeldRight stops the operator solving it by throwing.
    success: (ctx) => ctx.state.settled >= 0.5 && ctx.state.everHeldRight,
    failure: (ctx) => ctx.sim.bodyPose('cube').pos[2] < TABLE_Z - 0.15,
  }),

  new Level({
    id: 'L4_bimanual_lift',
    title: 'Lift together',
    teaches: 'Simultaneous bimanual control under a closed kinematic chain.',
    timeLimit: 75,
    gate: { window: 6, needed: 4 },
    qualityGate: { gripperOverforceTime: 4.0 },
    hint: 'The tray is too wide for one gripper. Pinch both ends and raise at the same rate — '
        + 'if one hand leads, the tray tips.',
    props: () => ({
      worldbody: `
    <body name="tray" pos="${(L.x + R.x) / 2} 0 ${TABLE_Z + 0.02}">
      <freejoint name="tray_free"/>
      <geom type="box" size="0.16 0.05 0.006" rgba="0.75 0.75 0.8 1" mass="0.5"
            friction="1.4 0.02 0.001" condim="4" solref="0.008 1"/>
      <geom type="box" size="0.012 0.05 0.025" pos="0.16 0 0.025" rgba="0.5 0.5 0.55 1" mass="0.05"/>
      <geom type="box" size="0.012 0.05 0.025" pos="-0.16 0 0.025" rgba="0.5 0.5 0.55 1" mass="0.05"/>
      <site name="tray_site" pos="0 0 0.01" size="0.01" rgba="0 0 0 0"/>
    </body>
    ${freeBody('ball', '<geom type="sphere" size="0.022" rgba="0.9 0.4 0.6 1" mass="0.05" '
      + 'friction="0.6 0.01 0.001" condim="4"/>',
      [(L.x + R.x) / 2, 0, TABLE_Z + 0.06])}
    <geom type="box" size="0.07 0.20 0.004" pos="${(L.x + R.x) / 2} 0 ${TABLE_Z + 0.30}"
          rgba="0.2 0.8 1 0.3" contype="0" conaffinity="0"/>
    <site name="goal_shelf" pos="${(L.x + R.x) / 2} 0 ${TABLE_Z + 0.30}"
          size="0.07 0.20 0.004" type="box" rgba="0 0 0 0"/>`,
    }),
    reset(ctx) {
      const r = ctx.rng;
      const cx = (L.x + R.x) / 2 + (r() - 0.5) * 0.06;
      const cy = (r() - 0.5) * 0.05;
      // Tray's long axis is local X; rotate 90 deg about Z so the grasp ends
      // line up with the left/right arms (which sit side-by-side along Y),
      // instead of pointing one end at the base and one away from it.
      placeFree(ctx.sim, 'tray', [cx, cy, TABLE_Z + 0.03], [0.7071068, 0, 0, 0.7071068]);
      placeFree(ctx.sim, 'ball', [cx, cy, TABLE_Z + 0.08]);
      ctx.state.holdT = 0;
    },
    tick(ctx, dt) {
      const tray = ctx.sim.bodyPose('tray');
      const ball = ctx.sim.bodyPose('ball').pos;
      const goal = ctx.sim.sitePose('goal_shelf').pos;
      const tilt = quatAngle(tray.quat, [0.7071068, 0, 0, 0.7071068]);
      const up = tray.pos[2] > goal[2] - 0.04;
      const ballOn = ball[2] > tray.pos[2] && dist(ball, tray.pos) < 0.14;
      ctx.state.holdT = (up && ballOn && tilt < 0.35) ? ctx.state.holdT + dt : 0;
      ctx.progress = Math.min(ctx.state.holdT / 1.5, 1);
      ctx.readout = `tilt ${(tilt * 57.3).toFixed(0)}deg`;
    },
    success: (ctx) => ctx.state.holdT >= 1.5,
    failure: (ctx) => ctx.sim.bodyPose('ball').pos[2] < TABLE_Z - 0.05,
  }),

  new Level({
    id: 'L5_insertion',
    title: 'Peg in hole',
    teaches: 'Precision under contact. 3 mm clearance — beyond visual servoing, you need compliance.',
    timeLimit: 90,
    gate: { window: 8, needed: 5 },
    qualityGate: { 'jerkIntegral.left': 1200, gripperOverforceTime: 5.0 },
    hint: 'Get the peg above the hole, then descend slowly. Fighting a jam makes it worse — '
        + 'lift 5 mm, re-align, try again.',
    props: () => ({
      worldbody: `${freeBody('peg',
        '<geom type="cylinder" size="0.011 0.05" rgba="0.85 0.45 0.15 1" mass="0.08" '
        + 'friction="0.9 0.02 0.001" condim="4" solref="0.006 1"/>',
        [L.x, L.y, TABLE_Z + 0.05])}
    <body name="socket" pos="${R.x} ${R.y} ${TABLE_Z}">
      <geom type="box" size="0.05 0.05 0.005" pos="0 0 0.005" rgba="0.35 0.38 0.45 1"/>
      ${boreRing(0.0125, 0.045, 0.04, 0.03, '0.3 0.32 0.4 1')}
      <site name="hole_site" pos="0 0 0.035" size="0.008" rgba="0.1 0.9 0.4 0.5"/>
    </body>`,
    }),
    reset(ctx) {
      const r = ctx.rng;
      placeFree(ctx.sim, 'peg',
        [L.x + (r() - 0.5) * 0.12, L.y + (r() - 0.5) * 0.12, TABLE_Z + 0.05],
        eulerToQuat(1.5708, 0, r() * Math.PI));
      ctx.state.holdT = 0;
    },
    tick(ctx, dt) {
      const p = ctx.sim.bodyPose('peg');
      const h = ctx.sim.sitePose('hole_site').pos;
      const lateral = Math.hypot(p.pos[0] - h[0], p.pos[1] - h[1]);
      const seated = lateral < 0.012 && p.pos[2] < h[2] + 0.035
        && quatAngle(p.quat, [1, 0, 0, 0]) < 0.25;
      ctx.state.holdT = seated ? ctx.state.holdT + dt : 0;
      ctx.progress = Math.min(ctx.state.holdT / 1.0, 1);
      ctx.readout = `offset ${(lateral * 1000).toFixed(0)}mm`;
    },
    success: (ctx) => ctx.state.holdT >= 1.0,
    failure: (ctx) => ctx.sim.bodyPose('peg').pos[2] < TABLE_Z - 0.15,
  }),

  new Level({
    id: 'L6_stabilize_and_act',
    title: 'Hold and pour',
    teaches: 'Asymmetric roles: one arm is a fixture, the other is the tool. '
           + 'Most real bimanual manipulation is this, not symmetric lifting.',
    timeLimit: 90,
    gate: { window: 6, needed: 4 },
    hint: 'Left arm holds the cup steady. Right arm tips the scoop over it. '
        + 'If the cup moves, the pour misses.',
    props: () => ({
      worldbody: `
    <body name="cup" pos="${L.x} ${L.y} ${TABLE_Z + 0.04}">
      <freejoint name="cup_free"/>
      <geom type="cylinder" size="0.035 0.005" rgba="0.9 0.9 0.92 1" mass="0.06"/>
      <geom type="cylinder" size="0.035 0.05" pos="0 0 0.05" rgba="0.9 0.9 0.92 0.35"
            mass="0.02" contype="1" conaffinity="1"/>
      <site name="cup_site" pos="0 0 0.01" size="0.01" rgba="0 0 0 0"/>
    </body>
    ${freeBody('scoop',
      '<geom type="box" size="0.035 0.025 0.004" rgba="0.4 0.45 0.55 1" mass="0.06"/>'
      + '<geom type="box" size="0.004 0.025 0.018" pos="-0.035 0 0.018" rgba="0.4 0.45 0.55 1" mass="0.01"/>'
      + '<geom type="box" size="0.035 0.004 0.018" pos="0 0.025 0.018" rgba="0.4 0.45 0.55 1" mass="0.01"/>'
      + '<geom type="box" size="0.035 0.004 0.018" pos="0 -0.025 0.018" rgba="0.4 0.45 0.55 1" mass="0.01"/>',
      [R.x, R.y, TABLE_Z + 0.03])}
    ${[0, 1, 2].map((i) => freeBody(`bead${i}`,
      `<geom type="sphere" size="0.009" rgba="0.95 0.75 0.2 1" mass="0.005" condim="4"/>`,
      [R.x - 0.02 + i * 0.02, R.y, TABLE_Z + 0.05])).join('\n')}`,
    }),
    reset(ctx) {
      const r = ctx.rng;
      placeFree(ctx.sim, 'cup', [L.x + (r() - 0.5) * 0.1, L.y + (r() - 0.5) * 0.1, TABLE_Z + 0.02]);
      const scoopX = R.x + (r() - 0.5) * 0.08;
      placeFree(ctx.sim, 'scoop', [scoopX, R.y, TABLE_Z + 0.03]);
      for (let i = 0; i < 3; i++) {
        placeFree(ctx.sim, `bead${i}`, [scoopX - 0.02 + i * 0.02, R.y, TABLE_Z + 0.05]);
      }
      ctx.state.inCup = 0;
    },
    tick(ctx) {
      const cup = ctx.sim.bodyPose('cup').pos;
      let n = 0;
      for (let i = 0; i < 3; i++) {
        const b = ctx.sim.bodyPose(`bead${i}`).pos;
        if (Math.hypot(b[0] - cup[0], b[1] - cup[1]) < 0.033
            && b[2] > cup[2] - 0.01 && b[2] < cup[2] + 0.10) n++;
      }
      ctx.state.inCup = n;
      ctx.progress = n / 3;
    },
    success: (ctx) => ctx.state.inCup >= 2,
  }),

  new Level({
    id: 'L7_long_horizon',
    title: 'Open, retrieve, close',
    teaches: 'Sequencing and articulated objects under a time budget. '
           + 'This is the level whose demos are actually worth training on.',
    timeLimit: 150,
    gate: { window: 8, needed: 5 },
    hint: 'Slide the drawer open with one hand, take the cube with the other, drop it in the bin, '
        + 'then close the drawer.',
    props: () => ({
      // Carcass is base/top plus a back-stop (thin in x, opposite the joint's
      // open direction) and two side panels (thin in y, running the full
      // depth) -- NOT a wall on both x-faces. A wall on the open-x-face used
      // to sit right on top of the handle's rest position and fully overlap
      // the drawer's open-travel span, which blocked the gripper's approach
      // (this geom isn't excluded from gripper contacts the way parent/child
      // drawer-vs-cabinet contacts are) and would have visually clipped the
      // slab as it slid past. The whole body is yawed 45deg so the drawer
      // opens toward the midpoint between straight-ahead and the right arm,
      // rather than dead-ahead -- quat is yaw=+45deg about Z (see eulerToQuat),
      // which turns local -x (the joint's open direction) into world
      // (-0.707,-0.707,0), i.e. toward both -x (the robot) and -y (right, per
      // R.y<0 above).
      worldbody: `
    <body name="cabinet" pos="${(L.x + R.x) / 2 + 0.10} 0 ${TABLE_Z}" quat="0.9238795 0 0 0.3826834">
      <geom type="box" size="0.09 0.10 0.005" pos="0 0 0.005" rgba="0.45 0.35 0.28 1"/>
      <geom type="box" size="0.005 0.10 0.06" pos="0.09 0 0.06" rgba="0.45 0.35 0.28 1"/>
      <geom type="box" size="0.09 0.005 0.06" pos="0 0.10 0.06" rgba="0.45 0.35 0.28 1"/>
      <geom type="box" size="0.09 0.005 0.06" pos="0 -0.10 0.06" rgba="0.45 0.35 0.28 1"/>
      <geom type="box" size="0.09 0.10 0.005" pos="0 0 0.12" rgba="0.45 0.35 0.28 1"/>
      <body name="drawer" pos="0 0 0.04">
        <joint name="drawer_slide" type="slide" axis="-1 0 0" range="0 0.14"
               damping="6" frictionloss="1.2"/>
        <geom type="box" size="0.075 0.09 0.004" rgba="0.6 0.5 0.4 1" mass="0.3"/>
        <geom type="box" size="0.075 0.005 0.025" pos="0 0.09 0.025" rgba="0.6 0.5 0.4 1" mass="0.05"/>
        <geom type="box" size="0.075 0.005 0.025" pos="0 -0.09 0.025" rgba="0.6 0.5 0.4 1" mass="0.05"/>
        <geom type="box" size="0.008 0.03 0.008" pos="-0.078 0 0.02" rgba="0.8 0.8 0.2 1" mass="0.02"/>
      </body>
    </body>
    ${freeBody('cube', box('0.02 0.02 0.02', '0.3 0.7 0.95 1', 0.1),
      [(L.x + R.x) / 2 + 0.10, 0, TABLE_Z + 0.06])}
    <body name="bin" pos="${R.x - 0.02} ${R.y - 0.22} ${TABLE_Z}">
      <geom type="box" size="0.06 0.06 0.004" pos="0 0 0.004" rgba="0.2 0.6 0.3 1"/>
      <geom type="box" size="0.004 0.06 0.03" pos="0.06 0 0.03" rgba="0.2 0.6 0.3 1"/>
      <geom type="box" size="0.004 0.06 0.03" pos="-0.06 0 0.03" rgba="0.2 0.6 0.3 1"/>
      <site name="bin_site" pos="0 0 0.02" size="0.01" rgba="0 0 0 0"/>
    </body>`,
    }),
    reset(ctx) {
      const r = ctx.rng;
      ctx.setJoint('drawer_slide', 0);
      placeFree(ctx.sim, 'cube',
        [(L.x + R.x) / 2 + 0.10 + (r() - 0.5) * 0.06, (r() - 0.5) * 0.10, TABLE_Z + 0.06]);
      ctx.state.stage = 0;
    },
    tick(ctx) {
      const open = ctx.getJoint('drawer_slide');
      const c = ctx.sim.bodyPose('cube').pos;
      const b = ctx.sim.sitePose('bin_site').pos;
      const inBin = Math.abs(c[0] - b[0]) < 0.05 && Math.abs(c[1] - b[1]) < 0.05
        && c[2] < b[2] + 0.05;
      if (ctx.state.stage === 0 && open > 0.10) ctx.state.stage = 1;
      if (ctx.state.stage === 1 && inBin) ctx.state.stage = 2;
      if (ctx.state.stage === 2 && open < 0.02) ctx.state.stage = 3;
      ctx.progress = ctx.state.stage / 3;
      ctx.readout = ['open the drawer', 'take the cube', 'drop it in the bin', 'close the drawer'][
        Math.min(ctx.state.stage, 3)];
    },
    success: (ctx) => ctx.state.stage === 3,
  }),
];

export function levelById(id) {
  return CURRICULUM.find((l) => l.id === id);
}

// --- small quaternion helpers (wxyz) ---------------------------------------

function eulerToQuat(roll, pitch, yaw) {
  const cr = Math.cos(roll / 2), sr = Math.sin(roll / 2);
  const cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
  const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2);
  return [
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ];
}

/** Hamilton product a*b: applies b as a perturbation local to a's frame. */
function quatMul(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

/** Geodesic angle between two quaternions, radians. */
function quatAngle(a, b) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  d = Math.min(Math.abs(d), 1);
  return 2 * Math.acos(d);
}
