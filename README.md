# yam-xr-gym

Browser-native VR training levels for bimanual I2RT YAM arms. MuJoCo runs in
WebAssembly inside the headset; there is nothing to install and no server in the
control loop. Tested against the official `i2rt-robotics/i2rt` arm models.

```bash
git clone https://github.com/i2rt-robotics/i2rt vendor/i2rt
npm install
npm run scene      # compose + flatten the bimanual MJCF (needs python + mujoco)
npm run dev        # then open on the headset over the LAN
```

WebXR requires a secure context. `localhost` counts; a LAN IP does not, so for
headset testing either run `vite --https` with a self-signed cert or tunnel it.

---

## Why this is not shaped like COBALT

COBALT runs the simulator server-side, streams WebRTC video to the client, and
takes input from a native Unity app on the Quest. That architecture is what buys
them 256 concurrent operators across 8 GPUs, and it is the right answer if you
are crowdsourcing at that scale.

The "WebXR, no download" constraint pushes the opposite way, and the tradeoff is
worth making explicitly rather than by accident:

| | COBALT (server sim) | this (client sim) |
|---|---|---|
| per-operator cost | fraction of a GPU | zero |
| control latency | 50–100 ms RTT | one frame |
| onboarding | install an app | open a URL |
| fidelity ceiling | Isaac / full MuJoCo | what a Quest CPU can step |
| domain randomization | server-side, unlimited | limited by client budget |
| operator trust | you control the binary | client can be tampered with |

That last row matters if the data is going into a training set. Client-side sim
means an operator can modify the physics or fabricate episodes. The recorder
logs object poses and RNG seed per episode specifically so demos can be
re-simulated server-side and checked against the submitted trajectory.

## Model assembly

The YAM MJCF ships in `i2rt/robot_models/arm/yam/` — six hinge joints, no
actuators, no gripper. `tools/build_yam_scene.py` attaches the `linear_4310`
gripper at `link6`, mirrors the arm with `left_`/`right_` prefixes via
`MjSpec.attach`, adds position servos, and flattens everything to one XML with a
flat mesh directory. The WASM VFS has no package-path resolution, so a flat
layout is not a style preference — it is the only thing the loader can consume.

Two things bite here and are handled in the script:

- `MjSpec.to_xml()` emits self-nested `<default class="X"><default class="X"/></default>`
  blocks after an attach, which MuJoCo then refuses to reload. Round-tripping is
  not tested upstream; validate the artifact you actually ship, not the spec
  object in memory.
- Both arms get their own mesh assets pointing at identical STLs. Deduplicating
  takes the payload from 7.3 MB to 5.2 MB.

### Collision proxies are the whole performance story

The stock geoms use the visual STLs for collision — 103,818 hull vertices.
Measured on the composed bimanual scene:

| collision geometry | native | notes |
|---|---|---|
| STL hulls (as shipped) | **18.8× realtime** | marginal on a Quest |
| capsule proxies, meshes visual-only | **111× realtime** | ~5.9× faster |

WASM runs roughly a third of native, so this is the difference between "drops
frames under contact" and "comfortable headroom". The proxies in `PROXY` in the
build script are eyeballed from link geometry; press `c` in the app to render
them and tighten any that are visibly wrong. Rebuild without them via
`--no-collision-proxies` to compare.

All eight levels compile and step cleanly. L6 is heaviest at nq=51, 14 contacts.

## Control

Clutched relative retargeting: Y/B toggle each arm frozen/unfrozen (left/right),
and while unfrozen only the delta since the toggle is applied on top of the EE
pose frozen at that moment. Absolute mapping is unusable — the operator's hand
and the YAM's wrist have different workspaces, so the arm hits a limit the
moment the hand leaves the reachable shell. Freezing and recentering is how the
operator ratchets across a workspace larger than their arm span.

Grip, held 2 s on either controller, resets the episode. X/A step to the
previous/next level.

Per-arm differential IK, damped least squares on the 6×6 site Jacobian, running
at a fixed 100 Hz independent of render rate.

### Tuning, and four bugs that look fine in a screenshot

`tools/ik_bench.py` is a line-for-line Python mirror of `src/control/ik.js`. Use
it to retune; debugging a controller inside a headset where the only diagnostic
is "the arm feels wrong" is not a good use of anyone's afternoon. Being a mirror,
it shares no code with the browser, so it cannot catch bug 4 below or anything
else in the binding layer — that is what `npm test` (`tools/ik_smoke.mjs`) is
for, and it runs the real JS control path against the real model in node.

1. **`mju_subQuat` returns the error in the site's frame; `mj_jacSite`'s
   rotational rows are world-frame.** Mixing them leaves ~60° of permanent
   residual. The arm still moves plausibly, it just never converges. Fix is one
   matrix multiply by `site.xmat`.
2. **The joint target is integrated open-loop.** When the arm is blocked by
   contact or a limit, the target runs away from the measured position —
   observed at 1.8 rad — and the loop diverges permanently. Fixed by leashing
   `qTarget` to within `maxLag` of measured `qpos`.
3. **At `posGain = 1` the DLS step always exceeds the velocity limit,** so the
   controller never decelerates near the goal and settles into a ~9 mm limit
   cycle. Dropping to 0.35 makes it proportional over the last centimetre.
4. **The WASM bindings take out-parameters as `DoubleBuffer`, not typed arrays.**
   Pass a `Float64Array` and the call is accepted, writes nothing, and does not
   throw — the array comes back untouched. Zeros are a plausible-looking result
   for every function involved, so this is invisible: an all-zero `mj_jacSite`
   means the arms simply never move, while the pose telemetry, clutch, and
   gripper all read correctly. Every out-parameter call goes through
   `src/sim/mjmath.js` so no call site has to remember this.

A smaller one: clipping the step per joint changes its *direction*, so
the gripper leaves the straight line the operator is drawing. Scale the vector
instead.

Current performance from `ik_bench.py`:

```
case                      steady       rot   t<20mm    t<5mm   leash
down 15cm               0.170 mm   0.01 deg    0.67s    0.73s    10%
fwd+left                0.183 mm   0.01 deg    0.76s    1.07s    12%
down + 45deg wrist      0.354 mm   0.01 deg    1.04s    1.74s    27%
up + yaw 60deg          8.350 mm   4.12 deg    2.43s       --    98%
lateral 20cm            0.468 mm   0.02 deg    1.43s    1.54s    20%
approach pose           0.119 mm   0.00 deg    1.17s    1.55s    16%
```

Slew is ~230 mm/s at the gripper (`maxStep` 0.01 rad/tick at 100 Hz). The
`up + yaw 60deg` row does not converge and should not: position plus a 60° yaw
is over-constrained for a 6-DoF arm in that region, so DLS settles on a
least-squares compromise and the leash sits at 98% holding a target the arm
cannot satisfy. That is the anti-windup working, not a tuning failure.

Gravity-compensation feedforward was tested and does not help — the residual was
never droop. Left out rather than carried as cargo.

## The curriculum

Ordered so each level adds exactly one failure mode, in the order novice
teleoperators actually fail:

| | level | isolates |
|---|---|---|
| L0 | Find your workspace | clutch, scale, reach envelope |
| L1 | Match the frame | orientation, not just position |
| L2 | Pick and place | grasp timing, approach direction |
| L3 | Handover | two arms, sequential |
| L4 | Lift together | two arms, simultaneous, closed chain |
| L5 | Peg in hole | precision under contact, 3 mm clearance |
| L6 | Hold and pour | asymmetric roles — fixture arm vs tool arm |
| L7 | Open, retrieve, close | sequencing, articulated object |

L0–L2 are single-arm deliberately. Handing someone two arms before they can
drive one produces operators who fixate on one hand and let the other drift,
which is hard to unlearn.

Gating requires *clean* successes, not just successes — an episode counts toward
unlocking only if it also passes the level's `qualityGate` (jerk, gripper
overforce, self-collision time). A sloppy success is not evidence of skill and
its demo is not worth training on, so the same predicate does double duty as the
dataset filter.

Levels are MJCF fragments plus three predicates, spliced into the base scene at
load. Per-episode variation writes freejoint qpos, which is free; only switching
levels recompiles.

## Recording

Fixed 30 Hz regardless of render rate. Logging per rendered frame would bake
72/90/whatever Hz into the action deltas depending on the operator's hardware.

Schema mirrors LeRobot for a bimanual arm: `observation.state` and `action` are
both 14-wide (6 joints + gripper, ×2). Controller poses, head pose, object
poses, and the RNG seed go in a sidecar — they stay out of the training tensors
but are what lets you re-simulate an episode later without the original
operator.

## Deployment

`vite.config.js` and `public/_headers` set COOP/COEP from the start. The
single-threaded MuJoCo build at the package root does not need them; the
multi-threaded build in `@mujoco/mujoco/mt` uses `SharedArrayBuffer` and does.
Setting them now makes that a one-line import change later rather than a
deployment problem found at the worst moment. Note this also means cross-origin
assets need CORP headers, which is why meshes are served from `/public` rather
than a CDN.

GitHub Pages cannot set these headers. Cloudflare Pages and Netlify honour
`_headers`.

## Known gaps

- **Nothing has run in an actual headset.** The physics, model assembly, level
  compilation and IK are all measured; the WebXR input path, render loop and HUD
  are written but unexercised. Expect the first session to be about frame
  pacing.
- Capsule collision proxies are approximate. Check them against the visual
  meshes (`c`) before trusting contact-heavy levels, especially L5.
- No camera observations. The recorder logs state and object poses only. If you
  want pixels you need an offscreen render pass, which will cost more than the
  physics does.
- Episode upload is a stub — `Recorder.upload(url)` POSTs newline-delimited JSON
  and nothing consumes it yet.
- L6 relies on three loose beads landing in a cup; it is the flakiest success
  predicate and the one most likely to need a tolerance pass after real use.
