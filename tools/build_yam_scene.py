#!/usr/bin/env python3
"""
Build a flat, browser-loadable bimanual YAM MJCF from the official i2rt models.

Output (default ./public/assets/yam/):
    bimanual_yam.xml      flat MJCF, no <include>, meshdir="meshes"
    meshes/*.stl          deduplicated meshes (one copy shared by both arms)

Why flatten: the MuJoCo WASM VFS has no notion of a package path, so every
asset must be written into MEMFS by exact filename before MjModel.loadFromXML.
One flat XML + one flat mesh dir keeps the loader trivial.
"""
import argparse, os, re, shutil, time
import numpy as np
import mujoco
import trimesh
import coacd

# --- joint servo gains (position actuators). Official i2rt values --
# vendor/i2rt/i2rt/robots/config/yam.yml -- not hand-tuned. They're soft
# because the real controller doesn't rely on PD stiffness to hold gravity
# load; it cancels gravity separately (see GRAVITY_COMP_FACTOR, applied at
# runtime via qfrc_applied) and only uses PD to correct the residual.
KP = [80.0, 80.0, 80.0, 10.0, 10.0, 10.0]
KD = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]
# Per-joint scale on the gravity-compensation feedforward torque, applied at
# runtime (src/control/ik.js) as qfrc_bias * GRAVITY_COMP_FACTOR. Kept here so
# the actuator gains and the comp factor they depend on stay next to each other.
#
# Unity, unlike yam.yml's [1.0, 1.1, 1.1, 1.2, 1.0, 1.0]. That is a hardware
# trim for the real motors; in sim the gravity model is the plant, so exact
# compensation is 1.0. i2rt's own sim path does the same (get_robot.py passes
# np.ones() rather than the yaml values). See the note in src/control/ik.js.
GRAVITY_COMP_FACTOR = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
GRIP_KP, GRIP_KD = 800.0, 20.0
# Per-joint peak motor torque, Nm. yam.yml's motor_list puts DM4340 on joints
# 1-3 and DM4310 on 4-6; the Damiao datasheets give DM4340 9 Nm rated / 27 Nm
# peak (40:1) and DM4310 3 Nm rated / 7-12.5 Nm peak (10:1), and i2rt's own CAN
# protocol ceilings (motor_drivers/utils.py MotorConstants.TORQUE_MAX) are 28
# and 10 -- two independent sources agreeing. See set_joint_torque_limits.
JOINT_PEAK_TORQUE = [27.0, 27.0, 27.0, 10.0, 10.0, 10.0]

# Capsule collision proxies replacing the STL hulls, per link.
# (fromto in the link's own frame, radius). Rough but cheap; refine visually.
PROXY = {
    "link1": ([-0.06, 0.0, 0.0, 0.02, 0.0, 0.0], 0.055),
    "link2": ([0.0, -0.033, 0.01, 0.0, -0.033, 0.25], 0.05),
    "link3": ([-0.055, -0.034, -0.005, -0.055, -0.034, -0.235], 0.045),
    # Wrist end pulled back from -0.07 to -0.064: giving the gripper real mesh
    # collision (see MESH_COLLISION) revealed this capsule's old reach poked
    # ~0.8mm into the gripper mount at rest, registering a false self-collision
    # on every idle frame.
    "link4": ([-0.055, 0.033, -0.005, -0.055, 0.033, -0.064], 0.042),
    "link5": ([0.0, 0.0, 0.0, 0.04, 0.0, -0.03], 0.04),
}

# gripper/tip_left/tip_right get no capsule proxy and no single-mesh convex
# hull either: both pad real empty space around a part that isn't
# capsule/box-shaped, and that slack is exactly what you feel as "the object
# registers as gripped before the fingers actually touch it," or "the palm
# hits things before it visually looks like it should." These instead collide
# on a CoACD convex decomposition of their own mesh -- a handful of small
# convex hulls whose union hugs the real (non-convex) shape, rather than one
# hull bridging every concavity. See decompose_collision_mesh / MESH_COLLISION.
MESH_COLLISION = {"gripper", "tip_left", "tip_right"}

# threshold/max_convex_hull trade decomposition tightness for part count (and
# build time); the mcts_*/resolution knobs are turned down from CoACD's
# defaults purely for speed -- this runs at asset-build time, not per frame,
# but a few minutes per mesh at the defaults is still needless. ~8 hulls gets
# the gripper palm from ~2x the real mesh volume (single convex hull) down to
# ~1.4x, in well under a minute.
COACD_ARGS = dict(
    threshold=0.08, max_convex_hull=8,
    mcts_nodes=8, mcts_iterations=25, mcts_max_depth=2,
    preprocess_resolution=20, resolution=1500,
)


def decompose_collision_mesh(stl_path):
    """Convex-decompose a mesh (CoACD) into a handful of convex hulls whose
    union tightly bounds the real shape. Returns [(vertices, faces), ...]."""
    coacd.set_log_level("error")
    tm = trimesh.load(stl_path, process=False)
    parts = coacd.run_coacd(coacd.Mesh(tm.vertices, tm.faces), **COACD_ARGS)
    return [(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int32))
            for v, f in parts]


def arm_with_gripper(arm_xml, grip_xml):
    arm = mujoco.MjSpec.from_file(arm_xml)
    grip = mujoco.MjSpec.from_file(grip_xml)
    arm.attach(grip, prefix="", frame=arm.body("link6").add_frame())
    return arm


def add_collision_proxies(spec, use_proxies, mesh_offsets=None):
    """Demote STL geoms to visual-only and add capsule collision proxies.
    gripper/tip_left/tip_right are demoted too -- their real collision geoms
    are convex-decomposition hulls added later, once both arms exist in the
    top-level spec (see add_gripper_collision_hulls).

    mesh_offsets, if given, records each MESH_COLLISION body's visual geom
    pos/quat (keyed by base name). The STL vertices for gripper/tip_left/
    tip_right are not authored with their origin at the body frame -- the
    original <geom pos=... quat=...> is what places the mesh correctly, and
    the CoACD hulls (decomposed straight from those same raw STL vertices,
    see decompose_collision_mesh) need that exact same placement or they land
    with zero offset relative to the body: right shape, wrong spot.
    """
    if not use_proxies:
        return
    for body in spec.bodies:
        # Strip a "left_"/"right_" side prefix only -- a plain split("_", 1)
        # also mangles unprefixed names that contain their own underscore
        # (tip_left/tip_right), turning them into "left"/"right" and silently
        # dropping them out of PROXY.
        base = body.name
        for side_prefix in ("left_", "right_"):
            if base.startswith(side_prefix):
                base = base[len(side_prefix):]
                break
        for g in body.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                if mesh_offsets is not None and base in MESH_COLLISION and base not in mesh_offsets:
                    mesh_offsets[base] = (np.array(g.pos), np.array(g.quat))
                g.contype, g.conaffinity = 0, 0
                g.group = 2
        if base in PROXY:
            ft, r = PROXY[base]
            c = body.add_geom(name=f"{body.name}_col",
                              type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                              fromto=ft, size=[r, 0, 0],
                              rgba=[0, 1, 0, 0.25], group=3,
                              condim=3, mass=0.0)
            c.density = 0.0


def add_gripper_collision_hulls(spec, sides, mesh_out, hull_parts, mesh_offsets):
    """Add the CoACD collision hulls (see decompose_collision_mesh) to both
    arms' gripper/tip_left/tip_right bodies. Runs on the fully-assembled
    top-level spec (bodies named e.g. "left_gripper") so each mesh asset is
    written once and shared by both arms, same as the visual STLs.

    Each hull geom gets the same pos/quat as the body's original visual mesh
    geom (mesh_offsets, from add_collision_proxies) -- the hulls are
    decomposed from that mesh's raw STL vertices, in that mesh's local frame,
    not the body frame, so without this they render at the right shape but
    the wrong place (this bit the wrist/gripper hardest since that's where
    the offset is largest)."""
    for base, parts in hull_parts.items():
        for i, (verts, faces) in enumerate(parts):
            name = f"{base}_col{i}"
            trimesh.Trimesh(vertices=verts, faces=faces).export(
                os.path.join(mesh_out, f"{name}.stl"))
            spec.add_mesh(name=name, file=f"{name}.stl")
    for side in sides:
        for base, parts in hull_parts.items():
            body = spec.body(f"{side}_{base}")
            pos, quat = mesh_offsets[base]
            for i in range(len(parts)):
                g = body.add_geom(name=f"{side}_{base}_col{i}",
                                  type=mujoco.mjtGeom.mjGEOM_MESH,
                                  meshname=f"{base}_col{i}",
                                  pos=pos, quat=quat,
                                  rgba=[0, 1, 0, 0.25], group=3,
                                  condim=3, mass=0.0)
                g.density = 0.0


def add_actuators(spec, sides):
    for side in sides:
        for i in range(6):
            a = spec.add_actuator(name=f"{side}_act{i+1}")
            a.target, a.trntype = f"{side}_joint{i+1}", mujoco.mjtTrn.mjTRN_JOINT
            a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
            a.gainprm[0], a.biasprm[1], a.biasprm[2] = KP[i], -KP[i], -KD[i]
            a.ctrlrange, a.ctrllimited = [-3.2, 3.2], True
        g = spec.add_actuator(name=f"{side}_gripper")
        g.target, g.trntype = f"{side}_joint7", mujoco.mjtTrn.mjTRN_JOINT
        g.gaintype, g.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        g.gainprm[0], g.biasprm[1], g.biasprm[2] = GRIP_KP, -GRIP_KP, -GRIP_KD
        g.ctrlrange, g.ctrllimited = [0.0, 0.0475], True
        # Gravity-comp feedforward: plain torque actuators, one per arm joint,
        # summed with the PD actuator above and clamped together by the joint's
        # actuatorfrcrange (see set_joint_torque_limits) -- gravity comp and the
        # servo share one budget because on hardware they share one motor.
        # Driven at runtime from qfrc_bias * GRAVITY_COMP_FACTOR
        # (src/control/ik.js), not from a fixed ctrl value here.
        for i in range(6):
            c = spec.add_actuator(name=f"{side}_gcomp{i+1}")
            c.target, c.trntype = f"{side}_joint{i+1}", mujoco.mjtTrn.mjTRN_JOINT
            c.gaintype, c.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
            c.gainprm[0] = 1.0


def set_joint_torque_limits(spec, sides):
    """Override the arm model's inherited actuatorfrcrange with the real motors'.

    i2rt's yam.xml carries actuatorfrcrange="-10 10" on every joint, but that
    model declares no actuators at all, so the attribute is inert there and was
    never sized against anything. Here it is live and clamps gcomp + PD
    together, so it has to be the motor's real peak torque or the arm cannot
    hold itself up: joints 1-3 are DM4340 and need ~15 Nm at joint2 at full
    extension, which 10 Nm cannot supply.

    Peak, not rated, because MuJoCo has no thermal model. A real YAM holding an
    extended pose does run joint2 above its 9 Nm continuous rating and heats up
    -- that is why i2rt's driver monitors temp_mos/temp_rotor -- but it can
    deliver the torque, so a hard clamp at the rated value would be wrong in the
    other direction.
    """
    for side in sides:
        for i in range(6):
            spec.joint(f"{side}_joint{i+1}").actfrcrange = JOINT_PEAK_TORQUE[i] * np.array([-1.0, 1.0])


def sanitize_xml(xml, mesh_out):
    """Fix things MjSpec.to_xml() emits that MuJoCo then refuses to reload, or
    that leak build-machine-local detail into the shipped asset."""
    # 0. meshdir was set to mesh_out's absolute path so the collision hulls
    # (written straight into mesh_out, see add_gripper_collision_hulls) would
    # resolve at compile time; the shipped XML needs the portable relative
    # form the browser's meshdir="meshes/" fetch convention expects. Matched
    # by regex, not an exact string, because to_xml() appends its own "/"
    # (and on Windows mesh_out itself is backslash-separated).
    xml = re.sub(r'meshdir="[^"]*"', 'meshdir="meshes/"', xml)
    # 1. attach() writes <default class="X"><default class="X"/></default>
    xml = re.sub(r'<default class="([^"]+)">\s*<default class="\1"\s*/>\s*</default>',
                 r'<default class="\1"/>', xml)
    # 2. left_link2.stl and right_link2.stl are the same file -> keep one copy
    xml = re.sub(r'\b(?:left|right)_(base|link[1-6]|gripper|tip_left|tip_right)\.stl',
                 r'\1.stl', xml)
    seen, out = set(), []
    for line in xml.splitlines():
        m = re.search(r'<mesh name="(?:left|right)_([^"]+)"', line)
        if m:
            if m.group(1) in seen:
                continue
            seen.add(m.group(1))
            line = re.sub(r'name="(?:left|right)_', 'name="', line)
        out.append(line)
    xml = "\n".join(out)
    xml = re.sub(r'mesh="(?:left|right)_(base|link[1-6]|gripper|tip_left|tip_right)"',
                 r'mesh="\1"', xml)
    return xml


def build(args):
    root = args.i2rt_models
    arm_xml = os.path.join(root, "arm/yam/yam.xml")
    grip_xml = os.path.join(root, "gripper/linear_4310/linear_4310.xml")
    grip_assets = os.path.join(root, "gripper/linear_4310/assets")

    out = args.out
    mesh_out = os.path.abspath(os.path.join(out, "meshes"))
    os.makedirs(mesh_out, exist_ok=True)
    for src in (os.path.join(root, "arm/yam/assets"), grip_assets):
        for fn in os.listdir(src):
            shutil.copy(os.path.join(src, fn), os.path.join(mesh_out, fn))

    hull_parts = {}
    if args.collision_proxies:
        for base in MESH_COLLISION:
            print(f"[coacd] decomposing {base}.stl ...")
            hull_parts[base] = decompose_collision_mesh(
                os.path.join(grip_assets, f"{base}.stl"))
            print(f"[coacd] {base}: {len(hull_parts[base])} convex hulls")

    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    spec.option.timestep = args.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Absolute for now so spec.compile() below can find both the copied
    # visual STLs and the collision hulls written directly into mesh_out;
    # sanitize_xml rewrites it back to the portable "meshes/" the browser
    # expects before the XML is written out.
    spec.meshdir = mesh_out

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], rgba=[0.25, 0.26, 0.30, 1])
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1],
                             type=int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL))

    table = spec.worldbody.add_body(name="table", pos=[0.40, 0.0, 0.0])
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.35, 0.6, 0.02], pos=[0, 0, 0.72],
                   rgba=[0.55, 0.45, 0.35, 1])

    mesh_offsets = {}
    sides = (("left", 0.25, -0.30), ("right", -0.25, 0.30))
    for side, y, yaw in sides:
        arm = arm_with_gripper(arm_xml, grip_xml)
        add_collision_proxies(arm, args.collision_proxies, mesh_offsets)
        f = spec.worldbody.add_frame(pos=[-0.05, y, 0.74],
                                     quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        spec.attach(arm, prefix=f"{side}_", frame=f)

    if hull_parts:
        add_gripper_collision_hulls(spec, [s[0] for s in sides], mesh_out, hull_parts, mesh_offsets)

    add_actuators(spec, [s[0] for s in sides])
    set_joint_torque_limits(spec, [s[0] for s in sides])
    model = spec.compile()
    xml = sanitize_xml(spec.to_xml(), mesh_out)

    xml_path = os.path.join(out, "bimanual_yam.xml")
    with open(xml_path, "w") as f:
        f.write(xml)

    # verify the artifact we actually ship
    check = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(check)
    mujoco.mj_forward(check, data)
    print(f"[ok] {xml_path}")
    print(f"     nq={check.nq} nv={check.nv} nu={check.nu} "
          f"nbody={check.nbody} ngeom={check.ngeom} nmesh={check.nmesh}")
    for s in ("left_grasp_site", "right_grasp_site"):
        print(f"     {s} @ {np.round(data.site(s).xpos, 4)}")
    print(f"     mesh payload: {sum(os.path.getsize(os.path.join(mesh_out, f)) for f in os.listdir(mesh_out)) / 1e6:.1f} MB")

    for _ in range(200):
        mujoco.mj_step(check, data)
    t0, N = time.perf_counter(), 4000
    for _ in range(N):
        mujoco.mj_step(check, data)
    sps = N / (time.perf_counter() - t0)
    print(f"     native: {sps:.0f} steps/s = {sps * args.timestep:.1f}x realtime "
          f"(expect ~1/3 of this in WASM)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--i2rt-models", default="./i2rt/i2rt/robot_models")
    p.add_argument("--out", default="./public/assets/yam")
    p.add_argument("--timestep", type=float, default=0.002)
    p.add_argument("--collision-proxies", action="store_true", default=True)
    p.add_argument("--no-collision-proxies", dest="collision_proxies", action="store_false")
    build(p.parse_args())
