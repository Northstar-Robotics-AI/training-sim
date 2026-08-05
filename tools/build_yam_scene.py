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

# --- joint servo gains (position actuators). Official i2rt values --
# vendor/i2rt/i2rt/robots/config/yam.yml -- not hand-tuned. They're soft
# because the real controller doesn't rely on PD stiffness to hold gravity
# load; it cancels gravity separately (see GRAVITY_COMP_FACTOR) and only uses
# PD to correct the residual.
KP = [80.0, 80.0, 80.0, 10.0, 10.0, 10.0]
KD = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]
# Per-joint scale on the gravity-compensation feedforward torque, applied at
# runtime (src/control/ik.js) as qfrc_bias * GRAVITY_COMP_FACTOR. Also from
# yam.yml; kept here so the actuator gains and the comp factor they depend on
# stay next to each other.
GRAVITY_COMP_FACTOR = [1.0, 1.1, 1.1, 1.2, 1.0, 1.0]
GRIP_KP, GRIP_KD = 800.0, 20.0

# Capsule collision proxies replacing the STL hulls, per link.
# (fromto in the link's own frame, radius). Rough but cheap; refine visually.
PROXY = {
    "link1": ([-0.06, 0.0, 0.0, 0.02, 0.0, 0.0], 0.055),
    "link2": ([0.0, -0.033, 0.01, 0.0, -0.033, 0.25], 0.05),
    "link3": ([-0.055, -0.034, -0.005, -0.055, -0.034, -0.235], 0.045),
    "link4": ([-0.055, 0.033, -0.005, -0.055, 0.033, -0.07], 0.042),
    "link5": ([0.0, 0.0, 0.0, 0.04, 0.0, -0.03], 0.04),
}


def arm_with_gripper(arm_xml, grip_xml):
    arm = mujoco.MjSpec.from_file(arm_xml)
    grip = mujoco.MjSpec.from_file(grip_xml)
    arm.attach(grip, prefix="", frame=arm.body("link6").add_frame())
    return arm


def add_collision_proxies(spec, use_proxies):
    """Demote STL geoms to visual-only and add capsule collision proxies."""
    if not use_proxies:
        return
    for body in spec.bodies:
        base = body.name.split("_", 1)[-1] if "_" in body.name else body.name
        for g in body.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
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
        # summed with the PD actuator above at each joint. Driven at runtime
        # from qfrc_bias * GRAVITY_COMP_FACTOR (src/control/ik.js), not from a
        # fixed ctrl value here.
        for i in range(6):
            c = spec.add_actuator(name=f"{side}_gcomp{i+1}")
            c.target, c.trntype = f"{side}_joint{i+1}", mujoco.mjtTrn.mjTRN_JOINT
            c.gaintype, c.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
            c.gainprm[0] = 1.0


def sanitize_xml(xml):
    """Fix two things MjSpec.to_xml() emits that MuJoCo then refuses to reload."""
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

    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    spec.option.timestep = args.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.meshdir = "meshes"

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], rgba=[0.25, 0.26, 0.30, 1])
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1],
                             type=int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL))

    table = spec.worldbody.add_body(name="table", pos=[0.40, 0.0, 0.0])
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.35, 0.6, 0.02], pos=[0, 0, 0.72],
                   rgba=[0.55, 0.45, 0.35, 1])

    sides = (("left", 0.25, -0.30), ("right", -0.25, 0.30))
    for side, y, yaw in sides:
        arm = arm_with_gripper(arm_xml, grip_xml)
        add_collision_proxies(arm, args.collision_proxies)
        f = spec.worldbody.add_frame(pos=[-0.05, y, 0.74],
                                     quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        spec.attach(arm, prefix=f"{side}_", frame=f)

    add_actuators(spec, [s[0] for s in sides])
    model = spec.compile()
    xml = sanitize_xml(spec.to_xml())

    out = args.out
    mesh_out = os.path.join(out, "meshes")
    os.makedirs(mesh_out, exist_ok=True)
    for src in (os.path.join(root, "arm/yam/assets"),
                os.path.join(root, "gripper/linear_4310/assets")):
        for fn in os.listdir(src):
            shutil.copy(os.path.join(src, fn), os.path.join(mesh_out, fn))
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
