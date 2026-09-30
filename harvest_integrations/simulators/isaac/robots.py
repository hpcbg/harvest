"""
Robot models for the HARVEST Isaac Sim layer -- the ONLY place that knows what
a tractor looks like or how its wheels are driven.

Everything above this module addresses a tractor by its HARVEST id and by field
coordinates: the harvest-sim/1.0 contract, the ROS 2 bridge, FleetInterface,
Diagnostics and FIWARE never learn whether the thing in the stage is a
procedural proxy, a stock NVIDIA AMR or a real ZETRABOT USD model.  Swapping in
the real model is therefore an entry in ``robot_models.yaml`` plus
``HARVEST_ISAAC_ROBOT_MODEL=<id>``, and no HARVEST change at all.

WHAT "PHYSICAL" MEANS HERE, because it is the whole point of the Isaac layer.
A tractor is a PhysX articulation: a chassis rigid body, four wheel rigid bodies
and four revolute joints with angular velocity drives.  It reaches a charger
because its wheels turn against the ground and the body accelerates, steers and
comes to a stop -- not because a transform is rewritten frame by frame.  The
poses HARVEST receives are then MEASURED from the simulated bodies, so arrival
and docking are observations rather than assertions.  ``motion.FieldKinematics``
(straight-line interpolation) remains what the GPU-free stub runs; the two
deliberately differ, and Diagnostics reports which one is connected.

Layout of a procedural vehicle (``/World/<id>`` is the articulation root):

    /World/tractor_3                     Xform (spawn pose only)
        /chassis                         UNSCALED Xform + RigidBody + Mass +
                                         ArticulationRoot  <- the base link
            /hull                        Cube (scaled) + Collision -- the body
            /cab, /bed, /beacon          visual only, no collision
        /wheel_front_left ...            Cylinder + RigidBody + Collision
        /wheel_front_left_joint ...      RevoluteJoint + angular velocity DriveAPI
    /World/PhysicsMaterials/tractor_tyre PhysicsMaterial bound to the wheels

Registry parsing is pure stdlib + PyYAML and imports nothing from Isaac, so the
registry can be validated by the ordinary test suite; every ``pxr``/``isaacsim``
import happens inside the build functions, which only ever run inside Isaac
Sim's own interpreter.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REGISTRY_FILE = Path(__file__).resolve().parent / "robot_models.yaml"
REGISTRY_SCHEMA = "harvest-isaac-robots/1.0"
_LOG = "[harvest-isaac]"

#: Selecting a model: an explicit argument wins, then this variable, then the
#: registry's ``default_model``.  Named after WISEPACK_ISAAC_ROBOT.
MODEL_ENV = "HARVEST_ISAAC_ROBOT_MODEL"


#: Steering constants for the skid-steer controller.  Deliberately conservative:
#: this is a farm vehicle crossing an open field, and a stable slow approach
#: beats a fast one that never converges.  See ``drive_towards`` for the
#: measurements behind each.
_YAW_GAIN = 2.5                 # deg/s of yaw command per degree of error
_PIVOT_DEG = 20.0               # beyond this, turn on the spot before driving
#: The tightest turn this vehicle takes AT SPEED without skidding, measured:
#: 6 m looked reasonable on paper (0.27 g at 4 m/s) and in PhysX the tractor
#: skidded -- forward speed collapsed from 3.0 to 0.6 m/s mid-corner while the
#: wheels kept turning at 10 rad/s, and the controller then had to pivot to
#: recover.  15 m keeps every turn inside the tyres' grip; a 20-degree
#: correction still closes in under two seconds, which on an 800 m farm is
#: nothing.
_MIN_TURN_RADIUS_M = 15.0
_CREEP_MPS = 0.6                # never crawl slower than this while driving


class RobotModelError(RuntimeError):
    """A model could not be resolved or built.  Always names what was tried."""


# --------------------------------------------------------------------------- #
#  The registry
# --------------------------------------------------------------------------- #
@dataclass
class RobotModel:
    """One entry of ``robot_models.yaml``, validated."""

    id: str
    display_name: str
    provider: str                      # "procedural" | "usd"
    enabled: bool
    resembles: str
    body: Dict[str, Any] = field(default_factory=dict)
    wheels: Dict[str, Any] = field(default_factory=dict)
    drive: Dict[str, Any] = field(default_factory=dict)
    colors: Dict[str, Any] = field(default_factory=dict)
    usd: Dict[str, Any] = field(default_factory=dict)

    # -- the few numbers the rest of the layer is allowed to ask for ---------
    @property
    def wheel_radius(self) -> float:
        return float(self.wheels.get("radius_m", 0.35))

    @property
    def track(self) -> float:
        return float(self.wheels.get("track_m", 1.0))

    @property
    def max_speed_mps(self) -> float:
        return float(self.drive.get("max_speed_mps", 4.0))

    @property
    def max_yaw_rate_dps(self) -> float:
        return float(self.drive.get("max_yaw_rate_dps", 45.0))

    @property
    def left_joints(self) -> List[str]:
        return [str(n) for n in (self.drive.get("left_joints") or [])]

    @property
    def right_joints(self) -> List[str]:
        return [str(n) for n in (self.drive.get("right_joints") or [])]

    def describe(self) -> Dict[str, Any]:
        """What Diagnostics is told about the model.  No paths, no host detail."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "provider": self.provider,
            "resembles": self.resembles,
            "length_m": float(self.body.get("length_m", 0.0)),
            "width_m": float(self.body.get("width_m", 0.0)),
            "wheel_radius_m": self.wheel_radius,
            "drive": str(self.drive.get("kind", "?")),
            "max_speed_mps": self.max_speed_mps,
        }


def load_registry(path: Path | str = REGISTRY_FILE) -> Dict[str, Any]:
    """Parse and validate the model registry.

    Validation is strict and happens here, once: a mistyped provider or an
    empty joint list must fail at selection with a message naming the entry,
    not halfway through building a stage where the cause is invisible.
    """
    import yaml                                            # noqa: PLC0415

    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise RobotModelError(f"robot registry {path} is missing") from exc
    except yaml.YAMLError as exc:
        raise RobotModelError(f"robot registry {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise RobotModelError(f"robot registry {path} is not a mapping")

    schema = str(raw.get("schema") or "")
    if schema.split("/")[0] != REGISTRY_SCHEMA.split("/")[0]:
        raise RobotModelError(
            f"robot registry {path} has schema {schema!r}, expected "
            f"{REGISTRY_SCHEMA!r}")

    models: Dict[str, RobotModel] = {}
    for entry in raw.get("models") or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            raise RobotModelError(f"robot registry {path} has an entry with no id")
        mid = str(entry["id"])
        provider = str(entry.get("provider") or "")
        if provider not in ("procedural", "usd"):
            raise RobotModelError(
                f"robot model {mid!r} has provider {provider!r}; expected "
                "'procedural' or 'usd'")
        model = RobotModel(
            id=mid,
            display_name=str(entry.get("display_name") or mid),
            provider=provider,
            enabled=bool(entry.get("enabled", True)),
            resembles=str(entry.get("resembles") or ""),
            body=dict(entry.get("body") or {}),
            wheels=dict(entry.get("wheels") or {}),
            drive=dict(entry.get("drive") or {}),
            colors=dict(entry.get("colors") or {}),
            usd=dict(entry.get("usd") or {}),
        )
        if str(model.drive.get("kind") or "") != "skid_steer":
            raise RobotModelError(
                f"robot model {mid!r} declares drive kind "
                f"{model.drive.get('kind')!r}; only 'skid_steer' is implemented")
        if model.enabled and not (model.left_joints and model.right_joints):
            raise RobotModelError(
                f"robot model {mid!r} is enabled but names no wheel joints; "
                "left_joints and right_joints are how the controller finds its DOFs")
        models[mid] = model

    if not models:
        raise RobotModelError(f"robot registry {path} lists no models")
    default = str(raw.get("default_model") or "")
    if default not in models:
        raise RobotModelError(
            f"robot registry {path} default_model {default!r} is not one of "
            f"{sorted(models)}")
    return {"schema": schema, "default_model": default, "models": models}


def resolve_model(requested: Optional[str] = None,
                  path: Path | str = REGISTRY_FILE) -> RobotModel:
    """Pick a model: explicit request, then ``HARVEST_ISAAC_ROBOT_MODEL``, then
    the registry default.  A disabled or unknown id is refused by name."""
    registry = load_registry(path)
    models: Dict[str, RobotModel] = registry["models"]
    wanted = (requested or os.environ.get(MODEL_ENV) or "").strip()
    source = "--robot-model" if requested else (MODEL_ENV if wanted else "default_model")
    if not wanted:
        wanted = registry["default_model"]
    model = models.get(wanted)
    if model is None:
        raise RobotModelError(
            f"unknown robot model {wanted!r} (from {source}); "
            f"the registry has {sorted(models)}")
    if not model.enabled:
        raise RobotModelError(
            f"robot model {wanted!r} (from {source}) is disabled in "
            f"{Path(path).name} -- enable it there once its asset exists")
    return model


# --------------------------------------------------------------------------- #
#  Building a vehicle in the stage (inside Isaac only)
# --------------------------------------------------------------------------- #
def _quat_from_yaw(yaw_deg: float) -> Tuple[float, float, float, float]:
    """(w, x, y, z) for a rotation about +Z."""
    half = math.radians(yaw_deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _yaw_from_quat(w: float, x: float, y: float, z: float) -> float:
    """Yaw in degrees from a (w, x, y, z) quaternion."""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny, cosy))


def _tyre_material(stage, model: RobotModel) -> str:
    """A shared high-friction physics material for the wheels.

    Created once per stage.  Default PhysX friction (0.5) lets a 950 kg body
    with four small contact patches spin its wheels instead of moving, which
    looks exactly like a broken controller.
    """
    from pxr import UsdPhysics, UsdShade                    # noqa: PLC0415

    path = "/World/PhysicsMaterials/tractor_tyre"
    if not stage.GetPrimAtPath(path):
        UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(path))
        api.CreateStaticFrictionAttr().Set(
            float(model.wheels.get("static_friction", 1.2)))
        api.CreateDynamicFrictionAttr().Set(
            float(model.wheels.get("dynamic_friction", 1.0)))
        api.CreateRestitutionAttr().Set(0.0)
    return path


def _bind_material(stage, prim_path: str, material_path: str) -> None:
    from pxr import UsdShade                                # noqa: PLC0415

    material = UsdShade.Material.Get(stage, material_path)
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(
        material, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics")


def _add_box(stage, path: str, size: Tuple[float, float, float],
             centre: Tuple[float, float, float], color, *,
             collision: bool, mass_kg: float = 0.0, rigid: bool = False):
    """A cube gprim scaled to ``size`` and translated to ``centre``.

    ``UsdGeom.Cube`` is unit-sized; scaling it (rather than authoring points)
    keeps the prim cheap and keeps PhysX on its fast box-collider path.

    NEVER USE THIS FOR A BODY A JOINT ATTACHES TO.  A joint's ``localPos`` is
    expressed in its body's own space, and that space carries this prim's SCALE:
    an anchor of 0.775 m on a prim scaled 2.6 becomes 2.0 m, so the wheels get
    constrained somewhere they were never meant to be.  Rigid bodies that carry
    joints are unscaled Xforms with a scaled child holding the collider -- see
    ``_build_procedural``.
    """
    from pxr import Gf, UsdGeom, UsdPhysics                 # noqa: PLC0415

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = cube.GetPrim()
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(prim)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    if mass_kg > 0.0:
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass_kg))
    return prim


def _add_wheel(stage, path: str, model: RobotModel,
               centre: Tuple[float, float, float], material_path: str):
    """One wheel: a Y-axis cylinder rigid body with a collider.

    The cylinder's axis is the vehicle's lateral axis, so the wheel rolls about
    it.  ``collider: sphere`` in the registry swaps the collision shape for a
    same-radius invisible sphere and keeps the cylinder as visual only -- the
    escape hatch if a PhysX release regresses on cylinder contacts.
    """
    from pxr import Gf, UsdGeom, UsdPhysics                 # noqa: PLC0415

    radius = model.wheel_radius
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(float(model.wheels.get("width_m", 0.24)))
    cyl.CreateAxisAttr("Y")
    cyl.AddTranslateOp().Set(Gf.Vec3d(*centre))
    cyl.CreateDisplayColorAttr(
        [Gf.Vec3f(*(model.colors.get("wheel") or [0.09, 0.09, 0.10]))])
    prim = cyl.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(
        float(model.wheels.get("mass_kg", 22.0)))

    if str(model.wheels.get("collider", "cylinder")) == "sphere":
        sphere = UsdGeom.Sphere.Define(stage, f"{path}/collider")
        sphere.CreateRadiusAttr(radius)
        UsdGeom.Imageable(sphere).CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        _bind_material(stage, f"{path}/collider", material_path)
    else:
        UsdPhysics.CollisionAPI.Apply(prim)
        _bind_material(stage, path, material_path)
    return prim


def _add_wheel_joint(stage, path: str, chassis_path: str, wheel_path: str,
                     anchor: Tuple[float, float, float], model: RobotModel):
    """A revolute joint about the lateral axis, with an angular VELOCITY drive.

    Velocity control means zero stiffness and large damping (Isaac's own
    guidance): the drive chases a target wheel speed and the body's motion is
    whatever the resulting tyre forces produce.  ``maxForce`` is the torque
    ceiling -- generous, because a 950 kg vehicle on four wheels needs real
    torque to pull away and the ceiling is not the interesting limit here; the
    controller's own speed clamp is.
    """
    from pxr import Gf, UsdPhysics                          # noqa: PLC0415

    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([chassis_path])
    joint.CreateBody1Rel().SetTargets([wheel_path])
    joint.CreateAxisAttr("Y")
    # Local frames: the joint sits at the wheel centre, expressed in each
    # body's own frame.  Both gprims are unrotated, so only translation differs.
    joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(float(model.drive.get("drive_damping", 1.0e5)))
    drive.CreateMaxForceAttr(float(model.drive.get("drive_max_force", 1.0e5)))
    drive.CreateTargetVelocityAttr(0.0)
    return joint.GetPrim()


def _build_procedural(stage, prim_path: str, model: RobotModel,
                      position: Tuple[float, float], heading_deg: float,
                      accent: Optional[Tuple[float, float, float]] = None) -> str:
    """Build the proxy vehicle at ``prim_path``; returns the base-link path.

    ``accent`` is the tractor's IDENTITY colour, painted on a roof stripe.  The
    task marker HARVEST assigned to this tractor carries the same colour, which
    is what makes "which tractor is going to which task" readable from a
    spectator camera without any text at all.
    """
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics    # noqa: PLC0415

    body = model.body
    length = float(body.get("length_m", 2.6))
    width = float(body.get("width_m", 1.35))
    height = float(body.get("height_m", 1.0))
    clearance = float(body.get("ground_clearance_m", 0.22))
    radius = model.wheel_radius
    track = model.track
    wheelbase = float(model.wheels.get("wheelbase_m", 1.55))

    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], 0.0))
    root.AddOrientOp().Set(Gf.Quatf(*_quat_from_yaw(heading_deg)))

    material = _tyre_material(stage, model)

    # Chassis: the base link, and the ONLY body whose pose HARVEST is told about.
    #
    # AN UNSCALED XFORM, with the hull box as a scaled CHILD.  This is not
    # tidiness, it is the difference between a vehicle and an ornament: a joint's
    # localPos is expressed in its body's own space, so on a chassis prim scaled
    # (2.6, 1.35, 1.0) the wheel anchor (0.775, 0.58, -0.36) silently becomes
    # (2.0, 0.78, -0.36).  Measured before the fix: the wheels were constrained
    # outside the body, the hull settled onto the ground, and the drives then
    # turned all four wheels at exactly their target with no load and no error
    # while the vehicle sat still -- which reads as a controller bug and is a
    # geometry bug.
    hull_z = clearance + height / 2.0
    chassis_path = f"{prim_path}/chassis"
    chassis = UsdGeom.Xform.Define(stage, chassis_path)
    chassis.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, hull_z))
    chassis_prim = chassis.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(chassis_prim)
    mass_api = UsdPhysics.MassAPI.Apply(chassis_prim)
    mass_api.CreateMassAttr(float(body.get("mass_kg", 950.0)))
    # Keep the centre of mass below the hull centre: a farm vehicle carries its
    # battery low, and so should this one, or hard steering rolls it.
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -height / 3.0))
    # The hull: the visible body and the chassis collider, scaled inside the
    # unscaled body frame where scaling is harmless.
    _add_box(stage, f"{chassis_path}/hull", (length, width, height),
             (0.0, 0.0, 0.0), model.colors.get("body") or [0.90, 0.90, 0.86],
             collision=True)

    # THE ARTICULATION ROOT GOES ON THE CHASSIS RIGID BODY, and this one line is
    # the difference between a vehicle and an ornament.  Applied to the parent
    # Xform instead, PhysX builds a FIXED-BASE articulation: the wheels then
    # turn at exactly their commanded velocity, with no load and no error,
    # while the body never moves a millimetre.  Measured here before the fix --
    # dof targets [11.28, 10.88, ...] == dof actual, body linear velocity
    # [0, 0, 0], 0.25 m of travel in 60 s -- which reads like a broken
    # controller and is not one.  On the rigid body, with no joint to the world,
    # the articulation is FLOATING and the chassis is its base link.
    UsdPhysics.ArticulationRootAPI.Apply(chassis_prim)
    physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(chassis_prim)
    # Wheels are children of the chassis and overlap its hull volume slightly;
    # self-collision would have them fighting the body they are attached to.
    physx_articulation.CreateEnabledSelfCollisionsAttr(False)
    # Four wheel contacts on a 950 kg body need more solver work than the
    # default 4/1 to stay quiet; this is the usual vehicle setting.
    physx_articulation.CreateSolverPositionIterationCountAttr(16)
    physx_articulation.CreateSolverVelocityIterationCountAttr(4)

    if body.get("cab", True):
        # Visual only, and deliberately: extra colliders on a vehicle that only
        # ever drives on a flat field buy nothing and can catch on each other.
        # Children of the chassis, so they ride with it, and positioned in the
        # chassis frame (z = 0 is the hull centre).
        cab_h = 0.85
        _add_box(stage, f"{chassis_path}/cab",
                 (length * 0.42, width * 0.82, cab_h),
                 (-length * 0.06, 0.0, height / 2.0 + cab_h / 2.0),
                 model.colors.get("cab") or [0.22, 0.24, 0.26], collision=False)
        _add_box(stage, f"{chassis_path}/bed",
                 (length * 0.34, width * 0.86, 0.22),
                 (length * 0.30, 0.0, height / 2.0 + 0.11),
                 model.colors.get("body") or [0.90, 0.90, 0.86], collision=False)
        # The beacon is RECOLOURED AT RUNTIME to show what the tractor is busy
        # with (travelling / working / charging / idle) -- see
        # TractorRobot.set_activity_colour.  At field scale it is also what makes
        # a vehicle findable in the viewport at a glance.
        _add_box(stage, f"{chassis_path}/beacon", (0.16, 0.16, 0.18),
                 (-length * 0.06, 0.0, height / 2.0 + cab_h + 0.09),
                 [0.95, 0.62, 0.05], collision=False)
        # The identity stripe: fixed for the life of the vehicle.
        _add_box(stage, f"{chassis_path}/stripe",
                 (length * 0.44, width * 0.84, 0.10),
                 (-length * 0.06, 0.0, height / 2.0 + cab_h + 0.01),
                 list(accent or (0.85, 0.85, 0.88)), collision=False)

    for name, sx, sy in (("front_left", +1.0, +1.0), ("front_right", +1.0, -1.0),
                         ("rear_left", -1.0, +1.0), ("rear_right", -1.0, -1.0)):
        centre = (sx * wheelbase / 2.0, sy * track / 2.0, radius)
        wheel_path = f"{prim_path}/wheel_{name}"
        _add_wheel(stage, wheel_path, model, centre, material)
        _add_wheel_joint(
            stage, f"{prim_path}/wheel_{name}_joint", chassis_path, wheel_path,
            # Anchor in the CHASSIS frame: the chassis gprim is itself
            # translated to hull_z, so subtract that.
            (centre[0], centre[1], centre[2] - hull_z), model)
    return chassis_path


def _resolve_usd_asset(model: RobotModel) -> str:
    """First reachable candidate for a ``usd`` model, or a message naming all."""
    from isaacsim.storage.native import get_assets_root_path  # noqa: PLC0415

    tried: List[str] = []
    root = None
    for candidate in model.usd.get("asset_path_candidates") or []:
        candidate = str(candidate)
        if candidate.startswith("/") and Path(candidate).exists():
            return candidate
        if not candidate.startswith("/") or not Path(candidate).exists():
            if root is None:
                root = get_assets_root_path() or ""
            if root:
                tried.append(root.rstrip("/") + candidate)
                continue
        tried.append(candidate)
    for path in tried:
        # A remote path cannot be stat'ed cheaply; hand the first to Usd and let
        # the reference either resolve or fail with Usd's own diagnostic.
        return path
    raise RobotModelError(
        f"robot model {model.id!r}: none of its asset_path_candidates could be "
        f"resolved (tried: {tried or model.usd.get('asset_path_candidates')})")


def _build_usd(stage, prim_path: str, model: RobotModel,
               position: Tuple[float, float], heading_deg: float) -> str:
    """Reference an existing USD model into the stage at ``prim_path``."""
    from pxr import Gf, UsdGeom                             # noqa: PLC0415

    asset = _resolve_usd_asset(model)
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.GetPrim().GetReferences().AddReference(asset)
    root.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], 0.0))
    root.AddOrientOp().Set(Gf.Quatf(*_quat_from_yaw(heading_deg)))
    base = str(model.usd.get("base_link") or "").strip("/")
    return f"{prim_path}/{base}" if base else prim_path


# --------------------------------------------------------------------------- #
#  The handle HARVEST's simulator app drives
# --------------------------------------------------------------------------- #
@dataclass
class DriveCommand:
    """What the controller decided, reported for Diagnostics."""
    linear_mps: float = 0.0
    yaw_rate_dps: float = 0.0
    reason: str = "idle"


class TractorRobot:
    """One vehicle in the stage: build it, drive it, measure it.

    Two phases, because PhysX only exposes an articulation once the timeline is
    playing: :meth:`build` authors prims on a stopped stage, :meth:`attach`
    binds the tensor view afterwards.  Calling the drive/measure methods before
    ``attach`` is a no-op that reports ``articulation_ready = False`` rather
    than raising -- the app must keep publishing telemetry through a stage
    rebuild, and a half-built vehicle is a normal transient state.
    """

    def __init__(self, entity_id: str, model: RobotModel, *,
                 prim_path: Optional[str] = None,
                 accent: Optional[Tuple[float, float, float]] = None):
        self.id = entity_id
        self.model = model
        #: Identity colour, shared with the marker of the task it is assigned.
        self.accent = accent
        self._beacon_colour: Optional[Tuple[float, float, float]] = None
        self.prim_path = prim_path or f"/World/{entity_id}"
        self.base_link_path = self.prim_path
        # The prim that CARRIES PhysicsArticulationRootAPI, which for a floating
        # vehicle is the base rigid body rather than the wrapper Xform.  Wrapping
        # the Xform instead resolves to a fixed-base articulation.
        self.articulation_path = self.prim_path
        self.articulation = None
        self._left_dofs: List[int] = []
        self._right_dofs: List[int] = []
        self._pose = (0.0, 0.0)
        self._heading_deg = 0.0
        self._speed_mps = 0.0
        self._spawn = (0.0, 0.0)
        self.last_command = DriveCommand()
        self.attach_error = ""

    # -- phase 1: author prims (stage stopped) -------------------------------
    def build(self, stage, position: Tuple[float, float],
              heading_deg: float = 0.0) -> None:
        self._spawn = (float(position[0]), float(position[1]))
        self._pose = self._spawn
        self._heading_deg = heading_deg
        if self.model.provider == "procedural":
            self.base_link_path = _build_procedural(
                stage, self.prim_path, self.model, position, heading_deg,
                self.accent)
            self.articulation_path = self.base_link_path
        else:
            self.base_link_path = _build_usd(
                stage, self.prim_path, self.model, position, heading_deg)
            # A USD model declares where its own articulation root sits; see the
            # WISEPACK note about PhysicsArticulationRootAPI landing on
            # <root>/root_joint rather than on the reference prim.
            declared = str(self.model.usd.get("articulation_root") or "").strip("/")
            self.articulation_path = (f"{self.prim_path}/{declared}" if declared
                                      else self.prim_path)

    # -- phase 2: bind the physics view (stage playing) ----------------------
    def attach(self) -> bool:
        """Bind the articulation and resolve the wheel DOF indices.

        Returns False (and records why) instead of raising: a vehicle that
        cannot be driven must still be reported, and the app must keep running.
        """
        from isaacsim.core.experimental.prims import Articulation  # noqa: PLC0415

        try:
            articulation = Articulation(self.articulation_path)
            names = list(articulation.dof_names)
            missing = [n for n in self.model.left_joints + self.model.right_joints
                       if n not in names]
            if missing:
                self.attach_error = (
                    f"wheel joints {missing} are not DOFs of {self.articulation_path} "
                    f"(it has {names}) -- check robot_models.yaml")
                return False
            self.articulation = articulation
            self._left_dofs = [names.index(n) for n in self.model.left_joints]
            self._right_dofs = [names.index(n) for n in self.model.right_joints]
            # Velocity control: zero stiffness, the damping authored on the
            # drive.  Set again through the tensor view because a USD model may
            # arrive with position drives configured.
            try:
                articulation.set_dof_drive_types(
                    "force", dof_indices=self._left_dofs + self._right_dofs)
            except Exception:            # a model may not allow this; harmless
                pass
            self.attach_error = ""
            return True
        except Exception as exc:                     # noqa: BLE001
            self.attach_error = f"{type(exc).__name__}: {exc}"
            return False

    @property
    def ready(self) -> bool:
        return self.articulation is not None

    # -- appearance ---------------------------------------------------------
    def set_activity_colour(self, colour: Tuple[float, float, float]) -> None:
        """Paint the beacon to show what the tractor is busy with.

        Only writes when the colour actually changes: this is called every frame
        and a USD attribute write per frame per vehicle is pure waste.  Failures
        are swallowed -- a beacon is a courtesy to the viewer, and a USD model
        without one must not break the drive loop.
        """
        if colour == self._beacon_colour:
            return
        self._beacon_colour = colour
        try:
            from pxr import Gf, UsdGeom                      # noqa: PLC0415
            import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: PLC0415

            prim = stage_utils.get_current_stage().GetPrimAtPath(
                f"{self.base_link_path}/beacon")
            if prim:
                UsdGeom.Gprim(prim).GetDisplayColorAttr().Set(
                    [Gf.Vec3f(*colour)])
        except Exception:                                    # noqa: BLE001
            pass

    # -- measurement (never echoed from the goal) ---------------------------
    def measure(self) -> None:
        """Read the simulated pose and speed back out of PhysX."""
        if self.articulation is None:
            return
        try:
            positions, orientations = self.articulation.get_world_poses()
            pos = positions.numpy()[0]
            quat = orientations.numpy()[0]
            self._pose = (float(pos[0]), float(pos[1]))
            self._heading_deg = _yaw_from_quat(*[float(q) for q in quat])
            linear, _ = self.articulation.get_velocities()
            vel = linear.numpy()[0]
            self._speed_mps = float(math.hypot(float(vel[0]), float(vel[1])))
        except Exception as exc:                     # noqa: BLE001
            self.attach_error = f"pose read failed: {type(exc).__name__}: {exc}"

    @property
    def pose(self) -> Tuple[float, float]:
        return self._pose

    @property
    def heading_deg(self) -> float:
        return self._heading_deg

    @property
    def speed_mps(self) -> float:
        return self._speed_mps

    # -- control ------------------------------------------------------------
    def drive_towards(self, target: Optional[Tuple[float, float]], *,
                      arrival_radius_m: float, speed_mps: float) -> DriveCommand:
        """Steer toward ``target``; brake inside ``arrival_radius_m``.

        Deliberately simple closed-loop steering, and simple is the right
        choice for an open field with no obstacles: turn toward the bearing,
        drive at a speed scaled by how well the vehicle is pointed and how
        close it is, stop when arrived.  Path planning belongs upstream of
        Isaac if HARVEST ever needs it, not here.
        """
        if target is None:
            self.last_command = DriveCommand(0.0, 0.0, "no target")
            return self._apply(self.last_command)

        dx, dy = target[0] - self._pose[0], target[1] - self._pose[1]
        distance = math.hypot(dx, dy)
        if distance <= arrival_radius_m:
            self.last_command = DriveCommand(0.0, 0.0, "arrived")
            return self._apply(self.last_command)

        bearing = math.degrees(math.atan2(dy, dx))
        error = (bearing - self._heading_deg + 180.0) % 360.0 - 180.0
        wanted_yaw = _YAW_GAIN * error
        top = min(speed_mps, self.model.max_speed_mps)

        # PIVOT WHEN MISALIGNED, THEN CAP THE TURN BY THE SPEED -- in that
        # order, and the order is the whole lesson here.  A skid-steer vehicle
        # cannot turn tightly at speed (the wheels just slip), so a fast drive
        # and a hard turn commanded together produce neither and the heading
        # error never closes.  Measured on the live farm before this existed:
        # tractors holding 1.3-2.1 m/s while rotating two degrees a second,
        # distance-to-target INCREASING, one driving clean off the field.
        #
        #   * beyond PIVOT_DEG of misalignment, do not drive at all -- turn on
        #     the spot, where a stationary skid-steer has all the authority it
        #     needs and no slip;
        #   * inside it, drive at a speed tapered by alignment and by how close
        #     the target is, and clamp the YAW to what that speed can sweep
        #     without slipping: the tightest honest turn is MIN_TURN_RADIUS, so
        #     the feasible rate is v / R.
        #
        # (An earlier attempt had this backwards -- capping SPEED at
        # omega * R -- which made a 5-degree heading error clamp the tractor to
        # 0.55 m/s and crawl across an 800 m farm.  v >= omega * R is a lower
        # bound on speed, not an upper one.)
        if abs(error) > _PIVOT_DEG:
            yaw_rate = max(-self.model.max_yaw_rate_dps,
                           min(self.model.max_yaw_rate_dps, wanted_yaw))
            command = DriveCommand(0.0, yaw_rate, "turning")
        else:
            alignment = max(0.0, math.cos(math.radians(error))) ** 2
            # Ease off over the last few metres so the vehicle rolls to a stop
            # instead of overshooting and hunting around the target.
            approach = min(1.0, distance / 8.0 + 0.2)
            speed = max(_CREEP_MPS, min(top, top * alignment * approach))
            feasible_yaw = min(self.model.max_yaw_rate_dps,
                               math.degrees(speed / _MIN_TURN_RADIUS_M))
            yaw_rate = max(-feasible_yaw, min(feasible_yaw, wanted_yaw))
            command = DriveCommand(speed, yaw_rate, "driving")
        self.last_command = command
        return self._apply(command)

    def _apply(self, command: DriveCommand) -> DriveCommand:
        """Skid-steer mixing: wheel angular velocities, in rad/s.

        The tensor articulation API takes SI units for revolute DOFs, so a
        wheel that should roll the body forward at ``v`` turns at ``v / r``;
        the yaw term is the usual differential-drive one over the track width.
        Verified on this machine by commanding a known speed and measuring the
        body displacement (``--self-test drive``).
        """
        if self.articulation is None:
            return command
        radius = self.model.wheel_radius
        yaw_rads = math.radians(command.yaw_rate_dps)
        left = (command.linear_mps - yaw_rads * self.model.track / 2.0) / radius
        right = (command.linear_mps + yaw_rads * self.model.track / 2.0) / radius
        try:
            import numpy as np                              # noqa: PLC0415

            self.articulation.set_dof_velocity_targets(
                np.array([[left] * len(self._left_dofs)], dtype=np.float32),
                dof_indices=self._left_dofs)
            self.articulation.set_dof_velocity_targets(
                np.array([[right] * len(self._right_dofs)], dtype=np.float32),
                dof_indices=self._right_dofs)
        except Exception as exc:                     # noqa: BLE001
            self.attach_error = f"drive failed: {type(exc).__name__}: {exc}"
        return command


__all__ = [
    "MODEL_ENV", "REGISTRY_FILE", "RobotModel", "RobotModelError",
    "TractorRobot", "DriveCommand", "load_registry", "resolve_model",
]
