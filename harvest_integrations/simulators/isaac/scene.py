"""
The HARVEST Isaac Sim world -- a deliberately simple agricultural demonstration.

Built from the ``harvest-sim/1.0`` scene description, which the ROS 2 bridge
derives from HARVEST's own ``config.yaml`` (``contract.scene_from_config``).  So
the field, the tractors and the chargers in the stage are exactly the ones
HARVEST models, at the same coordinates, and there is no second list of farm
structure anywhere -- the same rule as ``runtime.build_device_endpoints``.

What is here: a flat field with crop rows, one charging station per HARVEST
charger (a bright pad, a post and a head that changes colour when the station is
delivering power), one mobile robot per HARVEST tractor, sky and sun lighting,
and a framed spectator camera for the WebRTC stream.  What is deliberately NOT
here: buildings, terrain, obstacles, crops as geometry, cameras on the vehicles,
perception of any kind.  The demonstration is about tractors driving to chargers
and telling HARVEST that they arrived.

Which robot model represents a tractor is NOT decided here -- see ``robots.py``
and ``robot_models.yaml``.  This module asks the registry for a model and places
it; that is what lets a real ZETRABOT USD replace the proxy without touching
HARVEST or this file.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from . import streaming
from .robots import RobotModel, TractorRobot

_LOG = "[harvest-isaac]"

#: Charging-station proportions, in metres.  Big enough to spot from the
#: spectator camera at field scale, small enough not to dominate the field.
PAD_SIZE = (4.0, 4.0, 0.06)
POST_RADIUS, POST_HEIGHT = 0.16, 2.2
HEAD_SIZE = (0.5, 0.34, 0.62)

#: Charger head colours: idle grey-blue, and a bright green while the station is
#: delivering power.  The physical state HARVEST reports, made visible.
HEAD_IDLE = (0.35, 0.45, 0.55)
HEAD_ACTIVE = (0.20, 0.95, 0.35)
PAD_COLOR = (0.85, 0.80, 0.20)
FIELD_COLOR = (0.28, 0.42, 0.20)
ROW_COLOR = (0.36, 0.30, 0.18)


class HarvestWorld:
    """The stage for one scene fingerprint.

    Rebuilt from scratch whenever HARVEST sends a scene with a new fingerprint,
    which is the only time the stage changes shape.  Everything that moves
    afterwards moves because PhysX moved it.
    """

    def __init__(self, model: RobotModel):
        self.model = model
        self.robots: Dict[str, TractorRobot] = {}
        self.chargers: Dict[str, Dict[str, Any]] = {}
        self.field: Dict[str, float] = {"width": 100.0, "height": 100.0}
        self.fingerprint = ""
        self._charger_active: Dict[str, bool] = {}

    # ---------------------------------------------------------------- build --
    def build(self, scene: Dict[str, Any], fingerprint: str) -> None:
        """Author the whole stage.  The caller must have STOPPED the timeline.

        Mutating a playing stage is how you get a PhysX tensor view that
        outlives the prims it points at (WISEPACK rule: stop, rebuild, play).
        """
        import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: PLC0415
        from isaacsim.core.experimental.objects import DomeLight, DistantLight  # noqa: PLC0415

        self.fingerprint = fingerprint
        self.field = dict(scene.get("field") or self.field)
        self.robots.clear()
        self.chargers.clear()

        stage_utils.create_new_stage()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        stage = stage_utils.get_current_stage()

        entities = list(scene.get("entities") or [])
        tractors = [e for e in entities if e.get("kind") == "tractor"]
        chargers = [e for e in entities if e.get("kind") == "charger"]

        self._build_field(stage, tractors + chargers)
        DomeLight("/World/Sky").set_intensities(1200)
        # A sun as well as the dome: without a directional light nothing casts a
        # shadow and the vehicles look pasted onto the field rather than on it.
        sun = DistantLight("/World/Sun")
        sun.set_intensities(2600)
        sun.set_world_poses(orientations=[_quat_from_euler(-42.0, 0.0, 155.0)])

        for spec in chargers:
            self._build_charger(stage, str(spec["id"]),
                                _xy(spec.get("pose") or spec.get("home")))

        for index, spec in enumerate(tractors):
            entity_id = str(spec["id"])
            position = _xy(spec.get("home") or spec.get("pose"))
            # Point every tractor at the middle of the charging area, so the
            # first move of the demonstration is a drive rather than a
            # three-point turn.  Physics decides everything after that.
            heading = _bearing(position, self._charging_centre(chargers))
            robot = TractorRobot(entity_id, self.model)
            robot.build(stage, position, heading)
            self.robots[entity_id] = robot

        self._build_camera(stage, tractors, chargers)

    def _charging_centre(self, chargers: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Middle of the charging area, or the field middle when there is none."""
        points = [_xy(c.get("pose") or c.get("home")) for c in chargers]
        if not points:
            return (self.field.get("width", 100.0) / 2.0,
                    self.field.get("height", 100.0) / 2.0)
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))

    def _build_field(self, stage, placed: List[Dict[str, Any]]) -> None:
        """A flat field sized to contain every entity, with crop rows.

        A thick box rather than a physics ground plane: the field is both the
        collider the wheels push against and the thing you see, and one prim
        doing both cannot drift out of alignment with the other.
        """
        from pxr import Gf, UsdGeom, UsdPhysics              # noqa: PLC0415

        xs = [_xy(e.get("home") or e.get("pose"))[0] for e in placed] or [50.0]
        ys = [_xy(e.get("home") or e.get("pose"))[1] for e in placed] or [50.0]
        margin = 25.0
        x0, x1 = min(xs) - margin, max(xs) + margin
        y0, y1 = min(ys) - margin, max(ys) + margin
        width, depth = x1 - x0, y1 - y0
        centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        self.field = {"width": round(width, 1), "height": round(depth, 1),
                      "origin": [round(x0, 1), round(y0, 1)]}

        thickness = 0.5
        field = UsdGeom.Cube.Define(stage, "/World/Field")
        field.CreateSizeAttr(1.0)
        field.AddTranslateOp().Set(
            Gf.Vec3d(centre[0], centre[1], -thickness / 2.0))
        field.AddScaleOp().Set(Gf.Vec3f(width, depth, thickness))
        field.CreateDisplayColorAttr([Gf.Vec3f(*FIELD_COLOR)])
        UsdPhysics.CollisionAPI.Apply(field.GetPrim())
        # No RigidBodyAPI: without it the box is static geometry, which is what
        # a field is.  A static collider also costs PhysX nothing per step.

        # Crop rows: visual only, and the reason they are here is that a large
        # flat plane gives the eye nothing to judge motion against -- on the
        # WebRTC stream a tractor crossing bare ground barely looks like it is
        # moving.  Ten strips, no colliders, no cost worth measuring.
        rows = UsdGeom.Xform.Define(stage, "/World/CropRows")
        for i in range(10):
            y = y0 + depth * (i + 0.5) / 10.0
            strip = UsdGeom.Cube.Define(stage, f"/World/CropRows/row_{i}")
            strip.CreateSizeAttr(1.0)
            strip.AddTranslateOp().Set(Gf.Vec3d(centre[0], y, 0.02))
            strip.AddScaleOp().Set(Gf.Vec3f(width * 0.92, 0.35, 0.04))
            strip.CreateDisplayColorAttr([Gf.Vec3f(*ROW_COLOR)])
        del rows

    def _build_charger(self, stage, charger_id: str,
                       position: Tuple[float, float]) -> None:
        """One charging station: pad, post and head.  Static, visible, obvious."""
        from pxr import Gf, UsdGeom, UsdPhysics              # noqa: PLC0415

        base = f"/World/{charger_id}"
        UsdGeom.Xform.Define(stage, base)

        pad = UsdGeom.Cube.Define(stage, f"{base}/pad")
        pad.CreateSizeAttr(1.0)
        pad.AddTranslateOp().Set(
            Gf.Vec3d(position[0], position[1], PAD_SIZE[2] / 2.0))
        pad.AddScaleOp().Set(Gf.Vec3f(*PAD_SIZE))
        pad.CreateDisplayColorAttr([Gf.Vec3f(*PAD_COLOR)])
        # The pad is walked (driven) on, so it is a collider; a 6 cm lip is
        # small enough for the wheels to roll over at any approach angle.
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())

        post = UsdGeom.Cylinder.Define(stage, f"{base}/post")
        post.CreateRadiusAttr(POST_RADIUS)
        post.CreateHeightAttr(POST_HEIGHT)
        post.CreateAxisAttr("Z")
        # Offset to one side of the pad so a tractor parks ON the pad rather
        # than into the post.
        post_xy = (position[0], position[1] + PAD_SIZE[1] / 2.0 + 0.4)
        post.AddTranslateOp().Set(
            Gf.Vec3d(post_xy[0], post_xy[1], POST_HEIGHT / 2.0))
        post.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.58)])
        UsdPhysics.CollisionAPI.Apply(post.GetPrim())

        head = UsdGeom.Cube.Define(stage, f"{base}/head")
        head.CreateSizeAttr(1.0)
        head.AddTranslateOp().Set(
            Gf.Vec3d(post_xy[0], post_xy[1], POST_HEIGHT + HEAD_SIZE[2] / 2.0))
        head.AddScaleOp().Set(Gf.Vec3f(*HEAD_SIZE))
        head.CreateDisplayColorAttr([Gf.Vec3f(*HEAD_IDLE)])

        self.chargers[charger_id] = {
            "pose": [position[0], position[1]],
            # The pad centre is the dock target, NOT the post: a tractor should
            # come to rest on the pad.
            "dock_target": [position[0], position[1]],
            "head_path": f"{base}/head",
        }
        self._charger_active[charger_id] = False

    def _build_camera(self, stage, tractors: List[Dict[str, Any]],
                      chargers: List[Dict[str, Any]]) -> None:
        """A fixed spectator camera framing the whole demonstration.

        The stream opens on this camera because a fresh stage's default viewport
        looks at the origin, and the origin is 40 m from anything interesting.
        """
        from pxr import Gf, UsdGeom                          # noqa: PLC0415

        points = [_xy(e.get("home") or e.get("pose")) for e in tractors + chargers]
        if not points:
            points = [(50.0, 50.0)]
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        spread = max(20.0, max(
            max(p[0] for p in points) - min(p[0] for p in points),
            max(p[1] for p in points) - min(p[1] for p in points)))

        camera = UsdGeom.Camera.Define(stage, streaming.SPECTATOR_CAMERA)
        camera.CreateFocalLengthAttr(24.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 2000.0))
        eye = Gf.Vec3d(cx - spread * 0.9, cy - spread * 1.5, spread * 0.85)
        look = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(cx, cy, 1.0),
                                      Gf.Vec3d(0.0, 0.0, 1.0))
        # SetLookAt builds a WORLD -> camera matrix; a prim's transform is the
        # other direction.
        UsdGeom.Xformable(camera).AddTransformOp().Set(look.GetInverse())

    # ------------------------------------------------------------- runtime --
    def attach(self) -> Tuple[int, List[str]]:
        """Bind every vehicle's articulation once the timeline is playing.

        Returns (attached, problems): a vehicle that cannot be bound is reported
        and skipped, never fatal -- partial physical execution with an honest
        error beats no telemetry at all.
        """
        problems: List[str] = []
        attached = 0
        for robot in self.robots.values():
            if robot.attach():
                attached += 1
            else:
                problems.append(f"{robot.id}: {robot.attach_error}")
        return attached, problems

    def set_charger_active(self, charger_id: str, active: bool) -> None:
        """Colour a charger head by whether it is delivering power."""
        if self._charger_active.get(charger_id) == active:
            return
        info = self.chargers.get(charger_id)
        if not info:
            return
        from pxr import Gf, UsdGeom                          # noqa: PLC0415
        import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: PLC0415

        prim = stage_utils.get_current_stage().GetPrimAtPath(info["head_path"])
        if prim:
            UsdGeom.Gprim(prim).GetDisplayColorAttr().Set(
                [Gf.Vec3f(*(HEAD_ACTIVE if active else HEAD_IDLE))])
        self._charger_active[charger_id] = active

    def charger_pose(self, charger_id: Optional[str]) -> Optional[List[float]]:
        info = self.chargers.get(str(charger_id)) if charger_id else None
        return list(info["dock_target"]) if info else None


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def _xy(value: Any) -> Tuple[float, float]:
    if not value:
        return (50.0, 50.0)
    return (float(value[0]), float(value[1]))


def _bearing(origin: Tuple[float, float], target: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0]))


def _quat_from_euler(pitch_deg: float, roll_deg: float, yaw_deg: float):
    """(w, x, y, z) for an X-Y-Z rotation, used only for the sun's direction."""
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


__all__ = ["HarvestWorld"]
