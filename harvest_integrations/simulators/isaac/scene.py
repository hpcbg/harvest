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

#: Task marker proportions, in metres.  The zone is the work radius HARVEST
#: sent, so what you see is exactly the area the tractor has to reach.
TASK_POLE_RADIUS, TASK_POLE_HEIGHT = 0.10, 3.0
TASK_SIGN_SIZE = (1.8, 0.14, 1.0)
TASK_FLAG_SIZE = (0.9, 0.18, 0.55)
TASK_BEAM_SIZE = (0.30, 0.30, 9.0)

#: ONE COLOUR PER TASK STATE, and the whole point of the scene is that these
#: read at a glance from the spectator camera.  Pending work is present but
#: quiet; the two states that matter right now are loud; finished work goes
#: dark so the eye stops counting it.
TASK_COLORS: Dict[str, Any] = {
    "pending":   (0.55, 0.58, 0.62),      # grey — known, not due
    "assigned":  (0.98, 0.62, 0.05),      # amber — a tractor is on its way
    "active":    (0.15, 0.95, 0.30),      # green — being worked, now
    "completed": (0.10, 0.30, 0.14),      # dark green — done, deliberately dim
    "deferred":  (0.85, 0.45, 0.10),      # orange — was due, waiting again
    "missed":    (0.90, 0.12, 0.12),      # red — the window closed
}

#: Identity colours, assigned to tractors in sorted id order.  A task's flag is
#: painted in its assigned tractor's colour, and the tractor carries the same
#: colour as a roof stripe, so "which tractor is going to which task" is one
#: glance rather than a caption.
TRACTOR_ACCENTS = (
    (0.20, 0.55, 0.95),      # blue
    (0.95, 0.25, 0.75),      # magenta
    (0.98, 0.85, 0.15),      # yellow
    (0.30, 0.90, 0.90),      # cyan
    (0.70, 0.40, 0.95),      # violet
)


def accent_for(index: int):
    """Identity colour for the nth tractor.  Wraps rather than running out."""
    return TRACTOR_ACCENTS[index % len(TRACTOR_ACCENTS)]


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
        # Task markers, keyed by HARVEST task id.  Created once, then only
        # recoloured: authoring prims on a PLAYING stage is what invalidates the
        # PhysX tensor views the vehicles depend on.
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._task_drawn: Dict[str, Any] = {}
        self.accents: Dict[str, Any] = {}
        #: Task ids HARVEST sent that have no marker yet -- reported, never
        #: silently skipped, because a missing marker is a lie about the field.
        self.unknown_tasks: List[str] = []

    # ---------------------------------------------------------------- build --
    def build(self, scene: Dict[str, Any], fingerprint: str,
              tasks: Optional[Dict[str, Any]] = None) -> None:
        """Author the whole stage.  The caller must have STOPPED the timeline.

        Mutating a playing stage is how you get a PhysX tensor view that
        outlives the prims it points at (WISEPACK rule: stop, rebuild, play).

        ``tasks`` is HARVEST's task-goal map from the same command message.  The
        markers are created HERE, with the rest of the stage, and afterwards only
        recoloured -- see :meth:`sync_tasks`.
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

        # The field has to contain the work, not just the vehicles.
        self._build_field(stage, tractors + chargers
                          + [{"pose": t.get("location")}
                             for t in (tasks or {}).values()
                             if isinstance(t, dict) and t.get("location")])
        DomeLight("/World/Sky").set_intensities(1200)
        # A sun as well as the dome: without a directional light nothing casts a
        # shadow and the vehicles look pasted onto the field rather than on it.
        sun = DistantLight("/World/Sun")
        sun.set_intensities(2600)
        sun.set_world_poses(orientations=[_quat_from_euler(-42.0, 0.0, 155.0)])

        for spec in chargers:
            self._build_charger(stage, str(spec["id"]),
                                _xy(spec.get("pose") or spec.get("home")))

        # Identity colours first: the tractors and the task flags must agree.
        self.accents = {str(spec["id"]): accent_for(index)
                        for index, spec in enumerate(
                            sorted(tractors, key=lambda e: str(e["id"])))}

        for index, spec in enumerate(tractors):
            entity_id = str(spec["id"])
            position = _xy(spec.get("home") or spec.get("pose"))
            # Point every tractor at the middle of the charging area, so the
            # first move of the demonstration is a drive rather than a
            # three-point turn.  Physics decides everything after that.
            heading = _bearing(position, self._charging_centre(chargers))
            robot = TractorRobot(entity_id, self.model,
                                 accent=self.accents.get(entity_id))
            robot.build(stage, position, heading)
            self.robots[entity_id] = robot

        self.tasks = dict(tasks or {})
        self._task_drawn.clear()
        self.unknown_tasks = []
        for task_id, task in sorted(self.tasks.items()):
            self._build_task(stage, task_id, task)

        self._build_camera(stage, tractors, chargers, list(self.tasks.values()))

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
        # A GENEROUS margin, because the edge of this box is the edge of the
        # world: a vehicle that overshoots a turn near the boundary would drive
        # off solid ground and fall for ever.  Measured once, with a 25 m margin
        # and an unstable controller.
        margin = 60.0
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

    def _build_task(self, stage, task_id: str, task: Dict[str, Any]) -> None:
        """One task marker: work zone, pole, sign, assignment flag, beam.

        Visual only -- no colliders anywhere in here.  A tractor has to be able
        to drive onto its work zone, and a marker that could be bumped into
        would turn HARVEST's schedule into an obstacle course.
        """
        from pxr import Gf, UsdGeom                          # noqa: PLC0415

        position = _xy(task.get("location"))
        radius = max(2.0, float(task.get("work_radius_m", 6.0)))
        base = f"/World/Tasks/{task_id}"
        UsdGeom.Xform.Define(stage, base)

        # The work zone IS the radius HARVEST sent: what you see is the area
        # the tractor actually has to reach for the task to count as started.
        zone = UsdGeom.Cylinder.Define(stage, f"{base}/zone")
        zone.CreateRadiusAttr(radius)
        zone.CreateHeightAttr(0.08)
        zone.CreateAxisAttr("Z")
        zone.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], 0.05))

        pole = UsdGeom.Cylinder.Define(stage, f"{base}/pole")
        pole.CreateRadiusAttr(TASK_POLE_RADIUS)
        pole.CreateHeightAttr(TASK_POLE_HEIGHT)
        pole.CreateAxisAttr("Z")
        pole.AddTranslateOp().Set(
            Gf.Vec3d(position[0], position[1], TASK_POLE_HEIGHT / 2.0))
        pole.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.62, 0.64)])

        sign = UsdGeom.Cube.Define(stage, f"{base}/sign")
        sign.CreateSizeAttr(1.0)
        sign.AddTranslateOp().Set(
            Gf.Vec3d(position[0], position[1],
                     TASK_POLE_HEIGHT + TASK_SIGN_SIZE[2] / 2.0))
        sign.AddScaleOp().Set(Gf.Vec3f(*TASK_SIGN_SIZE))

        # The flag carries the ASSIGNED TRACTOR's identity colour, so the pairing
        # is readable from the spectator camera without any text.
        flag = UsdGeom.Cube.Define(stage, f"{base}/flag")
        flag.CreateSizeAttr(1.0)
        flag.AddTranslateOp().Set(
            Gf.Vec3d(position[0], position[1] + 0.5,
                     TASK_POLE_HEIGHT - TASK_FLAG_SIZE[2]))
        flag.AddScaleOp().Set(Gf.Vec3f(*TASK_FLAG_SIZE))

        # A tall beam, shown ONLY for the task being travelled to or worked.
        # This is what makes "where is the action" answerable across an 800 m
        # field, and why everything else stays quiet.
        beam = UsdGeom.Cube.Define(stage, f"{base}/beam")
        beam.CreateSizeAttr(1.0)
        beam.AddTranslateOp().Set(
            Gf.Vec3d(position[0], position[1], TASK_BEAM_SIZE[2] / 2.0))
        beam.AddScaleOp().Set(Gf.Vec3f(*TASK_BEAM_SIZE))

        self._task_drawn[task_id] = {
            "position": [position[0], position[1]],
            "radius": radius,
            "paths": {"zone": f"{base}/zone", "sign": f"{base}/sign",
                      "flag": f"{base}/flag", "beam": f"{base}/beam"},
            "state": None, "assigned": None, "emphasis": None,
        }

    def sync_tasks(self, tasks: Dict[str, Any]) -> None:
        """Recolour the markers to match HARVEST's current task states.

        Colour and visibility only.  No prim is created here, because this runs
        while the stage is PLAYING and authoring geometry then invalidates the
        PhysX views the vehicles are driven through.  A task id with no marker is
        recorded in :attr:`unknown_tasks` and picked up by the next rebuild.
        """
        from pxr import Gf, UsdGeom                          # noqa: PLC0415
        import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: PLC0415

        self.tasks = dict(tasks or {})
        unknown = [tid for tid in self.tasks if tid not in self._task_drawn]
        self.unknown_tasks = unknown
        if not self._task_drawn:
            return
        stage = stage_utils.get_current_stage()

        for task_id, drawn in self._task_drawn.items():
            task = self.tasks.get(task_id)
            state = str((task or {}).get("state") or "pending")
            assigned = (task or {}).get("assigned_tractor")
            emphasis = bool((task or {}).get("emphasis"))
            if (drawn["state"] == state and drawn["assigned"] == assigned
                    and drawn["emphasis"] == emphasis):
                continue          # nothing changed; touch no USD attribute
            drawn.update(state=state, assigned=assigned, emphasis=emphasis)

            colour = TASK_COLORS.get(state, TASK_COLORS["pending"])
            paths = drawn["paths"]
            for key in ("zone", "sign"):
                prim = stage.GetPrimAtPath(paths[key])
                if prim:
                    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set(
                        [Gf.Vec3f(*colour)])

            flag_prim = stage.GetPrimAtPath(paths["flag"])
            if flag_prim:
                accent = self.accents.get(str(assigned)) if assigned else None
                UsdGeom.Gprim(flag_prim).GetDisplayColorAttr().Set(
                    [Gf.Vec3f(*(accent or (0.30, 0.30, 0.32)))])
                UsdGeom.Imageable(flag_prim).CreateVisibilityAttr(
                    "inherited" if accent else "invisible")

            beam_prim = stage.GetPrimAtPath(paths["beam"])
            if beam_prim:
                UsdGeom.Gprim(beam_prim).GetDisplayColorAttr().Set(
                    [Gf.Vec3f(*colour)])
                UsdGeom.Imageable(beam_prim).CreateVisibilityAttr(
                    "inherited" if emphasis else "invisible")

    def task_position(self, task_id: Optional[str]) -> Optional[List[float]]:
        drawn = self._task_drawn.get(str(task_id)) if task_id else None
        return list(drawn["position"]) if drawn else None

    def _build_camera(self, stage, tractors: List[Dict[str, Any]],
                      chargers: List[Dict[str, Any]],
                      tasks: Optional[List[Dict[str, Any]]] = None) -> None:
        """A fixed spectator camera framing the whole demonstration.

        The stream opens on this camera because a fresh stage's default viewport
        looks at the origin, and the origin is 40 m from anything interesting.
        """
        from pxr import Gf, UsdGeom                          # noqa: PLC0415

        # The tasks are framed too: on an 800 m farm a camera framed on the
        # vehicles alone would leave most of the day's work off screen.
        points = [_xy(e.get("home") or e.get("pose")) for e in tractors + chargers]
        points += [_xy(t.get("location")) for t in (tasks or [])
                   if t.get("location")]
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
