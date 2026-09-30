"""
Isaac Sim 6.0.1 WebRTC livestreaming for HARVEST -- the ONLY place that knows how.

Adapted from WISEPACK's ``simulators/isaac/streaming.py``, whose configuration
is the one PROVEN on this machine, and reduced to what HARVEST needs (a plain
dict on the telemetry channel; HARVEST has no visualization-descriptor layer).
Verified against the INSTALLED 6.0.1 package rather than an older release:

  * ``omni.kit.livestream.app`` (10.1.1) captures the application framebuffer;
  * ``omni.kit.livestream.webrtc`` (10.3.2) is the WebRTC server it drives;
  * settings live under ``/exts/omni.kit.livestream.app/primaryStream/`` --
    ``signalPort`` (49100, TCP, negotiation), ``streamPort`` (47998, UDP,
    media), ``publicIp``, ``streamType`` and ``targetFps``.

The enable sequence is the one in the shipped example
``standalone_examples/api/isaacsim.simulation_app/livestream.py``: launch
``SimulationApp`` with ``headless=True`` and ``hide_ui=False``, then
``enable_extension("omni.kit.livestream.app")``.  Older releases used
``omni.services.livestream.webrtc`` and ``/app/livestream/enabled``; NEITHER is
present in this install, so code written against them fails at runtime rather
than at import.

NO BROWSER CLIENT IS SHIPPED.  There is no HTML or JavaScript in the installed
``omni.kit.livestream.*`` extensions -- NVIDIA moved to the native "Isaac Sim
WebRTC Streaming Client" application, and an HTTP GET to the signal port
returns 501.  So HARVEST publishes the endpoint and says which client opens it;
it never offers an iframe that could only render blank.

SECURITY, stated as measured rather than as intended: the stream has no
authentication and no encryption, and Kit BINDS THE SIGNAL PORT ON 0.0.0.0
whatever address HARVEST advertises.  ``HARVEST_ISAAC_STREAM_HOST`` therefore
controls the URL that is published, NOT who can reach the port.  Access control
is necessarily external: loopback plus an SSH forward, a firewall rule scoped to
one client, or an authenticated reverse proxy.
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any, Dict, Tuple

#: The extensions this backend requires, in the installed 6.0.1 package.
REQUIRED_EXTENSIONS: Tuple[str, ...] = (
    "omni.kit.livestream.app",
    "omni.kit.livestream.webrtc",
)

#: Settings prefix for the primary stream, per the installed extension's own
#: ``extension.toml``.  Stated once so a rename is a one-line change.
PRIMARY_STREAM_SETTING = "/exts/omni.kit.livestream.app/primaryStream"

#: The camera the stream opens on: a framed view of the whole demonstration
#: field, NOT wherever a fresh stage's default viewport happens to point (which
#: is at the origin, looking away from the field).  ``scene.py`` authors it and
#: ``select_spectator_camera`` points the viewport at it.
SPECTATOR_CAMERA = "/World/HarvestFieldCamera"

#: WHERE KIT ACTUALLY LISTENS.  Not configurable, and not a default we chose:
#: the livestream extension binds every interface.  Stated here so nothing in
#: HARVEST can imply that advertising 127.0.0.1 restricts access.
KIT_BIND_ADDRESS = "0.0.0.0"

#: The advertised address when the operator sets none.  Loopback, because the
#: stream is unauthenticated.
DEFAULT_ADVERTISED_HOST = "127.0.0.1"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc


@dataclass
class StreamingConfig:
    """Streaming tunables, all overridable, none containing a public address."""

    enabled: bool = False
    #: THE ADVERTISED ADDRESS -- what HARVEST tells an operator to point the
    #: client at.  It is NOT the bind address, and conflating the two is a real
    #: reporting bug (WISEPACK measured a native client connecting through the
    #: server's routable address while the UI displayed 127.0.0.1).  Loopback by
    #: default and deliberately so: publishing a routable address for an
    #: unauthenticated stream must be an explicit act, and HARVEST never
    #: discovers or guesses this host's public IP.
    host: str = DEFAULT_ADVERTISED_HOST
    #: Whether ``host`` was chosen by an operator or is the safe default.  It
    #: drives the wording: an unset default means "local or forwarded", an
    #: explicit value means "this is the endpoint the client should use".
    host_explicit: bool = False
    signal_port: int = 49100          # TCP, connection negotiation
    stream_port: int = 47998          # UDP, media
    viewer_port: int = 0              # separate viewer/UI port; 0 = not used
    viewer_url: str = ""
    width: int = 1280
    height: int = 720
    target_fps: int = 30

    @staticmethod
    def from_env() -> "StreamingConfig":
        host = os.environ.get("HARVEST_ISAAC_STREAM_HOST", "").strip()
        cfg = StreamingConfig(
            enabled=_env_flag("HARVEST_ISAAC_STREAMING", False),
            host=host or DEFAULT_ADVERTISED_HOST,
            host_explicit=bool(host),
            signal_port=_env_int("HARVEST_ISAAC_SIGNAL_PORT", 49100),
            stream_port=_env_int("HARVEST_ISAAC_STREAM_PORT", 47998),
            viewer_port=_env_int("HARVEST_ISAAC_VIEWER_PORT", 0),
            viewer_url=os.environ.get("HARVEST_ISAAC_STREAM_URL", ""),
            target_fps=_env_int("HARVEST_ISAAC_STREAM_FPS", 30),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        problems = []
        for name, port in (("signal_port", self.signal_port),
                           ("stream_port", self.stream_port)):
            if not 1 <= port <= 65535:
                problems.append(f"{name}={port} is not a valid TCP/UDP port")
        if self.signal_port == self.stream_port:
            problems.append("signal_port and stream_port must differ")
        if problems:
            raise ValueError("Isaac streaming configuration is invalid:\n  - "
                             + "\n  - ".join(problems))

    def resolved_viewer_url(self) -> str:
        """The endpoint an operator points the client at.

        An explicit ``HARVEST_ISAAC_STREAM_URL`` always wins -- that is how a
        reverse proxy or an SSH-forwarded port is expressed.
        """
        if self.viewer_url:
            return self.viewer_url
        port = self.viewer_port or self.signal_port
        return f"http://{self.host}:{port}"

    @property
    def bind_address(self) -> str:
        """Where Kit listens.  Reported, never configured -- see KIT_BIND_ADDRESS."""
        return KIT_BIND_ADDRESS

    @property
    def is_loopback_advertised(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")

    def endpoint_note(self) -> str:
        """What the advertised address actually means for a remote client."""
        if self.is_loopback_advertised and not self.host_explicit:
            return ("Local/forwarded endpoint.  For a remote client, set "
                    "HARVEST_ISAAC_STREAM_HOST to an address it can reach.")
        if self.is_loopback_advertised:
            return ("Loopback endpoint, explicitly configured.  Reachable only "
                    "from this host or through a port-forward.")
        return (f"Client endpoint, explicitly configured: {self.host}.  The "
                "client needs both ports -- signalling over TCP and media over UDP.")

    def client_hint(self) -> str:
        """How to actually watch it.  One string, so every surface agrees."""
        return (
            "Open the NVIDIA Isaac Sim WebRTC Streaming Client and connect to "
            f"{self.resolved_viewer_url()} -- the installed Isaac Sim 6.0.1 "
            "livestream package ships no in-browser client (an HTTP GET to the "
            "signal port returns 501).\n"
            f"{self.endpoint_note()}\n"
            f"The client needs BOTH ports: {self.signal_port}/TCP (signalling) "
            f"and {self.stream_port}/UDP (media).  A TCP-only SSH tunnel "
            "negotiates a connection and then shows no picture.\n"
            f"NOTE: Kit listens on {self.bind_address} -- every interface -- "
            "whatever address is advertised here, and the stream is "
            "unauthenticated; restricting access is a firewall decision.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            # KEPT APART ON PURPOSE: `bind_address` is where Kit listens,
            # `advertised_host` is what a client is told to dial.  Reporting one
            # as the other is what made WISEPACK's UI show 127.0.0.1 for a
            # stream a remote client had just connected to.
            "bind_address": self.bind_address,
            "advertised_host": self.host,
            "advertised_host_explicit": self.host_explicit,
            "endpoint_note": self.endpoint_note(),
            "viewer_url": self.resolved_viewer_url(),
            "signal_port": self.signal_port,
            "stream_port": self.stream_port,
            "viewer_port": self.viewer_port or None,
            "target_fps": self.target_fps,
            "camera": SPECTATOR_CAMERA,
            "required_extensions": list(REQUIRED_EXTENSIONS),
        }


def port_is_free(port: int, host: str = "0.0.0.0") -> bool:
    """True when nothing is already listening on ``port`` (TCP).

    Checked BEFORE enabling the stream: Kit falls back to "an unoccupied port"
    when its configured one is taken, so the URL HARVEST publishes would point
    at a different, older stream -- the worst kind of wrong, because it shows a
    picture and the picture is of something else.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


def launch_config(config: StreamingConfig, headless: bool) -> Dict[str, Any]:
    """The ``SimulationApp`` launch configuration for this viewing mode.

    ``hide_ui=False`` with ``headless=True`` is the combination the shipped
    livestream example uses: Kit renders the full application UI into an
    offscreen framebuffer, which is what the stream then carries.  Without it
    the client connects and shows a viewport with no panels.
    """
    launch: Dict[str, Any] = {
        "headless": bool(headless),
        # The scene is a ground plane, a few boxes and cylinders; the heavier
        # renderers buy nothing and cost startup time (WISEPACK choice).
        "renderer": "RaytracedLighting",
    }
    if config.enabled:
        launch.update({
            "width": config.width,
            "height": config.height,
            "window_width": 1920,
            "window_height": 1080,
            "hide_ui": False,
            "display_options": 3286,      # the default grid, per NVIDIA's example
        })
    return launch


def enable(simulation_app: Any, config: StreamingConfig) -> Dict[str, Any]:
    """Enable the WebRTC stream on a running Kit; returns what to report.

    Refuses rather than half-succeeds.  Kit reports a missing livestream
    extension as a warning buried in a few thousand startup lines and then runs
    happily with no stream at all, so the operator waits for a picture that is
    never coming.
    """
    from isaacsim.core.experimental.utils.app import (               # noqa: PLC0415
        enable_extension, is_extension_enabled)

    if not config.enabled:
        return {"enabled": False, "state": "disabled",
                "detail": "streaming off (HARVEST_ISAAC_STREAMING=1 enables it)"}

    # Ports BEFORE the server starts: the extension reads these settings when it
    # is enabled, and silently picks another port if the configured one is busy.
    simulation_app.set_setting(f"{PRIMARY_STREAM_SETTING}/signalPort",
                               config.signal_port)
    simulation_app.set_setting(f"{PRIMARY_STREAM_SETTING}/streamPort",
                               config.stream_port)
    simulation_app.set_setting(f"{PRIMARY_STREAM_SETTING}/streamType", "webrtc")
    simulation_app.set_setting(f"{PRIMARY_STREAM_SETTING}/targetFps",
                               config.target_fps)
    # publicIp helps a client behind NAT; only ever an address the operator
    # chose, never a discovered one.
    if config.host_explicit and not config.is_loopback_advertised:
        simulation_app.set_setting(f"{PRIMARY_STREAM_SETTING}/publicIp",
                                   config.host)
    simulation_app.set_setting("/app/window/drawMouse", True)

    # enable_extension's RESULT IS CHECKED, and that is the whole point of this
    # function: Kit reports a missing or broken livestream extension as a
    # warning buried in a few thousand startup lines and then runs happily with
    # no stream at all, so the operator waits for a picture that never comes.
    if not enable_extension("omni.kit.livestream.app") and not \
            is_extension_enabled("omni.kit.livestream.app"):
        raise RuntimeError(
            "Isaac Sim refused to enable omni.kit.livestream.app; this install "
            f"cannot serve a WebRTC stream (required: "
            f"{', '.join(REQUIRED_EXTENSIONS)})")
    report = {"enabled": True, "state": "serving", **config.to_dict()}
    report["detail"] = (f"WebRTC on {config.bind_address}:{config.signal_port}"
                        f"/TCP + {config.stream_port}/UDP")
    return report


def select_spectator_camera(camera_path: str = SPECTATOR_CAMERA
                            ) -> Tuple[bool, str]:
    """Point the active viewport -- and so the stream -- at the spectator camera.

    Called after EVERY stage build, because ``create_new_stage`` hands the
    viewport back to Kit's default perspective camera.  Until this existed the
    camera was authored and never selected: the stream opened on the default
    view of the origin, and seeing the farm at all meant flying there by hand.

    Returns ``(selected, detail)`` and never raises.  The camera is how the
    demonstration is watched, not part of it, so a run without a viewport
    (``HARVEST_ISAAC_VIEW_MODE=none`` on some installs) must carry on and say so.
    """
    try:
        from omni.kit.viewport.utility import get_active_viewport   # noqa: PLC0415

        viewport = get_active_viewport()
        if viewport is None:
            return False, "no active viewport to point at the spectator camera"
        viewport.camera_path = camera_path
        return True, f"viewport camera {camera_path}"
    except Exception as exc:                                 # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


__all__ = [
    "REQUIRED_EXTENSIONS", "PRIMARY_STREAM_SETTING", "SPECTATOR_CAMERA",
    "KIT_BIND_ADDRESS", "StreamingConfig", "port_is_free",
    "launch_config", "enable", "select_spectator_camera",
]
