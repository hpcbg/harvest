"""
harvest_integrations
====================

External-integration layer for HARVEST (TPI2/T3.x).

This package connects the HARVEST semantic device-agent model (defined in
``harvest_control``) to the outside world:

* ``devices``    -- protocol abstraction (Modbus TCP, OPC-UA, in-memory fake)
                    behind a single ``DeviceIO`` seam, plus a
                    ``DeviceFleetInterface`` backend for
                    :class:`harvest_control.FleetInterface`;
* ``simulators`` -- farm-device simulators that speak *real* Modbus/OPC-UA on
                    the wire, so the protocol clients are exercised end-to-end
                    without hardware;
* ``fiware``     -- NGSI-LD context-broker client, entity mapping and a
                    synchronisation daemon (HARVEST's model stays authoritative;
                    the broker mirrors it);
* ``codec``      -- JSON serialisation of fleet snapshots/commands, shared by
                    the HTTP API, the FIWARE sync and the ROS 2 bridge;
* ``runtime``    -- a background runner that exposes a live, continuously
                    advancing fleet over the boundary interfaces.

Design rules (mirroring TEMPO's adapter architecture):

* protocol code never leaks past ``devices`` -- everything above it sees only
  ``harvest_control`` dataclasses;
* new protocols are added by registering a ``DeviceIO`` factory
  (:func:`harvest_integrations.devices.register_protocol`) -- no change to any
  core HARVEST logic;
* everything here is optional: HARVEST's simulator, dashboard, MARL, predictor
  and ROI layers run unchanged without this package's third-party dependencies
  (``pymodbus``, ``asyncua``), which are imported lazily.
"""

__all__ = ["codec", "devices", "fiware", "runtime", "simulators"]
