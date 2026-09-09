"""
harvest_integrations.simulators
===============================

Farm-device simulators that speak *real* field protocols on the wire.

Following TEMPO's ``tempo_sim`` layout, the physics
(:mod:`farm_state`) is protocol-free; :mod:`modbus_server` and
:mod:`opcua_server` are thin "skins" over it, and :mod:`farm_sim` runs both
servers over one shared state so the Modbus view (chargers, loads, grid meter)
and the OPC-UA view (tractor batteries) always agree.

Run with::

    python -m harvest_integrations.simulators.farm_sim [--config config.yaml]

Standalone -- imports no ROS and none of HARVEST's simulation engine.
"""
