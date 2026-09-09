"""
Optional Isaac Sim simulation layer for HARVEST.

Adapts WISEPACK's proven Isaac Sim integration architecture to the
agricultural energy-management use case: HARVEST stays authoritative for
energy management, scheduling and semantic state; the simulator *executes or
visualises* the physical behaviour of entities HARVEST already models
(tractors driving to chargers, docking, charger activity) and reports the
measured physical state back.

Modules:

* ``contract``       -- the wire contract (topics, schema, scene/goal
                        derivation, fingerprints).  Pure stdlib, imported by
                        the ROS 2 bridge, the stub and Isaac Sim's own Python
                        interpreter alike, so the ends cannot drift;
* ``motion``         -- field kinematics shared by the real Isaac app and the
                        GPU-free stub, so ``isaac-demo`` exercises the exact
                        logic Isaac runs;
* ``stub``           -- rclpy stand-in simulator (``isaac-demo`` mode);
* ``harvest_isaac``  -- the standalone app run inside Isaac Sim's bundled
                        Python (never imported by HARVEST itself).

Nothing outside this directory imports ``isaacsim``/``omni``/``carb``
(WISEPACK's confinement rule); HARVEST core code never imports this package.
"""
