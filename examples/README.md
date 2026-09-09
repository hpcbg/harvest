# HARVEST integration examples

Small, standard-library-only scripts showing how external systems talk to the
HARVEST integration stack.  Start the stack first:

```bash
./run_harvest_dashboard.sh            # dashboard + device sim + FIWARE
```

| Script | Shows |
|---|---|
| `fleet_api_demo.py` | Reading live fleet snapshots and sending commands over the HTTP fleet API (`/api/fleet/*`) |
| `fiware_demo.py` | Reading the NGSI-LD mirror entities from Orion-LD and actuating the farm by PATCHing the `FarmCommand` entity |
| `device_io_demo.py` | Talking Modbus/OPC-UA directly through the `DeviceIO` protocol abstraction (requires `pip install -r requirements-integrations.txt`; run the farm simulator or expose its ports) |
| `isaac_sim_demo.py` | The Isaac Sim demonstrator: command a charge over the fleet API, watch the simulator drive the tractor to its charger and dock (start `isaac-demo` mode first, or `isaac` + `scripts/run_isaac_sim.sh`) |

Each script takes `--help`.  ROS 2 equivalents of `fleet_api_demo.py` are the
`/harvest/*` topics inside the `ros2-bridge` container (`full` mode), e.g.:

```bash
docker compose exec ros2-bridge bash -lc \
  'source /opt/ros/jazzy/setup.bash && source /tmp/ros_install/setup.bash &&
   ros2 topic echo --once /harvest/fleet/snapshot std_msgs/msg/String'
```
