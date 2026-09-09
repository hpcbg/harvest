"""
Farm-device simulator entry point: one process, one physics state, two
protocol servers.

    python -m harvest_integrations.simulators.farm_sim \
        [--config config.yaml] [--modbus-port 5020] \
        [--opcua-endpoint opc.tcp://0.0.0.0:4840/harvest/] \
        [--speedup 60] [--no-modbus | --no-opcua]

Serving both protocols from one process keeps the two views coherent: a
charge request written over OPC-UA shows up as charger power and grid draw on
the Modbus side within one tick.

``--config`` accepts a HARVEST ``config.yaml`` so the simulated farm matches
the configured fleet/chargers/loads; without it, built-in defaults are used
(3 tractors, 2 chargers, 3 loads -- pyyaml is then not required).
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .farm_state import FarmState

TICK_S = 1.0


def load_state(config_path: str | None) -> FarmState:
    if not config_path:
        return FarmState.from_config(None)
    import yaml
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return FarmState.from_config(cfg)


async def run(args: argparse.Namespace) -> None:
    state = load_state(args.config)
    tasks = []

    modbus = None
    if not args.no_modbus:
        from .modbus_server import ModbusFarmServer
        modbus = ModbusFarmServer(state)
        tasks.append(asyncio.create_task(
            modbus.serve(args.modbus_host, args.modbus_port)))

    opcua = None
    if not args.no_opcua:
        from .opcua_server import OpcUaFarmServer
        opcua = OpcUaFarmServer(state, args.opcua_endpoint)
        await opcua.start()

    print(f"[farm_sim] tractors={len(state.tractors)} chargers={len(state.chargers)} "
          f"loads={len(state.loads)} speedup={args.speedup}x", flush=True)
    if modbus:
        print(f"[farm_sim] Modbus TCP  : {args.modbus_host}:{args.modbus_port}", flush=True)
    if opcua:
        print(f"[farm_sim] OPC-UA      : {args.opcua_endpoint}", flush=True)

    try:
        while True:
            state.tick(TICK_S * args.speedup)
            if opcua:
                await opcua.sync()
            if modbus:
                modbus.sync()
            await asyncio.sleep(TICK_S)
    finally:
        if opcua:
            await opcua.stop()
        for t in tasks:
            t.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description="HARVEST farm device simulator")
    parser.add_argument("--config", default=None, help="HARVEST config.yaml (optional)")
    parser.add_argument("--modbus-host", default="0.0.0.0")
    parser.add_argument("--modbus-port", type=int, default=5020)
    parser.add_argument("--opcua-endpoint", default="opc.tcp://0.0.0.0:4840/harvest/")
    parser.add_argument("--speedup", type=float, default=60.0,
                        help="simulated seconds per real second (default 60)")
    parser.add_argument("--no-modbus", action="store_true")
    parser.add_argument("--no-opcua", action="store_true")
    # Tolerate ros2-launch style extra args, as TEMPO's sims do.
    args, _ = parser.parse_known_args()
    if args.no_modbus and args.no_opcua:
        sys.exit("farm_sim: nothing to serve (both protocols disabled)")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
