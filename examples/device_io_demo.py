#!/usr/bin/env python3
"""
Direct field-protocol access through the DeviceIO abstraction: read the grid
meter over Modbus and a tractor BMS over OPC-UA, using the same seam the
DeviceFleetInterface uses.

Requires `pip install -r requirements-integrations.txt` and a reachable farm
simulator, e.g. on the host:

    python -m harvest_integrations.simulators.farm_sim --config config.yaml

    python3 examples/device_io_demo.py \
        [--modbus-host 127.0.0.1] [--modbus-port 5020] \
        [--opcua opc.tcp://127.0.0.1:4840/harvest/] [--tractor tractor_1]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from harvest_integrations.devices import (  # noqa: E402
    DeviceEndpoint, PointSpec, create_device_io,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modbus-host", default="127.0.0.1")
    parser.add_argument("--modbus-port", type=int, default=5020)
    parser.add_argument("--opcua", default="opc.tcp://127.0.0.1:4840/harvest/")
    parser.add_argument("--tractor", default="tractor_1")
    args = parser.parse_args()

    grid = create_device_io(DeviceEndpoint(
        device_id="grid", kind="grid", protocol="modbus",
        options={"host": args.modbus_host, "port": args.modbus_port},
        points={
            "grid_draw_kw": PointSpec("grid_draw_kw", 1, 0.1),
            "grid_cap_kw": PointSpec("grid_cap_kw", 2, 0.1),
            "pv_kw": PointSpec("pv_kw", 3, 0.1),
            "price_eur_per_kwh": PointSpec("price_eur_per_kwh", 5, 0.001),
        }))
    bms = create_device_io(DeviceEndpoint(
        device_id=args.tractor, kind="tractor", protocol="opcua",
        options={"endpoint": args.opcua, "object_path": args.tractor},
        points={
            "soc_pct": PointSpec("soc_pct", "SoC_pct"),
            "charging": PointSpec("charging", "Charging"),
            "charge_request": PointSpec("charge_request", "Charge_Request",
                                        writable=True),
        }))
    try:
        print("Grid meter (Modbus):", grid.read())
        print(f"{args.tractor} BMS (OPC-UA):", bms.read())

        print(f"\nRequesting charge for {args.tractor} over OPC-UA ...")
        bms.write("charge_request", 1)
        import time
        time.sleep(3)
        print(f"{args.tractor} BMS:", bms.read())
        print("Grid meter:", grid.read())
        bms.write("charge_request", 0)
    finally:
        grid.close()
        bms.close()


if __name__ == "__main__":
    main()
