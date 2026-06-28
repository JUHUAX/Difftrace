# Benchmark Programs and Traffic Scripts

This directory contains compiled benchmark binaries and scripts for generating protocol traffic.

## `binaries/`

The directory contains client/server binaries used to generate and replay protocol traffic:

| Protocol | Client binary | Server binary |
| --- | --- | --- |
| BACnet | `bacnet_coverage_client` | `bacnet_server` |
| CIP / EtherNet/IP | `CIP_client` | `CIP_server` |
| IEC104 | `iec104_client` | `iec104_server` |
| MMS | `MMS_client` | `MMS_server` |
| Modbus | `modbus_client` | `modbus_server` |
| S7comm / Snap7 | `snap7_client` | `snap7_server` |

## `scripts/`

- `capture_bidirectional_protocol_pcaps.sh`: runs benchmark client/server communication and captures bidirectional traffic.
- `regenerate_protocol_pcaps.sh`: regenerates protocol pcaps from benchmark client/server runs.
- `bacnet_client.sh`: helper wrapper for BACnet client execution.

Example workflow:

```bash
cd /root/semvec/data_avaliable/benchmark
bash scripts/regenerate_protocol_pcaps.sh
```

If the scripts are moved to a different directory, update binary paths in the script or pass explicit paths where supported.

## Protocol sources

`protocol_sources.md` lists the upstream open-source protocol stacks or distributions used to build the benchmark binaries.
