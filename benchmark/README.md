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

- `regenerate_protocol_pcaps.sh`: regenerates the main experimental pcaps. It starts each benchmark server, runs the corresponding client, and captures only client-to-server payload traffic. The resulting one-way pcaps are the inputs used by the main DiffTrace evaluation pipeline.
- `capture_bidirectional_protocol_pcaps.sh`: captures full bidirectional client/server traffic. These pcaps include both requests and responses, and are mainly useful for debugging, inspecting complete protocol conversations, or running tools that require full-session context.
- `bacnet_client.sh`: BACnet-specific client wrapper. It repeatedly runs `bacnet_coverage_client` to trigger more BACnet services and message formats during pcap generation.

Note: these scripts preserve the absolute paths used in our original experimental environment, such as `/root/semvec/bitfield_groundtruth`. Before rerunning them in a different artifact location, update the `BASE`, binary, and output paths in the scripts, or adapt them to your local directory layout.

Example workflow:

```bash
cd /root/semvec/data_avaliable/benchmark
bash scripts/regenerate_protocol_pcaps.sh
```

For debugging full conversations, use:

```bash
bash scripts/capture_bidirectional_protocol_pcaps.sh all
```

## Protocol sources

`protocol_sources.md` lists the upstream open-source protocol stacks or distributions used to build the benchmark binaries.
