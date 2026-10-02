# Benchmark Programs and Traffic Scripts

This directory contains compiled benchmark binaries and scripts for generating protocol traffic.

The directory retains 12 Linux x86-64 client/server programs for six protocols, but no captures, request-count files, runtime logs, or results. Binaries depend on host dynamic libraries; upstream source links are not a complete custom-client source package or reproducible build recipe.

## `binaries/`

The directory contains client/server binaries used to generate and replay protocol traffic:

| Protocol | Client binary | Server binary |
| --- | --- | --- |
| BACnet | `bacnet_coverage_client` | `bacnet_server` |
| CIP / EtherNet/IP | `CIP_client` | `CIP_server` |
| IEC104 | `iec104_client` | `iec104_server` |
| MMS | `MMS_client` | `MMS_server` |
| Modbus | `modbus_client` | `modbus_server` |
| S7comm | `snap7_client` | `snap7_server` |

S7comm is the paper's protocol name; `snap7_*` are program filenames and need not be renamed. Some original scripts identify MMS paths as `iec61850`; this does not mean the entire IEC61850 stack was evaluated.

## `scripts/`

- `regenerate_protocol_pcaps.sh`: regenerates the main experimental pcaps. It starts each benchmark server, runs the corresponding client, and captures only client-to-server payload traffic. The resulting one-way pcaps are the inputs used by the main DiffTrace evaluation pipeline.
- `capture_bidirectional_protocol_pcaps.sh`: captures full bidirectional client/server traffic. These pcaps include both requests and responses, and are mainly useful for debugging, inspecting complete protocol conversations, or running tools that require full-session context.
- `bacnet_client.sh`: BACnet-specific client wrapper. It repeatedly runs `bacnet_coverage_client` to trigger more BACnet services and message formats during pcap generation.

These are not portable generators: besides original absolute paths, they reference unbundled `build_work/`, `generated_src/`, runtime directories, and some old binary names. Inspect and adapt every dependency rather than only changing `BASE`. Newly generated traffic is not guaranteed to reproduce the paper's exact 208 requests.

Check tcpdump/TShark, capture permissions, ports, and dynamic libraries before use. Helpers clean up server processes, overwrite captures/logs, and create runtime files; run in an isolated experiment environment, not directly on a shared machine. This documentation update did not run them.

Example workflow:

```bash
cd /path/to/artifact/benchmark
bash scripts/regenerate_protocol_pcaps.sh
```

For debugging full conversations, use:

```bash
bash scripts/capture_bidirectional_protocol_pcaps.sh all
```

## Protocol sources

`protocol_sources.md` lists the upstream open-source protocol stacks or distributions used to build the benchmark binaries.

Snap7, used for S7comm, is sourced from SourceForge. Exact build versions, dynamic-library dependencies, and custom-client build instructions still need verification; links alone do not establish fully validated reproducibility.
