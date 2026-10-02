# DiffTrace Pintool

This directory contains the custom Intel Pin pintool used by DiffTrace to collect dynamic execution evidence from protocol handlers.

It supplies byte taint and instruction/basic-block/comparison/branch/loop evidence for paper Stages 1/2. Offline bit-consumption tracking and arbitration are in `../difftrace/stage1/analyze_bitfields_planA.py`, not delegated to an LLM. The directory bundles no Pin SDK, compiled pintool, or traces.

## Files

- `pintool.cpp`: pintool entry point and instrumentation setup.
- `instrumentation.cpp/.h`: instruction instrumentation logic.
- `taintset.cpp/.h`, `taintstate.h`, `propagation.h`: taint state and propagation helpers.
- `logger.cpp/.h`: execution-event logging.
- `moduleinfo.cpp/.h`, `address.h`: module and address handling.
- `nethook.cpp/.h`: network I/O hooks used by the tracer.
- `loopdetector.h`, `loopdetector_old.h`: loop-related execution tracking helpers.
- `config.cpp/.h`: configuration and runtime options.
- `Makefile`, `makefile.rules`: build files.
- `INSTALL_PIN.sh`: helper for downloading/installing Intel Pin separately; inspect its paths before use.

## Build

Intel Pin is not included. Install Pin separately, then build the pintool:

```bash
cd /path/to/artifact/pintool
make PIN_ROOT=/path/to/pin
```

The build produces:

```text
obj-intel64/pintool.so
```

Linux x86-64 and a C++ build environment are required. The Makefile retains the original Pin 3.28 default path; the command explicitly overrides it. `INSTALL_PIN.sh` downloads the SDK, changes permissions, creates a system link, and edits shell configuration; do not run it without inspection. Build compatibility depends on the local Pin/compiler combination and was not revalidated in this documentation update.

## Run with a protocol handler

```bash
/path/to/pin/pin -t obj-intel64/pintool.so \
  -o /path/to/run/taint_record.log \
  -- /path/to/protocol_server [server-args]
```

The generated taint/execution log is then consumed by the DiffTrace Stage 1 and Stage 2 scripts.

`-o` is the supported log option; create the output directory first. Full replay normally uses `../difftrace/stage2/full.py`. Set Pin/pintool paths explicitly instead of relying on defaults that still reference the original workspace.
