# DiffTrace Pintool

This directory contains the custom Intel Pin pintool used by DiffTrace to collect dynamic execution evidence from protocol handlers.

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
- `INSTALL_PIN.md`: notes on installing Intel Pin separately.

## Build

Intel Pin is not included. Install Pin separately, then build the pintool:

```bash
cd /root/semvec/data_avaliable/pintool
export PIN_ROOT=/path/to/pin
make
```

The build produces:

```text
obj-intel64/pintool.so
```

## Run with a protocol handler

```bash
$PIN_ROOT/pin -t obj-intel64/pintool.so \
  -o /tmp/taint_record.log \
  -- /path/to/protocol_server [server-args]
```

The generated taint/execution log is then consumed by the DiffTrace Stage 1 and Stage 2 scripts.
