# DiffTrace Pintool

该目录包含 DiffTrace 使用的自定义 Intel Pin pintool，用于从协议处理程序中收集动态执行证据。

它为论文Stage 1/2提供字节污点及指令、基本块、比较、分支和循环相关执行证据。bit边界的离线消费追踪和证据裁决在`../difftrace/stage1/analyze_bitfields_planA.py`中完成，不由LLM确定。目录不包含Pin SDK、已编译pintool或执行轨迹。

## 文件说明

- `pintool.cpp`：pintool 入口和 instrumentation 初始化。
- `instrumentation.cpp/.h`：指令插桩逻辑。
- `taintset.cpp/.h`、`taintstate.h`、`propagation.h`：污点状态和传播辅助逻辑。
- `logger.cpp/.h`：执行事件日志记录。
- `moduleinfo.cpp/.h`、`address.h`：模块和地址处理。
- `nethook.cpp/.h`：tracer 使用的网络 I/O hook。
- `loopdetector.h`、`loopdetector_old.h`：循环相关执行跟踪辅助逻辑。
- `config.cpp/.h`：配置和运行时选项。
- `Makefile`、`makefile.rules`：构建文件。
- `INSTALL_PIN.sh`：单独下载安装 Intel Pin 的辅助脚本，使用前请检查其路径配置。

## 构建

artifact 不包含 Intel Pin。需要先单独安装 Pin，然后构建 pintool：

```bash
cd /path/to/artifact/pintool
make PIN_ROOT=/path/to/pin
```

构建产物为：

```text
obj-intel64/pintool.so
```

需要Linux x86-64和C++编译环境；Makefile保留了Pin 3.28的原默认路径，上述命令显式覆盖它。`INSTALL_PIN.sh`是原环境辅助脚本，会下载SDK、修改权限、创建系统链接并改写shell配置，不应未经检查直接运行。构建可用性取决于本地Pin和编译器兼容性，本轮未重新编译验证。

## 与协议处理程序一起运行

```bash
/path/to/pin/pin -t obj-intel64/pintool.so \
  -o /path/to/run/taint_record.log \
  -- /path/to/protocol_server [server-args]
```

生成的污点/执行日志随后由 DiffTrace Stage 1 和 Stage 2 脚本消费。

`-o`是本pintool支持的日志选项；输出目录需事先存在。完整协议重放通常由`../difftrace/stage2/full.py`启动，需显式设置Pin和pintool路径，不能依赖其仍指向原目录的默认值。
