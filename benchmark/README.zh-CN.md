# Benchmark 程序和流量脚本

该目录包含已编译的 benchmark 二进制程序，以及用于生成协议流量的脚本。

目录保留6种协议的12个Linux x86-64预编译client/server程序，不包含pcap、请求计数、运行日志或实验结果。二进制依赖目标机器的动态库；上游协议源码链接不等同于完整的自定义client源码和可复现构建配方。

## `binaries/`

该目录包含用于生成和重放协议流量的 client/server 二进制程序：

| Protocol | Client binary | Server binary |
| --- | --- | --- |
| BACnet | `bacnet_coverage_client` | `bacnet_server` |
| CIP / EtherNet/IP | `CIP_client` | `CIP_server` |
| IEC104 | `iec104_client` | `iec104_server` |
| MMS | `MMS_client` | `MMS_server` |
| Modbus | `modbus_client` | `modbus_server` |
| S7comm | `snap7_client` | `snap7_server` |

正文统一使用协议名S7comm；`snap7_*`是程序文件名，不需改名。部分原脚本把MMS记为`iec61850`，这也是内部路径标识，不代表评估了整个IEC61850协议栈。

## `scripts/`

- `regenerate_protocol_pcaps.sh`：重新生成主实验使用的 pcap。该脚本会启动每个 benchmark server，运行对应 client，并且只抓取 client-to-server 方向的 payload 流量。生成的单向 pcap 是 DiffTrace 主实验管线使用的输入。
- `capture_bidirectional_protocol_pcaps.sh`：抓取完整的双向 client/server 流量。该 pcap 同时包含请求和响应，主要用于 debug、检查完整协议会话，或运行需要完整 session 上下文的工具。
- `bacnet_client.sh`：BACnet 专用 client wrapper。它会重复运行 `bacnet_coverage_client`，以在生成 pcap 时触发更多 BACnet 服务和消息格式。

这些脚本不是可直接搬迁的生成器：除了原绝对路径，还引用工件未包含的`build_work/`、`generated_src/`、运行时文件目录，并使用部分旧二进制文件名。需要先逐项适配，而不只是改`BASE`；生成新流量也不保证正好恢复论文的208个请求。

运行前检查`tcpdump`/TShark、抓包权限、端口和动态库。脚本含清理服务器进程、覆盖旧抓包/日志及创建运行文件的操作，应在隔离实验环境中执行，不要直接在共享机器上试跑。本文档更新未执行这些脚本。

示例流程：

```bash
cd /path/to/artifact/benchmark
bash scripts/regenerate_protocol_pcaps.sh
```
如果需要 debug 完整会话，可以运行：

```bash
bash scripts/capture_bidirectional_protocol_pcaps.sh all
```

## 协议来源

`protocol_sources.md` 列出了用于构建 benchmark 二进制程序的上游开源协议栈或发行版本。

S7comm对应的Snap7来源为SourceForge。具体编译版本、动态库和自定义客户端构建说明仍需核对后补充，不能仅凭链接宣称已验证完全可复现。
