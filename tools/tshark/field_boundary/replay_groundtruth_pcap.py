#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_groundtruth_pcap.py

用途：
1. 从 pcap 中枚举所有合法 payload 候选；
2. 逐个发送到目标 server；
3. 从 pintool 总日志中切出当前数据包的执行记录。

本脚本只负责重放和执行日志采集。字段划分、日志预处理和位字段识别
请使用 analyze_replay_logs.py / run_experiment_analysis_from_logs.sh。

支持两种模式：
1. 手动先启动 pin + pintool + server，再用本脚本重放；
2. 加 `--taint`，由本脚本一键启动和停止污点分析环境。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SEMVEC_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DIFFTRACE_DIR = os.path.join(SEMVEC_ROOT, "difftrace")
if DIFFTRACE_DIR not in sys.path:
    sys.path.insert(0, DIFFTRACE_DIR)

from common import (
    DEFAULT_ENABLE_TAINT,
    DEFAULT_PIN_BIN,
    DEFAULT_PIN_LOG,
    DEFAULT_PROTO,
    DEFAULT_RECV,
    DEFAULT_RECV_TIMEOUT,
    DEFAULT_SERVER_ARGS,
    DEFAULT_SERVER_BIN,
    DEFAULT_TAINT_KILL_EXISTING,
    DEFAULT_TAINT_KILL_TIMEOUT,
    DEFAULT_TAINT_PREFIX,
    DEFAULT_TAINT_SHUTDOWN_TIMEOUT,
    DEFAULT_TAINT_STARTUP_TIME,
    DEFAULT_TAINT_STDOUT_LOG,
    DEFAULT_TAINT_TOOL,
    DEFAULT_TAINT_WORKDIR,
    DEFAULT_TARGET_HOST,
    DEFAULT_TARGET_PORT,
    DEFAULT_WAIT_MS,
    TaintProcessManager,
    build_taint_command,
    bytes_to_hex,
    collect_payload_candidates_from_pcap,
    ensure_dir,
    print_user_error,
    send_payload,
    slice_log,
    wait_for_file,
    wait_for_log_growth,
    write_json,
)


def _safe_name(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "item"


def _select_candidates(
    candidates: Sequence[Tuple[bytes, str, int]],
    packet_indices: Optional[Sequence[int]],
    limit: Optional[int],
) -> List[Tuple[bytes, str, int]]:
    if packet_indices:
        wanted = set(int(x) for x in packet_indices)
        selected = [item for item in candidates if int(item[2]) in wanted]
        missing = sorted(wanted - {int(item[2]) for item in selected})
        if missing:
            raise RuntimeError(f"pcap 包号不存在于合法候选集合中: {missing}")
    else:
        selected = list(candidates)

    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected


def should_enable_taint(args: argparse.Namespace) -> bool:
    enable = bool(args.taint)
    if getattr(args, "no_taint", False):
        enable = False
    return enable


def validate_server_port_expectation(args: argparse.Namespace) -> None:
    expected_port = getattr(args, "server_port_check", None)
    if expected_port is None:
        return

    target_port = int(args.target_port)
    expected_port = int(expected_port)

    if target_port != expected_port:
        raise RuntimeError(
            f"目标端口与期望的 server 监听端口不一致: "
            f"target-port={target_port}, server-port-check={expected_port}"
        )


def resolve_taint_command(args: argparse.Namespace) -> str:
    if args.taint_command:
        return args.taint_command
    return build_taint_command(
        args.taint_prefix,
        args.pin_bin,
        args.taint_tool,
        args.pin_log,
        args.server_bin,
        args.server_args,
    )


def resolve_server_name(args: argparse.Namespace) -> str:
    if args.taint_server_name:
        return args.taint_server_name
    if args.server_bin:
        return os.path.basename(args.server_bin)
    return ""


def _taint_unavailable_reason(taint_manager: TaintProcessManager, pin_log: str) -> str:
    pin_log_exists = os.path.exists(pin_log)
    if taint_manager.process is None:
        if pin_log_exists:
            return "污点进程未运行，但 pin 日志文件存在"
        return "污点进程未运行，且 pin 日志文件不存在"

    exit_code = taint_manager.process.poll()
    if exit_code is None:
        return "污点进程状态异常：检查阶段判定不可用，但进程似乎仍在运行"
    if exit_code < 0:
        base = f"污点进程被信号终止（signal={-exit_code}）"
    else:
        base = f"污点进程已退出（exit_code={exit_code}）"
    if not pin_log_exists:
        base += "，同时 pin 日志当前不可见"
    return base


def _restart_taint_process(args: argparse.Namespace, taint_manager: TaintProcessManager, reason: str) -> None:
    print(f"[taint] 检测到污点进程不可用，准备重启：{reason}")
    errors = []
    for attempt in range(1, max(1, int(args.taint_max_restarts)) + 1):
        print(f"[taint] 重启尝试 {attempt}/{args.taint_max_restarts}")
        try:
            taint_manager.restart()
        except Exception as exc:
            errors.append(f"attempt {attempt}: restart exception: {exc}")
            if args.taint_restart_delay > 0:
                time.sleep(args.taint_restart_delay)
            continue

        if taint_manager.process is None or taint_manager.process.poll() is not None:
            errors.append(f"attempt {attempt}: process exited after restart")
            if args.taint_restart_delay > 0:
                time.sleep(args.taint_restart_delay)
            continue

        if not wait_for_file(args.pin_log, args.taint_startup_time):
            errors.append(f"attempt {attempt}: pin log not visible: {args.pin_log}")
            if args.taint_restart_delay > 0:
                time.sleep(args.taint_restart_delay)
            continue

        print("[taint] 污点分析环境重启成功")
        return

    detail = "\n".join(errors[-5:])
    raise RuntimeError(
        "污点分析自动重启失败。\n"
        f"reason={reason}\n"
        f"restart_errors:\n{detail}"
    )


def _ensure_taint_ready(args: argparse.Namespace, taint_manager: Optional[TaintProcessManager], reason: str) -> None:
    if taint_manager is None:
        return
    if taint_manager.is_alive():
        return
    reason_text = _taint_unavailable_reason(taint_manager, args.pin_log)
    if not args.taint_auto_restart:
        raise RuntimeError(f"{reason}；{reason_text}")
    _restart_taint_process(args, taint_manager, f"{reason}; {reason_text}")


def cmd_replay(args: argparse.Namespace) -> None:
    ensure_dir(args.outdir)
    validate_server_port_expectation(args)
    taint_manager: Optional[TaintProcessManager] = None
    if should_enable_taint(args):
        if not args.server_bin:
            raise RuntimeError("--taint 模式下必须提供 --server-bin")
        command = resolve_taint_command(args)
        print(f"[taint] start command: {command}")
        taint_manager = TaintProcessManager(
            command=command,
            work_dir=args.taint_workdir,
            stdout_log=args.taint_stdout_log,
            startup_time=args.taint_startup_time,
            shutdown_timeout=args.taint_shutdown_timeout,
            kill_existing=args.taint_kill_existing,
            server_name=resolve_server_name(args),
            kill_wait=args.taint_kill_wait,
        )
        taint_manager.start()
        if taint_manager.process is None or taint_manager.process.poll() is not None:
            raise RuntimeError("污点分析进程启动失败或已退出")
        if not wait_for_file(args.pin_log, args.taint_startup_time):
            raise RuntimeError(f"污点分析已启动，但 pin 日志不可见: {args.pin_log}")
        print("[taint] 污点分析环境已启动")
    else:
        if not os.path.exists(args.pin_log):
            raise FileNotFoundError(f"pin log not found: {args.pin_log}")

    try:
        candidates = collect_payload_candidates_from_pcap(args.pcap, args.proto)
        selected = _select_candidates(candidates, args.index, args.limit)
        if not selected:
            raise RuntimeError("没有可重放的数据包")

        print(f"[replay] pcap={args.pcap}")
        print(f"[replay] proto_filter={args.proto}")
        print(f"[replay] valid_candidates={len(candidates)}")
        print(f"[replay] selected_packets={len(selected)}")

        for ordinal, (payload, proto, pkt_idx) in enumerate(selected, start=1):
            pkt_dir = os.path.join(args.outdir, f"pkt_{pkt_idx:04d}")
            ensure_dir(pkt_dir)

            raw_log_path = os.path.join(pkt_dir, "trace.log")
            meta_path = os.path.join(pkt_dir, "meta.json")
            max_attempts = 2 if taint_manager is not None else 1

            for attempt in range(1, max_attempts + 1):
                _ensure_taint_ready(args, taint_manager, f"重放到 pcap 包号 {pkt_idx} 前污点进程不可用")
                size_before = os.path.getsize(args.pin_log) if os.path.exists(args.pin_log) else 0
                t_start = time.time()
                send_info = send_payload(
                    payload=payload,
                    proto=proto,
                    host=args.target_host,
                    port=args.target_port,
                    recv=args.recv,
                    recv_timeout=args.recv_timeout,
                )
                size_after = wait_for_log_growth(args.pin_log, size_before, args.wait_ms)
                t_end = time.time()

                taint_died = taint_manager is not None and not taint_manager.is_alive()
                if taint_died:
                    if attempt < max_attempts and args.taint_auto_restart:
                        _restart_taint_process(
                            args,
                            taint_manager,
                            f"pcap 包号 {pkt_idx} 发送后进程退出，准备重发当前包",
                        )
                        continue
                    _ensure_taint_ready(args, taint_manager, f"pcap 包号 {pkt_idx} 发送后污点进程不可用")

                slice_len = slice_log(args.pin_log, size_before, size_after, raw_log_path)
                if slice_len == 0:
                    if attempt < max_attempts and taint_manager is not None and args.taint_auto_restart:
                        _restart_taint_process(
                            args,
                            taint_manager,
                            f"pcap 包号 {pkt_idx} 切片为空，重启后重发当前包",
                        )
                        continue
                    raise RuntimeError(
                        f"pcap 包号 {pkt_idx} 切片结果为空；请检查 server 是否收到请求、"
                        f"wait-ms 是否过小、或 pin log 是否正在写入"
                    )

                meta = {
                    "pcap": args.pcap,
                    "pcap_packet_index": pkt_idx,
                    "ordinal": ordinal,
                    "proto": proto,
                    "target": {
                        "host": args.target_host,
                        "port": args.target_port,
                    },
                    "payload_hex": bytes_to_hex(payload),
                    "pin_log": args.pin_log,
                    "trace_log": raw_log_path,
                    "slice_bytes": slice_len,
                    "size_before": size_before,
                    "size_after": size_after,
                    "time_start": t_start,
                    "time_end": t_end,
                    "duration_sec": t_end - t_start,
                    "send_info": send_info,
                    "attempt": attempt,
                }
                write_json(meta_path, meta)

                print(
                    f"[{ordinal:03d}/{len(selected):03d}] "
                    f"pkt_idx={pkt_idx} proto={proto} bytes={len(payload)} "
                    f"slice={slice_len}"
                    + (f" retry={attempt}" if attempt > 1 else "")
                )
                break
    finally:
        if taint_manager is not None:
            print("[taint] 停止污点分析环境")
            taint_manager.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量重放 pcap 合法数据包，并生成每包对应的原始执行日志 trace.log"
    )
    parser.add_argument("--pcap", required=True, help="pcap 文件路径")
    parser.add_argument(
        "--proto",
        choices=["auto", "tcp", "udp"],
        default=DEFAULT_PROTO,
        help="候选 payload 协议过滤",
    )
    parser.add_argument(
        "--index",
        type=int,
        nargs="*",
        help="指定要重放的原始 pcap 包号，可给多个；不指定则重放所有合法候选",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="只取前 N 个合法候选报文",
    )
    parser.add_argument("--target-host", default=DEFAULT_TARGET_HOST, help="目标主机")
    parser.add_argument("--target-port", type=int, default=DEFAULT_TARGET_PORT, help="目标端口")
    parser.add_argument("--pin-log", default=DEFAULT_PIN_LOG, help="pintool 总日志路径")
    parser.add_argument("--outdir", required=True, help="输出目录")
    parser.add_argument("--wait-ms", type=int, default=DEFAULT_WAIT_MS, help="每次发送后等待日志增长的毫秒数")
    parser.add_argument("--recv", action="store_true", default=DEFAULT_RECV, help="发送后尝试接收响应")
    parser.add_argument("--recv-timeout", type=float, default=DEFAULT_RECV_TIMEOUT, help="接收响应超时秒数")
    parser.add_argument("--taint", action="store_true", default=DEFAULT_ENABLE_TAINT, help="由脚本自动启动 pin+pintool+server")
    parser.add_argument("--no-taint", action="store_true", help="显式关闭自动启动污点分析")
    parser.add_argument("--taint-command", default="", help="自定义完整污点启动命令；若提供则忽略 pin/server 拼接参数")
    parser.add_argument("--taint-prefix", default=DEFAULT_TAINT_PREFIX, help="污点启动命令前缀")
    parser.add_argument("--pin-bin", default=DEFAULT_PIN_BIN, help="Pin 可执行文件路径")
    parser.add_argument("--taint-tool", default=DEFAULT_TAINT_TOOL, help="pintool so 路径")
    parser.add_argument("--server-bin", default=DEFAULT_SERVER_BIN, help="server 程序路径")
    parser.add_argument("--server-args", default=DEFAULT_SERVER_ARGS, help="server 启动参数，多个参数放在同一对引号中")
    parser.add_argument("--server-port-check", type=int, default=None, help="可选一致性检查：要求重放目标端口与 server 监听端口一致")
    parser.add_argument("--taint-server-name", default="", help="用于 kill_existing 的 server 名称")
    parser.add_argument("--taint-workdir", default=DEFAULT_TAINT_WORKDIR, help="污点分析启动工作目录")
    parser.add_argument("--taint-stdout-log", default=DEFAULT_TAINT_STDOUT_LOG, help="server/pin 标准输出重定向日志")
    parser.add_argument("--taint-startup-time", type=float, default=DEFAULT_TAINT_STARTUP_TIME, help="污点分析启动后等待秒数")
    parser.add_argument("--taint-shutdown-timeout", type=float, default=DEFAULT_TAINT_SHUTDOWN_TIMEOUT, help="污点分析停止超时秒数")
    parser.add_argument("--taint-kill-existing", action="store_true", default=DEFAULT_TAINT_KILL_EXISTING, help="启动前清理同名旧 server 进程")
    parser.add_argument("--taint-kill-wait", type=float, default=DEFAULT_TAINT_KILL_TIMEOUT, help="清理旧进程后的等待秒数")
    parser.add_argument("--taint-auto-restart", dest="taint_auto_restart", action="store_true", default=True, help="污点进程退出时自动重启并继续重放")
    parser.add_argument("--no-taint-auto-restart", dest="taint_auto_restart", action="store_false", help="关闭污点进程自动重启")
    parser.add_argument("--taint-max-restarts", type=int, default=3, help="自动重启最大尝试次数")
    parser.add_argument("--taint-restart-delay", type=float, default=1.0, help="自动重启失败后的等待秒数")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        cmd_replay(args)
    except Exception as exc:
        print_user_error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
