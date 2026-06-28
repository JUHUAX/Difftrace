#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
difftrace 公共模块
包含默认配置、工具函数、污点管理器等共享代码
"""

import json
import os
import random
import re
import shlex
import signal
import socket
import subprocess
import time
from typing import Dict, List, Optional, Tuple

# ================= 默认配置（可在此处修改） =================
DEFAULT_OUTDIR = "./out"  # 默认输出目录
DEFAULT_SEED = 1337  # 默认随机种子
DEFAULT_PROTO = "auto"  # 默认协议选择：auto|tcp|udp
DEFAULT_TARGET_HOST = "127.0.0.1"  # 默认目标主机
DEFAULT_TARGET_PORT = 102  # 默认目标端口
DEFAULT_WAIT_MS = 1500  # 默认发送后等待日志稳定的毫秒数
DEFAULT_RECV = False  # 默认是否接收响应
DEFAULT_RECV_TIMEOUT = 0.5  # 默认接收超时时间（秒）
DEFAULT_PIN_LOG = "../pintool_new/taint_record.log"  # 默认 pin 日志路径

DEFAULT_ENABLE_TAINT = False  # 默认是否启动污点分析
DEFAULT_TAINT_PREFIX = ""  # 污点分析前置环境/命令
DEFAULT_PIN_BIN = "pin"  # Pin 可执行文件路径
DEFAULT_TAINT_TOOL = "../pintool_new/obj-intel64/pintool.so"  # taint 插桩 so 路径
DEFAULT_SERVER_BIN = "../snap7/S7server"  # 被测 server 可执行文件路径
DEFAULT_SERVER_ARGS = ""  # server 启动参数
DEFAULT_TAINT_WORKDIR = "."  # 污点分析工作目录
DEFAULT_TAINT_STDOUT_LOG = "../pintool_new/taint_server.log"  # 污点分析标准输出日志
DEFAULT_TAINT_STARTUP_TIME = 3.0  # 污点分析启动等待时间（秒）
DEFAULT_TAINT_SHUTDOWN_TIMEOUT = 5.0  # 污点分析优雅退出超时（秒）
DEFAULT_TAINT_KILL_EXISTING = True  # 默认是否启动前清理残留进程
DEFAULT_TAINT_LOG_MARKER = True  # 默认是否写入发送标记到 taint stdout 日志
DEFAULT_TAINT_KILL_TIMEOUT = 2.0  # 默认清理残留进程后的等待时间（秒）

POLL_INTERVAL_SEC = 0.05  # 轮询日志尺寸的间隔（秒）
# ============================================================


# ================= 工具函数 =================

def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> dict:
    """读取 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    """写入 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def print_user_error(message: str) -> None:
    """打印用户友好的错误信息"""
    print(f"[error] {message}")


def hex_to_bytes(hex_str: str) -> bytes:
    """将 hex 字符串转换为 bytes"""
    hex_str = hex_str.strip().lower()
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    if len(hex_str) == 0 or len(hex_str) % 2 != 0:
        raise ValueError("hex string must be non-empty and even-length")
    return bytes.fromhex(hex_str)


def bytes_to_hex(data: bytes) -> str:
    """将 bytes 转换为 hex 字符串"""
    return data.hex()


def load_sample(outdir: str) -> dict:
    """从 outdir 加载 sample.json"""
    path = os.path.join(outdir, "sample.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"sample.json not found in outdir: {path}")
    return read_json(path)


def load_fields(outdir: str) -> dict:
    """从 outdir 加载 fields.json"""
    path = os.path.join(outdir, "fields.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"fields.json not found in outdir: {path}")
    return read_json(path)


def resolve_proto(requested: str, sample_proto: str) -> str:
    """解析协议类型"""
    if requested == "auto":
        if not sample_proto:
            raise ValueError("proto=auto requires sample.json to have proto")
        return sample_proto
    return requested


def next_run_id(runs_dir: str) -> int:
    """获取下一个 run ID"""
    existing = []
    if not os.path.exists(runs_dir):
        return 0
    for name in os.listdir(runs_dir):
        if name.startswith("run_") and name.endswith(".json"):
            parts = name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                existing.append(int(parts[1]))
    return max(existing) + 1 if existing else 0


# ================= 网络相关 =================

def send_payload(payload: bytes, proto: str, host: str, port: int,
                 recv: bool, recv_timeout: float) -> Dict[str, str]:
    """发送 payload 到目标"""
    info = {"proto": proto, "host": host, "port": port}
    if proto == "tcp":
        with socket.create_connection((host, port), timeout=3.0) as sock:
            sock.sendall(payload)
            info["sent_bytes"] = str(len(payload))
            if recv:
                sock.settimeout(recv_timeout)
                try:
                    data = sock.recv(4096)
                except socket.timeout:
                    data = b""
                info["recv_bytes"] = str(len(data))
                info["recv_preview_hex"] = data[:20].hex() if data else ""
    elif proto == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sent = sock.sendto(payload, (host, port))
            info["sent_bytes"] = str(sent)
        finally:
            sock.close()
    else:
        raise ValueError(f"unsupported proto: {proto}")
    return info


# ================= 日志切片相关 =================

def wait_for_log_growth(pin_log: str, size_before: int, wait_ms: int) -> int:
    """等待日志增长"""
    time.sleep(wait_ms / 1000.0)
    try:
        size_now = os.path.getsize(pin_log)
    except OSError:
        return size_before
    return max(size_now, size_before)


def slice_log(pin_log: str, size_before: int, size_after: int, out_path: str) -> int:
    """切片日志文件"""
    if size_after <= size_before:
        return 0
    with open(pin_log, "rb") as f_in:
        f_in.seek(size_before)
        data = f_in.read(size_after - size_before)
    with open(out_path, "wb") as f_out:
        f_out.write(data)
    return len(data)


def run_once(outdir: str, payload: bytes, proto: str, host: str, port: int,
             pin_log: str, wait_ms: int, role: str, recv: bool,
             recv_timeout: float) -> dict:
    """执行一次发送并切片日志"""
    if not os.path.exists(pin_log):
        raise FileNotFoundError(f"pin log not found: {pin_log}")

    runs_dir = os.path.join(outdir, "runs")
    ensure_dir(runs_dir)

    run_id = next_run_id(runs_dir)
    run_prefix = f"run_{run_id:03d}_{role}"
    log_path = os.path.join(runs_dir, f"{run_prefix}.log")
    meta_path = os.path.join(runs_dir, f"{run_prefix}.json")

    size_before = os.path.getsize(pin_log)
    t_start = time.time()
    send_info = send_payload(payload, proto, host, port, recv, recv_timeout)
    size_after = wait_for_log_growth(pin_log, size_before, wait_ms)
    t_end = time.time()

    slice_len = slice_log(pin_log, size_before, size_after, log_path)
    if slice_len == 0:
        raise RuntimeError("log slice is empty; check target or pin log output")

    meta = {
        "run_id": run_id,
        "role": role,
        "proto": proto,
        "target": {"host": host, "port": port},
        "payload_hex": bytes_to_hex(payload),
        "pin_log": pin_log,
        "log_slice_path": log_path,
        "slice_bytes": slice_len,
        "size_before": size_before,
        "size_after": size_after,
        "time_start": t_start,
        "time_end": t_end,
        "duration_sec": t_end - t_start,
        "send_info": send_info,
    }
    write_json(meta_path, meta)
    return meta


# ================= 污点分析进程管理 =================

class TaintProcessManager:
    """污点分析进程管理器（启动/停止）"""

    def __init__(self, command: str, work_dir: str, stdout_log: str,
                 startup_time: float, shutdown_timeout: float,
                 kill_existing: bool, server_name: str, kill_wait: float):
        self.command = command
        self.work_dir = work_dir
        self.stdout_log = stdout_log
        self.startup_time = startup_time
        self.shutdown_timeout = shutdown_timeout
        self.kill_existing = kill_existing
        self.server_name = server_name
        self.kill_wait = kill_wait
        self.process = None
        self.log_file = None
        self.consecutive_restart_count = 0

    def _cmdline_tokens(self, pid: int) -> List[str]:
        """读取指定 pid 的命令行 token。"""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read()
        except OSError:
            return []
        if not raw:
            return []
        return [
            token.decode("utf-8", errors="ignore")
            for token in raw.split(b"\x00")
            if token
        ]

    def _is_same_named_process(self, pid: int) -> bool:
        """匹配真正启动目标 server 的进程，但尽量避免误伤普通 wrapper。"""
        target = os.path.basename(self.server_name)
        if not target:
            return False
        tokens = self._cmdline_tokens(pid)
        if tokens and os.path.basename(tokens[0]) == target:
            return True
        for idx, token in enumerate(tokens[:-1]):
            if token == "--" and os.path.basename(tokens[idx + 1]) == target:
                return True
        shell_exec_pattern = re.compile(
            rf"(^|[\s;/&|])(?:exec\s+)?(?:[^\s]+/)?{re.escape(target)}(?:[\s\"']|$)"
        )
        if tokens and os.path.basename(tokens[0]) in {"bash", "sh"}:
            for token in tokens[1:]:
                if shell_exec_pattern.search(token):
                    return True
        try:
            exe_path = os.readlink(f"/proc/{pid}/exe")
        except OSError:
            return False
        return os.path.basename(exe_path) == target

    def _kill_existing_processes(self) -> None:
        if not self.server_name:
            return
        current_pid = os.getpid()
        try:
            result = subprocess.run(
                ["pgrep", "-f", self.server_name],
                capture_output=True,
                text=True,
                check=False,
            )
            if not result.stdout.strip():
                return
            for pid_str in result.stdout.strip().split("\n"):
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                if pid == current_pid:
                    continue
                if not self._is_same_named_process(pid):
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    continue
        finally:
            if self.kill_wait > 0:
                time.sleep(self.kill_wait)

    def start(self, startup_wait: Optional[float] = None) -> None:
        if self.kill_existing:
            self._kill_existing_processes()
        os.makedirs(os.path.dirname(os.path.abspath(self.stdout_log)) or ".", exist_ok=True)
        self.log_file = open(self.stdout_log, "a", buffering=1)
        self._log_marker("启动污点分析")
        self.process = subprocess.Popen(
            self.command,
            shell=True,
            cwd=self.work_dir,
            stdin=subprocess.PIPE,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        wait_time = self.startup_time if startup_wait is None else startup_wait
        time.sleep(wait_time)

    def is_alive(self) -> bool:
        """检查污点分析进程是否存活。"""
        return self.process is not None and self.process.poll() is None

    def restart(self, startup_wait: Optional[float] = None) -> None:
        """重启污点分析进程。"""
        self.stop()
        self.start(startup_wait=startup_wait)

    def reset_restart_backoff(self) -> None:
        """在一次正常恢复后重置连续重启计数。"""
        self.consecutive_restart_count = 0

    def stop(self) -> None:
        if not self.process:
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            return
        try:
            self._log_marker("停止污点分析")
            if self.process.poll() is None:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    self.process.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait(timeout=2)
        finally:
            if self.log_file:
                self.log_file.close()
            self.log_file = None
            self.process = None

    def log_packet_marker(self, label: str) -> None:
        self._log_marker(label)

    def _log_marker(self, label: str) -> None:
        if not self.log_file:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 80
        self.log_file.write(f"\n{sep}\n[{timestamp}] {label}\n{sep}\n")
        self.log_file.flush()


def build_taint_command(prefix: str, pin_bin: str, taint_tool: str,
                        pin_log: str, server_bin: str, server_args: str) -> str:
    """构建污点分析启动命令"""
    parts = []
    if prefix:
        # prefix 保持原样，允许用户传入环境变量或包装命令。
        parts.append(prefix)

    parts.append(shlex.quote(pin_bin))
    parts.append("-t")
    parts.append(shlex.quote(taint_tool))
    parts.append("-o")
    parts.append(shlex.quote(pin_log))
    parts.append("--")
    parts.append(shlex.quote(server_bin))

    if server_args:
        normalized_args = [shlex.quote(arg) for arg in shlex.split(server_args)]
        parts.extend(normalized_args)

    return " ".join(parts).strip()


def wait_for_file(path: str, timeout_sec: float) -> bool:
    """等待文件出现"""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(POLL_INTERVAL_SEC)
    return os.path.exists(path)


# ================= Pcap 采样相关 =================

try:
    from scapy.all import PcapReader, Raw, TCP, UDP
except Exception:
    PcapReader = None
    Raw = TCP = UDP = None


def collect_payload_candidates_from_pcap(
    pcap_path: str,
    proto_pref: str,
) -> List[Tuple[bytes, str, int]]:
    """收集 pcap 中符合协议条件的有效 payload 候选。

    返回三元组列表：`(payload, proto, pkt_idx)`，其中 `pkt_idx` 为原始 pcap 包号。
    """
    if PcapReader is None:
        raise RuntimeError("scapy is required for pcap mode but is not available")
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"pcap not found: {pcap_path}")

    tcp_payloads: List[Tuple[bytes, str, int]] = []
    udp_payloads: List[Tuple[bytes, str, int]] = []

    with PcapReader(pcap_path) as reader:
        for pkt_idx, pkt in enumerate(reader):
            if Raw is None or not pkt.haslayer(Raw):
                continue
            payload = bytes(pkt[Raw].load or b"")
            if not payload:
                continue
            if pkt.haslayer(TCP):
                tcp_payloads.append((payload, "tcp", pkt_idx))
            if pkt.haslayer(UDP):
                udp_payloads.append((payload, "udp", pkt_idx))

    if proto_pref == "udp":
        candidates = udp_payloads
    elif proto_pref == "tcp":
        candidates = tcp_payloads
    else:
        candidates = tcp_payloads if tcp_payloads else udp_payloads

    if not candidates:
        raise ValueError("no valid payloads in pcap for requested protocol")
    return candidates


def summarize_pcap_payloads(
    pcap_path: str,
    proto_pref: str,
) -> Dict[str, int]:
    """统计 pcap 原始包数与可作为 full/send 输入的候选包数。"""
    if PcapReader is None:
        raise RuntimeError("scapy is required for pcap mode but is not available")
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"pcap not found: {pcap_path}")

    total_packets = 0
    raw_nonempty = 0
    tcp_raw = 0
    udp_raw = 0

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            total_packets += 1
            if Raw is None or not pkt.haslayer(Raw):
                continue
            payload = bytes(pkt[Raw].load or b"")
            if not payload:
                continue
            raw_nonempty += 1
            if pkt.haslayer(TCP):
                tcp_raw += 1
            if pkt.haslayer(UDP):
                udp_raw += 1

    if proto_pref == "udp":
        usable = udp_raw
    elif proto_pref == "tcp":
        usable = tcp_raw
    else:
        usable = tcp_raw if tcp_raw else udp_raw

    return {
        "total_packets": total_packets,
        "raw_nonempty": raw_nonempty,
        "tcp_raw": tcp_raw,
        "udp_raw": udp_raw,
        "usable": usable,
    }


def sample_payload_from_pcap(pcap_path: str, seed: int, index: int,
                              proto_pref: str) -> Tuple[bytes, str, int]:
    """从 pcap 文件中采样 payload。

    返回值第三项为原始 pcap 包号（pkt_idx），不是候选列表中的位置。
    """
    candidates = collect_payload_candidates_from_pcap(pcap_path, proto_pref)

    if index is not None:
        if index < 0:
            raise IndexError("index must be >= 0 for pcap packet index")
        for payload, proto, pkt_idx in candidates:
            if pkt_idx == index:
                return payload, proto, pkt_idx
        raise IndexError(
            f"pcap packet index {index} not found in valid candidates "
            f"(protocol={proto_pref})"
        )

    rnd = random.Random(seed)
    chosen = rnd.randrange(0, len(candidates))
    payload, proto, pkt_idx = candidates[chosen]
    return payload, proto, pkt_idx


def sample_payload_from_hex(hex_str: str, proto: str) -> Tuple[bytes, str, int]:
    """从 hex 字符串采样 payload"""
    payload = hex_to_bytes(hex_str)
    return payload, proto, 0


def write_sample_metadata(outdir: str, meta: dict) -> None:
    """写入 sample.json"""
    ensure_dir(outdir)
    path = os.path.join(outdir, "sample.json")
    write_json(path, meta)
