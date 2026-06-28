#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
send.py - 选包并发送 payload
从 pcap/hex 选取 payload，写入 sample.json，发送到目标
"""

import argparse
import os
import sys
import time

from common import (
    DEFAULT_OUTDIR,
    DEFAULT_PROTO,
    DEFAULT_RECV,
    DEFAULT_RECV_TIMEOUT,
    DEFAULT_SEED,
    DEFAULT_TARGET_HOST,
    DEFAULT_TARGET_PORT,
    DEFAULT_PIN_LOG,
    DEFAULT_WAIT_MS,
    bytes_to_hex,
    ensure_dir,
    hex_to_bytes,
    load_sample,
    print_user_error,
    read_json,
    resolve_proto,
    sample_payload_from_hex,
    sample_payload_from_pcap,
    send_payload,
    slice_log,
    wait_for_log_growth,
    write_json,
    write_sample_metadata,
)


def cmd_send(args: argparse.Namespace) -> None:
    """send 子命令主逻辑"""
    ensure_dir(args.outdir)

    if args.mutations:
        if not os.path.exists(args.pin_log):
            raise FileNotFoundError(f"pin log not found: {args.pin_log}")

        data = read_json(args.mutations)
        if isinstance(data, dict):
            selections = [data]
        elif isinstance(data, list):
            selections = data
        else:
            raise ValueError("mutations.json 格式错误")
        if not selections:
            raise ValueError("mutations.json 中没有可用的 mutations")

        if args.proto == "auto":
            sample = load_sample(args.outdir)
            proto = resolve_proto(args.proto, sample.get("proto", ""))
        else:
            proto = args.proto

        def safe_name(value: str) -> str:
            cleaned = []
            for ch in value:
                if ch.isalnum() or ch in {"_", "-"}:
                    cleaned.append(ch)
                else:
                    cleaned.append("_")
            return "".join(cleaned) or "item"

        for selection_index, selection in enumerate(selections, start=1):
            if not isinstance(selection, dict):
                raise ValueError("mutations.json 里存在无效条目")
            field = selection.get("field")
            mutations = selection.get("mutations")
            if not isinstance(mutations, list) or not mutations:
                raise ValueError("mutations.json 中缺少 mutations 列表")
            if isinstance(field, dict):
                field_repr = field.get("repr")
                if not field_repr and "a" in field and "b" in field:
                    field_repr = f"{field['a']},{field['b']}"
            else:
                field_repr = None
            if not field_repr:
                field_repr = f"field_{selection_index}"

            field_dir_name = safe_name(field_repr.replace(",", "_"))
            field_dir = os.path.join(args.outdir, "mutations", field_dir_name)
            ensure_dir(field_dir)

            for idx, mut in enumerate(mutations, start=1):
                if isinstance(mut, dict):
                    payload_hex = mut.get("payload_hex")
                    mutation_name = mut.get("name")
                elif isinstance(mut, str):
                    payload_hex = mut
                    mutation_name = None
                else:
                    payload_hex = None
                    mutation_name = None
                if not payload_hex:
                    raise ValueError("mutation 缺少 payload_hex")
                payload = hex_to_bytes(payload_hex)
                name_part = safe_name(str(mutation_name)) if mutation_name else f"{args.role_prefix}{idx}"
                file_prefix = f"{idx:03d}_{name_part}"
                log_path = os.path.join(field_dir, f"{file_prefix}.log")
                meta_path = os.path.join(field_dir, f"{file_prefix}.json")

                size_before = os.path.getsize(args.pin_log)
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

                slice_len = slice_log(args.pin_log, size_before, size_after, log_path)
                if slice_len == 0:
                    raise RuntimeError("log slice is empty; check target or pin log output")

                meta = {
                    "field": field,
                    "field_repr": field_repr,
                    "mutation_index": idx,
                    "mutation_name": mutation_name or "",
                    "proto": proto,
                    "target": {"host": args.target_host, "port": args.target_port},
                    "payload_hex": bytes_to_hex(payload),
                    "pin_log": args.pin_log,
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
        return

    if args.payload_hex:
        # 直接使用 --payload-hex
        payload = hex_to_bytes(args.payload_hex)
        if args.proto == "auto":
            raise ValueError("--payload-hex 模式下必须指定 --proto tcp|udp")
        proto = args.proto
        meta = {
            "mode": "hex",
            "hex": args.payload_hex,
            "seed": args.seed,
            "index": 0,
            "proto": proto,
            "payload_hex": bytes_to_hex(payload),
            "role": args.role,
            "timestamp": time.time(),
        }
    else:
        if not args.mode:
            raise ValueError("请提供 --mode pcap|hex 或直接使用 --payload-hex")
        if args.mode == "pcap":
            if not args.pcap:
                raise ValueError("pcap 模式下必须提供 --pcap")
            payload, proto, chosen_pkt_index = sample_payload_from_pcap(
                args.pcap, args.seed, args.index, args.proto
            )
            meta = {
                "mode": "pcap",
                "pcap": args.pcap,
                "seed": args.seed,
                "index": chosen_pkt_index,
                "proto": proto,
                "payload_hex": bytes_to_hex(payload),
                "role": args.role,
                "timestamp": time.time(),
            }
        else:
            if not args.hex:
                raise ValueError("hex 模式下必须提供 --hex")
            if args.proto == "auto":
                raise ValueError("hex 模式下必须指定 --proto tcp|udp")
            payload, proto, chosen_index = sample_payload_from_hex(args.hex, args.proto)
            meta = {
                "mode": "hex",
                "hex": args.hex,
                "seed": args.seed,
                "index": chosen_index,
                "proto": proto,
                "payload_hex": bytes_to_hex(payload),
                "role": args.role,
                "timestamp": time.time(),
            }

    write_sample_metadata(args.outdir, meta)

    if meta.get("mode") == "pcap":
        print(f"[send] pcap_packet_index={meta.get('index')} proto={proto}")
    else:
        print(f"[send] mode={meta.get('mode')} proto={proto}")
    print(f"[send] payload_hex={bytes_to_hex(payload)}")

    info = send_payload(
        payload=payload,
        proto=proto,
        host=args.target_host,
        port=args.target_port,
        recv=args.recv,
        recv_timeout=args.recv_timeout,
    )
    if args.recv:
        recv_bytes = int(info.get("recv_bytes", "0"))
        preview = info.get("recv_preview_hex", "")
        if recv_bytes > 0:
            print(f"[recv] {recv_bytes} bytes: {preview}...")
        else:
            print("[recv] 0 bytes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="选包并发送 payload，写入 sample.json"
    )
    parser.add_argument("--mode", choices=["pcap", "hex"], default="",
                        help="数据来源模式")
    parser.add_argument("--pcap", help="pcap 文件路径")
    parser.add_argument("--hex", dest="hex", help="hex payload 字符串")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="随机种子")
    parser.add_argument("--index", type=int,
                        help="指定原始 pcap 包号（pkt_idx）")
    parser.add_argument("--target-host", default=DEFAULT_TARGET_HOST,
                        help="目标主机")
    parser.add_argument("--target-port", type=int, default=DEFAULT_TARGET_PORT,
                        help="目标端口")
    parser.add_argument("--proto", choices=["auto", "tcp", "udp"],
                        default=DEFAULT_PROTO, help="协议类型")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="输出目录")
    parser.add_argument("--payload-hex",
                        help="直接指定 payload hex（覆盖 --mode）")
    parser.add_argument("--mutations",
                        help="mutations.json 路径，逐个发送并切片")
    parser.add_argument("--pin-log", default=DEFAULT_PIN_LOG,
                        help="pin 日志路径")
    parser.add_argument("--wait-ms", type=int, default=DEFAULT_WAIT_MS,
                        help="发送后等待日志稳定的毫秒数")
    parser.add_argument("--recv", action="store_true", default=DEFAULT_RECV,
                        help="是否接收响应")
    parser.add_argument("--recv-timeout", type=float, default=DEFAULT_RECV_TIMEOUT,
                        help="接收超时时间（秒）")
    parser.add_argument("--role", default="baseline",
                        help="角色标识（如 baseline, mutated）")
    parser.add_argument("--role-prefix", default="mut",
                        help="mutations 发送时的 role 前缀")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        cmd_send(args)
    except FileNotFoundError as exc:
        print_user_error(str(exc))
        sys.exit(1)
    except ValueError as exc:
        print_user_error(str(exc))
        sys.exit(1)
    except RuntimeError as exc:
        print_user_error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
