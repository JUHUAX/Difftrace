#!/usr/bin/env python3
"""Preprocess replay program logs for RQ2-B groundtruth generation.

For every trace.preprocessed.log under replay_manual_latest/outputs, keep only
the region from the first Taint line through the last Instruction line, then
optionally compress highly repeated execution events. The source files are
never modified.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("/root/semvec/bitfield_groundtruth/replay_manual_latest/outputs")
DEFAULT_OUTPUT_DIR = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/preprocessed_logs"
)
@dataclass
class CompressionStats:
    repeat_summary_count: int = 0
    omitted_repeated_events: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim program logs for RQ2-B.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-compress-repeats",
        action="store_true",
        help="Only trim logs; do not insert RepeatSummary compression markers.",
    )
    parser.add_argument(
        "--repeat-threshold",
        type=int,
        default=20,
        help="Compress an event key only when its occurrence count exceeds this value.",
    )
    parser.add_argument(
        "--repeat-keep-head",
        type=int,
        default=3,
        help="Number of leading occurrences to keep for each compressed key.",
    )
    parser.add_argument(
        "--repeat-keep-tail",
        type=int,
        default=2,
        help="Number of trailing occurrences to keep for each compressed key.",
    )
    return parser.parse_args()


def line_event(line: str) -> str:
    parts = line.rstrip("\n").split("\t")
    return parts[2] if len(parts) >= 3 else ""


def split_line(line: str) -> list[str]:
    return line.rstrip("\n").split("\t")


def protocol_and_pkt(path: Path, input_root: Path) -> tuple[str, str]:
    rel = path.relative_to(input_root)
    protocol = rel.parts[0] if len(rel.parts) >= 3 else "unknown"
    pkt = rel.parts[1] if len(rel.parts) >= 3 else path.parent.name
    return protocol, pkt


def trim_lines(lines: list[str]) -> tuple[list[str], int | None, int | None]:
    first_taint = next((idx for idx, line in enumerate(lines) if line_event(line) == "Taint"), None)
    last_instruction = next(
        (idx for idx in range(len(lines) - 1, -1, -1) if line_event(lines[idx]) == "Instruction"),
        None,
    )
    if first_taint is None or last_instruction is None or first_taint > last_instruction:
        return [], first_taint, last_instruction
    return lines[first_taint : last_instruction + 1], first_taint + 1, last_instruction + 1


def function_name(parts: list[str]) -> str:
    if len(parts) >= 5 and parts[2] == "Function":
        return parts[4]
    return ""


def instruction_payload(parts: list[str]) -> str:
    return parts[3] if len(parts) >= 4 else ""


def instruction_assembly(payload: str) -> str:
    if ": " in payload:
        return payload.split(": ", 1)[1]
    return payload


def instruction_addr(payload: str) -> str:
    if ": " in payload:
        return payload.split(": ", 1)[0]
    return payload


def field_refs(parts: list[str]) -> str:
    return parts[4] if len(parts) >= 5 else ""


def value_info(parts: list[str]) -> str:
    if len(parts) <= 5:
        return ""
    values = [part for part in parts[5:] if part != "LOOP"]
    return "; ".join(values)


def is_loop_line(parts: list[str]) -> bool:
    return any(part == "LOOP" for part in parts[5:])


def shell_quote_value(value: object) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def compression_reason(kind: str, function: str, instruction: str = "") -> str:
    lowered = f"{function} {instruction}".lower()
    if "days_" in lowered or "date" in lowered or "time" in lowered:
        return "date/time conversion or periodic time-state logic repeatedly consumes tainted values"
    if "socket" in lowered or "encapsulation" in lowered or "connectionobject" in lowered:
        return "periodic socket/session maintenance repeatedly consumes tainted state"
    if "stringutils" in lowered or "getcharweight" in lowered or "linkedlist" in lowered:
        return "string/list helper repeatedly executes during collection traversal or sorting"
    if "asn" in lowered or "ber" in lowered or "der" in lowered:
        return "ASN.1/BER/DER encode-decode helper repeatedly processes structured data"
    if "cov" in lowered:
        return "periodic COV state machine repeatedly consumes tainted state"
    if kind == "instruction":
        return "same tainted instruction pattern appears repeatedly"
    if kind == "basicblock":
        return "same basic block appears repeatedly in a loop or recurring execution path"
    return "same helper function event appears repeatedly"


def summarize_values(values: list[str]) -> str:
    compact = [value for value in values if value]
    if not compact:
        return ""
    if compact[0] == compact[-1]:
        return compact[0]
    return f"{compact[0]} ... {compact[-1]}"


def build_repeat_summary(
    *,
    kind: str,
    function: str,
    action: str = "",
    instruction: str = "",
    refs: str = "",
    omitted_count: int,
    total_count: int,
    line_numbers: list[int],
    values: list[str],
    has_loop: bool,
) -> str:
    columns = [
        "THREADID",
        "0",
        "RepeatSummary",
        f"kind={kind}",
    ]
    if function:
        columns.append(f"function={function}")
    if action:
        columns.append(f"action={action}")
    if instruction:
        columns.append(f"instruction={shell_quote_value(instruction)}")
    if refs:
        columns.append(f"field_refs={refs}")
    columns.append(f"repeated={omitted_count}")
    columns.append(f"total_occurrences={total_count}")
    if line_numbers:
        columns.append(f"original_line_range={line_numbers[0]}..{line_numbers[-1]}")
    if has_loop:
        columns.append("loop=true")
    value_summary = summarize_values(values)
    if value_summary:
        columns.append(f"values={shell_quote_value(value_summary)}")
    columns.append(f"reason={shell_quote_value(compression_reason(kind, function, instruction))}")
    return "\t".join(columns) + "\n"


def annotate_lines(lines: list[str]) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    function_stack: list[str] = []
    for line_no, line in enumerate(lines, start=1):
        parts = split_line(line)
        event = parts[2] if len(parts) >= 3 else ""
        current_function = function_stack[-1] if function_stack else ""

        if event == "Function":
            action = parts[3] if len(parts) >= 4 else ""
            name = function_name(parts)
            key = ("function", action, name) if name else None
            annotated.append(
                {
                    "line": line,
                    "line_no": line_no,
                    "event": event,
                    "key": key,
                    "function": name,
                    "action": action,
                    "values": "",
                    "loop": False,
                    "field_refs": "",
                    "instruction": "",
                }
            )
            if action == "enter":
                function_stack.append(name)
            elif action == "exit":
                for idx in range(len(function_stack) - 1, -1, -1):
                    if function_stack[idx] == name:
                        del function_stack[idx:]
                        break
            continue

        key = None
        refs = ""
        asm = ""
        values = ""
        loop = False
        if event == "Instruction":
            payload = instruction_payload(parts)
            asm = instruction_assembly(payload)
            refs = field_refs(parts)
            values = value_info(parts)
            loop = is_loop_line(parts)
            key = ("instruction", current_function, instruction_addr(payload), asm, refs)
        elif event == "BasicBlock":
            payload = parts[3] if len(parts) >= 4 else ""
            key = ("basicblock", current_function, payload)
            asm = payload

        annotated.append(
            {
                "line": line,
                "line_no": line_no,
                "event": event,
                "key": key,
                "function": current_function,
                "action": "",
                "values": values,
                "loop": loop,
                "field_refs": refs,
                "instruction": asm,
            }
        )
    return annotated


def compress_repeated_events(
    lines: list[str],
    *,
    threshold: int,
    keep_head: int,
    keep_tail: int,
) -> tuple[list[str], CompressionStats]:
    if threshold <= 0:
        return lines, CompressionStats()

    annotated = annotate_lines(lines)
    counts = Counter(entry["key"] for entry in annotated if entry["key"] is not None)
    compressible = {key for key, count in counts.items() if count > threshold}
    if not compressible:
        return lines, CompressionStats()

    occurrences: dict[object, int] = defaultdict(int)
    summaries_emitted: set[object] = set()
    metadata: dict[object, dict[str, object]] = {}
    for entry in annotated:
        key = entry["key"]
        if key is None or key not in compressible:
            continue
        bucket = metadata.setdefault(
            key,
            {
                "line_numbers": [],
                "values": [],
                "loop": False,
                "function": entry["function"],
                "action": entry["action"],
                "instruction": entry["instruction"],
                "field_refs": entry["field_refs"],
                "kind": key[0] if isinstance(key, tuple) else "event",
            },
        )
        bucket["line_numbers"].append(entry["line_no"])
        bucket["values"].append(entry["values"])
        bucket["loop"] = bool(bucket["loop"]) or bool(entry["loop"])

    output: list[str] = []
    stats = CompressionStats()
    for entry in annotated:
        key = entry["key"]
        if key is None or key not in compressible:
            output.append(entry["line"])
            continue

        occurrences[key] += 1
        index = occurrences[key]
        count = counts[key]
        keep_current = index <= keep_head or index > count - keep_tail
        if keep_current:
            output.append(entry["line"])
            continue

        stats.omitted_repeated_events += 1
        if key in summaries_emitted:
            continue
        summaries_emitted.add(key)
        bucket = metadata[key]
        omitted_count = max(0, count - keep_head - keep_tail)
        output.append(
            build_repeat_summary(
                kind=str(bucket["kind"]),
                function=str(bucket["function"]),
                action=str(bucket.get("action", "")),
                instruction=str(bucket["instruction"]),
                refs=str(bucket["field_refs"]),
                omitted_count=omitted_count,
                total_count=count,
                line_numbers=list(bucket["line_numbers"]),
                values=list(bucket["values"]),
                has_loop=bool(bucket["loop"]),
            )
        )
        stats.repeat_summary_count += 1

    return output, stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(args.input_root.glob("*/pkt_*/trace.preprocessed.log"))
    manifest_rows: list[dict[str, object]] = []

    for seq, path in enumerate(inputs, start=1):
        protocol, pkt = protocol_and_pkt(path, args.input_root)
        output_name = f"{seq:06d}_{protocol}_{pkt}.log"
        output_path = args.output_dir / output_name
        source_lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        trimmed, first_taint_line, last_instruction_line = trim_lines(source_lines)
        compressed = trimmed
        compression_stats = CompressionStats()
        if trimmed and not args.no_compress_repeats:
            compressed, compression_stats = compress_repeated_events(
                trimmed,
                threshold=args.repeat_threshold,
                keep_head=args.repeat_keep_head,
                keep_tail=args.repeat_keep_tail,
            )
        status = "ok" if trimmed else "no_taint_or_instruction"

        if output_path.exists() and not args.overwrite:
            status = "exists"
        else:
            output_path.write_text("".join(compressed), encoding="utf-8")

        manifest_rows.append(
            {
                "seq": seq,
                "protocol_name": protocol,
                "pkt_id": pkt,
                "source_path": str(path),
                "output_path": str(output_path),
                "source_line_count": len(source_lines),
                "trimmed_line_count": len(trimmed),
                "output_line_count": len(compressed),
                "repeat_summary_count": compression_stats.repeat_summary_count,
                "omitted_repeated_events": compression_stats.omitted_repeated_events,
                "first_taint_line": first_taint_line if first_taint_line is not None else "",
                "last_instruction_line": (
                    last_instruction_line if last_instruction_line is not None else ""
                ),
                "status": status,
            }
        )
        print(
            f"[preprocess] {seq}/{len(inputs)} {protocol}/{pkt}: "
            f"{status}, lines {len(source_lines)} -> {len(trimmed)} -> {len(compressed)}, "
            f"repeat_summaries={compression_stats.repeat_summary_count}, "
            f"omitted={compression_stats.omitted_repeated_events}"
        )

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "seq",
            "protocol_name",
            "pkt_id",
            "source_path",
            "output_path",
            "source_line_count",
            "trimmed_line_count",
            "output_line_count",
            "repeat_summary_count",
            "omitted_repeated_events",
            "first_taint_line",
            "last_instruction_line",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[preprocess] total logs: {len(inputs)}")
    print(f"[preprocess] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
