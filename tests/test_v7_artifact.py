"""Synthetic checks only: no bundled captures, annotations, results, or API calls."""

import importlib.util
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from boundary_metrics import (average_metrics, bit_metrics, byte_metrics,
                              interval_metrics, merge_metrics, packet_paper_metrics)
from update_manifest import manifest_text


def load_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cost = load_module("experiments/RQ4/measure_rq4_llm_usage.py", "artifact_cost")
sota = load_module("tools/sota_evaluation/scripts/evaluate_boundary_predictions.py", "artifact_sota")
pg = load_module("tools/program_log/scripts/evaluate_program_log_field_boundary.py", "artifact_pg")
tg = load_module("tools/tshark/field_boundary/evaluate_tshark_vs_experiment.py", "artifact_tg")
rq3 = load_module("experiments/RQ3/evaluate_rq3_bitfield_ablation.py", "artifact_rq3")


class BoundaryMetricsTests(unittest.TestCase):
    def test_identical_byte_partition(self):
        result = byte_metrics({(0, 1), (2, 3)}, {(0, 1), (2, 3)}, 4)
        self.assertEqual((result["tp"], result["tn"], result["candidate_positions"]), (3, 2, 5))
        for key in ("accuracy", "f1", "perfection"):
            self.assertEqual(result[key], 1.0)

    def test_extra_interior_boundary_breaks_perfection(self):
        result = byte_metrics({(0, 3)}, {(0, 1), (2, 3)}, 4)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["accuracy"], 0.8)
        self.assertEqual(result["f1"], 0.8)
        self.assertEqual(result["perfection"], 0.0)

    def test_missing_boundary_breaks_both_fields(self):
        result = byte_metrics({(0, 1), (2, 3)}, {(0, 3)}, 4)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["perfection"], 0.0)

    def test_empty_prediction_is_not_perfect(self):
        result = byte_metrics({(0, 3)}, set(), 4)
        self.assertEqual((result["tp"], result["tn"], result["fn"]), (0, 3, 2))
        self.assertEqual(result["accuracy"], 0.6)
        self.assertEqual(result["perfection"], 0.0)

    def test_perfection_uses_boundaries_not_exact_intervals(self):
        result = byte_metrics({(0, 1), (2, 3)}, {(0, 3), (0, 1)}, 4)
        self.assertEqual(result["perfection"], 1.0)

    def test_invalid_range_fails(self):
        with self.assertRaises(ValueError):
            interval_metrics({(0, 5)}, set(), 4)

    def test_bit_endpoints_and_extra_boundary(self):
        reference = {(0, 0): {"[7:4]", "[3:0]"}}
        predicted = {(0, 0): {"[7:4]", "[3:2]", "[1:0]"}}
        result = bit_metrics(reference, predicted)
        self.assertEqual((result["tp"], result["tn"], result["fp"]), (3, 5, 1))
        self.assertEqual(result["candidate_positions"], 9)
        self.assertEqual(result["perfection"], 0.5)
        self.assertAlmostEqual(result["f1"], 6 / 7)

    def test_prediction_only_bit_parent_retains_false_positives(self):
        result = bit_metrics({(0, 0): {"[7:4]"}}, {(1, 1): {"[7:4]"}})
        self.assertEqual((result["fp"], result["fn"]), (2, 2))
        self.assertEqual(result["candidate_positions"], 18)

    def test_no_bit_reference_is_not_in_protocol_average(self):
        empty = bit_metrics({}, {})
        valid = bit_metrics({(0, 0): {"[7:4]"}}, {(0, 0): {"[7:4]"}})
        self.assertIsNone(empty["accuracy"])
        result = average_metrics([empty, valid])
        self.assertEqual(result["protocol_count"], 1)
        self.assertEqual(result["accuracy"], 1)

    def test_micro_and_protocol_average_differ(self):
        small = byte_metrics({(0, 0)}, set(), 1)
        large = byte_metrics({(0, 7)}, {(0, 7)}, 8)
        self.assertNotEqual(average_metrics([small, large])["accuracy"],
                            merge_metrics([small, large])["accuracy"])

    def test_sota_new_and_legacy_metrics(self):
        row = {"method": "toy", "variant": "default", "protocol": "toy", "sample_id": "pkt_0000",
               "status": "ok", "payload_length": 4, "fields": [(0, 1), (2, 3)]}
        result = sota.packet_metrics(row, {(0, 3)})
        summary = sota.summarize([result], "toy")
        self.assertEqual(summary["exact"]["f1"], 0)
        self.assertEqual(sota.flat(summary, "toy", "default")["f1_score"], 0.8)

    def test_pg_and_tg_protocol_summaries_share_calculation(self):
        fields, bits = {(0, 0)}, {(0, 0): {"[7:4]", "[3:0]"}}
        packet = {"paper_metrics": packet_paper_metrics(fields, fields, bits, bits, 1),
                  "field_boundary": pg.set_metrics(fields, fields).to_dict(),
                  "field_boundary_hit": pg.set_metrics({0, 1}, {0, 1}).to_dict(),
                  "bitfield_detection": pg.set_metrics({(0, 0)}, {(0, 0)}).to_dict(),
                  "bitfield_boundary": pg.boundary_metrics(bits, bits),
                  "bitfield_subfield_boundary_hit": pg.set_metrics({0, 4, 8}, {0, 4, 8}).to_dict()}
        self.assertEqual(pg.summarize_packets("toy", [packet])["paper_metrics"],
                         tg.summarize_packets("toy", [packet])["paper_metrics"])

    def test_rq3_rejects_old_summary_instead_of_relabelling(self):
        with self.assertRaises(ValueError):
            rq3.metric_values({"overall": {}})

    def test_pg_cli_and_rq3_metrics(self):
        # Fixture files live in a temporary directory, never in the artifact.
        with tempfile.TemporaryDirectory(prefix="difftrace-v7-test-") as temporary:
            base = Path(temporary)
            packet = base / "replay/toy/pkt_0000"
            packet.mkdir(parents=True)
            (packet / "meta.json").write_text(json.dumps({"payload_hex": "00"}))
            (packet / "fields.json").write_text(json.dumps({"fields": [{"a": 0, "b": 0}]}))
            (packet / "bitfields.json").write_text(json.dumps({"fields": [{"field_id": "0",
                "subfields": [{"label": "[7:4]"}, {"label": "[3:0]"}]}]}))
            truth = base / "reference.jsonl"
            truth.write_text("\n".join(json.dumps({"protocol_name": "toy", "sample_id": "pkt_0000",
                "field_id": field}) for field in ("b:0:0", "bit:0:0:4:7", "bit:0:0:0:3")))
            output = base / "metrics"
            subprocess.run([sys.executable, str(ROOT / "tools/program_log/scripts/evaluate_program_log_field_boundary.py"),
                "--groundtruth-jsonl", str(truth), "--replay-root", str(base / "replay"),
                "--outdir", str(output), "--groundtruth-md", str(base / "reference.md"),
                "--compare-md", str(base / "comparison.md")], check=True, capture_output=True)
            summary = json.loads((output / "field_boundary_metrics_summary.json").read_text())
            values = rq3.metric_values(summary)
            self.assertEqual((values["accuracy"], values["f1_score"], values["perfection"]), (1, 1, 1))
            self.assertTrue((output / "field_boundary_v7_metrics.csv").exists())

    def test_tg_cli(self):
        with tempfile.TemporaryDirectory(prefix="difftrace-tg-test-") as temporary:
            base = Path(temporary)
            packet = base / "replay/bacnet/pkt_0000"
            packet.mkdir(parents=True)
            (packet / "meta.json").write_text(json.dumps({"payload_hex": "00", "packet_index": 1}))
            (packet / "fields.json").write_text(json.dumps({"fields": [{"a": 0, "b": 0}]}))
            (packet / "bitfields.json").write_text(json.dumps({"fields": [{"field_id": "0",
                "subfields": [{"label": "[7:4]"}, {"label": "[3:0]"}]}]}))
            reference = base / "reference"
            reference.mkdir()
            (reference / "bacnet.json").write_text(json.dumps({"packets": [{"packet_index": 1,
                "selected_protocols": ["bacnet"], "protocols": [{"protocol": "bacnet", "offset": 0, "length": 1}],
                "fields": [{"protocol": "bacnet", "field_offset": 0, "field_length": 1,
                            "bit_ranges": ["[7:4]", "[3:0]"]}]}]}))
            output = base / "metrics"
            completed = subprocess.run([sys.executable, str(ROOT / "tools/tshark/field_boundary/evaluate_tshark_vs_experiment.py"),
                "--gt-root", str(reference), "--protocol", "bacnet", "--replay-root", str(base / "replay"),
                "--outdir", str(output), "--groundtruth-md", str(base / "reference.md"),
                "--compare-md", str(base / "comparison.md")], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((output / "metrics_summary.json").read_text())
            self.assertEqual(summary["overall"]["paper_metrics"]["bit"]["perfection"], 1)

    def test_sota_cli(self):
        with tempfile.TemporaryDirectory(prefix="difftrace-sota-test-") as temporary:
            base = Path(temporary)
            reference = base / "reference.jsonl"
            reference.write_text(json.dumps({"protocol_name": "toy", "sample_id": "pkt_0000", "field_id": "b:0:3"}))
            predictions = base / "predictions.jsonl"
            predictions.write_text(json.dumps({"method": "toy", "variant": "default", "protocol": "toy",
                "sample_id": "pkt_0000", "status": "ok", "payload_length": 4, "fields": [[0, 1], [2, 3]]}))
            output = base / "metrics"
            subprocess.run([sys.executable, str(ROOT / "tools/sota_evaluation/scripts/evaluate_boundary_predictions.py"),
                "--groundtruth", str(reference), "--predictions", str(predictions), "--outdir", str(output)],
                check=True, capture_output=True)
            self.assertIn("0.8000 | 0.8000 | 0.0000", (output / "boundary_metrics_summary.md").read_text())


class CostAndManifestTests(unittest.TestCase):
    def test_default_sample_size(self):
        with patch.object(sys, "argv", ["measure"]):
            self.assertEqual(cost.parse_args().sample_size, 100)

    def test_per_field_average_and_total(self):
        rows = [{"method": "full", "prompt_tokens": 10, "completion_tokens": 6,
                 "reasoning_tokens": 4, "total_tokens": 16, "elapsed_seconds": 2},
                {"method": "full", "prompt_tokens": 30, "completion_tokens": 10,
                 "reasoning_tokens": 8, "total_tokens": 40, "elapsed_seconds": 4}]
        result = cost.summarize_usage(rows)["full"]
        self.assertEqual(result["avg_prompt_tokens"], 20)
        self.assertEqual(result["avg_reasoning_tokens"], 6)
        self.assertEqual(result["avg_total_tokens"], 28)
        self.assertEqual(result["total_tokens"], 56)
        self.assertEqual(result["avg_elapsed_seconds"], 3)

    def test_nested_reasoning_counter(self):
        row = {}
        cost.flatten_usage("", {"completion_tokens_details": {"reasoning_tokens": 12}}, row)
        self.assertEqual(row["completion_tokens_details.reasoning_tokens"], 12)

    def test_missing_usage_is_unknown(self):
        result = cost.summarize_usage([{"method": "full"}], dry_run=True)
        self.assertIsNone(result["full"]["avg_total_tokens"])
        self.assertIn("N/A", cost.cost_markdown(result))

    def test_cost_cli_dry_run_and_insufficient_sample(self):
        direct = load_module("experiments/RQ4/run_rq4_no_latent_direct_summary.py", "artifact_direct")
        with tempfile.TemporaryDirectory(prefix="difftrace-cost-test-") as temporary:
            base = Path(temporary)
            full_input = base / "full.jsonl"
            no_input = base / "direct.csv"
            rows = [{"protocol_name": "toy", "sample_id": "pkt_0000", "field_id": f"b:{i}:{i}"}
                    for i in range(2)]
            full_input.write_text("\n".join(json.dumps(row) for row in rows))
            columns = direct.CONTEXT_COLS + [f"{group}_{feature}" for group in direct.GROUPS for feature in direct.GROUP_FEATURES]
            with no_input.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) + columns)
                writer.writeheader()
                writer.writerows({**row, **{column: 0 for column in columns}} for row in rows)
            command = [sys.executable, str(ROOT / "experiments/RQ4/measure_rq4_llm_usage.py"),
                       "--full-input", str(full_input), "--no-latent-input", str(no_input),
                       "--output-dir", str(base / "metrics"), "--dry-run"]
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("requested 100 shared fields, but only 2 available", rejected.stderr)
            subprocess.run(command + ["--sample-size", "2"], check=True, capture_output=True)
            summary = json.loads((base / "metrics/rq4_llm_usage_measurement_summary.json").read_text())
            self.assertEqual(summary["full_field_fusion"]["sampled_fields"], 2)
            self.assertEqual(summary["no_latent_direct"]["measurement"], "dry_run")
            self.assertIsNone(summary["no_latent_direct"]["avg_total_tokens"])

    def test_manifest_excludes_git_and_cache(self):
        with tempfile.TemporaryDirectory(prefix="difftrace-manifest-test-") as temporary:
            base = Path(temporary)
            (base / ".git").mkdir()
            (base / ".git/config").touch()
            (base / "__pycache__").mkdir()
            (base / "__pycache__/example.pyc").touch()
            (base / "example.py").touch()
            self.assertEqual(manifest_text(base), "./example.py\n")


if __name__ == "__main__":
    unittest.main()
