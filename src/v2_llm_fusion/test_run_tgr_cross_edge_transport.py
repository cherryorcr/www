import json
import tempfile
import unittest
from pathlib import Path

import torch

import run_tgr_cross_edge_transport as R


class CrossEdgeRunnerHelpersTest(unittest.TestCase):
    def test_missing_parent_provenance_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping, source = R._load_parent_provenance(Path(tmp), 4)
        self.assertIsNone(mapping)
        self.assertEqual(source, "missing_build_audit")

    def test_suffix_provenance_is_shifted_to_parent_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "build_audit.json"
            path.write_text(json.dumps({
                "structural_audit": {"records": [
                    {"source_original_ids": [10]},
                    {"source_original_ids": [11]},
                ]}
            }), encoding="utf-8")
            mapping, _ = R._load_parent_provenance(Path(tmp), 5)
        self.assertEqual(mapping, {10: 3, 11: 4})


if __name__ == "__main__":
    unittest.main()

