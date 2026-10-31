"""Protocol-level tests; no model downloads and no training."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from argparse import Namespace
import json

import torch

from protocol_data import concat_graphs, select_initial, subset_graph
from prepare_native_data import STAGE_CLASSES, _partition, _encoder_signature, build_stage_partitions, prepare


def graph(n=6, offset=0):
    return {
        "x": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) + 1,
        "edge_index": torch.stack((torch.arange(n), torch.arange(n).roll(-1))),
        "text_emb": torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) + 1,
        "texts": [f"text-{offset + i}" for i in range(n)],
        "weights": torch.arange(1, n + 1, dtype=torch.float32),
        "origin_ids": torch.arange(n, dtype=torch.long) + offset,
        "stage": torch.zeros(n, dtype=torch.long),
    }


class ProtocolDataTests(unittest.TestCase):
    def test_subset_is_induced_and_reindexed(self):
        got = subset_graph(graph(), [4, 1, 2])
        self.assertEqual(got["x"].shape[0], 3)
        self.assertEqual(got["origin_ids"].tolist(), [4, 1, 2])
        self.assertEqual(got["edge_index"].tolist(), [[1], [2]])

    def test_concat_offsets_edges(self):
        a, b = graph(2), graph(3, 10)
        got = concat_graphs([a, b])
        self.assertEqual(got["x"].shape[0], 5)
        self.assertEqual(got["edge_index"].max().item(), 4)
        self.assertEqual(got["texts"], ["text-0", "text-1", "text-10", "text-11", "text-12"])

    def test_selection_is_deterministic_and_conserves_mass(self):
        source = graph(6)
        source["weights"] = torch.tensor([0.5, 1.0, 2.0, 1.5, 3.0, 4.0])
        for mode in ("random", "joint_medoid"):
            one = select_initial(source, 3, seed=9, mode=mode)
            two = select_initial(source, 3, seed=9, mode=mode)
            torch.testing.assert_close(one["x"], two["x"])
            self.assertAlmostEqual(one["weights"].sum().item(), source["weights"].sum().item(), places=6)
            self.assertEqual(len(one["texts"]), 3)
            self.assertEqual(one["origin_ids"].unique().numel(), 3)
            if mode == "joint_medoid":
                torch.testing.assert_close((one["x"] * one["weights"][:, None]).sum(0),
                                           (source["x"] * source["weights"][:, None]).sum(0))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            select_initial(graph(), 2, 0, mode="labels")

    def test_selected_text_embedding_is_exact_encoding_of_real_text(self):
        source = graph()
        source["text_emb"] = torch.randn(6, 4, generator=torch.Generator().manual_seed(2))
        for mode in ("random", "joint_medoid"):
            selected = select_initial(source, 3, 1, mode)
            for row, original_id in enumerate(selected["origin_ids"].tolist()):
                self.assertEqual(selected["texts"][row], source["texts"][original_id])
                torch.testing.assert_close(selected["text_emb"][row], source["text_emb"][original_id], atol=0, rtol=0)
                if mode == "random":
                    torch.testing.assert_close(selected["x"][row], source["x"][original_id], atol=0, rtol=0)

    def test_identical_embeddings_keep_distinct_representatives_and_mass(self):
        source = graph()
        source["x"].zero_()
        source["text_emb"].fill_(1)
        for k in (1, 3, 6):
            for mode in ("random", "joint_medoid"):
                selected = select_initial(source, k, 12, mode)
                self.assertEqual(selected["origin_ids"].unique().numel(), k)
                self.assertTrue((selected["weights"] > 0).all())
                torch.testing.assert_close(selected["weights"].sum(), source["weights"].sum())
                if mode == "joint_medoid":
                    torch.testing.assert_close((selected["x"] * selected["weights"][:, None]).sum(0),
                                               (source["x"] * source["weights"][:, None]).sum(0))

    def test_partition_has_no_holdout_or_cross_stage_training_edges(self):
        n = 140
        labels = [f"class{i}" for i in range(7)]
        source = {
            "node_f": torch.arange(n * 3).reshape(n, 3).float(),
            "text_list": [f"source text {i}" for i in range(n)],
            "label_list": [labels[i % 7] for i in range(n)],
            "all_labels": labels, "labels_desc": ["description"] * 7,
            "original_node_ids": torch.arange(n) + 1000,
            "edge_index": torch.stack((torch.arange(n).repeat(2), torch.cat((torch.arange(n).roll(-1), torch.arange(n).roll(-7))))),
        }
        embedding = torch.randn(n, 4, generator=torch.Generator().manual_seed(19))
        parts, audit = build_stage_partitions(source, embedding)
        all_train_ids = []
        for t, item in enumerate(parts):
            train, evaluation = item["train"], item["eval"]
            self.assertNotIn("y", train)
            self.assertEqual(train["stage"].unique().tolist(), [t])
            self.assertEqual(evaluation["graph"]["stage"].unique().tolist(), [t])
            self.assertEqual(set(evaluation["y"].unique().tolist()), set(STAGE_CLASSES[t]))
            masks = evaluation["masks"]
            self.assertTrue(torch.stack(list(masks.values())).sum(0).eq(1).all())
            train_ids = set(train["origin_ids"].tolist())
            holdout_ids = set(evaluation["origin_ids"][~masks["train"]].tolist())
            self.assertFalse(train_ids & holdout_ids)
            all_train_ids.extend(train_ids)
            for node in train_ids:
                self.assertEqual(_partition(node), "train")
        union = concat_graphs([item["train"] for item in parts])
        edges = union["edge_index"]
        self.assertTrue((union["stage"][edges[0]] == union["stage"][edges[1]]).all())
        self.assertEqual(len(all_train_ids), len(set(all_train_ids)))
        self.assertGreater(audit["cross_stage_edges_dropped"], 0)
        # Reordering source rows must not change any original-ID partition.
        order = torch.randperm(n, generator=torch.Generator().manual_seed(4))
        remap = torch.argsort(order)
        shuffled = {**source, "node_f": source["node_f"][order],
                    "text_list": [source["text_list"][i] for i in order.tolist()],
                    "label_list": [source["label_list"][i] for i in order.tolist()],
                    "original_node_ids": source["original_node_ids"][order],
                    "edge_index": remap[source["edge_index"]]}
        changed, _ = build_stage_partitions(shuffled, embedding[order])
        for first, second in zip(parts, changed):
            for name in ("train", "checkpoint", "test"):
                a, b = first["eval"], second["eval"]
                self.assertEqual(set(a["origin_ids"][a["masks"][name]].tolist()),
                                 set(b["origin_ids"][b["masks"][name]].tolist()))

    def test_prepare_serializes_separate_label_boundary(self):
        n = 70
        labels = [f"class{i}" for i in range(7)]
        source = {"node_f": torch.randn(n, 3),
                  "edge_index": torch.stack((torch.arange(n), torch.arange(n).roll(-7))),
                  "text_list": [f"text{i}" for i in range(n)],
                  "label_list": [labels[i % 7] for i in range(n)],
                  "all_labels": labels, "labels_desc": ["description"] * 7}
        encoded = torch.randn(n, 4)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            torch.save(source, base / "source.pt")
            args = Namespace(source=base / "source.pt", output=base / "output", batch_size=4, split_seed=0)
            shared_class_slice = torch.randn(n + 7, 4)[n:]
            with patch("prepare_native_data._encode_texts", return_value=(encoded, shared_class_slice, {"test_fixture": True}, labels)):
                prepare(args)
            class_emb = torch.load(args.output / "class_text_emb.pt", weights_only=False)
            self.assertEqual(class_emb.shape, (7, 4))
            self.assertEqual(class_emb.untyped_storage().nbytes(), class_emb.numel() * class_emb.element_size())
            self.assertLess((args.output / "class_text_emb.pt").stat().st_size, 4096)
            manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["files"]), 7)
            for t in range(3):
                train = torch.load(args.output / f"stage_{t}_train.pt", weights_only=False)
                evaluation = torch.load(args.output / f"stage_{t}_eval.pt", weights_only=False)
                self.assertEqual(set(train), {"x", "edge_index", "text_emb", "texts", "weights", "origin_ids", "stage"})
                self.assertEqual(evaluation["masks"]["test"].dtype, torch.bool)
                self.assertTrue((train["stage"] == t).all())
                self.assertTrue((evaluation["graph"]["stage"] == t).all())

    def test_encoder_signature_changes_with_model_weight_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}")
            (path / "vocab.txt").write_text("test")
            (path / "pytorch_model.bin").write_bytes(b"weights1")
            _, one = _encoder_signature(directory)
            (path / "pytorch_model.bin").write_bytes(b"weights2")
            _, two = _encoder_signature(directory)
            self.assertNotEqual(one, two)
            self.assertEqual(one["max_length"], 512)
            self.assertTrue(one["eval_mode"])


if __name__ == "__main__":
    unittest.main()
