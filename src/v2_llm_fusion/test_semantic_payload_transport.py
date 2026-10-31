import unittest

import torch

from semantic_payload_transport import (
    PAYLOAD_KEYS,
    apply_payload_permutation,
    payload_multiset_digest,
    validate_bijection,
)


def make_block(n: int = 4) -> dict:
    return {
        "x": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
        "text_emb": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) + 100,
        "texts": [f"text-{i}" for i in range(n)],
        "mass": torch.arange(n, dtype=torch.float32) + 1,
        "depth": torch.arange(n, dtype=torch.long) + 10,
        "stage": torch.full((n,), 2, dtype=torch.long),
    }


class SemanticPayloadTransportTest(unittest.TestCase):
    def test_complete_transport_moves_all_fields_together(self):
        source = make_block()
        mapping = {0: 2, 1: 0, 2: 3, 3: 1}
        output = apply_payload_permutation(source, mapping)
        for source_row, target_row in mapping.items():
            for key in PAYLOAD_KEYS:
                if isinstance(source[key], torch.Tensor):
                    self.assertTrue(torch.equal(output[key][target_row], source[key][source_row]))
                else:
                    self.assertEqual(output[key][target_row], source[key][source_row])

    def test_payload_multiset_is_conserved(self):
        source = make_block()
        output = apply_payload_permutation(source, {0: 3, 1: 2, 2: 1, 3: 0})
        self.assertEqual(payload_multiset_digest(source), payload_multiset_digest(output))

    def test_invalid_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_bijection({0: 0, 1: 0}, 2)
        with self.assertRaises(ValueError):
            apply_payload_permutation(make_block(), {0: 1, 1: 0}, validate=True)

    def test_identity_is_content_preserving(self):
        source = make_block()
        output = apply_payload_permutation(source, {i: i for i in range(4)})
        self.assertEqual(payload_multiset_digest(source), payload_multiset_digest(output))
        for key in PAYLOAD_KEYS:
            if isinstance(source[key], torch.Tensor):
                self.assertTrue(torch.equal(source[key], output[key]))
            else:
                self.assertEqual(source[key], output[key])


if __name__ == "__main__":
    unittest.main()
