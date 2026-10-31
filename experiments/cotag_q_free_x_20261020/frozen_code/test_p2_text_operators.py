import unittest

import torch
import torch.nn.functional as F

import p2_text_operators as T


class Tokenizer:
    def __init__(self):
        self.words = {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self.words:
                self.words[word] = len(self.words) + 1
            ids.append(self.words[word])
        return [0] + ids + [0] if add_special_tokens else ids

    def decode(self, ids, **kwargs):
        words = {v: k for k, v in self.words.items()}
        return " ".join(words[i] for i in ids)


class Encoder:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.inputs = []

    def encode(self, texts):
        self.inputs.extend(texts)
        rows = []
        for text in texts:
            ids = self.tokenizer.encode(text)
            rows.append(torch.tensor([1., len(ids), sum(ids) % 19, sum(ids) % 11]))
        return F.normalize(torch.stack(rows), dim=1)


class TokenControlTests(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder()
        self.texts = [" ".join(f"doc{i}word{j}" for j in range(100)) for i in range(8)]
        torch.manual_seed(1)
        self.source = F.normalize(torch.randn(8, 4), dim=1)
        self.joint = torch.randn(8, 8)
        self.members = torch.arange(8)

    def build(self, operator, **kwargs):
        return T.construct(self.joint, self.source, self.members, self.texts,
                           self.encoder, operator=operator, **kwargs)

    def test_every_operator_respects_actual_input_cap(self):
        for operator in T.OPERATORS:
            for cap in (1, 7, 32, 64, 128):
                with self.subTest(operator=operator, cap=cap):
                    self.encoder.inputs = []
                    text, emb, audit = self.build(operator, budget=cap)
                    actual = sum(T.token_count(self.encoder.tokenizer, x)
                                 for x in self.encoder.inputs)
                    self.assertLessEqual(actual, cap)
                    self.assertEqual(actual, audit["encoder_content_tokens"])
                    self.assertLessEqual(T.token_count(self.encoder.tokenizer, text), cap)
                    self.assertTrue(torch.isfinite(emb).all())

    def test_pooling_never_reuses_full_source_embedding(self):
        for operator in ("mean_pool", "weighted_pool", "set_pool", "no_reencode"):
            first = self.build(operator)[1]
            self.source = torch.randn_like(self.source) * 100
            second = self.build(operator)[1]
            self.assertTrue(torch.equal(first, second), operator)

    def test_mean_and_no_reencode_are_declared_equivalent(self):
        a = self.build("mean_pool")
        b = self.build("no_reencode")
        self.assertEqual(a[0], b[0])
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertEqual(a[2]["input_sha256"], b[2]["input_sha256"])

    def test_short_sources_redistribute_budget(self):
        parts = T.bounded_parts(self.encoder.tokenizer, ["a", "b c d e f"], 5)
        self.assertEqual([T.token_count(self.encoder.tokenizer, x) for x in parts], [1, 4])

    def test_invalid_budget_rejected(self):
        for cap in (0, 241):
            with self.assertRaises(ValueError):
                self.build("concat", budget=cap)

    def test_no_text_is_zero_target(self):
        text, emb, audit = self.build("no_text")
        self.assertEqual(text, "")
        self.assertEqual(audit["encoder_sequences"], 0)
        self.assertEqual(float(emb.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
