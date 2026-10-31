import io
import unittest

import prepare_data as P


class PreparationTests(unittest.TestCase):
    def test_tagged_parser_does_not_mix_records(self):
        rows = list(P.acm_records(io.StringIO("#*First title\n#t2001\n#cSIGMOD\n#index0\n#%2\n#!An abstract\n\n#*Second title\n#t2000\n#index2\n")))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["references"], ["2"])
        self.assertEqual(rows[1]["references"], [])

    def test_alias_boundaries(self):
        self.assertEqual(P.labels_for("dblp", "ACL"), ["natural language processing"])
        self.assertEqual(P.labels_for("dblp", "miracle methods"), [])
        self.assertEqual(P.labels_for("acm", "Proceedings of ACM SIGMOD 2006"), ["databases"])

    def test_ambiguous_venues_are_detectable(self):
        self.assertEqual(len(P.labels_for("dblp", "Joint ACL ICML workshop")), 2)

    def test_title_normalization(self):
        self.assertEqual(P.canonical("A &amp; B: model!"), P.canonical("a & b model"))

    def test_stable_namespaced_split(self):
        self.assertEqual(P.stable_id("acm", "1"), P.stable_id("acm", "1"))
        self.assertNotEqual(P.stable_id("acm", "1"), P.stable_id("dblp", "1"))

    def test_training_only_boundaries(self):
        rows = [dict(dataset="acm", id=str(i), year=1990+i%26) for i in range(1000)]
        before = P.temporal_bounds(rows)
        for row in rows:
            if P.stable_id("acm", row["id"]) % 10 in (1, 2, 3):
                row["year"] = 9999
        self.assertEqual(before, P.temporal_bounds(rows))

    def test_single_year_aborts(self):
        with self.assertRaises(ValueError):
            P.temporal_bounds([dict(dataset="acm", id=str(i), year=2000) for i in range(100)])

    def test_edges_and_text_contract(self):
        rows = [dict(id=str(i), title=f"Substantive paper number {i}", abstract="body", year=1990+i%26,
                     label="databases", venue="SIGMOD", references=[]) for i in range(1000)]
        rows[0]["references"] = ["0", "1", "999999"]
        rows[1]["references"] = ["0"]
        final, edges, audit = P.finalize(rows, {}, "acm")
        self.assertEqual(audit["edge_filter_counts"]["forward_year_references_removed"], 1)
        self.assertEqual(audit["edge_filter_counts"]["self_references_removed"], 1)
        self.assertEqual(audit["directed_citations"], 1)
        self.assertEqual(len(edges), 2)
        self.assertTrue(all("SIGMOD" not in r["text"] and "databases" not in r["text"] for r in final))


if __name__ == "__main__":
    unittest.main()
