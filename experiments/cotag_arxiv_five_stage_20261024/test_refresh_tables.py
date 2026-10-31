"""Check protocol separation and aggregation before publishing table updates."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('refresh_tables', HERE / 'refresh_tables.py')
TABLES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TABLES)


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TableTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv = self.root / 'scores.csv'
        self.rows = [dict(arm='uq_dm_edgefree', order=o, seed=c,
                          f1=str(40 + i), accuracy=str(70 + i))
                     for i, (o, c) in enumerate(sorted(TABLES.EXPECTED))]

    def test_requires_all_nine_constructions(self):
        write_rows(self.csv, self.rows[:8])
        self.assertEqual(TABLES.load_complete(self.csv, False), {})

    def test_mean_and_sample_sd_are_over_memory_means(self):
        write_rows(self.csv, self.rows)
        value = TABLES.load_complete(self.csv, False)['uq_dm_edgefree', 'arxiv']
        self.assertEqual(value['memories'], 9)
        self.assertEqual(value['f1'], 44)
        self.assertAlmostEqual(value['f1_sd'], 7.5 ** .5)
        self.assertEqual(value['accuracy'], 74)

    def test_rejects_duplicate_construction(self):
        write_rows(self.csv, self.rows + self.rows[:1])
        with self.assertRaises(AssertionError):
            TABLES.load_complete(self.csv, False)

    def test_rejects_unregistered_seed(self):
        self.rows[0]['seed'] = '123'
        write_rows(self.csv, self.rows)
        with self.assertRaises(AssertionError):
            TABLES.load_complete(self.csv, False)

    def test_old_format_does_not_mix_datasets(self):
        rows = [dict(arm=r['arm'], order=r['order'], seed=r['seed'],
                     final_f1=r['f1'], accuracy=r['accuracy'], dataset='sportsfit')
                for r in self.rows]
        rows.append(dict(rows[0], dataset='bookhis'))
        write_rows(self.csv, rows)
        result = TABLES.load_complete(self.csv, True)
        self.assertEqual(set(result), {('uq_dm_edgefree', 'sportsfit')})

    def test_single_memory_marker(self):
        value = dict(f1=40, f1_sd=1, accuracy=70, accuracy_sd=2, memories=1)
        self.assertIn(r'\dagger', TABLES.metric(value))
        self.assertNotIn(r'\dagger', TABLES.metric(dict(value, memories=9)))
        self.assertEqual(TABLES.metric(None), '--')

    def test_no_three_stage_fallback_in_main_table(self):
        here = self.root / 'experiments/cotag_arxiv_five_stage_20261024'
        paper = self.root / 'paper_www'
        here.mkdir(parents=True)
        (paper / 'tables').mkdir(parents=True)
        values = {a: {d: dict(f1=98.76, f1_sd=1, accuracy=99.87,
                               accuracy_sd=1, memories=9)
                       for d in TABLES.DATASETS} for a, _, _ in TABLES.METHODS}
        (here / 'previous_table_snapshot.json').write_text(
            json.dumps(dict(values=values)), encoding='utf-8')
        with patch.multiple(TABLES, ROOT=self.root, HERE=here, PAPER=paper):
            TABLES.refresh()
        text = (paper / 'tables/main_comparison.tex').read_text(encoding='utf-8')
        method_rows = [line for line in text.splitlines()
                       if any(line.startswith(name + ' &') for _, name, _ in TABLES.METHODS)]
        self.assertEqual(len(method_rows), 11)
        self.assertTrue(all(line.split('&')[1].strip() == '--' for line in method_rows))
        archive = (paper / 'tables/arxiv_three_stage_archive.tex').read_text(encoding='utf-8')
        self.assertIn('98.76', archive)


if __name__ == '__main__':
    unittest.main()
