"""Regression: ordinary PLE parity must account for already-resident rows."""
import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'payload' / 'physical-runner'))
from fixture_semantic import ROW_BYTES, ordinary_page_traffic


def parity(rows, uploads):
    hits = rows - uploads
    return {'ple_row_uploads': uploads, 'ple_row_reuses': hits,
            'ple_row_h2d': uploads * ROW_BYTES,
            'ple_row_avoided_h2d': hits * ROW_BYTES,
            'value_upload_total': uploads * ROW_BYTES,
            'producer_kv_d2h': 0}


class OrdinaryRowReuse(unittest.TestCase):
    def test_cold_and_warm_unique(self):
        self.assertEqual(ordinary_page_traffic(parity(3, 3), [1, 2, 3]), (3, 0))
        self.assertEqual(ordinary_page_traffic(parity(3, 0), [1, 2, 3]), (0, 3))

    def test_repeated_one_may_start_cold_or_hot(self):
        self.assertEqual(ordinary_page_traffic(parity(57, 1), [4906] * 57), (1, 56))
        self.assertEqual(ordinary_page_traffic(parity(57, 0), [4906] * 57), (0, 57))

    def test_missing_contiguous_reuses_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'contiguous row reuse'):
            ordinary_page_traffic(parity(57, 57), [4906] * 57)

    def test_incorrect_real_h2d_bytes_are_rejected(self):
        event = parity(57, 1)
        event['ple_row_h2d'] += ROW_BYTES
        with self.assertRaisesRegex(ValueError, 'row bytes'):
            ordinary_page_traffic(event, [4906] * 57)


class ControlSourceScoping(unittest.TestCase):
    def test_seams_must_not_shadow_outer_prepared_payload_path(self):
        # This control was accidentally skipped by an UnboundLocalError after
        # the negative JSON cases bound `payload` as a local byte string.
        source = (Path(__file__).parent / 'control-checks.py').read_text()
        functions = [node for node in ast.walk(ast.parse(source))
                     if isinstance(node, ast.FunctionDef) and node.name == 'seams']
        self.assertEqual(len(functions), 1)
        assigned = {node.id for node in ast.walk(functions[0])
                    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))}
        self.assertNotIn('payload', assigned, 'source control payload path shadowed')


if __name__ == '__main__':
    unittest.main()
