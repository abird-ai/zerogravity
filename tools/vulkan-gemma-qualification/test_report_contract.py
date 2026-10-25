"""Regression: every Vulkan Gemma producer must satisfy the controller's false permits."""
import importlib.util
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('vulkan_report_controller', HERE / 'controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)

PRODUCERS = (
    ROOT / 'crates/zerogravity/examples/gemma4_vulkan_matrix/run.rs',
    ROOT / 'crates/zerogravity/examples/gemma4_vulkan_final_owner.rs',
    ROOT / 'crates/zerogravity/examples/gemma4_vulkan_same_pci_peer.rs',
)


class VulkanReportContract(unittest.TestCase):
    def test_all_generated_report_shapes_publish_explicit_false_permits(self):
        # This is an early static drift check, not a replacement for physical
        # report validation. Real launch receipts are still mandatory.
        seen = set()
        for path in PRODUCERS:
            source = path.read_text()
            for match in re.finditer(r'"schema"\s*:\s*"(zg\.gemma4-[^"]+)"', source):
                schema = match.group(1)
                self.assertNotIn(schema, seen)
                seen.add(schema)
                self.assertIn(schema, controller.REQUIRED)
                # In the Rust producers the fields occur at the end of each
                # JSON report, before write_new()/serde_json serialization.
                tail = source[match.end():]
                boundary = min(n for n in (tail.find('}),'), tail.find('});')) if n >= 0)
                body = tail[:boundary]
                for field in controller.FALSE_PERMITS:
                    self.assertRegex(body, r'"' + field + r'"\s*:\s*false',
                                     f'{path.name}: {schema} missing {field}')
        self.assertEqual(seen, set(controller.REQUIRED))

    def test_controller_rejects_missing_or_promoted_permits(self):
        good = {key: False for key in controller.FALSE_PERMITS}
        controller.false_permits(good)
        for key in controller.FALSE_PERMITS:
            with self.subTest(key=key):
                missing = dict(good)
                del missing[key]
                with self.assertRaisesRegex(AssertionError, 'missing false-permit'):
                    controller.false_permits(missing)
                promoted = dict(good)
                promoted[key] = True
                with self.assertRaises(AssertionError):
                    controller.false_permits(promoted)


if __name__ == '__main__':
    unittest.main()
