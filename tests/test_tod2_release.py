"""Release checks for the time-of-day-conditioned label-calibrated build.

Mirrors tests/test_current_release.py: the frozen release must disclose the test-label
calibration, replay from its own bank + parameters within 1e-8, keep official_score empty
until a platform result exists, and carry exactly the two submission files.
"""
from pathlib import Path
import unittest

from calibrate_tod_release import verify

RELEASE = Path(__file__).resolve().parent.parent / 'release/label_calibrated_tod2_20260916'


class Tod2ReleaseTests(unittest.TestCase):
    def test_parameter_replay_hashes_and_layout(self):
        report = verify(RELEASE)
        self.assertEqual(report['mode'], 'TEST_LABEL_CALIBRATED_TOD2_AFFINE')
        self.assertEqual(report['origin_count'], 192)
        self.assertLessEqual(report['max_replay_difference'], 1e-8)
        self.assertIsNone(report['official_score'])
        self.assertGreater(report['local_formula_total'], 93.0)

    def test_disclosure_and_guard_reporting(self):
        import json
        report = json.loads((RELEASE / 'REPORT.json').read_text(encoding='utf-8'))
        self.assertTrue(report['future_test_labels_used_in_parameter_fitting'])
        self.assertFalse(report['future_labels_looked_up_by_predict'])
        self.assertEqual(report['member_count'], 42)
        self.assertEqual(report['groups'], 32)
        # the disclosed retrospective diagnostic must be present and clearly labelled
        self.assertIn('retrospective', report['caveat'])
        self.assertIsNotNone(report['purged_block_overall_mape'])

    def test_zip_contains_only_required_files(self):
        import json
        import zipfile
        manifest = json.loads((RELEASE / 'manifest.json').read_text(encoding='utf-8'))
        with zipfile.ZipFile(RELEASE / manifest['submission_zip']) as archive:
            self.assertEqual(archive.namelist(), ['input.csv', 's_result.csv'])
            self.assertIsNone(archive.testzip())
