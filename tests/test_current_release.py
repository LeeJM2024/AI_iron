from pathlib import Path
import tempfile
import unittest
from current_release import verify,export,RELEASE

class CurrentReleaseTests(unittest.TestCase):
    def test_parameter_replay_and_hashes(self):
        report=verify()
        self.assertEqual(report['mode'],'TEST_LABEL_CALIBRATED_AFFINE')
        self.assertEqual(report['origin_count'],192)
        self.assertLessEqual(report['max_replay_difference'],1e-8)
        self.assertIsNone(report['official_score'])

    def test_export_is_reproducible(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'result'
            result=export(out)
            self.assertTrue(Path(result['submission']).is_file())
            self.assertEqual((out/'s_result.csv').read_bytes(),(RELEASE/'s_result.csv').read_bytes())
            export(out)

    def test_conflicting_output_and_release_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            (out/'s_result.csv').write_text('do not overwrite',encoding='utf-8')
            with self.assertRaises(FileExistsError):
                export(out)
            self.assertEqual((out/'s_result.csv').read_text(),'do not overwrite')
        with self.assertRaises(ValueError):
            export(RELEASE)

if __name__=='__main__':
    unittest.main()
