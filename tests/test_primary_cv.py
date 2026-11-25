"""Integration checks for the selected conditional-CV endpoint."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse
from spatial_membership_audit.primary_cv import estimate_cv, main


class PrimaryCVIntegration(unittest.TestCase):
    def setUp(self):
        # Conditional T equals 0 or 1/2 with equal probability: CV=1.
        self.p = np.array([[1., 0.], [0., .5], [0., 1.]])
        self.g = sparse.csr_matrix([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])

    def test_single_primary_endpoint_and_explicit_tolerance(self):
        with self.assertRaises(TypeError):
            estimate_cv(self.p, self.g)
        r = estimate_cv(self.p, self.g, accuracy_factor=1.25)
        self.assertEqual(r['primary_output'], 'cv')
        self.assertEqual(r['status'], 'RESOLVED')
        self.assertFalse(r['fallback_used'])
        self.assertLessEqual(r['cv_interval'][0], 1.)
        self.assertGreaterEqual(r['cv_interval'][1], 1.)
        self.assertLessEqual(max(r['cv'], 1/r['cv']), 1.25)

    def test_invalid_nonselected_category_and_budget(self):
        # Rounded sum is 1; exact binary sum nevertheless exceeds one.
        p = np.pad(self.p, ((0,0), (0,1)))
        p[0,2] = 1e-30
        with self.assertRaises(ValueError):
            estimate_cv(p, self.g, accuracy_factor=1.25)
        r = estimate_cv(self.p, self.g, accuracy_factor=1.25, max_seconds=1e-12)
        self.assertEqual(r['status'], 'UNRESOLVED')
        self.assertIsNone(r['cv'])
        self.assertFalse(r['fallback_used'])

    def test_command_line_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            np.save(folder/'p.npy', self.p)
            sparse.save_npz(folder/'g.npz', self.g)
            argv = ['--probabilities', str(folder/'p.npy'), '--adjacency', str(folder/'g.npz'),
                    '--accuracy-factor', '1.25', '--out', str(folder/'cv.json')]
            main(argv)
            result = json.loads((folder/'cv.json').read_text())
            self.assertEqual(result['status'], 'RESOLVED')
            original = (folder/'cv.json').read_bytes()
            with self.assertRaises(FileExistsError):
                main(argv)
            self.assertEqual((folder/'cv.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
