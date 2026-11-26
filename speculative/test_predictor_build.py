"""Regression tests for the RouteCrack predictor build path.

The C++ sources use C++17 features (structured bindings, init-captures,
brace-init of aggregate temporaries), so both build entry points must pass
-std=c++17. Before the fix they defaulted to the compiler's C++98 mode and
failed with 16 compile errors; build.sh then printed "Built:" unconditionally
even though no library had been produced.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CPP = _REPO / "speculative" / "cpp"
_LIB = _CPP / "libroutecrack.so"


class PredictorBuildTest(unittest.TestCase):
    def test_build_script_requests_cxx17_and_checks_status(self):
        script = (_CPP / "build.sh").read_text(encoding="utf-8")
        self.assertIn("-std=c++17", script, "build.sh must request C++17")
        self.assertIn("set -e", script, "build.sh must fail when g++ fails")

    def test_python_builder_requests_cxx17(self):
        source = (_REPO / "speculative" / "predictor.py").read_text(encoding="utf-8")
        self.assertIn("-std=c++17", source, "_build_lib must request C++17")

    @unittest.skipUnless(shutil.which("g++"), "g++ not available")
    def test_predictor_autobuild_produces_a_usable_library(self):
        """End-to-end: remove the cached .so and let Predictor() rebuild it."""
        import sys
        sys.path.insert(0, str(_REPO))
        from speculative.predictor import Predictor

        backup = None
        if _LIB.exists():
            fd, backup = tempfile.mkstemp(prefix="routecrack-so-", suffix=".bak")
            os.close(fd)
            shutil.move(str(_LIB), backup)
        try:
            predictor = Predictor()
            for token in range(100, 140):
                predictor.learn([1, 2, 3, token])
            result = predictor.predict([1, 2, 3], k=8)
            self.assertEqual(len(result), 8)
            self.assertTrue(_LIB.exists(), "auto-build must leave a usable .so")
        finally:
            if backup and os.path.exists(backup):
                if _LIB.exists():
                    os.remove(_LIB)
                shutil.move(backup, _LIB)


if __name__ == "__main__":
    unittest.main()
