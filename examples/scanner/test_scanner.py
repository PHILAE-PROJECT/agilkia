# -*- coding: utf-8 -*-
"""
Unit tests for the examples/scanner scripts.

@author: m.utting@uq.edu.au
"""

import unittest
from pathlib import Path
import subprocess


SCANNER_DIR = Path(__file__).parent


class TestExamplesScanner(unittest.TestCase):

    def ok(self, args):
        """Convenience function to run a command, check success, and capture output as text."""
        return subprocess.run(args, cwd=SCANNER_DIR, check=True, capture_output=True, text=True)

    def test_analyze(self):
        status = self.ok(["python", "analyse_scanette2.py", "--test"])
        with open("test_analyse.log", "w") as out:
            out.write("status=" + str(status))

    def test_generate(self):
        # Note: this assumes "log_split.json" as input.
        json = SCANNER_DIR / "log_split.json"
        self.assertTrue(json.exists(), "Needs log_split.json.  Run analyse_scanette2.py first.")
        status = self.ok(["python", "generate_missing_tests_scanette.py", "--test"])
        with open("test_generate.log", "w") as out:
            out.write("status=" + str(status))

    def test_read_write(self):
        """Convert Scanette *.csv to Agilkia *.json and back again, and check it is the same."""
        csv1 = SCANNER_DIR / "1026-steps.csv"
        json1 = SCANNER_DIR / "1026-steps.json"
        json2 = SCANNER_DIR / "tmp.json"
        csv2 = SCANNER_DIR / "tmp.csv"
        # clear any old tmp files
        if json2.exists(): json2.unlink()
        if csv2.exists(): csv2.unlink()
        # first convert csv1 to json1
        r1 = self.ok(["python", "read_scanette_csv.py", "1026-steps.csv"])
        self.assertEqual("  1026-steps.csv -> 1026-steps.json [1 traces]\n", r1.stdout)
        # now rename the output .json file
        self.assertTrue(json1.exists())
        json1.rename(json2)
        r2 = self.ok(["python", "write_scanette_csv.py", "tmp.json"])
        self.assertEqual("  tmp.json -> tmp.csv [1 traces]\n", r2.stdout)
        # with open("test_read_write.log", "w") as out:
        #    out.write("r1=" + str(r1))
        #    out.write("r2=" + str(r2))
        # Now check the two files are the same.
        self.assertEqual(csv1.read_text(), csv2.read_text())
        # clear all tmp files
        json2.unlink()
        csv2.unlink()
