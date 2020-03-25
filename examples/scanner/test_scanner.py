# -*- coding: utf-8 -*-
"""
Unit tests for the RandomTester class.

TODO:
    * test the examples/scanner scripts

@author: utting@usc.edu.au
"""

import unittest
import random
from pathlib import Path
import subprocess

import agilkia

THIS_DIR = Path(__file__).parent
SCANNER_DIR = THIS_DIR.parent / "examples" / "scanner"


class TestExamplesScanner(unittest.TestCase):

    def test_generate(self):
        status = subprocess.check_call(["python", "generate_missing_tests_scanette.py", "--test"],
                                  cwd=SCANNER_DIR)
        with open("test_generate.log", "w") as out:
            out.write("status=" + str(status))

    def test_analyze(self):
        status = subprocess.check_call(["python", "analyse_scanette2.py", "--test"],
                                  cwd=SCANNER_DIR)
        with open("test_analyse.log", "w") as out:
            out.write("status=" + str(status))
