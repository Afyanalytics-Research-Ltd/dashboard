"""Runs the Warehouse Studio JavaScript unit tests (warehouse_studio/tests/js/*.test.mjs) under Node.

Skipped when Node >= 18 is not installed.  Set WAREHOUSE_STUDIO_NODE to a node
binary to override discovery (nvm installs are found automatically).
"""

import glob
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from django.test import SimpleTestCase

HERE = Path(__file__).resolve().parent / 'js'


def find_node() -> str | None:
    candidates = [os.getenv('WAREHOUSE_STUDIO_NODE'), shutil.which('node')]
    candidates += sorted(glob.glob(os.path.expanduser('~/.nvm/versions/node/v*/bin/node')), reverse=True)
    for c in candidates:
        if c and os.path.exists(c):
            try:
                out = subprocess.run([c, '--version'], capture_output=True, text=True, timeout=10).stdout.strip()
            except OSError:
                continue
            if out.startswith('v') and int(out[1:].split('.')[0]) >= 18:
                return c
    return None


NODE = find_node()


@unittest.skipUnless(NODE, 'Node.js >= 18 not found')
class JavaScriptUnitTests(SimpleTestCase):
    def test_node_suites(self):
        files = sorted(str(p) for p in HERE.glob('*.test.mjs'))
        self.assertTrue(files, 'no JS test files found')
        proc = subprocess.run([NODE, '--test', '--test-reporter=spec', *files], capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            self.fail(f'JavaScript tests failed:\n{proc.stdout[-8000:]}\n{proc.stderr[-4000:]}')
