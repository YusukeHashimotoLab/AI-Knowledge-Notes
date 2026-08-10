#!/usr/bin/env python3
"""
Unit tests for tools/validate_mermaid.py

Covers the path-handling fix (F-01) and the Mermaid block extraction/validation
behaviour:
  - zero-error run on valid diagrams
  - single-line ``<div class="mermaid">...</div>`` blocks (they must be captured
    and validated, and must not swallow the markup that follows)
  - error run reports the correct count and a non-zero exit/return code
  - relative-path invocation (the original crash scenario)
  - absolute-path invocation
  - directory containing no HTML files

Bootstrapping note: the tool under test lives in ``../tools`` relative to this
file. Everything here is anchored on ``__file__`` (never ``Path.cwd()``), so the
suite passes both when run from ``wp/``::

    python3 -m unittest scripts.test_validate_mermaid -v

and from the parent directory::

    python3 -m unittest discover -s wp/scripts -p 'test_validate_mermaid.py'
"""

import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

# The tool under test lives in wp/tools/validate_mermaid.py. Resolve it relative
# to this file so the import works no matter what the current working directory
# is (the suite is run both from wp/ and from wp's parent).
_THIS_DIR = Path(__file__).resolve().parent
_TOOLS_DIR = _THIS_DIR.parent / "tools"
_TOOL_PATH = _TOOLS_DIR / "validate_mermaid.py"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from validate_mermaid import MermaidValidator  # noqa: E402


# --- Synthetic Mermaid fixtures -------------------------------------------------

# Inline single-line block with ';' separators (exercises the regex extractor).
VALID_INLINE = '<div class="mermaid">flowchart TD; A[Start]--&gt;B[End]</div>'

# Standard multi-line block.
VALID_MULTILINE = (
    '<div class="mermaid">\n'
    'flowchart LR\n'
    '  A[One] --> B[Two]\n'
    '</div>'
)

# xychart-beta is a valid mermaid@10 diagram type (regression: was a false error).
VALID_XYCHART = (
    '<div class="mermaid">\n'
    'xychart-beta\n'
    '  title "Demo"\n'
    '  line [1, 2, 3]\n'
    '</div>'
)

# <pre> wrapper: mermaid reads innerHTML, so the first token is literally
# "<pre>" and no diagram type is detected -> genuine error.
BROKEN_PRE = (
    '<div class="mermaid">\n'
    '<pre>\n'
    'flowchart TD\n'
    '  A --> B\n'
    '</pre>\n'
    '</div>'
)

# No recognizable diagram type -> genuine error.
BROKEN_NO_TYPE = '<div class="mermaid">this is not a diagram at all</div>'


def _page(body: str) -> str:
    return f"<!DOCTYPE html><html><head><title>t</title></head><body>{body}</body></html>"


def _write(dir_path: Path, name: str, body: str) -> Path:
    p = Path(dir_path) / name
    p.write_text(_page(body), encoding="utf-8")
    return p


def _run_quiet(fn, *args, **kwargs):
    """Call fn with stdout suppressed; return (result, captured_stdout)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


class TestZeroErrorRuns(unittest.TestCase):
    """Valid diagrams must produce no errors."""

    def test_inline_block_no_errors(self):
        with tempfile.TemporaryDirectory() as td:
            _write(td, "inline.html", VALID_INLINE)
            v = MermaidValidator(Path(td))
            _run_quiet(v.validate_all, Path(td))
            _run_quiet(v.print_report)
            self.assertEqual(v.errors, [])
            self.assertEqual(v.total_diagrams, 1)

    def test_multiline_and_xychart_no_errors(self):
        with tempfile.TemporaryDirectory() as td:
            _write(td, "multi.html", VALID_MULTILINE)
            _write(td, "xy.html", VALID_XYCHART)
            v = MermaidValidator(Path(td))
            _run_quiet(v.validate_all, Path(td))
            self.assertEqual(v.errors, [])
            self.assertEqual(v.total_diagrams, 2)


class TestSingleLineDiv(unittest.TestCase):
    """Regression: a diagram whose <div> opens and closes on ONE line.

    The original line-based extractor used if/elif so a line containing both
    ``<div class="mermaid">`` and ``</div>`` only ever matched the "opening"
    branch: the block was opened, never closed, its content was never captured
    and following markup was swallowed as diagram text. Single-line diagrams
    were therefore never validated.
    """

    def test_single_line_div_is_extracted_and_validated(self):
        with tempfile.TemporaryDirectory() as td:
            f = _write(td, "single.html", VALID_INLINE + "\n<h3>Next section</h3>")
            v = MermaidValidator(Path(td))
            blocks = v.extract_mermaid_blocks(f.read_text(encoding="utf-8"), f)
            self.assertEqual(len(blocks), 1)
            # Content is the diagram only - the following <h3> is NOT swallowed.
            self.assertEqual(blocks[0][1], "flowchart TD; A[Start]--&gt;B[End]")
            self.assertNotIn("<h3>", blocks[0][1])
            self.assertEqual(v.errors, [])

    def test_broken_single_line_div_is_reported(self):
        """A single-line diagram with no diagram type must raise an error."""
        with tempfile.TemporaryDirectory() as td:
            _write(td, "bad_single.html", BROKEN_NO_TYPE + "\n<p>after</p>")
            v = MermaidValidator(Path(td))
            _run_quiet(v.validate_all, Path(td))
            self.assertEqual(v.total_diagrams, 1)
            self.assertEqual(len(v.errors), 1)
            self.assertEqual(v.errors[0]['type'], 'syntax_error')

    def test_several_single_line_divs_all_counted(self):
        with tempfile.TemporaryDirectory() as td:
            body = "\n".join([
                VALID_INLINE,
                '<p>prose between diagrams</p>',
                '<div class="mermaid">sequenceDiagram; A->>B: hi</div>',
                VALID_MULTILINE,
            ])
            _write(td, "many.html", body)
            v = MermaidValidator(Path(td))
            _run_quiet(v.validate_all, Path(td))
            self.assertEqual(v.total_diagrams, 3)
            self.assertEqual(v.errors, [])

    def test_unclosed_div_still_reported(self):
        """An opening tag with no </div> is a genuine unclosed block."""
        with tempfile.TemporaryDirectory() as td:
            _write(td, "unclosed.html", '<div class="mermaid">\nflowchart TD\n  A --> B')
            v = MermaidValidator(Path(td))
            _run_quiet(v.validate_all, Path(td))
            self.assertEqual(len(v.errors), 1)
            self.assertEqual(v.errors[0]['type'], 'unclosed_block')


class TestErrorRuns(unittest.TestCase):
    """Broken diagrams must be counted and surface a non-zero exit code."""

    def test_broken_blocks_counted(self):
        with tempfile.TemporaryDirectory() as td:
            _write(td, "pre.html", BROKEN_PRE)
            _write(td, "notype.html", BROKEN_NO_TYPE)
            v = MermaidValidator(Path(td))
            _run_quiet(v.validate_all, Path(td))
            # print_report must not raise.
            _run_quiet(v.print_report)
            self.assertEqual(len(v.errors), 2)

    def test_cli_exit_code_zero_when_clean(self):
        with tempfile.TemporaryDirectory() as td:
            _write(td, "ok.html", VALID_MULTILINE)
            proc = subprocess.run(
                [sys.executable, str(_TOOL_PATH), td],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_cli_exit_code_nonzero_when_errors(self):
        with tempfile.TemporaryDirectory() as td:
            _write(td, "bad.html", BROKEN_NO_TYPE)
            proc = subprocess.run(
                [sys.executable, str(_TOOL_PATH), td],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)


class TestPathInvocation(unittest.TestCase):
    """Path handling: relative and absolute scan targets vs. absolute base_dir."""

    def test_relative_path_invocation(self):
        """Reproduces the original crash: absolute base_dir + relative target."""
        with tempfile.TemporaryDirectory() as td:
            sub = Path(td) / "sub"
            sub.mkdir()
            _write(sub, "bad.html", BROKEN_NO_TYPE)
            old_cwd = os.getcwd()
            try:
                os.chdir(td)
                # base_dir resolves to an absolute path; target is relative.
                v = MermaidValidator(Path.cwd())
                _run_quiet(v.validate_all, Path("sub"))
                # print_report used to raise ValueError here before the fix.
                _, report = _run_quiet(v.print_report)
            finally:
                os.chdir(old_cwd)
            self.assertEqual(len(v.errors), 1)
            # Reported path is relative to base_dir, not absolute.
            self.assertIn("sub/bad.html", report.replace(os.sep, "/"))
            self.assertNotIn(str(Path(td).resolve()), report)

    def test_absolute_path_invocation(self):
        with tempfile.TemporaryDirectory() as td:
            sub = Path(td) / "sub"
            sub.mkdir()
            _write(sub, "bad.html", BROKEN_NO_TYPE)
            v = MermaidValidator(Path(td))
            _run_quiet(v.validate_all, sub.resolve())  # absolute target
            _, report = _run_quiet(v.print_report)
            self.assertEqual(len(v.errors), 1)
            self.assertIn("sub/bad.html", report.replace(os.sep, "/"))

    def test_target_outside_base_dir_uses_relpath_fallback(self):
        """When the scan target is not under base_dir, relpath fallback applies."""
        with tempfile.TemporaryDirectory() as base_td, \
                tempfile.TemporaryDirectory() as scan_td:
            _write(scan_td, "bad.html", BROKEN_NO_TYPE)
            v = MermaidValidator(Path(base_td))  # unrelated base
            _run_quiet(v.validate_all, Path(scan_td))
            # print_report must not raise even though scan_td is outside base_td.
            _, report = _run_quiet(v.print_report)
            self.assertEqual(len(v.errors), 1)
            self.assertIn("bad.html", report)


class TestNoHtml(unittest.TestCase):
    """A directory with no HTML files is a clean, error-free run."""

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as td:
            v = MermaidValidator(Path(td))
            _, out = _run_quiet(v.validate_all, Path(td))
            self.assertEqual(v.errors, [])
            self.assertEqual(v.total_diagrams, 0)
            self.assertIn("No HTML files found", out)

    def test_cli_exit_code_zero_for_no_html(self):
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [sys.executable, str(_TOOL_PATH), td],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


def run_tests():
    """Run all tests (used when executing this file directly)."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for tc in (TestZeroErrorRuns, TestSingleLineDiv, TestErrorRuns,
               TestPathInvocation, TestNoHtml):
        suite.addTests(loader.loadTestsFromTestCase(tc))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
