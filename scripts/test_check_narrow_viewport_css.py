#!/usr/bin/env python3
"""Unit tests for scripts/check_narrow_viewport_css.py.

Each test builds a small knowledge tree in a temporary directory (two locale
sheets, the two copies of keyboard-scrollers.js, a converter template, an
inline-style page, a page that links the sheet, a dojo.css page, each page
loading the script) and checks what the script reports. The last test runs the script on this
repository, so a commit that edits one copy of the block and not the others
fails here as well as in ``--check``.

Run from wp/::

    python3 -m unittest scripts.test_check_narrow_viewport_css -v
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import check_narrow_viewport_css as cnv  # noqa: E402

REAL_SHEET_TO_INLINE = dict(cnv.SHEET_TO_INLINE)   # the subprocess test runs without the mock

RULES = """@media screen and (max-width: 768px) {
    mjx-container[display="true"],
    .katex-display {
        overflow-x: auto;
        overflow-y: hidden;
    }

    body {
        overflow-wrap: break-word;
    }
}
"""

FOCUS = '[data-kbd-scroll]:focus-visible,\npre[tabindex]:focus-visible {\n    outline-offset: -3px;\n}\n'
SHEET = ("body { margin: 0; }\n\n/* ====\n   19. Narrow-viewport overflow guards\n   ==== */\n\n"
         + RULES + "\n/* keyboard focus */\n" + FOCUS)

INLINE_COPY = """        /* Narrow-viewport overflow guards: a copy of section 19. */
        @media screen and (max-width: 768px) {
            mjx-container[display="true"], .katex-display { overflow-x: auto; overflow-y: hidden }
            body { overflow-wrap: break-word }
            .video-container iframe { max-width: 100%; height: auto; aspect-ratio: 16 / 9 }
        }
        [data-kbd-scroll]:focus-visible, pre[tabindex]:focus-visible { outline-offset: -3px }
"""


JS = "(function () { 'use strict'; }());\n"
TAG2 = '<script defer src="../../assets/js/keyboard-scrollers.js"></script>\n'   # series pages
TAG1 = '<script defer src="../assets/js/keyboard-scrollers.js"></script>\n'      # Dojo index pages
CONVERTER = ('HTML_FOOTER_TEMPLATE = """\n    </footer>\n'
             '<script defer src="{js_src}"></script>\n</body>\n</html>\n"""\n')


def inline_page(copy=INLINE_COPY, tag=TAG2):
    return ("<!DOCTYPE html>\n<html><head>\n    <style>\n        body { color: red; }\n"
            + copy + "    </style>\n</head><body><p>x</p>\n" + tag + "</body></html>\n")


LINKED_PAGE = ('<!DOCTYPE html>\n<html><head>\n<link href="../../assets/css/knowledge-base.css" '
               'rel="stylesheet"/>\n<style>p { margin: 0 }</style>\n</head><body>\n' + TAG2 + '</body></html>\n')
DOJO_PAGE = ('<!DOCTYPE html>\n<html><head>\n<link rel="stylesheet" href="../assets/css/dojo.css">\n'
             '</head><body>\n' + TAG1 + '</body></html>\n')


class CheckTest(unittest.TestCase):
    def setUp(self):
        # the fixture sheet has none of the repository's sheet-only rules
        self._map = mock.patch.dict(cnv.SHEET_TO_INLINE, {}, clear=True)
        self._map.start()
        self.addCleanup(self._map.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for loc in ("en", "jp"):
            self.write(f"knowledge/{loc}/assets/css/knowledge-base.css", SHEET)
            self.write(f"knowledge/{loc}/assets/js/keyboard-scrollers.js", JS)
        self.write("tools/convert_md_to_html.py", CONVERTER)
        self.write("knowledge/en/MI/series/chapter-1.html", inline_page())
        self.write("knowledge/jp/MI/series/chapter-1.html", inline_page())
        self.write("knowledge/en/MI/linked/chapter-1.html", LINKED_PAGE)
        self.write("knowledge/en/MI/index.html", DOJO_PAGE)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, rel, text):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def check(self):
        return cnv.check(str(self.root))

    def test_in_step(self):
        problems, stats = self.check()
        self.assertEqual(problems, [])
        self.assertEqual(stats, {"inline_pages": 2, "inline_ok": 2, "linked_pages": 2,
                                 "script_pages": 4, "script_ok": 4})

    def test_indentation_and_line_breaks_do_not_matter(self):
        copy = INLINE_COPY.replace("{ overflow-x: auto; overflow-y: hidden }",
                                   "{\n  overflow-x:auto;\n  overflow-y :hidden;\n}")
        self.write("knowledge/en/MI/series/chapter-1.html", inline_page(copy))
        self.assertEqual(self.check()[0], [])

    def test_locale_sheets_must_match(self):
        self.write("knowledge/jp/assets/css/knowledge-base.css",
                   SHEET.replace("break-word", "anywhere"))
        problems, _ = self.check()
        self.assertEqual(len(problems), 1)
        self.assertIn("differs from", problems[0])

    def test_missing_section(self):
        self.write("knowledge/jp/assets/css/knowledge-base.css", "body { margin: 0; }\n")
        problems, _ = self.check()
        self.assertTrue(any("not found" in p for p in problems))

    def test_inline_copy_with_changed_declaration(self):
        self.write("knowledge/en/MI/series/chapter-1.html",
                   inline_page(INLINE_COPY.replace("break-word }", "anywhere }")))
        problems, stats = self.check()
        self.assertEqual(len(problems), 1)
        self.assertIn("en/MI/series/chapter-1.html", problems[0])
        self.assertEqual(stats["inline_ok"], 1)

    def test_inline_copy_missing_rule(self):
        copy = "".join(l for l in INLINE_COPY.splitlines(True) if "body {" not in l)
        self.write("knowledge/jp/MI/series/chapter-1.html", inline_page(copy))
        problems, _ = self.check()
        self.assertEqual(len(problems), 1)
        self.assertIn("missing", problems[0])

    def test_focus_rule_must_follow_the_block(self):
        self.write("knowledge/en/MI/series/chapter-1.html",
                   inline_page(INLINE_COPY.replace("outline-offset: -3px", "outline-offset: 0")))
        problems, _ = self.check()
        self.assertEqual(len(problems), 1)
        self.assertIn("en/MI/series/chapter-1.html: the focus rule after the copy differs", problems[0])
        for loc in ("en", "jp"):
            self.write(f"knowledge/{loc}/assets/css/knowledge-base.css", SHEET.replace("\n/* keyboard focus */\n" + FOCUS, ""))
        problems, _ = self.check()
        self.assertTrue(any("no :focus-visible rule" in p for p in problems), problems)

    def test_inline_copy_missing_entirely(self):
        self.write("knowledge/jp/MI/series/chapter-1.html", inline_page(""))
        problems, _ = self.check()
        self.assertEqual(problems, ["knowledge/jp/MI/series/chapter-1.html: 0 inline copies (expected 1)"])

    def test_two_inline_copies(self):
        self.write("knowledge/jp/MI/series/chapter-1.html", inline_page(INLINE_COPY + INLINE_COPY))
        problems, _ = self.check()
        self.assertEqual(problems, ["knowledge/jp/MI/series/chapter-1.html: 2 inline copies (expected 1)"])

    def test_linked_page_must_not_carry_a_copy(self):
        self.write("knowledge/en/MI/linked/chapter-1.html",
                   LINKED_PAGE.replace("<style>p { margin: 0 }", "<style>p { margin: 0 }\n" + INLINE_COPY))
        problems, _ = self.check()
        self.assertEqual(len(problems), 1)
        self.assertIn("links the shared sheet", problems[0])

    def test_sheet_rule_with_an_inline_counterpart(self):
        sheet_rule = ".x { margin-top: 1rem; margin-bottom: var(--m) }"
        inline_rule = ".x { margin-top: 0; margin-bottom: var(--m, 1rem) }"
        cnv.SHEET_TO_INLINE[sheet_rule] = inline_rule
        sheet = SHEET.replace("    body {", "    .x {\n        margin-top: 1rem;\n        margin-bottom: var(--m);\n    }\n\n    body {")
        for loc in ("en", "jp"):
            self.write(f"knowledge/{loc}/assets/css/knowledge-base.css", sheet)
        line = "            " + inline_rule + "\n"
        good = INLINE_COPY.replace("            body {", line + "            body {")
        self.write("knowledge/en/MI/series/chapter-1.html", inline_page(good))
        self.write("knowledge/jp/MI/series/chapter-1.html", inline_page(good))
        self.assertEqual(self.check()[0], [])
        # a copy that kept the sheet's version of the rule is drift
        bad = INLINE_COPY.replace("            body {", "            " + sheet_rule + "\n            body {")
        self.write("knowledge/jp/MI/series/chapter-1.html", inline_page(bad))
        problems, _ = self.check()
        self.assertEqual(len(problems), 1)
        self.assertIn("jp/MI/series/chapter-1.html", problems[0])

    def test_script_copies_must_match(self):
        self.write("knowledge/jp/assets/js/keyboard-scrollers.js", JS.replace("strict", "sloppy"))
        problems, _ = self.check()
        self.assertEqual(problems, ["knowledge/jp/assets/js/keyboard-scrollers.js: differs from "
                                    "knowledge/en/assets/js/keyboard-scrollers.js (the canonical copy)"])

    def test_page_without_the_script(self):
        self.write("knowledge/en/MI/linked/chapter-1.html", LINKED_PAGE.replace(TAG2, ""))
        problems, stats = self.check()
        self.assertEqual(problems, ["knowledge/en/MI/linked/chapter-1.html: loads keyboard-scrollers.js 0 times (expected 1)"])
        self.assertEqual(stats["script_ok"], 3)

    def test_script_in_a_comment_does_not_count(self):
        self.write("knowledge/en/MI/linked/chapter-1.html", LINKED_PAGE.replace(TAG2, "<!--" + TAG2 + "-->"))
        self.assertIn("0 times", self.check()[0][0])

    def test_page_loading_the_script_twice(self):
        self.write("knowledge/jp/MI/series/chapter-1.html", inline_page(tag=TAG2 + TAG2))
        self.assertEqual(self.check()[0], ["knowledge/jp/MI/series/chapter-1.html: loads keyboard-scrollers.js 2 times (expected 1)"])

    def test_wrong_path_missing_defer_and_wrong_place(self):
        self.write("knowledge/en/MI/series/chapter-1.html", inline_page(tag=TAG1))           # one level short
        self.write("knowledge/jp/MI/series/chapter-1.html",
                   inline_page(tag=TAG2.replace("../../assets", "../../../en/assets")))      # other locale
        self.write("knowledge/en/MI/linked/chapter-1.html", LINKED_PAGE.replace(" defer", ""))
        self.write("knowledge/en/MI/index.html", DOJO_PAGE.replace(TAG1, TAG1 + "<p>after</p>\n"))
        problems, _ = self.check()
        self.assertEqual(sorted(problems), [
            "knowledge/en/MI/index.html: keyboard-scrollers.js is not loaded right before </body>",
            "knowledge/en/MI/linked/chapter-1.html: keyboard-scrollers.js is loaded without defer",
            "knowledge/en/MI/series/chapter-1.html: keyboard-scrollers.js src '../assets/js/keyboard-scrollers.js' "
            "does not reach knowledge/en/assets/js/keyboard-scrollers.js",
            "knowledge/jp/MI/series/chapter-1.html: keyboard-scrollers.js src '../../../en/assets/js/keyboard-scrollers.js' "
            "does not reach knowledge/jp/assets/js/keyboard-scrollers.js",
        ])

    def test_converter_template_must_load_the_script(self):
        self.write("tools/convert_md_to_html.py", CONVERTER.replace('<script defer src="{js_src}"></script>\n', ""))
        problems, _ = self.check()
        self.assertEqual(len(problems), 1)
        self.assertIn("tools/convert_md_to_html.py: page template", problems[0])

    def test_check_flag_sets_exit_code(self):
        script = _THIS_DIR / "check_narrow_viewport_css.py"
        # the script runs with its real SHEET_TO_INLINE, so the fixture carries those rules
        sheet, copy = SHEET, INLINE_COPY
        for sheet_rule, inline_rule in REAL_SHEET_TO_INLINE.items():
            sheet = sheet.replace("    body {", "    " + sheet_rule + "\n    body {")
            copy = copy.replace("            body {", "            " + inline_rule + "\n            body {")
        for loc in ("en", "jp"):
            self.write(f"knowledge/{loc}/assets/css/knowledge-base.css", sheet)
        self.write("knowledge/en/MI/series/chapter-1.html", inline_page(copy))
        self.write("knowledge/jp/MI/series/chapter-1.html", inline_page(copy))
        ok = subprocess.run([sys.executable, str(script), "--check", "--root", str(self.root)],
                            capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.write("knowledge/en/MI/series/chapter-1.html", inline_page(""))
        bad = subprocess.run([sys.executable, str(script), "--check", "--root", str(self.root)],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)
        self.assertIn("Drift: 1", bad.stdout)
        report = subprocess.run([sys.executable, str(script), "--root", str(self.root)],
                                capture_output=True, text=True)
        self.assertEqual(report.returncode, 0)


class RepositoryTest(unittest.TestCase):
    def test_repository_copies_are_in_step(self):
        problems, stats = cnv.check(cnv.REPO_ROOT)
        self.assertEqual(problems, [])
        self.assertGreater(stats["inline_pages"], 0)
        self.assertEqual(stats["inline_pages"], stats["inline_ok"])
        self.assertGreater(stats["script_pages"], 0)
        self.assertEqual(stats["script_pages"], stats["script_ok"])


if __name__ == "__main__":
    unittest.main()
