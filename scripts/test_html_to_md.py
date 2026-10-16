#!/usr/bin/env python3
"""Unit tests for tools/html_to_md.py: page furniture never reaches Markdown.

html_to_md.py (also used by batch_html_to_md.py and sync_md_html.py) turns a
page back into Markdown with html2text. Every knowledge page loads
assets/js/keyboard-scrollers.js with ``<script defer>`` right before
``</body>``; the Markdown must be exactly what the page gives without that tag,
whether the page has ``<main class="container">`` or is converted from
``<body>``.

html_to_md.py exits when html2text or beautifulsoup4 is missing (both are in
requirements.txt); the tests are skipped in that case.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
_DEPS = all(importlib.util.find_spec(m) for m in ("html2text", "bs4"))
if _DEPS:
    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))
    import html_to_md  # noqa: E402

SCRIPT = '<script defer src="../../assets/js/keyboard-scrollers.js"></script>\n'
SCRIPT_PAGE = """<!DOCTYPE html>
<html lang="en"><head><title>T - AI Terakoya</title></head><body>
<header><h1>T</h1></header>
{main_open}
<p>Intro with <code>code</code>.</p>
<pre><code>x = 1</code></pre>
{main_close}
<footer><p>F</p></footer>
{script}</body></html>
"""


@unittest.skipUnless(_DEPS, "html2text / beautifulsoup4 not installed")
class TestScriptTagDoesNotLeak(unittest.TestCase):
    def convert(self, with_main, with_script):
        html = SCRIPT_PAGE.format(main_open='<main class="container">' if with_main else '',
                                  main_close='</main>' if with_main else '',
                                  script=SCRIPT if with_script else '')
        return html_to_md.convert_html_to_markdown(html)[1]

    def test_markdown_is_the_same_with_and_without_the_script(self):
        for with_main in (True, False):            # False: converted from <body>
            plain, tagged = self.convert(with_main, False), self.convert(with_main, True)
            self.assertIn("Intro with `code`.", plain)
            self.assertEqual(tagged, plain)
            self.assertNotIn("keyboard-scrollers", tagged)
            self.assertNotIn("<script", tagged)


if __name__ == "__main__":
    unittest.main()
