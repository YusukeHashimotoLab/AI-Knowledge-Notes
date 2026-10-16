#!/usr/bin/env python3
"""Unit tests for tools/html_to_md.py: page furniture never reaches Markdown.

html_to_md.py (also used by batch_html_to_md.py and sync_md_html.py) turns a
page back into Markdown with html2text.

- Every knowledge page loads assets/js/keyboard-scrollers.js with
  ``<script defer>`` right before ``</body>``; the Markdown must be exactly what
  the page gives without that tag, whether the page has
  ``<main class="container">`` or is converted from ``<body>``.
- Every outermost table sits in ``<div class="table-scroll">`` (see
  tools/convert_md_to_html.py wrap_tables()). html2text drops ``<div>`` tags,
  so the wrapper cannot leak into the Markdown as raw HTML, but a div still
  adds a paragraph break (inside ``<details>`` it changes the output), so
  html_to_md.py unwraps it first: the Markdown must be exactly what the page
  gives without the wrapper.

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


PAGE = """<!DOCTYPE html>
<html lang="en"><head><title>T - AI Terakoya</title></head><body>
<header><h1>T</h1></header>
<main class="container">
<p>Intro.</p>
{t1}
<p>Between.</p>
<div class="comparison-table">
{t2}
</div>
<details><summary>More</summary>
{t3}
</details>
<div class="navigation"><a class="nav-button" href="index.html">Index</a></div>
</main>
<footer><p>F</p></footer>
</body></html>
"""
T1 = '<table>\n<thead><tr><th>a</th><th>b</th></tr></thead>\n<tbody><tr><td>1</td><td>2</td></tr></tbody>\n</table>'
T2 = '<table><tr><th>x</th></tr><tr><td>3</td></tr></table>'
T3 = '<table><tr><th>y</th></tr><tr><td>4</td></tr></table>'
W = '<div class="table-scroll">{}</div>'


@unittest.skipUnless(_DEPS, "html2text / beautifulsoup4 not installed")
class TestTableScrollWrapperDoesNotLeak(unittest.TestCase):
    def convert(self, wrapped):
        f = W.format if wrapped else (lambda t: t)
        return html_to_md.convert_html_to_markdown(PAGE.format(t1=f(T1), t2=f(T2), t3=f(T3)))[1]

    def test_markdown_is_the_same_with_and_without_wrapper(self):
        plain, wrapped = self.convert(False), self.convert(True)
        self.assertIn("1| 2", plain)                       # the tables really became Markdown tables
        self.assertEqual(wrapped, plain)
        self.assertNotIn("table-scroll", wrapped)
        self.assertNotIn("<div", wrapped)

    def test_html2text_alone_never_emits_the_div(self):
        import html2text
        h = html2text.HTML2Text()
        h.bypass_tables = True
        out = h.handle('<p>a</p>' + W.format(T2) + '<p>b</p>')
        self.assertIn("<table>", out)                      # tables kept as HTML in this mode
        self.assertNotIn("table-scroll", out)
        self.assertNotIn("<div", out)


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
