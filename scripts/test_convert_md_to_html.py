#!/usr/bin/env python3
"""
Unit tests for tools/convert_md_to_html.py (math protection).

Regression focus: the display-math toggle bug. A self-contained display
equation such as ``$$ S = QK^T $$`` both starts and ends with ``$$``, so the
old ``in_display_math = not in_display_math`` toggled the flag once and never
back. Every following line - prose and fenced code alike - was then treated as
math and had its underscores escaped to ``\\_``, corrupting Python examples in
the generated HTML.

Math is now protected by MathInlineProcessor (an inline pattern between code
spans and backslash escapes), with MathUnderscorePreprocessor keeping the
previous "_" escaping for "$...$" spans that turn out not to be math. Every
test checks the final HTML: the TeX comes out as written (apart from the
legacy escapes listed below), prose after an equation is still Markdown
(emphasis, snake_case words), and fenced code, raw HTML and single-line code
spans are byte-for-byte unchanged. Indented code keeps its backslashes, but
"_" inside "$...$" there still gets one, as before (a known-limit test).

Covered here:
  - single-line ``$$ ... $$`` (content kept, following prose/code unaffected)
  - multi-line ``$$`` blocks (content kept, block closes properly)
  - a truncated equation does not swallow the rest of the document
  - fenced, tilde and mermaid code, raw HTML and single-line code spans are
    never touched; indented code (and code spans spread over lines) keeps its
    backslashes, with the known limit for "_" inside "$...$"
  - inline ``$ ... $`` protection without touching surrounding prose
  - Markdown specials inside math: ``*``, ``\!``, ``\{``, ``\\[ \\]`` with ``_{}``
  - the backslash escapes the sources rely on: ``\\``, ``\*``, ``\_``, ``\|``
  - raw HTML, tables, block quotes, lists, link/image attributes, ``\$`` and
    dollar amounts around prose
  - a generated chapter loads its locale's keyboard-scrollers.js exactly once,
    with ``<script defer>`` right before ``</body>``, by a src that resolves

The tool under test lives in ``../tools`` relative to this file; everything is
anchored on ``__file__`` (never ``Path.cwd()``) so the suite passes both from
``wp/``::

    python3 -m unittest scripts.test_convert_md_to_html -v

and from wp's parent::

    python3 -m unittest discover -s wp/scripts -p 'test_convert_md_to_html.py'

``markdown`` / ``pyyaml`` are imported by the tool at module scope. If they are
not installed the whole module is skipped rather than erroring, so the shared
test suite still runs in environments that only need the scripts/ dependencies.
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_TOOLS_DIR = _THIS_DIR.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

try:  # pragma: no cover - environment dependent
    from convert_md_to_html import convert_markdown_to_html
    from convert_md_to_html import LOCALES, convert_chapter
    _IMPORT_ERROR = None
except ImportError as exc:  # markdown / yaml not installed
    convert_markdown_to_html = convert_chapter = LOCALES = None
    _IMPORT_ERROR = exc


def _html(text: str) -> str:
    return convert_markdown_to_html(text)


_skip = unittest.skipIf(_IMPORT_ERROR is not None,
                        f"convert_md_to_html import failed: {_IMPORT_ERROR}")


@_skip
class TestSingleLineDisplayMath(unittest.TestCase):
    """A line that both opens and closes with $$ must leave what follows alone."""

    def test_content_is_kept(self):
        self.assertEqual(_html('$$ Q_i = W_q x_i $$'), '<p>$$ Q_i = W_q x_i $$</p>')

    def test_following_prose_untouched(self):
        html = _html('\n'.join([
            '$$ S = QK^T $$',
            '',
            'The scale_factor keeps the dot products *small*.',
            'Another line with snake_case_words here.',
        ]))
        self.assertIn('<p>$$ S = QK^T $$</p>', html)
        self.assertIn('The scale_factor keeps the dot products <em>small</em>.', html)
        self.assertIn('Another line with snake_case_words here.', html)
        self.assertNotIn('\\_', html)

    def test_two_consecutive_single_line_equations(self):
        html = _html('\n'.join(['$$ a_1 = b_1 $$', '$$ a_2 = b_2 $$', 'plain_text_after *x*']))
        self.assertIn('$$ a_1 = b_1 $$\n$$ a_2 = b_2 $$\nplain_text_after <em>x</em>', html)

    def test_indented_single_line_equation(self):
        self.assertIn('$$ x_1 $$', _html('  $$ x_1 $$'))

    def test_bare_double_dollar_opens_a_block(self):
        html = _html('\n'.join(['$$', 'x_1 + y_2', '$$', 'after_text *x*']))
        self.assertIn('$$\nx_1 + y_2\n$$\nafter_text <em>x</em>', html)

    def test_trailing_prose_after_closing_delimiter(self):
        """Real corpus pattern: `$$...$$ where $x$ is ...` on a single line."""
        html = _html('\n'.join([
            r'$$\oint_C f(z) dz = 0$$ where $f(z)$ is *analytic* in region_R',
            'plain_text_after',
        ]))
        self.assertIn(r'$$\oint_C f(z) dz = 0$$ where $f(z)$ is <em>analytic</em> in region_R', html)
        self.assertIn('plain_text_after', html)


@_skip
class TestMultiLineDisplayMath(unittest.TestCase):
    """Multi-line $$ ... $$ blocks keep their body and then close."""

    def test_block_body_kept_and_block_closes(self):
        html = _html('\n'.join([
            'Attention is defined as:',
            '$$',
            r'\text{Attention}(Q_i, K_j) = \frac{Q_i K_j^T}{\sqrt{d_k}}',
            '$$',
            'Here d_k is the *key* dimension.',
        ]))
        self.assertIn('$$\n' + r'\text{Attention}(Q_i, K_j) = \frac{Q_i K_j^T}{\sqrt{d_k}}' + '\n$$', html)
        self.assertIn('Here d_k is the <em>key</em> dimension.', html)

    def test_multiple_blocks_alternate_correctly(self):
        html = _html('\n'.join(['$$', 'a_1 * b', '$$', 'prose *one*', '$$', 'b_2 * c', '$$', 'prose *two*']))
        self.assertIn('$$\na_1 * b\n$$\nprose <em>one</em>\n$$\nb_2 * c\n$$\nprose <em>two</em>', html)

    def test_closing_delimiter_at_end_of_content_line(self):
        html = _html('\n'.join(['$$', 'E = m c_0^2 $$', 'tail_text *x*']))
        self.assertIn('$$\nE = m c_0^2 $$\ntail_text <em>x</em>', html)

    def test_truncated_equation_does_not_swallow_the_document(self):
        """An unclosed equation must not turn the following paragraphs into math.

        The corpus contains equations truncated mid-expression, e.g.
        ``$$ ... P(y_t | y_{`` with no closing ``$$``.
        """
        html = _html('\n'.join([
            r'$$ y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i',
            '',
            'Where the *terms* are defined below.',
            '',
            '```python',
            'coef_matrix = model.coef_',
            '```',
        ]))
        self.assertIn('Where the <em>terms</em> are defined below.', html)
        self.assertIn('coef_matrix = model.coef_\n</code></pre>', html)
        self.assertNotIn('\\_', html)

    def test_content_on_opening_line_is_kept(self):
        html = _html('\n'.join(['$$ a_1 +', 'b_2^*', '$$', 'tail_text *x*']))
        self.assertIn('$$ a_1 +\nb_2^*\n$$\ntail_text <em>x</em>', html)


@_skip
class TestCodeIsLeftAlone(unittest.TestCase):
    """Fenced code, raw HTML and single-line code spans come out exactly as written.

    Indented code keeps its backslashes; "_" inside "$...$" in it is the one
    known exception, as before (test_known_limit_underscore_in_indented_code_as_before).
    """

    CODE = [
        'from sklearn.model_selection import train_test_split',
        'X_train, X_test = train_test_split(X, test_size=0.2)',
        'attn_scores = q @ k.transpose(-2, -1)',
    ]

    def _assert_code_verbatim(self, html):
        self.assertIn('\n'.join(self.CODE) + '\n</code></pre>', html)
        self.assertNotIn('\\_', html)

    def _fenced(self):
        return ['```python'] + self.CODE + ['```']

    def test_code_after_single_line_equation(self):
        self._assert_code_verbatim(_html('\n'.join(['$$ S = QK^T $$', ''] + self._fenced())))

    def test_code_after_multi_line_equation(self):
        self._assert_code_verbatim(_html('\n'.join(['$$', 'S = Q K^T', '$$', ''] + self._fenced())))

    def test_code_between_two_equations(self):
        html = _html('\n'.join(['$$ a_1 $$', ''] + self._fenced() + ['', '$$ b_2 $$']))
        self._assert_code_verbatim(html)
        self.assertIn('<p>$$ a_1 $$</p>', html)
        self.assertIn('<p>$$ b_2 $$</p>', html)

    def test_dollar_signs_inside_code_are_left_alone(self):
        html = _html('\n'.join(['```bash', 'echo $HOME_DIR and $USER_NAME', '```']))
        self.assertIn('echo $HOME_DIR and $USER_NAME\n</code></pre>', html)

    def test_tilde_fence_and_info_string(self):
        html = _html('\n'.join(['~~~python', 'a_b = c_d', '~~~', 'tail_text']))
        self.assertIn('a_b = c_d\n</code></pre>', html)
        self.assertIn('<p>tail_text</p>', html)

    def test_mermaid_fence_untouched(self):
        html = _html('\n'.join([
            '$$ y = W_1 x $$',
            '```mermaid',
            'flowchart LR',
            '  A[input_layer] --> B[hidden_layer]',
            '```',
        ]))
        self.assertIn('  A[input_layer] --> B[hidden_layer]', html)
        self.assertNotIn('\\_', html)

    def test_indented_code_keeps_backslashes(self):
        html = _html('Text.\n\n    $a\\\\b$ in code\n')
        self.assertIn('<pre><code>$a\\\\b$ in code\n</code></pre>', html)

    def test_known_limit_underscore_in_indented_code_as_before(self):
        """Documented limit, same as the previous converter: the line-based "_"
        escaping cannot tell indented code from a list paragraph, so "_" inside
        "$...$" in indented code gets a backslash. Fenced code is never touched."""
        self.assertIn('<pre><code>$x\\_i$\n</code></pre>', _html('Text.\n\n    $x_i$\n'))
        self.assertIn('<pre><code>$x_i$\n</code></pre>', _html('```\n$x_i$\n```\n'))

    def test_math_in_a_list_continuation_paragraph(self):
        html = _html('- matrix sum\n\n    $[A]_{ij} + [B]_{ij}$ and $\\text{avg\\_dev}$ and $y^* + z^*$\n')
        self.assertIn(r'<p>$[A]_{ij} + [B]_{ij}$ and $\text{avg\_dev}$ and $y^* + z^*$</p>', html)

    def test_code_spans_including_multi_line_ones(self):
        self.assertEqual(_html('Run `$a\\\\b$` and `\n$c\\\\d$\n` now.'),
                         '<p>Run <code>$a\\\\b$</code> and <code>$c\\\\d$</code> now.</p>')

    def test_code_span_between_dollar_amounts(self):
        self.assertEqual(_html('Cost $5 and `code_x` costs $6.'),
                         '<p>Cost $5 and <code>code_x</code> costs $6.</p>')


@_skip
class TestInlineMath(unittest.TestCase):
    """Inline $...$ spans are kept; surrounding prose is still Markdown."""

    def test_inline_span_kept_prose_untouched(self):
        self.assertEqual(_html('The value $Q_i K_j$ feeds the soft_max *step*.'),
                         '<p>The value $Q_i K_j$ feeds the soft_max <em>step</em>.</p>')

    def test_two_inline_spans_on_one_line(self):
        self.assertEqual(_html('Both $a_1$ and $b_2$ matter for x_axis labels.'),
                         '<p>Both $a_1$ and $b_2$ matter for x_axis labels.</p>')

    def test_no_math_line_is_unchanged(self):
        line = 'Use train_test_split from sklearn.model_selection here.'
        self.assertEqual(_html(line), f'<p>{line}</p>')

    def test_mid_line_display_math_kept(self):
        self.assertEqual(_html('Given $$a_1 = b_2$$ we continue with plain_text.'),
                         '<p>Given $$a_1 = b_2$$ we continue with plain_text.</p>')


@_skip
class TestMarkdownSpecialsInsideMath(unittest.TestCase):
    """What the previous converter corrupted: these must now survive."""

    def test_stars_are_not_emphasis(self):
        self.assertEqual(_html('at $y^* = m x$ and $y - y^*$, *real* emphasis.'),
                         '<p>at $y^* = m x$ and $y - y^*$, <em>real</em> emphasis.</p>')

    def test_tex_backslash_commands_are_kept(self):
        html = _html(r'$\ln\!\left(x\right)$ and $\{a, b\}$ and $a\#b$')
        self.assertIn(r'$\ln\!\left(x\right)$ and $\{a, b\}$ and $a\#b$', html)

    def test_escaped_bracket_display_with_subscripts(self):
        html = _html(r'\\[ F_{\mu\nu} = H^{\text{core}}_{\mu\nu} + \sum_{\lambda\sigma} P_{\lambda\sigma} \\]')
        self.assertEqual(html, r'<p>\[ F_{\mu\nu} = H^{\text{core}}_{\mu\nu} + \sum_{\lambda\sigma} P_{\lambda\sigma} \]</p>')

    def test_escaped_paren_inline(self):
        self.assertEqual(_html(r'see \\(x_{i}^* + y_{j}^*\\) here'),
                         r'<p>see \(x_{i}^* + y_{j}^*\) here</p>')

    def test_escaped_environment_delimiters(self):
        self.assertEqual(_html(r'\\begin{aligned}a &= b^*\\end{aligned}'),
                         r'<p>\begin{aligned}a &amp;= b^*\end{aligned}</p>')

    def test_environment_outside_delimiters(self):
        html = _html(r'Then \begin{aligned} a_1 &= b^* \\\\ c_2 &= d^* \end{aligned} holds.')
        self.assertIn(r'\begin{aligned} a_1 &amp;= b^* \\ c_2 &amp;= d^* \end{aligned} holds.', html)

    def test_math_in_block_quote_and_list(self):
        html = _html('> $$\n> a^* + b^*\n> $$\n\n- item $$\n  c^* + d^*\n  $$ end\n')
        self.assertIn('$$\na^* + b^*\n$$', html)
        self.assertIn('c^* + d^*', html)
        self.assertNotIn('<em>', html)


@_skip
class TestLegacyEscapes(unittest.TestCase):
    """Escapes that existing sources rely on keep the meaning they always had."""

    def test_double_backslash_is_one_backslash(self):
        self.assertIn(r'$\{x_i\}$ and $a \\ b$', _html(r'$\\{x_i\\}$ and $a \\\\ b$'))

    def test_escaped_star_underscore_and_pipe(self):
        self.assertIn(r'$\theta^*$ and \(a_i b_j\) and $|x|$',
                      _html(r'$\theta^\*$ and \\(a\_i b\_j\\) and $\|x\|$'))

    def test_escaped_pipe_in_table_cell(self):
        html = _html('| a | b |\n|---|---|\n| $P(w_i \\| w_j)$ | ok |\n')
        self.assertIn('<td>$P(w_i | w_j)$</td>', html)
        self.assertIn('<td>ok</td>', html)


@_skip
class TestNotMath(unittest.TestCase):
    """Things that look like math but are not, and HTML that must stay valid."""

    def test_escaped_dollar_is_not_a_delimiter(self):
        self.assertEqual(_html(r'costs \$5 and \$6, with $x + \$5 + y^*$ here'),
                         r'<p>costs \$5 and \$6, with $x + \$5 + y^*$ here</p>')

    def test_dollar_amounts_around_prose_stay_markdown(self):
        html = _html(r'Budget: $150/kWh (2024) \- **Target**: $100/kWh.')
        self.assertIn('<strong>Target</strong>', html)
        self.assertIn('$150/kWh (2024) - ', html)

    def test_raw_html_block_is_left_exactly_as_written(self):
        block = '<p>Raw \\(a\\\\b\\) and $x\\_y$ stay.</p>'
        self.assertIn(block, _html(block + '\n'))

    def test_attribute_values_stay_valid(self):
        html = _html('![$\\text{"hi"}$](x "t $y^*$")')
        self.assertIn('alt="$\\text{&quot;hi&quot;}$"', html)
        self.assertIn('title="t $y^*$"', html)

    def test_dollar_inside_inline_html_attributes(self):
        self.assertEqual(_html('Go <a href="/$x$">here</a> and <span title="$y$">hi</span>.'),
                         '<p>Go <a href="/$x$">here</a> and <span title="$y$">hi</span>.</p>')

    def test_reference_links_with_dollar_amounts(self):
        html = _html('[price $5][ref] and $6. ![img $5][ref] and $7.\n\n[ref]: /url\n')
        self.assertIn('<a href="/url">price $5</a> and $6.', html)
        self.assertIn('<img alt="img $5" src="/url" /> and $7.', html)

    def test_stars_in_braces_are_math_not_bold(self):
        self.assertEqual(_html('$a^{**}+b^{**}$'), '<p>$a^{**}+b^{**}$</p>')

    def test_brackets_and_less_than_in_real_math(self):
        self.assertEqual(
            _html(r'$[\alpha^*][\beta^*]$, $f(x)=[\alpha](x)$, $(0,1]$, $[0,1)$; for x<y, $a^*+b^*$.'),
            r'<p>$[\alpha^*][\beta^*]$, $f(x)=[\alpha](x)$, $(0,1]$, $[0,1)$; for x&lt;y, $a^*+b^*$.</p>')

    def test_spaced_and_shortcut_reference_links_with_dollar_amounts(self):
        html = _html('[price $5] [ref] and $6. [p2 $5] and $6. ![img $5] [ref] and $7.\n\n'
                     '[ref]: /url\n[p2 $5]: /url2\n')
        self.assertIn('<a href="/url">price $5</a> and $6.', html)
        self.assertIn('<a href="/url2">p2 $5</a> and $6.', html)
        self.assertIn('<img alt="img $5" src="/url" /> and $7.', html)

    def test_inline_html_between_dollar_amounts(self):
        self.assertEqual(_html('Cost $5 <a href="/x">link</a> $6.'),
                         '<p>Cost $5 <a href="/x">link</a> $6.</p>')

    def test_known_limit_brackets_or_less_than_without_tex_commands(self):
        """Documented limit, same output as the previous converter: a $...$ span
        with no TeX command that contains "[" or "<" is left to Markdown, because
        that is what a link or tag between two dollar amounts looks like."""
        self.assertEqual(_html('$f(x)=[a](x)$'), '<p>$f(x)=<a href="x">a</a>$</p>')
        self.assertEqual(_html('$[A^*][B^*]$'), '<p>$[A^<em>][B^</em>]$</p>')

    def test_math_left_to_markdown_keeps_its_underscores(self):
        """A span left to Markdown still gets the previous converter's "_" escaping."""
        self.assertEqual(_html('$[A]_{ij} + [B]_{ij}$ and $5 [x_y](u) $6'),
                         '<p>$[A]_{ij} + [B]_{ij}$ and $5 <a href="u">x_y</a> $6</p>')

    def test_a_tex_command_keeps_brackets_and_less_than_as_math(self):
        self.assertEqual(_html(r'$f(x)=[\mathrm{a}](x)$, $\lbrack A^*\rbrack[B^*]$, $a\lt b^*$'),
                         r'<p>$f(x)=[\mathrm{a}](x)$, $\lbrack A^*\rbrack[B^*]$, $a\lt b^*$</p>')

    def test_reference_links_between_dollar_amounts(self):
        html = _html('Cost $5 (see [pricing] [rates]) or $6. Cost $5 (see [p2]) or $6.\n\n'
                     '[rates]: /pricing\n[p2]: /p2\n')
        self.assertIn('Cost $5 (see <a href="/pricing">pricing</a>) or $6.', html)
        self.assertIn('Cost $5 (see <a href="/p2">p2</a>) or $6.', html)

    def test_half_open_interval_inside_a_link_label(self):
        self.assertEqual(_html('[$(0,1]$ の説明](interval.html)'),
                         '<p><a href="interval.html">$(0,1]$ の説明</a></p>')

    def test_link_label_broken_across_lines(self):
        html = _html('[price\n$5] [ref] and $6.\n\n[ref]: /url\n')
        self.assertIn('<a href="/url">price\n$5</a> and $6.', html)

    def test_dollar_amount_ending_inside_an_attribute(self):
        self.assertEqual(_html("Price $5 <span title='$6 with tax'>tax</span>."),
                         "<p>Price $5 <span title='$6 with tax'>tax</span>.</p>")

    def test_inline_link_between_dollar_amounts(self):
        self.assertEqual(_html('Cost $5 [link](https://x.org) $6.'),
                         '<p>Cost $5 <a href="https://x.org">link</a> $6.</p>')

    def test_relative_link_and_autolinks_between_dollar_amounts(self):
        self.assertEqual(_html('Cost $5 (see [pricing](pricing)) or $6.'),
                         '<p>Cost $5 (see <a href="pricing">pricing</a>) or $6.</p>')
        self.assertIn('<a href="https://example.com/pricing">https://example.com/pricing</a>) or $6.',
                      _html('Cost $5 (see <https://example.com/pricing>) or $6.'))
        self.assertNotIn('&lt;me', _html('Mail $5 <me@example.com> $6.'))

    def test_text_that_looks_like_a_placeholder(self):
        self.assertIn('<code>mathstashtoken000000x</code>', _html('`mathstashtoken000000x` and $x$'))


@_skip
class TestKeyboardScrollerScript(unittest.TestCase):
    """Every generated page loads its locale's keyboard-scrollers.js (the script
    that gives boxes which scroll on their own a Tab stop): once, deferred, right
    before </body>, by a relative src that reaches the file."""

    TAG_RE = re.compile(r'<script defer src="([^"]+)"></script>\n</body>\n</html>\n$')

    def test_generated_chapter_loads_the_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            for lang in ('en', 'jp'):
                series = Path(tmp) / 'knowledge' / lang / 'MI' / 'demo-series'
                series.mkdir(parents=True)
                (series / 'chapter-1.md').write_text('---\ntitle: T\n---\n\n# T\n\nText.\n', encoding='utf-8')
                self.assertTrue(convert_chapter(series, 'chapter-1.md', LOCALES[lang]))
                html = (series / 'chapter-1.html').read_text(encoding='utf-8')
                m = self.TAG_RE.search(html)
                self.assertIsNotNone(m, html[-300:])
                self.assertEqual(html.count('keyboard-scrollers.js'), 1)
                self.assertEqual((series / m.group(1)).resolve(),
                                 (Path(tmp) / 'knowledge' / lang / 'assets' / 'js' / 'keyboard-scrollers.js').resolve())


@_skip
class TestEndToEndConversion(unittest.TestCase):
    """Full Markdown -> HTML conversion must not corrupt code examples."""

    def test_code_block_after_single_line_equation(self):
        html = _html('\n'.join([
            '## Self-Attention',
            '',
            '$$ S = Q_i K_j^T $$',
            '',
            'The scores are then normalised.',
            '',
            '```python',
            'X_train, X_test = train_test_split(X, y, test_size=0.2)',
            'attn_weights = softmax(scores_matrix)',
            '```',
        ]))
        self.assertIn('train_test_split(X, y, test_size=0.2)', html)
        self.assertIn('attn_weights = softmax(scores_matrix)', html)
        self.assertNotIn('<em>', html)
        self.assertNotIn('\\_', html)
        self.assertIn('$$ S = Q_i K_j^T $$', html)

    def test_prose_emphasis_still_works_outside_math(self):
        self.assertIn('<em>emphasised</em>', _html('$$ a_1 $$\n\nThis is _emphasised_ text.\n'))


def run_tests():
    """Run all tests (used when executing this file directly)."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
