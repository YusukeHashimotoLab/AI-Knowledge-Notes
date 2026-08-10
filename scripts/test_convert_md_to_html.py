#!/usr/bin/env python3
"""
Unit tests for tools/convert_md_to_html.py (MathPreprocessor).

Regression focus: the display-math toggle bug. A self-contained display
equation such as ``$$ S = QK^T $$`` both starts and ends with ``$$``, so the
old ``in_display_math = not in_display_math`` toggled the flag once and never
back. Every following line - prose and fenced code alike - was then treated as
math and had its underscores escaped to ``\\_``, corrupting Python examples in
the generated HTML.

Covered here:
  - single-line ``$$ ... $$`` (content protected, state NOT flipped)
  - multi-line ``$$`` blocks (content protected, block closes properly)
  - fenced code blocks are never math-processed (before and after equations)
  - inline ``$ ... $`` protection without touching surrounding prose

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

import sys
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_TOOLS_DIR = _THIS_DIR.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

try:  # pragma: no cover - environment dependent
    from convert_md_to_html import MathPreprocessor, convert_markdown_to_html
    _IMPORT_ERROR = None
except ImportError as exc:  # markdown / yaml not installed
    MathPreprocessor = None
    convert_markdown_to_html = None
    _IMPORT_ERROR = exc


def _preprocess(text: str):
    """Run MathPreprocessor over a Markdown string, returning the lines."""
    return MathPreprocessor(None).run(text.split('\n'))


@unittest.skipIf(_IMPORT_ERROR is not None,
                 f"convert_md_to_html import failed: {_IMPORT_ERROR}")
class TestSingleLineDisplayMath(unittest.TestCase):
    """A line that both opens and closes with $$ must not flip the state."""

    def test_content_is_protected(self):
        out = _preprocess('$$ Q_i = W_q x_i $$')
        self.assertEqual(out, [r'$$ Q\_i = W\_q x\_i $$'])

    def test_state_not_flipped_following_prose_untouched(self):
        md = '\n'.join([
            '$$ S = QK^T $$',
            '',
            'The scale_factor keeps the dot products small.',
            'Another line with snake_case_words here.',
        ])
        out = _preprocess(md)
        self.assertEqual(out[0], '$$ S = QK^T $$')  # no underscores to escape
        self.assertEqual(out[2], 'The scale_factor keeps the dot products small.')
        self.assertEqual(out[3], 'Another line with snake_case_words here.')
        self.assertNotIn(r'\_', '\n'.join(out[1:]))

    def test_two_consecutive_single_line_equations(self):
        md = '\n'.join([
            '$$ a_1 = b_1 $$',
            '$$ a_2 = b_2 $$',
            'plain_text_after',
        ])
        out = _preprocess(md)
        self.assertEqual(out[0], r'$$ a\_1 = b\_1 $$')
        self.assertEqual(out[1], r'$$ a\_2 = b\_2 $$')
        self.assertEqual(out[2], 'plain_text_after')

    def test_indented_single_line_equation(self):
        out = _preprocess('  $$ x_1 $$')
        self.assertEqual(out, [r'  $$ x\_1 $$'])

    def test_bare_double_dollar_still_opens_a_block(self):
        """`$$` alone is a delimiter, not a self-contained equation."""
        md = '\n'.join(['$$', 'x_1 + y_2', '$$', 'after_text'])
        out = _preprocess(md)
        self.assertEqual(out[1], r'x\_1 + y\_2')
        self.assertEqual(out[3], 'after_text')

    def test_trailing_prose_after_closing_delimiter(self):
        """Real corpus pattern: `$$...$$ where $x$ is ...` on a single line."""
        md = '\n'.join([
            r'$$\oint_C f(z) dz = 0$$ where $f(z)$ is analytic in region_R',
            'plain_text_after',
        ])
        out = _preprocess(md)
        self.assertEqual(
            out[0],
            r'$$\oint\_C f(z) dz = 0$$ where $f(z)$ is analytic in region_R',
        )
        # State must not have flipped: the next line is prose.
        self.assertEqual(out[1], 'plain_text_after')


@unittest.skipIf(_IMPORT_ERROR is not None,
                 f"convert_md_to_html import failed: {_IMPORT_ERROR}")
class TestMultiLineDisplayMath(unittest.TestCase):
    """Multi-line $$ ... $$ blocks still protect their body and then close."""

    def test_block_body_protected_and_block_closes(self):
        md = '\n'.join([
            'Attention is defined as:',
            '$$',
            r'\text{Attention}(Q_i, K_j) = \frac{Q_i K_j^T}{\sqrt{d_k}}',
            '$$',
            'Here d_k is the key dimension.',
        ])
        out = _preprocess(md)
        self.assertEqual(out[1], '$$')
        self.assertIn(r'Q\_i', out[2])
        self.assertIn(r'd\_k', out[2])
        self.assertEqual(out[3], '$$')
        # State was reset by the closing delimiter.
        self.assertEqual(out[4], 'Here d_k is the key dimension.')

    def test_multiple_blocks_alternate_correctly(self):
        md = '\n'.join([
            '$$', 'a_1', '$$',
            'prose_one',
            '$$', 'b_2', '$$',
            'prose_two',
        ])
        out = _preprocess(md)
        self.assertEqual(out[1], r'a\_1')
        self.assertEqual(out[3], 'prose_one')
        self.assertEqual(out[5], r'b\_2')
        self.assertEqual(out[7], 'prose_two')

    def test_closing_delimiter_at_end_of_content_line(self):
        md = '\n'.join(['$$', 'E = m c_0^2 $$', 'tail_text'])
        out = _preprocess(md)
        self.assertEqual(out[1], r'E = m c\_0^2 $$')
        self.assertEqual(out[2], 'tail_text')

    def test_truncated_equation_does_not_swallow_the_document(self):
        """A blank line closes an open block (guards truncated source equations).

        The corpus contains equations truncated mid-expression, e.g.
        ``$$ ... P(y_t | y_{`` with no closing ``$$``. Without the blank-line
        guard every following line would be escaped as math.
        """
        md = '\n'.join([
            r'$$ y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i',
            '',
            'Where the terms are defined below.',
            '',
            '```python',
            'coef_matrix = model.coef_',
            '```',
        ])
        out = _preprocess(md)
        self.assertIn(r'\beta\_0', out[0])          # equation body protected
        self.assertEqual(out[2], 'Where the terms are defined below.')
        self.assertEqual(out[5], 'coef_matrix = model.coef_')
        self.assertNotIn(r'\_', '\n'.join(out[1:]))

    def test_content_on_opening_line_is_protected(self):
        md = '\n'.join(['$$ a_1 +', 'b_2', '$$', 'tail_text'])
        out = _preprocess(md)
        self.assertEqual(out[0], r'$$ a\_1 +')
        self.assertEqual(out[1], r'b\_2')
        self.assertEqual(out[3], 'tail_text')


@unittest.skipIf(_IMPORT_ERROR is not None,
                 f"convert_md_to_html import failed: {_IMPORT_ERROR}")
class TestFencedCodeBlocks(unittest.TestCase):
    """Code fences must never be math-processed (this runs before fenced_code)."""

    CODE = [
        '```python',
        'from sklearn.model_selection import train_test_split',
        'X_train, X_test = train_test_split(X, test_size=0.2)',
        'attn_scores = q @ k.transpose(-2, -1)',
        '```',
    ]

    def _assert_code_verbatim(self, out_lines):
        self.assertNotIn(r'\_', '\n'.join(out_lines))
        for original in self.CODE:
            self.assertIn(original, out_lines)

    def test_code_after_single_line_equation(self):
        md = '\n'.join(['$$ S = QK^T $$', ''] + self.CODE)
        self._assert_code_verbatim(_preprocess(md))

    def test_code_after_multi_line_equation(self):
        md = '\n'.join(['$$', 'S = Q K^T', '$$', ''] + self.CODE)
        self._assert_code_verbatim(_preprocess(md))

    def test_code_between_two_equations(self):
        md = '\n'.join(['$$ a_1 $$', ''] + self.CODE + ['', '$$ b_2 $$'])
        out = _preprocess(md)
        for original in self.CODE:
            self.assertIn(original, out)
        self.assertEqual(out[0], r'$$ a\_1 $$')
        self.assertEqual(out[-1], r'$$ b\_2 $$')

    def test_dollar_signs_inside_code_are_left_alone(self):
        md = '\n'.join([
            '```bash',
            'echo $HOME_DIR and $USER_NAME',
            '```',
        ])
        out = _preprocess(md)
        self.assertEqual(out[1], 'echo $HOME_DIR and $USER_NAME')

    def test_tilde_fence_and_info_string(self):
        md = '\n'.join(['~~~python', 'a_b = c_d', '~~~', 'tail_text'])
        out = _preprocess(md)
        self.assertEqual(out[1], 'a_b = c_d')
        self.assertEqual(out[2], '~~~')
        self.assertEqual(out[3], 'tail_text')

    def test_mermaid_fence_untouched(self):
        md = '\n'.join([
            '$$ y = W_1 x $$',
            '```mermaid',
            'flowchart LR',
            '  A[input_layer] --> B[hidden_layer]',
            '```',
        ])
        out = _preprocess(md)
        self.assertEqual(out[3], '  A[input_layer] --> B[hidden_layer]')
        self.assertNotIn(r'\_', out[3])


@unittest.skipIf(_IMPORT_ERROR is not None,
                 f"convert_md_to_html import failed: {_IMPORT_ERROR}")
class TestInlineMath(unittest.TestCase):
    """Inline $...$ spans are protected; surrounding prose is not."""

    def test_inline_span_protected_prose_untouched(self):
        out = _preprocess('The value $Q_i K_j$ feeds the soft_max step.')
        self.assertEqual(out[0], r'The value $Q\_i K\_j$ feeds the soft_max step.')

    def test_two_inline_spans_on_one_line(self):
        out = _preprocess('Both $a_1$ and $b_2$ matter for x_axis labels.')
        self.assertEqual(out[0], r'Both $a\_1$ and $b\_2$ matter for x_axis labels.')

    def test_no_math_line_is_unchanged(self):
        line = 'Use train_test_split from sklearn.model_selection here.'
        self.assertEqual(_preprocess(line), [line])

    def test_mid_line_display_math_protected(self):
        out = _preprocess('Given $$a_1 = b_2$$ we continue with plain_text.')
        self.assertEqual(out[0], r'Given $$a\_1 = b\_2$$ we continue with plain_text.')


@unittest.skipIf(_IMPORT_ERROR is not None,
                 f"convert_md_to_html import failed: {_IMPORT_ERROR}")
class TestEndToEndConversion(unittest.TestCase):
    """Full Markdown -> HTML conversion must not corrupt code examples."""

    def test_code_block_after_single_line_equation(self):
        md = '\n'.join([
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
        ])
        html = convert_markdown_to_html(md)
        # Code survives verbatim: no emphasis, no leaked backslash escapes.
        self.assertIn('train_test_split(X, y, test_size=0.2)', html)
        self.assertIn('attn_weights = softmax(scores_matrix)', html)
        self.assertNotIn('<em>', html)
        self.assertNotIn('\\_', html)
        # The equation keeps literal underscores for MathJax (protection worked;
        # Markdown resolves the \_ escape back to a plain underscore).
        self.assertIn('$$ S = Q_i K_j^T $$', html)

    def test_prose_emphasis_still_works_outside_math(self):
        html = convert_markdown_to_html('$$ a_1 $$\n\nThis is _emphasised_ text.\n')
        self.assertIn('<em>emphasised</em>', html)


def run_tests():
    """Run all tests (used when executing this file directly)."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
