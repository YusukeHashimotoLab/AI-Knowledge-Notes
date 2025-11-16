#!/usr/bin/env python3
"""
Unit tests for fix_broken_links.py

Tests all fix patterns to ensure correct behavior.
"""

import sys
import unittest
from pathlib import Path
from fix_broken_links import LinkFixer


class TestLinkFixer(unittest.TestCase):
    """Test cases for LinkFixer class."""

    def setUp(self):
        """Set up test fixture."""
        # Use current directory as base
        self.base_dir = Path.cwd()
        self.fixer = LinkFixer(base_dir=self.base_dir, dry_run=True)

    def test_file_context_chapter(self):
        """Test file context extraction for chapter file."""
        file_path = self.base_dir / "knowledge/en/MI/gnn-introduction/chapter-1.html"
        context = self.fixer.get_file_context(file_path)

        self.assertEqual(context['dojo'], 'MI')
        self.assertEqual(context['series'], 'gnn-introduction')
        self.assertEqual(context['filename'], 'chapter-1.html')
        self.assertEqual(context['depth'], 2)
        self.assertEqual(context['type'], 'chapter')

    def test_file_context_series_index(self):
        """Test file context extraction for series index."""
        file_path = self.base_dir / "knowledge/en/MI/gnn-introduction/index.html"
        context = self.fixer.get_file_context(file_path)

        self.assertEqual(context['dojo'], 'MI')
        self.assertEqual(context['series'], 'gnn-introduction')
        self.assertEqual(context['type'], 'series_index')
        self.assertEqual(context['depth'], 2)

    def test_file_context_dojo_index(self):
        """Test file context extraction for dojo index."""
        file_path = self.base_dir / "knowledge/en/MI/index.html"
        context = self.fixer.get_file_context(file_path)

        self.assertEqual(context['dojo'], 'MI')
        self.assertEqual(context['type'], 'dojo_index')
        self.assertEqual(context['depth'], 1)

    def test_fix_absolute_knowledge_path_depth2(self):
        """Test fixing absolute /knowledge/en/ path from depth 2."""
        context = {'depth': 2}

        result = self.fixer.fix_absolute_knowledge_paths('/knowledge/en/', context)
        self.assertEqual(result, '../../')

        result = self.fixer.fix_absolute_knowledge_paths('/knowledge/en/MI/', context)
        self.assertEqual(result, '../../MI/')

        result = self.fixer.fix_absolute_knowledge_paths(
            '/knowledge/en/MI/gnn-introduction/', context
        )
        self.assertEqual(result, '../../MI/gnn-introduction/')

    def test_fix_absolute_knowledge_path_depth1(self):
        """Test fixing absolute /knowledge/en/ path from depth 1."""
        context = {'depth': 1}

        result = self.fixer.fix_absolute_knowledge_paths('/knowledge/en/', context)
        self.assertEqual(result, '../')

        result = self.fixer.fix_absolute_knowledge_paths('/knowledge/en/MI/', context)
        self.assertEqual(result, '../MI/')

    def test_fix_breadcrumb_depth_chapter(self):
        """Test fixing breadcrumb depth for chapter files."""
        context = {'depth': 2, 'type': 'chapter'}

        # Chapter file (depth=2) going to root should use ../../
        result = self.fixer.fix_breadcrumb_depth('../../../index.html', context)
        self.assertEqual(result, '../../index.html')

    def test_fix_breadcrumb_depth_series_index(self):
        """Test fixing breadcrumb depth for series index."""
        context = {'depth': 2, 'type': 'series_index'}

        # Series index (depth=2) going to root should use ../
        result = self.fixer.fix_breadcrumb_depth('../../index.html', context)
        self.assertEqual(result, '../index.html')

    def test_fix_breadcrumb_depth_dojo_index(self):
        """Test fixing breadcrumb depth for dojo index."""
        context = {'depth': 1, 'type': 'dojo_index'}

        # Dojo index (depth=1) going to root should use ./
        result = self.fixer.fix_breadcrumb_depth('../../index.html', context)
        self.assertEqual(result, './index.html')

        # Dojo index using ../../FM/ should use ../FM/
        result = self.fixer.fix_breadcrumb_depth('../../FM/index.html', context)
        self.assertEqual(result, '../FM/index.html')

    def test_fix_asset_paths_depth2(self):
        """Test fixing asset paths from depth 2."""
        context = {'depth': 2}

        result = self.fixer.fix_asset_paths('/assets/css/variables.css', context)
        self.assertEqual(result, '../../assets/css/variables.css')

        result = self.fixer.fix_asset_paths('/assets/js/main.js', context)
        self.assertEqual(result, '../../assets/js/main.js')

    def test_fix_asset_paths_depth1(self):
        """Test fixing asset paths from depth 1."""
        context = {'depth': 1}

        result = self.fixer.fix_asset_paths('/assets/css/base.css', context)
        self.assertEqual(result, '../assets/css/base.css')

    def test_fix_asset_paths_depth0(self):
        """Test fixing asset paths from depth 0."""
        context = {'depth': 0}

        result = self.fixer.fix_asset_paths('/assets/css/base.css', context)
        self.assertEqual(result, 'assets/css/base.css')

    def test_files_similar_same_chapter(self):
        """Test filename similarity for same chapter number."""
        # Same chapter number should match
        self.assertTrue(
            self.fixer._files_similar(
                'chapter2-q-learning.html',
                'chapter2-q-learning-sarsa.html'
            )
        )

        # Different chapter numbers should not match
        self.assertFalse(
            self.fixer._files_similar(
                'chapter2-q-learning.html',
                'chapter3-q-learning.html'
            )
        )

    def test_files_similar_close_names(self):
        """Test filename similarity for close names."""
        self.assertTrue(
            self.fixer._files_similar(
                'chapter4-deep-learning-interpretation.html',
                'chapter4-deep-learning-interpretability.html'
            )
        )

    def test_files_similar_different_names(self):
        """Test filename similarity for different names."""
        self.assertFalse(
            self.fixer._files_similar(
                'chapter1-basics.html',
                'chapter2-advanced.html'
            )
        )

    def test_nonexistent_series_robotic_lab(self):
        """Test nonexistent series detection."""
        context = {'dojo': 'MI'}

        result = self.fixer.fix_nonexistent_series(
            '../robotic-lab-automation-introduction/index.html',
            context
        )
        self.assertIsNotNone(result)

        fixed_link, action = result
        # Should be marked for commenting since we can't auto-fix this
        self.assertEqual(action, 'comment')

    def test_pattern_priority(self):
        """Test that patterns are applied in correct priority."""
        # Absolute knowledge path should take priority over relative depth
        context = {'depth': 2, 'type': 'chapter'}

        link = '/knowledge/en/MI/'
        result = self.fixer.fix_absolute_knowledge_paths(link, context)
        self.assertEqual(result, '../../MI/')

        # Asset path should be recognized
        link = '/assets/css/base.css'
        result = self.fixer.fix_asset_paths(link, context)
        self.assertEqual(result, '../../assets/css/base.css')


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def setUp(self):
        """Set up test fixture."""
        self.base_dir = Path.cwd()
        self.fixer = LinkFixer(base_dir=self.base_dir, dry_run=True)

    def test_statistics_initialization(self):
        """Test that statistics are properly initialized."""
        self.assertEqual(self.fixer.stats.files_processed, 0)
        self.assertEqual(self.fixer.stats.files_modified, 0)
        self.assertEqual(self.fixer.stats.total_fixes, 0)

    def test_add_fix_statistics(self):
        """Test that statistics are updated correctly."""
        self.fixer.stats.add_fix('test_pattern')

        self.assertEqual(self.fixer.stats.total_fixes, 1)
        self.assertEqual(self.fixer.stats.fixes_by_pattern['test_pattern'], 1)

        self.fixer.stats.add_fix('test_pattern')
        self.assertEqual(self.fixer.stats.total_fixes, 2)
        self.assertEqual(self.fixer.stats.fixes_by_pattern['test_pattern'], 2)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestLinkFixer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
