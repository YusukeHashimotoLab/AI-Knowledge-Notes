#!/usr/bin/env python3
"""
Mermaid Diagram Validation Tool

このスクリプトは、変換されたHTMLファイル内のMermaidダイアグラムを検証し、
構文エラーや不完全なブロックを検出します。

使用方法:
    python validate_mermaid.py              # 全HTMLファイルを検証
    python validate_mermaid.py <directory>  # 指定ディレクトリのみ検証
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict
import sys


class MermaidValidator:
    """Mermaid diagram validator for HTML files."""

    def __init__(self, base_dir: Path = None):
        """
        Initialize validator.

        Args:
            base_dir: Base directory to search for HTML files (default: current directory)
        """
        self.base_dir = base_dir or Path.cwd()
        self.errors = []
        self.warnings = []
        self.total_diagrams = 0

    def extract_mermaid_blocks(self, html_content: str, file_path: Path) -> List[Tuple[int, str]]:
        """
        Extract all Mermaid blocks from HTML content.

        Args:
            html_content: HTML file content
            file_path: Path to the HTML file (for error reporting)

        Returns:
            List of (line_number, mermaid_content) tuples
        """
        blocks = []
        lines = html_content.split('\n')
        in_mermaid = False
        mermaid_content = []
        start_line = 0

        for i, line in enumerate(lines, 1):
            if '<div class="mermaid">' in line:
                in_mermaid = True
                start_line = i
                mermaid_content = []
            elif in_mermaid and '</div>' in line:
                # Found complete block
                blocks.append((start_line, '\n'.join(mermaid_content)))
                in_mermaid = False
            elif in_mermaid:
                mermaid_content.append(line)

        # Check for unclosed blocks
        if in_mermaid:
            self.errors.append({
                'file': file_path,
                'line': start_line,
                'type': 'unclosed_block',
                'message': 'Mermaid block is not closed (missing </div>)'
            })

        return blocks

    def validate_mermaid_syntax(self, content: str, file_path: Path, line_num: int) -> Dict:
        """
        Validate Mermaid diagram syntax.

        Args:
            content: Mermaid diagram content
            file_path: Path to the HTML file
            line_num: Line number where the diagram starts

        Returns:
            Dictionary with validation results
        """
        issues = []

        # Check for diagram type
        diagram_types = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram',
                        'stateDiagram', 'erDiagram', 'gantt', 'pie', 'timeline',
                        'journey', 'gitGraph', 'mindmap', 'quadrantChart']

        first_line = content.strip().split('\n')[0] if content.strip() else ''
        has_diagram_type = any(first_line.startswith(dtype) for dtype in diagram_types)

        if not has_diagram_type:
            issues.append({
                'severity': 'error',
                'message': f'No valid diagram type found (first line: "{first_line[:50]}...")'
            })

        # Check for common syntax issues
        lines = content.strip().split('\n')

        # Validate graph/flowchart direction for graph diagrams
        if first_line.startswith('graph') or first_line.startswith('flowchart'):
            parts = first_line.split()
            if len(parts) < 2:
                issues.append({
                    'severity': 'warning',
                    'message': f'Graph/flowchart missing direction (TD, LR, etc.)'
                })
            elif parts[1] not in ['TD', 'TB', 'BT', 'RL', 'LR']:
                issues.append({
                    'severity': 'warning',
                    'message': f'Unknown graph direction: {parts[1]}'
                })

        # Check for style syntax
        style_lines = [l for l in lines if l.strip().startswith('style ')]
        for style_line in style_lines:
            # Basic style syntax check: style <node> <properties>
            parts = style_line.strip().split(maxsplit=2)
            if len(parts) < 3:
                issues.append({
                    'severity': 'warning',
                    'message': f'Incomplete style definition: "{style_line.strip()}"'
                })

        # Check for empty content
        if not content.strip():
            issues.append({
                'severity': 'error',
                'message': 'Mermaid block is empty'
            })

        return {
            'file': file_path,
            'line': line_num,
            'diagram_type': first_line.split()[0] if has_diagram_type else 'unknown',
            'issues': issues
        }

    def validate_file(self, html_file: Path) -> None:
        """
        Validate all Mermaid diagrams in a single HTML file.

        Args:
            html_file: Path to the HTML file
        """
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append({
                'file': html_file,
                'line': 0,
                'type': 'file_error',
                'message': f'Could not read file: {e}'
            })
            return

        # Extract Mermaid blocks
        blocks = self.extract_mermaid_blocks(content, html_file)
        self.total_diagrams += len(blocks)

        # Validate each block
        for line_num, mermaid_content in blocks:
            result = self.validate_mermaid_syntax(mermaid_content, html_file, line_num)

            # Categorize issues
            for issue in result['issues']:
                if issue['severity'] == 'error':
                    self.errors.append({
                        'file': html_file,
                        'line': line_num,
                        'type': 'syntax_error',
                        'diagram_type': result['diagram_type'],
                        'message': issue['message']
                    })
                else:
                    self.warnings.append({
                        'file': html_file,
                        'line': line_num,
                        'type': 'syntax_warning',
                        'diagram_type': result['diagram_type'],
                        'message': issue['message']
                    })

    def validate_all(self, target_dir: Path = None) -> None:
        """
        Validate all HTML files in the target directory.

        Args:
            target_dir: Directory to search (default: all series directories)
        """
        if target_dir:
            html_files = list(target_dir.glob('**/*.html'))
        else:
            # Search all series directories
            html_files = []
            for series_dir in self.base_dir.glob('*-introduction'):
                html_files.extend(series_dir.glob('*.html'))

        if not html_files:
            print(f"⚠ No HTML files found in {target_dir or self.base_dir}")
            return

        print(f"🔍 Validating Mermaid diagrams in {len(html_files)} HTML files...")
        print("=" * 70)

        for html_file in sorted(html_files):
            self.validate_file(html_file)

    def print_report(self) -> None:
        """Print validation report."""
        print(f"\n📊 Validation Report")
        print("=" * 70)
        print(f"Total Mermaid diagrams found: {self.total_diagrams}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print()

        # Print errors
        if self.errors:
            print("❌ ERRORS:")
            print("-" * 70)
            for error in self.errors:
                rel_path = error['file'].relative_to(self.base_dir)
                print(f"\n  File: {rel_path}:{error['line']}")
                print(f"  Type: {error['type']}")
                if 'diagram_type' in error:
                    print(f"  Diagram: {error['diagram_type']}")
                print(f"  Message: {error['message']}")
            print()

        # Print warnings
        if self.warnings:
            print("⚠️  WARNINGS:")
            print("-" * 70)
            for warning in self.warnings:
                rel_path = warning['file'].relative_to(self.base_dir)
                print(f"\n  File: {rel_path}:{warning['line']}")
                print(f"  Type: {warning['type']}")
                if 'diagram_type' in warning:
                    print(f"  Diagram: {warning['diagram_type']}")
                print(f"  Message: {warning['message']}")
            print()

        # Summary
        if not self.errors and not self.warnings:
            print("✅ All Mermaid diagrams passed validation!")
        elif not self.errors:
            print(f"✅ No critical errors found (but {len(self.warnings)} warnings)")
        else:
            print(f"❌ Found {len(self.errors)} error(s) that need attention")

        print("=" * 70)


def main():
    """Main entry point."""
    base_dir = Path.cwd()

    # Parse command-line arguments
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.exists():
            print(f"❌ Error: Directory '{target}' does not exist")
            sys.exit(1)
        validator = MermaidValidator(base_dir)
        validator.validate_all(target)
    else:
        validator = MermaidValidator(base_dir)
        validator.validate_all()

    # Print report
    validator.print_report()

    # Exit with error code if errors were found
    sys.exit(1 if validator.errors else 0)


if __name__ == '__main__':
    main()
