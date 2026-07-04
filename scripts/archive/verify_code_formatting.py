#!/usr/bin/env python3
"""
Verification script for code formatting fixes.
Checks that all code-example divs have proper <pre><code> wrapping.
"""

import re
from pathlib import Path

def verify_code_formatting(base_dir: str):
    """Verify all code-example blocks have proper formatting."""
    base_path = Path(base_dir)
    results = {
        'total_files': 0,
        'files_with_code_examples': 0,
        'properly_formatted': 0,
        'improperly_formatted': 0,
        'problematic_files': []
    }

    for html_file in base_path.rglob("*.html"):
        results['total_files'] += 1

        try:
            # Try different encodings
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    content = html_file.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                print(f"Warning: Could not read {html_file}")
                continue

            # Check if file has code-example divs
            if '<div class="code-example">' not in content:
                continue

            results['files_with_code_examples'] += 1

            # Check formatting
            pattern = r'<div class="code-example">(?!\s*<pre><code)'
            if re.search(pattern, content):
                results['improperly_formatted'] += 1
                results['problematic_files'].append(str(html_file))
            else:
                results['properly_formatted'] += 1

        except Exception as e:
            print(f"Error checking {html_file}: {e}")

    return results

def main():
    base_dir = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en"

    print("=" * 80)
    print("CODE FORMATTING VERIFICATION REPORT")
    print("=" * 80)
    print()

    results = verify_code_formatting(base_dir)

    print(f"Total HTML files scanned: {results['total_files']}")
    print(f"Files with code examples: {results['files_with_code_examples']}")
    print(f"Properly formatted: {results['properly_formatted']}")
    print(f"Improperly formatted: {results['improperly_formatted']}")
    print()

    if results['improperly_formatted'] > 0:
        print("ISSUES FOUND:")
        for file_path in results['problematic_files']:
            print(f"  ❌ {file_path}")
        print()
        print("Status: FAILED - Issues need to be fixed")
    else:
        print("✅ Status: PASSED - All code blocks are properly formatted!")

    print()
    print("=" * 80)

    return 0 if results['improperly_formatted'] == 0 else 1

if __name__ == "__main__":
    exit(main())
