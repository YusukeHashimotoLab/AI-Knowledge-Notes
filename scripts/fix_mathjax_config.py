#!/usr/bin/env python3
"""
Fix MathJax configuration in HTML files to ensure proper math rendering.

Updates incomplete MathJax configs to include:
- processEnvironments: true
- skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
- ignoreHtmlClass: 'mermaid'
"""

import re
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).parent.parent
EN_DIR = BASE_DIR / "knowledge" / "en"

# Correct MathJax configuration
CORRECT_MATHJAX_CONFIG = """<script>
        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                ignoreHtmlClass: 'mermaid'
            }
        };
    </script>
<script async="" id="MathJax-script" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>"""


def needs_fix(content: str) -> bool:
    """Check if MathJax config needs fixing."""
    if 'MathJax' not in content:
        return False

    # Check for missing components
    has_process_environments = 'processEnvironments: true' in content
    has_code_skip = "'code']" in content or '"code"]' in content
    has_mermaid_ignore = 'ignoreHtmlClass' in content

    return not (has_process_environments and has_code_skip and has_mermaid_ignore)


def fix_mathjax_config(file_path: Path) -> Tuple[bool, str]:
    """Fix MathJax configuration in file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not needs_fix(content):
            return False, "No fix needed"

        # Pattern to match old MathJax config block
        # Match from <script> with MathJax to the MathJax script tag
        pattern = r'<script>\s*MathJax\s*=\s*\{.*?\};\s*</script>\s*<script[^>]*mathjax[^>]*></script>'

        if not re.search(pattern, content, re.DOTALL | re.IGNORECASE):
            return False, "MathJax config pattern not found"

        # Replace with correct config
        new_content = re.sub(
            pattern,
            CORRECT_MATHJAX_CONFIG,
            content,
            flags=re.DOTALL | re.IGNORECASE
        )

        if new_content == content:
            return False, "No changes made"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True, "MathJax config fixed"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Main execution."""
    print("=" * 70)
    print("Fixing MathJax Configuration")
    print("=" * 70)

    # Find all HTML files in EN directory
    html_files = list(EN_DIR.rglob("*.html"))

    print(f"\nScanning {len(html_files)} files...\n")

    # Statistics
    files_checked = 0
    files_fixed = 0
    files_by_category = {}

    for html_file in html_files:
        files_checked += 1

        success, message = fix_mathjax_config(html_file)

        if success:
            files_fixed += 1

            # Get category (e.g., FM, ML, MS, etc.)
            parts = html_file.relative_to(EN_DIR).parts
            category = parts[0] if parts else "unknown"

            if category not in files_by_category:
                files_by_category[category] = []
            files_by_category[category].append(html_file.name)

            if files_fixed <= 10:  # Show first 10
                print(f"  ✅ {html_file.relative_to(BASE_DIR)}")

    # Summary by category
    if files_fixed > 10:
        print(f"\n  ... and {files_fixed - 10} more files")

    print("\n" + "=" * 70)
    print("Summary by Category")
    print("=" * 70)

    for category, files in sorted(files_by_category.items()):
        print(f"{category}: {len(files)} files fixed")

    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)
    print(f"Files checked: {files_checked}")
    print(f"Files fixed:   {files_fixed}")
    print(f"Success rate:  {files_fixed}/{files_checked}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
