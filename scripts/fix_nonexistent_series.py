#!/usr/bin/env python3
"""
Remove references to non-existent series and fix file naming issues.

Handles:
1. References to series that don't exist
2. JP/EN language switching links (obsolete)
3. File naming inconsistencies
"""

import os
import sys
from pathlib import Path
import re

def fix_nonexistent_series_in_file(file_path):
    """Fix non-existent series references in a single HTML file."""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    fixes_applied = 0

    # Pattern 1: Remove references to non-existent series directories
    non_existent_series = [
        'machine-learning-basics',
        'mi-introduction',
        'deep-learning-advanced',
        'python-for-data-science',
        'prompt-engineering',
        'llm-applications',
        'llm-basics',
    ]

    for series in non_existent_series:
        # Remove entire link elements (both ../ and ../../ patterns)
        patterns = [
            rf'<a[^>]*href="\.\./{series}/"[^>]*>.*?</a>',
            rf'<a[^>]*href="\.\.\/\.\./{series}/"[^>]*>.*?</a>',
            rf'<a[^>]*href="\.\./{series}/index\.html"[^>]*>.*?</a>',
            rf'<a[^>]*href="\.\.\/\.\./{series}/index\.html"[^>]*>.*?</a>',
        ]

        for pattern in patterns:
            content, count = re.subn(pattern, '', content, flags=re.DOTALL)
            fixes_applied += count

        # Remove list items containing these links
        li_patterns = [
            rf'<li[^>]*>\s*<a[^>]*href="[^"]*{series}/?"[^>]*>.*?</a>\s*</li>',
            rf'<li[^>]*>\s*<a[^>]*href="[^"]*{series}/index\.html"[^>]*>.*?</a>\s*</li>',
        ]

        for pattern in li_patterns:
            content, count = re.subn(pattern, '', content, flags=re.DOTALL)
            fixes_applied += count

    # Pattern 2: Remove JP/EN language switching links (obsolete - replaced by locale switcher)
    jp_en_patterns = [
        r'<a[^>]*href="/jp/[^"]*"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/\.\.\/\.\.\/jp/[^"]*"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/\.\.\/\.\.\/knowledge_en\.html"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/mi_en\.html"[^>]*>.*?</a>',
    ]

    for pattern in jp_en_patterns:
        content, count = re.subn(pattern, '', content, flags=re.DOTALL)
        fixes_applied += count

    # Pattern 3: Remove references to non-existent page sections
    page_patterns = [
        r'<a[^>]*href="\.\.\/\.\.\/\.\.\/en/research\.html"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/\.\.\/\.\.\/en/publications\.html"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/\.\.\/\.\.\/en/news\.html"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/\.\.\/\.\.\/en/members\.html"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/\.\.\/\.\.\/en/contact\.html"[^>]*>.*?</a>',
        r'<a[^>]*href="/wp/knowledge/"[^>]*>.*?</a>',
    ]

    for pattern in page_patterns:
        content, count = re.subn(pattern, '', content, flags=re.DOTALL)
        fixes_applied += count

    # Pattern 4: Fix chapter filename inconsistencies
    # chapter4-real-world.html -> chapter-4.html or just remove if doesn't exist
    content, count = re.subn(
        r'href="chapter4-real-world\.html"',
        'href="chapter-4.html"',
        content
    )
    fixes_applied += count

    # Pattern 5: Fix ./chapter-X.html to just chapter-X.html (relative path)
    # Only if the file doesn't exist with ./
    content, count = re.subn(
        r'href="\./chapter-(\d+)\.html"',
        r'href="chapter-\1.html"',
        content
    )
    fixes_applied += count

    # Pattern 6: Fix chapter1-xxx naming to chapter-1 for consistency
    inconsistent_chapters = [
        ('chapter1-generative-model-basics', 'chapter-1'),
        ('chapter1-graph-basics', 'chapter-1'),
        ('chapter1-perceptron', 'chapter-1'),
        ('chapter1-signal-processing-basics', 'chapter-1'),
        ('chapter1-regression', 'chapter-1'),
        ('chapter1-anomaly-basics', 'chapter-1'),
    ]

    for old_name, new_name in inconsistent_chapters:
        content, count = re.subn(
            rf'href="\.\/{old_name}\.html"',
            f'href="./{new_name}.html"',
            content
        )
        fixes_applied += count

    # Pattern 7: Remove project references that don't exist
    content, count = re.subn(
        r'<a[^>]*href="/projects/[^"]*"[^>]*>.*?</a>',
        '',
        content,
        flags=re.DOTALL
    )
    fixes_applied += count

    if content == original_content:
        return False, "No changes needed"

    # Create backup
    backup_path = str(file_path) + '.bak'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)

    # Write fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return True, f"Applied {fixes_applied} fixes"

def get_all_html_files():
    """Get all HTML files in knowledge/en."""
    base_dir = Path("knowledge/en")
    return sorted(base_dir.rglob("*.html"))

def main():
    base_dir = Path("knowledge/en")

    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        sys.exit(1)

    print("Removing non-existent series references and fixing file names...\n")

    # Get all HTML files
    html_files = get_all_html_files()

    print(f"Scanning {len(html_files)} HTML files\n")

    files_fixed = 0
    total_fixes = 0

    for file_path in html_files:
        fixed, message = fix_nonexistent_series_in_file(file_path)
        if fixed:
            files_fixed += 1
            rel_path = file_path.relative_to(base_dir)
            print(f"✓ Fixed: {rel_path}")
            print(f"  {message}")

            # Extract number of fixes
            if "Applied" in message:
                num = int(message.split()[1])
                total_fixes += num

    print("\n" + "="*60)
    print(f"Summary:")
    print(f"  Files scanned: {len(html_files)}")
    print(f"  Files fixed: {files_fixed}")
    print(f"  Total fixes applied: {total_fixes}")
    print("="*60)

if __name__ == "__main__":
    main()
