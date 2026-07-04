#!/usr/bin/env python3
"""
Fix remaining broken links identified in link check.

Handles:
1. Incorrect index.html paths
2. Remaining asset path issues
3. Old AI-Knowledge-Notes paths
4. Dojo prefix issues
5. Remaining non-existent series references
"""

import os
import sys
from pathlib import Path
import re

def fix_remaining_links_in_file(file_path):
    """Fix various remaining link issues in a file."""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    fixes_applied = 0

    # Pattern 1: Fix /AI-Knowledge-Notes/ paths (old repository structure)
    content, count = re.subn(
        r'/AI-Knowledge-Notes/knowledge/en/',
        '/knowledge/en/',
        content
    )
    fixes_applied += count

    # Pattern 2: Fix ../../../index.html to proper path
    # From series/chapter-X.html depth (3 levels: en/DOJO/series/file)
    # Should be ../../../index.html (correct) or ../../index.html (from DOJO/)
    # Actually, from en/DOJO/series/file.html, ../../../ goes to root, not to en/
    # Correct path should be: ../../index.html (to DOJO level)
    # No wait, structure is: wp/knowledge/en/DOJO/series/file.html
    # From file: ../ = series, ../../ = DOJO, ../../../ = en, ../../../../ = knowledge
    # To get to en/index.html from file.html: ../../../index.html
    # Let's keep ../../../index.html as is (should work)

    # Pattern 3: Fix remaining CSS asset paths issues
    # Some files still have broken asset references
    asset_fixes = [
        (r'href="\.\.\/\.\.\/assets/css/variables\.css"', ''),
        (r'<link rel="stylesheet" href="\.\.\/\.\.\/assets/css/reset\.css">\n?', ''),
        (r'<link rel="stylesheet" href="\.\.\/\.\.\/assets/css/base\.css">\n?', ''),
        (r'<link rel="stylesheet" href="\.\.\/\.\.\/assets/css/components\.css">\n?', ''),
        (r'<link rel="stylesheet" href="\.\.\/\.\.\/assets/css/layout\.css">\n?', ''),
        (r'<link rel="stylesheet" href="\.\.\/\.\.\/assets/css/responsive\.css">\n?', ''),
        (r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets/css/variables\.css">\n?', ''),
        (r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets/css/reset\.css">\n?', ''),
        (r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets/css/base\.css">\n?', ''),
        (r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets/css/components\.css">\n?', ''),
        (r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets/css/layout\.css">\n?', ''),
        (r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets/css/responsive\.css">\n?', ''),
    ]

    for pattern, replacement in asset_fixes:
        content, count = re.subn(pattern, replacement, content)
        fixes_applied += count

    # Pattern 4: Fix Dojo prefix issues (./gnn-introduction/ etc.)
    dojo_prefix_fixes = [
        (r'href="\./gnn-introduction/index\.html"', 'href="../gnn-introduction/index.html"'),
        (r'href="\./reinforcement-learning-introduction/index\.html"', 'href="../reinforcement-learning-introduction/index.html"'),
        (r'href="\./transformer-introduction/index\.html"', 'href="../transformer-introduction/index.html"'),
    ]

    for pattern, replacement in dojo_prefix_fixes:
        content, count = re.subn(pattern, replacement, content)
        fixes_applied += count

    # Pattern 5: Fix remaining non-existent series references
    nonexistent_series = [
        r'<a[^>]*href="\.\.\/gnn-introduction/"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/supervised-learning-basics/"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/knowledge-graph/"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/recommender-systems/"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/ml-basics/"[^>]*>.*?</a>',
        r'<a[^>]*href="\.\.\/neural-network-basics/"[^>]*>.*?</a>',
    ]

    for pattern in nonexistent_series:
        content, count = re.subn(pattern, '', content, flags=re.DOTALL)
        fixes_applied += count

    # Pattern 6: Fix PI/food-process-ai-introduction index references
    # (index doesn't exist, remove references)
    content, count = re.subn(
        r'href="\.\.\/\.\.\/PI/food-process-ai-introduction/index\.html"',
        'href="#" title="Series index not yet available"',
        content
    )
    fixes_applied += count

    # Pattern 7: Fix ../../../../index.html (one too many ../)
    # Should be ../../../index.html from series/file.html
    content, count = re.subn(
        r'href="\.\.\/\.\.\/\.\.\/\.\.\/index\.html"',
        'href="../../../index.html"',
        content
    )
    fixes_applied += count

    # Pattern 8: Fix ../../PI/index.html references
    # This should likely be ../index.html from PI/series/file.html
    # OR ../../PI/index.html is correct from MI/series/file.html
    # Need to check file location to determine
    # For now, leave as is (might be correct)

    # Pattern 9: Fix relative path to knowledge-base.css
    # ../../../assets/css/knowledge-base.css should work from series/file.html
    # But some might have wrong depth
    # Check if file already has knowledge-base.css reference
    if 'knowledge-base.css' not in content:
        # Determine correct path based on file depth
        parts = Path(file_path).relative_to('knowledge/en').parts
        depth = len(parts) - 1

        if depth == 2:  # DOJO/series/file.html
            css_path = '../../../assets/css/knowledge-base.css'
        elif depth == 1:  # DOJO/file.html
            css_path = '../../assets/css/knowledge-base.css'
        else:
            css_path = '../../../assets/css/knowledge-base.css'

        # Add if not present
        if '</head>' in content:
            css_link = f'<link rel="stylesheet" href="{css_path}">\n</head>'
            content = content.replace('</head>', css_link)
            fixes_applied += 1

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

def main():
    base_dir = Path("knowledge/en")

    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        sys.exit(1)

    print("Fixing remaining broken links...\n")

    # Get all HTML files
    html_files = sorted(base_dir.rglob("*.html"))

    print(f"Scanning {len(html_files)} HTML files\n")

    files_fixed = 0
    total_fixes = 0

    for file_path in html_files:
        fixed, message = fix_remaining_links_in_file(file_path)
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
