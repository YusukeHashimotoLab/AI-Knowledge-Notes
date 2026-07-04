#!/usr/bin/env python3
"""
Fix missing index.html references in HTML files.

Handles two types of issues:
1. Navigation loop issues (incorrect relative paths)
2. References to non-existent series
"""

import os
import sys
from pathlib import Path
import re

def fix_missing_index_in_file(file_path):
    """Fix missing index references in a single HTML file."""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    fixes_applied = 0

    # Pattern 1: Fix navigation loop - ../../FM/index.html -> ../../index.html
    # This occurs in FM/index.html trying to link to itself incorrectly
    content, count = re.subn(
        r'href="\.\.\/\.\.\/FM/index\.html"',
        'href="../../index.html"',
        content
    )
    fixes_applied += count

    # Pattern 2: Fix ../../../../index.html -> ../../../index.html
    # From series/chapter-X.html depth (3 levels)
    content, count = re.subn(
        r'href="\.\.\/\.\.\/\.\.\/\.\.\/index\.html"',
        'href="../../../index.html"',
        content
    )
    fixes_applied += count

    # Pattern 3: Fix ../../../../../en/index.html -> ../../../../index.html
    content, count = re.subn(
        r'href="\.\.\/\.\.\/\.\.\/\.\.\/\.\.\/en/index\.html"',
        'href="../../../../index.html"',
        content
    )
    fixes_applied += count

    # Pattern 4: Fix ../../../../../jp/index.html (remove JP references)
    content, count = re.subn(
        r'<a[^>]*href="\.\.\/\.\.\/\.\.\/\.\.\/\.\.\/jp/index\.html"[^>]*>.*?</a>',
        '',
        content,
        flags=re.DOTALL
    )
    fixes_applied += count

    # Pattern 5: Fix ../../../en/knowledge/index.html references
    # These should point to root index
    content, count = re.subn(
        r'href="\.\.\/\.\.\/\.\.\/en/knowledge/index\.html"',
        'href="../../../index.html"',
        content
    )
    fixes_applied += count

    # Pattern 6: Fix ../../../jp/knowledge/index.html (remove)
    content, count = re.subn(
        r'<a[^>]*href="\.\.\/\.\.\/\.\.\/jp/knowledge/index\.html"[^>]*>.*?</a>',
        '',
        content,
        flags=re.DOTALL
    )
    fixes_applied += count

    # Pattern 7: Remove references to non-existent series index pages
    non_existent_series = [
        'inferential-bayesian-statistics',
        'robotic-lab-automation-introduction',
        'gnn-features-comparison',
        'materials-screening-workflow',
        'process-monitoring',
        'process-optimization',
        'ml-introduction',  # This exists but has wrong path
    ]

    for series in non_existent_series:
        # Remove entire link elements
        pattern = rf'<a[^>]*href="\.\./{series}/index\.html"[^>]*>.*?</a>'
        content, count = re.subn(pattern, '', content, flags=re.DOTALL)
        fixes_applied += count

        # Also try without ../
        pattern = rf'<a[^>]*href="\.\/{series}/index\.html"[^>]*>.*?</a>'
        content, count = re.subn(pattern, '', content, flags=re.DOTALL)
        fixes_applied += count

        # Remove list items containing these links
        pattern = rf'<li[^>]*>\s*<a[^>]*href="[^"]*{series}/index\.html"[^>]*>.*?</a>\s*</li>'
        content, count = re.subn(pattern, '', content, flags=re.DOTALL)
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

def get_files_with_missing_index():
    """Get list of files with missing index references from link check report."""

    report_path = Path("linkcheck_en_local.txt")
    if not report_path.exists():
        return []

    files_to_fix = set()
    current_file = None

    with open(report_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('File: '):
                current_file = line.replace('File: ', '').strip()
            elif line.startswith('Line ') and current_file:
                link = line.split(': ', 1)[1].strip() if ': ' in line else ''
                # Check if it's an index.html link
                if 'index.html' in link:
                    files_to_fix.add(current_file)

    return sorted(files_to_fix)

def main():
    base_dir = Path("knowledge/en")

    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        sys.exit(1)

    print("Fixing missing index references in HTML files...\n")

    # Get files with missing index
    files_to_fix = get_files_with_missing_index()

    if not files_to_fix:
        print("No files with missing index references found.")
        return

    print(f"Found {len(files_to_fix)} files with index references\n")

    files_fixed = 0
    total_fixes = 0

    for file_rel_path in files_to_fix:
        file_path = base_dir / file_rel_path

        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        fixed, message = fix_missing_index_in_file(file_path)
        if fixed:
            files_fixed += 1
            print(f"✓ Fixed: {file_rel_path}")
            print(f"  {message}")

            # Extract number of fixes
            if "Applied" in message:
                num = int(message.split()[1])
                total_fixes += num

    print("\n" + "="*60)
    print(f"Summary:")
    print(f"  Files checked: {len(files_to_fix)}")
    print(f"  Files fixed: {files_fixed}")
    print(f"  Total fixes applied: {total_fixes}")
    print("="*60)

if __name__ == "__main__":
    main()
