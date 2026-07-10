#!/usr/bin/env python3
"""
Fix broken asset paths in HTML files.

Changes all broken CSS/JS references to use the existing knowledge-base.css
or removes them if they're not needed.
"""

import os
import sys
from pathlib import Path
import re

def fix_asset_paths_in_file(file_path):
    """Fix asset paths in a single HTML file."""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    fixes_applied = 0

    # Pattern 1: Remove broken individual CSS files (variables, reset, base, etc.)
    # These should be consolidated into knowledge-base.css
    broken_css_patterns = [
        r'<link rel="stylesheet" href="\.\.\/\.\.\/assets\/css\/variables\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/assets\/css\/reset\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/assets\/css\/base\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/assets\/css\/layout\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/assets\/css\/components\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/assets\/css\/article\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/assets\/css\/responsive\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets\/css\/variables\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets\/css\/reset\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets\/css\/base\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets\/css\/components\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets\/css\/layout\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets\/css\/article\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/assets\/css\/responsive\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/\.\.\/assets\/css\/responsive\.css">\n?',
        r'<link rel="stylesheet" href="\.\.\/\.\.\/\.\.\/\.\.\/assets\/css\/article\.css">\n?',
    ]

    for pattern in broken_css_patterns:
        content, count = re.subn(pattern, '', content)
        fixes_applied += count

    # Pattern 2: Fix knowledge.css path (../../../../assets/css/knowledge.css)
    # This should point to ../../../assets/css/knowledge-base.css
    pattern_knowledge = r'../../../../assets/css/knowledge\.css'
    replacement_knowledge = '../../../assets/css/knowledge-base.css'
    content, count = re.subn(pattern_knowledge, replacement_knowledge, content)
    fixes_applied += count

    # Pattern 3: Remove broken JS files
    broken_js_patterns = [
        r'<script src="\.\.\/\.\.\/assets\/js\/navigation\.js"></script>\n?',
        r'<script src="\.\.\/\.\.\/assets\/js\/main\.js"></script>\n?',
    ]

    for pattern in broken_js_patterns:
        content, count = re.subn(pattern, '', content)
        fixes_applied += count

    # Pattern 4: Ensure knowledge-base.css is present (add if missing)
    # Determine correct relative path based on file location
    parts = Path(file_path).relative_to('knowledge/en').parts
    depth = len(parts) - 1  # Subtract 1 for the filename itself

    if depth == 2:  # e.g., FM/series/file.html
        correct_css_path = '../../../assets/css/knowledge-base.css'
    elif depth == 1:  # e.g., FM/file.html
        correct_css_path = '../../assets/css/knowledge-base.css'
    else:
        correct_css_path = '../../../assets/css/knowledge-base.css'  # Default

    # Check if knowledge-base.css is already referenced
    if 'knowledge-base.css' not in content:
        # Find the </head> tag and insert before it
        if '</head>' in content:
            css_link = f'    <link rel="stylesheet" href="{correct_css_path}">\n'
            content = content.replace('</head>', css_link + '</head>')
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

def get_files_with_broken_assets():
    """Get list of files with broken asset references from link check report."""

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
                # Check if it's an asset link
                if any(ext in link for ext in ['.css', '.js']):
                    files_to_fix.add(current_file)

    return sorted(files_to_fix)

def main():
    base_dir = Path("knowledge/en")

    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        sys.exit(1)

    print("Fixing asset paths in HTML files...\n")

    # Get files with broken assets
    files_to_fix = get_files_with_broken_assets()

    if not files_to_fix:
        print("No files with broken asset paths found in link check report.")
        return

    print(f"Found {len(files_to_fix)} files with broken asset references\n")

    files_fixed = 0
    total_fixes = 0

    for file_rel_path in files_to_fix:
        file_path = base_dir / file_rel_path

        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        fixed, message = fix_asset_paths_in_file(file_path)
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
