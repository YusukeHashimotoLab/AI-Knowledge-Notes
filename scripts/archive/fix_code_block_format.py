#!/usr/bin/env python3
"""
Fix code-block formatting by adding missing <pre> tags.

Changes:
  <div class="code-block"><code>
    →
  <div class="code-block"><pre><code>

And:
  </code></div>
    →
  </code></pre></div>
"""

import os
import sys
from pathlib import Path
from bs4 import BeautifulSoup
import re

def fix_code_blocks(file_path):
    """Fix code-block formatting in an HTML file."""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if file has the pattern to fix
    if '<div class="code-block"><code>' not in content:
        return False, "No code-block patterns found"

    original_content = content
    fixes_applied = 0

    # Pattern 1: Add <pre> after code-block div opening
    pattern1 = r'<div class="code-block"><code>'
    replacement1 = r'<div class="code-block"><pre><code>'
    content, count1 = re.subn(pattern1, replacement1, content)
    fixes_applied += count1

    # Pattern 2: Add </pre> before code-block div closing
    # Need to find </code></div> that follows a code-block
    # Use regex to match code-block sections
    pattern2 = r'(<div class="code-block"><pre><code>.*?)</code></div>'
    replacement2 = r'\1</code></pre></div>'
    content, count2 = re.subn(pattern2, replacement2, content, flags=re.DOTALL)
    fixes_applied += count2

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

    print("Searching for HTML files with code-block formatting issues...")

    # Find all HTML files
    html_files = list(base_dir.rglob("*.html"))

    files_fixed = 0
    total_fixes = 0

    for html_file in html_files:
        fixed, message = fix_code_blocks(html_file)
        if fixed:
            files_fixed += 1
            print(f"✓ Fixed: {html_file}")
            print(f"  {message}")
            # Extract number of fixes from message
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
