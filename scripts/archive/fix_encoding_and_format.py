#!/usr/bin/env python3
"""
Fix encoding issues and improper code formatting in HTML files.
Handles files with non-UTF-8 encoding and wraps code-example content properly.
"""

import re
import os
import chardet
from pathlib import Path
from typing import List, Tuple

def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet."""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

def fix_encoding_and_read(file_path: str) -> str:
    """Read file with proper encoding detection."""
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Detect and use actual encoding
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        # Write back as UTF-8
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return content

def find_problematic_files(base_dir: str) -> List[str]:
    """Find all HTML files with improper code-example formatting."""
    problematic = []
    base_path = Path(base_dir)

    for html_file in base_path.rglob("*.html"):
        try:
            content = fix_encoding_and_read(str(html_file))

            # Check if file has code-example divs
            if '<div class="code-example">' in content:
                # Check if any code-example div lacks proper <pre><code> wrapping
                pattern = r'<div class="code-example">(?!\s*<pre><code)'
                if re.search(pattern, content):
                    problematic.append(str(html_file))
        except Exception as e:
            print(f"Error processing {html_file}: {e}")

    return problematic

def detect_language(code_content: str) -> str:
    """Detect programming language from code content."""
    # Python indicators
    if any(keyword in code_content for keyword in ['import ', 'def ', 'class ', 'print(', 'numpy', 'pandas', '__init__']):
        return 'python'

    # JavaScript indicators
    if any(keyword in code_content for keyword in ['const ', 'let ', 'var ', 'function ', '=>', 'console.log']):
        return 'javascript'

    # Default to python (most common in this codebase)
    return 'python'

def fix_code_formatting(file_path: str) -> Tuple[bool, int]:
    """
    Fix code formatting in a single file.
    Returns (success, number_of_fixes)
    """
    try:
        content = fix_encoding_and_read(file_path)
        original_content = content
        fixes_count = 0

        def replace_code_block(match):
            nonlocal fixes_count
            indent = match.group(1)
            code_content = match.group(2)

            # Detect language
            language = detect_language(code_content)

            # Build properly formatted code block
            fixed = f'{indent}<div class="code-example"><pre><code class="language-{language}">{code_content}</code></pre></div>'
            fixes_count += 1
            return fixed

        # Match pattern: <div class="code-example">...content...</div>
        pattern = r'(\s*)<div class="code-example">(?!<pre><code)(.*?)</div>'
        content = re.sub(pattern, replace_code_block, content, flags=re.DOTALL)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, fixes_count

        return False, 0

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False, 0

def ensure_prism_js(file_path: str) -> bool:
    """Ensure file has Prism.js for syntax highlighting."""
    try:
        content = fix_encoding_and_read(file_path)

        # Check if Prism.js is already included
        if 'prism.css' in content and 'prism.js' in content:
            return False

        modified = False

        # Add Prism CSS if missing
        if 'prism.css' not in content and '</head>' in content:
            prism_css = '    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">\n'
            content = content.replace('</head>', f'{prism_css}</head>')
            modified = True

        # Add Prism JS if missing
        if 'prism.js' not in content and '</body>' in content:
            prism_js = '    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>\n    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>\n'
            content = content.replace('</body>', f'{prism_js}</body>')
            modified = True

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error adding Prism.js to {file_path}: {e}")
        return False

def main():
    base_dir = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en"

    print("=" * 80)
    print("CODE FORMATTING AND ENCODING FIX REPORT")
    print("=" * 80)
    print()

    # Find problematic files
    print("Phase 1: Identifying problematic files (with encoding fixes)...")
    problematic_files = find_problematic_files(base_dir)

    print(f"Found {len(problematic_files)} files with improper code formatting\n")

    if not problematic_files:
        print("No files need fixing!")
        return

    # Fix each file
    print("Phase 2: Fixing code formatting...")
    total_fixes = 0
    fixed_files = []
    prism_added = []

    for file_path in problematic_files:
        success, fix_count = fix_code_formatting(file_path)
        if success:
            fixed_files.append(file_path)
            total_fixes += fix_count
            print(f"  ✓ Fixed {fix_count} code blocks in: {os.path.relpath(file_path, base_dir)}")

            # Ensure Prism.js is included
            if ensure_prism_js(file_path):
                prism_added.append(file_path)

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files analyzed: {len(problematic_files)}")
    print(f"Total files fixed: {len(fixed_files)}")
    print(f"Total code blocks fixed: {total_fixes}")
    print(f"Files with Prism.js added: {len(prism_added)}")
    print()

    if fixed_files:
        print("Modified Files:")
        for file_path in fixed_files:
            print(f"  - {os.path.relpath(file_path, base_dir)}")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
