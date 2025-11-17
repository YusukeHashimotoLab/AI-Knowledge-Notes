#!/usr/bin/env python3
"""
Comprehensive QA check for English knowledge base articles.

Checks for:
- Broken internal links
- Common typos
- Format inconsistencies
- Missing required elements
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent / "knowledge" / "en"


def check_internal_links(file_path: Path, all_files: set) -> List[str]:
    """Check for broken internal links."""
    issues = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find all href links
        href_pattern = r'href=["\']([^"\']+)["\']'
        for match in re.finditer(href_pattern, content):
            link = match.group(1)

            # Skip external links
            if link.startswith(('http://', 'https://', 'mailto:', '#')):
                continue

            # Resolve relative links
            if link.startswith('../'):
                target_path = (file_path.parent / link).resolve()
            elif link.startswith('./'):
                target_path = (file_path.parent / link[2:]).resolve()
            else:
                target_path = (file_path.parent / link).resolve()

            # Check if target exists
            if not target_path.exists():
                issues.append(f"Broken link: {link}")

    except Exception as e:
        issues.append(f"Error checking links: {e}")

    return issues


def check_common_typos(file_path: Path) -> List[str]:
    """Check for common typos and format issues."""
    issues = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Common typos patterns
        typo_patterns = [
            (r'\bteh\b', 'teh → the'),
            (r'\badn\b', 'adn → and'),
            (r'\bfro\b(?!m)', 'fro → for'),
            (r'　', 'Full-width space detected'),
            (r'<h[1-6]>[^<]*  [^<]*</h[1-6]>', 'Double space in heading'),
        ]

        for pattern, message in typo_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues.append(f"Line {line_num}: {message}")

    except Exception as e:
        issues.append(f"Error checking typos: {e}")

    return issues


def check_required_elements(file_path: Path) -> List[str]:
    """Check for required HTML elements."""
    issues = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Required elements for chapter files
        if 'chapter' in file_path.name.lower():
            required = [
                (r'<h1>', 'Missing <h1> main heading'),
                (r'<main', 'Missing <main> tag'),
            ]

            for pattern, message in required:
                if not re.search(pattern, content):
                    issues.append(message)

    except Exception as e:
        issues.append(f"Error checking elements: {e}")

    return issues


def check_code_blocks(file_path: Path) -> List[str]:
    """Check code blocks for common issues."""
    issues = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for code blocks without language specification
        unspecified_code = re.findall(r'<pre><code>(?!class=)', content)
        if unspecified_code:
            issues.append(f"Found {len(unspecified_code)} code blocks without language specification")

        # Check for unclosed code tags
        pre_open = len(re.findall(r'<pre>', content))
        pre_close = len(re.findall(r'</pre>', content))
        if pre_open != pre_close:
            issues.append(f"Mismatched <pre> tags: {pre_open} open, {pre_close} close")

        code_open = len(re.findall(r'<code>', content))
        code_close = len(re.findall(r'</code>', content))
        if code_open != code_close:
            issues.append(f"Mismatched <code> tags: {code_open} open, {code_close} close")

    except Exception as e:
        issues.append(f"Error checking code blocks: {e}")

    return issues


def main():
    """Main QA check execution."""
    print("=" * 70)
    print("Comprehensive QA Check")
    print("=" * 70)

    # Collect all HTML files
    html_files = list(BASE_DIR.rglob("*.html"))
    all_file_paths = {f.resolve() for f in html_files}

    print(f"\nScanning {len(html_files)} files...\n")

    # Statistics
    stats = defaultdict(int)
    issues_by_file = {}

    # Process each file
    for html_file in html_files:
        file_issues = []

        # Run checks
        file_issues.extend(check_internal_links(html_file, all_file_paths))
        file_issues.extend(check_common_typos(html_file))
        file_issues.extend(check_required_elements(html_file))
        file_issues.extend(check_code_blocks(html_file))

        # Store results
        if file_issues:
            rel_path = html_file.relative_to(BASE_DIR.parent.parent)
            issues_by_file[str(rel_path)] = file_issues
            stats['files_with_issues'] += 1
            stats['total_issues'] += len(file_issues)

    # Report results
    print("\n" + "=" * 70)
    print("QA Check Results")
    print("=" * 70)

    if issues_by_file:
        print(f"\nFound issues in {stats['files_with_issues']} files:\n")

        # Show first 20 files with issues
        for i, (file_path, issues) in enumerate(list(issues_by_file.items())[:20], 1):
            print(f"{i}. {file_path}")
            for issue in issues[:3]:  # Show first 3 issues per file
                print(f"   - {issue}")
            if len(issues) > 3:
                print(f"   ... and {len(issues) - 3} more issues")
            print()

        if len(issues_by_file) > 20:
            print(f"... and {len(issues_by_file) - 20} more files with issues\n")
    else:
        print("\n✅ No issues found! All checks passed.\n")

    # Summary statistics
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Files scanned:        {len(html_files)}")
    print(f"Files with issues:    {stats['files_with_issues']}")
    print(f"Total issues found:   {stats['total_issues']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
