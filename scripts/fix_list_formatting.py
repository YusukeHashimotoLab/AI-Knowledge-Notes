#!/usr/bin/env python3
"""
Fix list formatting in HTML files
Converts dash-separated lists inside <p> tags to proper <ul><li> markup
"""

import re
import sys
from pathlib import Path


def fix_lists_in_content(content):
    """
    Find patterns like:
    <p>Text:
    - Item 1
    - Item 2</p>

    And convert to:
    <p>Text:</p>
    <ul>
    <li>Item 1</li>
    <li>Item 2</li>
    </ul>
    """

    # Pattern: <p> followed by text ending with :, then lines starting with -
    # This is a complex pattern, so we'll do it in multiple passes

    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is a <p> tag with text ending in :
        if '<p>' in line and line.rstrip().endswith(':'):
            # Look ahead to see if next lines are dash items
            dash_items = []
            j = i + 1

            while j < len(lines):
                next_line = lines[j].strip()

                # Check if it's a dash item
                if next_line.startswith('- '):
                    dash_items.append(next_line[2:])  # Remove "- "
                    j += 1
                # Check if it ends the paragraph
                elif next_line.endswith('</p>'):
                    # If we have items, this closes the list
                    if dash_items:
                        # If this line has content before </p>, add it as last item
                        if next_line != '</p>':
                            content_before_close = next_line[:-4].strip()
                            if content_before_close.startswith('- '):
                                dash_items.append(content_before_close[2:])
                    j += 1
                    break
                else:
                    # Not a list pattern
                    break

            # If we found dash items, convert to proper list
            if dash_items:
                result.append(line)  # Add the <p>text:</p>
                result.append('<ul>')
                for item in dash_items:
                    result.append(f'<li>{item}</li>')
                result.append('</ul>')
                i = j
                continue

        # No special processing needed
        result.append(line)
        i += 1

    return '\n'.join(result)


def process_file(file_path):
    """Process a single HTML file"""
    print(f"Processing: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        fixed_content = fix_lists_in_content(content)

        if fixed_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"  ✓ Fixed list formatting")
            return True
        else:
            print(f"  - No changes needed")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Process specific files
        files = [Path(f) for f in sys.argv[1:]]
    else:
        print("Usage: python fix_list_formatting.py <file1.html> [file2.html ...]")
        return

    total = 0
    fixed = 0

    for file_path in files:
        if file_path.exists():
            total += 1
            if process_file(file_path):
                fixed += 1
        else:
            print(f"File not found: {file_path}")

    print(f"\nSummary: {fixed}/{total} files modified")


if __name__ == '__main__':
    main()
