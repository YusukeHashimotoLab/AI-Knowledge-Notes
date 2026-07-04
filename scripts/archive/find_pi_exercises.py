#!/usr/bin/env python3
"""
Find PI files with exercise sections.

Shows which chapter files contain exercises and their current format.
"""

import re
from pathlib import Path
from typing import List, Tuple


def find_exercise_files(base_dir: Path) -> List[Tuple[Path, int, bool]]:
    """Find all PI files with exercise sections.

    Returns:
        List of (file_path, exercise_count, has_details_format) tuples
    """
    pi_dir = base_dir / 'knowledge' / 'en' / 'PI'
    files_with_exercises = []

    exercise_pattern = re.compile(
        r'<h2>[\d.]*\s*Exercises?</h2>',
        re.IGNORECASE
    )

    exercise_item_pattern = re.compile(
        r'<h4>Exercise\s+\d+',
        re.IGNORECASE
    )

    details_pattern = re.compile(
        r'<details>\s*<summary>💡\s*Hint</summary>',
        re.IGNORECASE
    )

    for html_file in sorted(pi_dir.glob('**/*.html')):
        if not html_file.stem.startswith('chapter'):
            continue

        try:
            content = html_file.read_text(encoding='utf-8')

            # Check for exercise section
            if not exercise_pattern.search(content):
                continue

            # Count exercises
            exercise_count = len(exercise_item_pattern.findall(content))

            # Check if already has details format
            has_details = bool(details_pattern.search(content))

            files_with_exercises.append((html_file, exercise_count, has_details))

        except Exception as e:
            print(f"Error reading {html_file}: {e}")

    return files_with_exercises


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent

    print("PI Files with Exercise Sections")
    print("=" * 80)
    print("")

    files = find_exercise_files(base_dir)

    if not files:
        print("No files with exercises found.")
        return

    # Categorize files
    converted = [f for f in files if f[2]]
    not_converted = [f for f in files if not f[2]]

    print(f"Found {len(files)} files with exercises:")
    print(f"  - Already converted: {len(converted)}")
    print(f"  - Need conversion: {len(not_converted)}")
    print("")

    if not_converted:
        print("Files Needing Conversion:")
        print("-" * 80)
        total_exercises = 0
        for file_path, ex_count, _ in not_converted:
            rel_path = file_path.relative_to(base_dir)
            print(f"  {rel_path}")
            print(f"    Exercises: {ex_count}")
            total_exercises += ex_count
        print("")
        print(f"Total exercises to convert: {total_exercises}")
        print("")

    if converted:
        print("Already Converted Files:")
        print("-" * 80)
        for file_path, ex_count, _ in converted:
            rel_path = file_path.relative_to(base_dir)
            print(f"  {rel_path}")
            print(f"    Exercises: {ex_count}")
        print("")

    print("=" * 80)
    print("")
    print("To convert files, run:")
    print("  python scripts/convert_exercises_pi.py --dry-run  # preview")
    print("  python scripts/convert_exercises_pi.py            # convert")


if __name__ == '__main__':
    main()
