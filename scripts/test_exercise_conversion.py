#!/usr/bin/env python3
"""
Test script for exercise conversion.

Validates the conversion output before running on all files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from convert_exercises_pi import PIExerciseConverter


def test_conversion():
    """Test conversion on a sample file."""
    base_dir = Path(__file__).parent.parent
    test_file = base_dir / 'knowledge/en/PI/process-data-analysis/chapter-1.html'

    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        return False

    print("Testing Exercise Conversion")
    print("=" * 60)
    print(f"Test file: {test_file.name}\n")

    # Create converter
    converter = PIExerciseConverter(base_dir, dry_run=True, verbose=False)

    # Read file
    content = test_file.read_text(encoding='utf-8')

    # Extract section
    section_data = converter.extract_exercise_section(content)
    if not section_data:
        print("ERROR: No exercise section found")
        return False

    section_content, start_pos, end_pos = section_data

    print("Original Section:")
    print("-" * 60)
    print(section_content[:500])
    print("...\n")

    # Convert
    new_section = converter.convert_section(section_content)

    print("Converted Section:")
    print("-" * 60)
    print(new_section[:1000])
    print("...\n")

    # Check conversions
    print("Validation:")
    print("-" * 60)

    # Check difficulty standardization
    if '(Basic)' in new_section:
        print("❌ FAIL: 'Basic' not converted to 'Easy'")
        return False
    if '(Intermediate)' in new_section:
        print("❌ FAIL: 'Intermediate' not converted to 'Medium'")
        return False
    if '(Advanced)' in new_section:
        print("❌ FAIL: 'Advanced' not converted to 'Hard'")
        return False
    print("✓ Difficulty labels standardized")

    # Check details elements
    hint_count = new_section.count('<details>')
    expected_hints = converter.stats['exercises_converted']
    expected_solutions = converter.stats['exercises_converted']
    expected_total = expected_hints + expected_solutions

    if converter.stats['hints_converted'] > 0:
        expected_total += 1  # Original hint callout converted

    if hint_count < expected_total:
        print(f"❌ FAIL: Expected {expected_total} details elements, found {hint_count}")
        return False
    print(f"✓ All exercises have details elements ({hint_count} total)")

    # Check for hint summaries
    hint_summary_count = new_section.count('<summary>💡 Hint</summary>')
    if hint_summary_count == 0:
        print("❌ FAIL: No hint summaries found")
        return False
    print(f"✓ Hint summaries present ({hint_summary_count})")

    # Check for solution summaries
    solution_summary_count = new_section.count('<summary>📝 Sample Solution</summary>')
    if solution_summary_count == 0:
        print("❌ FAIL: No solution summaries found")
        return False
    print(f"✓ Solution summaries present ({solution_summary_count})")

    # Check for old callout format
    if 'class="callout' in new_section:
        print("❌ FAIL: Old callout format still present")
        return False
    print("✓ No old callout format remaining")

    # Check structure preservation
    if '<h4>Exercise' not in new_section:
        print("❌ FAIL: Exercise headers missing")
        return False
    print("✓ Exercise headers preserved")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_conversion()
    sys.exit(0 if success else 1)
