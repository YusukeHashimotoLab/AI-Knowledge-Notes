#!/usr/bin/env python3
"""
Convert PI Exercise Formats to Details Structure

Scans PI HTML files and converts exercise sections from plain format to
collapsible <details> format with hints and sample solutions.

Transformations:
- Standardize difficulty: Basic→Easy, Intermediate→Medium, Advanced→Hard
- Convert existing hint callouts to <details> format
- Add placeholder <details> for hints if none exist
- Add placeholder <details> for sample solutions
- Preserve all other HTML structure

Usage:
    python convert_exercises_pi.py [--dry-run] [--verbose]
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import shutil
from datetime import datetime


@dataclass
class ExerciseBlock:
    """Represents an exercise with its components."""
    number: int
    difficulty: str
    title: str
    content: str
    hint: Optional[str] = None
    solution: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0


class PIExerciseConverter:
    """Converts PI exercise formats to details structure."""

    # Difficulty mapping
    DIFFICULTY_MAP = {
        'Basic': 'Easy',
        'Intermediate': 'Medium',
        'Advanced': 'Hard',
        'Easy': 'Easy',
        'Medium': 'Medium',
        'Hard': 'Hard'
    }

    # Pattern for exercise headers
    EXERCISE_PATTERN = re.compile(
        r'<h4>Exercise\s+(\d+)\s*\(([^)]+)\):\s*([^<]+)</h4>',
        re.IGNORECASE
    )

    # Pattern for existing hint callouts (more flexible)
    HINT_CALLOUT_PATTERN = re.compile(
        r'<div\s+class="callout\s+callout-tip">\s*'
        r'<h4>💡\s*Hint</h4>\s*'
        r'(.*?)'
        r'</div>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern for exercise section
    EXERCISE_SECTION_PATTERN = re.compile(
        r'(<section>\s*<h2>[\d.]*\s*Exercises?</h2>.*?</section>)',
        re.DOTALL | re.IGNORECASE
    )

    def __init__(self, base_dir: Path, dry_run: bool = False, verbose: bool = False):
        """Initialize converter.

        Args:
            base_dir: Base directory containing PI HTML files
            dry_run: If True, only show what would be changed
            verbose: If True, print detailed progress
        """
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'exercises_converted': 0,
            'hints_converted': 0,
            'hints_added': 0,
            'solutions_added': 0
        }

    def find_pi_files(self) -> List[Path]:
        """Find all HTML files in PI directory."""
        pi_dir = self.base_dir / 'knowledge' / 'en' / 'PI'
        if not pi_dir.exists():
            raise FileNotFoundError(f"PI directory not found: {pi_dir}")

        html_files = list(pi_dir.glob('**/*.html'))
        # Exclude index files as they typically don't have exercises
        html_files = [f for f in html_files if f.stem.startswith('chapter')]

        if self.verbose:
            print(f"Found {len(html_files)} chapter files in PI directory")

        return sorted(html_files)

    def extract_exercise_section(self, content: str) -> Optional[Tuple[str, int, int]]:
        """Extract exercise section from HTML content.

        Returns:
            Tuple of (exercise_section, start_pos, end_pos) or None
        """
        match = self.EXERCISE_SECTION_PATTERN.search(content)
        if match:
            return match.group(1), match.start(), match.end()
        return None

    def parse_exercises(self, section_content: str) -> List[ExerciseBlock]:
        """Parse exercises from section content.

        Args:
            section_content: HTML content of exercise section

        Returns:
            List of ExerciseBlock objects
        """
        exercises = []

        # Find all exercise headers
        exercise_matches = list(self.EXERCISE_PATTERN.finditer(section_content))

        for i, match in enumerate(exercise_matches):
            ex_num = int(match.group(1))
            difficulty = match.group(2).strip()
            title = match.group(3).strip()

            # Normalize difficulty
            difficulty = self.DIFFICULTY_MAP.get(difficulty, difficulty)

            # Find content between this exercise and next
            start_pos = match.end()
            if i + 1 < len(exercise_matches):
                end_pos = exercise_matches[i + 1].start()
            else:
                # Last exercise - find end of section or hint callout
                hint_match = self.HINT_CALLOUT_PATTERN.search(section_content, start_pos)
                if hint_match:
                    end_pos = hint_match.start()
                else:
                    end_pos = section_content.find('</section>', start_pos)
                    if end_pos == -1:
                        end_pos = len(section_content)

            content = section_content[start_pos:end_pos].strip()

            exercise = ExerciseBlock(
                number=ex_num,
                difficulty=difficulty,
                title=title,
                content=content,
                start_pos=match.start(),
                end_pos=end_pos
            )

            exercises.append(exercise)

        return exercises

    def extract_hint_callout(self, section_content: str) -> Optional[str]:
        """Extract existing hint callout if present.

        Args:
            section_content: HTML content of exercise section

        Returns:
            Hint text or None (preserves HTML tags)
        """
        match = self.HINT_CALLOUT_PATTERN.search(section_content)
        if match:
            # Extract content and preserve HTML structure
            content = match.group(1).strip()
            # Clean up any extra whitespace but preserve tags
            return content
        return None

    def generate_hint_placeholder(self, exercise: ExerciseBlock) -> str:
        """Generate placeholder hint based on exercise difficulty.

        Args:
            exercise: ExerciseBlock object

        Returns:
            HTML details element with hint
        """
        hints = {
            'Easy': 'Think about the basic principles covered in the chapter examples.',
            'Medium': 'Consider the trade-offs between different approaches and parameter settings.',
            'Hard': 'Break down the problem into smaller steps and validate each component.'
        }

        hint_text = hints.get(exercise.difficulty, 'Review the relevant examples and theory.')

        return f'''
<details>
<summary>💡 Hint</summary>
<p>{hint_text}</p>
</details>'''

    def generate_solution_placeholder(self, exercise: ExerciseBlock) -> str:
        """Generate placeholder solution structure.

        Args:
            exercise: ExerciseBlock object

        Returns:
            HTML details element with solution placeholder
        """
        return f'''
<details>
<summary>📝 Sample Solution</summary>
<p><em>Implementation approach:</em></p>
<ul>
<li>Step 1: [Key implementation point]</li>
<li>Step 2: [Analysis or comparison]</li>
<li>Step 3: [Validation and interpretation]</li>
</ul>
</details>'''

    def convert_exercise_to_details(self, exercise: ExerciseBlock,
                                   has_hint: bool = False) -> str:
        """Convert exercise to new format with details.

        Args:
            exercise: ExerciseBlock object
            has_hint: Whether exercise already has a hint

        Returns:
            Formatted HTML for exercise
        """
        html_parts = []

        # Exercise header
        html_parts.append(
            f'<h4>Exercise {exercise.number} ({exercise.difficulty}): {exercise.title}</h4>'
        )

        # Exercise content
        html_parts.append(exercise.content)

        # Add hint if not present
        if not has_hint:
            html_parts.append(self.generate_hint_placeholder(exercise))
            self.stats['hints_added'] += 1

        # Add solution placeholder
        html_parts.append(self.generate_solution_placeholder(exercise))
        self.stats['solutions_added'] += 1

        return '\n'.join(html_parts)

    def convert_hint_callout_to_details(self, hint_text: str) -> str:
        """Convert hint callout to details format.

        Args:
            hint_text: HTML content of hint (may include tags)

        Returns:
            HTML details element
        """
        self.stats['hints_converted'] += 1
        return f'''
<details>
<summary>💡 Hint</summary>
{hint_text}
</details>'''

    def convert_section(self, section_content: str) -> str:
        """Convert entire exercise section to new format.

        Args:
            section_content: HTML content of exercise section

        Returns:
            Converted section content
        """
        # Extract section header
        header_match = re.search(r'<section>\s*(<h2>[\d.]*\s*Exercises?</h2>)',
                                section_content, re.IGNORECASE)
        if not header_match:
            return section_content

        header = header_match.group(1)

        # Parse exercises
        exercises = self.parse_exercises(section_content)

        if not exercises:
            if self.verbose:
                print("  No exercises found in section")
            return section_content

        # Check for existing hint callout
        hint_text = self.extract_hint_callout(section_content)
        has_hint = hint_text is not None

        # Build new section
        new_parts = ['<section>', header]

        # Convert each exercise
        for exercise in exercises:
            converted = self.convert_exercise_to_details(exercise, has_hint=False)
            new_parts.append(converted)
            self.stats['exercises_converted'] += 1

        # Add converted hint if it existed
        if has_hint:
            new_parts.append(self.convert_hint_callout_to_details(hint_text))

        new_parts.append('</section>')

        return '\n'.join(new_parts)

    def process_file(self, file_path: Path) -> bool:
        """Process a single HTML file.

        Args:
            file_path: Path to HTML file

        Returns:
            True if file was modified, False otherwise
        """
        if self.verbose:
            print(f"\nProcessing: {file_path.relative_to(self.base_dir)}")

        self.stats['files_processed'] += 1

        # Read file
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"  ERROR reading file: {e}")
            return False

        # Extract exercise section
        section_data = self.extract_exercise_section(content)
        if not section_data:
            if self.verbose:
                print("  No exercise section found")
            return False

        section_content, start_pos, end_pos = section_data

        # Convert section
        new_section = self.convert_section(section_content)

        # Check if anything changed
        if new_section == section_content:
            if self.verbose:
                print("  No changes needed")
            return False

        # Build new content
        new_content = content[:start_pos] + new_section + content[end_pos:]

        if self.dry_run:
            print(f"  [DRY RUN] Would modify file")
            if self.verbose:
                print(f"  Changes: {self.stats['exercises_converted']} exercises converted")
            return True

        # Create backup
        backup_path = file_path.with_suffix('.html.bak')
        try:
            shutil.copy2(file_path, backup_path)
            if self.verbose:
                print(f"  Created backup: {backup_path.name}")
        except Exception as e:
            print(f"  WARNING: Could not create backup: {e}")

        # Write modified content
        try:
            file_path.write_text(new_content, encoding='utf-8')
            if self.verbose:
                print(f"  ✓ Modified file")
            self.stats['files_modified'] += 1
            return True
        except Exception as e:
            print(f"  ERROR writing file: {e}")
            return False

    def run(self, single_file: Optional[Path] = None) -> None:
        """Run conversion on PI files.

        Args:
            single_file: If provided, process only this file
        """
        print("PI Exercise Format Converter")
        print("=" * 60)

        if self.dry_run:
            print("DRY RUN MODE - No files will be modified\n")

        # Find files
        if single_file:
            # Convert to absolute path
            single_file = single_file.resolve()
            if not single_file.exists():
                print(f"ERROR: File not found: {single_file}")
                sys.exit(1)
            files = [single_file]
            print(f"Processing single file: {single_file.name}\n")
        else:
            try:
                files = self.find_pi_files()
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
                sys.exit(1)

            if not files:
                print("No chapter files found")
                return

            print(f"Processing {len(files)} files...\n")

        # Process each file
        for file_path in files:
            # Reset per-file stats
            prev_exercises = self.stats['exercises_converted']
            prev_hints = self.stats['hints_converted']

            self.process_file(file_path)

            # Show progress
            if self.verbose and self.stats['exercises_converted'] > prev_exercises:
                ex_count = self.stats['exercises_converted'] - prev_exercises
                hint_count = self.stats['hints_converted'] - prev_hints
                print(f"  Converted {ex_count} exercises, {hint_count} hints")

        # Print summary
        print("\n" + "=" * 60)
        print("Conversion Summary")
        print("=" * 60)
        print(f"Files processed:      {self.stats['files_processed']}")
        print(f"Files modified:       {self.stats['files_modified']}")
        print(f"Exercises converted:  {self.stats['exercises_converted']}")
        print(f"Hints converted:      {self.stats['hints_converted']}")
        print(f"Hints added:          {self.stats['hints_added']}")
        print(f"Solutions added:      {self.stats['solutions_added']}")
        print("=" * 60)

        if not self.dry_run and self.stats['files_modified'] > 0:
            print(f"\n✓ Backup files created with .bak extension")
            print("  To revert: mv file.html.bak file.html")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert PI exercise formats to details structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_exercises_pi.py
  python convert_exercises_pi.py --dry-run
  python convert_exercises_pi.py --verbose
  python convert_exercises_pi.py --dry-run --verbose
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress information'
    )

    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path(__file__).parent.parent,
        help='Base directory (default: script parent directory)'
    )

    parser.add_argument(
        '--file',
        type=Path,
        help='Process single file instead of all PI files'
    )

    args = parser.parse_args()

    # Run converter
    converter = PIExerciseConverter(
        base_dir=args.base_dir,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    try:
        converter.run(single_file=args.file)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
