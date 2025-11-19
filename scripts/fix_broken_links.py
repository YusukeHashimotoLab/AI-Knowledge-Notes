#!/usr/bin/env python3
"""
Comprehensive Link Fixing Script for AI Homepage

This script fixes broken links in HTML files based on patterns identified in link check reports.
It handles absolute path corrections, relative path depth issues, asset paths, and non-existent series links.

Author: Claude Code
Date: 2025-11-16
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from datetime import datetime

try:
    from bs4 import BeautifulSoup
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install beautifulsoup4 lxml tqdm")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class LinkFix:
    """Represents a single link fix operation."""
    file_path: Path
    line_number: int
    tag_name: str
    attribute: str
    old_value: str
    new_value: str
    pattern_type: str

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_number} [{self.pattern_type}] {self.old_value} → {self.new_value}"


@dataclass
class FixStats:
    """Statistics for link fixing operations."""
    files_processed: int = 0
    files_modified: int = 0
    total_fixes: int = 0
    fixes_by_pattern: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: List[str] = field(default_factory=list)

    def add_fix(self, pattern_type: str):
        """Record a fix."""
        self.total_fixes += 1
        self.fixes_by_pattern[pattern_type] += 1

    def report(self) -> str:
        """Generate statistics report."""
        lines = [
            "=" * 80,
            "Link Fix Report",
            "=" * 80,
            f"Files Processed: {self.files_processed}",
            f"Files Modified: {self.files_modified}",
            f"Total Fixes Applied: {self.total_fixes}",
            "",
            "Fixes by Pattern:",
        ]

        for pattern, count in sorted(self.fixes_by_pattern.items()):
            lines.append(f"  {pattern}: {count}")

        if self.errors:
            lines.extend([
                "",
                f"Errors Encountered: {len(self.errors)}",
                "See log for details."
            ])

        lines.append("=" * 80)
        return "\n".join(lines)


class LinkFixer:
    """Main class for fixing broken links in HTML files."""

    # Dojos (top-level categories)
    DOJOS = {'FM', 'MI', 'ML', 'MS', 'PI', 'NM'}

    # Known missing series (to comment out or fix path)
    MISSING_SERIES = {
        'llm-basics',
        'machine-learning-basics',
        'robotic-lab-automation-introduction',
        'inferential-bayesian-statistics',
    }

    def __init__(self, base_dir: Path, dry_run: bool = False, verbose: bool = False):
        """
        Initialize LinkFixer.

        Args:
            base_dir: Base directory containing knowledge/en/
            dry_run: If True, don't actually modify files
            verbose: Enable verbose logging
        """
        self.base_dir = Path(base_dir)
        self.knowledge_dir = self.base_dir / "knowledge" / "en"
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = FixStats()
        self.fixes: List[LinkFix] = []

        if not self.knowledge_dir.exists():
            raise ValueError(f"Knowledge directory not found: {self.knowledge_dir}")

        if verbose:
            logger.setLevel(logging.DEBUG)

    def calculate_relative_path(self, from_path: Path, to_path: Path) -> str:
        """
        Calculate relative path from one file to another.

        Args:
            from_path: Source file path
            to_path: Target file path

        Returns:
            Relative path string
        """
        try:
            # Get relative path
            rel_path = to_path.relative_to(from_path.parent)
            return str(rel_path)
        except ValueError:
            # Files are in different directory trees, need to go up
            from_parts = from_path.parent.parts
            to_parts = to_path.parts

            # Find common ancestor
            common_length = 0
            for i, (f, t) in enumerate(zip(from_parts, to_parts)):
                if f == t:
                    common_length = i + 1
                else:
                    break

            # Calculate how many levels to go up
            levels_up = len(from_parts) - common_length
            remaining_path = Path(*to_parts[common_length:])

            # Build relative path
            if levels_up == 0:
                return str(remaining_path)
            else:
                return str(Path(*(['..'] * levels_up)) / remaining_path)

    def get_file_context(self, file_path: Path) -> Dict[str, str]:
        """
        Extract context information from file path.

        Args:
            file_path: Path to HTML file

        Returns:
            Dictionary with dojo, series, filename info
        """
        try:
            rel_path = file_path.relative_to(self.knowledge_dir)
            parts = rel_path.parts

            context = {
                'dojo': parts[0] if len(parts) > 0 else None,
                'series': parts[1] if len(parts) > 1 else None,
                'filename': parts[-1] if len(parts) > 0 else None,
                'depth': len(parts) - 1,  # Depth from knowledge/en/
            }

            # Determine file type
            if context['filename'] == 'index.html':
                if context['depth'] == 0:
                    context['type'] = 'knowledge_index'
                elif context['depth'] == 1:
                    context['type'] = 'dojo_index'
                elif context['depth'] == 2:
                    context['type'] = 'series_index'
            elif context['filename'] and context['filename'].startswith('chapter'):
                context['type'] = 'chapter'
            else:
                context['type'] = 'other'

            return context

        except ValueError:
            return {'dojo': None, 'series': None, 'filename': None, 'depth': 0, 'type': 'other'}

    def fix_absolute_knowledge_paths(self, link: str, context: Dict[str, str]) -> Optional[str]:
        """
        Fix Pattern 1: Absolute /knowledge/en/ paths → Relative paths.

        Args:
            link: Original link
            context: File context information

        Returns:
            Fixed link or None if no fix needed
        """
        if not link.startswith('/knowledge/en/'):
            return None

        # Remove /knowledge/en/ prefix
        target = link.replace('/knowledge/en/', '')

        # Calculate relative path based on current depth
        depth = context['depth']

        if depth == 0:
            # Already at knowledge/en/
            return target
        elif depth == 1:
            # At dojo level (e.g., MI/)
            return f"../{target}"
        elif depth == 2:
            # At series level (e.g., MI/gnn-introduction/)
            return f"../../{target}"
        else:
            # Unknown depth, use depth-based calculation
            prefix = '../' * depth
            return f"{prefix}{target}"

    def fix_breadcrumb_depth(self, link: str, context: Dict[str, str]) -> Optional[str]:
        """
        Fix Pattern 2: Path depth issues in breadcrumbs.

        Chapter files going ../../../index.html should be ../../index.html

        Args:
            link: Original link
            context: File context information

        Returns:
            Fixed link or None if no fix needed
        """
        # Pattern: ../../../index.html in chapter file (depth=2)
        if link == '../../../index.html' and context['depth'] == 2:
            return '../../index.html'

        # Pattern: ../../index.html in series index (depth=2) or dojo index (depth=1)
        if link == '../../index.html':
            if context['type'] == 'series_index' or context['depth'] == 2:
                return '../index.html'
            elif context['type'] == 'dojo_index' or context['depth'] == 1:
                return './index.html'

        # Pattern: ../../FM/index.html from dojo level should be ../FM/index.html
        if link.startswith('../../') and context['depth'] == 1:
            # At dojo level, ../../ should be ../
            return link.replace('../../', '../', 1)

        return None

    def fix_asset_paths(self, link: str, context: Dict[str, str]) -> Optional[str]:
        """
        Fix Pattern 3: Asset paths /assets/ → relative paths.

        Args:
            link: Original link
            context: File context information

        Returns:
            Fixed link or None if no fix needed
        """
        if not link.startswith('/assets/'):
            return None

        # Assets are at knowledge/en/assets/
        depth = context['depth']

        if depth == 0:
            # At knowledge/en/, assets are in ./assets/
            return link.replace('/assets/', 'assets/')
        elif depth == 1:
            # At dojo level
            return link.replace('/assets/', '../assets/')
        elif depth == 2:
            # At series level
            return link.replace('/assets/', '../../assets/')
        else:
            # General case
            prefix = '../' * depth
            return link.replace('/assets/', f'{prefix}assets/')

    def fix_nonexistent_series(self, link: str, context: Dict[str, str]) -> Optional[Tuple[str, str]]:
        """
        Fix Pattern 4: Non-existent series links.

        Returns tuple of (fixed_link, action) where action is 'fix' or 'comment'

        Args:
            link: Original link
            context: File context information

        Returns:
            Tuple of (fixed_link, action) or None if no fix needed
        """
        # Check if link references a missing series
        for missing in self.MISSING_SERIES:
            if f'/{missing}/' in link or link.endswith(f'/{missing}/index.html'):
                # Try to fix cross-dojo references
                if link.startswith('../') and not link.startswith('../../'):
                    # Likely missing dojo prefix
                    # Example: ../pi-introduction/ from MI should be ../../PI/pi-introduction/

                    # Try to infer correct dojo
                    if 'pi-' in missing or missing == 'pi-introduction':
                        fixed = link.replace('../', '../../PI/', 1)
                        return (fixed, 'fix')

                # Can't fix, comment out
                return (link, 'comment')

        return None

    def fix_wrong_filename(self, link: str, file_path: Path) -> Optional[str]:
        """
        Fix Pattern 5: Wrong file references.

        Args:
            link: Original link
            file_path: Current file path

        Returns:
            Fixed link or None if no fix needed
        """
        # Check if linked file exists or is special type
        if (link.startswith('http://') or link.startswith('https://') or
            link.startswith('#') or link.startswith('mailto:') or
            link.startswith('tel:')):
            return None

        # Skip anchors
        link_without_anchor = link.split('#')[0]
        if not link_without_anchor:
            return None

        # Skip directory links (ending with /)
        if link_without_anchor.endswith('/'):
            return None

        # Resolve target path
        try:
            if link_without_anchor.startswith('/'):
                # Absolute path from base
                target = self.base_dir / link_without_anchor.lstrip('/')
            else:
                # Relative path
                target = (file_path.parent / link_without_anchor).resolve()

            if target.exists():
                return None

            # Try to find similar file
            target_dir = target.parent
            target_name = target.name

            if not target_dir.exists():
                return None

            # Look for similar files
            similar_files = []
            for f in target_dir.iterdir():
                if f.is_file() and f.suffix == '.html':
                    # Skip if trying to match index.html with chapter files or vice versa
                    target_is_index = 'index' in target_name.lower()
                    candidate_is_index = 'index' in f.name.lower()
                    target_is_chapter = 'chapter' in target_name.lower()
                    candidate_is_chapter = 'chapter' in f.name.lower()

                    # Don't match index to chapter or vice versa
                    if target_is_index != candidate_is_index:
                        continue
                    if target_is_chapter != candidate_is_chapter:
                        continue

                    # Check similarity (but not the same name)
                    if f.name != target_name and self._files_similar(target_name, f.name):
                        similar_files.append(f)

            if len(similar_files) == 1:
                # Found unique match
                fixed_name = similar_files[0].name

                # If original link had an anchor, preserve it
                if '#' in link:
                    anchor = link.split('#', 1)[1]
                    fixed_name = f"{fixed_name}#{anchor}"

                # Preserve directory path if present
                if '/' in link:
                    dir_part = link.rsplit('/', 1)[0]
                    return f"{dir_part}/{fixed_name}"
                else:
                    return fixed_name

        except (ValueError, OSError) as e:
            logger.debug(f"Error checking link {link} from {file_path}: {e}")

        return None

    def _files_similar(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """
        Check if two filenames are similar.

        Args:
            name1: First filename
            name2: Second filename
            threshold: Similarity threshold (0-1)

        Returns:
            True if files are similar
        """
        # Remove extensions for comparison
        name1_base = name1.replace('.html', '')
        name2_base = name2.replace('.html', '')

        # Extract chapter numbers if present
        import re
        chapter_pattern = r'chapter[-_]?(\d+)'

        match1 = re.search(chapter_pattern, name1_base, re.IGNORECASE)
        match2 = re.search(chapter_pattern, name2_base, re.IGNORECASE)

        # If both have chapter numbers and they're different, not similar
        if match1 and match2:
            if match1.group(1) != match2.group(1):
                return False

        # Remove common prefixes/suffixes for comparison
        clean1 = name1_base.replace('chapter', '').replace('index', '')
        clean2 = name2_base.replace('chapter', '').replace('index', '')

        # Normalize separators
        clean1 = clean1.replace('-', '').replace('_', '').lower()
        clean2 = clean2.replace('-', '').replace('_', '').lower()

        # Check if one contains the other (must be substantial overlap)
        if len(clean1) >= 3 and len(clean2) >= 3:
            if clean1 in clean2 or clean2 in clean1:
                # Check the overlap is significant
                # Use a lower threshold (0.6) if one fully contains the other
                overlap = min(len(clean1), len(clean2))
                longer = max(len(clean1), len(clean2))
                if overlap / longer >= 0.6:  # Lower threshold for containment
                    return True

        # If they're very short and one contains the other, consider similar
        if (len(clean1) < 3 or len(clean2) < 3) and (clean1 in clean2 or clean2 in clean1):
            return True

        # Check character-by-character similarity
        if len(clean1) == 0 or len(clean2) == 0:
            return False

        matches = sum(1 for a, b in zip(clean1, clean2) if a == b)
        longer = max(len(clean1), len(clean2))
        similarity = matches / longer

        return similarity >= threshold

    def process_html_file(self, file_path: Path) -> List[LinkFix]:
        """
        Process a single HTML file and identify fixes.

        Args:
            file_path: Path to HTML file

        Returns:
            List of LinkFix objects
        """
        fixes = []
        context = self.get_file_context(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            soup = BeautifulSoup(content, 'lxml')

            # Process all links (a tags with href, link tags with href, img/script with src)
            for tag in soup.find_all(['a', 'link', 'img', 'script']):
                attr = 'href' if tag.name in ['a', 'link'] else 'src'
                link = tag.get(attr)

                if not link or link.startswith('#') or link.startswith('http://') or link.startswith('https://'):
                    continue

                original_link = link
                fixed_link = None
                pattern_type = None

                # Apply fix patterns in order
                # Pattern 1a: Absolute /knowledge/en/ paths
                if link.startswith('/knowledge/en/'):
                    fixed_link = self.fix_absolute_knowledge_paths(link, context)
                    pattern_type = 'absolute_knowledge_path'

                # Pattern 1b: Absolute /en/ paths (site root paths)
                elif link.startswith('/en/'):
                    # /en/ paths are from site root, need to adjust based on depth
                    # knowledge/en/ files are 2 levels deep from site root
                    fixed_link = '../../../' + link.lstrip('/')
                    pattern_type = 'absolute_site_path'

                # Pattern 3: Asset paths
                elif link.startswith('/assets/'):
                    fixed_link = self.fix_asset_paths(link, context)
                    pattern_type = 'asset_path'

                # Pattern 2: Breadcrumb depth
                elif '../../../index.html' in link or '../../index.html' in link:
                    fixed_link = self.fix_breadcrumb_depth(link, context)
                    pattern_type = 'breadcrumb_depth'

                # Pattern 4: Non-existent series
                result = self.fix_nonexistent_series(link, context)
                if result:
                    fixed_link, action = result
                    if action == 'comment':
                        # Mark for commenting out
                        pattern_type = 'nonexistent_series_comment'
                    else:
                        pattern_type = 'nonexistent_series_fix'

                # Pattern 5: Wrong filename
                if not fixed_link:
                    fixed_link = self.fix_wrong_filename(link, file_path)
                    if fixed_link:
                        pattern_type = 'wrong_filename'

                # Record fix if found
                if fixed_link and fixed_link != original_link:
                    # Find line number (approximate)
                    line_number = content[:content.find(str(tag))].count('\n') + 1 if str(tag) in content else 0

                    fixes.append(LinkFix(
                        file_path=file_path,
                        line_number=line_number,
                        tag_name=tag.name,
                        attribute=attr,
                        old_value=original_link,
                        new_value=fixed_link,
                        pattern_type=pattern_type or 'unknown'
                    ))

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats.errors.append(f"{file_path}: {e}")

        return fixes

    def apply_fixes(self, file_path: Path, fixes: List[LinkFix]) -> bool:
        """
        Apply fixes to a file.

        Args:
            file_path: Path to file
            fixes: List of fixes to apply

        Returns:
            True if file was modified
        """
        if not fixes:
            return False

        try:
            # Create backup
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')

            if not self.dry_run:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Save backup
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                # Apply fixes using BeautifulSoup for accurate replacement
                soup = BeautifulSoup(content, 'lxml')

                for fix in fixes:
                    # Find and update tags
                    for tag in soup.find_all(fix.tag_name):
                        attr_value = tag.get(fix.attribute)
                        if attr_value == fix.old_value:
                            if fix.pattern_type == 'nonexistent_series_comment':
                                # Comment out the entire tag
                                tag.replace_with(f'<!-- TODO: Add when series exists - {tag} -->')
                            else:
                                # Update attribute
                                tag[fix.attribute] = fix.new_value

                # Write modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(soup))

                logger.info(f"Applied {len(fixes)} fixes to {file_path}")

            # Record stats
            for fix in fixes:
                self.stats.add_fix(fix.pattern_type)

            return True

        except Exception as e:
            logger.error(f"Error applying fixes to {file_path}: {e}")
            self.stats.errors.append(f"{file_path}: {e}")
            return False

    def process_all_files(self) -> None:
        """Process all HTML files in knowledge directory."""
        # Find all HTML files
        html_files = list(self.knowledge_dir.rglob('*.html'))

        logger.info(f"Found {len(html_files)} HTML files to process")

        # Process files with progress bar
        for file_path in tqdm(html_files, desc="Processing files", disable=not sys.stdout.isatty()):
            self.stats.files_processed += 1

            # Process file
            fixes = self.process_html_file(file_path)

            if fixes:
                # Log fixes if verbose
                if self.verbose:
                    for fix in fixes:
                        logger.debug(str(fix))

                # Apply fixes
                if self.apply_fixes(file_path, fixes):
                    self.stats.files_modified += 1
                    self.fixes.extend(fixes)

    def write_report(self, output_path: Path) -> None:
        """
        Write detailed fix report.

        Args:
            output_path: Path to output report file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Link Fix Report - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                # Write statistics
                f.write(self.stats.report())
                f.write("\n\n")

                # Write detailed fixes
                f.write("Detailed Fixes:\n")
                f.write("-" * 80 + "\n\n")

                # Group by file
                fixes_by_file = defaultdict(list)
                for fix in self.fixes:
                    fixes_by_file[fix.file_path].append(fix)

                for file_path in sorted(fixes_by_file.keys()):
                    f.write(f"\nFile: {file_path.relative_to(self.knowledge_dir)}\n")
                    for fix in fixes_by_file[file_path]:
                        f.write(f"  Line {fix.line_number}: [{fix.pattern_type}]\n")
                        f.write(f"    {fix.old_value} → {fix.new_value}\n")

            logger.info(f"Report written to {output_path}")

        except Exception as e:
            logger.error(f"Error writing report: {e}")

    def restore_backups(self) -> None:
        """Restore all files from backups."""
        backup_files = list(self.knowledge_dir.rglob('*.html.bak'))

        logger.info(f"Found {len(backup_files)} backup files")

        for backup_path in tqdm(backup_files, desc="Restoring backups"):
            original_path = backup_path.with_suffix('')

            try:
                # Restore from backup
                with open(backup_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                with open(original_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                # Remove backup
                backup_path.unlink()

                logger.debug(f"Restored {original_path}")

            except Exception as e:
                logger.error(f"Error restoring {backup_path}: {e}")

        logger.info("All backups restored")

    def clean_backups(self) -> None:
        """Remove all backup files."""
        backup_files = list(self.knowledge_dir.rglob('*.html.bak'))

        logger.info(f"Found {len(backup_files)} backup files to remove")

        for backup_path in backup_files:
            try:
                backup_path.unlink()
                logger.debug(f"Removed {backup_path}")
            except Exception as e:
                logger.error(f"Error removing {backup_path}: {e}")

        logger.info("All backups removed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fix broken links in HTML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be fixed
  %(prog)s --dry-run

  # Apply fixes
  %(prog)s

  # Apply fixes with verbose logging
  %(prog)s --verbose

  # Restore from backups (undo)
  %(prog)s --restore

  # Clean backup files
  %(prog)s --clean-backups
        """
    )

    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path.cwd(),
        help='Base directory (default: current directory)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fixed without modifying files'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('link_fix_report.txt'),
        help='Output report file (default: link_fix_report.txt)'
    )

    parser.add_argument(
        '--restore',
        action='store_true',
        help='Restore files from backups (undo fixes)'
    )

    parser.add_argument(
        '--clean-backups',
        action='store_true',
        help='Remove all backup files'
    )

    args = parser.parse_args()

    try:
        fixer = LinkFixer(
            base_dir=args.base_dir,
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        if args.restore:
            # Restore from backups
            logger.info("Restoring files from backups...")
            fixer.restore_backups()
            logger.info("Restore complete")
            return 0

        if args.clean_backups:
            # Clean backups
            logger.info("Cleaning backup files...")
            fixer.clean_backups()
            logger.info("Cleanup complete")
            return 0

        # Process files
        logger.info(f"Starting link fixing (dry_run={args.dry_run})...")
        fixer.process_all_files()

        # Print summary
        print("\n" + fixer.stats.report())

        # Write detailed report
        fixer.write_report(args.output)

        if args.dry_run:
            print("\nDRY RUN: No files were modified")
            print("Run without --dry-run to apply fixes")
        else:
            print(f"\nBackup files created with .bak extension")
            print(f"To undo changes, run: {sys.argv[0]} --restore")

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
