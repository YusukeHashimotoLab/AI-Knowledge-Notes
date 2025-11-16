#!/usr/bin/env python3
"""
Bidirectional synchronization between Markdown and HTML files.
Automatically detects which file is newer and syncs in the appropriate direction.

This script enables efficient workflows where you can edit either Markdown or HTML
and keep them synchronized automatically.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import subprocess

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    # Create dummy classes for when watchdog is not installed
    class FileSystemEventHandler:
        """Dummy handler when watchdog not available."""
        pass

    class Observer:
        """Dummy observer when watchdog not available."""
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base path for English knowledge base
BASE_PATH = Path("knowledge/en")

# Supported Dojos
DOJOS = ["FM", "MI", "ML", "MS", "PI"]


class FilePair:
    """Represents a Markdown-HTML file pair."""

    def __init__(self, md_path: Path):
        """
        Initialize file pair.

        Args:
            md_path: Path to Markdown file (may or may not exist)
        """
        self.md_path = md_path
        self.html_path = md_path.with_suffix('.html')

    @property
    def md_exists(self) -> bool:
        """Check if Markdown file exists."""
        return self.md_path.exists()

    @property
    def html_exists(self) -> bool:
        """Check if HTML file exists."""
        return self.html_path.exists()

    @property
    def md_mtime(self) -> float:
        """Get Markdown file modification time."""
        return self.md_path.stat().st_mtime if self.md_exists else 0

    @property
    def html_mtime(self) -> float:
        """Get HTML file modification time."""
        return self.html_path.stat().st_mtime if self.html_exists else 0

    def needs_sync(self) -> Optional[str]:
        """
        Determine if files need syncing and in which direction.

        Returns:
            'md2html' if MD is newer, 'html2md' if HTML is newer, None if in sync
        """
        if not self.md_exists and not self.html_exists:
            return None

        if self.md_exists and not self.html_exists:
            return 'md2html'

        if self.html_exists and not self.md_exists:
            return 'html2md'

        # Both exist - check modification times
        # Add 1 second tolerance to avoid false positives from file system timing
        time_diff = abs(self.md_mtime - self.html_mtime)
        if time_diff < 1:
            return None

        if self.md_mtime > self.html_mtime:
            return 'md2html'
        else:
            return 'html2md'

    def __str__(self):
        """String representation."""
        return f"FilePair({self.md_path.name})"


def find_file_pairs(directory: Path) -> List[FilePair]:
    """
    Find all Markdown-HTML file pairs in a directory.

    Args:
        directory: Directory to search

    Returns:
        List of FilePair objects
    """
    pairs = []

    # Find all chapter Markdown files
    for md_file in sorted(directory.glob("chapter*.md")):
        pairs.append(FilePair(md_file))

    # Find HTML files without corresponding Markdown
    for html_file in sorted(directory.glob("chapter*.html")):
        md_file = html_file.with_suffix('.md')
        if not md_file.exists():
            # Create pair from HTML perspective
            pairs.append(FilePair(md_file))

    return pairs


def sync_md_to_html(md_path: Path, dry_run: bool = False) -> bool:
    """
    Sync Markdown to HTML using convert_md_to_html_en.py.

    Args:
        md_path: Path to Markdown file
        dry_run: If True, only log what would be done

    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would convert MD→HTML: {md_path}")
        return True

    logger.info(f"Converting MD→HTML: {md_path}")

    try:
        # Import converter module
        converter_script = Path(__file__).parent / "convert_md_to_html_en.py"
        if not converter_script.exists():
            logger.error(f"Converter script not found: {converter_script}")
            return False

        # Call converter
        result = subprocess.run(
            [sys.executable, str(converter_script), str(md_path)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"✓ Successfully converted {md_path.name}")
            return True
        else:
            logger.error(f"✗ Conversion failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"✗ Error converting {md_path}: {e}")
        return False


def sync_html_to_md(html_path: Path, dry_run: bool = False) -> bool:
    """
    Sync HTML to Markdown using html_to_md.py.

    Args:
        html_path: Path to HTML file
        dry_run: If True, only log what would be done

    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would convert HTML→MD: {html_path}")
        return True

    logger.info(f"Converting HTML→MD: {html_path}")

    try:
        # Import converter module
        converter_script = Path(__file__).parent / "html_to_md.py"
        if not converter_script.exists():
            logger.error(f"Converter script not found: {converter_script}")
            return False

        # Call converter
        result = subprocess.run(
            [sys.executable, str(converter_script), str(html_path)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"✓ Successfully converted {html_path.name}")
            return True
        else:
            logger.error(f"✗ Conversion failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"✗ Error converting {html_path}: {e}")
        return False


def sync_file_pair(pair: FilePair, force_direction: Optional[str] = None, dry_run: bool = False) -> bool:
    """
    Synchronize a file pair.

    Args:
        pair: FilePair object
        force_direction: Force sync direction ('md2html' or 'html2md')
        dry_run: If True, only log what would be done

    Returns:
        True if successful or no sync needed, False otherwise
    """
    # Determine sync direction
    if force_direction:
        direction = force_direction
    else:
        direction = pair.needs_sync()

    if not direction:
        logger.debug(f"Files in sync: {pair}")
        return True

    # Perform sync
    if direction == 'md2html':
        if not pair.md_exists:
            logger.warning(f"Cannot sync MD→HTML: {pair.md_path} does not exist")
            return False
        return sync_md_to_html(pair.md_path, dry_run)
    else:  # html2md
        if not pair.html_exists:
            logger.warning(f"Cannot sync HTML→MD: {pair.html_path} does not exist")
            return False
        return sync_html_to_md(pair.html_path, dry_run)


def sync_directory(directory: Path, force_direction: Optional[str] = None, dry_run: bool = False) -> Tuple[int, int]:
    """
    Synchronize all files in a directory.

    Args:
        directory: Directory to synchronize
        force_direction: Force sync direction
        dry_run: If True, only log what would be done

    Returns:
        Tuple of (successful_count, total_count)
    """
    logger.info(f"\nSynchronizing directory: {directory}")
    logger.info("-" * 60)

    pairs = find_file_pairs(directory)

    if not pairs:
        logger.warning(f"No file pairs found in {directory}")
        return 0, 0

    success_count = 0
    sync_needed = 0

    for pair in pairs:
        if force_direction or pair.needs_sync():
            sync_needed += 1
            if sync_file_pair(pair, force_direction, dry_run):
                success_count += 1

    logger.info(f"Synced {success_count}/{sync_needed} files that needed updating")
    return success_count, sync_needed


class SyncEventHandler(FileSystemEventHandler):
    """Event handler for watch mode."""

    def __init__(self, force_direction: Optional[str] = None):
        """Initialize handler."""
        self.force_direction = force_direction
        self.last_sync = {}

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Only handle chapter files
        if not path.name.startswith('chapter'):
            return

        # Only handle .md and .html files
        if path.suffix not in ['.md', '.html']:
            return

        # Debounce - avoid syncing the same file too frequently
        current_time = time.time()
        last_time = self.last_sync.get(str(path), 0)
        if current_time - last_time < 2:  # 2 second debounce
            return

        self.last_sync[str(path)] = current_time

        logger.info(f"\nFile modified: {path}")

        # Create file pair
        if path.suffix == '.md':
            pair = FilePair(path)
        else:
            pair = FilePair(path.with_suffix('.md'))

        # Sync
        sync_file_pair(pair, self.force_direction, dry_run=False)


def watch_mode(paths: List[Path], force_direction: Optional[str] = None):
    """
    Watch directories for changes and auto-sync.

    Args:
        paths: List of directories to watch
        force_direction: Force sync direction
    """
    if not WATCHDOG_AVAILABLE:
        logger.error("Watch mode requires watchdog. Install with: pip install watchdog")
        sys.exit(1)

    logger.info("Starting watch mode...")
    logger.info("Monitoring for file changes (Ctrl+C to stop):")
    for path in paths:
        logger.info(f"  - {path}")
    logger.info("=" * 60)

    event_handler = SyncEventHandler(force_direction)
    observer = Observer()

    for path in paths:
        observer.schedule(event_handler, str(path), recursive=False)

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nStopping watch mode...")
        observer.stop()

    observer.join()
    logger.info("Watch mode stopped.")


def find_series_directories(dojo: str) -> List[Path]:
    """
    Find all series directories within a Dojo.

    Args:
        dojo: Dojo name (FM, MI, ML, MS, PI)

    Returns:
        List of series directory paths
    """
    dojo_path = BASE_PATH / dojo
    if not dojo_path.exists():
        logger.warning(f"Dojo directory not found: {dojo_path}")
        return []

    # Find all directories that contain chapter files
    series_dirs = []
    for item in dojo_path.iterdir():
        if item.is_dir():
            if list(item.glob("chapter*.md")) or list(item.glob("chapter*.html")):
                series_dirs.append(item)

    return sorted(series_dirs)


def main():
    """Main synchronization function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Bidirectional synchronization between Markdown and HTML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Auto-sync all files in a series (detects which is newer)
  python sync_md_html.py knowledge/en/ML/transformer-introduction/

  # Force sync MD→HTML for entire series
  python sync_md_html.py knowledge/en/ML/transformer-introduction/ --force-direction md2html

  # Force sync HTML→MD for a single file
  python sync_md_html.py knowledge/en/ML/transformer-introduction/chapter-1.html --force-direction html2md

  # Dry run to see what would be synced
  python sync_md_html.py knowledge/en/ML/transformer-introduction/ --dry-run

  # Watch mode for live development
  python sync_md_html.py knowledge/en/ML/transformer-introduction/ --watch

  # Sync entire Dojo
  python sync_md_html.py ML

  # Sync entire knowledge base
  python sync_md_html.py knowledge/en/
        '''
    )

    parser.add_argument(
        'path',
        type=str,
        nargs='?',
        default='knowledge/en/',
        help='File, directory, or Dojo to sync (default: entire knowledge base)'
    )

    parser.add_argument(
        '--force-direction',
        choices=['md2html', 'html2md'],
        help='Force sync direction (default: auto-detect based on modification time)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    parser.add_argument(
        '--watch',
        action='store_true',
        help='Watch for file changes and auto-sync (requires watchdog)'
    )

    args = parser.parse_args()

    logger.info("Starting Markdown-HTML bidirectional sync...")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN MODE - No files will be modified]")
        logger.info("=" * 60)

    path = Path(args.path)
    total_success = 0
    total_files = 0
    watch_paths = []

    # Determine what to sync
    if not path.exists():
        # Check if it's a Dojo name
        if args.path.upper() in DOJOS:
            series_dirs = find_series_directories(args.path.upper())
            for series_path in series_dirs:
                success, total = sync_directory(series_path, args.force_direction, args.dry_run)
                total_success += success
                total_files += total
                watch_paths.append(series_path)
        else:
            logger.error(f"Path does not exist: {path}")
            sys.exit(1)
    elif path.is_file():
        # Single file
        if path.suffix == '.md':
            pair = FilePair(path)
        elif path.suffix == '.html':
            pair = FilePair(path.with_suffix('.md'))
        else:
            logger.error(f"Invalid file type: {path} (must be .md or .html)")
            sys.exit(1)

        if sync_file_pair(pair, args.force_direction, args.dry_run):
            total_success = 1
        total_files = 1
        watch_paths.append(path.parent)
    elif path.is_dir():
        # Check if it's a series directory
        if list(path.glob("chapter*.md")) or list(path.glob("chapter*.html")):
            success, total = sync_directory(path, args.force_direction, args.dry_run)
            total_success += success
            total_files += total
            watch_paths.append(path)
        else:
            # Might be a Dojo or knowledge base root
            for dojo in DOJOS:
                dojo_path = path / dojo
                if dojo_path.exists():
                    series_dirs = find_series_directories(dojo)
                    for series_path in series_dirs:
                        success, total = sync_directory(series_path, args.force_direction, args.dry_run)
                        total_success += success
                        total_files += total
                        watch_paths.append(series_path)
    else:
        logger.error(f"Invalid path: {path}")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Synchronization complete!")
    logger.info(f"Successfully synced: {total_success}/{total_files} files")

    # Start watch mode if requested
    if args.watch:
        if watch_paths:
            watch_mode(watch_paths, args.force_direction)
        else:
            logger.warning("No directories to watch")


if __name__ == "__main__":
    main()
