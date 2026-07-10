#!/usr/bin/env python3
"""
Batch Translation Script for MI Series

Translates all chapter files in a series directory.

Usage:
    python scripts/translate_series.py knowledge/en/MI/materials-applications-introduction
"""

import sys
import argparse
from pathlib import Path
from translate_html import HTMLTranslator


def translate_series(series_dir, api_key=None):
    """Translate all chapter files in a series directory."""
    series_path = Path(series_dir)

    if not series_path.exists():
        print(f"❌ Error: Directory not found: {series_path}")
        sys.exit(1)

    # Find all chapter HTML files
    chapter_files = sorted(series_path.glob("chapter-*.html"))

    if not chapter_files:
        print(f"❌ Error: No chapter-*.html files found in {series_path}")
        sys.exit(1)

    print(f"📚 Found {len(chapter_files)} chapter files to translate")
    print(f"📂 Series: {series_path.name}\n")

    # Initialize translator
    translator = HTMLTranslator(api_key=api_key)

    # Translate each file
    for i, chapter_file in enumerate(chapter_files, 1):
        print(f"\n{'='*60}")
        print(f"Chapter {i}/{len(chapter_files)}: {chapter_file.name}")
        print(f"{'='*60}\n")

        try:
            translator.translate_file(chapter_file)
        except Exception as e:
            print(f"❌ Error translating {chapter_file.name}: {e}")
            print("Continuing with next file...\n")
            continue

    print(f"\n\n{'='*60}")
    print(f"🎉 Series translation complete!")
    print(f"{'='*60}\n")

    # Final verification
    print("📊 Final verification:")
    for chapter_file in chapter_files:
        with open(chapter_file, 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        japanese_count = len(re.findall(r'[一-龯ぁ-んァ-ヶー]', content))

        status = "✅" if japanese_count == 0 else f"⚠️  ({japanese_count} Japanese chars)"
        print(f"  {chapter_file.name}: {status}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Translate all chapter files in an MI series directory"
    )
    parser.add_argument(
        'series_dir',
        help="Path to series directory (e.g., knowledge/en/MI/materials-applications-introduction)"
    )
    parser.add_argument(
        '--api-key',
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)"
    )

    args = parser.parse_args()

    try:
        translate_series(args.series_dir, api_key=args.api_key)
    except KeyboardInterrupt:
        print("\n\n⚠️  Translation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
