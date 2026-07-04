#!/usr/bin/env python3
"""
Chapter Description Insertion Script

This script scans HTML chapter files in knowledge/en/ directory and automatically
adds <p class="chapter-description"> elements for chapters that are missing them.

Features:
- Identifies files missing chapter-description
- Extracts chapter title and learning objectives
- Generates appropriate description text
- Inserts description after <h1> and before first <h2> or <hr/>
- Supports dry-run mode for safe previewing
- Provides comprehensive progress reporting

Usage:
    python scripts/add_chapter_description.py --dry-run  # Preview changes
    python scripts/add_chapter_description.py            # Apply changes
    python scripts/add_chapter_description.py --directory knowledge/en/ML  # Specific dir

Requirements:
    - Python 3.9+
    - beautifulsoup4>=4.12.0
    - lxml>=5.0.0

Author: AI Terakoya Content Team
Version: 1.0
Created: 2025-11-17
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from bs4 import BeautifulSoup, NavigableString
except ImportError:
    print("Error: BeautifulSoup4 is required. Install with: pip install beautifulsoup4 lxml")
    sys.exit(1)


class ChapterDescriptionGenerator:
    """Generates contextually appropriate chapter descriptions."""

    DESCRIPTION_TEMPLATES = {
        'fundamentals': (
            "This chapter covers the fundamentals of {topic}, which {purpose}. "
            "You will learn {concepts}."
        ),
        'basics': (
            "This chapter introduces the basics of {topic}. "
            "You will learn {concepts}."
        ),
        'introduction': (
            "This chapter provides an introduction to {topic}. "
            "You will explore {concepts}."
        ),
        'advanced': (
            "This chapter covers advanced topics in {topic}. "
            "You will master {concepts}."
        ),
        'practical': (
            "This chapter focuses on practical applications of {topic}. "
            "You will learn {concepts}."
        ),
        'default': (
            "This chapter covers {topic}. "
            "You will learn {concepts}."
        )
    }

    @staticmethod
    def extract_key_topics(title: str, objectives: List[str], sections: List[str]) -> Dict[str, str]:
        """Extract key topics from title, objectives, and section headings."""
        # Extract main topic from title
        topic_match = re.search(r'Chapter \d+:\s*(.+?)(?:\s*-|$)', title, re.IGNORECASE)
        main_topic = topic_match.group(1).strip() if topic_match else title

        # Extract subtitle if present (text after dash)
        subtitle_match = re.search(r'Chapter \d+:\s*.+?\s*-\s*(.+)', title, re.IGNORECASE)
        subtitle = subtitle_match.group(1).strip() if subtitle_match else ""

        # Build a natural description from objectives and sections
        if objectives:
            # Use first objective as primary concept
            first_obj = objectives[0]
            first_obj = re.sub(r'^[✅✓]\s*', '', first_obj)
            first_obj = re.sub(r'^(?:understand|learn|master|explain)\s+', '', first_obj, flags=re.IGNORECASE)
            first_obj = re.sub(r'^(?:the|how to)\s+', '', first_obj, flags=re.IGNORECASE)
            # Keep full first objective with some cleanup
            first_obj = first_obj.strip().rstrip('.')

            # Try to extract 2-3 key topics from all objectives
            key_terms = []
            for obj in objectives[:5]:
                # Extract noun phrases (simplified approach)
                obj = re.sub(r'^[✅✓]\s*', '', obj)
                obj = re.sub(r'^(?:understand|learn|master|explain|implement|apply|grasp|select)\s+', '', obj, flags=re.IGNORECASE)
                obj = re.sub(r'^(?:the|how to|appropriate)\s+', '', obj, flags=re.IGNORECASE)

                # Find key phrases (before "and", "in", "for", etc.)
                phrases = re.split(r',\s*|\s+and\s+', obj)
                for phrase in phrases[:1]:  # Take first phrase
                    phrase = phrase.strip().rstrip('.')
                    if 15 <= len(phrase) <= 50 and phrase not in key_terms:
                        key_terms.append(phrase)
                        if len(key_terms) >= 3:
                            break
                if len(key_terms) >= 3:
                    break

            # Create natural concept description
            if len(key_terms) >= 3:
                concepts = f"{key_terms[0]}, {key_terms[1]}, and {key_terms[2]}"
            elif len(key_terms) == 2:
                concepts = f"{key_terms[0]} and {key_terms[1]}"
            elif len(key_terms) == 1:
                concepts = key_terms[0]
            elif first_obj:
                # Fallback to cleaned first objective
                concepts = first_obj if len(first_obj) < 120 else first_obj[:117] + "..."
            else:
                concepts = "essential concepts and techniques"
        else:
            concepts = "essential concepts and techniques"

        # Determine purpose from subtitle or first section
        purpose = subtitle if subtitle else ""
        if not purpose and sections:
            first_section = re.sub(r'^\d+\.\d+\s*', '', sections[0])
            if len(first_section) < 50:
                purpose = first_section

        return {
            'topic': main_topic,
            'concepts': concepts,
            'purpose': purpose.lower() if purpose else "forms the foundation of this area",
            'domain': ChapterDescriptionGenerator._extract_domain(main_topic)
        }

    @staticmethod
    def _extract_domain(topic: str) -> str:
        """Extract domain from topic."""
        topic_lower = topic.lower()
        if any(word in topic_lower for word in ['machine learning', 'deep learning', 'neural']):
            return 'machine learning'
        elif any(word in topic_lower for word in ['data', 'feature', 'preprocessing']):
            return 'data science'
        elif any(word in topic_lower for word in ['material', 'ceramic', 'metal', 'composite']):
            return 'materials science'
        elif any(word in topic_lower for word in ['process', 'manufacturing', 'chemical']):
            return 'process informatics'
        elif any(word in topic_lower for word in ['physics', 'quantum', 'statistical']):
            return 'physics'
        else:
            return 'this field'

    @staticmethod
    def select_template(title: str) -> str:
        """Select appropriate template based on chapter title."""
        title_lower = title.lower()

        if 'fundamental' in title_lower:
            return ChapterDescriptionGenerator.DESCRIPTION_TEMPLATES['fundamentals']
        elif 'basic' in title_lower or 'introduction' in title_lower:
            return ChapterDescriptionGenerator.DESCRIPTION_TEMPLATES['basics']
        elif 'advanced' in title_lower:
            return ChapterDescriptionGenerator.DESCRIPTION_TEMPLATES['advanced']
        elif 'practical' in title_lower or 'application' in title_lower:
            return ChapterDescriptionGenerator.DESCRIPTION_TEMPLATES['practical']
        else:
            return ChapterDescriptionGenerator.DESCRIPTION_TEMPLATES['default']

    @classmethod
    def generate(cls, title: str, objectives: List[str], sections: List[str]) -> str:
        """Generate chapter description based on content analysis."""
        topics = cls.extract_key_topics(title, objectives, sections)
        template = cls.select_template(title)

        try:
            description = template.format(**topics)
            return description
        except KeyError as e:
            # Fallback to default template
            default = cls.DESCRIPTION_TEMPLATES['default']
            try:
                return default.format(**topics)
            except KeyError:
                # Last resort fallback
                return f"This chapter covers {topics['topic']}. You will learn {topics['concepts']}."


class ChapterAnalyzer:
    """Analyzes chapter HTML files and extracts relevant content."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.soup = None
        self.encoding = 'utf-8'

    def load(self) -> bool:
        """Load and parse HTML file."""
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                self.soup = BeautifulSoup(f.read(), 'lxml')
            return True
        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            return False

    def has_description(self) -> bool:
        """Check if chapter description already exists."""
        if not self.soup:
            return False
        desc = self.soup.find('p', class_='chapter-description')
        return desc is not None

    def extract_title(self) -> Optional[str]:
        """Extract chapter title from <h1> tag."""
        if not self.soup:
            return None

        # Try main content h1 first
        main = self.soup.find('main')
        if main:
            h1 = main.find('h1')
            if h1:
                return h1.get_text(strip=True)

        # Fallback to header h1
        header = self.soup.find('header')
        if header:
            h1 = header.find('h1')
            if h1:
                return h1.get_text(strip=True)

        # Last resort: any h1
        h1 = self.soup.find('h1')
        return h1.get_text(strip=True) if h1 else None

    def extract_objectives(self) -> List[str]:
        """Extract learning objectives from the chapter."""
        if not self.soup:
            return []

        objectives = []
        main = self.soup.find('main')
        if not main:
            return objectives

        # Find "Learning Objectives" section
        obj_heading = main.find('h2', string=re.compile(r'Learning Objectives?', re.IGNORECASE))
        if obj_heading:
            # Get next ul element
            ul = obj_heading.find_next_sibling('ul')
            if ul:
                for li in ul.find_all('li', recursive=False):
                    objectives.append(li.get_text(strip=True))

        return objectives

    def extract_section_headings(self) -> List[str]:
        """Extract first-level section headings (h2 tags)."""
        if not self.soup:
            return []

        sections = []
        main = self.soup.find('main')
        if not main:
            return sections

        for h2 in main.find_all('h2'):
            text = h2.get_text(strip=True)
            # Skip meta sections
            if not any(skip in text.lower() for skip in [
                'learning objective', 'chapter summary', 'practice problem',
                'reference', 'next chapter', 'what we learned'
            ]):
                sections.append(text)

        return sections[:5]  # Limit to first 5 sections

    def find_insertion_point(self) -> Optional[Tuple[any, str]]:
        """Find the appropriate location to insert chapter description.

        Returns:
            Tuple of (element, position) where position is 'after_h1' or 'main_start'
        """
        if not self.soup:
            return None

        main = self.soup.find('main')
        if not main:
            return None

        # Check if h1 exists in main
        h1 = main.find('h1')
        if h1:
            return (h1, 'after_h1')

        # If no h1 in main, insert at the beginning of main (after opening tag)
        # This handles cases where h1 is in header, not main
        return (main, 'main_start')

    def insert_description(self, description_text: str) -> bool:
        """Insert chapter description into the HTML."""
        result = self.find_insertion_point()
        if not result:
            return False

        element, position = result

        # Create description element
        desc_p = self.soup.new_tag('p', attrs={'class': 'chapter-description'})
        desc_p.string = description_text

        if position == 'after_h1':
            # Insert after h1
            element.insert_after('\n')
            element.insert_after(desc_p)
            desc_p.insert_after('\n')
        elif position == 'main_start':
            # Insert at beginning of main, before first child
            if element.contents:
                # Insert before first child
                first_child = element.contents[0]
                if isinstance(first_child, NavigableString) and first_child.strip() == '':
                    # If first child is just whitespace, insert after it
                    if len(element.contents) > 1:
                        first_child.insert_after(desc_p)
                        first_child.insert_after('\n')
                        desc_p.insert_after('\n')
                    else:
                        element.insert(0, '\n')
                        element.insert(0, desc_p)
                        element.insert(0, '\n')
                else:
                    first_child.insert_before('\n')
                    first_child.insert_before(desc_p)
                    first_child.insert_before('\n')
            else:
                # Empty main tag
                element.insert(0, '\n')
                element.insert(0, desc_p)
                element.insert(0, '\n')

        return True

    def save(self) -> bool:
        """Save modified HTML back to file."""
        if not self.soup:
            return False

        try:
            # Format HTML nicely
            html_str = str(self.soup)

            with open(self.file_path, 'w', encoding=self.encoding) as f:
                f.write(html_str)

            return True
        except Exception as e:
            print(f"Error saving {self.file_path}: {e}")
            return False


class ChapterDescriptionAdder:
    """Main class for adding chapter descriptions to HTML files."""

    def __init__(self, root_dir: Path, dry_run: bool = False, verbose: bool = True):
        self.root_dir = root_dir
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = {
            'total_files': 0,
            'with_description': 0,
            'without_description': 0,
            'processed': 0,
            'errors': 0
        }

    def find_chapter_files(self) -> List[Path]:
        """Find all chapter HTML files in the directory."""
        pattern = 'chapter*.html'
        files = list(self.root_dir.rglob(pattern))

        # Filter out backup and temporary files
        files = [f for f in files if not any(
            suffix in f.name for suffix in ['.backup', '.bak', '.tmp', '.temp', '_temp']
        )]

        return sorted(files)

    def process_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Process a single chapter file.

        Returns:
            Tuple of (success, description_text)
        """
        analyzer = ChapterAnalyzer(file_path)

        if not analyzer.load():
            return False, None

        # Check if already has description
        if analyzer.has_description():
            self.stats['with_description'] += 1
            if self.verbose:
                print(f"  ✓ Already has description: {file_path.relative_to(self.root_dir)}")
            return True, None

        self.stats['without_description'] += 1

        # Extract content
        title = analyzer.extract_title()
        objectives = analyzer.extract_objectives()
        sections = analyzer.extract_section_headings()

        if not title:
            if self.verbose:
                print(f"  ✗ No title found: {file_path.relative_to(self.root_dir)}")
            self.stats['errors'] += 1
            return False, None

        # Generate description
        generator = ChapterDescriptionGenerator()
        description = generator.generate(title, objectives, sections)

        if self.dry_run:
            if self.verbose:
                print(f"\n  📄 {file_path.relative_to(self.root_dir)}")
                print(f"     Title: {title}")
                print(f"     Generated: {description}")
            return True, description

        # Insert description
        if not analyzer.insert_description(description):
            if self.verbose:
                print(f"  ✗ Failed to insert: {file_path.relative_to(self.root_dir)}")
            self.stats['errors'] += 1
            return False, None

        # Save file
        if not analyzer.save():
            if self.verbose:
                print(f"  ✗ Failed to save: {file_path.relative_to(self.root_dir)}")
            self.stats['errors'] += 1
            return False, None

        self.stats['processed'] += 1
        if self.verbose:
            print(f"  ✅ Added description: {file_path.relative_to(self.root_dir)}")
            print(f"     {description}")

        return True, description

    def run(self) -> Dict:
        """Run the chapter description addition process."""
        print(f"\n{'='*80}")
        print(f"Chapter Description Addition Tool")
        print(f"{'='*80}")
        print(f"Root directory: {self.root_dir}")
        print(f"Mode: {'DRY RUN (no changes will be made)' if self.dry_run else 'LIVE (files will be modified)'}")
        print(f"{'='*80}\n")

        # Find all chapter files
        chapter_files = self.find_chapter_files()
        self.stats['total_files'] = len(chapter_files)

        if not chapter_files:
            print("No chapter files found!")
            return self.stats

        print(f"Found {len(chapter_files)} chapter files\n")

        # Process each file
        for file_path in chapter_files:
            self.process_file(file_path)

        # Print summary
        self._print_summary()

        return self.stats

    def _print_summary(self):
        """Print processing summary."""
        print(f"\n{'='*80}")
        print(f"Summary")
        print(f"{'='*80}")
        print(f"Total files scanned:          {self.stats['total_files']}")
        print(f"Already with description:     {self.stats['with_description']}")
        print(f"Missing description:          {self.stats['without_description']}")

        if self.dry_run:
            print(f"Would be processed:           {self.stats['without_description'] - self.stats['errors']}")
        else:
            print(f"Successfully processed:       {self.stats['processed']}")

        print(f"Errors:                       {self.stats['errors']}")
        print(f"{'='*80}\n")

        if self.dry_run and self.stats['without_description'] > 0:
            print("Run without --dry-run to apply changes.\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Add chapter descriptions to HTML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes (dry run)
  python scripts/add_chapter_description.py --dry-run

  # Apply changes to all files
  python scripts/add_chapter_description.py

  # Process specific directory
  python scripts/add_chapter_description.py --directory knowledge/en/ML

  # Quiet mode (minimal output)
  python scripts/add_chapter_description.py --quiet
        """
    )

    parser.add_argument(
        '--directory',
        type=str,
        default='knowledge/en',
        help='Root directory to scan (default: knowledge/en)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (only show summary)'
    )

    args = parser.parse_args()

    # Resolve directory path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    root_dir = project_root / args.directory

    if not root_dir.exists():
        print(f"Error: Directory does not exist: {root_dir}")
        sys.exit(1)

    if not root_dir.is_dir():
        print(f"Error: Not a directory: {root_dir}")
        sys.exit(1)

    # Run the tool
    adder = ChapterDescriptionAdder(
        root_dir=root_dir,
        dry_run=args.dry_run,
        verbose=not args.quiet
    )

    stats = adder.run()

    # Exit code based on errors
    sys.exit(1 if stats['errors'] > 0 else 0)


if __name__ == '__main__':
    main()
