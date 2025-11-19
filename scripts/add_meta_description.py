#!/usr/bin/env python3
"""
Add meta description tags to HTML files missing them.
Generates descriptions from page title and learning objectives.
"""

import re
from pathlib import Path
from typing import Tuple, Optional
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent.parent


def extract_title(soup: BeautifulSoup) -> Optional[str]:
    """Extract title from HTML."""
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text(strip=True)

    h1 = soup.find('h1')
    if h1:
        return h1.get_text(strip=True)

    return None


def extract_learning_objectives(soup: BeautifulSoup) -> list:
    """Extract learning objectives from HTML."""
    objectives = []

    # Find learning objectives section
    learning_section = soup.find('div', class_='learning-objectives')
    if not learning_section:
        learning_section = soup.find('section', class_='learning-objectives')

    if learning_section:
        # Extract list items
        items = learning_section.find_all('li')
        for item in items[:3]:  # Take first 3
            text = item.get_text(strip=True)
            # Remove emoji/checkmarks
            text = re.sub(r'[✅✓☑]', '', text).strip()
            objectives.append(text)

    return objectives


def generate_description(title: str, objectives: list, max_length: int = 155) -> str:
    """Generate meta description from title and objectives."""
    if not objectives:
        # No objectives, use title only
        desc = title
    else:
        # Combine title and first objective
        desc = f"{title}. {objectives[0]}"

    # Truncate to max length
    if len(desc) > max_length:
        desc = desc[:max_length - 3] + "..."

    return desc


def add_meta_description(file_path: Path) -> Tuple[bool, str]:
    """Add meta description to HTML file if missing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if meta description already exists
        if re.search(r'<meta\s+name=["\']description["\']', content, re.IGNORECASE):
            return False, "Meta description already exists"

        soup = BeautifulSoup(content, 'html.parser')

        # Extract information
        title = extract_title(soup)
        if not title:
            return False, "No title found"

        objectives = extract_learning_objectives(soup)
        description = generate_description(title, objectives)

        # Create meta tag
        meta_tag = f'<meta content="{description}" name="description"/>'

        # Find insertion point (after charset or viewport meta)
        # Look for <meta charset> or <meta viewport>
        meta_pattern = r'(<meta\s+[^>]*(?:charset|viewport)[^>]*>)'
        match = re.search(meta_pattern, content, re.IGNORECASE)

        if match:
            # Insert after existing meta tag
            insert_pos = match.end()
            new_content = content[:insert_pos] + '\n' + meta_tag + content[insert_pos:]
        else:
            # Insert after <head>
            head_match = re.search(r'<head[^>]*>', content, re.IGNORECASE)
            if head_match:
                insert_pos = head_match.end()
                new_content = content[:insert_pos] + '\n' + meta_tag + content[insert_pos:]
            else:
                return False, "No suitable insertion point found"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True, "Meta description added"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Main execution."""
    print("=" * 70)
    print("Adding Meta Description Tags")
    print("=" * 70)

    # Find all HTML files
    html_files = list(BASE_DIR.glob("knowledge/**/*.html"))

    files_processed = 0
    files_updated = 0
    errors = []

    for html_file in html_files:
        files_processed += 1
        success, message = add_meta_description(html_file)

        if success:
            files_updated += 1
            if files_updated <= 10:  # Show first 10
                print(f"  ✅ {html_file.relative_to(BASE_DIR)}")
        elif "Error:" in message:
            errors.append((html_file.name, message))

    if files_updated > 10:
        print(f"\n  ... and {files_updated - 10} more files")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Files processed: {files_processed}")
    print(f"Files updated:   {files_updated}")
    if errors:
        print(f"Errors:          {len(errors)}")
        print("\nFirst 5 errors:")
        for fname, msg in errors[:5]:
            print(f"  - {fname}: {msg}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
