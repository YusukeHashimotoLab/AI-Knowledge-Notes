#!/usr/bin/env python3
"""
Add bidirectional locale switcher links between English and Japanese articles.

English version: Add link to Japanese version
Japanese version: Add locale switcher with link to English version
"""

import re
from pathlib import Path
from typing import Tuple, Optional

BASE_DIR = Path(__file__).parent.parent
EN_DIR = BASE_DIR / "knowledge" / "en"
JP_DIR = BASE_DIR / "knowledge" / "jp"


def get_corresponding_path(file_path: Path, from_lang: str) -> Optional[Path]:
    """Get the corresponding file path in the other language."""
    if from_lang == "en":
        # Convert en/... -> jp/...
        relative = file_path.relative_to(EN_DIR)
        jp_path = JP_DIR / relative
        return jp_path if jp_path.exists() else None
    else:
        # Convert jp/... -> en/...
        relative = file_path.relative_to(JP_DIR)
        en_path = EN_DIR / relative
        return en_path if en_path.exists() else None


def add_jp_link_to_en_file(file_path: Path) -> Tuple[bool, str]:
    """Add Japanese link to English locale switcher."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if JP link already exists
        if 'href="../../../jp/' in content or '🇯🇵 JP' in content:
            return False, "JP link already exists"

        # Find the locale-switcher div
        pattern = r'(<div class="locale-switcher">\s*<span class="current-locale">🌐 EN</span>\s*<span class="locale-separator">\|</span>)'

        if not re.search(pattern, content):
            return False, "Locale switcher not found"

        # Get relative path to JP version
        relative_path = file_path.relative_to(EN_DIR)
        jp_relative = f"../../../jp/{relative_path}"

        # Insert JP link after the separator
        replacement = r'\1\n<a href="' + jp_relative + r'" class="locale-link">🇯🇵 JP</a>\n<span class="locale-separator">|</span>'

        new_content = re.sub(pattern, replacement, content)

        if new_content == content:
            return False, "No changes made"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True, "JP link added successfully"

    except Exception as e:
        return False, f"Error: {e}"


def add_locale_switcher_to_jp_file(file_path: Path) -> Tuple[bool, str]:
    """Add complete locale switcher with EN link to Japanese file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if locale switcher already exists
        if 'locale-switcher' in content or '🌐 JP' in content:
            return False, "Locale switcher already exists"

        # Find <body> tag
        body_match = re.search(r'<body>\s*', content)
        if not body_match:
            return False, "<body> tag not found"

        # Get relative path to EN version
        relative_path = file_path.relative_to(JP_DIR)
        en_relative = f"../../../en/{relative_path}"

        # Create locale switcher HTML
        locale_switcher = f'''<div class="locale-switcher">
<span class="current-locale">🌐 JP</span>
<span class="locale-separator">|</span>
<a href="{en_relative}" class="locale-link">🇬🇧 EN</a>
<span class="locale-separator">|</span>
<span class="locale-meta">Last sync: 2025-11-16</span>
</div>
'''

        # Insert after <body>
        insert_pos = body_match.end()
        new_content = content[:insert_pos] + locale_switcher + content[insert_pos:]

        # Check if CSS styles exist
        if '.locale-switcher' not in content:
            # Add CSS styles before </head>
            css_styles = '''
    <style>
        /* Locale Switcher Styles */
        .locale-switcher {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 6px;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .current-locale {
            font-weight: 600;
            color: #7b2cbf;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        .locale-separator {
            color: #adb5bd;
            font-weight: 300;
        }

        .locale-link {
            color: #f093fb;
            text-decoration: none;
            font-weight: 500;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }

        .locale-link:hover {
            background: rgba(240, 147, 251, 0.1);
            color: #d07be8;
            transform: translateY(-1px);
        }

        .locale-meta {
            color: #868e96;
            font-size: 0.85rem;
            font-style: italic;
            margin-left: auto;
        }

        @media (max-width: 768px) {
            .locale-switcher {
                font-size: 0.85rem;
                padding: 0.4rem 0.8rem;
            }
            .locale-meta {
                display: none;
            }
        }
    </style>
'''
            new_content = new_content.replace('</head>', css_styles + '</head>')

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True, "Locale switcher added successfully"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Main execution."""
    print("=" * 70)
    print("Adding Bidirectional Locale Links")
    print("=" * 70)

    # Statistics
    en_files_processed = 0
    en_files_updated = 0
    jp_files_processed = 0
    jp_files_updated = 0

    # Process English files
    print("\n[1/2] Processing English files...")
    en_html_files = list(EN_DIR.rglob("*.html"))

    for en_file in en_html_files:
        # Check if corresponding JP file exists
        jp_file = get_corresponding_path(en_file, "en")
        if not jp_file:
            continue

        en_files_processed += 1
        success, message = add_jp_link_to_en_file(en_file)

        if success:
            en_files_updated += 1
            if en_files_updated <= 5:  # Show first 5
                print(f"  ✅ {en_file.relative_to(BASE_DIR)}")

    print(f"\nEnglish files: {en_files_updated}/{en_files_processed} updated")

    # Process Japanese files
    print("\n[2/2] Processing Japanese files...")
    jp_html_files = list(JP_DIR.rglob("*.html"))

    for jp_file in jp_html_files:
        # Check if corresponding EN file exists
        en_file = get_corresponding_path(jp_file, "jp")
        if not en_file:
            continue

        jp_files_processed += 1
        success, message = add_locale_switcher_to_jp_file(jp_file)

        if success:
            jp_files_updated += 1
            if jp_files_updated <= 5:  # Show first 5
                print(f"  ✅ {jp_file.relative_to(BASE_DIR)}")

    print(f"\nJapanese files: {jp_files_updated}/{jp_files_processed} updated")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"English files updated:  {en_files_updated}/{en_files_processed}")
    print(f"Japanese files updated: {jp_files_updated}/{jp_files_processed}")
    print(f"Total changes:          {en_files_updated + jp_files_updated}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
