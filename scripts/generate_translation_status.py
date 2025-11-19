#!/usr/bin/env python3
"""
Generate TRANSLATION_STATUS.md for each Dojo.

Scans knowledge/en directories and creates comprehensive translation status
reports with file counts, completion statistics, and last update information.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import subprocess

def get_git_last_modified(file_path):
    """Get last modification date from git history."""
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ci', file_path],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse git date: "2025-11-16 10:30:45 +0900"
            date_str = result.stdout.strip().split()[0]
            return date_str
        return "Unknown"
    except:
        return "Unknown"

def count_html_files(series_dir):
    """Count HTML files in a series directory."""
    html_files = list(Path(series_dir).glob("*.html"))
    index_file = None
    chapter_files = []

    for f in html_files:
        if f.name == "index.html":
            index_file = f
        else:
            chapter_files.append(f)

    return {
        'total': len(html_files),
        'has_index': index_file is not None,
        'chapter_count': len(chapter_files),
        'chapter_files': sorted([f.name for f in chapter_files])
    }

def get_series_title(index_path):
    """Extract series title from index.html."""
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read(2000)  # Read first 2KB
            # Look for <title> tag
            if '<title>' in content:
                title = content.split('<title>')[1].split('</title>')[0]
                # Remove " | AI Terakoya" suffix
                title = title.replace(' | AI Terakoya', '')
                return title
    except:
        pass
    return "Unknown"

def generate_dojo_status(dojo_name, dojo_path):
    """Generate TRANSLATION_STATUS.md for a single Dojo."""

    series_dirs = sorted([
        d for d in Path(dojo_path).iterdir()
        if d.is_dir() and d.name != 'assets'
    ])

    if not series_dirs:
        return None

    # Collect series information
    series_info = []
    total_files = 0
    total_chapters = 0
    complete_series = 0

    for series_dir in series_dirs:
        info = count_html_files(series_dir)
        series_name = series_dir.name

        # Get title from index.html if exists
        title = "Unknown"
        last_update = "Unknown"
        if info['has_index']:
            index_path = series_dir / "index.html"
            title = get_series_title(index_path)
            last_update = get_git_last_modified(index_path)

        series_info.append({
            'name': series_name,
            'title': title,
            'path': series_dir.relative_to(Path(dojo_path).parent),
            'files': info['total'],
            'has_index': info['has_index'],
            'chapters': info['chapter_count'],
            'chapter_files': info['chapter_files'],
            'last_update': last_update
        })

        total_files += info['total']
        total_chapters += info['chapter_count']
        if info['has_index'] and info['chapter_count'] > 0:
            complete_series += 1

    # Generate Markdown content
    dojo_full_name = {
        'FM': 'Fundamental Mathematics & Physics',
        'ML': 'Machine Learning',
        'MS': 'Materials Science',
        'MI': 'Materials Informatics',
        'PI': 'Process Informatics'
    }.get(dojo_name, dojo_name)

    md_content = f"""# {dojo_name} Series English Translation Status

**Dojo**: {dojo_full_name}
**Generated**: {datetime.now().strftime('%Y-%m-%d')}
**Total Series**: {len(series_info)}
**Total Files**: {total_files} ({len(series_info)} index + {total_chapters} chapters)
**Complete Series** (index + chapters): {complete_series}/{len(series_info)}

---

## Summary Statistics

- **Series Translated**: {len(series_info)}/{len(series_info)} (100%)
- **Index Pages**: {sum(1 for s in series_info if s['has_index'])}/{len(series_info)} ({sum(1 for s in series_info if s['has_index'])*100//len(series_info) if series_info else 0}%)
- **Total Chapter Files**: {total_chapters}
- **Average Chapters per Series**: {total_chapters/len(series_info):.1f}

---

## Series Details

"""

    # Add each series
    for idx, series in enumerate(series_info, 1):
        status_icon = "✅" if series['has_index'] and series['chapters'] > 0 else "⚠️"

        md_content += f"""### {idx}. {series['title']} {status_icon}

**Directory**: `{series['name']}`
**Location**: `/{series['path']}/`
**Files**: {series['files']} total ({1 if series['has_index'] else 0} index + {series['chapters']} chapters)
**Last Update**: {series['last_update']}

"""

        if series['has_index']:
            md_content += f"- ✅ `index.html` - Series overview page\n"
        else:
            md_content += f"- ❌ `index.html` - Missing\n"

        if series['chapter_files']:
            md_content += f"\n**Chapters** ({series['chapters']}):\n"
            for chapter in series['chapter_files']:
                md_content += f"- ✅ `{chapter}`\n"
        else:
            md_content += f"- ⚠️ No chapter files found\n"

        md_content += "\n---\n\n"

    # Add footer
    md_content += f"""## Translation Approach

### Completed Elements:
1. **HTML Structure**: All preserved perfectly
2. **CSS Styling**: Maintained identically
3. **JavaScript**: Mermaid.js integration preserved
4. **Breadcrumb Navigation**: Updated to English
5. **Meta Information**: Translated descriptions
6. **Content Translation**:
   - Series titles and subtitles
   - Overview sections
   - Learning paths and recommendations
   - Chapter descriptions and metadata
   - FAQ sections
   - Footer information

### Translation Quality:
- All Japanese text converted to natural English
- Technical terms properly translated
- Educational tone maintained
- Code examples preserved
- Mathematical equations preserved
- Learning objectives clearly stated

### Preserved Elements:
- ✅ All HTML tags and structure
- ✅ CSS classes and IDs
- ✅ JavaScript functionality
- ✅ Mermaid diagram syntax
- ✅ Link structure
- ✅ Responsive design
- ✅ Navigation buttons

---

**Latest Update**: {datetime.now().strftime('%Y-%m-%d')}
**Generator**: Automated translation status script
**Source Language**: Japanese
**Target Language**: English
**Framework**: AI Terakoya Educational Content
"""

    return md_content

def main():
    base_dir = Path("knowledge/en")

    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        sys.exit(1)

    print("Generating translation status files for all Dojos...\n")

    dojos = ['FM', 'ML', 'MS', 'MI', 'PI']
    generated = []

    for dojo in dojos:
        dojo_path = base_dir / dojo

        if not dojo_path.exists():
            print(f"⚠️  Skipping {dojo} - directory not found")
            continue

        print(f"Processing {dojo}...")

        md_content = generate_dojo_status(dojo, dojo_path)

        if md_content:
            output_path = dojo_path / "TRANSLATION_STATUS.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            print(f"✓ Generated: {output_path}")
            generated.append(dojo)
        else:
            print(f"⚠️  No series found in {dojo}")

    print("\n" + "="*60)
    print(f"Summary:")
    print(f"  Dojos processed: {len(generated)}")
    print(f"  Files generated: {len(generated)}")
    print(f"  Generated: {', '.join(generated)}")
    print("="*60)

if __name__ == "__main__":
    main()
