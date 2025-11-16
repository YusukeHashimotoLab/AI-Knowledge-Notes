#!/usr/bin/env python3
"""
Analyze broken links from linkcheck_en_local.txt and categorize them
for decision making on how to fix.
"""

import re
from pathlib import Path
from collections import defaultdict

def parse_link_report(report_path):
    """Parse linkcheck report and categorize broken links."""

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract broken link entries
    broken_links = []
    current_file = None

    for line in content.split('\n'):
        if line.startswith('File: '):
            current_file = line.replace('File: ', '').strip()
        elif line.startswith('Line ') and current_file:
            # Extract link target
            parts = line.split(': ', 1)
            if len(parts) == 2:
                line_num = parts[0].replace('Line ', '').strip()
                link = parts[1].strip()
                broken_links.append({
                    'file': current_file,
                    'line': line_num,
                    'link': link
                })

    return broken_links

def categorize_links(broken_links):
    """Categorize broken links by type."""

    categories = {
        'missing_chapters': [],
        'missing_index': [],
        'non_existent_series': [],
        'asset_paths': [],
        'navigation_loops': [],
        'other': []
    }

    for link in broken_links:
        link_text = link['link']

        # Missing chapter files
        if re.match(r'^chapter-?\d+\.html$', link_text):
            categories['missing_chapters'].append(link)

        # Missing index.html
        elif 'index.html' in link_text:
            if '../' in link_text or '../../' in link_text:
                categories['missing_index'].append(link)
            else:
                categories['missing_index'].append(link)

        # Non-existent series directories
        elif re.search(r'\.\./([\w-]+)/index\.html', link_text):
            match = re.search(r'\.\./([\w-]+)/index\.html', link_text)
            series_name = match.group(1)
            # Check if it's a known missing series
            if any(x in series_name for x in ['robotic-lab', 'gnn-features', 'materials-screening',
                                                'llm-basics', 'machine-learning-basics']):
                categories['non_existent_series'].append(link)
            else:
                categories['missing_index'].append(link)

        # Asset paths (CSS, images, etc.)
        elif any(ext in link_text for ext in ['.css', '.js', '.png', '.jpg', '.svg']):
            categories['asset_paths'].append(link)

        # Navigation loops (../../FM/index.html type)
        elif re.match(r'\.\./.+/index\.html', link_text) and 'knowledge' in link_text:
            categories['navigation_loops'].append(link)

        # Everything else
        else:
            categories['other'].append(link)

    return categories

def analyze_missing_chapters(missing_chapters):
    """Analyze missing chapter files by series."""

    series_missing = defaultdict(list)

    for link in missing_chapters:
        # Extract series from file path
        path_parts = Path(link['file']).parts
        if len(path_parts) >= 2:
            dojo = path_parts[0]
            series = path_parts[1]
            series_key = f"{dojo}/{series}"
            series_missing[series_key].append(link['link'])

    return series_missing

def generate_report(categories, series_missing):
    """Generate analysis report."""

    report = """# Broken Links Analysis Report

Generated: 2025-11-16

## Summary by Category

"""

    total = sum(len(v) for v in categories.values())

    report += f"**Total Broken Links**: {total}\n\n"

    for category, links in categories.items():
        if links:
            count = len(links)
            pct = (count / total * 100) if total > 0 else 0
            report += f"- **{category.replace('_', ' ').title()}**: {count} ({pct:.1f}%)\n"

    report += "\n---\n\n"

    # Missing Chapters Detail
    report += f"""## 1. Missing Chapters ({len(categories['missing_chapters'])} links)

**Analysis**: These are chapter files referenced in navigation but not yet created.

### Breakdown by Series:

"""

    for series, chapters in sorted(series_missing.items()):
        unique_chapters = sorted(set(chapters))
        report += f"**{series}**: {len(unique_chapters)} missing\n"
        for ch in unique_chapters:
            report += f"  - {ch}\n"
        report += "\n"

    report += f"""**Recommendation**:
- **Option A**: Create template files for all missing chapters (2-3 days)
- **Option B**: Remove chapter links from navigation (4-6 hours)
- **Option C**: Create chapters only for high-priority series (1-2 days)

---

"""

    # Missing Index Detail
    report += f"""## 2. Missing Index Files ({len(categories['missing_index'])} links)

**Analysis**: Series index.html files that are referenced but don't exist.

### Examples:

"""

    missing_index_series = set()
    for link in categories['missing_index'][:20]:  # Show first 20
        report += f"- File: `{link['file']}`\n"
        report += f"  Link: `{link['link']}`\n\n"

        # Extract series name
        match = re.search(r'/([\w-]+)/index\.html', link['link'])
        if match:
            missing_index_series.add(match.group(1))

    report += f"\n**Missing Index Series** ({len(missing_index_series)}):\n"
    for series in sorted(missing_index_series):
        report += f"- {series}\n"

    report += f"""
**Recommendation**: Create index.html for these series or remove references.

---

"""

    # Non-Existent Series
    report += f"""## 3. Non-Existent Series ({len(categories['non_existent_series'])} links)

**Analysis**: Links to series directories that don't exist at all.

### Series Not Yet Created:

"""

    non_existent = set()
    for link in categories['non_existent_series']:
        match = re.search(r'\.\./?([\w-]+)/', link['link'])
        if match:
            non_existent.add(match.group(1))

    for series in sorted(non_existent):
        count = sum(1 for l in categories['non_existent_series'] if series in l['link'])
        report += f"- **{series}**: {count} references\n"

    report += f"""
**Recommendation**: Remove all references to non-existent series.

---

"""

    # Asset Paths
    if categories['asset_paths']:
        report += f"""## 4. Asset Path Issues ({len(categories['asset_paths'])} links)

**Analysis**: Broken paths to CSS, JS, images, or other assets.

### Examples:

"""
        for link in categories['asset_paths'][:10]:
            report += f"- `{link['link']}` in `{link['file']}`\n"

        report += f"""
**Recommendation**: Fix asset paths or copy missing assets.

---

"""

    # Navigation Loops
    if categories['navigation_loops']:
        report += f"""## 5. Navigation Loop Issues ({len(categories['navigation_loops'])} links)

**Analysis**: Incorrect relative paths causing navigation loops.

### Examples:

"""
        for link in categories['navigation_loops'][:10]:
            report += f"- `{link['link']}` in `{link['file']}`\n"

        report += f"""
**Recommendation**: Fix relative path logic in templates.

---

"""

    # Other
    if categories['other']:
        report += f"""## 6. Other Issues ({len(categories['other'])} links)

### Examples:

"""
        for link in categories['other'][:10]:
            report += f"- `{link['link']}` in `{link['file']}`\n"

    report += "\n---\n\n"

    # Action Plan
    report += """## Recommended Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. Remove references to non-existent series (10 links)
2. Fix navigation loop issues
3. Verify and fix asset paths

### Phase 2: Index Files (2-4 hours)
1. Create missing index.html for existing series
2. Update navigation to remove broken index references

### Phase 3: Missing Chapters (Decision Required)

**Option A - Full Creation** (2-3 days):
- Create all missing chapter files using templates
- Pros: Complete navigation, professional appearance
- Cons: Time-consuming, may create placeholder content

**Option B - Navigation Update** (4-6 hours):
- Remove chapter links from navigation until ready
- Pros: Fast, no placeholder content
- Cons: Users can't see full course structure

**Option C - Hybrid Approach** (1-2 days):
- Create chapters for top 10 most-accessed series
- Remove navigation for low-priority series
- Pros: Balanced effort/value
- Cons: Requires usage analytics

### Recommended: Option C (Hybrid)
1. Identify top 10 series (by page views or strategic importance)
2. Create missing chapters for those 10 series
3. Update navigation to hide unfinished chapters in other series
4. Document which series are "complete" vs "in progress"

"""

    return report

def main():
    report_path = Path("linkcheck_en_local.txt")

    if not report_path.exists():
        print(f"Error: {report_path} not found")
        return

    print("Analyzing broken links...")

    broken_links = parse_link_report(report_path)
    print(f"Found {len(broken_links)} broken link entries")

    categories = categorize_links(broken_links)
    series_missing = analyze_missing_chapters(categories['missing_chapters'])

    report = generate_report(categories, series_missing)

    output_path = Path("BROKEN_LINKS_ANALYSIS.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✓ Analysis complete: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    total = sum(len(v) for v in categories.values())
    for category, links in categories.items():
        if links:
            print(f"  {category.replace('_', ' ').title()}: {len(links)}")
    print(f"  Total: {total}")
    print("="*60)

if __name__ == "__main__":
    main()
