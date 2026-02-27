#!/usr/bin/env python3
"""
Spin Wave Papers Processor
Extracts metadata from TXT files and generates manifest.json
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

SOURCE_DIR = Path("/Users/yusukehashimoto/Documents/pycharm/Write_review/projects/spin_wave/papers_satoh")
OUTPUT_DIR = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/restricted/jp/research/spin-wave-references")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

# Theme classification keywords (title weight: 3x, body weight: 1x)
THEMES = {
    "spin-wave-physics": {
        "name_ja": "スピン波物理",
        "keywords": ["spin wave", "spin-wave", "magnon", "FMR", "ferromagnetic resonance",
                     "YIG", "yttrium iron garnet", "magnon-magnon", "dipole-exchange"]
    },
    "magnon-phonon": {
        "name_ja": "マグノン-フォノン結合",
        "keywords": ["phonon", "magnetoelastic", "acoustic", "magneto-acoustic",
                     "magnon-phonon", "lattice", "elastic"]
    },
    "optical-control": {
        "name_ja": "光制御",
        "keywords": ["optical", "laser", "photon", "Faraday", "Kerr", "photoinduced",
                     "light-induced", "optical control", "femtosecond"]
    },
    "ultrafast-dynamics": {
        "name_ja": "超高速ダイナミクス",
        "keywords": ["ultrafast", "THz", "terahertz", "pump-probe", "femtosecond",
                     "picosecond", "demagnetization", "all-optical"]
    },
    "multiferroics": {
        "name_ja": "マルチフェロイクス",
        "keywords": ["multiferroic", "BiFeO", "BFO", "magnetoelectric", "ferroelectric",
                     "antiferromagnetic", "multiferroics"]
    },
    "spintronics": {
        "name_ja": "スピントロニクス",
        "keywords": ["spin current", "spin Hall", "STT", "spin transfer", "spin pumping",
                     "spin Seebeck", "spin orbit", "spin-orbit"]
    },
    "brillouin-scattering": {
        "name_ja": "ブリルアン散乱",
        "keywords": ["Brillouin", "BLS", "inelastic", "light scattering",
                     "magneto-optical", "spectroscopy"]
    },
    "theoretical": {
        "name_ja": "理論・計算",
        "keywords": ["theory", "theoretical", "DFT", "simulation", "first-principles",
                     "ab initio", "calculation", "numerical", "model"]
    }
}


def extract_year_from_filename(filename):
    """Extract year from filename patterns like PRB86_134403(2012).txt"""
    # Pattern: (YYYY)
    match = re.search(r'\((\d{4})\)', filename)
    if match:
        return match.group(1)

    # Pattern: arXiv like 2107.07265 -> 2021
    match = re.match(r'^(\d{2})(\d{2})\.\d+', filename)
    if match:
        year = int(match.group(1))
        if year >= 90:
            return f"19{year}"
        else:
            return f"20{match.group(1)}"

    return None


def extract_metadata(txt_path):
    """Extract author, title, year, and journal from TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')[:50]  # First 50 lines for metadata
    except Exception as e:
        return None

    metadata = {
        "source": txt_path.name,
        "title": None,
        "title_line": None,
        "first_author": None,
        "authors": None,
        "year": None,
        "journal": None,
        "theme": None,
        "output": None,
        "status": "pending",
        "error": None
    }

    # Extract title (usually first non-empty line or line with journal prefix)
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        if line and len(line) > 20 and not line.startswith('arXiv'):
            # Skip journal headers like "PHYSICAL REVIEW B"
            if re.match(r'^(PHYSICAL REVIEW|NATURE|SCIENCE|JOURNAL|Phys\. Rev\.)', line, re.I):
                continue
            # Skip DOI/PACS lines
            if 'DOI:' in line or 'PACS' in line:
                continue
            metadata["title"] = line
            metadata["title_line"] = i
            break

    # Extract authors (usually after title, contains * or initials pattern)
    for i, line in enumerate(lines[1:20], 1):
        line = line.strip()
        if not line:
            continue
        # Author patterns: "A. B. Name*" or "Name, A. B."
        if re.search(r'[A-Z]\.\s*[A-Z]?\.*\s*[A-Z][a-z]+', line) or '*' in line:
            # Skip affiliation lines
            if any(x in line.lower() for x in ['university', 'institute', 'department', 'laboratory']):
                continue
            metadata["authors"] = line
            # Extract first author (before first comma or 'and')
            author_match = re.match(r'^([A-Z]\.\s*[A-Z]?\.*\s*)?([A-Z][a-z]+)', line)
            if author_match:
                metadata["first_author"] = author_match.group(2).lower()
            break

    # Extract year
    year = extract_year_from_filename(txt_path.name)
    if not year:
        # Try to find in content
        for line in lines[:30]:
            match = re.search(r'\b(19[89]\d|20[0-2]\d)\b', line)
            if match:
                year = match.group(1)
                break
    metadata["year"] = year or "unknown"

    # Classify theme
    title_lower = (metadata["title"] or "").lower()
    body_lower = content[:5000].lower()

    best_theme = "theoretical"  # Default
    best_score = 0

    for theme_id, theme_data in THEMES.items():
        score = 0
        for kw in theme_data["keywords"]:
            kw_lower = kw.lower()
            # Title match (3x weight)
            if kw_lower in title_lower:
                score += 3
            # Body match (1x weight)
            score += body_lower.count(kw_lower)

        if score > best_score:
            best_score = score
            best_theme = theme_id

    metadata["theme"] = best_theme

    # Generate output filename
    author = metadata["first_author"] or txt_path.stem[:10].lower().replace("_", "-")
    year = metadata["year"]

    # Generate keyword from theme and title
    keyword = best_theme.split("-")[0][:10]
    if metadata["title"]:
        # Extract a distinctive word from title
        title_words = re.findall(r'\b[A-Za-z]{4,}\b', metadata["title"])
        for word in title_words[:5]:
            if word.lower() not in ['with', 'from', 'this', 'that', 'have', 'been']:
                keyword = word.lower()[:15]
                break

    output_name = f"{author}-{year}-{keyword}.html"
    # Clean filename
    output_name = re.sub(r'[^\w\-.]', '', output_name)
    metadata["output"] = output_name

    return metadata


def scan_all_papers():
    """Scan all TXT files and generate manifest"""
    txt_files = sorted(SOURCE_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} TXT files")

    papers = []
    seen_outputs = set()

    for i, txt_path in enumerate(txt_files):
        if i % 500 == 0:
            print(f"Processing {i}/{len(txt_files)}...")

        metadata = extract_metadata(txt_path)
        if metadata:
            # Handle duplicate output names
            base_output = metadata["output"]
            counter = 1
            while metadata["output"] in seen_outputs:
                name_parts = base_output.rsplit('.', 1)
                metadata["output"] = f"{name_parts[0]}-{counter}.html"
                counter += 1

            seen_outputs.add(metadata["output"])
            papers.append(metadata)

    manifest = {
        "project": "spin-wave-references",
        "total_papers": len(papers),
        "processed": 0,
        "last_batch": 0,
        "last_updated": datetime.now().isoformat(),
        "themes": {k: {"name_ja": v["name_ja"], "count": 0} for k, v in THEMES.items()},
        "papers": papers
    }

    # Count themes
    for paper in papers:
        if paper["theme"] in manifest["themes"]:
            manifest["themes"][paper["theme"]]["count"] += 1

    # Save manifest
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nManifest saved to {MANIFEST_PATH}")
    print(f"Total papers: {len(papers)}")
    print("\nTheme distribution:")
    for theme_id, theme_data in manifest["themes"].items():
        print(f"  {theme_data['name_ja']}: {theme_data['count']}")

    return manifest


if __name__ == "__main__":
    manifest = scan_all_papers()
