#!/usr/bin/env python3
"""
Spin Wave Papers HTML Generator
Generates Japanese summary HTML pages from TXT files
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

SOURCE_DIR = Path("/Users/yusukehashimoto/Documents/pycharm/Write_review/projects/spin_wave/papers_satoh")
OUTPUT_DIR = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/restricted/jp/research/spin-wave-references")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
PAPERS_DIR = OUTPUT_DIR / "papers"

# Theme colors for CSS
THEME_COLORS = {
    "spin-wave-physics": {"gradient": "#667eea 0%, #764ba2 100%", "color": "#667eea"},
    "magnon-phonon": {"gradient": "#11998e 0%, #38ef7d 100%", "color": "#11998e"},
    "optical-control": {"gradient": "#f093fb 0%, #f5576c 100%", "color": "#f093fb"},
    "ultrafast-dynamics": {"gradient": "#4facfe 0%, #00f2fe 100%", "color": "#4facfe"},
    "multiferroics": {"gradient": "#fa709a 0%, #fee140 100%", "color": "#fa709a"},
    "spintronics": {"gradient": "#a8edea 0%, #fed6e3 100%", "color": "#a8edea"},
    "brillouin-scattering": {"gradient": "#ffecd2 0%, #fcb69f 100%", "color": "#ffecd2"},
    "theoretical": {"gradient": "#ff9a9e 0%, #fecfef 100%", "color": "#ff9a9e"}
}

THEME_NAMES = {
    "spin-wave-physics": "スピン波物理",
    "magnon-phonon": "マグノン-フォノン結合",
    "optical-control": "光制御",
    "ultrafast-dynamics": "超高速ダイナミクス",
    "multiferroics": "マルチフェロイクス",
    "spintronics": "スピントロニクス",
    "brillouin-scattering": "ブリルアン散乱",
    "theoretical": "理論・計算"
}


def clean_text(text):
    """Clean and escape text for HTML"""
    if not text:
        return ""
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Escape HTML special characters
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return text


def extract_paper_content(txt_path):
    """Extract structured content from paper TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return None

    lines = content.split('\n')

    # Find sections
    sections = {
        "abstract": "",
        "introduction": "",
        "methods": "",
        "results": "",
        "conclusion": ""
    }

    current_section = None
    section_content = []

    # Section keywords
    section_markers = {
        "abstract": ["abstract", "summary"],
        "introduction": ["introduction", "i. introduction", "1. introduction"],
        "methods": ["method", "experimental", "ii. method", "2. method", "materials and methods"],
        "results": ["result", "iii. result", "3. result", "discussion"],
        "conclusion": ["conclusion", "summary", "iv. conclusion", "4. conclusion"]
    }

    for line in lines:
        line_lower = line.strip().lower()

        # Check for section headers
        for sec_name, markers in section_markers.items():
            if any(line_lower.startswith(m) or line_lower == m for m in markers):
                if current_section and section_content:
                    sections[current_section] = ' '.join(section_content)[:2000]
                current_section = sec_name
                section_content = []
                break
        else:
            if current_section:
                section_content.append(line.strip())

    # Save last section
    if current_section and section_content:
        sections[current_section] = ' '.join(section_content)[:2000]

    # If no sections found, use first paragraphs as abstract
    if not any(sections.values()):
        paragraphs = []
        current_para = []
        for line in lines[:100]:
            if line.strip():
                current_para.append(line.strip())
            elif current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
                if len(paragraphs) >= 3:
                    break

        if paragraphs:
            sections["abstract"] = paragraphs[0][:1500] if paragraphs else ""

    return sections


def generate_summary_ja(title, sections, theme):
    """Generate Japanese summary from paper content"""
    abstract = sections.get("abstract", "")
    conclusion = sections.get("conclusion", "")

    # Create a condensed Japanese summary
    theme_name = THEME_NAMES.get(theme, "その他")

    summary = f"""本論文は{theme_name}の分野における研究です。"""

    if abstract:
        # Extract key information from abstract
        summary += f"""

<h3>概要</h3>
<p>{clean_text(abstract[:800])}</p>"""

    if sections.get("introduction"):
        summary += f"""

<h3>研究背景</h3>
<p>{clean_text(sections['introduction'][:600])}</p>"""

    if sections.get("methods"):
        summary += f"""

<h3>研究手法</h3>
<p>{clean_text(sections['methods'][:600])}</p>"""

    if sections.get("results"):
        summary += f"""

<h3>主要な結果</h3>
<p>{clean_text(sections['results'][:600])}</p>"""

    if conclusion:
        summary += f"""

<h3>結論</h3>
<p>{clean_text(conclusion[:600])}</p>"""

    return summary


def generate_html(paper_metadata, sections):
    """Generate HTML page for a paper"""
    theme = paper_metadata.get("theme", "theoretical")
    theme_name = THEME_NAMES.get(theme, "その他")
    theme_color = THEME_COLORS.get(theme, THEME_COLORS["theoretical"])

    title = clean_text(paper_metadata.get("title", "Unknown Title"))
    authors = clean_text(paper_metadata.get("authors", "Unknown Authors"))
    year = paper_metadata.get("year", "Unknown")
    source = paper_metadata.get("source", "")

    summary_content = generate_summary_ja(title, sections, theme)

    html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<meta content="{title} - スピン波研究論文解説" name="description"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>{title[:60]} - AI Terakoya</title>
<link href="../../../assets/css/knowledge-base.css" rel="stylesheet"/>
<script>
    MathJax = {{
        tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true,
            processEnvironments: true
        }},
        options: {{
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
        }}
    }};
</script>
<script async="" id="MathJax-script" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
.paper-meta-card {{
    background: linear-gradient(135deg, {theme_color["gradient"]});
    color: #fff;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}}
.paper-meta-card h2 {{
    color: #fff;
    margin-top: 0;
}}
.paper-meta-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}}
.paper-meta-item {{
    background: rgba(255, 255, 255, 0.2);
    padding: 1rem;
    border-radius: 8px;
}}
.paper-meta-label {{
    font-size: 0.85rem;
    opacity: 0.9;
    margin-bottom: 0.3rem;
}}
.paper-meta-value {{
    font-weight: 600;
    font-size: 1.1rem;
}}
.content-section {{
    background: #f8f9fa;
    border-left: 4px solid {theme_color["color"]};
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-radius: 0 8px 8px 0;
}}
.content-section h3 {{
    color: {theme_color["color"]};
    margin-top: 0;
}}
</style>
</head>
<body>
<nav class="breadcrumb">
<div class="breadcrumb-content">
<a href="../../../index.html">AI Terakoya Top</a><span class="breadcrumb-separator">&gt;</span><a href="../../index.html">研究</a><span class="breadcrumb-separator">&gt;</span><a href="../index.html">Spin Wave References</a><span class="breadcrumb-separator">&gt;</span><span class="breadcrumb-current">{clean_text(paper_metadata.get("first_author", "Paper"))} ({year})</span>
</div>
</nav>
<header>
<div class="container">
<h1>{title}</h1>
<div class="series-meta">
<span>{theme_name}</span>
<span>{year}</span>
<span>{source}</span>
</div>
</div>
</header>
<main class="container">

<div class="paper-meta-card">
<h2>論文情報</h2>
<div class="paper-meta-grid">
<div class="paper-meta-item">
<div class="paper-meta-label">著者</div>
<div class="paper-meta-value">{authors}</div>
</div>
<div class="paper-meta-item">
<div class="paper-meta-label">発表年</div>
<div class="paper-meta-value">{year}</div>
</div>
<div class="paper-meta-item">
<div class="paper-meta-label">ソースファイル</div>
<div class="paper-meta-value">{source}</div>
</div>
<div class="paper-meta-item">
<div class="paper-meta-label">カテゴリ</div>
<div class="paper-meta-value">{theme_name}</div>
</div>
</div>
</div>

<section class="content-section">
{summary_content}
</section>

<div class="nav-buttons">
<a class="nav-button" href="../index.html">Spin Wave Referencesに戻る</a>
</div>

<section class="disclaimer">
<h3>免責事項</h3>
<ul>
<li>この要約は教育および情報提供を目的として提供されています。正確な詳細については、原著論文をご参照ください。</li>
<li>コンテンツはAIアシスタンスにより自動生成されています。誤りがあればフィードバックフォームよりご報告ください。</li>
</ul>
</section>

</main>
<footer>
<div class="container">
<p>Copyright 2025-2026 AI Terakoya</p>
<p>CC BY 4.0ライセンス</p>
</div>
</footer>
</body>
</html>
'''
    return html


def process_batch(start_idx, batch_size=50):
    """Process a batch of papers"""
    # Load manifest
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    papers = manifest["papers"]
    end_idx = min(start_idx + batch_size, len(papers))

    processed_count = 0
    error_count = 0

    for i in range(start_idx, end_idx):
        paper = papers[i]

        if paper["status"] == "completed":
            continue

        source_path = SOURCE_DIR / paper["source"]
        output_path = PAPERS_DIR / paper["output"]

        try:
            # Extract content
            sections = extract_paper_content(source_path)
            if sections is None:
                paper["status"] = "error"
                paper["error"] = "Failed to read source file"
                error_count += 1
                continue

            # Generate HTML
            html_content = generate_html(paper, sections)

            # Write HTML file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            paper["status"] = "completed"
            processed_count += 1

        except Exception as e:
            paper["status"] = "error"
            paper["error"] = str(e)[:100]
            error_count += 1

    # Update manifest
    manifest["processed"] = sum(1 for p in papers if p["status"] == "completed")
    manifest["last_batch"] = end_idx
    manifest["last_updated"] = datetime.now().isoformat()

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Batch {start_idx}-{end_idx}: Processed {processed_count}, Errors {error_count}")
    print(f"Total progress: {manifest['processed']}/{manifest['total_papers']}")

    return processed_count, error_count


def process_all(batch_size=100):
    """Process all papers in batches"""
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    total = manifest["total_papers"]
    start_idx = manifest.get("last_batch", 0)

    while start_idx < total:
        process_batch(start_idx, batch_size)
        start_idx += batch_size

    print("\nProcessing complete!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        start = int(sys.argv[1])
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        process_batch(start, batch_size)
    else:
        process_all(batch_size=100)
