#!/usr/bin/env python3
"""
Spin Wave Papers Batch Processor
Generates high-quality Japanese explanation HTML pages from TXT files.
Uses Claude API for deep analysis when available, falls back to template-based generation.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import subprocess

# Configuration
SOURCE_DIR = Path("/Users/yusukehashimoto/Documents/pycharm/Write_review/projects/spin_wave/papers_satoh")
OUTPUT_DIR = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/restricted/jp/research/spin-wave-references/papers")
MANIFEST_PATH = OUTPUT_DIR.parent / "manifest.json"

# HTML Template
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<meta content="{meta_description}" name="description"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>{title_short} - AI Terakoya</title>
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
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
.concept-box {{
    background: #f8f9fa;
    border-left: 4px solid #667eea;
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-radius: 0 8px 8px 0;
}}
.concept-box h4 {{
    color: #667eea;
    margin-top: 0;
}}
.result-highlight {{
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
}}
.result-highlight h4 {{
    color: #2e7d32;
    margin-top: 0;
}}
.equation-box {{
    background: #fff3e0;
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    overflow-x: auto;
}}
.equation-box h4 {{
    color: #e65100;
    margin-top: 0;
}}
.method-box {{
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
}}
.method-box h4 {{
    color: #1565c0;
    margin-top: 0;
}}
.impact-box {{
    background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    border: 2px solid #ffc107;
}}
.impact-box h4 {{
    color: #ff8f00;
    margin-top: 0;
}}
.keyword-box {{
    background: #f3e5f5;
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
}}
.keyword-box h4 {{
    color: #7b1fa2;
    margin-top: 0;
}}
.keyword-tag {{
    display: inline-block;
    margin: 0.3rem 0.5rem 0.3rem 0;
    padding: 0.4rem 0.8rem;
    background: white;
    border-radius: 16px;
    font-size: 0.9rem;
    color: #7b1fa2;
    border: 1px solid #ce93d8;
}}
.related-papers {{
    background: #e3f2fd;
    padding: 1.5rem;
    border-radius: 8px;
    margin-top: 2rem;
}}
.related-papers h4 {{
    color: #1565c0;
    margin-top: 0;
}}
.related-link {{
    display: inline-block;
    margin: 0.3rem 0.5rem 0.3rem 0;
    padding: 0.4rem 0.8rem;
    background: white;
    border-radius: 4px;
    text-decoration: none;
    color: #1565c0;
    font-size: 0.9rem;
    transition: all 0.3s ease;
}}
.related-link:hover {{
    background: #1565c0;
    color: white;
}}
.abstract-box {{
    background: #fafafa;
    border: 1px solid #e0e0e0;
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    font-style: italic;
}}
</style>
</head>
<body>
<nav class="breadcrumb">
<div class="breadcrumb-content">
<a href="../../../index.html">AI Terakoya Top</a><span class="breadcrumb-separator">></span><a href="../../index.html">研究</a><span class="breadcrumb-separator">></span><a href="../index.html">Spin Wave References</a><span class="breadcrumb-separator">></span><span class="breadcrumb-current">{author_short} ({year})</span>
</div>
</nav>
<header>
<div class="container">
<h1>{title}</h1>
<p style="font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.95;">{subtitle}</p>
<div class="series-meta">
<span>{category}</span>
<span>{year}</span>
<span>{journal}</span>
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
<div class="paper-meta-label">掲載誌/arXiv</div>
<div class="paper-meta-value">{journal}</div>
</div>
<div class="paper-meta-item">
<div class="paper-meta-label">カテゴリ</div>
<div class="paper-meta-value">{category}</div>
</div>
</div>
</div>

<section>
<h2>概要（Abstract）</h2>
<div class="abstract-box">
<p>{abstract}</p>
</div>
</section>

<section>
<h2>研究の背景と意義</h2>
{background_section}
</section>

<section>
<h2>主要な研究内容</h2>
{main_content_section}
</section>

<section>
<h2>研究手法</h2>
{methodology_section}
</section>

<section>
<h2>主要な結果</h2>
{results_section}
</section>

<section>
<h2>科学的インパクト</h2>
<div class="impact-box">
<h4>スピン波物理学への貢献</h4>
{impact_content}
</div>
</section>

<section>
<h2>キーワード</h2>
<div class="keyword-box">
{keywords_html}
</div>
</section>

<section class="related-papers">
<h4>関連論文</h4>
<p>スピン波物理学の関連研究：</p>
<a class="related-link" href="../index.html">Spin Wave References一覧へ</a>
</section>

<div class="nav-buttons">
<a class="nav-button" href="../index.html">Spin Wave Referencesに戻る</a>
</div>

<section class="disclaimer">
<h3>免責事項</h3>
<ul>
<li>この要約は教育および情報提供を目的として提供されています。正確な詳細については、原著論文をご参照ください。</li>
<li>コンテンツはAIアシスタンスにより作成されています。誤りがあればフィードバックフォームよりご報告ください。</li>
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

# Keyword translations for common spin wave physics terms
KEYWORD_TRANSLATIONS = {
    'spin wave': 'スピン波（Spin Wave）',
    'magnon': 'マグノン（Magnon）',
    'ferromagnet': '強磁性体（Ferromagnet）',
    'antiferromagnet': '反強磁性体（Antiferromagnet）',
    'ferrimagnet': 'フェリ磁性体（Ferrimagnet）',
    'magneto-optical': '磁気光学（Magneto-optical）',
    'faraday': 'ファラデー効果（Faraday Effect）',
    'kerr': 'カー効果（Kerr Effect）',
    'brillouin': 'ブリルアン散乱（Brillouin Scattering）',
    'raman': 'ラマン散乱（Raman Scattering）',
    'phonon': 'フォノン（Phonon）',
    'magnon-phonon': 'マグノン-フォノン結合（Magnon-Phonon Coupling）',
    'ultrafast': '超高速（Ultrafast）',
    'femtosecond': 'フェムト秒（Femtosecond）',
    'terahertz': 'テラヘルツ（Terahertz）',
    'skyrmion': 'スカーミオン（Skyrmion）',
    'multiferroic': 'マルチフェロイック（Multiferroic）',
    'magnetization': '磁化（Magnetization）',
    'demagnetization': '消磁（Demagnetization）',
    'precession': '歳差運動（Precession）',
    'damping': '減衰（Damping）',
    'gilbert': 'ギルバート減衰（Gilbert Damping）',
    'landau-lifshitz': 'ランダウ-リフシッツ方程式（Landau-Lifshitz Equation）',
    'dispersion': '分散関係（Dispersion Relation）',
    'anisotropy': '異方性（Anisotropy）',
    'exchange': '交換相互作用（Exchange Interaction）',
    'dzyaloshinskii': 'ジャロシンスキー-守谷相互作用（DMI）',
    'spin-orbit': 'スピン軌道結合（Spin-Orbit Coupling）',
    'inverse faraday': '逆ファラデー効果（Inverse Faraday Effect）',
    'pump-probe': 'ポンプ-プローブ法（Pump-Probe）',
    'time-resolved': '時間分解（Time-Resolved）',
    'yig': 'イットリウム鉄ガーネット（YIG）',
    'garnet': 'ガーネット（Garnet）',
    'thin film': '薄膜（Thin Film）',
    'nanostructure': 'ナノ構造（Nanostructure）',
    'domain wall': '磁壁（Domain Wall）',
    'spin current': 'スピン流（Spin Current）',
    'spin pumping': 'スピンポンピング（Spin Pumping）',
    'spin hall': 'スピンホール効果（Spin Hall Effect）',
    'magnonics': 'マグノニクス（Magnonics）',
    'spintronics': 'スピントロニクス（Spintronics）',
}

# Category detection patterns
CATEGORY_PATTERNS = {
    'マグノニクス・スピントロニクス': ['magnonic', 'spintronic', 'spin current', 'spin transport', 'spin hall'],
    'マルチフェロイック': ['multiferroic', 'magnetoelectric', 'bifeo3', 'bfo'],
    '超高速スピンダイナミクス': ['ultrafast', 'femtosecond', 'pump-probe', 'time-resolved', 'demagnetization'],
    'テラヘルツ・マイクロ波': ['terahertz', 'thz', 'microwave', 'ghz'],
    'スカーミオン・トポロジカル': ['skyrmion', 'topological', 'chiral', 'helix'],
    '光スピン制御': ['optical', 'laser', 'photon', 'faraday', 'kerr', 'magneto-optical'],
    '磁気共鳴・散乱': ['resonance', 'fmr', 'brillouin', 'raman', 'neutron', 'x-ray'],
    'スピン波理論': ['theoretical', 'theory', 'calculation', 'simulation', 'first-principles', 'dft'],
    'マグノン-フォノン結合': ['phonon', 'lattice', 'acoust', 'magnon-phonon', 'magnetoelastic'],
    '反強磁性スピン波': ['antiferromagnet', 'afm', 'nio', 'mno', 'feo'],
}


def extract_year(text: str, filename: str) -> str:
    """Extract publication year from text or filename."""
    # Try to find year in text
    year_patterns = [
        r'\((\d{4})\)',  # (2020)
        r'(\d{4})\)',    # 2020)
        r'20[0-2]\d',    # 2000-2029
        r'19[89]\d',     # 1980-1999
    ]
    for pattern in year_patterns:
        match = re.search(pattern, text[:2000])
        if match:
            year = match.group(1) if '(' in pattern else match.group(0)
            if 1950 <= int(year) <= 2030:
                return year

    # Try filename
    match = re.search(r'(\d{4})', filename)
    if match:
        year = match.group(1)
        if 1950 <= int(year) <= 2030:
            return year

    # arXiv pattern
    match = re.search(r'(\d{2})(\d{2})\.\d+', filename)
    if match:
        year = '20' + match.group(1)
        if 2000 <= int(year) <= 2030:
            return year

    return "不明"


def extract_authors(text: str) -> str:
    """Extract author names from text."""
    lines = text.split('\n')[:30]
    authors = []

    for i, line in enumerate(lines):
        line = line.strip()
        # Skip empty lines and title
        if not line or i == 0:
            continue
        # Skip arXiv identifiers
        if 'arxiv' in line.lower():
            continue
        # Skip affiliations (usually contain university, institute, etc.)
        if any(word in line.lower() for word in ['university', 'institute', 'department', 'laboratory', 'cnrs', 'cea']):
            continue
        # Author line patterns
        if re.match(r'^[A-Z][a-z]+.*[A-Z]', line) and len(line) < 200:
            # Clean up the line
            author_line = re.sub(r'\d+[,\s]*', '', line)
            author_line = re.sub(r'[∗†‡§¶]', '', author_line)
            author_line = re.sub(r'\s+', ' ', author_line).strip()
            if author_line and not any(word in author_line.lower() for word in ['abstract', 'introduction', 'we ']):
                authors.append(author_line)
                if len(authors) >= 2:
                    break

    if authors:
        return ', '.join(authors[:2]) + (' et al.' if len(authors) > 2 else '')
    return "著者情報抽出中"


def extract_title(text: str, filename: str) -> str:
    """Extract paper title from text."""
    lines = text.split('\n')[:20]

    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Skip journal headers
        if any(word in line.upper() for word in ['PHYSICAL REVIEW', 'NATURE', 'SCIENCE', 'RAPID COMMUNICATIONS']):
            continue
        # Skip arXiv identifiers
        if 'arxiv' in line.lower():
            continue
        # Title is usually the first substantial line
        if len(line) > 20 and not line.startswith('('):
            # Clean up the title
            title = re.sub(r'\s+', ' ', line).strip()
            return title[:200]  # Limit length

    # Fallback to filename
    return filename.replace('.txt', '').replace('_', ' ')


def extract_abstract(text: str) -> str:
    """Extract abstract from text."""
    # Look for Abstract section
    abstract_match = re.search(r'(?:Abstract|ABSTRACT)[:\.]?\s*\n?(.*?)(?:\n\n|\nI\.\s|1\.\s*Introduction|PACS|Keywords)',
                               text, re.DOTALL | re.IGNORECASE)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        abstract = re.sub(r'\s+', ' ', abstract)
        return abstract[:1500]

    # Fallback: use first paragraph after title/authors
    lines = text.split('\n')
    in_abstract = False
    abstract_lines = []

    for i, line in enumerate(lines[3:30]):  # Skip first 3 lines (usually title/authors)
        line = line.strip()
        if not line:
            if abstract_lines:
                break
            continue
        # Skip headers and affiliations
        if any(word in line.lower() for word in ['university', 'institute', 'arxiv', 'doi:', 'pacs']):
            continue
        if len(line) > 50:  # Substantial text
            abstract_lines.append(line)
            if len(' '.join(abstract_lines)) > 500:
                break

    if abstract_lines:
        return ' '.join(abstract_lines)[:1500]

    return "本論文の詳細な要約については原著論文をご参照ください。"


def detect_category(text: str) -> str:
    """Detect paper category based on content."""
    text_lower = text.lower()

    category_scores = {}
    for category, keywords in CATEGORY_PATTERNS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            category_scores[category] = score

    if category_scores:
        return max(category_scores, key=category_scores.get)
    return "スピン波物理学"


def extract_keywords(text: str) -> List[str]:
    """Extract relevant keywords from text."""
    text_lower = text.lower()
    keywords = []

    for english, japanese in KEYWORD_TRANSLATIONS.items():
        if english in text_lower:
            keywords.append(japanese)
            if len(keywords) >= 10:
                break

    if not keywords:
        keywords = ['スピン波（Spin Wave）', 'マグノン（Magnon）']

    return keywords


def extract_journal(text: str, filename: str) -> str:
    """Extract journal information."""
    # Common journal patterns
    journal_patterns = [
        (r'Physical Review [A-Z]+ \d+', 'Physical Review'),
        (r'Phys\. Rev\. [A-Z]+ \d+', 'Physical Review'),
        (r'Nature [A-Za-z]* \d+', 'Nature'),
        (r'Science \d+', 'Science'),
        (r'Applied Physics Letters \d+', 'Applied Physics Letters'),
        (r'Journal of [A-Za-z ]+', 'Journal'),
        (r'PRL \d+', 'Physical Review Letters'),
        (r'PRB \d+', 'Physical Review B'),
    ]

    for pattern, name in journal_patterns:
        match = re.search(pattern, text[:1000], re.IGNORECASE)
        if match:
            return match.group(0)

    # arXiv
    arxiv_match = re.search(r'arXiv:(\d+\.\d+)', text[:1000], re.IGNORECASE)
    if arxiv_match:
        return f"arXiv:{arxiv_match.group(1)}"

    # From filename
    if 'PRB' in filename:
        return 'Physical Review B'
    if 'PRL' in filename:
        return 'Physical Review Letters'

    return "学術論文"


def generate_background_section(text: str, category: str) -> str:
    """Generate background section HTML."""
    # Extract some context from the introduction
    intro_match = re.search(r'(?:Introduction|INTRODUCTION|I\.\s*INTRODUCTION)(.*?)(?:II\.|2\.|Method|Experiment)',
                            text, re.DOTALL | re.IGNORECASE)

    background = ""
    if intro_match:
        intro_text = intro_match.group(1)[:1000]
        intro_text = re.sub(r'\[\d+[,\d]*\]', '', intro_text)  # Remove citations
        intro_text = re.sub(r'\s+', ' ', intro_text).strip()
        background = intro_text

    category_backgrounds = {
        'マグノニクス・スピントロニクス': 'スピン波（マグノン）を情報キャリアとして利用するマグノニクス分野の研究です。',
        'マルチフェロイック': '強誘電性と磁性が共存するマルチフェロイック材料に関する研究です。',
        '超高速スピンダイナミクス': 'フェムト秒〜ピコ秒スケールでの超高速磁化ダイナミクスに関する研究です。',
        'テラヘルツ・マイクロ波': 'テラヘルツ〜マイクロ波帯でのスピン波励起・制御に関する研究です。',
        'スカーミオン・トポロジカル': 'トポロジカルに保護されたスピン構造であるスカーミオンに関する研究です。',
        '光スピン制御': 'レーザー光を用いたスピン状態の制御・検出に関する研究です。',
        '磁気共鳴・散乱': '磁気共鳴や散乱実験による磁性研究です。',
        'スピン波理論': 'スピン波の理論的・計算科学的研究です。',
        'マグノン-フォノン結合': 'マグノンとフォノンの相互作用に関する研究です。',
        '反強磁性スピン波': '反強磁性体におけるスピン波励起に関する研究です。',
    }

    category_intro = category_backgrounds.get(category, 'スピン波物理学の基礎・応用研究です。')

    html = f'''<div class="concept-box">
<h4>研究分野: {category}</h4>
<p>{category_intro}</p>
</div>
'''

    if background:
        html += f'<p>{background[:800]}...</p>'

    return html


def generate_methodology_section(text: str) -> str:
    """Generate methodology section HTML."""
    # Look for method/experimental section
    method_keywords = ['experiment', 'method', 'sample', 'measurement', 'calculation', 'simulation']

    method_text = ""
    for keyword in method_keywords:
        match = re.search(rf'{keyword}[s]?[:\.]?\s*(.*?)(?:\n\n|III\.|3\.)', text, re.DOTALL | re.IGNORECASE)
        if match:
            method_text = match.group(1)[:600]
            break

    if method_text:
        method_text = re.sub(r'\[\d+[,\d]*\]', '', method_text)
        method_text = re.sub(r'\s+', ' ', method_text).strip()

    html = '''<div class="method-box">
<h4>研究手法</h4>
'''

    # Detect method types
    text_lower = text.lower()
    methods = []
    if any(w in text_lower for w in ['pump-probe', 'laser', 'optical']):
        methods.append('光学測定（Optical Measurement）')
    if any(w in text_lower for w in ['brillouin', 'bls']):
        methods.append('ブリルアン散乱分光（BLS）')
    if any(w in text_lower for w in ['raman']):
        methods.append('ラマン分光（Raman Spectroscopy）')
    if any(w in text_lower for w in ['neutron']):
        methods.append('中性子散乱（Neutron Scattering）')
    if any(w in text_lower for w in ['x-ray', 'xmcd', 'xfmr']):
        methods.append('X線分光（X-ray Spectroscopy）')
    if any(w in text_lower for w in ['fmr', 'ferromagnetic resonance']):
        methods.append('強磁性共鳴（FMR）')
    if any(w in text_lower for w in ['dft', 'first-principles', 'ab initio']):
        methods.append('第一原理計算（First-Principles Calculation）')
    if any(w in text_lower for w in ['micromagnetic', 'simulation']):
        methods.append('マイクロマグネティックシミュレーション')
    if any(w in text_lower for w in ['landau-lifshitz', 'llg']):
        methods.append('LLG方程式シミュレーション')

    if methods:
        html += '<ul>\n'
        for m in methods[:5]:
            html += f'<li>{m}</li>\n'
        html += '</ul>\n'

    html += '</div>\n'

    if method_text:
        html += f'<p>{method_text}...</p>'

    return html


def generate_results_section(text: str) -> str:
    """Generate results section HTML."""
    # Look for results/discussion
    results_match = re.search(r'(?:Results|Discussion|Conclusions?)[:\.]?\s*(.*?)(?:\n\n\n|References|Acknowledgment)',
                              text, re.DOTALL | re.IGNORECASE)

    results_text = ""
    if results_match:
        results_text = results_match.group(1)[:800]
        results_text = re.sub(r'\[\d+[,\d]*\]', '', results_text)
        results_text = re.sub(r'\s+', ' ', results_text).strip()

    html = '''<div class="result-highlight">
<h4>主要な発見</h4>
<p>本研究では、スピン波物理学に関する重要な知見が得られました。詳細な結果については原著論文をご参照ください。</p>
</div>
'''

    if results_text:
        html += f'<p>{results_text}...</p>'

    return html


def generate_main_content_section(text: str, title: str) -> str:
    """Generate main content section."""
    # Extract key sentences about the main research
    sentences = re.split(r'[.。]', text[:3000])
    key_sentences = []

    keywords = ['show', 'demonstrate', 'find', 'observe', 'reveal', 'report', 'study', 'investigate']
    for sent in sentences:
        if any(kw in sent.lower() for kw in keywords) and len(sent) > 50:
            clean_sent = re.sub(r'\[\d+[,\d]*\]', '', sent)
            clean_sent = re.sub(r'\s+', ' ', clean_sent).strip()
            if clean_sent:
                key_sentences.append(clean_sent)
            if len(key_sentences) >= 3:
                break

    html = '<div class="concept-box">\n'
    html += f'<h4>{title[:100]}</h4>\n'

    if key_sentences:
        html += '<ul>\n'
        for sent in key_sentences:
            html += f'<li>{sent[:200]}...</li>\n'
        html += '</ul>\n'
    else:
        html += '<p>本論文の主要な研究内容については原著論文をご参照ください。</p>\n'

    html += '</div>\n'
    return html


def generate_impact_content(category: str) -> str:
    """Generate impact content based on category."""
    impacts = {
        'マグノニクス・スピントロニクス': '本研究は、スピン波を用いた次世代情報処理技術の発展に貢献しています。低消費電力デバイスやマグノンベース論理回路への応用が期待されます。',
        'マルチフェロイック': '電場による磁性制御という新しいパラダイムを開拓し、次世代メモリやセンサ技術への応用が期待されます。',
        '超高速スピンダイナミクス': 'フェムト秒スケールでの磁化制御は、超高速磁気記録や光磁気デバイスの基盤技術となります。',
        'テラヘルツ・マイクロ波': 'テラヘルツ周波数帯でのスピン波制御は、高速通信や量子情報処理への応用が期待されます。',
        'スカーミオン・トポロジカル': 'トポロジカルに保護されたスピン構造は、安定で低消費電力の情報キャリアとして注目されています。',
        '光スピン制御': '全光学的なスピン制御は、超高速・低消費電力のスピントロニクスデバイス実現への鍵となります。',
        '磁気共鳴・散乱': '先端的な分光・散乱実験は、磁性材料の基礎物性理解に不可欠です。',
        'スピン波理論': '理論的予測は実験研究の指針となり、新材料・新現象の発見につながります。',
        'マグノン-フォノン結合': 'マグノンとフォノンの結合は、熱スピントロニクスやフォノニクスとの融合研究を推進します。',
        '反強磁性スピン波': '反強磁性体のテラヘルツダイナミクスは、超高速スピントロニクスの新しいフロンティアです。',
    }
    return impacts.get(category, '本研究は、スピン波物理学の基礎理解と応用技術の発展に貢献しています。')


def generate_keywords_html(keywords: List[str]) -> str:
    """Generate keywords HTML."""
    html = ''
    for kw in keywords:
        html += f'<span class="keyword-tag">{kw}</span>\n'
    return html


def generate_output_filename(txt_filename: str, authors: str, year: str, title: str) -> str:
    """Generate output HTML filename."""
    # Extract first author's last name
    author_match = re.match(r'([A-Za-z]+)', authors)
    first_author = author_match.group(1).lower() if author_match else 'unknown'

    # Extract key words from title
    title_words = re.findall(r'[a-z]+', title.lower())
    key_word = title_words[0] if title_words else 'paper'

    # Create filename
    base_name = f"{first_author}-{year}-{key_word}"

    # Ensure uniqueness with hash if needed
    hash_suffix = hashlib.md5(txt_filename.encode()).hexdigest()[:4]

    return f"{base_name}-{hash_suffix}.html"


def process_paper(txt_path: Path, output_dir: Path) -> Dict:
    """Process a single paper TXT file and generate HTML."""
    try:
        # Read TXT file
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        if len(text) < 100:
            return {'status': 'skipped', 'reason': 'Too short', 'file': txt_path.name}

        # Extract metadata
        filename = txt_path.name
        title = extract_title(text, filename)
        authors = extract_authors(text)
        year = extract_year(text, filename)
        abstract = extract_abstract(text)
        category = detect_category(text)
        keywords = extract_keywords(text)
        journal = extract_journal(text, filename)

        # Generate sections
        background_section = generate_background_section(text, category)
        methodology_section = generate_methodology_section(text)
        results_section = generate_results_section(text)
        main_content_section = generate_main_content_section(text, title)
        impact_content = generate_impact_content(category)
        keywords_html = generate_keywords_html(keywords)

        # Short versions for headers
        author_short = authors.split(',')[0].split()[0] if authors else 'Unknown'
        title_short = title[:60] + '...' if len(title) > 60 else title

        # Generate HTML
        html_content = HTML_TEMPLATE.format(
            meta_description=f"{authors} {year} - {title[:100]}",
            title_short=title_short,
            title=title,
            subtitle=category,
            category=category,
            year=year,
            journal=journal,
            authors=authors,
            author_short=author_short,
            abstract=abstract,
            background_section=background_section,
            main_content_section=main_content_section,
            methodology_section=methodology_section,
            results_section=results_section,
            impact_content=impact_content,
            keywords_html=keywords_html,
        )

        # Generate output filename
        output_filename = generate_output_filename(filename, authors, year, title)
        output_path = output_dir / output_filename

        # Write HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return {
            'status': 'success',
            'input': txt_path.name,
            'output': output_filename,
            'title': title[:100],
            'authors': authors,
            'year': year,
            'category': category,
        }

    except Exception as e:
        return {
            'status': 'error',
            'file': txt_path.name,
            'error': str(e)
        }


def batch_process(start_idx: int = 0, count: int = 500, verbose: bool = True) -> Dict:
    """Process a batch of papers."""
    # Get list of TXT files
    txt_files = sorted(SOURCE_DIR.glob('*.txt'))

    if start_idx >= len(txt_files):
        return {'error': f'Start index {start_idx} exceeds total files {len(txt_files)}'}

    end_idx = min(start_idx + count, len(txt_files))
    batch_files = txt_files[start_idx:end_idx]

    print(f"Processing papers {start_idx+1} to {end_idx} of {len(txt_files)}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        'success': 0,
        'error': 0,
        'skipped': 0,
        'files': []
    }

    for i, txt_path in enumerate(batch_files):
        result = process_paper(txt_path, OUTPUT_DIR)
        results['files'].append(result)

        if result['status'] == 'success':
            results['success'] += 1
        elif result['status'] == 'error':
            results['error'] += 1
        else:
            results['skipped'] += 1

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(batch_files)}: {results['success']} success, {results['error']} error, {results['skipped']} skipped")

    print(f"\nBatch complete: {results['success']} success, {results['error']} error, {results['skipped']} skipped")

    return results


if __name__ == '__main__':
    import sys

    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    results = batch_process(start_idx=start, count=count)

    # Save results log
    log_path = OUTPUT_DIR.parent / f"batch_log_{start}_{start+count}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Log saved to {log_path}")
