#!/usr/bin/env python3
"""
Spin Wave Papers Index Generator
Generates main index.html and theme sub-index pages
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

OUTPUT_DIR = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/restricted/jp/research/spin-wave-references")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
THEMES_DIR = OUTPUT_DIR / "themes"

THEME_CONFIG = {
    "spin-wave-physics": {
        "name_ja": "スピン波物理",
        "description": "スピン波（マグノン）の基礎物理、磁気共鳴、YIGフィルムなどの研究",
        "gradient": "#667eea 0%, #764ba2 100%",
        "color": "#667eea"
    },
    "magnon-phonon": {
        "name_ja": "マグノン-フォノン結合",
        "description": "磁気弾性効果、音響波との結合、フォノンモードの研究",
        "gradient": "#11998e 0%, #38ef7d 100%",
        "color": "#11998e"
    },
    "optical-control": {
        "name_ja": "光制御",
        "description": "レーザーによる磁化制御、光磁気効果、光誘起現象の研究",
        "gradient": "#f093fb 0%, #f5576c 100%",
        "color": "#f093fb"
    },
    "ultrafast-dynamics": {
        "name_ja": "超高速ダイナミクス",
        "description": "フェムト秒レーザー、テラヘルツ分光、超高速消磁の研究",
        "gradient": "#4facfe 0%, #00f2fe 100%",
        "color": "#4facfe"
    },
    "multiferroics": {
        "name_ja": "マルチフェロイクス",
        "description": "BiFeO3などの強誘電-強磁性共存系、磁気電気効果の研究",
        "gradient": "#fa709a 0%, #fee140 100%",
        "color": "#fa709a"
    },
    "spintronics": {
        "name_ja": "スピントロニクス",
        "description": "スピン流、スピンホール効果、スピン移行トルクの研究",
        "gradient": "#a8edea 0%, #fed6e3 100%",
        "color": "#a8edea"
    },
    "brillouin-scattering": {
        "name_ja": "ブリルアン散乱",
        "description": "ブリルアン光散乱（BLS）によるマグノン分光研究",
        "gradient": "#ffecd2 0%, #fcb69f 100%",
        "color": "#ffecd2"
    },
    "theoretical": {
        "name_ja": "理論・計算",
        "description": "第一原理計算、理論モデル、シミュレーション研究",
        "gradient": "#ff9a9e 0%, #fecfef 100%",
        "color": "#ff9a9e"
    }
}


def load_manifest():
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_main_index(manifest):
    """Generate main index.html"""
    papers = manifest["papers"]
    total = manifest["total_papers"]

    # Count by theme and year
    by_theme = defaultdict(list)
    by_year = defaultdict(int)

    for paper in papers:
        theme = paper.get("theme", "theoretical")
        year = paper.get("year", "unknown")
        by_theme[theme].append(paper)
        by_year[year] += 1

    # Sort years
    years_sorted = sorted([y for y in by_year.keys() if y != "unknown"], key=lambda x: int(x) if x.isdigit() else 0)
    min_year = years_sorted[0] if years_sorted else "N/A"
    max_year = years_sorted[-1] if years_sorted else "N/A"

    # Generate theme cards HTML
    theme_cards_html = ""
    for theme_id, config in THEME_CONFIG.items():
        count = len(by_theme.get(theme_id, []))
        theme_cards_html += f'''
      <a href="themes/{theme_id}.html" class="theme-card">
        <div class="theme-card-header" style="background: linear-gradient(135deg, {config['gradient']});">
          <span class="theme-name">{config['name_ja']}</span>
          <span class="theme-count">{count}論文</span>
        </div>
        <div class="theme-card-body">
          <p>{config['description']}</p>
        </div>
      </a>'''

    # Recent papers (last 20 by year)
    recent_papers = sorted(papers, key=lambda x: x.get("year", "0000"), reverse=True)[:20]
    recent_papers_html = ""
    for paper in recent_papers:
        title = (paper.get("title") or "Unknown")[:60]
        author = paper.get("first_author") or "unknown"
        year = paper.get("year") or "unknown"
        output = paper.get("output") or ""
        theme = paper.get("theme") or "theoretical"
        theme_name = THEME_CONFIG.get(theme, {}).get("name_ja", "その他")

        recent_papers_html += f'''
      <div class="paper-card">
        <div class="paper-card-header theme-{theme}">{theme_name}</div>
        <div class="paper-card-body">
          <div class="paper-title">{title}</div>
          <div class="paper-authors">{author}</div>
          <div class="paper-meta"><span>{year}</span></div>
          <a href="papers/{output}" class="paper-link">詳細を読む</a>
        </div>
      </div>'''

    html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<meta content="Spin Wave Research Papers - スピン波研究論文の日本語解説コレクション（{total}論文収録）" name="description"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Spin Wave References - スピン波研究論文コレクション - AI Terakoya</title>
<link href="../../assets/css/knowledge-base.css" rel="stylesheet"/>
<style>
  .theme-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 2rem;
  }}
  .theme-card {{
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    overflow: hidden;
    transition: transform 0.3s, box-shadow 0.3s;
    text-decoration: none;
    color: inherit;
  }}
  .theme-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
  }}
  .theme-card-header {{
    padding: 1.5rem;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .theme-name {{
    font-weight: 600;
    font-size: 1.1rem;
  }}
  .theme-count {{
    background: rgba(255,255,255,0.3);
    padding: 0.3rem 0.8rem;
    border-radius: 12px;
    font-size: 0.85rem;
  }}
  .theme-card-body {{
    padding: 1.2rem;
  }}
  .theme-card-body p {{
    margin: 0;
    font-size: 0.9rem;
    color: #666;
  }}
  .paper-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 2rem;
  }}
  .paper-card {{
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    overflow: hidden;
    transition: transform 0.3s;
  }}
  .paper-card:hover {{
    transform: translateY(-2px);
  }}
  .paper-card-header {{
    padding: 0.8rem 1.2rem;
    color: white;
    font-weight: 600;
    font-size: 0.85rem;
  }}
  .theme-spin-wave-physics {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
  .theme-magnon-phonon {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
  .theme-optical-control {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
  .theme-ultrafast-dynamics {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
  .theme-multiferroics {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }}
  .theme-spintronics {{ background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; }}
  .theme-brillouin-scattering {{ background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333; }}
  .theme-theoretical {{ background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); color: #333; }}
  .paper-card-body {{
    padding: 1.2rem;
  }}
  .paper-title {{
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    line-height: 1.4;
  }}
  .paper-authors {{
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 0.5rem;
  }}
  .paper-meta {{
    font-size: 0.8rem;
    color: #888;
    margin-bottom: 0.8rem;
  }}
  .paper-link {{
    display: inline-block;
    padding: 0.5rem 1rem;
    background: #667eea;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    font-size: 0.85rem;
    transition: background 0.3s;
  }}
  .paper-link:hover {{
    background: #5a67d8;
  }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
  }}
  .stat-card {{
    background: #f7fafc;
    padding: 1.5rem;
    border-radius: 8px;
    text-align: center;
  }}
  .stat-number {{
    font-size: 2rem;
    font-weight: 700;
    color: #667eea;
  }}
  .stat-label {{
    font-size: 0.9rem;
    color: #666;
    margin-top: 0.3rem;
  }}
  .search-box {{
    margin: 2rem 0;
    padding: 1.5rem;
    background: #f7fafc;
    border-radius: 8px;
  }}
  .search-input {{
    width: 100%;
    padding: 1rem;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    font-size: 1rem;
  }}
  .search-input:focus {{
    outline: none;
    border-color: #667eea;
  }}
</style>
</head>
<body>
<nav class="breadcrumb">
  <div class="breadcrumb-content">
    <a href="../../index.html">AI Terakoya Top</a>
    <span class="breadcrumb-separator">&gt;</span>
    <a href="../index.html">研究</a>
    <span class="breadcrumb-separator">&gt;</span>
    <span class="breadcrumb-current">Spin Wave References</span>
  </div>
</nav>

<div class="locale-switcher">
  <span class="current-locale">JP</span>
  <span class="locale-separator">|</span>
  <span class="locale-meta">最終更新: {datetime.now().strftime('%Y-%m-%d')}</span>
</div>

<header>
  <div class="container">
    <h1>Spin Wave Research References</h1>
    <p class="subtitle">スピン波研究論文コレクション</p>
    <div class="series-meta">
      <span>{total}論文</span>
      <span>{min_year}-{max_year}年</span>
      <span>8つの研究テーマ</span>
    </div>
  </div>
</header>

<main class="container">
  <section class="intro">
    <h2>このコレクションについて</h2>
    <p>本コレクションは、スピン波（マグノン）研究に関する重要論文を日本語で解説したものです。スピン波物理、マグノン-フォノン結合、光制御、超高速ダイナミクス、マルチフェロイクス、スピントロニクス、ブリルアン散乱、理論計算の8つのテーマに分類されています。</p>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-number">{total}</div>
        <div class="stat-label">収録論文数</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">{len(years_sorted)}</div>
        <div class="stat-label">年間スパン</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">8</div>
        <div class="stat-label">研究テーマ</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">{len(by_theme.get('optical-control', []))}</div>
        <div class="stat-label">光制御論文</div>
      </div>
    </div>
  </section>

  <section>
    <h2>研究テーマ別</h2>
    <div class="theme-grid">
{theme_cards_html}
    </div>
  </section>

  <section>
    <h2>最新論文</h2>
    <div class="paper-grid">
{recent_papers_html}
    </div>
  </section>

  <div class="nav-buttons">
    <a class="nav-button" href="../index.html">研究トップに戻る</a>
  </div>
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
    output_path = OUTPUT_DIR / "index.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Generated: {output_path}")


def generate_theme_index(theme_id, papers, config):
    """Generate theme sub-index page"""
    papers_sorted = sorted(papers, key=lambda x: x.get("year", "0000"), reverse=True)

    papers_html = ""
    for paper in papers_sorted:
        title = (paper.get("title") or "Unknown")[:70]
        author = paper.get("first_author") or "unknown"
        year = paper.get("year") or "unknown"
        output = paper.get("output") or ""

        papers_html += f'''
      <div class="paper-card">
        <div class="paper-card-body">
          <div class="paper-title">{title}</div>
          <div class="paper-authors">{author}</div>
          <div class="paper-meta"><span>{year}</span></div>
          <a href="../papers/{output}" class="paper-link">詳細を読む</a>
        </div>
      </div>'''

    html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<meta content="{config['name_ja']} - スピン波研究論文（{len(papers)}論文）" name="description"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>{config['name_ja']} - Spin Wave References - AI Terakoya</title>
<link href="../../../assets/css/knowledge-base.css" rel="stylesheet"/>
<style>
  .paper-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 2rem;
  }}
  .paper-card {{
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    overflow: hidden;
    transition: transform 0.3s;
  }}
  .paper-card:hover {{
    transform: translateY(-2px);
  }}
  .paper-card-body {{
    padding: 1.2rem;
  }}
  .paper-title {{
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    line-height: 1.4;
  }}
  .paper-authors {{
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 0.5rem;
  }}
  .paper-meta {{
    font-size: 0.8rem;
    color: #888;
    margin-bottom: 0.8rem;
  }}
  .paper-link {{
    display: inline-block;
    padding: 0.5rem 1rem;
    background: {config['color']};
    color: white;
    text-decoration: none;
    border-radius: 4px;
    font-size: 0.85rem;
    transition: opacity 0.3s;
  }}
  .paper-link:hover {{
    opacity: 0.9;
  }}
  .theme-header {{
    background: linear-gradient(135deg, {config['gradient']});
    color: white;
    padding: 2rem;
    border-radius: 8px;
    margin-bottom: 2rem;
  }}
  .theme-header h2 {{
    color: white;
    margin: 0 0 0.5rem 0;
  }}
  .theme-header p {{
    margin: 0;
    opacity: 0.9;
  }}
</style>
</head>
<body>
<nav class="breadcrumb">
  <div class="breadcrumb-content">
    <a href="../../../index.html">AI Terakoya Top</a>
    <span class="breadcrumb-separator">&gt;</span>
    <a href="../../index.html">研究</a>
    <span class="breadcrumb-separator">&gt;</span>
    <a href="../index.html">Spin Wave References</a>
    <span class="breadcrumb-separator">&gt;</span>
    <span class="breadcrumb-current">{config['name_ja']}</span>
  </div>
</nav>

<header>
  <div class="container">
    <h1>{config['name_ja']}</h1>
    <div class="series-meta">
      <span>{len(papers)}論文</span>
      <span>Spin Wave References</span>
    </div>
  </div>
</header>

<main class="container">
  <div class="theme-header">
    <h2>{config['name_ja']}</h2>
    <p>{config['description']}</p>
  </div>

  <section>
    <h2>論文一覧（{len(papers)}件）</h2>
    <div class="paper-grid">
{papers_html}
    </div>
  </section>

  <div class="nav-buttons">
    <a class="nav-button" href="../index.html">Spin Wave Referencesに戻る</a>
  </div>
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
    output_path = THEMES_DIR / f"{theme_id}.html"
    THEMES_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Generated: {output_path}")


def generate_all_indexes():
    """Generate all index pages"""
    manifest = load_manifest()
    papers = manifest["papers"]

    # Group by theme
    by_theme = defaultdict(list)
    for paper in papers:
        theme = paper.get("theme", "theoretical")
        by_theme[theme].append(paper)

    # Generate main index
    generate_main_index(manifest)

    # Generate theme indexes
    for theme_id, config in THEME_CONFIG.items():
        generate_theme_index(theme_id, by_theme.get(theme_id, []), config)

    print(f"\nAll indexes generated!")


if __name__ == "__main__":
    generate_all_indexes()
