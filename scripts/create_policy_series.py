#!/usr/bin/env python3
"""
Create Materials Science Policy and Strategy series template files.
"""

from pathlib import Path
import re

BASE_DIR = Path(__file__).parent.parent

def create_japanese_index():
    """Create Japanese index.html for policy series."""

    # Read template
    template_path = BASE_DIR / "knowledge/jp/MS/materials-microstructure-introduction/index.html"
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace series-specific content
    replacements = {
        '材料組織学入門シリーズ': '材料科学の政策と戦略入門シリーズ',
        'Introduction to Materials Microstructure Series': 'Introduction to Materials Science Policy and Strategy',
        '結晶粒構造から相変態まで - 組織制御の基礎をマスター': '日本と世界の材料科学政策 - 研究戦略からキャリア展望まで',
        '5 Chapters': '5章',
        '125-175 min': '125-175分',
        '35': '20',  # Code examples count
        'Intermediate': '初級〜中級',

        # Overview paragraph
        '本シリーズは材料組織学とその制御法を基礎から実践まで扱う中級コースです。結晶粒・粒界、相変態、析出、転位といった組織学の基本概念を理解しながら、Pythonを用いた実践的な組織解析スキルを習得します。本シリーズはマテリアルズインフォマティクス（MI）における組織データ解析の基礎知識を提供します。':
        '本シリーズは材料科学を取り巻く政策・戦略・社会的文脈を理解するためのコースです。世界各国（日本・米国・EU・中国など）の材料科学政策、サステナビリティ規制、研究資金の獲得戦略、産業標準、サプライチェーン政策、未来ロードマップまで幅広くカバーします。材料科学の技術的知識だけでなく、社会実装や戦略的意思決定に必要な知識を習得できます。',
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    # Replace Mermaid learning path
    old_mermaid = r'flowchart LR.*?style E fill:#f093fb.*?\s+'
    new_mermaid = '''flowchart LR
    A[第1章<br/>政策ランドスケープ] --> B[第2章<br/>サステナビリティ規制]
    B --> C[第3章<br/>研究資金戦略]
    C --> D[第4章<br/>産業標準・供給網]
    D --> E[第5章<br/>未来展望]

    style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    style B fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    style D fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    style E fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
            '''
    content = re.sub(old_mermaid, new_mermaid, content, flags=re.DOTALL)

    # Replace chapter cards
    chapters = [
        {
            'num': '第1章',
            'title': '材料科学の政策ランドスケープ',
            'description': 'グローバルな材料科学政策の概要、主要国の国家戦略（日本・米国・EU・中国・韓国）、産官学連携の仕組み、材料科学が社会課題解決に果たす役割を学びます。',
            'time': '25-35分',
            'examples': '4',
            'level': '初級'
        },
        {
            'num': '第2章',
            'title': 'サステナビリティと環境規制',
            'description': 'EUグリーンディール、循環経済とマテリアルフロー分析、ライフサイクルアセスメント（LCA）、REACH規制・RoHS指令などの化学物質規制、バッテリー規制について学びます。',
            'time': '25-35分',
            'examples': '4',
            'level': '初級〜中級'
        },
        {
            'num': '第3章',
            'title': '研究資金と助成金戦略',
            'description': '主要研究資金源（科研費・JST・NEDO・NSF・ERCなど）、助成金申請の戦略とベストプラクティス、産学連携資金の獲得方法、研究資金トレンドとホットトピックを学びます。',
            'time': '25-35分',
            'examples': '4',
            'level': '中級'
        },
        {
            'num': '第4章',
            'title': '産業標準とサプライチェーン政策',
            'description': 'ISO材料関連規格、業界別標準（航空宇宙・自動車・半導体）、クリティカルマテリアル政策、サプライチェーン強靭化戦略、貿易政策と材料調達への影響を学びます。',
            'time': '25-35分',
            'examples': '4',
            'level': '中級'
        },
        {
            'num': '第5章',
            'title': '戦略ロードマップと未来展望',
            'description': '各国のマテリアル技術ロードマップ、新興技術領域（次世代電池・水素材料・量子材料）、2030/2050年の材料科学ビジョン、キャリアパスと政策理解の重要性を学びます。',
            'time': '30-40分',
            'examples': '4',
            'level': '中級〜上級'
        }
    ]

    # Build chapter cards HTML
    chapter_cards_html = ''
    for i, ch in enumerate(chapters, 1):
        chapter_cards_html += f'''
        <div class="chapter-card">
            <span class="chapter-number">{ch['num']}</span>
            <div class="chapter-title">{ch['title']}</div>
            <p class="chapter-description">
                {ch['description']}
            </p>
            <div class="chapter-meta">
                <span>⏱️ {ch['time']}</span>
                <span>💻 {ch['examples']}コード例</span>
                <span>📊 {ch['level']}</span>
            </div>
            <div style="margin-top: 1rem;">
                <a href="chapter-{i}.html" class="nav-button" style="display: inline-block;">学習を開始 →</a>
            </div>
        </div>'''

    # Replace chapter cards section
    old_cards = r'<div class="chapter-grid">.*?</div>\s+</div>\s+<h2>学習目標</h2>'
    new_cards = f'<div class="chapter-grid">{chapter_cards_html}\n    </div>\n    <h2>学習目標</h2>'
    content = re.sub(old_cards, new_cards, content, flags=re.DOTALL)

    # Replace learning objectives
    old_objectives = r'<h2>学習目標</h2>.*?<h2>推奨学習パターン</h2>'
    new_objectives = '''<h2>学習目標</h2>
    <p>本シリーズを完了することで、以下のスキルと知識を獲得できます：</p>
    <ul>
        <li>✅ 主要国（日本・米国・EU・中国・韓国）の材料科学政策と国家戦略を理解し、研究方向性への影響を説明できる</li>
        <li>✅ サステナビリティ規制（EUグリーンディール・循環経済・LCA）を理解し、規制準拠の材料選択ができる</li>
        <li>✅ 研究資金源（科研費・JST・NEDO・NSF・ERC）の特徴を把握し、効果的な申請戦略を立案できる</li>
        <li>✅ 産業標準（ISO・業界別規格）とサプライチェーン政策を理解し、リスク評価ができる</li>
        <li>✅ グローバルな技術ロードマップと未来ビジョンを把握し、キャリア戦略に政策視点を統合できる</li>
        <li>✅ 政策文書・データベースの分析にPythonツールを活用できる</li>
        <li>✅ 材料科学の社会実装における政策・規制・標準の役割を理解できる</li>
    </ul>
    <h2>推奨学習パターン</h2>'''
    content = re.sub(old_objectives, new_objectives, content, flags=re.DOTALL)

    # Replace learning patterns
    old_patterns = r'<h2>推奨学習パターン</h2>.*?<h2>前提知識</h2>'
    new_patterns = '''<h2>推奨学習パターン</h2>
    <div class="info-box">
        <h3>パターン1：標準学習 - 理論と実践のバランス（5日間）</h3>
        <ul>
            <li>1日目：第1章（政策ランドスケープ）</li>
            <li>2日目：第2章（サステナビリティと環境規制）</li>
            <li>3日目：第3章（研究資金戦略）</li>
            <li>4日目：第4章（産業標準とサプライチェーン）</li>
            <li>5日目：第5章（未来展望） + 総合復習</li>
        </ul>
    </div>

    <div class="info-box">
        <h3>パターン2：集中学習 - 政策マスター（2-3日間）</h3>
        <ul>
            <li>1日目：第1-2章（基礎理論：政策とサステナビリティ）</li>
            <li>2日目：第3-4章（応用理論：資金獲得と標準化）</li>
            <li>3日目：第5章（未来展望） + 各章演習問題</li>
        </ul>
    </div>

    <div class="info-box">
        <h3>パターン3：実務重視 - 戦略スキル習得（半日）</h3>
        <ul>
            <li>第1-4章：コード例のみ実行（理論は参照）</li>
            <li>第5章：深掘り学習と実際の政策データで分析演習</li>
            <li>必要に応じて理論セクションに戻る</li>
        </ul>
    </div>
    <h2>前提知識</h2>'''
    content = re.sub(old_patterns, new_patterns, content, flags=re.DOTALL)

    # Replace prerequisites table
    old_prereq = r'<h2>前提知識</h2>.*?</tbody>\s+</table>'
    new_prereq = '''<h2>前提知識</h2>
    <table>
        <thead>
            <tr>
                <th>分野</th>
                <th>必要レベル</th>
                <th>説明</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>材料科学基礎</strong></td>
                <td>入門レベル完了</td>
                <td>材料分類、基本物性、応用分野の基礎知識</td>
            </tr>
            <tr>
                <td><strong>Python</strong></td>
                <td>基礎レベル</td>
                <td>基本文法、データ可視化（matplotlib）、データ分析（pandas）の基礎</td>
            </tr>
            <tr>
                <td><strong>英語読解力</strong></td>
                <td>中級レベル</td>
                <td>政策文書・学術論文の読解（本シリーズで日本語解説あり）</td>
            </tr>
            <tr>
                <td><strong>社会科学基礎</strong></td>
                <td>不要</td>
                <td>政策・経済・法規制の知識は本シリーズで学習</td>
            </tr>
        </tbody>
    </table>'''
    content = re.sub(old_prereq, new_prereq, content, flags=re.DOTALL)

    # Update breadcrumb
    content = content.replace('materials-microstructure-introduction', 'materials-science-policy-strategy-introduction')
    content = content.replace('材料組織学', '材料科学政策・戦略')
    content = content.replace('Materials Microstructure', 'Materials Science Policy')

    # Write to new location
    output_path = BASE_DIR / "knowledge/jp/MS/materials-science-policy-strategy-introduction/index.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Created: {output_path}")
    return True


def main():
    """Main execution."""
    print("=" * 70)
    print("Creating Materials Science Policy Series Templates")
    print("=" * 70)

    print("\n[1/1] Creating Japanese index.html...")
    create_japanese_index()

    print("\n" + "=" * 70)
    print("Template Creation Complete")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review generated index.html")
    print("2. Create chapter-1.html template")
    print("3. Add detailed content to chapters")

    return 0


if __name__ == "__main__":
    exit(main())
