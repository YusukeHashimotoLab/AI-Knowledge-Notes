#!/usr/bin/env python3
"""
パンくずリスト自動追加スクリプト

全ページに階層的なパンくずリストを追加:
AI寺子屋トップ > ドメイン > シリーズ > Chapter
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# ドメイン名マッピング
DOMAIN_NAMES = {
    'MI': 'マテリアルズ・インフォマティクス',
    'PI': 'プロセス・インフォマティクス',
    'ML': '機械学習',
    'MS': '材料科学',
    'FM': '基礎数理'
}

# パンくずリストCSS
BREADCRUMB_CSS = """
        /* Breadcrumb styles */
        .breadcrumb {
            background: #f7fafc;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.9rem;
        }

        .breadcrumb-content {
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .breadcrumb a {
            color: #667eea;
            text-decoration: none;
            transition: color 0.2s;
        }

        .breadcrumb a:hover {
            color: #764ba2;
            text-decoration: underline;
        }

        .breadcrumb-separator {
            color: #a0aec0;
            margin: 0 0.25rem;
        }

        .breadcrumb-current {
            color: #4a5568;
            font-weight: 500;
        }
"""


def get_breadcrumb_path(file_path: Path, base_dir: Path) -> List[Tuple[str, str]]:
    """
    ファイルパスからパンくずリストのパスを生成

    Returns:
        List of (title, url) tuples
    """
    breadcrumbs = [('AI寺子屋トップ', '/AI-Knowledge-Notes/knowledge/jp/index.html')]

    relative_path = file_path.relative_to(base_dir)
    parts = relative_path.parts

    if len(parts) < 1:
        return breadcrumbs

    # ドメインレベル (MI, MS, etc.)
    if len(parts) >= 1:
        domain = parts[0]
        domain_name = DOMAIN_NAMES.get(domain, domain)
        breadcrumbs.append((domain_name, f'/AI-Knowledge-Notes/knowledge/jp/{domain}/index.html'))

    # シリーズレベル
    if len(parts) >= 2 and parts[1] != 'index.html':
        series_dir = parts[1]
        # シリーズ名を読みやすく変換
        series_name = series_dir.replace('-introduction', '').replace('-', ' ').title()
        breadcrumbs.append((series_name, f'/AI-Knowledge-Notes/knowledge/jp/{domain}/{series_dir}/index.html'))

    # チャプターレベル
    if len(parts) >= 3 and parts[2] != 'index.html':
        chapter_file = parts[2]
        chapter_name = chapter_file.replace('.html', '')

        # 様々な命名パターンに対応
        if chapter_file.startswith('chapter-'):
            # chapter-1.html, chapter-2.html
            chapter_num = chapter_file.replace('chapter-', '').replace('.html', '')
            breadcrumbs.append((f'Chapter {chapter_num}', ''))
        elif chapter_file.startswith('chapter'):
            # chapter1-introduction.html, chapter2-fundamentals.html
            chapter_num = chapter_name.split('-')[0].replace('chapter', '')
            breadcrumbs.append((f'Chapter {chapter_num}', ''))
        else:
            # その他のファイル名（そのまま表示）
            display_name = chapter_name.replace('-', ' ').title()
            breadcrumbs.append((display_name, ''))

    return breadcrumbs


def generate_breadcrumb_html(breadcrumbs: List[Tuple[str, str]]) -> str:
    """
    パンくずリストのHTML生成
    """
    items = []

    for i, (title, url) in enumerate(breadcrumbs):
        if i > 0:
            items.append('<span class="breadcrumb-separator">›</span>')

        if url:  # リンクあり
            items.append(f'<a href="{url}">{title}</a>')
        else:  # 現在ページ
            items.append(f'<span class="breadcrumb-current">{title}</span>')

    html = f'''    <nav class="breadcrumb">
        <div class="breadcrumb-content">
            {''.join(items)}
        </div>
    </nav>
'''
    return html


def add_breadcrumb_to_file(file_path: Path, base_dir: Path, dry_run: bool = False) -> bool:
    """
    HTMLファイルにパンくずリストを追加

    Returns:
        True if modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 既にパンくずリストが存在する場合はスキップ
        if 'class="breadcrumb"' in content:
            return False

        # CSSの追加
        if BREADCRUMB_CSS.strip() not in content:
            # </style>の直前に追加
            content = content.replace('</style>', f'{BREADCRUMB_CSS}    </style>')

        # パンくずリストHTML生成
        breadcrumbs = get_breadcrumb_path(file_path, base_dir)
        breadcrumb_html = generate_breadcrumb_html(breadcrumbs)

        # <header>タグの直前に挿入
        if '<header>' in content or '<header ' in content:
            # <header>または<header ...>の前に挿入
            content = re.sub(
                r'(\s*)(<header[^>]*>)',
                f'\\1{breadcrumb_html}\\1\\2',
                content,
                count=1
            )
        else:
            # headerがない場合は<body>直後に挿入
            content = re.sub(
                r'(<body[^>]*>\s*)',
                f'\\1{breadcrumb_html}',
                content,
                count=1
            )

        if dry_run:
            print(f"[DRY RUN] Would modify: {file_path.relative_to(base_dir)}")
            return True

        # ファイルに書き込み
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """
    メイン処理
    """
    import argparse

    parser = argparse.ArgumentParser(description='全ページにパンくずリストを追加')
    parser.add_argument('--dry-run', action='store_true', help='変更内容を表示するのみ')
    parser.add_argument('--domain', help='特定ドメインのみ処理 (MI, MS, etc.)')
    args = parser.parse_args()

    base_dir = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp')

    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return

    # 処理対象ドメイン
    domains = [args.domain] if args.domain else ['MI', 'PI', 'ML', 'MS', 'FM']

    total_files = 0
    modified_files = 0

    for domain in domains:
        domain_dir = base_dir / domain
        if not domain_dir.exists():
            print(f"Warning: Domain directory not found: {domain_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing domain: {domain} ({DOMAIN_NAMES.get(domain, domain)})")
        print(f"{'='*60}")

        # ドメインindex.html
        domain_index = domain_dir / 'index.html'
        if domain_index.exists():
            total_files += 1
            if add_breadcrumb_to_file(domain_index, base_dir, args.dry_run):
                modified_files += 1
                print(f"  ✓ {domain_index.relative_to(base_dir)}")

        # 各シリーズディレクトリ
        for series_dir in sorted(domain_dir.iterdir()):
            if not series_dir.is_dir():
                continue

            # シリーズindex.html
            series_index = series_dir / 'index.html'
            if series_index.exists():
                total_files += 1
                if add_breadcrumb_to_file(series_index, base_dir, args.dry_run):
                    modified_files += 1
                    print(f"  ✓ {series_index.relative_to(base_dir)}")

            # 各チャプターファイル（全HTMLファイルを処理）
            for html_file in sorted(series_dir.glob('*.html')):
                # index.htmlは既に処理済みなのでスキップ
                if html_file.name == 'index.html':
                    continue

                total_files += 1
                if add_breadcrumb_to_file(html_file, base_dir, args.dry_run):
                    modified_files += 1
                    print(f"  ✓ {html_file.relative_to(base_dir)}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Modified files: {modified_files}")
    print(f"  Skipped (already has breadcrumb): {total_files - modified_files}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n⚠️  DRY RUN mode - no files were actually modified")
        print("Run without --dry-run to apply changes")


if __name__ == '__main__':
    main()
