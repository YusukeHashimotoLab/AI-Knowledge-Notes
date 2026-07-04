#!/usr/bin/env python3
"""
Add disclaimer section to all HTML files that are missing it.

Adds a standardized disclaimer section before the footer in both
English and Japanese knowledge base articles.
"""

import re
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).parent.parent
EN_DIR = BASE_DIR / "knowledge" / "en"
JP_DIR = BASE_DIR / "knowledge" / "jp"

# English disclaimer template
EN_DISCLAIMER = """<section class="disclaimer">
<h3>Disclaimer</h3>
<ul>
<li>This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).</li>
<li>This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.</li>
<li>The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.</li>
<li>To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.</li>
<li>The content may be changed, updated, or discontinued without notice.</li>
<li>The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.</li>
</ul>
</section>
"""

# Japanese disclaimer template
JP_DISCLAIMER = """<section class="disclaimer">
<h3>免責事項</h3>
<ul>
<li>本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。</li>
<li>本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。</li>
<li>外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。</li>
<li>本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。</li>
<li>本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。</li>
<li>本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。</li>
</ul>
</section>
"""


def has_disclaimer(content: str) -> bool:
    """Check if content already has a disclaimer."""
    return 'disclaimer' in content.lower() or '免責事項' in content


def add_disclaimer_to_file(file_path: Path, is_japanese: bool = False) -> Tuple[bool, str]:
    """Add disclaimer section to HTML file if missing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if already has disclaimer
        if has_disclaimer(content):
            return False, "Disclaimer already exists"

        # Select appropriate template
        disclaimer = JP_DISCLAIMER if is_japanese else EN_DISCLAIMER

        # Try to insert before </main>
        if '</main>' in content:
            new_content = content.replace('</main>', disclaimer + '\n</main>', 1)
        # Try to insert before <footer>
        elif '<footer>' in content or '<footer ' in content:
            new_content = re.sub(r'(<footer[^>]*>)', disclaimer + r'\n\1', content, count=1)
        # Try to insert before </body>
        elif '</body>' in content:
            new_content = content.replace('</body>', disclaimer + '\n</body>', 1)
        else:
            return False, "No suitable insertion point found"

        # Verify change was made
        if new_content == content:
            return False, "No changes made"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True, "Disclaimer added successfully"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Main execution."""
    print("=" * 70)
    print("Adding Disclaimer Sections to HTML Files")
    print("=" * 70)

    # Process English files
    print("\n[1/2] Processing English files...")
    en_files = list(EN_DIR.rglob("*.html"))
    en_processed = 0
    en_added = 0

    for en_file in en_files:
        en_processed += 1
        success, message = add_disclaimer_to_file(en_file, is_japanese=False)

        if success:
            en_added += 1
            if en_added <= 5:  # Show first 5
                print(f"  ✅ {en_file.relative_to(BASE_DIR)}")

    if en_added > 5:
        print(f"\n  ... and {en_added - 5} more files")

    print(f"\nEnglish: {en_added}/{en_processed} files updated")

    # Process Japanese files
    print("\n[2/2] Processing Japanese files...")
    jp_files = list(JP_DIR.rglob("*.html"))
    jp_processed = 0
    jp_added = 0

    for jp_file in jp_files:
        jp_processed += 1
        success, message = add_disclaimer_to_file(jp_file, is_japanese=True)

        if success:
            jp_added += 1
            if jp_added <= 5:  # Show first 5
                print(f"  ✅ {jp_file.relative_to(BASE_DIR)}")

    if jp_added > 5:
        print(f"\n  ... and {jp_added - 5} more files")

    print(f"\nJapanese: {jp_added}/{jp_processed} files updated")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"English files:  {en_added}/{en_processed} updated")
    print(f"Japanese files: {jp_added}/{jp_processed} updated")
    print(f"Total:          {en_added + jp_added} disclaimers added")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
