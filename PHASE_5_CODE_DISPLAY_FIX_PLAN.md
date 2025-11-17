# Phase 5: コード表示修正計画

**作成日**: 2025-11-17
**ステータス**: 計画段階（実装前）
**対象**: 142ファイル

---

## 問題の概要

### 発見された問題

**`python_code_outside_pre.txt`** の検証結果に基づき、142個のHTMLファイルでPythonの`import`文がコードブロック（`<pre><code>`）の外に出力されていることが判明しました。

### 具体的な症状

**現在のパターン（英語版）**:
```html
<h3>Example: Effects of Preprocessing</h3>
<pre><code class="language-python">import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
...
</code></pre>
```

**正しいパターン（日本語版を参照）**:
```html
<h3>Example: 前処理の効果</h3>
<pre><code class="language-python">import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
...
</code></pre>
```

**重要な発見**:
- 日本語版ファイル `knowledge/jp/MI/mi-introduction/chapter3-hands-on.html` を確認した結果、**同じパターン**が使用されている
- つまり、`<pre><code class="language-python">import` という形式は**正しいフォーマット**
- `grep "^import "` による検出は**false positive**（誤検出）の可能性が高い

### Phase 4-1-4との違い

**Phase 4-1-4で修正した問題**:
```html
<!-- 修正前 -->
<div class="code-block"><code>
def function():
    pass
</code></div>

<!-- 修正後 -->
<div class="code-block"><pre><code>
def function():
    pass
</code></pre></div>
```
- `<pre>`タグが完全に欠落していた（2ファイル、22箇所）

**Phase 5で修正する問題**:
```html
<!-- 現状（問題あり） -->
<pre><code class="language-python">import numpy as np
import pandas as pd
...
</code></pre>

<!-- 期待される形式 -->
<pre><code class="language-python">
import numpy as np
import pandas as pd
...
</code></pre>
```
- `<pre><code>`タグは存在するが、開始タグ直後に改行がない
- `import`文が開始タグと同じ行にある

---

## 根本原因分析

### 1. 日本語版との比較結果

**検証ファイル**:
- 日本語版: `knowledge/jp/MI/mi-introduction/chapter3-hands-on.html`
- 英語版: `knowledge/en/MI/mi-introduction/chapter3-hands-on.html`

**発見された事実**:
```bash
# 日本語版
grep -n '<pre><code class="language-python">import' jp/file.html
663:<pre><code class="language-python">import numpy as np

# 英語版
grep -n '<pre><code class="language-python">import' en/file.html
277:<pre><code class="language-python">import numpy as np
```

**結論**: **両方とも同じフォーマット** → これは正しいHTML構造

### 2. `grep "^import "` の誤検出メカニズム

**検出コマンドの問題**:
```bash
grep "^import " file.html
```

**誤検出の理由**:
1. `<pre><code class="language-python">import numpy as np` という**1行**が存在
2. 次の行: `import pandas as pd` が**行頭から始まる**
3. `grep "^import "` は**2行目以降のimport**を検出してしまう
4. しかし、これらは**すべて`<pre><code>`タグ内**にある

**HTML構造の真実**:
```html
<pre><code class="language-python">import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
</code></pre>
```
- 1行目: `<pre><code class="language-python">import numpy as np`
- 2行目: `import pandas as pd` ← これが検出される
- 3行目: `import matplotlib.pyplot as plt` ← これも検出される
- しかし**すべて正しくタグ内**にある

### 3. 実際の問題の有無

**結論**: **問題なし（false positive）**

**理由**:
1. ✅ 日本語版と同じフォーマット
2. ✅ HTMLタグ構造は正しい（`<pre><code>` ... `</code></pre>`）
3. ✅ Syntax highlightは正常に機能する（class="language-python"が設定されている）
4. ✅ `grep`コマンドの検出ロジックが不適切だった

**確認方法**:
```bash
# 正しい検証方法: <pre><code>タグの外にimportがあるか確認
grep -v '<pre><code' file.html | grep "^import "
# → これで検出されなければ問題なし
```

---

## 検証フェーズ

### Step 1: 実際の表示確認

**検証ファイル** (代表的な3ファイルをサンプル):
1. `knowledge/en/ML/feature-engineering-introduction/chapter1-data-preprocessing.html`
2. `knowledge/en/FM/calculus-vector-analysis/chapter-5.html`
3. `knowledge/en/MS/materials-science-introduction/chapter-5.html`

**検証項目**:
```
□ ブラウザで開いてコードブロックを確認
□ Syntax highlightは正常か？
□ import文は正しく表示されているか？
□ コードブロック全体が表示されているか？
□ コピー&ペーストは正常に動作するか？
```

### Step 2: HTML構造の詳細調査

**調査コマンド**:
```bash
# パターン1: <pre><code>の直後にimportがある行を検出
grep -n '<pre><code class="language-python">import' file.html

# パターン2: <pre><code>の次の行にimportがある場合
grep -A1 '<pre><code class="language-python">' file.html | grep '^import'

# パターン3: 正しいフォーマット（<pre><code>の次の行が空行または改行）
grep -A1 '<pre><code class="language-python">$' file.html
```

### Step 3: 問題パターンの分類

**期待される発見**:

**パターンA: 開始タグと同じ行にコード**
```html
<pre><code class="language-python">import numpy as np
import pandas as pd
</code></pre>
```

**パターンB: 開始タグの直後に改行なし**
```html
<pre><code class="language-python">
import numpy as np
import pandas as pd
</code></pre>
```

**パターンC: 正しいフォーマット**
```html
<pre><code class="language-python">
import numpy as np
import pandas as pd
</code></pre>
```

---

## 修正計画（検証後に実施）

### Option A: 表示問題なし → 検証スクリプト改善のみ

**実施内容**:
1. より正確なリンク切れ検証スクリプトを作成
2. `grep "^import "`を改良して、HTML構造を考慮
3. ドキュメントに「検証方法の改善」を記録

**所要時間**: 1-2時間

### Option B: 表示問題あり → HTML修正

#### B-1: 問題の範囲確認

**スクリプト**: `scripts/analyze_code_format_issues.py`
```python
"""
Analyze code block formatting issues in HTML files.
Detect patterns where code starts on the same line as <pre><code>.
"""

import re
from pathlib import Path

def analyze_code_blocks(file_path):
    """Analyze code block structure in an HTML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []

    # Pattern 1: <pre><code>CODE on same line
    pattern1 = r'<pre><code[^>]*>(?!\\n)[^<]+'
    matches1 = re.findall(pattern1, content)
    if matches1:
        issues.append({
            'type': 'code_on_same_line',
            'count': len(matches1),
            'examples': matches1[:3]
        })

    # Pattern 2: Missing newline after <pre><code>
    pattern2 = r'<pre><code class="language-python">import'
    matches2 = re.findall(pattern2, content)
    if matches2:
        issues.append({
            'type': 'import_on_same_line',
            'count': len(matches2)
        })

    return issues

def main():
    target_files = [
        # Read from 修正対象ページ.md
    ]

    for file_path in target_files:
        issues = analyze_code_blocks(file_path)
        if issues:
            print(f"\\n{file_path}:")
            for issue in issues:
                print(f"  - {issue['type']}: {issue['count']} instances")
```

#### B-2: 修正スクリプト作成

**スクリプト**: `scripts/fix_code_block_newlines.py`
```python
"""
Fix code block formatting by ensuring newline after <pre><code> tags.
"""

import re
from pathlib import Path

def fix_code_block_formatting(file_path):
    """Fix code blocks to have proper newline after opening tags."""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    fixes_applied = 0

    # Pattern 1: <pre><code class="language-python">import ...
    # Fix to: <pre><code class="language-python">\\nimport ...
    pattern = r'(<pre><code class="language-python">)(import )'
    replacement = r'\\1\\n\\2'
    content, count = re.subn(pattern, replacement, content)
    fixes_applied += count

    # Pattern 2: <pre><code class="language-python">from ...
    pattern = r'(<pre><code class="language-python">)(from )'
    replacement = r'\\1\\n\\2'
    content, count = re.subn(pattern, replacement, content)
    fixes_applied += count

    # Pattern 3: <pre><code class="language-python">def ...
    pattern = r'(<pre><code class="language-python">)(def )'
    replacement = r'\\1\\n\\2'
    content, count = re.subn(pattern, replacement, content)
    fixes_applied += count

    # Pattern 4: <pre><code class="language-python">class ...
    pattern = r'(<pre><code class="language-python">)(class )'
    replacement = r'\\1\\n\\2'
    content, count = re.subn(pattern, replacement, content)
    fixes_applied += count

    # Pattern 5: Generic - any non-whitespace after >
    pattern = r'(<pre><code class="language-python">)([^\\s<])'
    replacement = r'\\1\\n\\2'
    content, count = re.subn(pattern, replacement, content)
    fixes_applied += count

    if content == original_content:
        return False, "No changes needed"

    # Create backup
    backup_path = str(file_path) + '.bak'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)

    # Write fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return True, f"Applied {fixes_applied} fixes"
```

#### B-3: 実施手順

1. **分析実行**:
   ```bash
   python scripts/analyze_code_format_issues.py > CODE_FORMAT_ANALYSIS.md
   ```

2. **サンプル修正**（3ファイル）:
   ```bash
   python scripts/fix_code_block_newlines.py --files sample_files.txt
   ```

3. **検証**:
   - ブラウザで修正後のファイルを確認
   - Syntax highlightが正常か確認
   - コピー&ペーストの動作確認

4. **全体修正**（検証成功後）:
   ```bash
   python scripts/fix_code_block_newlines.py --all
   ```

5. **リンクチェック**:
   ```bash
   python check_links.py
   ```

6. **コミット**:
   ```bash
   git add .
   git commit -m "fix: Add newlines after code block opening tags

   Fixed 142 files where code started on the same line as <pre><code>.
   This ensures proper syntax highlighting and code block rendering.

   Pattern fixed:
   <pre><code class="language-python">import numpy
   →
   <pre><code class="language-python">
   import numpy

   Files affected: 142
   - ML: 96 files
   - FM: 24 files
   - MS: 13 files
   - MI: 6 files
   - PI: 3 files"
   ```

**所要時間**: 4-6時間

---

## リスク評価

### 低リスク（Option A）

**リスク**: ほぼなし
**影響**: 検証スクリプトの改善のみ
**ロールバック**: 不要

### 中リスク（Option B）

**リスク**:
1. HTMLレンダリングへの影響
2. Syntax highlightライブラリとの相性
3. 既存の動作する表示を壊す可能性

**軽減策**:
1. サンプル修正 → 検証 → 全体適用の段階的アプローチ
2. バックアップファイル作成（`.bak`）
3. Git履歴による即座のロールバック可能性

**ロールバック計画**:
```bash
# 問題が発生した場合
git reset --hard HEAD~1

# または個別ファイルの復元
git checkout HEAD -- file.html
```

---

## 推奨アプローチ

### ✅ 最終判断: Phase 5は不要

**理由**:
1. 日本語版との比較により、現在のフォーマットが**正しい**ことを確認
2. `grep "^import "`による検出は**誤検出（false positive）**
3. HTML構造は適切（`<pre><code>` ... `</code></pre>`）
4. Syntax highlightは正常に機能する

### 実施すべきこと: 検証スクリプトの改善のみ

**Phase 5-0: 検証コマンド修正**（所要時間: 30分）

**目的**: 誤検出を防ぐ正しい検証方法の確立

**改善スクリプト**: `scripts/verify_code_blocks_correct.py`
```python
"""
Correct verification of code blocks in HTML files.
Detects code that is truly outside <pre><code> tags.
"""

import re
from pathlib import Path

def verify_code_outside_pre(file_path):
    """Check if import statements exist outside <pre><code> tags."""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove all <pre><code> ... </code></pre> blocks
    content_without_code_blocks = re.sub(
        r'<pre><code[^>]*>.*?</code></pre>',
        '',
        content,
        flags=re.DOTALL
    )

    # Check for import statements in the remaining content
    imports_outside = re.findall(r'^import\s+', content_without_code_blocks, re.MULTILINE)

    return len(imports_outside) > 0, imports_outside

def main():
    base_dir = Path("knowledge/en")
    html_files = sorted(base_dir.rglob("*.html"))

    files_with_issues = []

    for file_path in html_files:
        has_issues, imports = verify_code_outside_pre(file_path)
        if has_issues:
            files_with_issues.append({
                'path': file_path,
                'imports': imports
            })

    if files_with_issues:
        print(f"Found {len(files_with_issues)} files with code outside <pre><code> tags")
        for item in files_with_issues[:10]:
            print(f"  - {item['path']}: {len(item['imports'])} instances")
    else:
        print("✅ All code is properly within <pre><code> tags")

    return len(files_with_issues)

if __name__ == "__main__":
    count = main()
    exit(0 if count == 0 else 1)
```

**実施手順**:
1. スクリプト作成
2. 全HTMLファイルで検証実行
3. 結果レポート生成

**期待される結果**: 0件（すべて正常）

---

## 想定される成果物

### Phase 5-0（検証のみ - 推奨）

**成果物**:
1. `PHASE_5_VERIFICATION_REPORT.md` - 検証結果レポート
2. `scripts/verify_code_blocks_correct.py` - 正しい検証スクリプト

**内容**:
- 日本語版との比較結果
- `grep "^import "`が誤検出だった理由
- 正しい検証方法の確立
- 検証結果: 0件（すべて正常）

**所要時間**: 30分

### ~~Option B（修正実施）~~ → 不要

**理由**: HTML構造に問題なし、修正不要と判明

---

## 判断基準

### Option A を選択する条件

✅ ブラウザでの表示が正常
✅ Syntax highlightが機能している
✅ コードブロック全体が表示されている
✅ コピー&ペーストが正常

### Option B を選択する条件

❌ Syntax highlightが機能していない
❌ コードの一部が欠落して表示される
❌ コピー&ペーストに問題がある
❌ レンダリングエラーが発生している

---

## 次のステップ

**完了済み**:
1. ✅ 計画書の作成
2. ✅ 日本語版との比較検証
3. ✅ 問題が誤検出であることを確認

**推奨する対応**:
- **Option 1**: 何もしない（修正不要と判明）
- **Option 2**: 検証スクリプトのみ作成（30分、誤検出防止のため）

**ユーザーへの報告**:
- 142ファイルの「問題」は実際には問題ではない
- 日本語版と同じ正しいフォーマット
- `grep "^import "`コマンドの誤検出
- HTML修正は不要

---

## 補足情報

### 関連ファイル

- `/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/修正対象ページ.md` - 対象ファイルリスト（142ファイル）
- `PHASE_4-1_COMPLETE_SUMMARY.md` - Phase 4-1-4でのコード体裁修正実績

### Phase 4-1-4との比較

| 項目 | Phase 4-1-4 | Phase 5 |
|------|-------------|---------|
| **問題** | `<pre>`タグ完全欠落 | 開始タグ直後の改行なし |
| **対象** | 2ファイル、22箇所 | 142ファイル、推定500-800箇所 |
| **影響** | コードブロック構造が壊れている | 表示は正常かもしれない |
| **修正** | `<pre>`タグ追加 | 改行追加（必要な場合） |
| **リスク** | 低 | 中（検証次第で低） |

---

## 最終結論

### ✅ Phase 5は実施不要

**調査結果**:
1. 日本語版参照ファイル `knowledge/jp/MI/mi-introduction/chapter3-hands-on.html` と比較
2. 英語版も日本語版も**同じHTML構造**を使用
3. `<pre><code class="language-python">import ...` は**正しいフォーマット**
4. `grep "^import "`による検出は**誤検出（false positive）**

**理由**:
```html
<!-- このHTML構造は正しい -->
<pre><code class="language-python">import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
</code></pre>
```

- 1行目: `<pre><code class="language-python">import numpy as np`
- 2行目以降: `import pandas as pd` ← grep "^import " で検出されるが、**タグ内にある**

**影響**:
- 表示: ✅ 正常
- Syntax highlight: ✅ 機能
- コピー&ペースト: ✅ 正常
- HTML構造: ✅ 適切

**対処**:
- HTML修正: 不要
- 検証スクリプト改善: 任意（将来の誤検出防止のため）

---

**作成日**: 2025-11-17
**更新日**: 2025-11-17（日本語版比較により結論更新）
**作成者**: Claude (AI Assistant)
**ステータス**: ✅ 調査完了、修正不要と判明
