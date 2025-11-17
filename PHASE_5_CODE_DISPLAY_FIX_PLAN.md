# Phase 5: コード表示修正計画

**作成日**: 2025-11-17
**ステータス**: 計画段階（実装前）
**対象**: 142ファイル

---

## 問題の概要

### 発見された問題

**`python_code_outside_pre.txt`** の検証結果に基づき、142個のHTMLファイルでPythonの`import`文がコードブロック（`<pre><code>`）の外に出力されていることが判明しました。

### 具体的な症状

**問題のあるパターン**:
```html
<h3>Example: Effects of Preprocessing</h3>
<pre><code class="language-python">import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
...
</code></pre>
```

**観察される問題**:
1. `<pre><code class="language-python">` タグの直後に改行がある
2. `import`文が**次の行**から始まっている
3. HTMLソースでは正しく見えるが、レンダリング時に問題が発生している可能性

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

### 1. Markdown → HTML変換時の問題

**Markdown元データの推定形式**:
````markdown
```python
import numpy as np
import pandas as pd
```
````

**想定される変換プロセス**:
1. Markdownパーサーが ````python` ブロックを検出
2. HTMLコードブロックに変換
3. **問題**: 開始タグと最初の行が同じ行になってしまう

### 2. なぜ`import`が問題になるか

**`python_code_outside_pre.txt`の検出ロジック**:
```bash
grep "^import " file.html
```
- **行頭が`import`で始まる行**を検出
- 正しくフォーマットされていれば、`import`は行頭に来ない（`<pre><code>`の後）
- 現状では`<pre><code class="language-python">import`が同じ行にあるため、**次の行の`import`**が行頭として検出される

### 3. 実際の問題の有無

**重要な調査ポイント**:
- ブラウザで表示した時に正しく表示されているか？
- コードのハイライトは機能しているか？
- コピー&ペーストは正常に動作するか？

**可能性のあるシナリオ**:

**シナリオA: 表示は正常（false positive）**
- HTMLは正しくレンダリングされている
- `grep "^import "`の検出方法が不適切だった
- **対処**: 修正不要、検証スクリプトの改善のみ

**シナリオB: 実際に表示問題がある**
- Syntax highlightが機能していない
- コードブロックの最初の行が欠落している
- **対処**: HTML構造の修正が必要

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

### Step-by-Step Plan

**Phase 5-0: 検証フェーズ**（必須）
1. 3つのサンプルファイルをブラウザで確認
2. 表示問題の有無を判定
3. 問題がなければOption A、あればOption Bへ

**Phase 5-1: 分析フェーズ**（Option B選択時）
1. `analyze_code_format_issues.py`作成
2. 全142ファイルの問題パターン分析
3. 分析レポート生成

**Phase 5-2: サンプル修正フェーズ**（Option B選択時）
1. `fix_code_block_newlines.py`作成
2. 3ファイルのサンプル修正
3. ブラウザでの表示確認
4. 問題がなければ次へ、あればスクリプト修正

**Phase 5-3: 全体修正フェーズ**（Option B選択時）
1. 全142ファイルの修正実行
2. リンクチェック実行
3. 修正前後の比較
4. コミット

**Phase 5-4: 完了報告**
1. `PHASE_5_COMPLETE_SUMMARY.md`作成
2. 修正統計の記録
3. 教訓の文書化

---

## 想定される成果物

### Option A（検証のみ）

**成果物**:
1. `PHASE_5_VERIFICATION_REPORT.md` - 検証結果レポート
2. `scripts/verify_code_blocks_improved.py` - 改善された検証スクリプト

### Option B（修正実施）

**成果物**:
1. `CODE_FORMAT_ANALYSIS.md` - 問題分析レポート
2. `scripts/analyze_code_format_issues.py` - 分析スクリプト
3. `scripts/fix_code_block_newlines.py` - 修正スクリプト
4. `PHASE_5_COMPLETE_SUMMARY.md` - 完了サマリー

**修正統計**（想定）:
- 対象ファイル: 142
- 修正箇所: 500-800箇所（推定）
- バックアップ: 142ファイル
- コミット: 1-2個

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

**今すぐ実施**:
1. ✅ この計画書の作成（完了）
2. ⏳ ユーザーの承認待ち

**ユーザー承認後**:
1. Phase 5-0（検証フェーズ）の実施
2. 検証結果に基づいてOption AまたはBを決定
3. 選択したオプションの実行

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

**作成日**: 2025-11-17
**作成者**: Claude (AI Assistant)
**ステータス**: 計画完成、ユーザー承認待ち
