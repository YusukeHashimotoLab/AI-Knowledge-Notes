# 英語版記事 ガイドライン準拠化計画

**作成日**: 2025-11-17
**ステータス**: 計画段階
**対象**: knowledge/en/ 配下の全HTMLファイル（566ファイル）
**目標**: 執筆ガイドライン準拠度 72% → 95%以上

---

## Executive Summary

英語版記事と執筆ガイドライン（`執筆ガイドライン.md` v2.0）との準拠度分析により、**72/100点**の現状が判明しました。主な問題は：

1. **🔴 重大**: 参考文献欠落（PI）、バージョン情報なし、演習形式不統一
2. **🟡 主要**: Docstring欠落、難易度ラベル不統一、chapter-description欠落
3. **🟢 軽微**: 見出し形式、免責事項、学術参考文献不足

本計画では、3つのPhaseで段階的に準拠度を向上させます。

---

## 現状分析サマリー

### 準拠度スコア

| カテゴリ | 現状 | 目標 | ギャップ |
|---------|------|------|---------|
| HTML構造 | 95/100 | 95/100 | 0 |
| コンテンツ品質 | 85/100 | 90/100 | +5 |
| コード例 | 75/100 | 95/100 | +20 |
| 演習問題 | 65/100 | 95/100 | +30 |
| 参考文献 | 60/100 | 90/100 | +30 |
| メタデータ | 70/100 | 90/100 | +20 |
| **総合** | **72/100** | **95/100** | **+23** |

### 対象ファイル統計

**調査サンプル**:
- ML/feature-engineering-introduction/chapter1-data-preprocessing.html (78点)
- PI/process-data-analysis/chapter-1.html (65点)
- MS/materials-science-introduction/chapter-5.html (70点)

**推定対象範囲**:
- 全566 HTMLファイル（knowledge/en/）
- 推定101シリーズ × 平均4-5章 = 約400-500章ファイル
- Index、その他ページ: 約60-160ファイル

---

## Phase 1: 重大な不備の修正（優先度🔴）

### 目標

**準拠度**: 72% → 85%
**所要時間**: 2-3日
**対象**: 最重要3項目

---

### Task 1-1: PI章への参考文献セクション追加

#### 対象ファイル
- `knowledge/en/PI/process-data-analysis/chapter-1.html`
- 他のPIシリーズで参考文献が欠落しているファイル（要調査）

#### 問題
- 参考文献セクションが完全欠落
- ガイドライン違反: セクション7.1、3.2（line 199）

#### 修正内容

**追加する参考文献セクション**:
```html
<h2>References</h2>
<ol>
  <li>Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
      <em>Time Series Analysis: Forecasting and Control</em> (5th ed.).
      Wiley. ISBN: 978-1118675021</li>

  <li>Hyndman, R. J., & Athanasopoulos, G. (2021).
      <em>Forecasting: Principles and Practice</em> (3rd ed.).
      OTexts.
      URL: <a href="https://otexts.com/fpp3/">https://otexts.com/fpp3/</a>
      (Accessed: 2025-11-17)</li>

  <li>Seabold, S., & Perktold, J. (2010).
      "statsmodels: Econometric and statistical modeling with python."
      <em>Proceedings of the 9th Python in Science Conference</em>, 57-61.
      DOI: <a href="https://doi.org/10.25080/Majora-92bf1922-011">10.25080/Majora-92bf1922-011</a></li>

  <li>Truong, C., Oudre, L., & Vayatis, N. (2020).
      "Selective review of offline change point detection methods."
      <em>Signal Processing</em>, 167, 107299.
      DOI: <a href="https://doi.org/10.1016/j.sigpro.2019.107299">10.1016/j.sigpro.2019.107299</a></li>

  <li>pandas documentation. (2024). "Time Series / Date functionality."
      URL: <a href="https://pandas.pydata.org/docs/user_guide/timeseries.html">https://pandas.pydata.org/docs/user_guide/timeseries.html</a>
      (Accessed: 2025-11-17)</li>
</ol>
```

#### 実施手順
1. PIシリーズ全体で参考文献の有無を調査（`grep -r "<h2>References</h2>" knowledge/en/PI/`）
2. 欠落ファイルリストを作成
3. 各章の内容に応じた参考文献を選定
4. HTMLファイルに追加（ナビゲーションの前）
5. リンク有効性確認

#### 成果物
- `scripts/add_references_pi.py` - 自動追加スクリプト
- 修正ファイルリスト（推定3-10ファイル）

---

### Task 1-2: 全コード例へのバージョン情報追加

#### 対象ファイル
- 全566 HTMLファイル（コード例を含むファイル）
- 推定400-500ファイル

#### 問題
- すべてのコード例でライブラリバージョン未記載
- ガイドライン違反: セクション4.1（line 226）

#### 修正内容

**修正前**:
```html
<pre><code class="language-python">import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
```

**修正後**:
```html
<pre><code class="language-python"># Requirements:
# - Python 3.9+
# - numpy>=1.24.0, <2.0.0
# - pandas>=2.0.0, <2.2.0
# - matplotlib>=3.7.0
# - scikit-learn>=1.3.0, <1.5.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
```

#### 実施手順

**Option A: スクリプトによる一括追加**
1. パターン分析: 各Dojoで使用される主要ライブラリを特定
2. バージョンマッピング作成:
   ```python
   VERSION_MAP = {
       'ML': {
           'numpy': '>=1.24.0, <2.0.0',
           'pandas': '>=2.0.0, <2.2.0',
           'scikit-learn': '>=1.3.0, <1.5.0',
           'torch': '>=2.0.0',
           'tensorflow': '>=2.13.0'
       },
       'FM': {
           'numpy': '>=1.24.0',
           'scipy': '>=1.11.0',
           'sympy': '>=1.12'
       },
       # ... 他のDojo
   }
   ```
3. スクリプト作成: `scripts/add_version_info.py`
4. テスト実行（3ファイル）
5. 全体実行
6. 手動検証（ランダム10ファイル）

**Option B: Dojo別に段階実行**
1. ML Dojo（96ファイル）
2. FM Dojo（24ファイル）
3. MS/MI/PI Dojo（26ファイル）

#### 注意点
- コードブロックの最初の行に挿入（import文の前）
- 既存のコメントを上書きしない
- `<pre><code class="language-python">` の直後に改行追加

#### 成果物
- `scripts/add_version_info.py` - バージョン情報追加スクリプト
- `VERSION_REQUIREMENTS.md` - Dojo別バージョン要件一覧
- バックアップファイル（`.bak`）

---

### Task 1-3: PI演習問題のフォーマット修正

#### 対象ファイル
- `knowledge/en/PI/process-data-analysis/chapter-1.html`
- 他のPIシリーズで `<div class="callout">` 形式の演習問題を使用しているファイル

#### 問題
- `<details>` タグではなく `<div class="callout">` を使用
- ガイドライン違反: セクション6.1（line 406）

#### 修正内容

**修正前**:
```html
<div class="callout callout-tip">
<h4>💡 Hint</h4>
<p>For Exercise 3, try different cost functions ("rbf", "normal", "l1", "l2")...</p>
</div>
```

**修正後**:
```html
<h3>Exercise 3 (Difficulty: Hard) ⏱️ 30-60 minutes</h3>
<p>Apply the code from Example 8 to the provided `chemical_process_full.csv` dataset...</p>

<details>
<summary>💡 Hint</summary>
<p>For Exercise 3, try different cost functions ("rbf", "normal", "l1", "l2")...</p>
<ul>
  <li>Try <code>model="l1"</code> for sparse change points</li>
  <li>Use <code>pen=10</code> as a starting point and adjust</li>
</ul>
</details>

<details>
<summary>✅ Sample Answer</summary>
<pre><code class="language-python"># Complete solution code
import ruptures as rpt

# Load data
df = pd.read_csv('chemical_process_full.csv')
signal = df['temperature'].values

# Apply PELT algorithm with L1 cost
model = "l1"
algo = rpt.Pelt(model=model).fit(signal)
result = algo.predict(pen=10)

print(f"Detected change points: {result}")
# Visualization code...
</code></pre>
<p><strong>Explanation</strong>: The L1 cost function is effective for detecting abrupt changes...</p>
</details>
```

#### 実施手順
1. PIシリーズ全体で `<div class="callout">` パターンを検索
2. 各演習問題に対応する解答コードを作成
3. `<details>` 形式に変換
4. 難易度（Easy/Medium/Hard）と時間見積もりを追加
5. ブラウザでの表示確認

#### 成果物
- `scripts/convert_exercises_pi.py` - 演習形式変換スクリプト
- 修正ファイルリスト

---

### Phase 1 完了基準

- [ ] PI章に参考文献セクション追加（5-8件以上）
- [ ] 全コード例にバージョン情報追加（Python 3.9+、主要ライブラリ）
- [ ] PI演習問題を `<details>` 形式に変換
- [ ] リンクチェック実行（broken links確認）
- [ ] ブラウザでの表示確認（3 Dojo × 各1ファイル）
- [ ] Git commit（"feat: Phase 1 - Fix critical guideline violations"）

**期待される準拠度向上**: 72% → 85%

---

## Phase 2: 主要な不備の修正（優先度🟡）

### 目標

**準拠度**: 85% → 92%
**所要時間**: 3-4日
**対象**: 主要3項目

---

### Task 2-1: コード例へのDocstring追加

#### 対象ファイル
- 全コード例（推定1,500-2,000ブロック）

#### 問題
- コード例にメタデータdocstringが欠落
- ガイドライン違反: セクション4.1（lines 230-236）

#### 修正内容

**修正前**:
```html
<pre><code class="language-python"># Requirements:
# - numpy>=1.24.0
# - pandas>=2.0.0

import numpy as np
import pandas as pd
```

**修正後**:
```html
<pre><code class="language-python">"""
Example: Effects of Preprocessing on Model Performance

Purpose: Demonstrate how data scaling improves classification accuracy
Target: Beginner to Intermediate
Execution time: ~10 seconds
Dependencies: numpy, pandas, scikit-learn
"""

# Requirements:
# - numpy>=1.24.0, <2.0.0
# - pandas>=2.0.0, <2.2.0
# - scikit-learn>=1.3.0

import numpy as np
import pandas as pd
```

#### 実施手順

**Option A: 半自動化アプローチ**
1. コード例のタイトルを抽出（`<h3>` タグから）
2. Docstringテンプレート生成
3. 手動でPurpose、Target、Execution timeを記入
4. スクリプトで一括挿入

**Option B: AI支援アプローチ**
1. Claude/GPT-4でコード内容を分析
2. 適切なdocstringを自動生成
3. 人間がレビュー・承認
4. 一括適用

#### テンプレート
```python
"""
Example: {title}

Purpose: {1-2 sentence description of what this code does and why}
Target: {Beginner | Intermediate | Advanced}
Execution time: ~{N} seconds
Dependencies: {comma-separated list}
"""
```

#### 成果物
- `scripts/add_docstrings.py` - Docstring追加スクリプト
- `DOCSTRING_TEMPLATE.md` - テンプレートガイド

---

### Task 2-2: 演習問題の難易度ラベル統一

#### 対象ファイル
- 全演習問題を含むファイル（推定300-400ファイル）

#### 問題
- 難易度ラベルが不統一
  - ML: "Easy/Medium/Hard" ✅
  - PI: "Basic/Intermediate/Advanced" ❌
- ガイドライン違反: セクション6.1

#### 修正内容

**検索パターン**:
```regex
(Difficulty|難易度):\s*(Basic|Intermediate|Advanced|初級|中級|上級)
```

**置換**:
- Basic → Easy
- Intermediate → Medium
- Advanced → Hard

**時間見積もり追加**:
```html
<!-- 修正前 -->
<h3>Exercise 1 (Difficulty: Basic)</h3>

<!-- 修正後 -->
<h3>Exercise 1 (Difficulty: Easy) ⏱️ 5-10 minutes</h3>
```

#### 時間見積もり基準（ガイドライン6.1）
- Easy: 5-10分
- Medium: 15-30分
- Hard: 30-60分

#### 実施手順
1. 難易度ラベルパターン検索
2. マッピング作成（Basic→Easy等）
3. 一括置換スクリプト実行
4. 時間見積もり追加
5. 検証

#### 成果物
- `scripts/standardize_difficulty.py` - 難易度統一スクリプト

---

### Task 2-3: chapter-description要素の追加

#### 対象ファイル
- 全章ファイル（推定400-500ファイル）

#### 問題
- `<p class="chapter-description">` 要素が欠落
- ガイドライン違反: セクション3.2（line 176）

#### 修正内容

**挿入位置**: Learning Objectivesセクションの直後

**テンプレート**:
```html
<h2>Learning Objectives</h2>
<ul>
  <li>✅ ...</li>
  <li>✅ ...</li>
</ul>

<hr/>

<p class="chapter-description">
This chapter covers {main topic}. Through {N} practical {language} examples,
you will learn {skill 1}, {skill 2}, and {skill 3} applicable to {application area}.
</p>

<h2>1.1 First Section</h2>
```

#### 実施手順

**Option A: 半自動生成**
1. 各章のh1タイトルとLearning Objectivesを抽出
2. テンプレートに基づいて文章生成
3. 人間がレビュー・編集
4. HTMLに挿入

**Option B: AIによる自動生成**
1. 章全体の内容をAIで分析
2. 2-3文の説明文を生成
3. 人間がレビュー
4. 承認後に一括挿入

#### サンプル
```html
<!-- ML/feature-engineering -->
<p class="chapter-description">
This chapter covers the fundamental techniques of data preprocessing,
the critical first step in any machine learning pipeline. Through 12
practical Python examples, you will learn handling missing values,
outlier detection, and feature scaling applicable to real-world datasets.
</p>

<!-- PI/process-data-analysis -->
<p class="chapter-description">
This chapter introduces time series analysis methods essential for
process monitoring and optimization. Through 10 Python implementations,
you will master data preparation, statistical testing, forecasting, and
anomaly detection applicable to chemical manufacturing processes.
</p>
```

#### 成果物
- `scripts/add_chapter_description.py` - Description追加スクリプト
- `CHAPTER_DESCRIPTIONS.csv` - 各章のdescription一覧

---

### Phase 2 完了基準

- [ ] 全コード例にdocstring追加
- [ ] 演習問題の難易度ラベル統一（Easy/Medium/Hard）
- [ ] 演習問題に時間見積もり追加
- [ ] chapter-description要素追加（全章）
- [ ] ランダムサンプル検証（各Dojo 5ファイル）
- [ ] Git commit（"feat: Phase 2 - Improve code examples and exercise format"）

**期待される準拠度向上**: 85% → 92%

---

## Phase 3: 軽微な不備の修正とQA（優先度🟢）

### 目標

**準拠度**: 92% → 95%+
**所要時間**: 2-3日
**対象**: 仕上げと品質保証

---

### Task 3-1: 見出し・ラベルの統一

#### 対象
- Learning Objectives見出し（"What You Will Learn" → "Learning Objectives"）
- 演習問題セクション見出し
- その他のセクション見出し

#### 実施手順
1. 非標準見出しを検索
2. 標準形式に置換
3. 検証

---

### Task 3-2: 免責事項セクションの追加

#### 対象ファイル
- MS Dojoで免責事項が欠落しているファイル

#### 修正内容
```html
<section class="disclaimer">
  <h2>⚠️ Disclaimer</h2>
  <p>This educational content is provided "AS IS" without warranties...</p>
</section>
```

---

### Task 3-3: 学術参考文献の拡充

#### 目標
- 各章に3-5本の学術論文/専門書を追加
- DOIリンク付き
- 最新の研究成果を含む

#### 実施手順
1. 各章のトピックに関連する論文を検索（Google Scholar、arXiv）
2. 適切な引用形式で追加
3. アクセス日付を記録

---

### Task 3-4: 品質保証（QA）

#### チェック項目

**自動チェック**:
- [ ] リンク切れチェック（`check_links.py`）
- [ ] HTMLバリデーション
- [ ] コードブロック構文チェック
- [ ] 画像リンク有効性

**手動チェック**:
- [ ] ブラウザでの表示確認（各Dojo 3ファイル）
- [ ] Syntax highlightの動作
- [ ] Mermaid図の表示
- [ ] LaTeX数式の表示
- [ ] `<details>` タグの動作
- [ ] ナビゲーションリンク
- [ ] レスポンシブデザイン（Mobile/Tablet/Desktop）

**準拠度チェック**:
- [ ] ガイドライン準拠度再評価（サンプル20ファイル）
- [ ] チェックリスト完全性確認（セクション8）
- [ ] 目標95%達成確認

---

### Task 3-5: ドキュメント更新

#### 作成・更新するドキュメント

1. **`ENGLISH_ARTICLE_COMPLIANCE_REPORT.md`** - 最終準拠度レポート
   - Phase 1-3の実施内容
   - 修正統計（ファイル数、コード例数、演習問題数）
   - Before/After比較
   - 残存課題

2. **`GUIDELINE_COMPLIANCE_CHECKLIST.md`** - 準拠度チェックリスト
   - 各章で確認すべき項目
   - 新規記事執筆時の参照用

3. **執筆ガイドライン更新**
   - 英語版特有の注意事項追加
   - コード例のベストプラクティス更新

---

### Phase 3 完了基準

- [ ] 見出し・ラベル統一完了
- [ ] 免責事項追加完了
- [ ] 学術参考文献拡充（平均3本以上/章）
- [ ] QAチェック全項目クリア
- [ ] ドキュメント更新完了
- [ ] 準拠度95%以上達成
- [ ] Git commit（"feat: Phase 3 - Final compliance improvements and QA"）

**期待される準拠度向上**: 92% → 95%+

---

## スクリプト一覧

### Phase 1用スクリプト

1. **`scripts/add_references_pi.py`**
   - 機能: PI章に参考文献セクション追加
   - 入力: 対象ファイルリスト
   - 出力: 修正済みHTML

2. **`scripts/add_version_info.py`**
   - 機能: コード例にバージョン情報追加
   - 入力: Dojo名、ファイルパス
   - 出力: 修正済みHTML
   - オプション: `--dojo`, `--dry-run`, `--backup`

3. **`scripts/convert_exercises_pi.py`**
   - 機能: 演習問題を `<details>` 形式に変換
   - 入力: PIファイル
   - 出力: 修正済みHTML

### Phase 2用スクリプト

4. **`scripts/add_docstrings.py`**
   - 機能: コード例にdocstring追加
   - 入力: HTMLファイル、docstringマッピング
   - 出力: 修正済みHTML

5. **`scripts/standardize_difficulty.py`**
   - 機能: 演習難易度ラベル統一
   - 入力: 全HTMLファイル
   - 出力: 修正済みHTML

6. **`scripts/add_chapter_description.py`**
   - 機能: chapter-description要素追加
   - 入力: HTMLファイル、description CSV
   - 出力: 修正済みHTML

### Phase 3用スクリプト

7. **`scripts/verify_compliance.py`**
   - 機能: ガイドライン準拠度検証
   - 入力: HTMLファイル、ガイドライン
   - 出力: 準拠度スコアレポート

8. **`scripts/qa_check_all.py`**
   - 機能: 総合QAチェック
   - 機能: リンク、HTML、コード、表示の検証
   - 出力: QAレポート

---

## タイムライン

### Week 1: Phase 1（重大な不備）
- **Day 1-2**: Task 1-1（参考文献追加）+ Task 1-2準備
- **Day 3-4**: Task 1-2（バージョン情報追加、テスト→全体）
- **Day 5**: Task 1-3（演習形式変換）+ 検証
- **Day 6**: Phase 1 QA + Git commit

### Week 2: Phase 2（主要な不備）
- **Day 7-8**: Task 2-1（docstring追加、サンプル→全体）
- **Day 9**: Task 2-2（難易度統一）
- **Day 10-11**: Task 2-3（chapter-description追加）
- **Day 12**: Phase 2 QA + Git commit

### Week 3: Phase 3（仕上げ）
- **Day 13**: Task 3-1 + 3-2（見出し統一、免責事項）
- **Day 14**: Task 3-3（学術参考文献拡充）
- **Day 15-16**: Task 3-4（総合QA）
- **Day 17**: Task 3-5（ドキュメント更新）+ 最終commit

**総所要時間**: 17日（約2.5週間）

---

## リスク管理

### 主要リスク

| リスク | 影響 | 確率 | 軽減策 |
|--------|------|------|--------|
| スクリプトによる一括修正でHTML破損 | 高 | 中 | バックアップ作成、段階実行、検証 |
| バージョン情報の誤り | 中 | 中 | 実環境でのテスト、専門家レビュー |
| 作業量過多（566ファイル） | 高 | 高 | 優先度付け、自動化、並行作業 |
| Git競合 | 中 | 低 | ブランチ戦略、定期commit |
| QA不足による品質低下 | 高 | 中 | サンプル検証、自動テスト |

### 軽減戦略

1. **バックアップ**: 各Phase前に全ファイルバックアップ（`.phase-N.bak`）
2. **段階実行**: サンプル3-5ファイルでテスト → 検証 → 全体適用
3. **Git戦略**: Feature branch（`feature/guideline-compliance`）作成
4. **並行作業**: Dojo別に並行実行可能（ML/FM/MS/MI/PI独立）
5. **ロールバック**: 問題発生時は即座に `git reset`

---

## 成功基準

### 定量的基準
- [ ] 準拠度スコア95%以上達成
- [ ] 全コード例にバージョン情報（100%）
- [ ] 全コード例にdocstring（100%）
- [ ] 全演習問題が `<details>` 形式（100%）
- [ ] 参考文献セクション存在率100%
- [ ] リンク切れ0件（Phase 4で320件→0件）

### 定性的基準
- [ ] ガイドライン準拠度レポート完成
- [ ] 新規記事執筆用チェックリスト完成
- [ ] 既存記事の品質が一貫して高い
- [ ] ブラウザ表示が正常（全Dojo確認）
- [ ] ユーザーフィードバック収集準備完了

---

## 次のステップ

### 承認後の実施順序

1. **Feature branch作成**: `git checkout -b feature/guideline-compliance`
2. **Phase 1-1開始**: PI参考文献追加（最小影響、すぐ効果）
3. **Phase 1-2準備**: バージョンマッピング作成、スクリプト開発
4. **並行作業**: 複数Dojoで同時進行可能

### 承認前の確認事項

- [ ] この計画の実施に承認いただけますか？
- [ ] タイムライン（2.5週間）は許容範囲ですか？
- [ ] 特定のDojoから優先的に開始すべきですか？
- [ ] 自動化スクリプトの開発を先行すべきですか？
- [ ] サンプル修正（3ファイル）を先に見たいですか？

---

**作成日**: 2025-11-17
**作成者**: Claude (AI Assistant)
**ステータス**: 計画完成、承認待ち
**推定工数**: 17日（約120-140時間）
