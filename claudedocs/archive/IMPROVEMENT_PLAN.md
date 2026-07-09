# AI Terakoya Knowledge Base - 修正案

**作成日**: 2025-12-01
**作成者**: Claude Code (Opus 4.5)

---

## 1. 概要

本ドキュメントは、AI Terakoya Knowledge Baseの評価結果に基づく具体的な修正案を提示します。

### 現状サマリー
- **総合スコア**: 94/100（優秀）
- **HTMLリンク**: 100%修正完了
- **残存問題**: 軽微な問題のみ

---

## 2. 修正項目一覧

### 2.1 優先度: 高

#### 項目なし
HTMLファイル内のリンク問題は全て修正済みです。

---

### 2.2 優先度: 中

#### A. Missing Anchors修正（JP版 8件）

**問題**: Markdownファイル内のアンカーリンクが対応するIDを持たない

| ファイル | 行 | アンカー | 対応方法 |
|----------|-----|----------|----------|
| MI/bayesian-optimization-introduction/index.md | 986 | #学習の進め方 | アンカーID追加 |
| MI/chemoinformatics-introduction/index.md | 799 | #学習の進め方 | アンカーID追加 |
| MI/experimental-data-analysis-introduction/index.md | 793 | #学習の進め方 | アンカーID追加 |
| MI/gnn-introduction/index.md | 1121 | #学習の進め方 | アンカーID追加 |
| MI/index.md | 4597 | #応用分野別ガイド | アンカーID追加 |
| MI/index.md | 4721 | #ライセンスと利用規約 | アンカーID追加 |
| MI/materials-databases-introduction/index.md | 776 | #学習の進め方 | アンカーID追加 |
| MI/mi-introduction/index.md | 626 | #データ利用とライセンス | アンカーID追加 |

**修正方法**:
```markdown
# 修正前
## 学習の進め方

# 修正後
## 学習の進め方 {#学習の進め方}
```

または、対応するHTMLに直接IDを追加:
```html
<h2 id="学習の進め方">学習の進め方</h2>
```

---

#### B. 「準備中」シリーズのコンテンツ作成

**問題**: 多数のシリーズが「準備中」として表示されている

**ML Dojo（優先度順）**:
1. llm-basics - LLM基礎
2. prompt-engineering - プロンプトエンジニアリング
3. deep-learning-fundamentals - ディープラーニング基礎
4. statistics-for-ml - 機械学習のための統計
5. pytorch-basics - PyTorch基礎

**MI Dojo**:
1. materials-databases - 材料データベース
2. bayesian-optimization - ベイズ最適化
3. experimental-data-analysis - 実験データ解析

**推奨アクション**:
- 各シリーズに最低でもindex.htmlとchapter-1.htmlを作成
- 段階的にコンテンツを追加

---

#### C. 画像alt属性の追加

**問題**: 一部の画像でalt属性が空または不足

**検出コマンド**:
```bash
grep -r 'alt=""' knowledge/
grep -r '<img[^>]*>' knowledge/ | grep -v 'alt='
```

**修正方法**:
```html
# 修正前
<img src="diagram.png">

# 修正後
<img src="diagram.png" alt="ニューラルネットワークの構造図">
```

---

### 2.3 優先度: 低

#### D. SEOメタデータ最適化

**問題**: 一部ページでmeta descriptionが不足または最適化されていない

**推奨追加メタタグ**:
```html
<meta name="description" content="[ページ固有の説明文 150-160文字]">
<meta name="keywords" content="マテリアルズインフォマティクス, 機械学習, 材料科学">
<meta property="og:title" content="[ページタイトル]">
<meta property="og:description" content="[説明文]">
<meta property="og:type" content="article">
```

---

#### E. CI/CDへのリンクチェック統合

**推奨**: GitHub Actionsワークフローの追加

```yaml
# .github/workflows/link-check.yml
name: Link Check

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  link-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install beautifulsoup4 lxml tqdm
      - run: python scripts/check_links.py --path knowledge/en
      - run: python scripts/check_links.py --path knowledge/jp
```

---

#### F. リンクチェッカー誤検出対策

**問題**: 化学式（SMILES表記）がリンクとして誤検出される

**推奨**: check_links.pyに除外パターンを追加

```python
# 除外パターン（化学式表記）
EXCLUDE_PATTERNS = [
    r'^[A-Z]$',           # 単一大文字（C, O, N等）
    r'^=[A-Z]',           # =O, =N等
    r'^\[.*\]$',          # [OH], [C@@H]等
    r'^NC\(=O',           # アミド結合
    r'^\w+\d*-\w+',       # C=C4等の化学式
]
```

---

## 3. 実装スケジュール案

### Phase 1: 即時対応（1日）
- [ ] Missing anchors修正（8件）

### Phase 2: 短期対応（1-2週間）
- [ ] 優先度の高い「準備中」シリーズ3つの作成
- [ ] 画像alt属性の追加

### Phase 3: 中期対応（1ヶ月）
- [ ] 残りの「準備中」シリーズの作成
- [ ] SEOメタデータ最適化
- [ ] CI/CDワークフロー追加

### Phase 4: 継続的改善
- [ ] 新規コンテンツ追加時のリンクチェック
- [ ] 定期的な品質レビュー

---

## 4. 修正スクリプト

### 4.1 Missing Anchors自動修正

```python
#!/usr/bin/env python3
"""Fix missing anchors in JP MD files"""
from pathlib import Path
import re

base_jp = Path("knowledge/jp")

anchor_fixes = [
    ("MI/bayesian-optimization-introduction/index.md", "学習の進め方"),
    ("MI/chemoinformatics-introduction/index.md", "学習の進め方"),
    ("MI/experimental-data-analysis-introduction/index.md", "学習の進め方"),
    ("MI/gnn-introduction/index.md", "学習の進め方"),
    ("MI/index.md", "応用分野別ガイド"),
    ("MI/index.md", "ライセンスと利用規約"),
    ("MI/materials-databases-introduction/index.md", "学習の進め方"),
    ("MI/mi-introduction/index.md", "データ利用とライセンス"),
]

for file_path, anchor in anchor_fixes:
    full_path = base_jp / file_path
    if full_path.exists():
        content = full_path.read_text(encoding='utf-8')
        # Add anchor ID to heading
        pattern = rf'^(##?\s*{re.escape(anchor)})\s*$'
        replacement = rf'\1 {{#{anchor}}}'
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if new_content != content:
            full_path.write_text(new_content, encoding='utf-8')
            print(f"Fixed: {file_path}")
```

### 4.2 画像alt属性チェック

```bash
#!/bin/bash
# find_missing_alt.sh
echo "=== 画像alt属性が不足しているファイル ==="
grep -rl '<img[^>]*>' knowledge/ | while read file; do
    if grep -q '<img[^>]*[^a]src=' "$file" || grep -q 'alt=""' "$file"; then
        echo "$file"
    fi
done
```

---

## 5. 品質維持のためのチェックリスト

### 新規コンテンツ作成時
- [ ] 全ての内部リンクが有効か確認
- [ ] 画像にalt属性があるか確認
- [ ] meta descriptionが設定されているか確認
- [ ] 難易度・読了時間が明記されているか確認
- [ ] 前提条件・次ステップが記載されているか確認
- [ ] EN/JP両版が同期しているか確認

### 定期レビュー（月次）
- [ ] リンクチェッカー実行
- [ ] 新規「準備中」シリーズの進捗確認
- [ ] ユーザーフィードバックの確認

---

## 6. 結論

AI Terakoyaは既に高品質な状態にあります（94/100）。本修正案の実施により、さらに完成度を高めることができます。

**最も効果的な改善**:
1. Missing anchors修正（即座に実行可能）
2. 「準備中」シリーズのコンテンツ作成（ユーザー価値向上）
3. CI/CD統合（品質維持の自動化）

---

*このドキュメントはClaude Code (Opus 4.5) によって自動生成されました。*
