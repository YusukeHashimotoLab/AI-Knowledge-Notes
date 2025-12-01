# AI Terakoya Knowledge Base - 修正案 v2

**作成日**: 2025-12-01
**更新日**: 2025-12-01
**作成者**: Claude Code (Opus 4.5)

---

## 1. 概要

本ドキュメントは、AI Terakoya Knowledge Baseの最新評価結果に基づく修正案です。
前回の修正案（v1）からの進捗を反映しています。

### 現状サマリー

| 指標 | EN版 | JP版 | 合計 |
|------|------|------|------|
| HTMLファイル数 | 573 | 622 | 1,195 |
| シリーズ数 | 77 | 85 | 162 |
| 壊れたリンク (HTML) | 0 | 0 | **0** |
| 壊れたリンク (MD誤検出) | 2 | 102 | 104 |
| Missing Anchors (MD) | 0 | 8 | 8 |

**総合スコア**: 95/100（優秀） ↑1pt

---

## 2. 前回からの進捗

### 完了した項目

| 項目 | ステータス | 備考 |
|------|----------|------|
| HTMLリンク修正 | ✅ 完了 | EN/JP共に100%修正 |
| LLM基礎入門シリーズ | ✅ 完了 | index.html + chapter-1.html |
| PyTorch基礎入門シリーズ | ✅ 完了 | index.html + chapter-1.html |
| プロンプトエンジニアリング入門 | ✅ 完了 | index.html + chapter-1.html |
| PyTorch Geometric入門 | ✅ 完了 | index.html + chapter-1.html |
| CI/CDワークフロー | ✅ 完了 | link-check.yml, html-validate.yml |
| ML/index.htmlリンク追加 | ✅ 完了 | 4シリーズ追加 |

### 新規作成シリーズ一覧

```
knowledge/jp/ML/
├── llm-basics-introduction/          # LLM基礎入門
│   ├── index.html (27.5KB)
│   └── chapter-1.html (37.6KB)
├── pytorch-basics-introduction/      # PyTorch基礎入門
│   ├── index.html (25.5KB)
│   └── chapter-1.html (35.1KB)
├── prompt-engineering-introduction/  # プロンプトエンジニアリング入門
│   ├── index.html (26.9KB)
│   └── chapter-1.html (38.2KB)
└── pytorch-geometric-introduction/   # PyTorch Geometric入門
    ├── index.html (23.4KB)
    └── chapter-1.html (38.1KB)
```

---

## 3. 残存課題

### 3.1 優先度: 高

#### 該当なし
HTMLファイル内の壊れたリンクは全て修正済みです。

---

### 3.2 優先度: 中

#### A. Missing Anchors修正（JP版 8件）

**問題**: Markdownファイル内のアンカーリンクが対応するIDを持たない

| ファイル | アンカー | 状態 |
|----------|----------|------|
| MI/bayesian-optimization-introduction/index.md | #learning-path | 未修正 |
| MI/chemoinformatics-introduction/index.md | #learning-path | 未修正 |
| MI/experimental-data-analysis-introduction/index.md | #learning-path | 未修正 |
| MI/gnn-introduction/index.md | #learning-path | 未修正 |
| MI/index.md | #application-guide | 未修正 |
| MI/index.md | #license-terms | 未修正 |
| MI/materials-databases-introduction/index.md | #learning-path | 未修正 |
| MI/mi-introduction/index.md | #data-license | 未修正 |

**影響**: HTMLファイルには影響なし（MDソースファイルのみ）

**修正方法**:
```markdown
## 学習の進め方 {#learning-path}
```

---

#### B. 追加コンテンツ作成（推奨）

**ML Dojoで追加推奨のシリーズ**:

| シリーズ | 優先度 | 理由 |
|---------|--------|------|
| ディープラーニング基礎 | 高 | CNN/RNN/Transformerの前提知識 |
| 機械学習のための統計 | 高 | 基礎知識として重要 |
| scikit-learn入門 | 中 | 実践的なML入門として |
| TensorFlow入門 | 中 | PyTorchと並ぶ主要フレームワーク |

**MI Dojoで追加推奨のシリーズ**:

| シリーズ | 優先度 | 理由 |
|---------|--------|------|
| 材料データベース活用 | 高 | MI実践の基盤 |
| ベイズ最適化実践 | 高 | 材料探索の主要手法 |
| 実験データ解析 | 中 | 実験科学者向け |

---

#### C. EN版コンテンツ同期

**問題**: JP版で新規作成したシリーズがEN版に存在しない

| シリーズ | JP版 | EN版 |
|---------|------|------|
| LLM基礎入門 | ✅ | ❌ |
| PyTorch基礎入門 | ✅ | ❌ |
| プロンプトエンジニアリング入門 | ✅ | ❌ |
| PyTorch Geometric入門 | ✅ | ❌ |

**推奨**: JP版コンテンツの英訳版を作成

---

### 3.3 優先度: 低

#### D. 画像alt属性の追加

**問題**: 一部の画像でalt属性が空または不足

**検出コマンド**:
```bash
grep -r 'alt=""' knowledge/ --include="*.html"
```

---

#### E. SEOメタデータ最適化

**推奨追加メタタグ**:
```html
<meta name="description" content="[ページ固有の説明文]">
<meta property="og:title" content="[ページタイトル]">
<meta property="og:description" content="[説明文]">
<meta property="og:type" content="article">
```

---

#### F. リンクチェッカー改善

**問題**: 化学式（SMILES表記）がリンクとして誤検出される（102件）

**推奨**: check_links.pyに除外パターンを追加

```python
EXCLUDE_PATTERNS = [
    r'^[A-Z]$',           # 単一大文字（C, O, N等）
    r'^=[A-Z]',           # =O, =N等
    r'^\[.*\]$',          # [OH], [C@@H]等
    r'^NC\(=O',           # アミド結合
]
```

---

## 4. 推奨アクション（優先度順）

### Phase 1: 即時対応

| タスク | 工数 | 効果 |
|--------|------|------|
| Missing Anchors修正（MD） | 30分 | 品質向上 |

### Phase 2: 短期対応

| タスク | 工数 | 効果 |
|--------|------|------|
| ディープラーニング基礎シリーズ作成 | 2-3時間 | コンテンツ充実 |
| 機械学習のための統計シリーズ作成 | 2-3時間 | 基礎強化 |
| EN版新シリーズ翻訳（4シリーズ） | 4-6時間 | バイリンガル対応 |

### Phase 3: 中期対応

| タスク | 工数 | 効果 |
|--------|------|------|
| 材料データベース活用シリーズ作成 | 2-3時間 | MI充実 |
| ベイズ最適化実践シリーズ作成 | 2-3時間 | MI充実 |
| 画像alt属性追加 | 1-2時間 | アクセシビリティ |
| SEOメタデータ最適化 | 2-3時間 | 検索性向上 |

### Phase 4: 継続的改善

| タスク | 頻度 | 目的 |
|--------|------|------|
| リンクチェック実行 | PR毎 | 品質維持 |
| コンテンツ同期確認 | 月次 | EN/JP整合性 |
| ユーザーフィードバック確認 | 月次 | 改善点発見 |

---

## 5. 品質維持チェックリスト

### 新規コンテンツ作成時

- [ ] 全ての内部リンクが有効か確認
- [ ] 画像にalt属性があるか確認
- [ ] meta descriptionが設定されているか確認
- [ ] 難易度・読了時間が明記されているか確認
- [ ] 前提条件・次ステップが記載されているか確認
- [ ] コード例が100%動作するか確認
- [ ] EN/JP両版の同期を検討

### 定期レビュー（月次）

- [ ] CI/CDリンクチェック結果確認
- [ ] 新規「準備中」シリーズの進捗確認
- [ ] アクセス解析・ユーザーフィードバック確認

---

## 6. 技術スタック確認

### 現在のCI/CD構成

```
.github/workflows/
├── link-check.yml      # リンク検証（EN/JP並列実行）
├── html-validate.yml   # HTML5準拠検証
└── README.md           # ワークフロー説明
```

### 利用ツール

| ツール | 用途 | 状態 |
|--------|------|------|
| check_links.py | リンク検証 | ✅ 稼働中 |
| convert_md_to_html_en.py | MD→HTML変換 | ✅ 利用可能 |
| sync_md_html.py | 双方向同期 | ✅ 利用可能 |
| markdownlint | MDリント | ✅ 設定済み |
| html-validate | HTML検証 | ✅ 設定済み |

---

## 7. 結論

AI Terakoyaは**優秀な状態（95/100）**にあります。

### 達成事項

1. **HTMLリンク**: 100%修正完了（EN/JP共に壊れたリンク0件）
2. **新規シリーズ**: 4シリーズ作成（LLM基礎、PyTorch基礎、プロンプトエンジニアリング、PyTorch Geometric）
3. **CI/CD**: 自動リンクチェック・HTML検証パイプライン構築

### 今後の注力ポイント

1. **コンテンツ拡充**: ディープラーニング基礎、統計基礎などの追加
2. **EN版同期**: 新規JP版シリーズの英訳
3. **アクセシビリティ**: 画像alt属性の追加

---

*このドキュメントはClaude Code (Opus 4.5) によって自動生成されました。*
*作成日: 2025-12-01*
