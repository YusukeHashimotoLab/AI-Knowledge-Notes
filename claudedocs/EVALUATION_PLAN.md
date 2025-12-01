# AI Terakoya 包括的評価・改善プラン

## 現状分析サマリー

### サイト規模
- **総HTMLファイル数**: 1,187件
- **言語**: 日本語 (JP) / 英語 (EN)
- **Dojo数**: 5つ (FM, MI, ML, MS, PI)

### Dojo別ファイル数

| Dojo | EN files | EN series | JP files | JP series |
|------|----------|-----------|----------|-----------|
| FM   | 75       | 14        | 85       | 14        |
| MI   | 113      | 21        | 145      | 26        |
| ML   | 157      | 30        | 170      | 30        |
| MS   | 114      | 21        | 106      | 21        |
| PI   | 113      | 20        | 107      | 18        |

### 主要な問題

#### 1. リンク切れ問題 (928件)
- **AI-Knowledge-Notes絶対パス問題**: 227件
  - `/AI-Knowledge-Notes/knowledge/en/...` という誤った絶対パスがbreadcrumbに使用されている
- **存在しないシリーズへのリンク**: 約200件
  - `en/MI/gnn-introduction/` が存在しない（JPにはある）
  - `en/MI/reinforcement-learning-introduction/` が存在しない
  - `en/MI/transformer-introduction/` が存在しない
- **章ファイル名不整合**: 約300件
  - `chapter-4.html` vs `chapter4-xxx.html` の混在
  - 次章リンクが存在しないファイルを指している
- **日本語対応ファイル欠落**: 約50件
  - `jp/MI/mi-global-projects/` が存在しない
- **画像ファイル欠落**: 2件
  - `band_structure.png`, `dos.png`

#### 2. サイト構造の問題
- **breadcrumb**: 「AI Terakoya Top」が `../index.html`（Dojoトップ）を指しており、グローバルポータルではない
- **言語カバレッジの非対称性**: EN/JPで異なるシリーズが存在
- **欠落indexファイル**: `FM/inferential-bayesian-statistics/index.html`

#### 3. コンテンツ品質の懸念
- 翻訳状態の不統一（TRANSLATION_STATUS.mdが一部のDojoにのみ存在）
- コンテンツの完全性確認が必要

---

## 改善プラン

### Phase 1: リンク切れ修正 (優先度: 最高)

#### 1.1 AI-Knowledge-Notes絶対パス修正 (227件)
**問題**: breadcrumbに `/AI-Knowledge-Notes/knowledge/en/...` という誤った絶対パスが含まれている

**対策**:
- スクリプトで一括置換
- `/AI-Knowledge-Notes/knowledge/en/` → 相対パス `../../../` または適切な相対パス

**影響ファイル**: 主にMI, ML, MS, PIの全章ファイル

#### 1.2 存在しないENシリーズの作成 (約200件に影響)
**欠落シリーズ**:
- `en/MI/gnn-introduction/` (JPから翻訳)
- `en/MI/reinforcement-learning-introduction/` (JPから翻訳)
- `en/MI/transformer-introduction/` (JPから翻訳)

**対策**: JPの対応シリーズを英語に翻訳

#### 1.3 章ファイル名・リンク不整合修正 (約300件)
**問題**: 次章/前章リンクが存在しないファイル名を指している

**対策**:
- 実際のファイル名を確認し、リンクを修正
- または欠落している章ファイルを作成

#### 1.4 JP欠落シリーズの作成 (約50件に影響)
**欠落シリーズ**:
- `jp/MI/mi-global-projects/`

**対策**: ENから翻訳

#### 1.5 欠落画像ファイルの追加
- `en/MI/high-throughput-computing-introduction/band_structure.png`
- `en/MI/high-throughput-computing-introduction/dos.png`

---

### Phase 2: サイト構造の整合性確保 (優先度: 高)

#### 2.1 breadcrumb修正
- 全ファイルのbreadcrumb「AI Terakoya Top」を正しいパス（`../../index.html`）に修正

#### 2.2 欠落indexファイル作成
- `en/FM/inferential-bayesian-statistics/index.html` を作成

#### 2.3 言語カバレッジの整理
- EN/JP間で異なるシリーズを特定し、対応を決定
  - 翻訳して追加
  - または参照リンクを削除

---

### Phase 3: コンテンツ品質評価 (優先度: 中)

#### 3.1 TRANSLATION_STATUS.md統一
- 全Dojoに統一フォーマットのTRANSLATION_STATUS.mdを作成

#### 3.2 コンテンツ完全性チェック
- 各章のMathJax/Mermaid動作確認
- コード例の動作確認
- 日本語/英語の整合性確認

#### 3.3 品質レポート作成
- 各Dojoの品質スコアカード作成

---

## 実装順序

```
Phase 1.1 (AI-Knowledge-Notes修正)
    ↓
Phase 1.3 (章リンク修正)
    ↓
Phase 2.1 (breadcrumb修正)
    ↓
Phase 1.2 (ENシリーズ翻訳) ← 最も時間がかかる
    ↓
Phase 1.4 (JPシリーズ翻訳)
    ↓
Phase 2.2, 2.3 (構造整理)
    ↓
Phase 3 (品質評価)
```

---

## 推定作業量

| Phase | タスク | 影響ファイル数 | 複雑度 |
|-------|--------|---------------|--------|
| 1.1 | AI-Knowledge-Notes修正 | ~227 | 低（一括置換） |
| 1.2 | ENシリーズ翻訳 | ~20章 | 高（翻訳作業） |
| 1.3 | 章リンク修正 | ~300 | 中（個別確認必要） |
| 1.4 | JPシリーズ翻訳 | ~6章 | 高（翻訳作業） |
| 1.5 | 画像追加 | 2 | 低 |
| 2.1 | breadcrumb修正 | ~1000 | 低（一括置換） |
| 2.2 | index作成 | 1 | 中 |
| 2.3 | カバレッジ整理 | ~50 | 中 |
| 3.x | 品質評価 | 全体 | 高 |

---

## 次のアクション

ユーザーの承認後、以下の順序で実行:

1. **Phase 1.1**: AI-Knowledge-Notes絶対パスを相対パスに一括修正
2. **Phase 2.1**: breadcrumbパス修正
3. **Phase 1.3**: 章リンク不整合の修正
4. **Phase 1.2/1.4**: 翻訳作業（オプション）
5. **Phase 3**: 品質レポート作成
