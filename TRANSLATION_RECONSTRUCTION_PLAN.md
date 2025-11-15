# 翻訳再構築プラン - AI Terakoya English Platform

**Date**: 2025年11月9日
**Status**: 計画段階
**対象**: 274ファイル（FM 23 + ML 119 + MS 37 + PI 95 + その他）

---

## 現状分析

### 問題の深刻度

**CRITICAL**: 辞書ベース翻訳が完全に失敗し、単語レベルで日英が混在

**例**:
```html
<!-- 現状（読めない） -->
<p>この Series、AIAgent（AI Agents）をBasics fromstep-by-step学べる全4 Chapter構成のPractice的教育コンテンツis。</p>

<!-- 正しい翻訳 -->
<p>This series is a practical educational content consisting of 4 chapters where you can learn AI Agents step-by-step from basics.</p>
```

### 原因分析

1. **辞書ベース翻訳の限界**: 文脈を無視した単語置換
2. **部分マッチング**: 「学べる」が「学」+「べる」に分割され誤訳
3. **助詞処理の失敗**: 「の」「を」「に」などが削除されず残存
4. **語順の問題**: 日本語の語順のまま英単語を挿入

### 必要なアプローチ

**× 辞書ベース翻訳** → 文脈無視で単語置換、失敗
**○ AI完全翻訳** → 文脈理解した自然な英語、成功

---

## 翻訳戦略

### Phase 1: パイロット翻訳（検証）

**目的**: 翻訳品質とプロセスの検証
**対象**: 各カテゴリーから1ファイルずつ（計5ファイル）
**方法**: Subagent個別翻訳
**期待品質**: 100%英語、自然な文章、技術的正確性

**パイロットファイル**:
1. FM/quantum-mechanics/chapter-1.html
2. ML/ai-agents-introduction/index.html
3. MS/electron-microscopy-introduction/index.html
4. PI/pi-introduction/chapter-1.html
5. MI/mi-introduction/chapter4-real-world.html（参照用、既に完璧）

**成功基準**:
- 日本語文字 < 1%（メタデータIDのみ許容）
- 読みやすい自然な英語
- 技術用語の正確性
- HTML構造保持
- コード・数式の保持

---

### Phase 2: バッチ翻訳（大規模実行）

**前提**: Phase 1で品質確認済み

**実行方法**: カテゴリー別並列処理

#### Batch 1: FM（23ファイル）- 優先度 HIGH
- **理由**: 最小ファイル数、数学的内容で検証しやすい
- **並列度**: 4 agents（各5-6ファイル）
- **想定時間**: 30-40分

#### Batch 2: MS（37ファイル）- 優先度 CRITICAL
- **理由**: 前回の翻訳が最も壊れている
- **並列度**: 6 agents（各6ファイル）
- **想定時間**: 40-50分

#### Batch 3: ML（119ファイル）- 優先度 HIGH
- **理由**: 最大ファイル数、段階的処理が必要
- **並列度**: 10 agents（各12ファイル）
- **想定時間**: 60-80分

#### Batch 4: PI（95ファイル）- 優先度 MEDIUM
- **理由**: 前回翻訳で比較的良好だったが再確認
- **並列度**: 8 agents（各12ファイル）
- **想定時間**: 50-70分

---

## Subagent翻訳プロトコル

### 各Agentの責務

**Input**:
- 日本語HTMLファイルパス（JP source）
- 出力先英語ファイルパス（EN target）
- 参照用MI翻訳サンプル

**Process**:
1. **読み込み**: JPソースファイル全体を読む
2. **構造分析**: HTML構造、コードブロック、数式を識別
3. **翻訳**: 文脈を考慮した自然な英語に翻訳
4. **保持**: HTML/CSS/JS/コード/MathJax/リンクを完全保持
5. **検証**: 日本語残存チェック（< 1%）
6. **書き込み**: ENファイルに上書き

**重要ルール**:
- **文単位で翻訳**: 単語置換ではなく文脈理解
- **技術用語の一貫性**: MI翻訳を参考
- **自然な英語**: ネイティブレベルの流暢さ
- **構造保持**: HTMLタグ、属性、クラス名は変更しない
- **コード保持**: Pythonコード、数式は一切変更しない

### 翻訳ガイドライン

#### 技術用語の扱い

**一貫性のある翻訳**:
- 機械学習 → Machine Learning
- 深層学習 → Deep Learning
- ニューラルネットワーク → Neural Networks
- 強化学習 → Reinforcement Learning
- 畳み込み → Convolution
- 量子力学 → Quantum Mechanics
- 統計力学 → Statistical Mechanics

**カタカナ→英語**:
- エージェント → Agent
- モデル → Model
- アルゴリズム → Algorithm
- パラメータ → Parameter

#### 自然な英語表現

**日本語直訳を避ける**:
```
❌ This chapter, we learn basics of AI agents.
✅ In this chapter, we learn the basics of AI agents.

❌ By reading this, following things you can understand.
✅ By reading this, you will understand the following concepts.

❌ Let's try to implement actual code.
✅ Let's implement the actual code.
```

#### メタデータの扱い

**変更する**:
- `lang="ja"` → `lang="en"`
- ページタイトル、説明文 → 英語に翻訳
- ナビゲーション、パンくず → 英語に翻訳

**保持する**:
- コースID（例: "コースID: ML-D08"）
- 著者名（日本人名）
- 組織名（日本語の場合）

---

## 並列処理戦略

### Agent配分

**同時実行Agent数**: 最大10（トークン制限考慮）

**優先順位付きキュー**:
1. Priority 1（CRITICAL）: MS 37ファイル
2. Priority 2（HIGH）: FM 23ファイル + ML 119ファイル
3. Priority 3（MEDIUM）: PI 95ファイル

### リソース管理

**トークン予算**: 200,000 tokens total
- Phase 1パイロット: ~20,000 tokens（5ファイル）
- Phase 2バッチ: ~150,000 tokens（269ファイル）
- 予備: ~30,000 tokens

**時間見積もり**:
- Phase 1: 20-30分
- Phase 2 Batch 1-2: 60-90分
- Phase 2 Batch 3-4: 90-150分
- **Total**: 3-4.5時間

---

## 品質保証プロセス

### 自動検証（各ファイル）

1. **日本語文字数チェック**: < 1%（メタデータID除く）
2. **HTML検証**: 正しいHTML構造
3. **リンク検証**: 壊れたリンクなし
4. **lang属性**: `lang="en"`に変更済み

### サンプリング検証（バッチ後）

各カテゴリーから2-3ファイルをランダム選択:
- 読みやすさ（ネイティブレベル）
- 技術的正確性
- 用語の一貫性
- HTMLレンダリング

### 最終レポート

- 翻訳済みファイル数
- 成功/失敗カウント
- 平均日本語残存率
- 発見された問題
- 推奨される改善

---

## 実行計画

### Step 1: パイロット翻訳（検証）✅

```bash
# 5個のSubagentを並列起動
Agent 1: FM/quantum-mechanics/chapter-1.html
Agent 2: ML/ai-agents-introduction/index.html
Agent 3: MS/electron-microscopy-introduction/index.html
Agent 4: PI/pi-introduction/chapter-1.html
Agent 5: MI/mi-introduction/chapter4-real-world.html（検証用参照）
```

**判定**:
- 全5ファイルが品質基準を満たす → Phase 2へ
- 1ファイルでも失敗 → プロトコル改善してリトライ

### Step 2: FM完全翻訳（23ファイル）

```bash
# 4個のSubagentを並列起動
Agent 1: quantum-mechanics series（6ファイル）
Agent 2: quantum-field-theory-introduction series（6ファイル）
Agent 3: classical-statistical-mechanics series（6ファイル）
Agent 4: probability-stochastic-processes series（5ファイル）
```

### Step 3: MS完全翻訳（37ファイル）

```bash
# 6個のSubagentを並列起動
Agent 1: electron-microscopy-introduction（6ファイル）
Agent 2: crystallography-introduction（6ファイル）
Agent 3: ceramic-materials-introduction（6ファイル）
Agent 4: composite-materials-introduction（6ファイル）
Agent 5: metallic-materials-introduction（6ファイル）
Agent 6: materials-properties-introduction（7ファイル）
```

### Step 4: ML完全翻訳（119ファイル）- 分割実行

**ML Batch 1（60ファイル）**:
```bash
# 5個のSubagentを並列起動（各12ファイル）
Agent 1-5: 各シリーズ
```

**ML Batch 2（59ファイル）**:
```bash
# 5個のSubagentを並列起動（各12ファイル）
Agent 6-10: 残りシリーズ
```

### Step 5: PI再検証翻訳（95ファイル）

```bash
# 8個のSubagentを並列起動（各12ファイル）
Agent 1-8: 各シリーズ
```

### Step 6: 最終検証とレポート

- 全274ファイルの日本語チェック
- サンプリング品質確認
- 最終レポート生成

---

## リスク管理

### リスク1: Agent失敗

**対策**:
- 失敗ファイルリストを保持
- リトライキューで再実行
- 最大3回まで自動リトライ

### リスク2: トークン不足

**対策**:
- バッチサイズ削減（10→5 agents）
- ファイルサイズ別優先順位
- 大容量ファイルは個別処理

### リスク3: 品質不一致

**対策**:
- パイロットフェーズで早期検出
- MI参照サンプル提供
- 翻訳ガイドライン明確化

### リスク4: 時間超過

**対策**:
- 段階的実行（1日で全部やらない）
- チェックポイントで進捗保存
- 優先度順実行（CRITICAL → HIGH → MEDIUM）

---

## 成功基準

### 最小基準（必須）

- ✅ 全274ファイル翻訳完了
- ✅ 日本語残存 < 1%（メタデータID除く）
- ✅ HTML構造保持100%
- ✅ リンク機能100%

### 品質基準（目標）

- ✅ 自然な英語（ネイティブレベル）
- ✅ 技術用語一貫性
- ✅ MI翻訳と同等品質
- ✅ ユーザーテスト合格

---

## 次のアクション

1. **ユーザー承認**: このプラン内容の確認
2. **Phase 1開始**: パイロット翻訳（5ファイル、30分）
3. **品質検証**: パイロット結果の確認
4. **Phase 2実行判断**: GO/NO-GO決定
5. **大規模翻訳**: 承認後に段階的実行

---

**準備完了 - ユーザーの承認待ち**
