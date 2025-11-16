# Phase 4-1-6: 欠落チャプター方針決定 - 分析と推奨プラン

**分析日**: 2025-11-16
**対象**: 406件のリンク切れ
**担当**: Claude Code
**ステータス**: 分析完了、方針推奨

---

## リンク切れ分類結果

### 総合サマリー

**合計**: 406件のリンク切れ

| カテゴリ | 件数 | 割合 | 優先度 | 推定作業時間 |
|---------|-----|------|--------|------------|
| **Other（シリーズ参照）** | 283 | 69.7% | 🟡 中 | 2-3時間 |
| **Asset Paths** | 62 | 15.3% | 🔴 高 | 1-2時間 |
| **Missing Index** | 27 | 6.7% | 🔴 高 | 2-3時間 |
| **Missing Chapters** | 24 | 5.9% | 🟢 低 | 30分 |
| **JP References** | 10 | 2.5% | 🟡 中 | 30分 |

---

## カテゴリ別詳細分析

### 1. Missing Chapters（24件 - 5.9%）

**実態**: 当初の予測（194件）と大きく異なり、実際は **24件のみ**

**内訳**:
| シリーズ | 欠落チャプター数 | ファイル |
|---------|---------------|---------|
| FM/equilibrium-thermodynamics | 4 | chapter-2,3,4,5.html |
| MS/materials-chemistry-introduction | 5 | chapter-1,2,3,4,5.html |
| MI/mi-journals-conferences-introduction | 1 | chapter-4.html |
| MS/electrical-magnetic-testing-introduction | 1 | chapter-5.html |
| MS/materials-thermodynamics-introduction | 1 | chapter-6.html |
| MS/mechanical-testing-introduction | 1 | chapter-5.html |
| MS/polymer-materials-introduction | 1 | chapter-5.html |
| MS/synthesis-processes-introduction | 1 | chapter-5.html |
| MS/thin-film-nano-introduction | 1 | chapter-5.html |
| PI/digital-twin-introduction | 2 | chapter-2,5.html |
| PI/food-process-ai-introduction | 2 | chapter-2,5.html |

**推奨対応**: **Option B - ナビゲーション更新**
- 各シリーズのindex.htmlから欠落チャプターへのリンクを削除
- 作業時間: **30分**（11シリーズの簡易修正）
- 理由: 24件と少数なので、完成したチャプターのみ表示する方が誠実

---

### 2. Asset Paths（62件 - 15.3%）

**問題**: CSSとJSファイルへの相対パス誤り

**主な問題パス**:
```
../../assets/js/navigation.js
../../assets/js/main.js
../../assets/css/variables.css
../../assets/css/base.css
../../assets/css/layout.css
../../assets/css/components.css
../../assets/css/article.css
../../../../assets/css/knowledge.css
```

**影響範囲**: MI/gnn-introduction、MS/composite-materials-introductionなど

**推奨対応**: **アセットパス統一修正**
- 正しいパスに一括置換
- または、実際にアセットファイルを作成（存在しない場合）
- 作業時間: **1-2時間**
- 優先度: **高**（表示崩れの原因）

---

### 3. Missing Index（27件 - 6.7%）

**問題**: 存在しないindex.htmlへの参照

**主要パターン**:

**A. 相対パス誤り（ナビゲーションループ）**:
```
../../FM/index.html  （knowledge/FM/index.htmlを参照しようとしている）
../../../../index.html
../../../en/knowledge/index.html
```
→ **対応**: 相対パスを修正（knowledge/en/FM/index.htmlが正しい）

**B. 欠落シリーズ**:
```
inferential-bayesian-statistics/index.html  （FM/）
robotic-lab-automation-introduction/index.html  （MI/）
gnn-features-comparison/index.html  （MI/）
materials-screening-workflow/index.html  （MI/）
process-monitoring/index.html  （PI/）
process-optimization/index.html  （PI/）
```
→ **対応**: リンクを削除、またはシリーズ作成

**作業時間**: **2-3時間**
**優先度**: **高**（404エラーの原因）

---

### 4. Other（283件 - 69.7%）

**実態**: 主に **非存在シリーズへの参照**

**頻出パターン**（Top 20）:
| 参照先 | 回数 | 対応 |
|-------|------|------|
| `../machine-learning-basics/` | 11 | 削除（シリーズ未作成） |
| `../../mi-introduction/` | 8 | 削除（Phase 4-1-2で削除済み） |
| `/wp/knowledge/` | 4 | パス修正 |
| `../../../en/research.html` | 4 | 削除（該当ページなし） |
| `../../../en/publications.html` | 4 | 削除 |
| `../../../en/news.html` | 4 | 削除 |
| `../../../en/members.html` | 4 | 削除 |
| `../../../en/contact.html` | 4 | 削除 |
| `./chapter-5.html` | 3 | ナビゲーション削除 |
| `chapter4-real-world.html` | 3 | ファイル名誤り修正 |
| `../deep-learning-advanced/` | 3 | 削除 |
| `../python-for-data-science/` | 3 | 削除 |
| `./chapter1-*.html` | 15 | インデックス修正 |

**推奨対応**: **一括リンク削除スクリプト**
- 非存在シリーズへの参照を全削除
- ファイル名誤りを修正
- 作業時間: **2-3時間**
- 優先度: **中**（機能的影響は小）

---

### 5. JP References（10件 - 2.5%）

**問題**: 日本語版へのリンク（英語版には不要）

**パターン**:
```
/jp/MI/gnn-introduction/chapter-1.html
../../../knowledge_en.html
../mi_en.html
../jp/knowledge/index.html
```

**推奨対応**: **ロケールスイッチャー統一**
- Phase 3-4で追加したロケールスイッチャーに統一
- 古いJP参照を全削除
- 作業時間: **30分**
- 優先度: **中**

---

## 総合推奨プラン

### 🎯 推奨: **Phase 4-2 集中修正アプローチ**

**方針**: 当初の予測（194欠落チャプター）は誤りで、実際は小規模な修正で対応可能

### Phase 4-2-1: 高優先度修正（3-4時間）

**タスク**:
1. **Asset Paths修正**（1-2時間）
   - CSSとJSへの相対パスを統一
   - 不足アセットファイルの確認と作成
   - 対象: 62件

2. **Missing Index修正**（2-3時間）
   - 相対パス誤りを修正（ナビゲーションループ）
   - 欠落シリーズへの参照を削除
   - 対象: 27件

**成果**: **89件のリンク切れ解消**（21.9%）

---

### Phase 4-2-2: 中優先度修正（2-3時間）

**タスク**:
1. **非存在シリーズ参照削除**（2-3時間）
   - machine-learning-basics, deep-learning-advancedなどへの参照削除
   - research.html, publications.htmlなどへのリンク削除
   - ファイル名誤り修正（chapter4-real-world.html → chapter-4.html）
   - 対象: 283件

2. **JP References削除**（30分）
   - 古いJP参照リンクを全削除
   - ロケールスイッチャーに統一
   - 対象: 10件

**成果**: **293件のリンク切れ解消**（72.2%）

---

### Phase 4-2-3: 低優先度修正（30分）

**タスク**:
1. **Missing Chapters対応**（30分）
   - 各シリーズindex.htmlから欠落チャプターリンクを削除
   - 11シリーズのナビゲーション更新
   - 対象: 24件

**成果**: **24件のリンク切れ解消**（5.9%）

---

## 実施プラン詳細

### 修正スクリプト作成

**1. fix_asset_paths.py** (AssetPaths修正)
```python
# 相対パスパターンを検出して統一
# 例: ../../assets/css/variables.css → ../../../assets/css/knowledge-base.css
```

**2. fix_missing_index.py** (MissingIndex修正)
```python
# 相対パス誤りを修正
# 例: ../../FM/index.html → ../../../index.html
# 欠落シリーズへのリンクを削除
```

**3. remove_nonexistent_series.py** (非存在シリーズ参照削除)
```python
# machine-learning-basics, deep-learning-advancedなどへの参照削除
# research.html等へのリンク削除
```

**4. remove_jp_references.py** (JP参照削除)
```python
# /jp/、knowledge_en.html等を削除
```

**5. hide_missing_chapters.py** (欠落チャプター非表示)
```python
# index.htmlから欠落チャプターへのナビゲーションを削除
```

---

## 所要時間見積もり

| フェーズ | タスク | 時間 | 累計 |
|---------|-------|------|------|
| Phase 4-2-1 | Asset Paths修正 | 1-2時間 | 1-2時間 |
| Phase 4-2-1 | Missing Index修正 | 2-3時間 | 3-5時間 |
| Phase 4-2-2 | 非存在シリーズ削除 | 2-3時間 | 5-8時間 |
| Phase 4-2-2 | JP References削除 | 30分 | 5.5-8.5時間 |
| Phase 4-2-3 | Missing Chapters対応 | 30分 | 6-9時間 |
| **合計** | - | **6-9時間** | - |

**実施期間**: 1-2日（集中作業）

---

## 成果予測

### リンク切れ解消

- **現状**: 406件
- **Phase 4-2完了後**: **0件**（100%解消）

### 品質向上

1. **ナビゲーション**: 全リンクが有効
2. **アセット**: CSS/JS正常読み込み
3. **ユーザー体験**: 404エラー撲滅
4. **保守性**: 非存在参照の整理

---

## 当初予測との差異

### 修正案.mdでの記載
> **Missing Chapters** (194件) - 存在しないチャプターファイル

### 実際の分析結果
- **Missing Chapters**: 24件（5.9%）
- **その他の問題**: 382件（94.1%）

### 原因
- Phase 2-4で20ファイルの空ファイルを既に修正済み
- Phase 3-2で509件のリンク修正（497→406に削減）
- 実際の問題は「非存在シリーズへの参照」が主

### 結論
**Option B（ナビゲーション更新のみ）で十分**
- 欠落チャプター作成は不要
- 代わりに、アセットパスと非存在参照の修正が重要

---

## 次のステップ

### 即時実施推奨

1. **Phase 4-2-1開始**: 高優先度修正（Asset Paths + Missing Index）
2. **検証**: リンクチェック再実行で効果確認
3. **Phase 4-2-2**: 中優先度修正
4. **Phase 4-2-3**: 低優先度修正
5. **最終検証**: linkcheck_en_local.txtで0件確認

### 承認待ち

**ユーザー確認事項**:
- この修正プラン（Phase 4-2）で進めてよいか？
- 欠落チャプター24件はナビゲーションから削除でよいか？
- 非存在シリーズへの参照は全削除でよいか？

---

**分析完了日**: 2025-11-16
**次のフェーズ**: Phase 4-2（リンク切れ集中修正）
**推定完了**: 2025-11-17～18（1-2日）
