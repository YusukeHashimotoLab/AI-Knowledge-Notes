# MI Dojo Extension Project Proposal
# 特徴量抽出シリーズ新規追加プロジェクト

**Date**: 2025-11-02
**Status**: 🟡 Proposal - Awaiting PRESIDENT Approval
**Priority**: Medium
**Estimated Duration**: 3-4 hours (3 workers parallel)

---

## 📋 Project Overview

### Objective
MIドメインに**材料特徴量抽出の比較**に関する新しい教育コンテンツを追加し、組成ベース特徴量（Magpie等）とグラフニューラルネットワーク系特徴量（CGCNN、MPNN等）の体系的な理解を提供する。

### Background
- 既存のMIドメインにはGNN入門シリーズが存在（5章構成）
- しかし、**組成ベース特徴量抽出**と**GNN特徴量**の**体系的な比較**コンテンツが不足
- ユーザー要望：Magpie等の組成ベース手法とCGCNN/MPNN等のGNN手法の特徴と比較

### Strategic Alignment
- MS Dojoプロジェクトの成功パターンを活用
- Phase 7品質基準（Academic Review ≥90点）を適用
- 既存MIドメインとの整合性維持

---

## 🎯 Deliverables

### New Content Series

#### Series 1: **composition-features-introduction** （組成ベース特徴量入門）
**Directory**: `/wp/knowledge/jp/MI/composition-features-introduction/`

**Chapter Structure** (5-6 chapters):
1. **組成ベース特徴量の基礎**
   - 組成表記と特徴量抽出の概念
   - 元素特性データベース（周期表情報、Zunger擬ポテンシャル半径等）
   - 統計量特徴量（平均、分散、最大・最小）

2. **Magpie特徴量**
   - Magpieフレームワークの原理
   - 145次元特徴量の構成（元素特性、化学量論、電子構造）
   - Python実装（matminer活用）
   - 実データでの特徴量抽出デモ

3. **その他の組成ベース手法**
   - Meredig特徴量（元素統計量18次元）
   - Valence orbital特徴量（価電子軌道情報）
   - Compositional属性（平均原子量、電気陰性度差等）
   - 各手法の比較と適用場面

4. **記述子エンジニアリング**
   - 特徴量選択（Recursive Feature Elimination、LASSO）
   - 次元削減（PCA、t-SNE、UMAP）
   - ドメイン知識の組み込み
   - 特徴量重要度分析

5. **Pythonワークフロー実践**
   - matminerによる統合ワークフロー
   - 複数特徴量セットの比較
   - 機械学習モデルとの統合
   - ベンチマークデータセット（Materials Project、OQMD）での実験

**Code Examples**: 35-40個（実行可能、matminer/pymatgen活用）
**Exercises**: 30-40問（Easy/Medium/Hard、詳細解答）
**References**: 30-42件（ページ番号必須）
**Target Lines**: 8,000-10,000行

---

#### Series 2: **gnn-features-comparison-introduction** （GNN特徴量比較入門）
**Directory**: `/wp/knowledge/jp/MI/gnn-features-comparison-introduction/`

**Chapter Structure** (5-6 chapters):
1. **GNNベース特徴量の基礎**
   - 結晶構造のグラフ表現（ノード＝原子、エッジ＝結合）
   - メッセージパッシング機構
   - グラフ畳み込みの原理

2. **CGCNN（Crystal Graph Convolutional Neural Networks）**
   - アーキテクチャ詳細（畳み込み層、プーリング）
   - 結晶構造からの特徴量抽出メカニズム
   - PyTorch実装とpre-trainedモデル活用
   - 形成エネルギー・バンドギャップ予測デモ

3. **MPNN（Message Passing Neural Networks）とSchNet**
   - MPNN一般フレームワーク
   - SchNet（連続フィルタ畳み込み）
   - MEGNet（Materials Graph Networks）
   - 各モデルの特徴と性能比較

4. **GNN特徴量 vs 組成ベース特徴量**
   - 表現能力の違い（局所構造情報 vs 組成統計量）
   - 計算コストとデータ要件
   - 予測性能の比較（ベンチマークデータセット）
   - ハイブリッドアプローチ（組成+構造）

5. **Pythonワークフロー実践**
   - PyTorch Geometricによる実装
   - 事前学習済みモデルの転移学習
   - 特徴量可視化（t-SNE、UMAP）
   - Materials Project APIとの統合

**Code Examples**: 35-40個（PyTorch Geometric、DGL活用）
**Exercises**: 30-40問（Easy/Medium/Hard、詳細解答）
**References**: 30-42件（ページ番号必須）
**Target Lines**: 8,000-10,000行

---

## 📊 Technical Specifications

### Quality Standards (Phase 7)
- ✅ Academic Review: ≥90/100 (first attempt)
- ✅ References: 6-7 per chapter (page numbers required)
- ✅ Exercises: 8-10 per chapter (Easy/Medium/Hard + detailed solutions)
- ✅ Learning objectives: 3-level assessment per chapter
- ✅ Code examples:実行可能、詳細コメント、Google Colab対応

### Design Requirements
- ✅ MS gradient統一（#667eea → #764ba2 for MI domain）
- ✅ MathJax 3.x数式レンダリング
- ✅ Mermaid.js図表統合
- ✅ Prism.js syntax highlighting
- ✅ Responsive design（mobile-first）
- ✅ WCAG 2.1 Level AA accessibility

### Content Guidelines
- 📖 article-writing-guidelines.md完全準拠
- 🔬 実データ・実装重視（理論だけでなく実践）
- 🎓 学習段階：初級者向け導入 → 中級者向け実装 → 上級者向け比較分析
- 🌐 国際標準論文引用（Nature, Science, npj Computational Materials等）

---

## 👥 Resource Allocation

### Worker Assignment Strategy

#### **Worker1**: composition-features-introduction
**Rationale**:
- Worker1の材料特性知識（materials-properties 92.3/100実績）
- 組成ベース特徴量は材料科学の基礎知識と親和性高い
- matminer/pymatgenライブラリ活用経験

**Estimated Time**: 2-2.5時間

---

#### **Worker2**: gnn-features-comparison-introduction
**Rationale**:
- Worker2の最高品質実績（electron-microscopy 93/100）
- GNN実装は高度な技術が必要（PyTorch Geometric）
- 比較分析の深い洞察力を活用

**Estimated Time**: 2.5-3時間（GNN実装の複雑性を考慮）

---

#### **Worker3**: Integration & Quality Assurance
**Rationale**:
- Worker3の高貢献実績（Phase 4コア完成）
- 2シリーズ間の整合性確保
- 既存GNN入門シリーズとのクロスリファレンス統合
- ナビゲーション更新（MI domain index.html）

**Tasks**:
1. 2シリーズ間のクロスリンク設定
2. 既存gnn-introductionシリーズとの整合性チェック
3. MI domain index.htmlへのナビゲーション追加
4. 全シリーズの品質ゲート総合レビュー

**Estimated Time**: 1-1.5時間

---

### Parallel Execution Plan

```
T+0:00  Boss1 → Worker1: composition-features指示
        Boss1 → Worker2: gnn-features-comparison指示

T+2:30  Worker1完成 → Worker3統合開始
        Worker2継続作業

T+3:00  Worker2完成 → Worker3統合完了
        Boss1品質ゲート実施

T+3:30  全シリーズ完成、PRESIDENT最終承認

Total: 3-3.5時間（並行実行）
```

---

## 📚 Key References (Preliminary)

### Composition-Based Features
1. Ward, L., et al. (2016). "A general-purpose machine learning framework for predicting properties of inorganic materials." *npj Computational Materials*, 2, 16028.
2. Meredig, B., et al. (2014). "Combinatorial screening for new materials in unconstrained composition space with machine learning." *Physical Review B*, 89, 094104.
3. Jain, A., et al. (2013). "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." *APL Materials*, 1, 011002.

### GNN-Based Features
1. Xie, T., Grossman, J.C. (2018). "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties." *Physical Review Letters*, 120, 145301.
2. Schütt, K.T., et al. (2017). "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions." *NeurIPS*, 991-1001.
3. Chen, C., et al. (2019). "Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals." *Chemistry of Materials*, 31, 3564-3572.

---

## 🎯 Success Criteria

### Quantitative Metrics
- ✅ Academic Review ≥90/100 (both series)
- ✅ Total files: 12-14 (index + 5-6 chapters × 2 series + integration)
- ✅ Total lines: 16,000-20,000行
- ✅ Code examples: 70-80個（実行可能）
- ✅ Exercises: 60-80問（詳細解答付き）
- ✅ References: 60-84件（ページ番号必須）

### Qualitative Metrics
- ✅ Publication Ready状態での完成
- ✅ 既存MIドメインとのシームレスな統合
- ✅ 国際水準の品質（MIT OCW等に匹敵）
- ✅ 実践的なPythonコード（Google Colab対応）
- ✅ 初級～上級学習者への段階的な学習パス提供

---

## 🚀 Implementation Plan

### Phase 1: Project Initialization (15 min)
- Boss1がPRESIDENT承認取得
- Worker1・Worker2・Worker3へタスク指示送信
- ディレクトリ構造作成確認

### Phase 2: Parallel Content Generation (2.5-3 hours)
- Worker1: composition-features-introduction生成
- Worker2: gnn-features-comparison-introduction生成
- 各WorkerがPhase 7標準を最初から適用

### Phase 3: Integration & QA (1 hour)
- Worker3が2シリーズ統合
- Boss1がAcademic Reviewer品質ゲート実施
- 90点未満の章は即座改善

### Phase 4: Final Approval (15 min)
- Boss1がPRESIDENTに最終報告
- プロジェクト完了宣言

---

## 💰 Risk Assessment

### Technical Risks
- **Low**: 既存GNN入門シリーズとの整合性問題
  - Mitigation: Worker3による統合レビュー
- **Low**: PyTorch Geometricコード実装の複雑性
  - Mitigation: Worker2の高度な技術力活用

### Schedule Risks
- **Low**: Worker2のGNN実装時間超過
  - Mitigation: 3時間のバッファ確保
- **Very Low**: 全体スケジュール遅延
  - Mitigation: 並行実行による効率化

### Quality Risks
- **Very Low**: Academic Review 90点未達
  - Mitigation: MS Dojo Phase 7標準の実証済み有効性

---

## 📝 Approval Request

### PRESIDENT Decision Points
1. **Approve Project Initiation?** (Yes/No/Modify)
2. **Approve Worker Allocation?** (Yes/No/Reassign)
3. **Approve Timeline (3-3.5 hours)?** (Yes/No/Adjust)
4. **Special Instructions?** (Any additional guidance)

---

## 📎 Related Documentation

- MS Dojo Project Final Status: `./tmp/MS_DOJO_PROJECT_FINAL_STATUS.md`
- Article Writing Guidelines: `/claudedocs/article-writing-guidelines.md`
- Existing GNN Series: `/wp/knowledge/jp/MI/gnn-introduction/`
- Existing MI Domain: `/wp/knowledge/jp/MI/`

---

**Prepared by**: Boss1
**Submission Date**: 2025-11-02
**Awaiting**: PRESIDENT Approval

---

## 🎊 Expected Outcome

この2シリーズ追加により、MIドメインは以下を達成します：

✅ 組成ベース特徴量とGNN特徴量の**体系的な比較学習パス**提供
✅ 実践的なPythonコード（matminer、PyTorch Geometric）による**即座活用可能**なリソース
✅ 初級者から上級者まで対応する**段階的な学習体験**
✅ 国際水準の品質（Academic Review ≥90点）による**信頼性の高い教育コンテンツ**

MS Dojoプロジェクトで確立された品質基準とプロセスを活用し、最高品質の成果物を効率的に構築します。
