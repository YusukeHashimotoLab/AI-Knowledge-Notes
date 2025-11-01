# GROMACS入門シリーズ

## 作成日
2025-11-01

## ファイル構成

### index.html（シリーズトップページ）
- タイトル: 🧬 GROMACS入門 - 生体分子シミュレーションの実践
- 全5章構成、2.5時間、35コード例、中級レベル
- CT colors: #11998e → #38ef7d（グラデーション）
- 章構成:
  1. GROMACSの基礎と環境構築（30分）
  2. タンパク質シミュレーションの実践（35分）
  3. 膜タンパク質系のシミュレーション（30分）
  4. 高度な解析手法（35分）
  5. GPU加速と性能最適化（30分）

### chapter-1.html（第1章: GROMACSの基礎と環境構築）

#### コンテンツ構成
- **学習目標**: 3レベル（基本理解、実践スキル、応用力）
- **セクション数**: 9セクション
  1. 本章の概要
  2. 分子動力学法の基礎（ニュートン方程式、力場、数値積分、アンサンブル）
  3. GROMACSの特徴（GPU対応、力場サポート、解析ツール、並列計算）
  4. インストール（ソースビルド、GPU対応確認、Conda簡易版）
  5. トポロジーとパラメータファイル（ファイル形式、.top構造、.mdp設定）
  6. 基本MDワークフロー（5ステップ: 準備→溶媒和→最小化→平衡化→本計算）
  7. 学習目標の確認
  8. 演習問題（10問）
  9. 参考文献（7件）

#### コード例（5個）
1. ソースからのインストール（GPU対応版）
2. GROMACSバージョンとGPUサポート確認
3. 水分子系の簡単なMDシミュレーション
4. リゾチームタンパク質のエネルギー最小化
5. GPU並列実行とパフォーマンス測定

#### 演習問題（10問）
**Easy（3問）**:
1. GROMACSバージョン確認
2. 力場の選択
3. エネルギー最小化の収束判定

**Medium（4問）**:
1. トポロジーファイルの理解
2. MDパラメータの設定
3. Leap-frog積分法の理解
4. GPU高速化の評価

**Hard（3問）**:
1. 力場の選択基準（GPCR、DNA-タンパク質、抗体）
2. MDワークフローの設計（100 ns計画）
3. 力場パラメータのカスタマイズ（GAFF vs 量子化学計算）

#### 参考文献（7件）
1. Abraham, M. J., et al. (2015). GROMACS: High performance molecular simulations. SoftwareX.
2. Maier, J. A., et al. (2015). ff14SB: AMBER protein force field. JCTC.
3. Huang, J., & MacKerell, A. D., Jr. (2013). CHARMM36 force field. JCC.
4. Berendsen, H. J. C., et al. (1995). GROMACS message-passing implementation. CPC.
5. GROMACS Documentation (2024). User Guide.
6. Lemkul, J. A. (2019). GROMACS Tutorial Suite. LiveCoMS.
7. Páll, S., et al. (2015). GPU acceleration in GROMACS. JCP.

## デザイン仕様

### カラーパレット（CT domain）
- Primary: #11998e（深緑）
- Accent: #38ef7d（ライムグリーン）
- Gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%)

### アイコン
- 🧬 (DNA) - 生体分子シミュレーションを象徴

### レイアウト
- GPU Computing シリーズと完全同一の構造
- Breadcrumb navigation
- Sticky header navigation
- Mermaid diagrams（フローチャート）
- Code highlighting (Prism.js)
- Math rendering (MathJax)
- Responsive design (mobile-first)

## 技術スタック

### フロントエンド
- HTML5
- CSS3 (Custom properties, Grid, Flexbox)
- JavaScript (Vanilla)

### ライブラリ
- Prism.js 1.29.0（コードハイライト）
- MathJax 3（数式レンダリング）
- Mermaid 10（ダイアグラム）

### 教育的特徴
- 実行可能なコード例（全て動作検証済み想定）
- 段階的な難易度設定（Easy → Medium → Hard）
- 実践的なワークフロー（水分子 → リゾチーム → GPU最適化）
- 詳細な解説付き演習問題
- 外部リンク付き参考文献

## ファイルサイズ
- index.html: 30KB
- chapter-1.html: 72KB
- 合計: 102KB

## 品質チェック項目
- ✅ GPU Computingシリーズと同一デザイン
- ✅ CT colors適用（#11998e → #38ef7d）
- ✅ 5コード例（実践的な内容）
- ✅ 10演習問題（3 Easy + 4 Medium + 3 Hard）
- ✅ 7参考文献（DOI/URLリンク付き）
- ✅ Mermaidダイアグラム（フローチャート）
- ✅ 数式表示（MathJax）
- ✅ Breadcrumbナビゲーション
- ✅ レスポンシブデザイン
- ✅ アクセシビリティ対応

## 次のステップ
- Chapter 2-5の作成（タンパク質シミュレーション、膜系、解析、GPU最適化）
- コード例の動作確認
- 演習問題の解答検証
- 相互リンクの確認
