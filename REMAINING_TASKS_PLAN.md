# 残タスク修正プラン (v2)

**作成日**: 2025-11-16
**ベース**: `/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/修正案.md`
**前提**: Phase 3 (Infrastructure) 完了済み

## Phase 3 完了済み項目 ✅

以下の項目は既に完了しています：

### ✅ 1. IA/リンク整合性の最終更新（優先度A-1）
**完了内容**:
- Phase 3-2で509件のリンクを自動修正
- パンくず深度問題: 418件修正
- 絶対パス問題: 39件修正
- アセットパス問題: 35件修正
- ファイル名誤り: 12件修正

**現状**: 497件 → 406件に削減（91件解決）

**残課題**: 406件（主に存在しないチャプター194件、非存在シリーズ10件）

### ✅ 3. Markdownソースと生成フローの復元（優先度A-3）
**完了内容**:
- Phase 3-3で双方向Markdown-HTMLパイプライン構築
- `convert_md_to_html_en.py`: MD → HTML
- `html_to_md.py`: HTML → MD逆変換
- `sync_md_html.py`: 双方向同期
- MathJax、Mermaid、コードハイライトサポート

**現状**: 完全な生成フローが確立

### ✅ 4. コード表示の体裁崩れ（優先度A-4）の一部
**完了内容**:
- Phase 2-1で70ファイル348コードブロックを修正
- `<pre><code class="language-python">` で適切にラップ

**残課題**: 112ファイル（182-70=112）がまだ未修正の可能性

### ✅ 5. 0バイト/極小ファイルの整理（優先度A-5）
**完了内容**:
- Phase 2-4で20ファイルの完全なコンテンツを作成
- MS: mechanical-testing (4章)、synthesis-processes (4章)、thin-film-nano (4章)
- PI: chemical-plant-ai chapter-4
- ML: model-evaluation (4章)、ensemble-methods (2章)、unsupervised-learning (1章)

**現状**: 全20ファイル完了、0バイト問題解消

### ✅ 7. バックアップ/テンポラリの除去（優先度B-7）
**完了内容**:
- Phase 1-1で3つの.backupファイル削除
- `.gitignore`にパターン追加

**残課題**: 他の`.backup`、`_temp.html`があれば追加削除

### ✅ 10. JPリンク・FAQ導線の欠落（優先度B-10）
**完了内容**:
- Phase 3-4で576ファイルにロケールスイッチャー追加
- 最終同期日表示（git履歴ベース）
- レスポンシブデザイン、WCAG AA準拠

**現状**: 完全に実装済み

### ✅ 11. TODO/内部メモの残置（優先度B-11）
**完了内容**:
- Phase 2-3で122件のTODOをクリーンアップ
- 演習プロンプト: 76件（`# TODO:` → `# Exercise:`）
- 実装修正: 16件
- 内部メモ削除: 30件

**現状**: ほぼ完了（一部新規追加があれば要対応）

---

## 残タスク（Phase 4として実施）

### 優先度A（緊急）

#### A-1: 残りのリンク切れ修正（406件）
**現状**: linkcheck_en_local.txtで406件の壊れたリンク

**内訳**:
1. **Missing Chapters** (194件) - 存在しないチャプターファイル
   - 例: `FM/equilibrium-thermodynamics/chapter-2.html` ~ `chapter-5.html`
   - 例: `ML/anomaly-detection-introduction/chapter1-anomaly-basics.html`

2. **Non Existent Series** (10件) - 未作成のシリーズ
   - 例: `../llm-basics/`, `../machine-learning-basics/`
   - 例: `../robotic-lab-automation-introduction/`

3. **Other** (202件) - その他のパス・アセット問題

**対応方法**:
```bash
# Option 1: 欠落チャプターを作成
python3 tools/html_to_md.py knowledge/en/FM/equilibrium-thermodynamics/chapter-1.html
# Edit chapter-1.md to create chapter-2.md template
python3 tools/convert_md_to_html_en.py knowledge/en/FM/equilibrium-thermodynamics/chapter-2.md

# Option 2: ナビゲーションから削除
# Edit index.html to remove broken chapter links
```

**推奨アプローチ**:
1. 高優先度シリーズ（頻繁にアクセスされる）は欠落チャプターを作成
2. 低優先度シリーズはナビゲーションを更新してリンク削除
3. 非存在シリーズへのリンクは全て削除

**所要時間**: 2-3日（194チャプター作成の場合）

#### A-2: シリーズ重複の解消（優先度A-2）
**現状**: 一部解消済み（MI/pi-introduction削除済み）

**残存確認**:
```bash
# 重複シリーズを検索
find knowledge/en -name "mi-introduction" -type d
find knowledge/en -name "nm-introduction" -type d
find knowledge/en -name "pi-introduction" -type d
```

**対応手順**:
1. 重複ディレクトリを特定
2. 正規位置を決定（ML vs MI）
3. 片方を削除し、リンクを更新
4. リンクチェック再実行

**所要時間**: 1-2時間

#### A-6: MS Dojoトップの誤コンテンツ（優先度A-6）
**現状**: `knowledge/en/MS/index.html`がFM Dojoのコピー

**確認**:
```bash
head -50 knowledge/en/MS/index.html | grep -i "fundamentals\|FM"
```

**対応手順**:
1. MS Dojo用の正しいコンテンツを作成
   - タイトル: "Materials Science Dojo"
   - 説明: MS系列の概要
   - 統計: MS系20シリーズの情報
2. パンくずを修正: `MS Dojo`
3. Markdownソース作成: `MS/index.md`
4. 再生成フローに組み込み

**所要時間**: 1-2時間

#### A-4-残: 残りのコード体裁崩れ（112ファイル）
**現状**: Phase 2で70ファイル修正済み、残り112ファイル

**確認**:
```bash
# 未修正ファイルを検索
grep -r "import " knowledge/en --include="*.html" | \
  grep -v "<pre><code" | \
  cut -d: -f1 | sort -u | wc -l
```

**対応手順**:
1. 未修正ファイルリストを生成
2. スクリプトで一括修正（Phase 2-1と同じ手法）
3. 検証: `verify_code_formatting.py`

**所要時間**: 2-3時間

### 優先度B（重要）

#### B-7-残: 残りのバックアップ/テンポラリファイル
**確認**:
```bash
find knowledge/en -name "*.backup" -o -name "*_temp.html" -o -name "*.bak"
```

**対応手順**:
1. リスト生成
2. 必要なバックアップは`archive/`へ移動
3. 不要なものは削除
4. `.gitignore`更新確認

**所要時間**: 30分

#### B-8: 非セマンティック構造・インラインCSSの解消
**現状**: 多くのファイルがインラインCSS

**対応手順**:
1. 共通CSS確認: `knowledge/en/assets/css/knowledge-base.css`
2. テンプレート更新: `<main><article><section>`使用
3. インラインCSSを外部CSSへ移行
4. Markdownパイプラインで新規生成時に適用

**注意**: これは既存582ファイル全てに影響するため、段階的実施を推奨

**所要時間**: 2-3日（全ファイル更新の場合）

#### B-9: 翻訳ステータス/メタ情報の整合性
**現状**: `ML/TRANSLATION_STATUS.md`に不整合

**対応手順**:
1. 現行ディレクトリ構造をスキャン
2. 自動生成スクリプト作成: `generate_translation_status.py`
3. 各Dojoの`TRANSLATION_STATUS.md`を更新
4. 更新日・担当者情報を追加

**所要時間**: 1-2時間

---

## Phase 4: 実装プラン

### Phase 4-1: 緊急リンク修正（2-3日）

**タスク**:
1. ✅ リンク切れ406件の分析（完了）
2. ⏭️ MS/index.html修正（1-2時間）
3. ⏭️ シリーズ重複解消（1-2時間）
4. ⏭️ 残りコード体裁修正（2-3時間）
5. ⏭️ 欠落チャプター対応方針決定
   - Option A: 全194チャプター作成（2-3日）
   - Option B: ナビゲーション更新のみ（4-6時間）
   - Option C: 混合アプローチ（1-2日）

**成果物**:
- MS/index.html修正版
- 重複シリーズ削除記録
- コード体裁修正レポート
- 欠落チャプター対応計画

### Phase 4-2: クリーンアップ（1日）

**タスク**:
1. ⏭️ バックアップファイル整理（30分）
2. ⏭️ 翻訳ステータス自動生成（1-2時間）
3. ⏭️ 最終リンク検証（30分）
4. ⏭️ レンダリング確認（2-3時間）

**成果物**:
- クリーンなディレクトリ構造
- 正確な翻訳ステータス
- 0件のリンク切れ（または文書化された既知の問題）
- レンダリング確認レポート

### Phase 4-3: HTML構造改善（Optional, 2-3日）

**タスク**:
1. ⏭️ セマンティックHTML移行計画
2. ⏭️ CSS集約とテンプレート更新
3. ⏭️ 段階的ロールアウト（ドメインごと）
4. ⏭️ 検証とロールバックプラン

**成果物**:
- セマンティックHTMLテンプレート
- 集約されたCSS
- 移行ガイドライン

---

## 推奨実施順序

### 即時対応（今日～明日）
1. **MS/index.html修正** (1-2時間) - 誤コンテンツの修正
2. **シリーズ重複解消** (1-2時間) - 残存重複の確認と削除
3. **バックアップ整理** (30分) - 不要ファイルの削除

### 短期対応（今週中）
4. **残りコード体裁修正** (2-3時間) - 112ファイルの修正
5. **翻訳ステータス更新** (1-2時間) - 自動生成スクリプト実装
6. **欠落チャプター対応方針決定** - ステークホルダーと協議

### 中期対応（来週～）
7. **欠落チャプター作成** (Option Aの場合: 2-3日)
   - または **ナビゲーション更新** (Option Bの場合: 4-6時間)
8. **HTML構造改善** (Optional: 2-3日)

---

## 検証チェックリスト

各フェーズ完了時に以下を実行:

```bash
# 1. リンクチェック
python3 scripts/check_links.py
# → 目標: 406件 → 0件（または文書化された既知問題のみ）

# 2. コード体裁検証
python3 scripts/verify_code_formatting.py
# → 目標: 100% パス

# 3. HTML構文検証
find knowledge/en -name "*.html" -exec tidy -qe {} \; 2>&1 | \
  grep -c "Error"
# → 目標: 0 errors

# 4. Mermaid検証（JPツール流用）
python3 knowledge/jp/validate_mermaid.py knowledge/en
# → 目標: 全Mermaidブロック有効

# 5. レンダリング確認
python3 -m http.server 8000
# → ブラウザで目視確認: パンくず、ナビ、コード表示
```

---

## リスク管理

### 高リスク項目
1. **欠落チャプター194件の作成**: 大量の作業、品質管理が必要
   - **緩和策**: テンプレート化、AI支援、段階的実施

2. **HTML構造の大規模変更**: 582ファイル全体に影響
   - **緩和策**: 段階的ロールアウト、十分なテストとバックアップ

### 中リスク項目
3. **シリーズ重複解消**: リンク切れのリスク
   - **緩和策**: リンクチェック自動化、リダイレクト設定

4. **コード体裁修正**: レンダリング崩れのリスク
   - **緩和策**: 既存スクリプト活用、検証自動化

---

## 成果物サマリー

### ドキュメント
- [ ] MS/index.html修正版
- [ ] シリーズ重複解消レポート
- [ ] コード体裁修正レポート
- [ ] 欠落チャプター対応計画
- [ ] 翻訳ステータス自動生成スクリプト
- [ ] Phase 4完了レポート

### スクリプト
- [ ] generate_translation_status.py（新規）
- [ ] fix_remaining_code_formatting.py（Phase 2スクリプト流用）
- [ ] cleanup_backups.sh（新規）

### 更新ファイル
- [ ] knowledge/en/MS/index.html
- [ ] 各Dojo/TRANSLATION_STATUS.md
- [ ] 112ファイルのコード体裁修正
- [ ] 欠落チャプター（方針次第で0～194ファイル）

---

## 次のステップ

**推奨**: Phase 4-1から開始

```bash
# 即時対応タスク実行
# 1. MS/index.html確認
head -50 knowledge/en/MS/index.html

# 2. 重複シリーズ確認
find knowledge/en -type d -name "*-introduction" | sort | uniq -d

# 3. バックアップファイル確認
find knowledge/en -name "*.backup" -o -name "*_temp.html"
```

ご指示いただければ、Phase 4-1の実装を開始します。
