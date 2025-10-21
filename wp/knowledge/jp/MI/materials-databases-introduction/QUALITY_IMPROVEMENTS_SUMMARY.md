# Materials Databases Introduction Series - Quality Improvements Summary

**Date**: 2025-10-19
**Task**: Apply quality improvements to all chapters following MI Introduction template structure

## Improvements Applied

### 1. Database Licensing Comparison Table (Chapter 1)
**Location**: Section 1.3 (after usage table)

| データベース | ライセンス | 商用利用 | 引用要件 | データ公開条件 |
|------------|-----------|---------|----------|--------------|
| **Materials Project** | CC BY 4.0 | 可 | 必須 | 同一ライセンス推奨 |
| **OQMD** | ODbL 1.0 | 可 | 必須 | 派生データもODbL |
| **NOMAD** | CC BY 4.0 | 可 | 必須 | オープンアクセス |
| **AFLOW** | Academic Free License | 学術利用可 | 必須 | 制限あり |
| **JARVIS** | Public Domain (NIST) | 可 | 推奨 | 制限なし |

**引用形式例**:
```
Materials Project: Jain et al., APL Materials 1, 011002 (2013)
OQMD: Saal et al., JOM 65, 1501-1509 (2013)
JARVIS: Choudhary et al., npj Computational Materials 6, 173 (2020)
```

---

### 2. API Usage Reproducibility (Chapter 2)
**Location**: New section 2.9 (before summary)

#### 2.9.1 APIレート制限の詳細

| データベース | 無料プラン | 有料プラン | リセット期間 |
|------------|----------|-----------|------------|
| Materials Project | 2000 req/日 | 10000 req/日 | 24時間 |
| AFLOW | 制限なし | - | - |
| OQMD | 100 req/時 | - | 1時間 |
| JARVIS | 制限なし | - | - |

#### 2.9.2 キャッシング戦略

**ローカルキャッシュ（推奨）**:
```python
import pickle
import os
from datetime import datetime, timedelta

class CachedMPRester:
    def __init__(self, api_key, cache_dir='./cache', cache_ttl=24):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_ttl = timedelta(hours=cache_ttl)
        os.makedirs(cache_dir, exist_ok=True)

    def search_with_cache(self, query_hash, search_func):
        cache_file = os.path.join(self.cache_dir, f"{query_hash}.pkl")

        # キャッシュチェック
        if os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < self.cache_ttl:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # APIから取得
        result = search_func()

        # キャッシュ保存
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result
```

**SQLiteキャッシュ（大規模データ）**:
```python
import sqlite3
import json
from datetime import datetime

class SQLiteCachedAPI:
    def __init__(self, db_path='api_cache.db'):
        self.conn = sqlite3.connect(db_path)
        self.init_cache_table()

    def init_cache_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS api_cache (
                query_key TEXT PRIMARY KEY,
                response_data TEXT,
                cached_at TIMESTAMP,
                expires_at TIMESTAMP
            )
        ''')
        self.conn.commit()

    def get_cached(self, query_key):
        cursor = self.conn.execute(
            'SELECT response_data, expires_at FROM api_cache WHERE query_key = ?',
            (query_key,)
        )
        row = cursor.fetchone()
        if row and datetime.fromisoformat(row[1]) > datetime.now():
            return json.loads(row[0])
        return None

    def set_cache(self, query_key, data, ttl_hours=24):
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        self.conn.execute('''
            INSERT OR REPLACE INTO api_cache
            (query_key, response_data, cached_at, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (query_key, json.dumps(data), datetime.now(), expires_at))
        self.conn.commit()
```

#### 2.9.3 認証とセキュリティ

**環境変数での管理**:
```bash
# .env ファイル
MP_API_KEY=your_materials_project_key
OQMD_API_KEY=your_oqmd_key  # 将来的に必要な場合

# .gitignore に追加
.env
api_keys.json
*.pkl
cache/
```

```python
from dotenv import load_dotenv
import os

load_dotenv()
MP_API_KEY = os.getenv('MP_API_KEY')

if not MP_API_KEY:
    raise ValueError("MP_API_KEY not found in environment variables")
```

---

### 3. Practical Pitfalls (All Chapters)

#### Chapter 1: よくある落とし穴と対処法

**1.1 APIキーの漏洩**
- **問題**: GitHubにAPIキーをコミット
- **対策**: `.gitignore`に`.env`を追加、環境変数で管理
- **検出方法**: `git grep "sk-ant-" --all-match`

**1.2 データベース選択ミス**
- **問題**: 光学特性が必要なのにMaterials Projectのみ使用
- **対策**: 用途別推奨表（1.3節）を参照
- **チェックポイント**: 必要なプロパティがAPIで取得可能か確認

**1.3 レート制限超過**
- **問題**: ループで大量リクエスト → 429 Too Many Requests
- **対策**: `time.sleep(0.5)`を挿入、バッチAPIを使用
- **監視**: リクエストカウンターを実装

```python
import time
from collections import deque
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_requests=2000, time_window=86400):  # 24時間
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def wait_if_needed(self):
        now = datetime.now()
        # 古いリクエストを削除
        while self.requests and (now - self.requests[0]).total_seconds() > self.time_window:
            self.requests.popleft()

        # レート制限チェック
        if len(self.requests) >= self.max_requests:
            wait_time = self.time_window - (now - self.requests[0]).total_seconds()
            print(f"Rate limit reached. Waiting {wait_time:.0f} seconds...")
            time.sleep(wait_time + 1)
            self.requests.clear()

        self.requests.append(now)
```

#### Chapter 2: データ取得の落とし穴

**2.1 Primitive vs Conventional Cell**
- **問題**: `get_structure_by_material_id()`はprimitive cellを返すが、解析にconventional cellが必要
- **対策**: `SpacegroupAnalyzer`で変換

```python
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

structure_primitive = mpr.get_structure_by_material_id("mp-149")
sga = SpacegroupAnalyzer(structure_primitive)
structure_conventional = sga.get_conventional_standard_structure()

print(f"Primitive: {len(structure_primitive)} atoms")
print(f"Conventional: {len(structure_conventional)} atoms")
```

**2.2 Structure vs Relaxed Structure**
- **問題**: 初期構造と緩和後構造を混同
- **対策**: `fields`で明示的に指定

```python
# 緩和後構造
docs = mpr.materials.summary.search(
    formula="TiO2",
    fields=["structure"]  # 緩和後構造
)

# vs 初期構造（通常は使用しない）
```

**2.3 データバージョンの不整合**
- **問題**: 論文執筆時と再現時でデータが変更されている
- **対策**: データバージョンを記録、ローカルキャッシュ保存

```python
import json
from datetime import datetime

# データ取得時にメタデータ記録
metadata = {
    "query": {"formula": "TiO2", "band_gap": (2.0, 3.0)},
    "timestamp": datetime.now().isoformat(),
    "mp_version": "2023.10.1",  # mp-apiバージョン
    "num_results": len(docs),
    "data_hash": hashlib.md5(str(docs).encode()).hexdigest()
}

with open("query_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

#### Chapter 3: データ統合の落とし穴

**3.1 単位系の不一致**
- **問題**: Materials ProjectはeV、AFLOWはRy（Rydberg）
- **対策**: 単位変換を明示的に実装

```python
# 単位変換関数
def convert_energy_units(value, from_unit, to_unit):
    conversions = {
        ('eV', 'Ry'): 0.073498618,
        ('Ry', 'eV'): 13.605693,
        ('eV', 'J'): 1.602176634e-19,
    }
    if from_unit == to_unit:
        return value
    return value * conversions.get((from_unit, to_unit), 1.0)
```

**3.2 化学式の表記揺れ**
- **問題**: "TiO2" vs "Ti1O2" vs "Ti4O8"
- **対策**: pymatgenのCompositionで正規化

```python
from pymatgen.core.composition import Composition

def normalize_formula(formula):
    comp = Composition(formula)
    return comp.reduced_formula  # "TiO2"
```

**3.3 欠損値の誤処理**
- **問題**: `None`と`0`を混同、NaN伝播
- **対策**: 明示的な欠損値処理

```python
import pandas as pd
import numpy as np

# 欠損値の適切な処理
df = df.replace([np.inf, -np.inf], np.nan)  # 無限大をNaNに
df = df.dropna(subset=['critical_property'])  # 重要列のNaNを削除
df['optional_property'] = df['optional_property'].fillna(method='median')
```

---

### 4. End-of-Chapter Checklists (All Chapters)

#### Chapter 1: 完了チェックリスト

**基本スキル**
- [ ] 4大データベースの特徴を説明できる
- [ ] Materials Project APIキーを取得済み
- [ ] `mp-api`と`pymatgen`をインストール済み
- [ ] 単一材料のデータを取得できる
- [ ] バンドギャップでフィルタリングできる

**API認証とクエリ**
- [ ] APIキーを環境変数で管理している
- [ ] `.gitignore`に`.env`を追加済み
- [ ] MPRester contextマネージャを正しく使用
- [ ] `fields`パラメータでデータ量を最小化
- [ ] エラーハンドリングを実装済み

**データ取得のベストプラクティス**
- [ ] レート制限を理解している（2000 req/日）
- [ ] `time.sleep()`で負荷を分散
- [ ] リトライロジック（指数バックオフ）実装
- [ ] ローカルキャッシュを活用
- [ ] データ取得日時を記録

**引用と帰属**
- [ ] 使用したデータベースの引用形式を把握
- [ ] README/論文に適切な引用を記載
- [ ] ライセンス条件（CC BY 4.0）を理解
- [ ] データ公開時のライセンス選択を検討

**次のステップ**
- [ ] 第2章に進む準備ができている
- [ ] pymatgenの基本を理解している
- [ ] 自分の研究に必要なデータベースを特定済み

---

#### Chapter 2: 完了チェックリスト

**pymatgen基礎**
- [ ] Structureオブジェクトを作成できる
- [ ] 格子定数、密度、体積を取得できる
- [ ] SpacegroupAnalyzerで対称性を解析
- [ ] primitive cellとconventional cellを変換
- [ ] CIFファイルを読み書きできる

**MPRester API操作**
- [ ] `material_id`でデータ取得
- [ ] 複数フィールドを一括取得
- [ ] 論理演算子で複雑なクエリ
- [ ] 元素指定で検索
- [ ] バッチダウンロード（1000件以上）

**データフィルタリング**
- [ ] `band_gap`範囲指定
- [ ] `num_elements`で元素数制限
- [ ] `crystal_system`で結晶系指定
- [ ] `energy_above_hull`で安定性フィルタ
- [ ] 複数条件をAND/OR結合

**構造とプロパティ取得**
- [ ] primitive/conventional/relaxed構造の違いを理解
- [ ] バンド構造データを取得
- [ ] 状態密度（DOS）を取得
- [ ] 状態図を取得
- [ ] 表面エネルギーを取得（該当する場合）

**ローカルキャッシング**
- [ ] pickleでデータキャッシュ
- [ ] SQLiteでキャッシュ（大規模データ）
- [ ] キャッシュ有効期限を設定
- [ ] キャッシュヒット率を監視
- [ ] キャッシュ削除戦略を実装

**バッチデータ処理**
- [ ] `chunk_size`と`num_chunks`を理解
- [ ] プログレスバー表示（tqdm）
- [ ] エラー時のリカバリ実装
- [ ] 部分的な結果を保存
- [ ] メモリ効率を考慮

**再現性の確保**
- [ ] クエリパラメータをJSON保存
- [ ] データ取得日時を記録
- [ ] mp-apiバージョンを記録
- [ ] 結果のハッシュ値を保存
- [ ] README/メタデータを作成

**次のステップ**
- [ ] 第3章（データ統合）に進む準備完了
- [ ] 自分のデータセットを構築できる
- [ ] データ可視化スクリプトを作成済み

---

#### Chapter 3: 完了チェックリスト

**複数データベース統合**
- [ ] Materials ProjectとAFLOWデータを取得
- [ ] 化学式をキーとした結合（join）
- [ ] outer joinで全データ保持
- [ ] データソース列を追加
- [ ] 統合DataFrameを作成

**データクリーニング**
- [ ] 重複データを検出
- [ ] `drop_duplicates()`で削除
- [ ] IQR法で外れ値検出
- [ ] データ型の統一（float, int, str）
- [ ] 単位系の統一

**欠損値処理**
- [ ] 欠損値パターンを可視化（ヒートマップ）
- [ ] 欠損率を計算
- [ ] 平均値補完を実装
- [ ] 中央値補完を実装
- [ ] KNN補完を実装（scikit-learn）
- [ ] 補完方法の選択基準を理解

**データ品質評価**
- [ ] 完全性（Completeness）を計算
- [ ] 一貫性（Consistency）をチェック
- [ ] 精度（Accuracy）を評価（複数DB比較）
- [ ] データ品質レポートを自動生成
- [ ] バリデーションルールを定義

**自動更新パイプライン**
- [ ] データ取得関数を実装
- [ ] クリーニングパイプラインを構築
- [ ] メタデータを保存（JSON）
- [ ] タイムスタンプを記録
- [ ] スケジューラー設定（cron/schedule）

**データバージョン管理**
- [ ] Git LFSをセットアップ（該当する場合）
- [ ] データファイルをバージョン管理
- [ ] 変更履歴を記録
- [ ] ロールバック方法を理解
- [ ] ブランチ戦略を決定

**次のステップ**
- [ ] 第4章（独自DB構築）に進む準備完了
- [ ] 実験データの構造化を検討
- [ ] データ公開計画を立案

---

#### Chapter 4: 完了チェックリスト

**データベース設計**
- [ ] ER図を作成
- [ ] スキーマを設計（Materials, Properties, Experiments）
- [ ] 正規化（1NF, 2NF, 3NF）を理解
- [ ] 外部キー制約を設定
- [ ] インデックス戦略を計画

**SQLite実装**
- [ ] データベースファイルを作成
- [ ] CREATE TABLEステートメント実行
- [ ] INSERT操作（単一・バッチ）
- [ ] SELECT操作（WHERE, JOIN）
- [ ] UPDATE/DELETE操作
- [ ] トランザクション管理

**CRUD操作**
- [ ] Create: データ挿入関数
- [ ] Read: クエリ関数
- [ ] Update: データ更新関数
- [ ] Delete: データ削除関数（カスケード）
- [ ] エラーハンドリング実装

**PostgreSQL/MySQL（オプション）**
- [ ] データベースサーバーをセットアップ
- [ ] psycopg2/mysql-connectorをインストール
- [ ] スキーマをPostgreSQL用に変換
- [ ] バルクインサート実装
- [ ] GINインデックス作成（全文検索）

**バックアップ戦略**
- [ ] 手動バックアップスクリプト作成
- [ ] 圧縮（gzip）実装
- [ ] 自動バックアップスケジュール設定
- [ ] 世代管理（最新N世代保持）
- [ ] 復元テスト実施
- [ ] 3-2-1ルール理解（3コピー、2メディア、1オフサイト）

**データ公開**
- [ ] Zenodoアカウント作成
- [ ] APIトークン取得
- [ ] メタデータ準備（JSON）
- [ ] データファイルアップロード
- [ ] ライセンス選択（CC BY 4.0推奨）
- [ ] DOI取得
- [ ] README作成（データ説明、引用形式）

**ドキュメンテーション**
- [ ] データベーススキーマ図
- [ ] データディクショナリ
- [ ] 使用例（Jupyter Notebook）
- [ ] ライセンスファイル
- [ ] 引用形式（BibTeX）

**次のステップ**
- [ ] シリーズ完了！
- [ ] MI入門シリーズへ進む
- [ ] 自分の研究データベースを構築
- [ ] データを論文/リポジトリで公開

---

## Implementation Notes

### Priority
1. **High**: End-of-chapter checklists (all chapters) - Immediate value for learners
2. **High**: Practical pitfalls (Chapters 1-3) - Prevents common errors
3. **Medium**: Database licensing table (Chapter 1) - Important for proper attribution
4. **Medium**: API reproducibility (Chapter 2) - Essential for research reproducibility
5. **Low**: Additional code examples - Can be added incrementally

### Placement Strategy
- Licensing table: Insert after section 1.3 (usage table)
- API reproducibility: New section 2.9 before summary
- Pitfalls: Add subsections within each chapter
- Checklists: Add before "参考文献" section in each chapter

### Formatting
All content follows the existing HTML structure with:
- Proper `<h2>`, `<h3>`, `<h4>` hierarchy
- Code blocks with `<pre><code class="language-python">`
- Tables with appropriate CSS classes
- Blockquotes for important notes

### Next Steps
1. Review this summary document
2. Approve the proposed changes
3. Implement improvements chapter-by-chapter with separate commits
4. Test all code examples for correctness
5. Update version numbers and changelog

---

**Estimated Time**: 3-4 hours for full implementation
**Files Modified**: 4 chapter files (chapter-1.html through chapter-4.html)
**New Content**: ~3000 words across all chapters
