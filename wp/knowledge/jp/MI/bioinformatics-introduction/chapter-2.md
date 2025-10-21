---
title: "第2章：配列解析と機械学習"
subtitle: "タンパク質機能予測の実践"
series: "バイオインフォマティクス入門シリーズ v1.0"
series_id: "bioinformatics-introduction"
chapter_number: 2
chapter_id: "chapter2-sequence-ml"
level: "beginner-intermediate"
difficulty: "初級〜中級"
reading_time: "25-30分"
code_examples: 9
exercises: 3
mermaid_diagrams: 2
created_at: "2025-10-17"
updated_at: "2025-10-17"
version: "1.0"
prerequisites:
  - "第1章（タンパク質構造とPDB）"
  - "Python基礎"
  - "機械学習基礎"
learning_objectives:
  - "BLAST検索を実行し結果を解釈できる"
  - "配列から特徴量を抽出できる"
  - "機械学習モデルでタンパク質機能を予測できる"
  - "酵素活性予測モデルを構築できる"
keywords:
  - "配列アライメント"
  - "BLAST"
  - "特徴量抽出"
  - "機械学習"
  - "酵素活性予測"
authors:
  - name: "Dr. Yusuke Hashimoto"
    affiliation: "Tohoku University"
    email: "yusuke.hashimoto.b8@tohoku.ac.jp"
license: "CC BY 4.0"
language: "ja"
---

# 第2章：配列解析と機械学習

**タンパク質機能予測の実践**

## 学習目標

- ✅ BLAST検索を実行し、相同タンパク質を発見できる
- ✅ アミノ酸配列から物理化学的特徴量を抽出できる
- ✅ 機械学習モデルでタンパク質の局在・機能を予測できる
- ✅ Random Forest、LightGBMを使った酵素活性予測ができる
- ✅ モデルの性能評価と改善ができる

**読了時間**: 25-30分 | **コード例**: 9個 | **演習問題**: 3問

---

## 2.1 配列アライメント

### アライメントとは

**配列アライメント**は、2つ以上の配列を並べて類似性を評価する手法です。

<div class="mermaid">
graph LR
    A[クエリ配列] --> B[アライメント<br>アルゴリズム]
    C[データベース] --> B
    B --> D[相同配列]
    D --> E[機能推定]
    D --> F[進化解析]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#ffebee
    style F fill:#fff9c4
</div>

**用途**:
- 機能未知タンパク質の機能推定
- 進化的関係の解析
- 重要な残基の同定

---

### BLAST検索

**BLAST（Basic Local Alignment Search Tool）**は、最も広く使われる配列類似性検索ツールです。

**Example 1: Biopythonを使ったBLAST検索**

```python
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import SeqIO

# クエリ配列（例: インスリン）
query_sequence = """
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN
"""

print("BLAST検索を実行中...")

# NCBIのBLASTサーバーで検索
result_handle = NCBIWWW.qblast(
    "blastp",  # タンパク質BLAST
    "nr",  # non-redundantデータベース
    query_sequence,
    hitlist_size=10  # 上位10件
)

# 結果をファイルに保存
with open("blast_results.xml", "w") as out_handle:
    out_handle.write(result_handle.read())

result_handle.close()

print("検索完了。結果を解析中...")

# 結果を解析
with open("blast_results.xml") as result_handle:
    blast_records = NCBIXML.parse(result_handle)

    for blast_record in blast_records:
        print(f"\n=== BLAST検索結果 ===")
        print(f"クエリ長: {blast_record.query_length} aa")
        print(f"ヒット数: {len(blast_record.alignments)}")

        # 上位5件を表示
        for alignment in blast_record.alignments[:5]:
            for hsp in alignment.hsps:
                print(f"\n--- ヒット ---")
                print(f"配列: {alignment.title[:60]}...")
                print(f"E-value: {hsp.expect:.2e}")
                print(f"スコア: {hsp.score}")
                print(f"同一性: {hsp.identities}/{hsp.align_length} "
                      f"({100*hsp.identities/hsp.align_length:.1f}%)")
                print(f"アライメント:")
                print(f"Query: {hsp.query[:60]}")
                print(f"       {hsp.match[:60]}")
                print(f"Sbjct: {hsp.sbjct[:60]}")
```

**出力例**:
```
=== BLAST検索結果 ===
クエリ長: 110 aa
ヒット数: 50

--- ヒット ---
配列: insulin [Homo sapiens]
E-value: 2.5e-75
スコア: 229
同一性: 110/110 (100.0%)
Align ment:
Query: MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED
       ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Sbjct: MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED
```

---

### ローカル vs グローバルアライメント

**グローバルアライメント（Needleman-Wunsch）**:
- 配列全体を比較
- 類似した長さの配列に適用

**ローカルアライメント（Smith-Waterman）**:
- 最も類似した部分領域を検出
- 長さが異なる配列に適用（BLASTで使用）

**Example 2: Biopythonでのアライメント**

```python
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

# 2つの配列
seq1 = "ACDEFGHIKLMNPQRSTVWY"
seq2 = "ACDEFGHIKLQNPQRSTVWY"  # 1文字異なる

# グローバルアライメント
alignments = pairwise2.align.globalxx(seq1, seq2)

print("=== グローバルアライメント ===")
print(format_alignment(*alignments[0]))

# ローカルアライメント
local_alignments = pairwise2.align.localxx(seq1, seq2)

print("\n=== ローカルアライメント ===")
print(format_alignment(*local_alignments[0]))

# スコアリング: match +2, mismatch -1, gap -0.5
gap_penalty = -0.5
alignments_scored = pairwise2.align.globalms(
    seq1, seq2,
    match=2,
    mismatch=-1,
    open=-0.5,
    extend=-0.1
)

print("\n=== スコアリング付きアライメント ===")
print(format_alignment(*alignments_scored[0]))
print(f"スコア: {alignments_scored[0][2]:.1f}")
```

---

## 2.2 配列からの特徴量抽出

### アミノ酸の物理化学的性質

**20種類のアミノ酸**は、異なる物理化学的性質を持ちます：

| 性質 | アミノ酸 |
|----|---------|
| 疎水性 | A, V, I, L, M, F, W, P |
| 親水性（極性） | S, T, N, Q, Y, C |
| 塩基性（正電荷） | K, R, H |
| 酸性（負電荷） | D, E |
| 芳香族 | F, Y, W |

**Example 3: アミノ酸組成の計算**

```python
from collections import Counter
import numpy as np

def calculate_aa_composition(sequence):
    """
    アミノ酸組成を計算

    Returns:
    --------
    dict: 各アミノ酸の出現頻度（%）
    """
    # 大文字に統一
    sequence = sequence.upper()

    # 20種類の標準アミノ酸
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"

    # カウント
    aa_count = Counter(sequence)

    # 頻度（%）に変換
    total = len(sequence)
    composition = {aa: 100 * aa_count.get(aa, 0) / total
                   for aa in standard_aa}

    return composition

# テスト配列
sequence = """
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFY
TPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSIC
SLYQLENYCN
"""
sequence = sequence.replace("\n", "").replace(" ", "")

comp = calculate_aa_composition(sequence)

print("=== アミノ酸組成 ===")
for aa in sorted(comp.keys(), key=lambda x: comp[x], reverse=True):
    if comp[aa] > 0:
        print(f"{aa}: {comp[aa]:.1f}%")

# 物理化学的性質の集計
hydrophobic = ["A", "V", "I", "L", "M", "F", "W", "P"]
charged = ["K", "R", "H", "D", "E"]
polar = ["S", "T", "N", "Q", "Y", "C"]

hydrophobic_pct = sum(comp[aa] for aa in hydrophobic)
charged_pct = sum(comp[aa] for aa in charged)
polar_pct = sum(comp[aa] for aa in polar)

print(f"\n疎水性アミノ酸: {hydrophobic_pct:.1f}%")
print(f"荷電アミノ酸: {charged_pct:.1f}%")
print(f"極性アミノ酸: {polar_pct:.1f}%")
```

---

### k-mer表現

**k-mer**は、長さkの連続部分配列です（自然言語処理のn-gramに相当）。

**Example 4: k-mer特徴量の抽出**

```python
from collections import Counter

def extract_kmers(sequence, k=3):
    """
    k-mer特徴量を抽出

    Parameters:
    -----------
    sequence : str
        アミノ酸配列
    k : int
        k-merの長さ

    Returns:
    --------
    dict: k-merの出現頻度
    """
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        kmers.append(kmer)

    # 頻度計算
    kmer_counts = Counter(kmers)
    total = len(kmers)

    # 頻度（%）に変換
    kmer_freq = {kmer: 100 * count / total
                 for kmer, count in kmer_counts.items()}

    return kmer_freq

# テスト
sequence = "ACDEFGHIKLMNPQRSTVWY" * 3

# 3-mer
kmers_3 = extract_kmers(sequence, k=3)

print("=== 上位10個の3-mer ===")
for kmer, freq in sorted(kmers_3.items(),
                         key=lambda x: x[1],
                         reverse=True)[:10]:
    print(f"{kmer}: {freq:.2f}%")

# 特徴ベクトルの作成
def create_kmer_vector(sequence, k=3):
    """
    k-mer特徴ベクトルを作成（機械学習用）
    """
    # 全ての可能なk-merを生成（サイズ: 20^k）
    # 実際には頻出k-merのみ使用
    kmers = extract_kmers(sequence, k)

    # ベクトル化（頻度ベース）
    vector = list(kmers.values())
    return vector

# 機械学習用ベクトル
feature_vector = create_kmer_vector(sequence, k=2)
print(f"\n特徴ベクトル次元: {len(feature_vector)}")
```

---

### 物理化学的記述子

**Example 5: 疎水性プロファイルの計算**

```python
import matplotlib.pyplot as plt
import numpy as np

# Kyte-Doolittle疎水性スケール
hydrophobicity_scale = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5,
    'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
    'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

def calculate_hydrophobicity_profile(sequence, window=9):
    """
    疎水性プロファイルを計算

    Parameters:
    -----------
    sequence : str
        アミノ酸配列
    window : int
        移動平均のウィンドウサイズ

    Returns:
    --------
    list: 各位置の疎水性スコア
    """
    profile = []

    for i in range(len(sequence) - window + 1):
        segment = sequence[i:i+window]
        # ウィンドウ内の平均疎水性
        hydrophobicity = np.mean([
            hydrophobicity_scale.get(aa, 0)
            for aa in segment
        ])
        profile.append(hydrophobicity)

    return profile

# テスト配列
sequence = """
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFY
"""
sequence = sequence.replace("\n", "").replace(" ", "")

# 疎水性プロファイル
profile = calculate_hydrophobicity_profile(sequence, window=9)

# 可視化
plt.figure(figsize=(12, 5))
plt.plot(range(len(profile)), profile, linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('位置', fontsize=12)
plt.ylabel('疎水性スコア', fontsize=12)
plt.title('Kyte-Doolittle疎水性プロファイル', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hydrophobicity_profile.png', dpi=300)
plt.show()

# 膜貫通領域の予測（疎水性が高い領域）
threshold = 1.6  # 膜貫通領域の閾値
tm_regions = []

for i, score in enumerate(profile):
    if score > threshold:
        tm_regions.append(i)

if tm_regions:
    print(f"膜貫通領域候補: {min(tm_regions)}-{max(tm_regions)}")
else:
    print("膜貫通領域は検出されませんでした")
```

---

## 2.3 機械学習による機能予測

### タンパク質の局在予測

**Example 6: 細胞内局在の予測**

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# サンプルデータの生成（実際はデータベースから取得）
def generate_sample_data(n_samples=500):
    """
    サンプルデータ生成（デモ用）
    """
    np.random.seed(42)

    data = []

    # 核（Nuclear）: 塩基性が高い
    for _ in range(n_samples // 5):
        features = {
            'hydrophobic_pct': np.random.normal(30, 5),
            'charged_pct': np.random.normal(35, 5),  # 高い
            'polar_pct': np.random.normal(25, 5),
            'aromatic_pct': np.random.normal(10, 3),
            'length': np.random.normal(300, 50),
            'localization': 'Nuclear'
        }
        data.append(features)

    # 細胞質（Cytoplasmic）
    for _ in range(n_samples // 5):
        features = {
            'hydrophobic_pct': np.random.normal(35, 5),
            'charged_pct': np.random.normal(25, 5),
            'polar_pct': np.random.normal(30, 5),
            'aromatic_pct': np.random.normal(10, 3),
            'length': np.random.normal(350, 60),
            'localization': 'Cytoplasmic'
        }
        data.append(features)

    # 膜（Membrane）: 疎水性が高い
    for _ in range(n_samples // 5):
        features = {
            'hydrophobic_pct': np.random.normal(50, 5),  # 高い
            'charged_pct': np.random.normal(15, 5),
            'polar_pct': np.random.normal(25, 5),
            'aromatic_pct': np.random.normal(10, 3),
            'length': np.random.normal(280, 40),
            'localization': 'Membrane'
        }
        data.append(features)

    # ミトコンドリア
    for _ in range(n_samples // 5):
        features = {
            'hydrophobic_pct': np.random.normal(38, 5),
            'charged_pct': np.random.normal(28, 5),
            'polar_pct': np.random.normal(24, 5),
            'aromatic_pct': np.random.normal(10, 3),
            'length': np.random.normal(320, 55),
            'localization': 'Mitochondrial'
        }
        data.append(features)

    # 分泌（Secreted）
    for _ in range(n_samples // 5):
        features = {
            'hydrophobic_pct': np.random.normal(32, 5),
            'charged_pct': np.random.normal(22, 5),
            'polar_pct': np.random.normal(36, 5),  # 高い
            'aromatic_pct': np.random.normal(10, 3),
            'length': np.random.normal(250, 45),
            'localization': 'Secreted'
        }
        data.append(features)

    return pd.DataFrame(data)

# データ生成
df = generate_sample_data(n_samples=500)

print("=== データセット概要 ===")
print(df.head())
print(f"\n局在の分布:")
print(df['localization'].value_counts())

# 特徴量とラベルを分離
X = df.drop('localization', axis=1)
y = df['localization']

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forestモデル
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 評価
print("\n=== 評価結果 ===")
print(classification_report(y_test, y_pred))

# 特徴量重要度
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== 特徴量重要度 ===")
print(feature_importance)

# 新しいタンパク質の予測
new_protein = {
    'hydrophobic_pct': 48.0,
    'charged_pct': 18.0,
    'polar_pct': 24.0,
    'aromatic_pct': 10.0,
    'length': 290.0
}

prediction = model.predict([list(new_protein.values())])
print(f"\n新しいタンパク質の予測局在: {prediction[0]}")
```

---

## 2.4 ケーススタディ：酵素活性予測

### データ収集と前処理

**Example 7: 酵素データセットの準備**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def extract_features_from_sequence(sequence):
    """
    配列から特徴量を抽出
    """
    # アミノ酸組成
    aa_count = {aa: sequence.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    aa_comp = {f"aa_{aa}": count / len(sequence)
               for aa, count in aa_count.items()}

    # 物理化学的性質
    hydrophobic = sum(sequence.count(aa) for aa in "AVILMFWP")
    charged = sum(sequence.count(aa) for aa in "KRHDE")
    polar = sum(sequence.count(aa) for aa in "STNQYC")

    features = {
        **aa_comp,
        'length': len(sequence),
        'hydrophobic_pct': 100 * hydrophobic / len(sequence),
        'charged_pct': 100 * charged / len(sequence),
        'polar_pct': 100 * polar / len(sequence),
        'molecular_weight': sum(
            aa_count.get(aa, 0) * mw
            for aa, mw in {
                'A': 89, 'C': 121, 'D': 133, 'E': 147,
                'F': 165, 'G': 75, 'H': 155, 'I': 131,
                'K': 146, 'L': 131, 'M': 149, 'N': 132,
                'P': 115, 'Q': 146, 'R': 174, 'S': 105,
                'T': 119, 'V': 117, 'W': 204, 'Y': 181
            }.items()
        )
    }

    return features

# サンプルデータ（実際はUniProtなどから取得）
enzyme_data = [
    {
        'sequence': "ACDEFGHIKLMNPQRSTVWY" * 10,
        'activity': 8.5  # log(kcat/Km)
    },
    # 実際には数百〜数千のデータ
]

print("特徴量抽出のデモ:")
features = extract_features_from_sequence(enzyme_data[0]['sequence'])
print(f"特徴量数: {len(features)}")
print(f"サンプル特徴量:")
for key, value in list(features.items())[:5]:
    print(f"  {key}: {value:.3f}")
```

---

### モデル訓練と評価

**Example 8: LightGBMによる酵素活性予測**

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# サンプルデータ生成（実際はデータベースから）
np.random.seed(42)

n_samples = 200
X = np.random.randn(n_samples, 25)  # 25特徴量
# 活性は特定の特徴に依存
y = (2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2] +
     np.random.randn(n_samples) * 0.5)

# 訓練/テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LightGBMモデル
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 評価
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"=== モデル性能 ===")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")

# 可視化
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)
plt.xlabel('実測値 (log(kcat/Km))', fontsize=12)
plt.ylabel('予測値', fontsize=12)
plt.title(f'酵素活性予測 (R²={r2:.3f})', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('enzyme_activity_prediction.png', dpi=300)
plt.show()

# クロスバリデーション
cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=5,
    scoring='r2'
)

print(f"\nCV R²: {cv_scores.mean():.3f} ± "
      f"{cv_scores.std():.3f}")
```

---

### ハイパーパラメータチューニング

**Example 9: Optunaによる最適化**

```python
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

def objective(trial):
    """
    Optunaの目的関数
    """
    # ハイパーパラメータの探索空間
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.01, 0.3
        ),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int(
            'min_child_samples', 5, 50
        ),
        'random_state': 42
    }

    # モデル
    model = lgb.LGBMRegressor(**params)

    # クロスバリデーション
    scores = cross_val_score(
        model, X_train, y_train,
        cv=3,
        scoring='r2'
    )

    return scores.mean()

# 最適化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"\n=== 最適化結果 ===")
print(f"最良R²: {study.best_value:.3f}")
print(f"最良パラメータ:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# 最良モデルで再訓練
best_model = lgb.LGBMRegressor(**study.best_params)
best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)
r2_best = r2_score(y_test, y_pred_best)

print(f"\nテストR²（最適化後）: {r2_best:.3f}")
```

---

## 2.5 本章のまとめ

### 学んだこと

1. **配列アライメント**
   - BLAST検索による相同配列探索
   - ローカル vs グローバルアライメント

2. **特徴量抽出**
   - アミノ酸組成
   - k-mer表現
   - 物理化学的記述子（疎水性など）

3. **機械学習**
   - Random Forestによる局在予測
   - LightGBMによる酵素活性予測
   - ハイパーパラメータチューニング

### 次の章へ

第3章では、**分子ドッキングと相互作用解析**を学びます。

**[第3章：分子ドッキングと相互作用解析 →](./chapter-3.html)**

---

## データライセンスと引用

### 配列データベース

#### 1. NCBI (National Center for Biotechnology Information)
- **ライセンス**: Public Domain
- **引用**: NCBI Resource Coordinators. (2018). "Database resources of the National Center for Biotechnology Information." *Nucleic Acids Research*, 46(D1), D8-D13.
- **BLAST**: https://blast.ncbi.nlm.nih.gov/
- **用途**: タンパク質配列検索、相同性解析

#### 2. UniProt
- **ライセンス**: CC BY 4.0
- **引用**: The UniProt Consortium. (2023). "UniProt: the Universal Protein Knowledgebase in 2023." *Nucleic Acids Research*, 51(D1), D523-D531.
- **アクセス**: https://www.uniprot.org/
- **用途**: アノテーション済み配列データ、酵素活性情報

#### 3. Pfam (Protein families database)
- **ライセンス**: CC0 1.0
- **引用**: Mistry, J. et al. (2021). "Pfam: The protein families database in 2021." *Nucleic Acids Research*, 49(D1), D412-D419.
- **アクセス**: https://pfam.xfam.org/
- **用途**: タンパク質ドメイン、機能分類

### ライブラリライセンス

| ライブラリ | バージョン | ライセンス | 用途 |
|--------|----------|--------|------|
| Biopython | 1.81+ | BSD-3-Clause | BLAST、配列解析 |
| scikit-learn | 1.3+ | BSD-3-Clause | 機械学習 |
| LightGBM | 4.0+ | MIT | 勾配ブースティング |
| Optuna | 3.3+ | MIT | ハイパーパラメータ最適化 |

---

## コード再現性

### 乱数シード設定

```python
# 全ての乱数を固定
import numpy as np
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# scikit-learn
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=SEED)

# LightGBM
import lightgbm as lgb
model = lgb.LGBMRegressor(random_state=SEED)
```

### BLAST検索の再現性確保

```python
from Bio.Blast import NCBIWWW
from datetime import datetime

# タイムスタンプ記録
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# BLAST実行
result_handle = NCBIWWW.qblast(
    "blastp",
    "nr",
    query_sequence,
    hitlist_size=10
)

# 結果保存（再現可能）
filename = f"blast_results_{timestamp.replace(' ', '_')}.xml"
with open(filename, "w") as f:
    f.write(result_handle.read())

print(f"BLAST results saved: {filename}")
```

---

## よくある落とし穴と対処法

### 1. BLAST E-valueの誤解釈

**問題**: E-valueが低ければ全て有意と誤解

**NG**:
```python
if hsp.expect < 0.05:  # 統計的有意性と混同
    accept_hit()
```

**OK**:
```python
# E-value閾値は用途により変える
if application == 'homology_search':
    threshold = 1e-10  # 厳しい閾値
elif application == 'remote_homology':
    threshold = 1e-3   # 緩い閾値

if hsp.expect < threshold and hsp.identities/hsp.align_length > 0.3:
    # 配列同一性も考慮
    accept_hit()
```

### 2. 配列アライメントのギャップペナルティ設定ミス

**問題**: デフォルト値のまま使用

**NG**:
```python
# デフォルト値は必ずしも最適ではない
alignments = pairwise2.align.globalxx(seq1, seq2)
```

**OK**:
```python
# タンパク質の種類に応じて調整
if protein_type == 'short_peptide':
    gap_open = -10
    gap_extend = -0.5
elif protein_type == 'structured':
    gap_open = -5
    gap_extend = -2

alignments = pairwise2.align.globalms(
    seq1, seq2,
    match=2,
    mismatch=-1,
    open=gap_open,
    extend=gap_extend
)
```

### 3. k-mer表現の次元爆発

**問題**: k値が大きすぎてメモリ不足

**NG**:
```python
k = 5  # 20^5 = 3,200,000次元！
kmers = extract_kmers(sequence, k=k)
```

**OK**:
```python
# k=2 or 3が実用的
k = 3  # 20^3 = 8,000次元
kmers = extract_kmers(sequence, k=k)

# または頻出k-merのみ使用
from collections import Counter
kmer_counts = Counter(kmers.keys())
top_kmers = [k for k, _ in kmer_counts.most_common(1000)]
```

### 4. 機械学習モデルの過学習

**問題**: 訓練データに過度にフィット

**NG**:
```python
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None  # 無制限の深さ
)
```

**OK**:
```python
# 正則化パラメータを設定
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# クロスバリデーションで検証
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

### 5. 配列長の違いを無視

**問題**: 異なる長さの配列を直接比較

**NG**:
```python
# 絶対値で比較（長さが異なる）
if feature1 > feature2:
    select_protein1()
```

**OK**:
```python
# 正規化してから比較
feature1_norm = feature1 / len(sequence1)
feature2_norm = feature2 / len(sequence2)

if feature1_norm > feature2_norm:
    select_protein1()
```

### 6. 不均衡データセット

**問題**: クラス間のサンプル数が大きく異なる

**NG**:
```python
# 不均衡なまま学習
model.fit(X_train, y_train)
```

**OK**:
```python
from sklearn.utils.class_weight import compute_class_weight

# クラス重みを計算
classes = np.unique(y_train)
class_weights = compute_class_weight(
    'balanced',
    classes=classes,
    y=y_train
)

# 重み付き学習
model = RandomForestClassifier(
    class_weight=dict(zip(classes, class_weights)),
    random_state=42
)
model.fit(X_train, y_train)
```

---

## 品質チェックリスト

### データ取得段階
- [ ] 配列がFASTA形式で正しく保存されている
- [ ] 非標準アミノ酸（X, B, Z など）の処理方針を決定
- [ ] 配列長の分布を確認（極端に短い/長い配列の除外）
- [ ] 重複配列の除去

### BLAST検索段階
- [ ] E-value閾値が用途に適切（1e-3 〜 1e-10）
- [ ] クエリ配列長が十分（最低20残基）
- [ ] データベース選択が適切（nr, swissprot等）
- [ ] ヒット数が十分（最低3-5件）

### 特徴量抽出段階
- [ ] アミノ酸組成の合計が100%
- [ ] k-mer の k 値が適切（2-3）
- [ ] 疎水性スケールが標準的なもの（Kyte-Doolittle等）
- [ ] 特徴量に欠損値がない

### 機械学習段階
- [ ] 訓練/テストデータの分割が層化サンプリング
- [ ] クロスバリデーション実施（k=5 or 10）
- [ ] 特徴量スケーリング実施（StandardScaler等）
- [ ] モデル性能指標が複数（Accuracy, F1, AUC等）

### 酵素活性予測特有
- [ ] 活性値の単位が明確（kcat/Km, IC50等）
- [ ] 測定条件（pH, 温度）が統一
- [ ] 外れ値の検出と除去
- [ ] 実験誤差の考慮

---

## 参考文献

1. Altschul, S. F. et al. (1990). "Basic local alignment search tool."
   *Journal of Molecular Biology*, 215(3), 403-410.

2. Kyte, J. & Doolittle, R. F. (1982). "A simple method for displaying
   the hydropathic character of a protein." *Journal of Molecular Biology*,
   157(1), 105-132.

---

## ナビゲーション

**[← 第1章](./chapter-1.html)** | **[第3章 →](./chapter-3.html)** | **[目次](./index.html)**
