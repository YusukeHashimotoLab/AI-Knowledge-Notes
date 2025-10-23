# Quality Enhancement Appendix for Chapters 2-5
# ハイスループット計算入門シリーズ 品質向上付録

このファイルには、第2章〜第5章に追加すべき共通の品質向上セクションが含まれています。

---

## Chapter 2: DFT計算の自動化 - 追加セクション

### データライセンスと引用

**使用ソフトウェアのライセンス**:

| ソフトウェア | ライセンス | 引用要件 | 入手方法 |
|------------|----------|---------|---------|
| ASE | LGPL 2.1+ | 論文引用推奨 | pip install ase |
| pymatgen | MIT | 論文引用必須 | pip install pymatgen |
| VASP | 商用 | ライセンス購入必須 | https://www.vasp.at/ |
| Quantum ESPRESSO | GPL v2 | 論文引用推奨 | https://www.quantum-espresso.org/ |

**必須引用文献**:

1. **ASE**: Larsen, A. H., et al. (2017). "The atomic simulation environment—a Python library for working with atoms." *Journal of Physics: Condensed Matter*, 29(27), 273002.

2. **pymatgen**: Ong, S. P., et al. (2013). "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis." *Computational Materials Science*, 68, 314-319.

3. **VASP**: Kresse, G., & Furthmüller, J. (1996). "Efficient iterative schemes for ab initio total-energy calculations using a plane-wave basis set." *Physical Review B*, 54(16), 11169.

### 実践的な落とし穴（DFT自動化特有）

#### 落とし穴1: POTCARファイルの管理ミス

**問題**: 異なる擬ポテンシャルが混在してデータの一貫性が失われる

**症状**:
- 同じ元素で異なるPOTCAR（PBE vs LDA）を使用
- 材料間の比較が無意味になる
- 論文レビューアから指摘を受ける

**解決策**:
```python
import hashlib
import json

def verify_potcar_consistency(potcar_directory):
    """
    POTCAR ファイルの一貫性を検証
    """
    potcar_registry = {}

    for element in ['Li', 'Co', 'O', 'Ni', 'Mn']:
        potcar_path = f"{potcar_directory}/POTCAR.{element}"

        # ファイルのMD5ハッシュを計算
        with open(potcar_path, 'rb') as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()

        potcar_registry[element] = {
            'path': potcar_path,
            'md5': md5_hash,
            'type': extract_potcar_type(potcar_path)  # PBE/LDA/PAW等
        }

    # レジストリを保存
    with open('potcar_registry.json', 'w') as f:
        json.dump(potcar_registry, f, indent=2)

    print("POTCAR レジストリを作成しました")
    return potcar_registry

def extract_potcar_type(potcar_path):
    """POTCARタイプを抽出"""
    with open(potcar_path, 'r') as f:
        first_line = f.readline()
        if 'PAW_PBE' in first_line:
            return 'PAW_PBE'
        elif 'PAW_LDA' in first_line:
            return 'PAW_LDA'
        else:
            return 'UNKNOWN'
```

**教訓**: すべての計算で同一のPOTCARセットを使用し、MD5ハッシュで管理

#### 落とし穴2: k-point収束テストの省略

**問題**: k-point密度が不十分で結果が不正確

**症状**:
- 1材料だけテストして全材料に適用
- 構造サイズが変わってもk-point数固定
- エネルギーが5 meV/atomレベルで変動

**解決策**:
```python
def kpoint_convergence_test(structure, kpt_densities=[500, 1000, 1500, 2000]):
    """
    k-point 収束テストを自動実行

    Parameters:
    -----------
    structure : pymatgen.Structure
        テストする構造
    kpt_densities : list
        テストするk-point密度

    Returns:
    --------
    optimal_density : int
        最適なk-point密度
    """
    energies = []

    for density in kpt_densities:
        # k-point設定
        kpts = Kpoints.automatic_density(structure, density)

        # VASP計算実行
        energy = run_vasp_calculation(structure, kpts)
        energies.append(energy)

        print(f"密度 {density}: エネルギー = {energy:.6f} eV/atom")

    # 収束判定（前回との差が1 meV/atom以下）
    for i in range(1, len(energies)):
        delta = abs(energies[i] - energies[i-1])
        if delta < 0.001:  # 1 meV/atom
            optimal_density = kpt_densities[i-1]
            print(f"収束: 最適密度 = {optimal_density}")
            return optimal_density

    print("警告: 収束しませんでした")
    return kpt_densities[-1]
```

**教訓**: 材料系ごとにk-point収束テストを実行する

#### 落とし穴3: エラーハンドリングの不足

**問題**: 計算失敗時に無限ループまたは停止

**症状**:
- VASPが"ZBRENT: fatal error"で停止
- 自動リスタートスクリプトが同じエラーで無限ループ
- ジョブキューを占有

**解決策**:
```python
class VASPErrorHandler:
    """
    VASPエラーを分類して適切に対処
    """

    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.retry_count = 0

    def handle_error(self, error_type, directory):
        """
        エラータイプに応じて対処

        Returns:
        --------
        action : str
            'retry', 'modify', 'skip'のいずれか
        """
        self.retry_count += 1

        if self.retry_count > self.max_retries:
            print(f"最大リトライ回数を超過: {directory}")
            return 'skip'

        # エラータイプごとの対処
        if error_type == 'ZBRENT':
            # ISMEAR を変更
            modify_incar(directory, {'ISMEAR': 0, 'SIGMA': 0.05})
            return 'retry'

        elif error_type == 'EDDDAV':
            # ALGO を変更
            modify_incar(directory, {'ALGO': 'Normal'})
            return 'retry'

        elif error_type == 'RHOSYG':
            # 対称性を緩和
            modify_incar(directory, {'ISYM': 0})
            return 'retry'

        elif error_type == 'DENTET':
            # Tetrahedron法を無効化
            modify_incar(directory, {'ISMEAR': 0})
            return 'retry'

        else:
            print(f"未知のエラー: {error_type}")
            return 'skip'

# 使用例
handler = VASPErrorHandler(max_retries=3)

for attempt in range(5):
    try:
        run_vasp(directory)
        break
    except VASPError as e:
        action = handler.handle_error(e.error_type, directory)

        if action == 'skip':
            log_failed_calculation(directory, e.error_type)
            break
        elif action == 'retry':
            print(f"リトライ {attempt+1}")
            continue
```

**教訓**: エラータイプを分類し、各エラーに適した対処法を実装する

### 品質チェックリスト（DFT自動化）

#### 計算開始前

**ソフトウェア設定**
- [ ] ASE/pymatgenのバージョンを記録した
- [ ] VASPのバージョンとコンパイルオプションを確認した
- [ ] POTCARの整合性を検証した（MD5ハッシュ）
- [ ] k-point収束テストを実施した

**入力生成**
- [ ] INCARパラメータを文書化した
- [ ] K-point密度の根拠を明示した
- [ ] エネルギーカットオフの妥当性を確認した
- [ ] 入力生成スクリプトをバージョン管理した

**エラー対策**
- [ ] エラーハンドラーを実装した
- [ ] 最大リトライ回数を設定した
- [ ] 失敗ログの出力先を決めた
- [ ] ディスク容量を確認した

#### 計算完了後

**結果検証**
- [ ] すべての計算が収束したか確認した
- [ ] エネルギーの妥当性を検証した（-100 ~ 0 eV/atom）
- [ ] 力の収束を確認した（< 0.05 eV/Å）
- [ ] バンドギャップが妥当な範囲か確認した

**データ保存**
- [ ] OUTCAR, CONTCAR, vasprun.xmlを保存した
- [ ] 計算設定（INCAR, KPOINTS, POTCAR）を記録した
- [ ] メタデータ（日時、ホスト名）を保存した
- [ ] データベースに登録した

### コードの再現性仕様

**必須ライブラリバージョン**:
```bash
# Python 3.10以上
ase==3.22.1
pymatgen==2023.10.11
numpy==1.24.0
scipy==1.10.0
```

**VASP設定の記録例**:
```python
calculation_metadata = {
    'software': {
        'vasp_version': '6.3.0',
        'compilation': 'Intel 2021.2 + MKL',
        'mpi': 'Intel MPI 2021.2'
    },
    'pseudopotentials': {
        'Li': 'PAW_PBE Li 17Jan2003',
        'Co': 'PAW_PBE Co 06Sep2000',
        'O': 'PAW_PBE O 08Apr2002',
        'md5_registry': 'potcar_md5_hashes.json'
    },
    'convergence_criteria': {
        'energy': 1e-5,  # eV
        'force': 0.01,   # eV/Å
        'kpoint_density': 1000,  # per Å³
        'encut': 520  # eV
    }
}
```

---

## Chapter 3: ジョブスケジューリングと並列化 - 追加セクション

### 実践的な落とし穴（SLURM/並列化特有）

#### 落とし穴1: ジョブスクリプトのデバッグ困難

**問題**: ジョブが失敗してもエラーメッセージが見つからない

**症状**:
- slurm-xxxxx.outが空ファイル
- エラーが標準エラーに出ているが確認していない
- ジョブが"COMPLETED"なのに結果がない

**解決策**:
```bash
#!/bin/bash
#SBATCH --job-name=vasp_debug
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00

# デバッグ情報を詳細に記録
set -x  # コマンドの実行ログを出力
set -e  # エラー時に即座に停止

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "========================"

# 環境変数の確認
echo "=== Environment ==="
module list
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "==================="

# VASPの存在確認
which vasp_std || { echo "Error: vasp_std not found"; exit 1; }

# 入力ファイルの存在確認
for file in INCAR POSCAR KPOINTS POTCAR; do
    if [ ! -f $file ]; then
        echo "Error: $file not found"
        exit 1
    fi
done

# VASP実行
echo "=== Starting VASP ==="
mpirun -np $SLURM_NTASKS vasp_std 2>&1 | tee vasp_output.log

# 終了ステータス確認
if [ $? -eq 0 ]; then
    echo "VASP completed successfully"
else
    echo "VASP failed with exit code $?"
    exit 1
fi

# 収束確認
if grep -q "reached required accuracy" OUTCAR; then
    echo "SUCCESS: Calculation converged"
    touch CONVERGED
else
    echo "WARNING: Calculation did not converge"
    touch NOT_CONVERGED
fi

echo "=== Job completed at $(date) ==="
```

**教訓**: 詳細なログ出力とエラーチェックを実装する

#### 落とし穴2: メモリ不足による計算失敗

**問題**: OOM (Out of Memory) killerに殺される

**症状**:
- ジョブが突然終了（exit code 137）
- slurm-*.errに"Killed"だけ記録
- 計算途中でノード全体が応答不能

**解決策**:
```python
def estimate_memory_requirement(structure, encut=520, kpoints=(8,8,8)):
    """
    必要メモリを見積もり

    Returns:
    --------
    memory_gb : float
        必要メモリ（GB）
    """
    n_atoms = len(structure)
    n_kpoints = kpoints[0] * kpoints[1] * kpoints[2]
    n_bands = int(n_atoms * 6)  # 経験則

    # VASPメモリ使用量の近似式
    # Memory ≈ (n_bands × n_kpoints × 複素数サイズ × 行列数) / 1e9
    memory_per_band = n_kpoints * 16  # バイト（複素数double = 16 byte）
    total_memory = memory_per_band * n_bands * 10  # 安全係数10

    memory_gb = total_memory / 1e9

    print(f"推定メモリ: {memory_gb:.2f} GB")
    print(f"  原子数: {n_atoms}")
    print(f"  k-point数: {n_kpoints}")
    print(f"  バンド数: {n_bands}")

    # 安全マージン（+50%）
    recommended_memory = memory_gb * 1.5

    return recommended_memory

# SLURMスクリプト生成時に使用
structure = Structure.from_file("POSCAR")
required_memory = estimate_memory_requirement(structure)

# SLURMスクリプトに反映
slurm_script = f"""#!/bin/bash
#SBATCH --mem={int(required_memory)}G
...
"""
```

**教訓**: メモリ要求量を計算サイズから見積もる

#### 落とし穴3: 並列効率の未測定

**問題**: 48コア使っても12コアと速度が変わらない

**症状**:
- スケーリング効率が50%以下
- コア数を増やしても時間短縮しない
- リソースの無駄遣い

**解決策**:
```python
import matplotlib.pyplot as plt
import numpy as np

def analyze_scaling_efficiency(benchmark_results):
    """
    並列効率を解析してプロット

    Parameters:
    -----------
    benchmark_results : dict
        {cores: time(seconds)}の辞書
    """
    cores = np.array(list(benchmark_results.keys()))
    times = np.array(list(benchmark_results.values()))

    # スピードアップ計算
    base_time = times[0]
    speedup = base_time / times
    ideal_speedup = cores / cores[0]

    # 並列効率
    efficiency = (speedup / (cores / cores[0])) * 100

    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # スピードアップ
    ax1.plot(cores, speedup, 'o-', label='実測', linewidth=2)
    ax1.plot(cores, ideal_speedup, '--', label='理想（線形）', linewidth=2)
    ax1.set_xlabel('コア数', fontsize=12)
    ax1.set_ylabel('スピードアップ', fontsize=12)
    ax1.set_title('並列スケーリング', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)

    # 効率
    ax2.plot(cores, efficiency, 'o-', color='green', linewidth=2)
    ax2.axhline(y=80, color='red', linestyle='--', label='80%目標')
    ax2.axhline(y=50, color='orange', linestyle='--', label='50%最低ライン')
    ax2.set_xlabel('コア数', fontsize=12)
    ax2.set_ylabel('並列効率 (%)', fontsize=12)
    ax2.set_title('並列効率', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=300)

    # 最適コア数を提案
    optimal_cores = cores[efficiency >= 80][-1] if any(efficiency >= 80) else cores[0]
    print(f"\n推奨コア数: {optimal_cores} （効率 {efficiency[cores == optimal_cores][0]:.1f}%）")

    return optimal_cores

# ベンチマーク結果例
results = {
    12: 3600,  # 12コアで3600秒
    24: 2000,  # 24コアで2000秒（効率83%）
    48: 1200,  # 48コアで1200秒（効率75%）
    96: 800,   # 96コアで800秒（効率56%）
}

optimal = analyze_scaling_efficiency(results)
```

**教訓**: ベンチマークを実施して最適コア数を決定する

### 品質チェックリスト（ジョブスケジューリング）

#### ジョブ投入前
- [ ] リソース要求量を見積もった（コア数、メモリ、時間）
- [ ] 並列効率をベンチマークした
- [ ] ジョブスクリプトをローカルでテストした
- [ ] ログ出力先を確認した

#### ジョブ実行中
- [ ] ジョブ状態を定期的に監視している（squeue）
- [ ] ログファイルを確認している
- [ ] リソース使用率を監視している（seff）
- [ ] 失敗ジョブを記録している

#### ジョブ完了後
- [ ] すべてのジョブが完了したか確認した
- [ ] 失敗ジョブのエラーログを解析した
- [ ] リソース使用効率を評価した
- [ ] 次回のために最適パラメータを記録した

---

## Chapter 4: データ管理とワークフロー - 追加セクション

### 実践的な落とし穴（ワークフロー管理特有）

#### 落とし穴1: データベース設計の失敗

**問題**: クエリが遅すぎて使い物にならない

**症状**:
- 10,000材料から検索に10分以上
- データベースサイズが100GB超
- インデックスが適切に設定されていない

**解決策**:
```python
from pymongo import MongoClient, ASCENDING, DESCENDING

def setup_optimized_database():
    """
    最適化されたデータベースを構築
    """
    client = MongoClient('localhost', 27017)
    db = client['materials']
    collection = db['calculations']

    # 複合インデックスの作成
    collection.create_index([
        ('formula', ASCENDING),
        ('properties.band_gap', ASCENDING)
    ], name='formula_bandgap_idx')

    collection.create_index([
        ('properties.energy', ASCENDING)
    ], name='energy_idx')

    collection.create_index([
        ('structure.space_group', ASCENDING)
    ], name='spacegroup_idx')

    collection.create_index([
        ('calculation_metadata.date', DESCENDING)
    ], name='date_idx')

    # テキスト検索用インデックス
    collection.create_index([
        ('formula', 'text'),
        ('tags', 'text')
    ], name='text_search_idx')

    # インデックス一覧を表示
    indexes = collection.list_indexes()
    print("作成されたインデックス:")
    for idx in indexes:
        print(f"  - {idx['name']}: {idx.get('key', {})}")

    return collection

# クエリ最適化の例
def optimized_query(collection):
    """
    最適化されたクエリ実行
    """
    # 悪い例（遅い）
    # results = list(collection.find({'properties.band_gap': {'$gt': 1.0}}))

    # 良い例（インデックス活用）
    results = collection.find(
        {'properties.band_gap': {'$gt': 1.0, '$lt': 3.0}},
        projection={'formula': 1, 'properties.band_gap': 1, '_id': 0}
    ).hint('formula_bandgap_idx').limit(1000)

    return list(results)
```

**教訓**: 頻繁に使用するクエリにインデックスを作成する

### 品質チェックリスト（データ管理）

#### データベース設計時
- [ ] スキーマを文書化した
- [ ] インデックスを適切に設定した
- [ ] バックアップ戦略を決定した
- [ ] クエリ性能をテストした

#### データ保存時
- [ ] すべてのメタデータを記録した
- [ ] データ検証を実施した（スキーマ整合性）
- [ ] バックアップを取得した
- [ ] データ圧縮を検討した

---

## Chapter 5: クラウドHPC活用 - 追加セクション

### 実践的な落とし穴（クラウドHPC特有）

#### 落とし穴1: コスト爆発

**問題**: 月末に予想外の$10,000請求

**症状**:
- インスタンスの停止忘れ
- ストレージの削除忘れ
- データ転送料の見落とし

**解決策**:
```python
import boto3
from datetime import datetime, timedelta

class AWSCostMonitor:
    """
    AWSコストを監視して警告
    """

    def __init__(self, budget_limit=1000):
        self.ce_client = boto3.client('ce', region_name='us-east-1')
        self.budget_limit = budget_limit

    def get_current_month_cost(self):
        """
        今月のコストを取得
        """
        start_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        response = self.ce_client.get_cost_and_usage(
            TimePeriod={'Start': start_date, 'End': end_date},
            Granularity='MONTHLY',
            Metrics=['UnblendedCost']
        )

        cost = float(response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount'])
        return cost

    def check_budget(self):
        """
        予算超過をチェック
        """
        current_cost = self.get_current_month_cost()
        percentage = (current_cost / self.budget_limit) * 100

        print(f"今月のコスト: ${current_cost:.2f} / ${self.budget_limit}")
        print(f"予算消費率: {percentage:.1f}%")

        if percentage > 80:
            print("⚠️  警告: 予算の80%を超過しました")
            self.send_alert(current_cost, percentage)

        return current_cost

    def send_alert(self, cost, percentage):
        """
        アラート送信（メール、Slack等）
        """
        # 実装例: SNS経由でメール送信
        sns = boto3.client('sns', region_name='us-east-1')
        sns.publish(
            TopicArn='arn:aws:sns:us-east-1:123456789012:billing-alert',
            Message=f"AWSコスト警告: ${cost:.2f} ({percentage:.1f}%)",
            Subject='AWS予算超過アラート'
        )

# 使用例（定期実行）
monitor = AWSCostMonitor(budget_limit=1000)
monitor.check_budget()
```

**教訓**: コスト監視を自動化し、アラートを設定する

### 品質チェックリスト（クラウドHPC）

#### クラスタ起動前
- [ ] 予算上限を設定した
- [ ] スポットインスタンス価格を確認した
- [ ] データ転送量を見積もった
- [ ] 自動停止設定を有効化した

#### 実行中
- [ ] コストを毎日確認している
- [ ] 不要なリソースを削除している
- [ ] スケールダウンが機能しているか確認した
- [ ] データバックアップを取得した

#### 完了後
- [ ] すべてのインスタンスを停止/削除した
- [ ] ストレージを削除または最小化した
- [ ] 最終コストを記録した
- [ ] コスト分析レポートを作成した

---

## 全章共通: 参考文献追加

### HPC関連の必須文献

1. **SLURM**: Yoo, A. B., Jette, M. A., & Grondona, M. (2003). "SLURM: Simple Linux Utility for Resource Management." In *Job scheduling strategies for parallel processing* (pp. 44-60). Springer, Berlin, Heidelberg.

2. **MPI**: Message Passing Interface Forum (2015). "MPI: A Message-Passing Interface Standard Version 3.1." https://www.mpi-forum.org/docs/

3. **Docker**: Merkel, D. (2014). "Docker: lightweight Linux containers for consistent development and deployment." *Linux Journal*, 2014(239), 2.

4. **Singularity**: Kurtzer, G. M., Sochat, V., & Bauer, M. W. (2017). "Singularity: Scientific containers for mobility of compute." *PLOS ONE*, 12(5), e0177459.

### データ管理関連

5. **MongoDB**: Chodorow, K. (2013). *MongoDB: the definitive guide: powerful and scalable data storage.* O'Reilly Media, Inc.

6. **FAIR Principles**: Wilkinson, M. D., et al. (2016). "The FAIR Guiding Principles for scientific data management and stewardship." *Scientific Data*, 3(1), 1-9.

### クラウドHPC

7. **AWS Parallel Cluster**: Amazon Web Services (2023). "AWS ParallelCluster User Guide." https://docs.aws.amazon.com/parallelcluster/

8. **Cloud Cost Optimization**: Tak, B. C., Urgaonkar, B., & Sivasubramaniam, A. (2011). "To move or not to move: The economics of cloud computing." In *Proceedings of the 3rd USENIX conference on Hot topics in cloud computing* (pp. 5-5).

---

## テンプレート: 環境記録スクリプト

すべての計算プロジェクトで使用すべき環境記録スクリプト：

```python
#!/usr/bin/env python3
"""
環境情報を完全記録するスクリプト
すべての計算プロジェクトで実行してください
"""

import json
import subprocess
import platform
from datetime import datetime
import os

def record_complete_environment(output_file='environment_record.json'):
    """
    計算環境を完全に記録

    Returns:
    --------
    env_data : dict
        環境情報の辞書
    """
    env_data = {
        'metadata': {
            'recorded_at': datetime.now().isoformat(),
            'recorder_version': '1.0'
        },
        'system': {
            'hostname': subprocess.check_output(['hostname']).decode().strip(),
            'os': platform.system(),
            'os_version': platform.release(),
            'python_version': platform.python_version(),
            'architecture': platform.machine()
        },
        'python_packages': {},
        'software_versions': {},
        'environment_variables': {},
        'git_info': {}
    }

    # Python パッケージ
    try:
        pip_freeze = subprocess.check_output(['pip', 'freeze']).decode()
        env_data['python_packages'] = {
            line.split('==')[0]: line.split('==')[1]
            for line in pip_freeze.split('\n')
            if '==' in line
        }
    except Exception as e:
        env_data['python_packages'] = {'error': str(e)}

    # VASP バージョン
    try:
        vasp_version = subprocess.check_output(['vasp_std', '--version']).decode()
        env_data['software_versions']['vasp'] = vasp_version.strip()
    except:
        env_data['software_versions']['vasp'] = 'not_found'

    # SLURM バージョン
    try:
        slurm_version = subprocess.check_output(['sinfo', '--version']).decode()
        env_data['software_versions']['slurm'] = slurm_version.strip()
    except:
        env_data['software_versions']['slurm'] = 'not_found'

    # 環境変数（重要なもののみ）
    important_vars = ['PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'SLURM_CLUSTER_NAME']
    for var in important_vars:
        env_data['environment_variables'][var] = os.environ.get(var, 'not_set')

    # Git 情報
    try:
        env_data['git_info'] = {
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
            'branch': subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip(),
            'remote': subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode().strip(),
            'status': subprocess.check_output(['git', 'status', '--short']).decode().strip()
        }
    except:
        env_data['git_info'] = {'error': 'not_a_git_repository'}

    # JSON保存
    with open(output_file, 'w') as f:
        json.dump(env_data, f, indent=2)

    print(f"環境情報を保存しました: {output_file}")
    return env_data

if __name__ == '__main__':
    record_complete_environment()
```

---

## まとめ：品質向上のポイント

### データライセンスと引用
- すべての使用データセット/ソフトウェアのライセンスを明示
- 論文引用方法を具体的に提示
- 商用利用時の注意事項を記載

### 実践的な落とし穴
- 各章で5個以上の具体的な問題と解決策を提示
- 症状・原因・解決策・教訓の4点セットで記述
- 実行可能なコード例を含める

### 品質チェックリスト
- 計算開始前/実行中/完了後の3段階で整理
- 各項目をチェックボックス形式で提示
- 具体的で実行可能な項目のみ含める

### コードの再現性
- 必須ソフトウェアのバージョンを明示
- 動作確認済み環境をリスト化
- インストールスクリプトを提供
- トラブルシューティングガイドを含める

### 参考文献
- 必須文献と推奨文献を分類
- DOIリンクを必ず含める
- オンラインリソースのURLを提供

---

**このファイルの使用方法**:
各章の最後に、該当する章の追加セクションをコピー&ペーストしてください。
