---
title: "第3章: 材料探索環境の構築"
chapter: 3
series: "reinforcement-learning-introduction"
learning_time: "25-30分"
estimated_words: 4200
code_examples: 7
exercises: 3
keywords:
  - "OpenAI Gym"
  - "カスタム環境"
  - "状態空間設計"
  - "報酬設計"
  - "DFT統合"
created_at: "2025-10-17"
updated_at: "2025-10-19"
version: "2.0"

# データソース
data_sources:
  - name: "Matminer"
    license: "BSD-3-Clause"
    url: "https://github.com/hackingmaterials/matminer"
    citation: "Ward et al. (2018), Computational Materials Science, 152, 60-69"
    usage: "材料記述子（Magpie, SOAP等）の生成"

  - name: "Materials Project API"
    license: "CC BY 4.0 (academic use)"
    url: "https://materialsproject.org"
    citation: "Jain et al. (2013), APL Materials, 1(1), 011002"
    usage: "バンドギャップ、形成エネルギーデータ"

  - name: "ASE (Atomic Simulation Environment)"
    license: "LGPL-2.1"
    url: "https://wiki.fysik.dtu.dk/ase/"
    citation: "Hjorth Larsen et al. (2017), J. Phys.: Condens. Matter 29, 273002"
    usage: "DFT計算インターフェース"

# 実践的な落とし穴
practical_pitfalls:
  - category: "状態空間設計ミス"
    description: "組成のみで多形を区別できない"
    solution: "構造記述子（SOAP, Coulomb Matrix）を追加"

  - category: "報酬スケーリング"
    description: "バンドギャップ（0-5 eV）と形成エネルギー（-5-0 eV）のスケール不一致"
    solution: "正規化または重み付けで調整"

  - category: "DFT計算コスト"
    description: "1材料あたり数時間かかり探索が非効率"
    solution: "サロゲートモデルで高速予測、有望材料のみDFT"

  - category: "終了条件設定"
    description: "最大ステップ数のみだと非効率"
    solution: "目標達成 OR 改善なし10連続 OR 最大ステップ"
---

# 第3章: 材料探索環境の構築

## 学習目標

この章では、以下を習得します：

- OpenAI Gymカスタム環境の実装方法
- 材料記述子と状態空間の設計
- 効果的な報酬関数の設計原則
- DFT計算・実験装置との統合方法

---

## 3.1 OpenAI Gym環境の基礎

### Gym環境の構成要素

OpenAI Gymは、強化学習環境の標準インターフェースです。すべてのGym環境は以下のメソッドを実装します：

```python
import gym
import numpy as np

class CustomEnv(gym.Env):
    """カスタムGym環境のテンプレート"""

    def __init__(self):
        super(CustomEnv, self).__init__()

        # 行動空間と観測空間の定義（必須）
        self.action_space = gym.spaces.Discrete(4)  # 離散行動（4種類）
        self.observation_space = gym.spaces.Box(
            low=0, high=10, shape=(4,), dtype=np.float32
        )  # 連続状態（4次元、範囲 [0, 10]）

    def reset(self):
        """環境を初期状態にリセット

        Returns:
            observation: 初期状態
        """
        self.state = np.random.uniform(0, 10, 4).astype(np.float32)
        return self.state

    def step(self, action):
        """行動を実行し、環境を1ステップ進める

        Args:
            action: 実行する行動

        Returns:
            observation: 次の状態
            reward: 報酬
            done: エピソード終了フラグ
            info: 追加情報（辞書）
        """
        # 行動に応じて状態を更新
        self.state = self._update_state(action)

        # 報酬を計算
        reward = self._compute_reward()

        # 終了条件をチェック
        done = self._is_done()

        # 追加情報
        info = {'distance': self._compute_distance()}

        return self.state, reward, done, info

    def render(self, mode='human'):
        """環境を可視化（オプション）"""
        print(f"Current state: {self.state}")

    def _update_state(self, action):
        """状態更新ロジック"""
        # 実装は環境による
        pass

    def _compute_reward(self):
        """報酬計算ロジック"""
        pass

    def _is_done(self):
        """終了条件チェック"""
        pass

    def _compute_distance(self):
        """追加情報の計算"""
        pass
```

### 行動空間と観測空間の定義

Gymは多様な空間タイプをサポート：

```python
from gym import spaces

# 離散行動（整数 0, 1, 2, 3）
action_space = spaces.Discrete(4)

# 連続行動（実数ベクトル [-1, 1]^3）
action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

# 辞書形式（複数の入力）
observation_space = spaces.Dict({
    'composition': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
    'temperature': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
    'pressure': spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
})

# タプル形式
action_space = spaces.Tuple((
    spaces.Discrete(5),      # 元素選択
    spaces.Box(low=0, high=1, shape=(1,))  # 組成比率
))

# マルチバイナリ（複数のバイナリ選択）
action_space = spaces.MultiBinary(10)  # 10個の元素をON/OFF
```

---

## 3.2 材料記述子と状態空間の設計

### 材料記述子の選択

状態空間は、**材料の特性を数値ベクトルで表現**したものです。効果的な記述子の選択が重要です。

#### 1. 組成ベース記述子

**元素割合**:
```python
# 例: Li2MnO3の組成ベクトル
composition = {
    'Li': 2/6,   # 33.3%
    'Mn': 1/6,   # 16.7%
    'O': 3/6     # 50.0%
}

# 周期表全体のベクトル（118次元）
state = np.zeros(118)
state[2] = 0.333   # Li (原子番号3)
state[24] = 0.167  # Mn (原子番号25)
state[7] = 0.500   # O (原子番号8)
```

**Magpie記述子**（Ward et al., 2016）:
```python
from matminer.featurizers.composition import ElementProperty

featurizer = ElementProperty.from_preset("magpie")
# 組成から132次元の記述子を生成
# - 平均原子番号、平均電気陰性度、平均イオン半径など
composition = "Li2MnO3"
features = featurizer.featurize(Composition(composition))
```

#### 2. 構造ベース記述子

**格子定数**:
```python
# 結晶格子
state = np.array([
    a, b, c,           # 格子定数
    alpha, beta, gamma # 角度
])
```

**Smooth Overlap of Atomic Positions (SOAP)**:
```python
from dscribe.descriptors import SOAP
from ase import Atoms

# 原子構造から記述子生成
atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
soap = SOAP(species=['H', 'O'], rcut=5.0, nmax=8, lmax=6)
state = soap.create(atoms)  # 高次元ベクトル
```

#### 3. プロセスパラメータ

**合成条件**:
```python
# 合成プロセスの状態
state = np.array([
    temperature,      # 温度 [K]
    pressure,         # 圧力 [Pa]
    time,             # 時間 [s]
    heating_rate,     # 昇温速度 [K/min]
    atmosphere_O2     # 酸素分圧 [Pa]
])
```

### 実例: バンドギャップ探索環境

```python
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

class BandgapDiscoveryEnv(gym.Env):
    """バンドギャップ最適化環境

    目標: 特定のバンドギャップ（例: 3.0 eV）を持つ材料を発見
    """

    def __init__(self, target_bandgap=3.0, element_pool=None):
        super(BandgapDiscoveryEnv, self).__init__()

        self.target_bandgap = target_bandgap

        # 使用可能な元素（デフォルト: 典型的な半導体元素）
        if element_pool is None:
            self.element_pool = ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W',
                                  'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
                                  'As', 'Se', 'Sr', 'Y', 'In', 'Sn', 'Sb', 'Te', 'O', 'S', 'N']
        else:
            self.element_pool = element_pool

        self.n_elements = len(self.element_pool)

        # 行動空間: 3元素を選択 + 各元素の比率
        # 簡略化: 3元素の離散選択（組み合わせ）
        self.action_space = gym.spaces.MultiDiscrete([self.n_elements] * 3)

        # 状態空間: Magpie記述子（132次元）
        self.featurizer = ElementProperty.from_preset("magpie")
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(132,), dtype=np.float32
        )

        # 履歴（試した組成）
        self.history = []
        self.current_composition = None

    def reset(self):
        """ランダムな初期組成"""
        self.history = []
        action = self.action_space.sample()
        self.current_composition = self._action_to_composition(action)
        return self._get_state()

    def step(self, action):
        """新しい材料組成を試す"""
        self.current_composition = self._action_to_composition(action)

        # 状態（記述子）
        state = self._get_state()

        # バンドギャップを予測（サロゲートモデル or DFT）
        predicted_bandgap = self._predict_bandgap(self.current_composition)

        # 報酬: 目標との差の負の値
        error = abs(predicted_bandgap - self.target_bandgap)
        reward = -error

        # ボーナス報酬（目標に近い場合）
        if error < 0.1:
            reward += 10.0  # 非常に近い

        # 履歴に追加
        self.history.append({
            'composition': self.current_composition,
            'bandgap': predicted_bandgap,
            'reward': reward
        })

        # 終了条件: 目標に到達 or 最大ステップ数
        done = error < 0.05 or len(self.history) >= 100

        info = {
            'composition': self.current_composition,
            'predicted_bandgap': predicted_bandgap,
            'error': error
        }

        return state, reward, done, info

    def _action_to_composition(self, action):
        """行動を組成文字列に変換

        Args:
            action: [elem1_idx, elem2_idx, elem3_idx]

        Returns:
            組成文字列（例: "TiO2"）
        """
        elements = [self.element_pool[idx] for idx in action]

        # 重複除去
        unique_elements = list(set(elements))

        # 簡略化: 等量混合
        if len(unique_elements) == 1:
            comp_str = unique_elements[0]
        elif len(unique_elements) == 2:
            comp_str = f"{unique_elements[0]}{unique_elements[1]}"
        else:
            comp_str = f"{unique_elements[0]}{unique_elements[1]}{unique_elements[2]}"

        return comp_str

    def _get_state(self):
        """現在の組成から記述子を生成"""
        try:
            comp = Composition(self.current_composition)
            features = self.featurizer.featurize(comp)
            return np.array(features, dtype=np.float32)
        except:
            # 無効な組成の場合、ゼロベクトル
            return np.zeros(132, dtype=np.float32)

    def _predict_bandgap(self, composition):
        """バンドギャップを予測

        実際には:
        - 機械学習モデル（事前学習済み）
        - DFT計算（pymatgen + VASP）
        - データベース検索（Materials Project）

        ここでは簡易的なルールベース
        """
        try:
            comp = Composition(composition)

            # 簡易ルール: 酸素を含む化合物はバンドギャップが大きい傾向
            if 'O' in comp:
                base_gap = 2.5
            elif 'S' in comp:
                base_gap = 1.8
            elif 'N' in comp:
                base_gap = 2.0
            else:
                base_gap = 1.0

            # 金属元素の影響
            metals = ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta']
            for metal in metals:
                if metal in comp:
                    base_gap += 0.5

            # ランダムノイズ（実験誤差）
            noise = np.random.normal(0, 0.2)
            return max(0, base_gap + noise)

        except:
            return 0.0

    def render(self, mode='human'):
        print(f"Current composition: {self.current_composition}")
        if self.history:
            last = self.history[-1]
            print(f"Predicted bandgap: {last['bandgap']:.2f} eV")
            print(f"Target: {self.target_bandgap:.2f} eV")
            print(f"Reward: {last['reward']:.2f}")


# 環境のテスト
env = BandgapDiscoveryEnv(target_bandgap=3.0)

state = env.reset()
print(f"初期状態: {state.shape}")

for step in range(10):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    print(f"\nStep {step+1}:")
    print(f"  組成: {info['composition']}")
    print(f"  予測バンドギャップ: {info['predicted_bandgap']:.2f} eV")
    print(f"  報酬: {reward:.2f}")

    if done:
        print("目標到達！")
        break
```

**出力例**:
```
初期状態: (132,)

Step 1:
  組成: TiO
  予測バンドギャップ: 3.12 eV
  報酬: -0.12

Step 2:
  組成: ZrO
  予測バンドギャップ: 2.95 eV
  報酬: -0.05
目標到達！
```

---

## 3.3 効果的な報酬関数の設計

### 報酬設計の原則

報酬関数は、**エージェントが何を最適化すべきかを定義**します。不適切な報酬は、望まない行動や学習失敗を引き起こします。

#### 原則1: 明確な目標

**悪い例**:
```python
# 曖昧な報酬
reward = 1 if 'good_material' else 0  # "good"の定義が不明確
```

**良い例**:
```python
# 明確な目標（バンドギャップ）
target = 3.0
predicted = 2.8
reward = -abs(predicted - target)  # 目標との距離
```

#### 原則2: スケーリング

報酬の範囲を適切に設定：

**悪い例**:
```python
# 報酬が極端に大きい
reward = 1e10 if success else -1e10  # 学習が不安定
```

**良い例**:
```python
# [-1, 1]程度に正規化
reward = -error / max_error  # error ∈ [0, max_error]
```

#### 原則3: シェイピング（中間報酬）

疎報酬を密報酬に変換：

**疎報酬（学習が困難）**:
```python
reward = 1.0 if distance < 0.1 else 0.0
```

**密報酬（学習が容易）**:
```python
# 距離に応じた連続的な報酬
reward = -distance

# さらに階層的な報酬
if distance < 0.5:
    reward += 5.0  # 近い
if distance < 0.1:
    reward += 10.0  # 非常に近い
```

#### 原則4: 多目的最適化

複数の目標を重み付け：

```python
# バンドギャップと安定性の両方を最適化
bandgap_error = abs(predicted_bandgap - target_bandgap)
stability = formation_energy  # 負の値が安定

# 重み付き報酬
w1, w2 = 0.7, 0.3
reward = -w1 * bandgap_error - w2 * max(0, stability)
```

### 報酬設計の実例

#### 例1: 触媒活性最大化

```python
class CatalystOptimizationEnv(gym.Env):
    """触媒活性を最大化する環境"""

    def _compute_reward(self, activity, selectivity, stability):
        """多目的報酬

        Args:
            activity: 触媒活性（高いほど良い）
            selectivity: 選択性（目的生成物への選択性、高いほど良い）
            stability: 安定性（負の形成エネルギー、低いほど安定）

        Returns:
            総合報酬
        """
        # 各指標を正規化 [0, 1]
        activity_norm = activity / 100.0  # 仮に最大100
        selectivity_norm = selectivity  # 既に [0, 1]
        stability_norm = -stability / 5.0  # 仮に最大-5 eV

        # 重み付き和（活性を重視）
        weights = {'activity': 0.5, 'selectivity': 0.3, 'stability': 0.2}
        reward = (weights['activity'] * activity_norm +
                  weights['selectivity'] * selectivity_norm +
                  weights['stability'] * stability_norm)

        # ペナルティ: 不安定な材料
        if stability > 0:  # 正の形成エネルギー（不安定）
            reward -= 1.0

        return reward
```

#### 例2: 合成コスト制約

```python
def reward_with_cost_constraint(self, performance, synthesis_cost, max_cost=1000):
    """コスト制約付き報酬

    Args:
        performance: 材料性能
        synthesis_cost: 合成コスト [USD/kg]
        max_cost: コスト上限

    Returns:
        報酬
    """
    # 性能に基づく基本報酬
    base_reward = performance

    # コスト制約違反のペナルティ
    if synthesis_cost > max_cost:
        penalty = (synthesis_cost - max_cost) / max_cost
        base_reward -= 10.0 * penalty

    # コストが低いほどボーナス
    cost_bonus = max(0, (max_cost - synthesis_cost) / max_cost)
    base_reward += 2.0 * cost_bonus

    return base_reward
```

---

## 3.4 DFT計算との統合

### Materials Projectからのデータ取得

実際の材料特性を取得し、報酬に使用：

```python
from mp_api.client import MPRester
import os

class MPIntegratedEnv(gym.Env):
    """Materials Project統合環境"""

    def __init__(self, mp_api_key=None):
        super(MPIntegratedEnv, self).__init__()

        # Materials Project APIキー
        if mp_api_key is None:
            mp_api_key = os.getenv("MP_API_KEY")

        self.mpr = MPRester(mp_api_key)

        # ... (環境設定) ...

    def _get_bandgap_from_mp(self, composition):
        """Materials Projectからバンドギャップを取得

        Args:
            composition: 組成（例: "TiO2"）

        Returns:
            バンドギャップ [eV]（データがない場合はNone）
        """
        try:
            # 組成で検索
            docs = self.mpr.materials.summary.search(
                formula=composition,
                fields=["material_id", "band_gap", "formation_energy_per_atom"]
            )

            if docs:
                # 最も安定な構造（形成エネルギーが最小）を選択
                stable_doc = min(docs, key=lambda x: x.formation_energy_per_atom)
                return stable_doc.band_gap
            else:
                return None

        except Exception as e:
            print(f"Materials Project検索エラー: {e}")
            return None

    def step(self, action):
        composition = self._action_to_composition(action)

        # Materials Projectからデータ取得
        bandgap = self._get_bandgap_from_mp(composition)

        if bandgap is not None:
            # 実データで報酬計算
            error = abs(bandgap - self.target_bandgap)
            reward = -error
        else:
            # データがない場合、予測モデルを使用 or ペナルティ
            reward = -10.0  # 未知の材料へのペナルティ

        # ... (状態、終了条件など) ...

        return state, reward, done, info
```

**注意**: Materials Projectへの大量リクエストは避け、ローカルキャッシュを活用してください。

### ASEによるDFT計算統合（高度）

```python
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.optimize import BFGS

class DFTIntegratedEnv(gym.Env):
    """DFT計算統合環境（計算コスト大）"""

    def _calculate_bandgap_dft(self, composition):
        """DFT計算でバンドギャップを取得

        警告: 非常に時間がかかる（1材料あたり数時間〜数日）
        実用的には事前計算データベースを使用

        Args:
            composition: 組成

        Returns:
            バンドギャップ [eV]
        """
        # 結晶構造を生成（pymatgenなどで）
        structure = self._generate_structure(composition)

        # ASE Atomsオブジェクトに変換
        atoms = Atoms(
            symbols=structure.species,
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True
        )

        # VASP計算設定
        calc = Vasp(
            xc='PBE',
            encut=520,
            kpts=(4, 4, 4),
            ismear=0,
            sigma=0.05,
            directory='vasp_calc'
        )
        atoms.calc = calc

        # 構造最適化
        opt = BFGS(atoms)
        opt.run(fmax=0.05)

        # バンドギャップ計算
        # ... (VASPのOUTCAR解析) ...

        return bandgap

    def step(self, action):
        # DFT計算は時間がかかるため、
        # 実際には以下のような工夫が必要:
        # 1. 事前計算データベースを構築
        # 2. サロゲートモデルで高速予測
        # 3. アクティブラーニングで重要な材料のみDFT計算
        pass
```

**実用的アプローチ**:
1. **事前学習**: Materials Projectなどのデータでサロゲートモデルを訓練
2. **強化学習**: サロゲートモデルで高速探索
3. **検証**: 有望な材料のみDFT計算で精密評価

---

## 3.5 実験装置との統合（クローズドループ）

### REST APIによる自動実験装置制御

```python
import requests

class RoboticLabEnv(gym.Env):
    """ロボット実験装置統合環境"""

    def __init__(self, api_endpoint="http://lab-robot.example.com/api"):
        super(RoboticLabEnv, self).__init__()
        self.api_endpoint = api_endpoint

        # ... (環境設定) ...

    def _synthesize_and_measure(self, composition, temperature, time):
        """材料を合成し、特性を測定

        Args:
            composition: 組成
            temperature: 合成温度 [K]
            time: 合成時間 [min]

        Returns:
            測定結果（バンドギャップ、XRDパターンなど）
        """
        # ロボットに合成リクエスト
        payload = {
            'composition': composition,
            'temperature': temperature,
            'time': time,
            'measurement': ['bandgap', 'xrd']
        }

        response = requests.post(
            f"{self.api_endpoint}/synthesize",
            json=payload,
            headers={'Authorization': 'Bearer YOUR_API_KEY'}
        )

        if response.status_code == 200:
            result = response.json()
            return result['bandgap'], result['xrd_pattern']
        else:
            raise Exception(f"実験失敗: {response.text}")

    def step(self, action):
        """行動 = 合成条件"""
        composition, temperature, time = self._decode_action(action)

        # 実験実行（数分〜数時間）
        bandgap, xrd = self._synthesize_and_measure(composition, temperature, time)

        # 報酬計算
        reward = -abs(bandgap - self.target_bandgap)

        # 状態更新（実験履歴を含む）
        state = self._update_state(composition, temperature, time, bandgap, xrd)

        done = len(self.history) >= self.max_experiments

        return state, reward, done, {'bandgap': bandgap}
```

**課題**:
- **実験コスト**: 1回あたり数千円〜数万円
- **時間**: 合成・測定に数時間〜数日
- **安全性**: ロボットの誤作動、危険物質の扱い

**解決策**:
- **シミュレーション先行**: サロゲートモデルで事前探索
- **ベイズ最適化併用**: 効率的な実験点選択
- **バッチ実験**: 並列で複数材料を合成

---

## 演習問題

### 問題1 (難易度: easy)

以下の2つの報酬関数の違いを説明し、どちらが学習しやすいか理由とともに答えてください。

**報酬A**:
```python
reward = 10.0 if abs(bandgap - 3.0) < 0.1 else 0.0
```

**報酬B**:
```python
reward = -abs(bandgap - 3.0)
```

<details>
<summary>ヒント</summary>

報酬Aは疎報酬、報酬Bは密報酬です。学習シグナルの頻度を考えてみましょう。

</details>

<details>
<summary>解答例</summary>

**報酬Aの特徴**:
- **疎報酬**: バンドギャップが2.9〜3.1 eVの範囲に入ったときのみ報酬10.0、それ以外は0.0
- **学習が困難**: ほとんどの探索で報酬0、どの方向に進めば良いかわからない
- **探索が非効率**: ランダム探索に近くなる

**報酬Bの特徴**:
- **密報酬**: すべての行動で報酬が得られる（目標との距離）
- **学習が容易**: 目標に近づくと報酬が改善するため、勾配が明確
- **探索が効率的**: 報酬の変化から学習できる

**結論**: **報酬Bの方が学習しやすい**

ただし、報酬Bには局所最適解に陥りやすいという欠点もあります。実用的には、報酬Bをベースに、報酬Aのようなボーナスを追加するハイブリッド設計が有効です。

```python
# ハイブリッド報酬
reward = -abs(bandgap - 3.0)  # 密報酬
if abs(bandgap - 3.0) < 0.1:
    reward += 10.0  # ボーナス（疎報酬の要素）
```

</details>

---

### 問題2 (難易度: medium)

材料探索において、以下の3つの状態表現を比較し、それぞれの長所・短所を述べてください。

1. **組成のみ**: `["Li2MnO3"]`（文字列）
2. **元素割合**: `[0.33, 0.17, 0.50]`（Li, Mn, Oの割合）
3. **Magpie記述子**: 132次元ベクトル（平均原子番号、電気陰性度など）

<details>
<summary>ヒント</summary>

ニューラルネットワークは数値入力を必要とします。また、記述子の次元数と学習の複雑さの関係を考えてみましょう。

</details>

<details>
<summary>解答例</summary>

**1. 組成文字列の長所・短所**:

**長所**:
- 人間が理解しやすい
- データベース検索に直接使用可能

**短所**:
- ニューラルネットワークに直接入力できない（数値変換が必要）
- 類似組成の関係性を捉えにくい（"TiO2"と"ZrO2"が似ていることを学習しにくい）

**2. 元素割合の長所・短所**:

**長所**:
- 数値ベクトルなのでNNに入力可能
- 低次元（3次元など）で扱いやすい

**短所**:
- 元素の化学的性質を反映しない（TiとZrが似ていることを表現できない）
- 元素の順序が任意（[Li, Mn, O]と[O, Mn, Li]が異なるベクトルになる）

**3. Magpie記述子の長所・短所**:

**長所**:
- 元素の化学的性質を反映（電気陰性度、イオン半径など）
- 類似組成が似たベクトルになる
- 機械学習で高い予測性能

**短所**:
- 高次元（132次元）で学習が複雑
- 解釈性が低い（どの次元が何を表すか直感的でない）

**推奨**:
- **初期探索**: Magpie記述子（汎用性が高い）
- **特定タスク**: タスク専用の記述子（例: 触媒ならd軌道占有数）
- **ハイブリッド**: 組成 + プロセスパラメータ

</details>

---

### 問題3 (難易度: hard)

バンドギャップ探索環境において、以下の改善を実装してください：

1. **履歴を考慮した状態**: これまで試した材料の情報を状態に含める
2. **探索ボーナス**: 未知の領域を探索した場合に追加報酬
3. **早期終了**: 10ステップ連続で改善がない場合、エピソード終了

<details>
<summary>ヒント</summary>

履歴は辞書形式で保存し、状態には「最良材料との距離」などを追加します。探索ボーナスは、過去の材料との類似度で計算できます。

</details>

<details>
<summary>解答例</summary>

```python
import numpy as np
from scipy.spatial.distance import euclidean

class ImprovedBandgapEnv(gym.Env):
    """改善版バンドギャップ探索環境"""

    def __init__(self, target_bandgap=3.0):
        super(ImprovedBandgapEnv, self).__init__()

        self.target_bandgap = target_bandgap

        # 行動・状態空間（簡略化）
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(15,), dtype=np.float32)

        # 履歴
        self.history = []
        self.best_error = float('inf')
        self.no_improvement_count = 0

    def reset(self):
        self.history = []
        self.best_error = float('inf')
        self.no_improvement_count = 0

        initial_state = self._get_state(np.random.uniform(0, 1, 10))
        return initial_state

    def step(self, action):
        # バンドギャップ予測（簡易モデル）
        predicted_bandgap = np.sum(action) * 3.0  # 仮の予測

        # 誤差
        error = abs(predicted_bandgap - self.target_bandgap)

        # 基本報酬
        reward = -error

        # 改善1: 履歴を考慮した状態
        state = self._get_state(action)

        # 改善2: 探索ボーナス
        exploration_bonus = self._compute_exploration_bonus(action)
        reward += 0.1 * exploration_bonus

        # 改善3: 早期終了
        if error < self.best_error:
            self.best_error = error
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        done = error < 0.05 or self.no_improvement_count >= 10 or len(self.history) >= 100

        # 履歴に追加
        self.history.append({
            'action': action,
            'bandgap': predicted_bandgap,
            'error': error
        })

        info = {'bandgap': predicted_bandgap, 'exploration_bonus': exploration_bonus}

        return state, reward, done, info

    def _get_state(self, action):
        """履歴を考慮した状態

        状態構成:
        - 現在の行動（10次元）
        - 最良材料との距離（1次元）
        - 履歴サイズ（1次元）
        - 改善なし連続回数（1次元）
        - 平均誤差（1次元）
        - 最良誤差（1次元）
        """
        state = np.zeros(15, dtype=np.float32)

        # 現在の行動
        state[:10] = action

        # 最良材料との距離
        if self.history:
            best_action = min(self.history, key=lambda x: x['error'])['action']
            state[10] = euclidean(action, best_action) / 10.0  # 正規化
        else:
            state[10] = 1.0

        # 履歴サイズ
        state[11] = len(self.history) / 100.0  # 正規化

        # 改善なし連続回数
        state[12] = self.no_improvement_count / 10.0

        # 平均誤差
        if self.history:
            state[13] = np.mean([h['error'] for h in self.history])
        else:
            state[13] = 10.0

        # 最良誤差
        state[14] = self.best_error

        return state

    def _compute_exploration_bonus(self, action):
        """探索ボーナス

        過去の行動と離れているほど高いボーナス
        """
        if not self.history:
            return 1.0  # 最初は常に探索

        # 過去の行動との最小距離
        min_distance = min(
            euclidean(action, h['action'])
            for h in self.history
        )

        # 距離が大きいほどボーナス（最大1.0）
        bonus = min(1.0, min_distance / 5.0)

        return bonus


# テスト
env = ImprovedBandgapEnv()
state = env.reset()

for step in range(50):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    print(f"Step {step+1}: Bandgap={info['bandgap']:.2f}, "
          f"Reward={reward:.2f}, Exploration={info['exploration_bonus']:.2f}")

    if done:
        print(f"終了: 最良誤差={env.best_error:.4f}, "
              f"改善なし連続={env.no_improvement_count}回")
        break
```

**ポイント**:
- 履歴情報を状態に含めることで、エージェントが過去の経験を活用
- 探索ボーナスにより、未知領域の探索を促進
- 早期終了により、無駄な探索を削減

</details>

---

## このセクションのまとめ

- **OpenAI Gym**は強化学習環境の標準インターフェース
- **状態空間**は材料記述子で設計（組成、構造、プロセスパラメータ）
- **報酬関数**は明確な目標、適切なスケーリング、中間報酬が重要
- **DFT統合**はサロゲートモデルで高速化し、重要な材料のみ精密計算
- **実験装置統合**はREST APIでクローズドループ最適化を実現

次章では、化学プロセス制御や合成経路設計など、実世界での応用事例を学びます。

---

## 参考文献

1. Brockman et al. "OpenAI Gym" *arXiv* (2016) - Gym環境の標準
2. Ward et al. "A general-purpose machine learning framework for predicting properties of inorganic materials" *npj Computational Materials* (2016) - Magpie記述子
3. Brockherde et al. "Bypassing the Kohn-Sham equations with machine learning" *Nature Communications* (2017) - DFT加速
4. Ng et al. "Policy invariance under reward transformations" *ICML* (1999) - 報酬シェイピング理論

---

**次章**: [第4章: 実世界応用とクローズドループ](chapter-4.html)
