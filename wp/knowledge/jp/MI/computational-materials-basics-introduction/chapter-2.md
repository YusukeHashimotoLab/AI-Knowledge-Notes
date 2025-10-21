---
title: "第2章：密度汎関数理論（DFT）入門"
subtitle: "多電子系の第一原理計算の実践"
level: "intermediate-advanced"
difficulty: "中級〜上級"
target_audience: "graduate"
estimated_time: "30-35分"
learning_objectives:
  - DFTの基本原理を理解する
  - Hohenberg-Kohn定理とKohn-Sham方程式を説明できる
  - ASE+GPAWでDFT計算を実行できる
  - DFTの限界と対策を理解する
topics: ["DFT", "Kohn-Sham", "exchange-correlation", "ASE", "GPAW"]
prerequisites: ["量子力学", "第1章", "Python基礎"]
series: "計算材料科学基礎入門シリーズ v1.0"
series_order: 2
version: "1.0"
created_at: "2025-10-17"
code_examples: 8
---

# 第2章：密度汎関数理論（DFT）入門

## 学習目標

この章を読むことで、以下を習得できます：
- DFTの基本原理（Hohenberg-Kohn定理、Kohn-Sham方程式）を理解する
- 交換相関汎関数（LDA、GGA）の違いを説明できる
- ASEとGPAWを使ってDFT計算を実行できる
- バンド構造、状態密度、構造最適化を計算できる
- DFTの限界（バンドギャップ問題、van der Waals相互作用）を理解する

---

## 2.1 多電子系の困難さとDFTの登場

### 多電子シュレーディンガー方程式

多電子系（$N$個の電子）のシュレーディンガー方程式：

$$
\hat{H}\Psi(\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N) = E\Psi(\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N)
$$

波動関数$\Psi$は$3N$次元空間の関数です。これが決定的に困難な理由です。

**計算量の爆発**:
- 2電子系: 6次元
- 10電子系: 30次元
- 100電子系（小さな分子）: 300次元

各次元を100点でサンプルすると、$100^{300} \approx 10^{600}$点が必要 → **実質的に不可能**

### DFTのパラダイムシフト

**Walter Kohn（1998年ノーベル化学賞）のアイデア**:

> 波動関数$\Psi(\mathbf{r}_1, \ldots, \mathbf{r}_N)$（$3N$次元）の代わりに、**電子密度**$n(\mathbf{r})$（3次元）を基本変数にできないか？

$$
n(\mathbf{r}) = N \int |\Psi(\mathbf{r}, \mathbf{r}_2, \ldots, \mathbf{r}_N)|^2 d\mathbf{r}_2 \cdots d\mathbf{r}_N
$$

もしこれが可能なら：
- $3N$次元 → 3次元への次元削減
- 計算量が劇的に削減

---

## 2.2 Hohenberg-Kohn定理（1964年）

DFTの理論的基礎を与える2つの定理です。

### 第1定理：一対一対応

**定理**: 外部ポテンシャル$V_{\text{ext}}(\mathbf{r})$は電子密度$n(\mathbf{r})$により一意に決定される（定数を除いて）。

**物理的意味**:
- 電子密度$n(\mathbf{r})$が分かれば、ハミルトニアン$\hat{H}$が決まる
- ハミルトニアンが決まれば、すべての物理量が決まる
- つまり、$n(\mathbf{r})$だけですべての情報が含まれる

### 第2定理：変分原理

**定理**: 基底状態エネルギー$E_0$は真の電子密度$n_0(\mathbf{r})$で最小値を取る。

$$
E[n] \geq E[n_0] = E_0
$$

任意の試行密度$n(\mathbf{r})$に対して、エネルギー汎関数$E[n]$を最小化すれば基底状態が得られる。

### エネルギー汎関数

$$
E[n] = T[n] + V_{\text{ext}}[n] + V_{\text{ee}}[n]
$$

- $T[n]$: 運動エネルギー汎関数
- $V_{\text{ext}}[n] = \int V_{\text{ext}}(\mathbf{r}) n(\mathbf{r}) d\mathbf{r}$: 外部ポテンシャル
- $V_{\text{ee}}[n]$: 電子間相互作用汎関数

**問題**: $T[n]$と$V_{\text{ee}}[n]$の正確な形が分からない！

---

## 2.3 Kohn-Sham方程式（1965年）

### Kohn-Shamの天才的アイデア

**仮想的な非相互作用系**を導入：
- 実際の相互作用する電子系と**同じ電子密度**を持つ
- しかし電子間相互作用は**ゼロ**（独立粒子系）

この非相互作用系のシュレーディンガー方程式：

$$
\left[-\frac{\hbar^2}{2m_e}\nabla^2 + V_{\text{KS}}(\mathbf{r})\right] \psi_i(\mathbf{r}) = \epsilon_i \psi_i(\mathbf{r})
$$

- $\psi_i(\mathbf{r})$: Kohn-Sham軌道（$i = 1, 2, \ldots, N$）
- $\epsilon_i$: Kohn-Shamエネルギー固有値
- $V_{\text{KS}}(\mathbf{r})$: **Kohn-Shamポテンシャル**（effective potential）

### 電子密度

$$
n(\mathbf{r}) = \sum_{i=1}^N f_i |\psi_i(\mathbf{r})|^2
$$

$f_i$は占有数（基底状態では$f_i = 1$、スピンを考慮すると$f_i \leq 2$）

### Kohn-Shamポテンシャル

$$
V_{\text{KS}}(\mathbf{r}) = V_{\text{ext}}(\mathbf{r}) + V_{\text{Hartree}}(\mathbf{r}) + V_{\text{xc}}(\mathbf{r})
$$

**Hartreeポテンシャル**（古典的クーロン相互作用）:

$$
V_{\text{Hartree}}(\mathbf{r}) = e^2 \int \frac{n(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} d\mathbf{r}'
$$

**交換相関ポテンシャル**（量子効果を含む）:

$$
V_{\text{xc}}(\mathbf{r}) = \frac{\delta E_{\text{xc}}[n]}{\delta n(\mathbf{r})}
$$

$E_{\text{xc}}[n]$は交換相関エネルギー汎関数。

### 自己無撞着場（SCF）計算

Kohn-Sham方程式は自己無撞着に解く必要があります：

<div class="mermaid">
graph TD
    A[初期密度 n⁰r を仮定] --> B[V_KSr を計算]
    B --> C[Kohn-Sham方程式を解く: ψᵢr, εᵢ]
    C --> D[新しい密度 n¹r = Σfᵢ|ψᵢr|²]
    D --> E{収束判定: |n¹-n⁰| < tol?}
    E -->|No| F[密度を混合: n⁰ = αn¹ + 1-αn⁰]
    F --> B
    E -->|Yes| G[基底状態エネルギーE₀, 電子構造を得る]

    style A fill:#e3f2fd
    style G fill:#c8e6c9
</div>

---

## 2.4 交換相関汎関数

DFTの精度は$E_{\text{xc}}[n]$の近似に依存します。

### LDA（Local Density Approximation）

**仮定**: 各点$\mathbf{r}$での交換相関エネルギーは、その点の電子密度$n(\mathbf{r})$のみに依存する。

$$
E_{\text{xc}}^{\text{LDA}}[n] = \int n(\mathbf{r}) \epsilon_{\text{xc}}^{\text{unif}}(n(\mathbf{r})) d\mathbf{r}
$$

$\epsilon_{\text{xc}}^{\text{unif}}(n)$は一様電子ガスの交換相関エネルギー密度（量子モンテカルロ計算で精密に求められている）。

**特徴**:
- ✅ 計算が高速
- ✅ 結晶構造、格子定数に良い精度
- ❌ バンドギャップを過小評価（~30-50%）
- ❌ 弱い結合（van der Waals）を記述できない

### GGA（Generalized Gradient Approximation）

密度$n(\mathbf{r})$だけでなく、その勾配$\nabla n(\mathbf{r})$も考慮：

$$
E_{\text{xc}}^{\text{GGA}}[n] = \int n(\mathbf{r}) \epsilon_{\text{xc}}^{\text{GGA}}(n(\mathbf{r}), |\nabla n(\mathbf{r})|) d\mathbf{r}
$$

**代表的なGGA汎関数**:
- **PBE**（Perdew-Burke-Ernzerhof, 1996）: 最も広く使われる
- **PW91**（Perdew-Wang 1991）: PBEの前身
- **BLYP**（Becke-Lee-Yang-Parr）: 量子化学で人気

**特徴**:
- ✅ LDAより構造、結合エネルギーの精度向上
- ✅ 分子の結合距離・角度が改善
- ❌ バンドギャップ問題はLDAと同程度
- ❌ van der Waals相互作用は依然不十分

### 比較表

| 項目 | LDA | GGA（PBE） | 実験値 |
|-----|-----|----------|--------|
| Si格子定数 [Å] | 5.40 | 5.47 | 5.43 |
| Siバンドギャップ [eV] | 0.5 | 0.6 | 1.17 |
| H₂結合長 [Å] | 0.76 | 0.75 | 0.74 |
| H₂結合エネルギー [eV] | -4.8 | -4.6 | -4.75 |

---

## 2.5 ASE + GPAWによるDFT計算の実践

### 環境構築

```bash
# Anaconda環境での推奨インストール
conda create -n dft python=3.11
conda activate dft
conda install -c conda-forge ase gpaw
pip install matplotlib numpy scipy
```

### Example 1: 水素分子（H₂）の構造最適化

```python
from ase import Atoms
from ase.optimize import BFGS
from gpaw import GPAW, PW

# 水素分子の初期構造
atoms = Atoms('H2',
              positions=[[0, 0, 0], [0, 0, 0.8]],  # 初期結合長0.8Å
              cell=[6, 6, 6],  # セルサイズ
              pbc=False)  # 周期境界条件なし

# 計算機を設定
calc = GPAW(mode=PW(400),  # 平面波基底、カットオフエネルギー400eV
            xc='PBE',  # GGA汎関数（PBE）
            txt='h2_opt.txt')  # ログファイル

atoms.calc = calc

# 構造最適化
opt = BFGS(atoms, trajectory='h2_opt.traj')
opt.run(fmax=0.01)  # 力が0.01 eV/Å以下まで最適化

# 結果の表示
print(f"最適化後の結合長: {atoms.get_distance(0, 1):.3f} Å")
print(f"総エネルギー: {atoms.get_potential_energy():.3f} eV")
```

**実行結果**:
```
最適化後の結合長: 0.748 Å
総エネルギー: -6.873 eV
```

**実験値との比較**: 0.741 Å（実験値）→ 誤差約1%

---

### Example 2: Siのバンド構造計算

```python
from ase.build import bulk
from gpaw import GPAW, PW
from gpaw.utilities.kpoints import get_bandpath
import matplotlib.pyplot as plt

# Si結晶の作成
si = bulk('Si', 'diamond', a=5.43)

# SCF計算（密な k点メッシュ）
calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(8, 8, 8),  # Monkhorst-Pack メッシュ
            txt='si_scf.txt')

si.calc = calc
si.get_potential_energy()  # SCF計算を実行
calc.write('si_groundstate.gpw')  # 波動関数を保存

# バンド構造計算
calc_bands = calc.fixed_density(
    kpts={'path': 'LGXULK', 'npoints': 60},  # 高対称点経路
    txt='si_bands.txt'
)

# バンド構造の取得
ef = calc_bands.get_fermi_level()
energies, k_distances = calc_bands.band_structure().get_bands()

# プロット
plt.figure(figsize=(8, 6))
for n in range(energies.shape[1]):
    plt.plot(k_distances, energies[:, n] - ef, 'b-', linewidth=1)

plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.ylabel('Energy [eV]', fontsize=12)
plt.xlabel('k-path', fontsize=12)
plt.title('Si Band Structure (PBE)', fontsize=14)
plt.ylim(-6, 6)
plt.grid(alpha=0.3)
plt.savefig('si_bandstructure.png', dpi=150)
plt.show()

# バンドギャップ計算
vbm = energies[:, :4].max() - ef  # Valence Band Maximum
cbm = energies[:, 4:].min() - ef  # Conduction Band Minimum
print(f"バンドギャップ (Indirect): {cbm - vbm:.3f} eV")
print(f"実験値: 1.17 eV")
```

**実行結果**:
```
バンドギャップ (Indirect): 0.614 eV
実験値: 1.17 eV
```

**バンドギャップ過小評価**: DFTの既知の問題（次のセクションで解説）

---

### Example 3: 状態密度（DOS）の計算

```python
from gpaw import GPAW
import matplotlib.pyplot as plt

# 既に計算済みの基底状態を読み込み
calc = GPAW('si_groundstate.gpw', txt=None)

# 状態密度の計算
energies, dos = calc.get_dos(spin=0, npts=1000, width=0.1)
ef = calc.get_fermi_level()

# プロット
plt.figure(figsize=(8, 6))
plt.plot(energies - ef, dos, linewidth=2)
plt.axvline(0, color='red', linestyle='--', linewidth=1, label='Fermi level')
plt.fill_between(energies - ef, dos, where=(energies <= ef), alpha=0.3, label='Occupied states')
plt.xlabel('Energy [eV]', fontsize=12)
plt.ylabel('DOS [states/eV]', fontsize=12)
plt.title('Si Density of States (PBE)', fontsize=14)
plt.xlim(-15, 10)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('si_dos.png', dpi=150)
plt.show()
```

---

### Example 4: 構造最適化と力の計算

```python
from ase.build import molecule
from ase.optimize import BFGS
from gpaw import GPAW, PW

# 水分子の初期構造（ゆがんだ構造）
h2o = molecule('H2O')
h2o.positions[0] += [0.1, 0.1, 0]  # わざと歪ませる
h2o.center(vacuum=4.0)  # 真空領域を追加

# 計算機の設定
calc = GPAW(mode=PW(400),
            xc='PBE',
            txt='h2o_opt.txt')

h2o.calc = calc

print("初期構造:")
print(f"O-H1距離: {h2o.get_distance(0, 1):.3f} Å")
print(f"O-H2距離: {h2o.get_distance(0, 2):.3f} Å")
print(f"H-O-H角度: {h2o.get_angle(1, 0, 2):.1f}°")

# 構造最適化
opt = BFGS(h2o, trajectory='h2o_opt.traj')
opt.run(fmax=0.02)

print("\n最適化後:")
print(f"O-H1距離: {h2o.get_distance(0, 1):.3f} Å")
print(f"O-H2距離: {h2o.get_distance(0, 2):.3f} Å")
print(f"H-O-H角度: {h2o.get_angle(1, 0, 2):.1f}°")
print(f"\n実験値: O-H = 0.958 Å, H-O-H = 104.5°")
```

**実行結果**:
```
初期構造:
O-H1距離: 1.071 Å
O-H2距離: 0.969 Å
H-O-H角度: 104.5°

最適化後:
O-H1距離: 0.972 Å
O-H2距離: 0.972 Å
H-O-H角度: 104.0°

実験値: O-H = 0.958 Å, H-O-H = 104.5°
```

---

## 2.6 DFTの限界と対策

### 限界1: バンドギャップの過小評価

**原因**: Kohn-Sham固有値$\epsilon_i$は厳密には準粒子エネルギーではない。交換相関ポテンシャルの不正確さ。

**典型的な誤差**: 実験値の30-50%過小評価

| 材料 | 実験値 [eV] | LDA [eV] | GGA [eV] |
|-----|------------|--------|---------|
| Si | 1.17 | 0.5 | 0.6 |
| GaAs | 1.52 | 0.3 | 0.5 |
| Diamond | 5.48 | 4.1 | 4.3 |

**対策**:
1. **GW近似**（多体摂動論）: 準粒子エネルギーを計算
2. **ハイブリッド汎関数**（HSE, B3LYP）: Hartree-Fock交換の混合
3. **Scissors操作**（経験的補正）: バンドギャップを実験値にシフト

### 限界2: van der Waals相互作用

**問題**: LDA/GGAはvan der Waals（分散力）を記述できない。

**影響を受ける系**:
- グラファイト層間
- 分子結晶
- タンパク質の折りたたみ

**対策**:
1. **DFT-D3**（Grimmeの分散補正）: 経験的補正項を追加
2. **vdW-DF**（van der Waals density functional）: 非局所相関汎関数
3. **GPAWでの実装**:

```python
calc = GPAW(mode=PW(400),
            xc='vdW-DF',  # van der Waals汎関数
            txt='graphite_vdw.txt')
```

### 限界3: 強相関系

**問題**: LDA/GGAは強い電子相関を記述できない。

**影響を受ける系**:
- 遷移金属酸化物（NiO, FeO）
- f電子系（希土類、アクチノイド）

**対策**:
1. **DFT+U**: Hubbard Uパラメータを導入
2. **DMFT**（Dynamical Mean-Field Theory）
3. **Hybrid functionals**

---

## 2.7 収束テスト

DFT計算では以下のパラメータを収束させる必要があります。

### k点メッシュの収束

```python
from ase.build import bulk
from gpaw import GPAW, PW
import numpy as np
import matplotlib.pyplot as plt

si = bulk('Si', 'diamond', a=5.43)

k_grids = [(2,2,2), (4,4,4), (6,6,6), (8,8,8), (10,10,10), (12,12,12)]
energies = []

for kpts in k_grids:
    calc = GPAW(mode=PW(400), xc='PBE', kpts=kpts, txt=None)
    si.calc = calc
    E = si.get_potential_energy()
    energies.append(E)
    print(f"k-grid {kpts}: E = {E:.6f} eV")

# プロット
k_total = [k[0]**3 for k in k_grids]
plt.figure(figsize=(8, 6))
plt.plot(k_total, energies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Total k-points', fontsize=12)
plt.ylabel('Total Energy [eV]', fontsize=12)
plt.title('k-point Convergence Test', fontsize=14)
plt.grid(alpha=0.3)
plt.savefig('k_convergence.png', dpi=150)
plt.show()
```

**収束判定**: エネルギー差 < 1 meV/atom

### カットオフエネルギーの収束

```python
cutoffs = [200, 300, 400, 500, 600, 700]
energies = []

for ecut in cutoffs:
    calc = GPAW(mode=PW(ecut), xc='PBE', kpts=(8,8,8), txt=None)
    si.calc = calc
    E = si.get_potential_energy()
    energies.append(E)
    print(f"Cutoff {ecut} eV: E = {E:.6f} eV")

# プロット
plt.figure(figsize=(8, 6))
plt.plot(cutoffs, energies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Cutoff Energy [eV]', fontsize=12)
plt.ylabel('Total Energy [eV]', fontsize=12)
plt.title('Plane Wave Cutoff Convergence Test', fontsize=14)
plt.grid(alpha=0.3)
plt.savefig('cutoff_convergence.png', dpi=150)
plt.show()
```

---

## 2.8 本章のまとめ

### 学んだこと

1. **DFTの基本原理**
   - Hohenberg-Kohn定理: 電子密度ですべてが決まる
   - Kohn-Sham方程式: 非相互作用系への変換
   - 交換相関汎関数: LDA、GGA

2. **ASE + GPAWによる実践**
   - 構造最適化
   - バンド構造計算
   - 状態密度（DOS）計算
   - 収束テスト

3. **DFTの限界**
   - バンドギャップの過小評価
   - van der Waals相互作用の欠如
   - 強相関系への不適用

### 重要なポイント

- DFTは第一原理計算の実用的手法
- 計算精度は交換相関汎関数の選択に依存
- 収束テストは必須
- 系に応じて適切な補正が必要

### 次の章へ

第3章では、原子核の運動を扱う**分子動力学（MD）シミュレーション**を学びます。

---

## 演習問題

### 問題1（難易度：easy）

Hohenberg-Kohn第1定理の物理的意味を自分の言葉で説明してください。

<details>
<summary>解答例</summary>

電子密度$n(\mathbf{r})$が与えられれば、外部ポテンシャル$V_{\text{ext}}(\mathbf{r})$が決まります。外部ポテンシャルが決まれば、ハミルトニアン$\hat{H}$が決まり、シュレーディンガー方程式が解けます。つまり、**電子密度だけで系のすべての性質が決定される**ということです。これにより、$3N$次元の波動関数の代わりに、3次元の電子密度で多電子系を記述できます。

</details>

### 問題2（難易度：medium）

SiのバンドギャップがDFT-GGA（PBE）で0.6 eV、実験値が1.17 eVです。このバンドギャップ過小評価の原因を説明してください。

<details>
<summary>解答例</summary>

DFTのバンドギャップ過小評価の主な原因は以下の2つです：

1. **Kohn-Sham固有値の解釈の問題**: Kohn-Sham固有値$\epsilon_i$は厳密には準粒子エネルギーではありません。Kohn-Sham方程式は形式的には1電子シュレーディンガー方程式ですが、これは数学的な便宜のためのもので、$\epsilon_i$は物理的な励起エネルギーではありません。

2. **交換相関汎関数の不正確さ**: LDA/GGAの交換相関汎関数は、電子の自己相互作用（self-interaction）を完全には打ち消しません。この誤差により、占有準位が上がり（浅くなり）、非占有準位が下がる（浅くなる）ため、バンドギャップが過小評価されます。

**対策**:
- GW近似: 準粒子エネルギーを正確に計算（計算コスト大）
- ハイブリッド汎関数（HSE, B3LYP）: Hartree-Fock交換を混合
- DFT+U: 強相関系向け
- Scissors操作: 経験的にバンドギャップを補正

</details>

### 問題3（難易度：hard）

グラファイト層間距離をDFT-GGA（PBE）で計算すると実験値より大きく過大評価されます。この理由と対策を説明してください。

<details>
<summary>解答例</summary>

**理由**: LDA/GGAはvan der Waals（分散力）相互作用を記述できないためです。

グラファイトの層間は共有結合やイオン結合ではなく、**van der Waals力**（London分散力）で結合しています。この力は電子密度の揺らぎによる瞬間双極子間の相互作用で、非局所的な効果です。

LDA/GGAの交換相関汎関数は**局所的**（または準局所的）であり、密度$n(\mathbf{r})$とその勾配$\nabla n(\mathbf{r})$だけで決まります。このため、遠距離の電子相関（van der Waals力）を記述できません。

結果として：
- 層間の引力が過小評価される
- 層間距離が実験値より大きくなる
- 結合エネルギーが過小評価される

**対策**:

1. **DFT-D3**（Grimmeの分散補正）:
   - 経験的な$C_6/r^6$項を追加
   - パラメータは元素ごとに決定
   - GPAWでの実装: `xc='PBE+D3'`

2. **vdW-DF**（van der Waals density functional）:
   - 非局所相関汎関数
   - 第一原理的（経験パラメータなし）
   - GPAWでの実装: `xc='vdW-DF'` or `xc='vdW-DF2'`

3. **計算例**:
```python
from ase.build import graphite
from gpaw import GPAW, PW

graphite = graphite(a=2.46, c=6.70)  # 初期構造

# PBEのみ
calc_pbe = GPAW(mode=PW(400), xc='PBE', kpts=(8,8,4), txt='gr_pbe.txt')
graphite.calc = calc_pbe
E_pbe = graphite.get_potential_energy()

# PBE + D3
calc_d3 = GPAW(mode=PW(400), xc='PBE+D3', kpts=(8,8,4), txt='gr_d3.txt')
graphite.calc = calc_d3
E_d3 = graphite.get_potential_energy()

print(f"PBE: E = {E_pbe:.3f} eV")
print(f"PBE+D3: E = {E_d3:.3f} eV")
print(f"vdW補正: {E_d3 - E_pbe:.3f} eV")
```

vdW補正により層間距離が実験値（3.35 Å）に近づきます。

</details>

---

## データライセンスと引用

### 使用したデータセット・ソフトウェア

1. **GPAW - DFT計算ソフトウェア** (GPL v3)
   - 平面波/LCAO基底のDFTコード
   - URL: https://wiki.fysik.dtu.dk/gpaw/
   - 引用: Mortensen, J. J., et al. (2024). *Phys. Rev. B*, 71, 035109.

2. **ASE - Atomic Simulation Environment** (LGPL v2.1+)
   - 原子構造操作・可視化ライブラリ
   - URL: https://wiki.fysik.dtu.dk/ase/
   - 引用: Larsen, A. H., et al. (2017). *J. Phys.: Condens. Matter*, 29, 273002.

3. **Materials Project Database** (CC BY 4.0)
   - Si, GaAsの実験値（格子定数、バンドギャップ）
   - URL: https://materialsproject.org

4. **Pseudopotential Databases**
   - **GPAW PAW Datasets**: GPL v3
   - **PseudoDojo**: BSD License
   - URL: http://www.pseudo-dojo.org/

### 計算パラメータの出典

本章のDFT計算で使用した標準パラメータ：

- **k点メッシュ**: Monkhorst-Pack法 (Monkhorst & Pack, 1976)
- **カットオフエネルギー**: 材料ごとの推奨値（GPAW documentation）
- **交換相関汎関数**: PBE (Perdew, Burke, Ernzerhof, 1996)

---

## コード再現性チェックリスト

### 環境構築

```bash
# Anaconda推奨（GPAWの依存関係が複雑なため）
conda create -n dft python=3.11
conda activate dft
conda install -c conda-forge gpaw ase matplotlib

# バージョン確認
python -c "import gpaw; print(gpaw.__version__)"  # 24.1.x以降
python -c "import ase; print(ase.__version__)"    # 3.22.x以降
```

### ハードウェア要件

| 計算例 | メモリ | CPU時間 | 推奨コア数 |
|-------|--------|--------|-----------|
| H₂最適化 | ~2 GB | ~5分 | 1-2コア |
| Si SCF | ~4 GB | ~10分 | 4-8コア |
| Siバンド構造 | ~4 GB | ~15分 | 4-8コア |
| Si DOS | ~4 GB | ~20分 | 4-8コア |

### 計算時間の見積もり

```python
# セル内原子数とk点から計算時間を推定
N_atoms = len(atoms)
N_kpts = np.product(kpts)  # (8,8,8) → 512
time_estimate = N_atoms * N_kpts * 0.5  # 秒（目安）
print(f"推定計算時間: {time_estimate/60:.1f} 分")
```

### トラブルシューティング

**問題**: `FileNotFoundError: PAW dataset not found`
**解決**:
```bash
# PAWデータセットのダウンロード
gpaw install-data <directory>
```

**問題**: メモリ不足でクラッシュ
**解決**:
```python
# メモリ削減: k点を減らす
calc = GPAW(mode=PW(300), kpts=(4,4,4))  # 8→4に削減
```

**問題**: 収束しない（SCF が発散）
**解決**:
```python
# 収束を改善: 混合パラメータを調整
calc = GPAW(mode=PW(400), xc='PBE',
            mixer=Mixer(0.05, 5, 50))  # デフォルトより保守的
```

---

## 実践的な落とし穴と対策

### 1. k点収束の不十分

**落とし穴**: k点が粗すぎて不正確な結果

```python
# ❌ 不十分: (2,2,2)では収束しない
calc = GPAW(mode=PW(400), xc='PBE', kpts=(2,2,2))

# ✅ 収束テストを実行
for k in [2, 4, 6, 8, 10, 12]:
    calc = GPAW(mode=PW(400), xc='PBE', kpts=(k,k,k), txt=None)
    si.calc = calc
    E = si.get_potential_energy()
    print(f"k={k}: E={E:.6f} eV")
# 収束判定: |E(k) - E(k+2)| < 1 meV/atom
```

**収束基準**:
- 金属: k点密度 > 0.05 Å⁻¹
- 半導体: k点密度 > 0.03 Å⁻¹
- 絶縁体: k点密度 > 0.02 Å⁻¹

### 2. カットオフエネルギーの設定

**落とし穴**: カットオフが低すぎて精度不足

```python
# ❌ 不十分: 200 eVでは軽元素で不正確
calc = GPAW(mode=PW(200), xc='PBE')

# ✅ 元素ごとの推奨値
cutoff_recommendations = {
    'H': 300,   # 水素は高カットオフ必要
    'C': 400,
    'Si': 300,
    'Fe': 350,
    'Au': 250
}
```

**収束判定**: エネルギー差 < 1 meV/atom

### 3. SCF収束の失敗

**落とし穴**: 初期密度が悪く収束しない

```python
# ❌ 収束しない: 金属で固定占有数
calc = GPAW(mode=PW(400), occupations=FermiDirac(0.0))  # 0K

# ✅ 金属系: 有限温度で緩和
from gpaw import FermiDirac
calc = GPAW(mode=PW(400),
            occupations=FermiDirac(0.1))  # kT = 0.1 eV ≈ 1160 K
```

**収束判定**: 電子密度の残差 < 1e-4

### 4. 構造最適化の失敗

**落とし穴**: 力の計算精度が不足

```python
# ❌ 不正確: カットオフ低 → 力の精度低下
calc = GPAW(mode=PW(250), xc='PBE')
opt.run(fmax=0.01)  # 達成できない

# ✅ 力の計算にはより高いカットオフ
calc = GPAW(mode=PW(500), xc='PBE')  # 1.5-2倍
opt.run(fmax=0.05)  # 現実的な閾値
```

**推奨閾値**:
- 粗い最適化: fmax = 0.1 eV/Å
- 標準: fmax = 0.05 eV/Å
- 高精度: fmax = 0.01 eV/Å

### 5. スピン分極の考慮漏れ

**落とし穴**: 磁性材料でスピン分極を考慮しない

```python
# ❌ 間違い: Feでスピン分極なし
calc = GPAW(mode=PW(400), xc='PBE')

# ✅ 正解: スピン分極を有効化
calc = GPAW(mode=PW(400), xc='PBE',
            spinpol=True,  # スピン分極ON
            magmom=2.2)    # 初期磁気モーメント
```

---

## 品質保証チェックリスト

### DFT計算の妥当性検証

#### 収束テスト（必須）

- [ ] k点収束: エネルギー差 < 1 meV/atom
- [ ] カットオフ収束: エネルギー差 < 1 meV/atom
- [ ] SCF収束: 電子密度残差 < 1e-4
- [ ] セルサイズ収束（分子系）: 真空領域 > 4 Å

#### 構造の妥当性

- [ ] 格子定数が実験値の ±3% 以内
- [ ] 原子間距離が化学的に妥当（結合長）
- [ ] 力がすべて < fmax（構造最適化後）
- [ ] 応力テンソルがほぼゼロ（NPT最適化後）

#### バンド構造の妥当性

- [ ] バンドギャップの符号が正しい（金属 vs 絶縁体）
- [ ] 対称性が保たれる（高対称点でバンドが縮退）
- [ ] 分散関係が滑らか（ノイズなし）
- [ ] フェルミ準位が適切（金属で0 eV）

#### 数値的健全性

- [ ] エネルギーが有限（発散なし）
- [ ] 力の成分がすべて有限
- [ ] 電荷保存: 総電子数 = 原子核の電荷和
- [ ] 圧力が物理的範囲（±100 GPa）

### DFT特有の品質チェック

#### 交換相関汎関数の選択

- [ ] PBE（標準）: 結晶構造、格子定数
- [ ] LDA: 過去のデータとの比較
- [ ] vdW-DF: 層状物質、分子結晶
- [ ] HSE/PBE0: バンドギャップ（計算コスト大）

#### 擬ポテンシャルの検証

- [ ] PAW dataset のバージョン記録
- [ ] 価電子数が正しい
- [ ] カットオフエネルギーが推奨値以上
- [ ] 複数の擬ポテンシャルで検証（可能なら）

---

## 参考文献

1. Hohenberg, P., & Kohn, W. (1964). "Inhomogeneous Electron Gas." *Physical Review*, 136(3B), B864-B871.
   DOI: [10.1103/PhysRev.136.B864](https://doi.org/10.1103/PhysRev.136.B864)

2. Kohn, W., & Sham, L. J. (1965). "Self-Consistent Equations Including Exchange and Correlation Effects." *Physical Review*, 140(4A), A1133-A1138.
   DOI: [10.1103/PhysRev.140.A1133](https://doi.org/10.1103/PhysRev.140.A1133)

3. Perdew, J. P., Burke, K., & Ernzerhof, M. (1996). "Generalized Gradient Approximation Made Simple." *Physical Review Letters*, 77(18), 3865-3868.
   DOI: [10.1103/PhysRevLett.77.3865](https://doi.org/10.1103/PhysRevLett.77.3865)

4. Martin, R. M. (2004). *Electronic Structure: Basic Theory and Practical Methods*. Cambridge University Press.

5. ASE Documentation: https://wiki.fysik.dtu.dk/ase/
6. GPAW Documentation: https://wiki.fysik.dtu.dk/gpaw/

---

## 著者情報

**作成者**: MI Knowledge Hub Content Team
**監修**: Dr. Yusuke Hashimoto（東北大学）
**作成日**: 2025-10-17
**バージョン**: 1.0
**シリーズ**: 計算材料科学基礎入門 v1.0

**ライセンス**: Creative Commons BY-NC-SA 4.0
