---
title: "第4章：フォノン計算と熱力学特性"
subtitle: "格子振動から熱伝導まで"
level: "intermediate-advanced"
difficulty: "中級〜上級"
target_audience: "graduate"
estimated_time: "20-25分"
learning_objectives:
  - 格子振動とフォノンの関係を理解する
  - Phonopyでフォノン計算を実行できる
  - 熱力学特性（比熱、自由エネルギー）を計算できる
  - フォノンバンド構造を解釈できる
topics: ["phonon", "lattice-dynamics", "thermodynamics", "Phonopy"]
prerequisites: ["固体物理基礎", "第1章", "第2章"]
series: "計算材料科学基礎入門シリーズ v1.0"
series_order: 4
version: "1.0"
created_at: "2025-10-17"
code_examples: 6
---

# 第4章：フォノン計算と熱力学特性

## 学習目標

この章を読むことで、以下を習得できます：
- 格子振動（フォノン）の理論的基礎を理解する
- 調和近似と動力学行列の概念を説明できる
- Phonopyで実際にフォノン計算を実行できる
- フォノンバンド構造と状態密度を解釈できる
- 熱力学特性（比熱、自由エネルギー、エントロピー）を計算できる

---

## 4.1 格子振動の理論

### 調和振動子モデル

結晶中の原子は平衡位置$\mathbf{R}_I^0$の周りで振動します。変位を$\mathbf{u}_I$とすると：

$$
\mathbf{R}_I = \mathbf{R}_I^0 + \mathbf{u}_I
$$

ポテンシャルエネルギーをTaylor展開（調和近似）：

$$
U = U_0 + \sum_I \frac{\partial U}{\partial \mathbf{u}_I}\Bigg|_0 \mathbf{u}_I + \frac{1}{2}\sum_{I,J} \mathbf{u}_I \cdot \frac{\partial^2 U}{\partial \mathbf{u}_I \partial \mathbf{u}_J}\Bigg|_0 \cdot \mathbf{u}_J + O(\mathbf{u}^3)
$$

平衡位置では第1項がゼロなので：

$$
U \approx U_0 + \frac{1}{2}\sum_{I,J} \mathbf{u}_I \cdot \mathbf{\Phi}_{IJ} \cdot \mathbf{u}_J
$$

$\mathbf{\Phi}_{IJ}$は**力定数行列**（force constant matrix）：

$$
\Phi_{IJ}^{\alpha\beta} = \frac{\partial^2 U}{\partial u_I^\alpha \partial u_J^\beta}\Bigg|_0
$$

### 運動方程式

原子$I$の運動方程式：

$$
M_I \frac{d^2 u_I^\alpha}{dt^2} = -\sum_{J,\beta} \Phi_{IJ}^{\alpha\beta} u_J^\beta
$$

平面波解を仮定：

$$
u_I^\alpha = \frac{1}{\sqrt{M_I}} e_I^\alpha(\mathbf{k}) e^{i(\mathbf{k}\cdot\mathbf{R}_I - \omega t)}
$$

**動力学行列**（dynamical matrix）を導入：

$$
D_{IJ}^{\alpha\beta}(\mathbf{k}) = \frac{1}{\sqrt{M_I M_J}} \Phi_{IJ}^{\alpha\beta} e^{i\mathbf{k}\cdot(\mathbf{R}_I - \mathbf{R}_J)}
$$

### 固有値問題

$$
\sum_{J,\beta} D_{IJ}^{\alpha\beta}(\mathbf{k}) e_J^\beta(\mathbf{k}) = \omega^2(\mathbf{k}) e_I^\alpha(\mathbf{k})
$$

- $\omega(\mathbf{k})$: フォノンの角振動数（フォノン分散）
- $\mathbf{e}(\mathbf{k})$: フォノンの固有ベクトル（偏極ベクトル）

**フォノンバンドの数**: $3N_{\text{atom}}$本（単位胞内原子数$N_{\text{atom}}$に対して）

---

## 4.2 フォノンの分類

### 音響モード（Acoustic modes）

- $\omega(\mathbf{k}) \to 0$ as $\mathbf{k} \to 0$
- Γ点（$\mathbf{k}=0$）で周波数がゼロ
- 3本（1本の縦波LA、2本の横波TA）
- 物理的意味: 並進運動

### 光学モード（Optical modes）

- $\omega(\mathbf{k}) \neq 0$ at $\mathbf{k} = 0$
- Γ点で有限の周波数
- $3(N_{\text{atom}} - 1)$本
- 物理的意味: 単位胞内の相対運動

### 縦波（Longitudinal）vs 横波（Transverse）

- **縦波（L）**: 振動方向が波の進行方向と平行
- **横波（T）**: 振動方向が波の進行方向と垂直

**単原子結晶（Si, Cuなど）**:
- 音響モード: 3本（1 LA + 2 TA）
- 光学モード: なし

**2原子結晶（NaCl, GaAsなど）**:
- 音響モード: 3本
- 光学モード: 3本（1 LO + 2 TO）

---

## 4.3 Phonopyによる実践

### インストール

```bash
pip install phonopy
# またはcondaで
conda install -c conda-forge phonopy
```

### Example 1: Siのフォノン計算（GPAW使用）

**Step 1: 基底状態計算**

```python
from ase.build import bulk
from gpaw import GPAW, PW

# Si結晶の作成
si = bulk('Si', 'diamond', a=5.43)

# DFT計算
calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(8, 8, 8),
            txt='si_gs.txt')

si.calc = calc
si.get_potential_energy()
calc.write('si_groundstate.gpw')
```

**Step 2: Phonopy用のスーパーセル作成**

```bash
# phonopy_disp.confを作成
cat > phonopy_disp.conf <<EOF
DIM = 2 2 2
ATOM_NAME = Si
EOF

# Phonopyで変位構造を生成
phonopy -d --dim="2 2 2" --gpaw
```

これにより`supercell-XXX.py`ファイルが生成されます。

**Step 3: 力の計算**

```python
# supercell-001.pyを実行（各変位構造で）
from gpaw import GPAW

calc = GPAW('si_groundstate.gpw', txt=None)
atoms = calc.get_atoms()
forces = atoms.get_forces()

# FORCE_SETSファイルに書き込み（Phonopyが読み込む形式）
import numpy as np
np.savetxt('forces_001.dat', forces)
```

**全変位構造に対して繰り返す**（通常はスクリプトで自動化）

**Step 4: フォノン計算**

```python
from phonopy import Phonopy
from phonopy.interface.gpaw import read_gpaw
import matplotlib.pyplot as plt

# Phonopyオブジェクトの作成
unitcell, calc_forces = read_gpaw('si_groundstate.gpw')
phonon = Phonopy(unitcell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

# 力定数行列の設定（FORCE_SETSから）
phonon.set_displacement_dataset(dataset)
phonon.produce_force_constants()

# バンド構造の計算
path = [[[0, 0, 0], [0.5, 0, 0.5], [0.625, 0.25, 0.625],
         [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
labels = ["$\\Gamma$", "X", "U", "K", "$\\Gamma$", "L"]
qpoints, connections = phonon.get_band_structure_plot_data(path)

phonon.plot_band_structure(path, labels=labels)
plt.ylabel('Frequency (THz)')
plt.savefig('si_phonon_band.png', dpi=150)
plt.show()

# 状態密度（DOS）
phonon.set_mesh([20, 20, 20])
phonon.set_total_DOS()
dos_freq, dos_val = phonon.get_total_DOS()

plt.figure(figsize=(8, 6))
plt.plot(dos_freq, dos_val, linewidth=2)
plt.xlabel('Frequency (THz)', fontsize=12)
plt.ylabel('DOS (states/THz)', fontsize=12)
plt.title('Si Phonon DOS', fontsize=14)
plt.grid(alpha=0.3)
plt.savefig('si_phonon_dos.png', dpi=150)
plt.show()
```

---

### Example 2: 完全な自動化スクリプト

```python
from ase.build import bulk
from gpaw import GPAW, PW
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import numpy as np

def calculate_phonon(symbol='Si', a=5.43, dim=(2,2,2)):
    """
    完全自動フォノン計算
    """
    # 1. 基底状態計算
    print("Step 1: Ground state calculation...")
    atoms = bulk(symbol, 'diamond', a=a)
    calc = GPAW(mode=PW(400), xc='PBE', kpts=(8,8,8), txt=None)
    atoms.calc = calc
    atoms.get_potential_energy()

    # 2. Phonopyセットアップ
    print("Step 2: Generate displacements...")
    cell = PhonopyAtoms(symbols=[symbol]*len(atoms),
                       cell=atoms.cell,
                       scaled_positions=atoms.get_scaled_positions())

    phonon = Phonopy(cell, np.diag(dim))
    phonon.generate_displacements(distance=0.01)

    # 3. 力の計算
    print(f"Step 3: Calculate forces for {len(phonon.supercells_with_displacements)} supercells...")
    set_of_forces = []

    for i, scell in enumerate(phonon.supercells_with_displacements):
        supercell = bulk(symbol, 'diamond', a=a).repeat(dim)
        supercell.positions = scell.positions
        supercell.calc = calc
        forces = supercell.get_forces()
        drift_force = forces.sum(axis=0)
        for force in forces:
            force -= drift_force / len(forces)  # ドリフト補正
        set_of_forces.append(forces)

    # 4. フォノン計算
    print("Step 4: Phonon calculation...")
    phonon.produce_force_constants(forces=set_of_forces)

    # バンド構造
    path = [[[0, 0, 0], [0.5, 0, 0.5], [0.625, 0.25, 0.625],
             [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
    labels = ["$\\Gamma$", "X", "U", "K", "$\\Gamma$", "L"]

    phonon.auto_band_structure(plot=True, labels=labels, filename=f'{symbol}_band.png')

    # DOS
    phonon.auto_total_dos(plot=True, filename=f'{symbol}_dos.png')

    print("Done!")
    return phonon

# 実行
si_phonon = calculate_phonon('Si', a=5.43, dim=(2,2,2))
```

---

## 4.4 熱力学特性の計算

### 自由エネルギー

**Helmholtz自由エネルギー**（NVT）:

$$
F(T) = U_0 + k_B T \sum_{\mathbf{q},j} \ln\left[2\sinh\left(\frac{\hbar\omega_{\mathbf{q}j}}{2k_B T}\right)\right]
$$

- $U_0$: ゼロ点エネルギー
- $\omega_{\mathbf{q}j}$: 波数$\mathbf{q}$、バンド$j$のフォノン周波数

### 内部エネルギー

$$
U(T) = U_0 + \sum_{\mathbf{q},j} \hbar\omega_{\mathbf{q}j} \left[\frac{1}{2} + n_B(\omega_{\mathbf{q}j}, T)\right]
$$

$n_B(\omega, T)$はBose-Einstein分布関数：

$$
n_B(\omega, T) = \frac{1}{e^{\hbar\omega/(k_B T)} - 1}
$$

### エントロピー

$$
S(T) = -\left(\frac{\partial F}{\partial T}\right)_V = k_B \sum_{\mathbf{q},j} \left[\frac{\hbar\omega_{\mathbf{q}j}}{k_B T} n_B(\omega_{\mathbf{q}j}, T) - \ln(1 - e^{-\hbar\omega_{\mathbf{q}j}/(k_B T)})\right]
$$

### 比熱（定積）

$$
C_V(T) = \left(\frac{\partial U}{\partial T}\right)_V = k_B \sum_{\mathbf{q},j} \left(\frac{\hbar\omega_{\mathbf{q}j}}{k_B T}\right)^2 \frac{e^{\hbar\omega_{\mathbf{q}j}/(k_B T)}}{(e^{\hbar\omega_{\mathbf{q}j}/(k_B T)} - 1)^2}
$$

### Phonopyでの実装

```python
import numpy as np
import matplotlib.pyplot as plt

# 温度範囲
temperatures = np.arange(0, 1000, 10)

# 熱力学特性を計算
si_phonon.set_mesh([20, 20, 20])
si_phonon.set_thermal_properties(t_step=10, t_max=1000, t_min=0)

tp_dict = si_phonon.get_thermal_properties_dict()
temps = tp_dict['temperatures']
free_energy = tp_dict['free_energy']  # kJ/mol
entropy = tp_dict['entropy']  # J/K/mol
heat_capacity = tp_dict['heat_capacity']  # J/K/mol

# プロット
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 自由エネルギー
axes[0,0].plot(temps, free_energy, linewidth=2)
axes[0,0].set_xlabel('Temperature (K)', fontsize=12)
axes[0,0].set_ylabel('Free Energy (kJ/mol)', fontsize=12)
axes[0,0].set_title('Helmholtz Free Energy', fontsize=14)
axes[0,0].grid(alpha=0.3)

# エントロピー
axes[0,1].plot(temps, entropy, linewidth=2, color='orange')
axes[0,1].set_xlabel('Temperature (K)', fontsize=12)
axes[0,1].set_ylabel('Entropy (J/K/mol)', fontsize=12)
axes[0,1].set_title('Entropy', fontsize=14)
axes[0,1].grid(alpha=0.3)

# 比熱
axes[1,0].plot(temps, heat_capacity, linewidth=2, color='green')
axes[1,0].axhline(3*8.314, color='red', linestyle='--', label='Dulong-Petit (3R)')
axes[1,0].set_xlabel('Temperature (K)', fontsize=12)
axes[1,0].set_ylabel('Heat Capacity (J/K/mol)', fontsize=12)
axes[1,0].set_title('Heat Capacity at Constant Volume', fontsize=14)
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# 内部エネルギー（F = U - TSから計算）
internal_energy = free_energy + temps * entropy / 1000  # kJ/mol
axes[1,1].plot(temps, internal_energy, linewidth=2, color='purple')
axes[1,1].set_xlabel('Temperature (K)', fontsize=12)
axes[1,1].set_ylabel('Internal Energy (kJ/mol)', fontsize=12)
axes[1,1].set_title('Internal Energy', fontsize=14)
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('si_thermodynamics.png', dpi=150)
plt.show()

# デバイ温度の計算
# 比熱が3R（Dulong-Petit極限）の63%に達する温度
R = 8.314  # J/K/mol
target_cv = 0.63 * 3 * R
idx = np.argmin(np.abs(heat_capacity - target_cv))
theta_D = temps[idx]
print(f"デバイ温度: {theta_D:.1f} K")
print(f"実験値（Si）: 645 K")
```

---

## 4.5 熱膨張係数

### 準調和近似（Quasi-harmonic approximation）

体積を変えてフォノン計算を複数回実行：

$$
\alpha(T) = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_P = -\frac{1}{V}\frac{(\partial^2 F/\partial T \partial V)}{(\partial^2 F/\partial V^2)}
$$

### 実装例

```python
import numpy as np
import matplotlib.pyplot as plt

# 異なる格子定数で計算
lattice_constants = np.linspace(5.35, 5.51, 5)  # Å
phonons = []

for a in lattice_constants:
    print(f"Calculating phonon for a = {a:.3f} Å...")
    phonon = calculate_phonon('Si', a=a, dim=(2,2,2))
    phonons.append(phonon)

# 各温度、各体積で自由エネルギーを計算
temps = np.arange(0, 1000, 50)
volumes = (lattice_constants / 5.43)**3  # 規格化体積
free_energies = np.zeros((len(temps), len(volumes)))

for i, phonon in enumerate(phonons):
    phonon.set_mesh([20, 20, 20])
    phonon.set_thermal_properties(t_step=50, t_max=1000, t_min=0)
    tp = phonon.get_thermal_properties_dict()
    free_energies[:, i] = tp['free_energy']

# 各温度で平衡体積を見つける（F最小）
V_eq = np.zeros(len(temps))
for i, T in enumerate(temps):
    # 2次多項式フィット
    coeffs = np.polyfit(volumes, free_energies[i], 2)
    V_eq[i] = -coeffs[1] / (2 * coeffs[0])  # 極値

# 熱膨張係数
alpha = np.gradient(V_eq, temps) / V_eq

# プロット
plt.figure(figsize=(8, 6))
plt.plot(temps, alpha * 1e6, linewidth=2)
plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel('Thermal expansion coefficient (10$^{-6}$ K$^{-1}$)', fontsize=12)
plt.title('Si Thermal Expansion (QHA)', fontsize=14)
plt.grid(alpha=0.3)
plt.savefig('si_thermal_expansion.png', dpi=150)
plt.show()

print(f"室温（300K）での熱膨張係数: {alpha[6]*1e6:.2f} × 10⁻⁶ K⁻¹")
print(f"実験値: 2.6 × 10⁻⁶ K⁻¹")
```

---

## 4.6 フォノン計算の応用

### 1. 格子熱伝導率

**Boltzmann輸送方程式**からの推定（簡易版）:

$$
\kappa = \frac{1}{3} \sum_{\mathbf{q},j} C_{\mathbf{q}j} v_{\mathbf{q}j}^2 \tau_{\mathbf{q}j}
$$

- $C_{\mathbf{q}j}$: モードごとの比熱
- $v_{\mathbf{q}j}$: 群速度（$\partial\omega/\partial\mathbf{q}$）
- $\tau_{\mathbf{q}j}$: 緩和時間（フォノン散乱）

**PhonoPyでの群速度計算**:

```python
# 群速度の計算
si_phonon.set_group_velocity()
group_velocities = si_phonon.get_group_velocity()

print("Group velocities at Gamma point:")
print(group_velocities[0])  # [band, cartesian_direction]
```

### 2. 超伝導の臨界温度（Tc）

**McMillan式**（簡易版）:

$$
T_c = \frac{\omega_{\text{log}}}{1.2} \exp\left[-\frac{1.04(1+\lambda)}{\lambda - \mu^*(1+0.62\lambda)}\right]
$$

- $\omega_{\text{log}}$: 対数平均フォノン周波数
- $\lambda$: 電子-フォノン結合定数
- $\mu^*$: クーロン擬ポテンシャル

### 3. 相転移の検出

**虚数フォノンモード**が存在すると、結晶構造が不安定：

```python
# 負の振動数（虚数モード）のチェック
frequencies = si_phonon.get_frequencies(q=[0, 0, 0])
if np.any(frequencies < -1e-3):
    print("Warning: Imaginary phonon modes detected!")
    print("Structure may be unstable.")
```

---

## 4.7 本章のまとめ

### 学んだこと

1. **格子振動の理論**
   - 調和近似
   - 動力学行列
   - フォノン分散関係

2. **フォノンの分類**
   - 音響モード vs 光学モード
   - 縦波 vs 横波

3. **Phonopyによる実践**
   - フォノンバンド構造
   - フォノン状態密度
   - 完全自動化スクリプト

4. **熱力学特性**
   - 自由エネルギー
   - 内部エネルギー
   - エントロピー
   - 比熱
   - デバイ温度

5. **応用**
   - 熱膨張係数（準調和近似）
   - 熱伝導率
   - 相転移の検出

### 重要なポイント

- フォノンは量子化された格子振動
- 調和近似で十分な精度（多くの場合）
- 熱力学特性はフォノンから計算可能
- 実験との良い一致（Si: デバイ温度、熱膨張係数）

### 次の章へ

第5章では、DFT計算と機械学習を統合した最新手法を学びます。

---

## 演習問題

### 問題1（難易度：easy）

音響モードと光学モードの違いを、物理的意味とともに説明してください。

<details>
<summary>解答例</summary>

**音響モード（Acoustic modes）**:

**数**: 3本（1 LA + 2 TA）
**特徴**: Γ点（$\mathbf{k}=0$）で周波数がゼロ（$\omega \to 0$ as $\mathbf{k} \to 0$）
**物理的意味**: 結晶全体の並進運動、音波に対応
- すべての原子が同じ方向・位相で振動
- 長波長極限で弾性波（音波）になる

**光学モード（Optical modes）**:

**数**: $3(N_{\text{atom}} - 1)$本（単位胞内原子数$N_{\text{atom}}$）
**特徴**: Γ点で有限の周波数（$\omega(\mathbf{k}=0) \neq 0$）
**物理的意味**: 単位胞内の原子の相対運動、光（赤外線）と結合
- 異なる原子が逆位相で振動
- イオン結晶では電気双極子モーメントが生じる → 赤外吸収

**例（NaCl結晶）**:
- 音響モード: NaとClが同じ方向に動く
- 光学モード: NaとClが逆方向に動く → 双極子 → 光吸収

</details>

### 問題2（難易度：medium）

デバイ温度$\theta_D$の物理的意味と、比熱との関係を説明してください。

<details>
<summary>解答例</summary>

**デバイ温度$\theta_D$の定義**:

デバイモデルでは、すべてのフォノンモードを単一のデバイ周波数$\omega_D$で近似します：

$$
\theta_D = \frac{\hbar\omega_D}{k_B}
$$

**物理的意味**:

1. **量子効果の境界温度**:
   - $T \ll \theta_D$: 量子効果が支配的（低温）
   - $T \gg \theta_D$: 古典的振る舞い（高温）

2. **フォノン励起の目安**:
   - $\theta_D$は最高フォノン周波数に対応
   - $T < \theta_D$: 高エネルギーフォノンは励起されない
   - $T > \theta_D$: すべてのフォノンモードが励起

**比熱との関係**:

**低温（$T \ll \theta_D$）**: Debye $T^3$則

$$
C_V \propto T^3
$$

フォノンの量子効果が顕著。

**高温（$T \gg \theta_D$）**: Dulong-Petit則

$$
C_V \to 3Nk_B = 3R
$$

すべての自由度が励起され、古典的極限。

**典型的な値**:
- Si: $\theta_D \approx 645$ K
- Diamond: $\theta_D \approx 2230$ K（硬い格子 → 高周波）
- Pb: $\theta_D \approx 105$ K（柔らかい格子 → 低周波）

**実用的意味**:
- $\theta_D$が高い → 室温でも量子効果が重要
- $\theta_D$が低い → 室温で古典的

</details>

### 問題3（難易度：hard）

準調和近似（QHA）で熱膨張係数を計算する際、なぜ複数の体積でフォノン計算が必要なのか説明してください。

<details>
<summary>解答例</summary>

**調和近似の限界**:

調和近似では、自由エネルギー$F(V, T)$の体積依存性が以下のようになります：

$$
F_{\text{harm}}(V, T) = U_0(V) + k_B T \sum_{\mathbf{q},j} \ln\left[2\sinh\left(\frac{\hbar\omega_{\mathbf{q}j}(V)}{2k_B T}\right)\right]
$$

ここで$\omega_{\mathbf{q}j}(V)$は体積$V$に依存します。

しかし、調和近似では**格子定数が一定**と仮定しているため、以下の問題があります：

1. **熱膨張が記述できない**: $(∂V/∂T)_P = 0$
2. **熱容量が定積のみ**: $C_P = C_V$（実際は$C_P > C_V$）

**準調和近似（QHA）の導入**:

QHAでは、**異なる体積で独立に調和近似を適用**します：

**手順**:

1. 複数の体積$V_1, V_2, \ldots, V_n$でフォノン計算
2. 各体積で自由エネルギー$F(V_i, T)$を計算
3. 各温度$T$で自由エネルギーを最小化する体積$V_{\text{eq}}(T)$を見つける：
   $$\left(\frac{\partial F}{\partial V}\right)_T = 0$$
4. 熱膨張係数を計算：
   $$\alpha(T) = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_P$$

**なぜ複数の体積が必要か**:

- 体積依存性$F(V, T)$を知るため
- 自由エネルギーの曲率$(\partial^2 F/\partial V^2)$が必要
- 最小値の位置$V_{\text{eq}}(T)$が温度依存

**QHAの仮定**:

- フォノン周波数の体積依存性を考慮
- しかし、フォノン間の非調和相互作用は無視
- 各体積で独立に調和近似

**QHAの限界**:

- 高温（融点近く）では不正確
- 強い非調和性（例: 負の熱膨張材料）は記述困難
- 完全な非調和計算（TDEP, SSCHAなど）が必要

**計算コスト**:

- 5-10個の体積でフォノン計算 → 5-10倍のコスト
- しかし、熱膨張係数が得られる利点

</details>

---

## データライセンスと引用

### 使用したソフトウェア

1. **Phonopy - Phonon calculation tool** (BSD 3-Clause)
   - フォノン計算ソフトウェア
   - URL: https://phonopy.github.io/phonopy/
   - 引用: Togo, A., & Tanaka, I. (2015). *Scr. Mater.*, 108, 1-5.

2. **GPAW** (GPL v3) + **ASE** (LGPL v2.1+)
   - DFT力計算バックエンド
   - 第2章と同様のライセンス

### 計算パラメータ

- **変位振幅**: 0.01 Å（調和近似の標準値）
- **Supercellサイズ**: 2×2×2以上（収束必須）
- **q点メッシュ**: 20×20×20以上（DOSとenthalpy計算）

---

## コード再現性チェックリスト

### 環境構築

```bash
conda create -n phonon python=3.11
conda activate phonon
conda install -c conda-forge phonopy gpaw ase matplotlib
pip install spglib  # 対称性解析（推奨）
```

### 計算時間の見積もり

| 系 | Supercell | 変位数 | DFT時間/変位 | 総時間 |
|---|----------|-------|-------------|-------|
| Si (2原子) | 2×2×2 | 3-6 | 10分 | ~1時間 |
| NaCl (2原子) | 2×2×2 | 6-12 | 15分 | ~3時間 |
| GaAs (2原子) | 3×3×3 | 6-12 | 30分 | ~6時間 |

### トラブルシューティング

**問題**: 虚数フォノン（負の周波数²）
**解決**:
```python
# 1. 構造最適化が不十分
# → 力の閾値を0.01 eV/Å以下に
# 2. Supercellが小さい
# → 2×2×2 → 3×3×3に拡大
# 3. 対称性の破れ
# → spglibで対称性をチェック
```

---

## 実践的な落とし穴と対策

### 1. Supercellサイズ不足

```python
# ❌ 不十分: 1×1×1（収束しない）
phonon = Phonopy(unitcell, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# ✅ 推奨: 2×2×2以上（収束テスト必須）
for dim in [(2,2,2), (3,3,3), (4,4,4)]:
    phonon = Phonopy(unitcell, np.diag(dim))
    # フォノン周波数の収束をチェック
```

### 2. 構造最適化の不完全

```python
# ❌ 不十分: 粗い最適化 → 虚数フォノン
opt.run(fmax=0.1)  # 力が大きすぎ

# ✅ 推奨: 高精度最適化
opt.run(fmax=0.01)  # 0.01 eV/Å以下
```

### 3. q点メッシュ不足（DOS計算）

```python
# ❌ 粗い: 10×10×10（構造を見逃す）
phonon.set_mesh([10, 10, 10])

# ✅ 推奨: 20×20×20以上
phonon.set_mesh([20, 20, 20])
```

### 4. 単位換算ミス

```python
# Phonopy内部単位: THz
# 変換:
# - cm⁻¹ = THz × 33.356
# - meV = THz × 4.136
# - K (温度) = THz × 47.992
```

---

## 品質保証チェックリスト

### フォノン計算の妥当性

- [ ] Γ点（k=0）の音響モードが0 Hz（誤差 < 0.1 THz）
- [ ] 音響モードが3本（並進自由度）
- [ ] すべての周波数が実数（虚数モードなし）
- [ ] 高対称点でバンドが縮退
- [ ] DOSの総和が3N（自由度数）

### 熱力学特性の妥当性

- [ ] 高温極限で比熱 → 3R（Dulong-Petit）
- [ ] デバイ温度が文献値の±20%以内
- [ ] 熱膨張係数が正（通常の材料）
- [ ] エントロピーが温度とともに単調増加

---

## 参考文献

1. Dove, M. T. (1993). *Introduction to Lattice Dynamics*. Cambridge University Press.

2. Togo, A., & Tanaka, I. (2015). "First principles phonon calculations in materials science." *Scripta Materialia*, 108, 1-5.
   DOI: [10.1016/j.scriptamat.2015.07.021](https://doi.org/10.1016/j.scriptamat.2015.07.021)

3. Phonopy Documentation: https://phonopy.github.io/phonopy/

4. Shulumba, N., et al. (2017). "Temperature-dependent elastic properties of Ti$_x$Zr$_{1-x}$N alloys." *Applied Physics Letters*, 111, 061901.

---

## 著者情報

**作成者**: MI Knowledge Hub Content Team
**監修**: Dr. Yusuke Hashimoto（東北大学）
**作成日**: 2025-10-17
**バージョン**: 1.0
**シリーズ**: 計算材料科学基礎入門 v1.0

**ライセンス**: Creative Commons BY-NC-SA 4.0
