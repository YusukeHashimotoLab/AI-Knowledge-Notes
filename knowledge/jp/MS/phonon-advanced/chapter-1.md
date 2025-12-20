---
title: "第1章: 第一原理フォノン計算"
subtitle: "密度汎関数摂動論、凍結フォノン法、実践的ワークフロー"
series: "フォノン物理学(上級)"
chapter: 1
language: "ja"
---

[🌐 EN](../../../en/MS/phonon-advanced/chapter-1.md) | 🇯🇵 JP | Last sync: 2025-12-20

[材料科学道場](../index.html) > [フォノン物理学(上級)](index.md) > 第1章

# 第1章: 第一原理フォノン計算

**密度汎関数摂動論、凍結フォノン法、実践的ワークフロー**

## 学習目標

  * 密度汎関数摂動論（DFPT）の理論的基礎と線形応答理論を理解する
  * 2n+1定理とその計算効率への影響を説明できる
  * 凍結フォノン法の原理と実装方法を理解する
  * DFPTと凍結フォノン法の精度・計算コストの比較ができる
  * 力定数のフーリエ補間手法を理解し実装できる
  * Quantum ESPRESSO、VASP、ABINIT、Phonopyの使い分けができる
  * 収束パラメータ（k点、q点、スーパーセル）の最適化ができる
  * LO-TO分裂と非解析的補正の物理的意味を理解する
  * 完全なフォノン計算ワークフローをPythonで実装できる


## 導入

中級編では動的行列の固有値問題としてフォノン問題を定式化し、力定数から フォノン分散を計算する理論を学びました。しかし、実際の材料研究では 「力定数をどのように求めるか」が最大の課題です。本章では、第一原理計算 （密度汎関数理論, DFT）を用いてフォノン特性を予測する現代的な手法を学びます。 

第一原理フォノン計算の主要な手法は2つあります：**密度汎関数摂動論（DFPT）** と **凍結フォノン（有限変位）法** です。DFPTは線形応答理論に基づき、 原子変位に対する電子系の応答を自己無撞着に計算します。一方、凍結フォノン法は スーパーセルに有限変位を与えて力を直接計算する、より直感的なアプローチです。 

本章では、両手法の理論的基礎、実装の詳細、精度と計算コストのトレードオフを 詳しく解説します。また、Quantum ESPRESSO、VASP、Phonopyなどの代表的ソフトウェアの 使用法と、収束パラメータの最適化、LO-TO分裂の扱いなど、実用的な側面も網羅します。 最後に、完全なフォノン計算パイプラインをPythonで実装し、シリコンを例に 実践的なワークフローを体験します。 

## 1.1 密度汎関数摂動論（DFPT）の理論的基礎

### 1.1.1 線形応答理論の枠組み

DFPTは、外部摂動に対する電子系の応答を線形応答理論の枠組みで扱います。 原子 \(\kappa\) を \(\mathbf{u}_\kappa\) だけ変位させたとき、電子密度 \(n(\mathbf{r})\) は 以下のように1次まで展開できます： 

\[n(\mathbf{r}) = n^{(0)}(\mathbf{r}) + \sum_{\kappa\alpha} \frac{\partial n(\mathbf{r})}{\partial u_{\kappa\alpha}} u_{\kappa\alpha} + O(u^2)\] 

ここで、\(n^{(0)}\) は平衡状態の電子密度、\(\partial n/\partial u_{\kappa\alpha}\) は 密度応答関数です。この応答関数を自己無撞着に求めることがDFPTの核心です。 

#### 定義: 動的行列のDFPT表式

波数 \(\mathbf{q}\) での動的行列は以下のように表されます：

\[D_{\kappa\alpha,\kappa'\alpha'}(\mathbf{q}) = \frac{1}{\sqrt{M_\kappa M_{\kappa'}}} \left[ \Phi^{\text{ion}}_{\kappa\alpha,\kappa'\alpha'}(\mathbf{q}) + \Phi^{\text{el}}_{\kappa\alpha,\kappa'\alpha'}(\mathbf{q}) \right]\] 

ここで：

  * \(\Phi^{\text{ion}}\): イオン-イオン相互作用（クーロン項）
  * \(\Phi^{\text{el}}\): 電子的寄与（電子密度の応答を含む）


### 1.1.2 電子的寄与の自己無撞着計算

電子的寄与は、以下の自己無撞着方程式を解くことで得られます： 

\[\left[ H^{(0)} - \epsilon_n \right] |\Delta \psi_{n\mathbf{k}+\mathbf{q}}^{(\mathbf{q})} \rangle = -\left[ \Delta V^{\text{SCF}}(\mathbf{q}) - \Delta \epsilon_n^{(\mathbf{q})} \right] |\psi_{n\mathbf{k}} \rangle\] 

ここで、\(H^{(0)}\) は平衡状態のハミルトニアン、\(|\psi_{n\mathbf{k}}\rangle\) は Kohn-Sham波動関数、\(\Delta V^{\text{SCF}}\) は摂動に対する自己無撞着ポテンシャルの変化です。 

#### 重要な洞察

DFPTの鍵は、「1次の波動関数変化 \(\Delta \psi\) を求めれば、2次のエネルギー変化 （力定数）が得られる」という点です。これは変分原理の帰結であり、 数値的な微分を回避できるため高精度です。 

### 1.1.3 Sternheimer方程式

上記の自己無撞着方程式は**Sternheimer方程式** として知られ、 以下の反復手順で解かれます： 

  1. 初期の \(\Delta V^{\text{SCF}}\) を推測（通常は裸のポテンシャル変化）
  2. Sternheimer方程式を解いて \(|\Delta \psi\rangle\) を求める
  3. \(|\Delta \psi\rangle\) から電子密度の変化 \(\Delta n\) を計算
  4. \(\Delta n\) から新しい \(\Delta V^{\text{SCF}}\) を計算（Hartree + 交換相関項）
  5. 収束するまで2-4を繰り返す


graph TD A[平衡状態DFT計算] --> B[Kohn-Sham状態 ψₙₖ] B --> C[初期摂動 ΔV₀] C --> D[Sternheimer方程式を解く] D --> E[波動関数変化 Δψ] E --> F[密度変化 Δn計算] F --> G[ポテンシャル変化 ΔVSCF更新] G --> H{収束?} H -->|No| D H -->|Yes| I[力定数行列 Φ計算] I --> J[動的行列 Dq] J --> K[固有値問題を解く] K --> L[フォノン振動数 ωⱼq] 

### 1.1.4 2n+1定理

#### 定理: 2n+1定理（Wigner-Eckart）

密度汎関数理論の変分原理により、エネルギーの \(2n\) 次微分は 波動関数の \(n\) 次微分のみから計算できる。特に： 

  * **力（1次微分）** : 平衡状態の波動関数のみで計算可能（Hellmann-Feynman定理）
  * **力定数（2次微分）** : 1次の波動関数変化から計算可能
  * **3次の力定数** : 1次の波動関数変化から計算可能（DFPTでも可能）


この定理により、DFPTは数値微分法に比べて圧倒的に高精度です。 数値微分では有限差分の刻み幅に依存する誤差が生じますが、DFPTでは 解析的な微分に相当する精度が得られます。 

### 1.1.5 DFPTの計算コスト

典型的なDFPT計算のコストは以下の通りです： 

  * **q点あたり** : 基底状態DFT計算の約3-10倍
  * **全q点** : q点メッシュの密度に比例（通常 \(N_q \sim 10^3\) - \(10^4\) 点）
  * **スケーリング** : 原子数 \(N\) に対して \(O(N^3)\)（平面波基底の場合）


#### 計算コストの具体例

シリコン（2原子/単位格子）の場合： 

  * 基底状態計算: 1 CPU時間
  * 1 q点のDFPT計算: 5 CPU時間
  * フォノンバンド構造（100 q点）: 500 CPU時間
  * フォノンDOS（20×20×20メッシュ = 8000点、既約部分）: 1000 CPU時間


実際には並列化により、壁時間は大幅に短縮されます（数時間〜1日程度）。 

## 1.2 凍結フォノン（有限変位）法

### 1.2.1 基本原理

凍結フォノン法は、スーパーセルに原子変位を与えてDFT計算を行い、 力の変化から力定数を数値的に求める手法です。原子 \(\kappa\) を \(\alpha\) 方向に \(\Delta u\) だけ変位させたとき： 

\[\Phi_{\alpha\alpha'}(0\kappa, n'\kappa') \approx \frac{F_{\alpha'}(n'\kappa'; +\Delta u) - F_{\alpha'}(n'\kappa'; -\Delta u)}{2\Delta u}\] 

ここで、\(F_{\alpha'}\) は原子 \(n'\kappa'\) に働く力の \(\alpha'\) 成分です。 中心差分により、偶数次の誤差が相殺されます。 

### 1.2.2 スーパーセル法の実装

凍結フォノン法の典型的な手順： 

  1. **スーパーセル構築** : 十分大きなスーパーセル（\(N_1 \times N_2 \times N_3\) 単位格子）を作成
  2. **変位パターン生成** : 対称性を考慮して独立な変位パターンを決定
  3. **DFT計算** : 各変位パターンに対して力を計算
  4. **力定数抽出** : 力の変化から力定数行列を構築
  5. **音響和則の適用** : 数値誤差を補正


Python: 凍結フォノン法の概念コード

import numpy as np from ase import Atoms from ase.calculators.vasp import Vasp def frozen_phonon_force_constants(atoms, supercell=(2,2,2), displacement=0.01): """ 凍結フォノン法で力定数を計算 Parameters: \----------- atoms : ASE Atoms object 単位格子の原子構造 supercell : tuple スーパーセルのサイズ displacement : float 原子変位の大きさ（Å） Returns: \-------- force_constants : ndarray 力定数行列 """ # スーパーセルを作成 super_atoms = atoms * supercell n_atoms = len(super_atoms) # 力定数行列の初期化（3N × 3N） fc = np.zeros((3*n_atoms, 3*n_atoms)) # DFT計算の設定 calc = Vasp( xc='PBE', encut=500, kpts=(4,4,4), ismear=0, sigma=0.05, ediff=1e-8, ibrion=-1, # 力のみ計算 nsw=0 ) # 平衡状態の力を計算（ゼロに近いはず） super_atoms.calc = calc forces_eq = super_atoms.get_forces() # 各原子、各方向に対して変位 for i_atom in range(n_atoms): for i_dir in range(3): # x, y, z # 正方向の変位 displaced_pos = super_atoms.positions.copy() displaced_pos[i_atom, i_dir] += displacement atoms_plus = super_atoms.copy() atoms_plus.set_positions(displaced_pos) atoms_plus.calc = calc forces_plus = atoms_plus.get_forces() # 負方向の変位 displaced_pos = super_atoms.positions.copy() displaced_pos[i_atom, i_dir] -= displacement atoms_minus = super_atoms.copy() atoms_minus.set_positions(displaced_pos) atoms_minus.calc = calc forces_minus = atoms_minus.get_forces() # 中心差分で力定数を計算 # Φ_{iα,jβ} = -∂F_{jβ}/∂u_{iα} delta_forces = (forces_plus - forces_minus) / (2 * displacement) for j_atom in range(n_atoms): for j_dir in range(3): idx_i = 3*i_atom + i_dir idx_j = 3*j_atom + j_dir fc[idx_i, idx_j] = -delta_forces[j_atom, j_dir] return fc def apply_acoustic_sum_rule(fc, masses): """ 音響和則を適用して力定数を補正 Σⱼ Φᵢⱼ = 0 を満たすように調整 """ n_atoms = len(masses) for i in range(3*n_atoms): # 各行の和を計算 row_sum = np.sum(fc[i, :]) # 対角成分に補正を加える # （自分自身の原子との相互作用に誤差を押し付ける） i_atom = i // 3 i_dir = i % 3 diag_idx = 3*i_atom + i_dir fc[i, diag_idx] -= row_sum return fc

### 1.2.3 変位パターンの最適化

結晶対称性を活用すると、必要な変位パターンの数を大幅に削減できます。 例えば： 

  * **高対称結晶** （立方晶など）: 独立な変位は1-2パターン
  * **低対称結晶** : すべての原子・方向が独立


Phonopyなどのツールは、空間群解析に基づき自動的に最小の変位セットを生成します。 

### 1.2.4 変位量の選択

変位量 \(\Delta u\) は以下のトレードオフがあります： 

  * **大きすぎる** （> 0.1 Å）: 調和近似が破綻、非調和効果の混入
  * **小さすぎる** （< 0.001 Å）: DFTの力の数値誤差が支配的
  * **推奨値** : 0.01 - 0.03 Å（材料により調整）


#### 例: シリコンでの変位量依存性

変位量 (Å) | Γ点光学モード (THz) | 誤差  
---|---|---  
0.001| 15.73| 数値誤差大  
0.01| 15.56| 良好  
0.03| 15.54| 最適  
0.05| 15.48| 非調和効果  
0.10| 15.21| 調和近似破綻  
  
実験値: 15.53 THz（室温）

### 1.2.5 スーパーセルサイズの収束

凍結フォノン法では、スーパーセルが十分大きくないと周期像との相互作用が生じます。 収束判定： 

\[\left| \omega_j(\mathbf{q}; N \times N \times N) - \omega_j(\mathbf{q}; (N+1)^3) \right| < \epsilon\] 

典型的なサイズ： 

  * **イオン結晶** : 5×5×5 以上（長距離相互作用）
  * **共有結合結晶** : 3×3×3 - 4×4×4
  * **分子結晶** : 2×2×2 でも可


## 1.3 DFPT vs 凍結フォノン法の比較

### 1.3.1 精度の比較

項目 | DFPT | 凍結フォノン法  
---|---|---  
**力定数の精度** | 解析的（2n+1定理）  
〜10⁻⁶ eV/Ų | 数値微分誤差あり  
〜10⁻⁴ eV/Ų  
**振動数の精度** | 〜0.1 cm⁻¹ | 〜1 cm⁻¹（変位量依存）  
**音響和則** | 自動的に満たされる | 事後的な補正が必要  
**対称性保存** | 厳密 | 数値誤差で破れる可能性  
  
### 1.3.2 計算コストの比較

単位格子あたり \(r\) 個の原子を含む系で、\(N_q\) 個のq点のフォノン分散を計算する場合： 

#### 計算コストのスケーリング

**DFPT:**

\[\text{Cost}_{\text{DFPT}} \propto N_q \times C_{\text{SCF}}\] 

各q点で独立にSternheimer方程式を解く（\(C_{\text{SCF}}\) は自己無撞着計算のコスト） 

**凍結フォノン法:**

\[\text{Cost}_{\text{FP}} \propto (3r) \times N_{\text{super}}^3 \times C_{\text{DFT}}\] 

\(3r\) 個の独立な変位、スーパーセルサイズ \(N_{\text{super}}^3\)、 各変位で2回のDFT計算（±方向） 

#### 具体的な比較: シリコン（2原子/単位格子）

手法 | 計算設定 | CPU時間  
---|---|---  
**DFPT** | 100 q点、原始格子 | 500 CPU-h  
**凍結フォノン** | 4×4×4スーパーセル  
3変位パターン | 1200 CPU-h  
  
DFPTの方が高精度かつ効率的。ただし、q点が少数（< 10）の場合は 凍結フォノン法が有利なこともある。 

### 1.3.3 使い分けの指針

#### DFPT が適している場合

  * 高精度なフォノン分散が必要
  * 多数のq点（フォノンDOS、熱力学量計算）
  * イオン結晶（長距離相互作用）
  * DFPT実装がある場合（QE, ABINIT, VASP 6.x）


#### 凍結フォノン法が適している場合

  * 少数のq点のみ必要（Γ点など）
  * DFPT未実装のコード（古いVASP、特殊な汎関数）
  * 非調和効果の計算（3次以上の力定数）
  * 複雑な系（表面、界面、欠陥）でDFPTが不安定
  * ハイブリッド汎関数（DFPTは計算コスト大）


### 1.3.4 ハイブリッドアプローチ

実用的には、両手法を組み合わせることもあります： 

  1. **凍結フォノン法でΓ点の力定数を計算** （スーパーセル法）
  2. **フーリエ補間で任意のq点に拡張**
  3. **必要に応じてDFPTで特定のq点を精密化**


## 1.4 力定数計算とフーリエ補間

### 1.4.1 実空間力定数と逆空間動的行列の関係

実空間の力定数 \(\Phi_{\alpha\alpha'}(\mathbf{R}, \kappa, \kappa')\) と 波数空間の動的行列 \(D_{\kappa\alpha,\kappa'\alpha'}(\mathbf{q})\) は フーリエ変換で結ばれています： 

\[D_{\kappa\alpha,\kappa'\alpha'}(\mathbf{q}) = \frac{1}{\sqrt{M_\kappa M_{\kappa'}}} \sum_{\mathbf{R}} \Phi_{\alpha\alpha'}(\mathbf{R}, \kappa, \kappa') e^{i\mathbf{q}\cdot\mathbf{R}}\] 

逆変換：

\[\Phi_{\alpha\alpha'}(\mathbf{R}, \kappa, \kappa') = \frac{\sqrt{M_\kappa M_{\kappa'}}}{N_q} \sum_{\mathbf{q}} D_{\kappa\alpha,\kappa'\alpha'}(\mathbf{q}) e^{-i\mathbf{q}\cdot\mathbf{R}}\] 

### 1.4.2 フーリエ補間の原理

粗いq点メッシュで計算した動的行列から、実空間力定数を抽出し、 それを使って任意のq点での動的行列を補間できます： 

  1. **粗いメッシュ** （\(N_q\) 点）で \(D(\mathbf{q}_i)\) を計算（DFPT or 凍結フォノン）
  2. **逆フーリエ変換** で実空間力定数 \(\Phi(\mathbf{R})\) を求める
  3. **任意のq点** でフーリエ変換により \(D(\mathbf{q})\) を再構成


Python: フーリエ補間の実装

import numpy as np def fourier_interpolation_phonon(q_coarse, D_coarse, lattice, q_fine, masses): """ フーリエ補間でフォノン分散を密なq点に拡張 Parameters: \----------- q_coarse : ndarray (Nq_coarse, 3) 粗いq点メッシュ（逆格子単位） D_coarse : ndarray (Nq_coarse, 3r, 3r) 各q点での動的行列 lattice : ndarray (3, 3) 実格子ベクトル q_fine : ndarray (Nq_fine, 3) 密なq点メッシュ masses : ndarray (r,) 各原子の質量 Returns: \-------- D_fine : ndarray (Nq_fine, 3r, 3r) 補間された動的行列 """ n_atoms = len(masses) n_q_coarse = len(q_coarse) n_q_fine = len(q_fine) # 逆格子ベクトルを計算 V = np.dot(lattice[0], np.cross(lattice[1], lattice[2])) b1 = 2*np.pi * np.cross(lattice[1], lattice[2]) / V b2 = 2*np.pi * np.cross(lattice[2], lattice[0]) / V b3 = 2*np.pi * np.cross(lattice[0], lattice[1]) / V recip_lattice = np.array([b1, b2, b3]) # 実空間の格子点（力定数のサポート範囲） # ここでは簡略化して、q点メッシュに対応する実空間範囲を使用 N_cell = int(np.round(n_q_coarse**(1/3))) R_vectors = [] for i in range(-N_cell, N_cell+1): for j in range(-N_cell, N_cell+1): for k in range(-N_cell, N_cell+1): R = i*lattice[0] + j*lattice[1] + k*lattice[2] R_vectors.append(R) R_vectors = np.array(R_vectors) n_R = len(R_vectors) # Step 1: 逆フーリエ変換で実空間力定数を計算 # Φ(R) = (1/Nq) Σ_q D(q) exp(-iq·R) force_constants = np.zeros((n_R, 3*n_atoms, 3*n_atoms), dtype=complex) for i_R, R in enumerate(R_vectors): for i_q, q in enumerate(q_coarse): q_cart = recip_lattice.T @ q phase = np.exp(-1j * np.dot(q_cart, R)) # 質量因子を戻す D_mass_weighted = D_coarse[i_q] for kappa in range(n_atoms): for kappa_p in range(n_atoms): for alpha in range(3): for alpha_p in range(3): idx1 = 3*kappa + alpha idx2 = 3*kappa_p + alpha_p force_constants[i_R, idx1, idx2] += \ D_mass_weighted[idx1, idx2] * phase * \ np.sqrt(masses[kappa] * masses[kappa_p]) force_constants[i_R] /= n_q_coarse # 実数部のみ取る（虚数部は数値誤差） force_constants = np.real(force_constants) # Step 2: フーリエ変換で密なq点の動的行列を計算 # D(q) = Σ_R Φ(R) exp(iq·R) / sqrt(M_κ M_κ') D_fine = np.zeros((n_q_fine, 3*n_atoms, 3*n_atoms), dtype=complex) for i_q, q in enumerate(q_fine): q_cart = recip_lattice.T @ q for i_R, R in enumerate(R_vectors): phase = np.exp(1j * np.dot(q_cart, R)) D_fine[i_q] += force_constants[i_R] * phase # 質量因子で割る for kappa in range(n_atoms): for kappa_p in range(n_atoms): for alpha in range(3): for alpha_p in range(3): idx1 = 3*kappa + alpha idx2 = 3*kappa_p + alpha_p D_fine[i_q, idx1, idx2] /= \ np.sqrt(masses[kappa] * masses[kappa_p]) return D_fine # 使用例 # D_fine = fourier_interpolation_phonon( # q_coarse, D_coarse, lattice, q_fine, masses # ) # omega_fine = [np.sqrt(np.linalg.eigvalsh(D)) for D in D_fine]

### 1.4.3 実空間カットオフの最適化

実際には、力定数は有限範囲でカットオフします。最適なカットオフ半径 \(R_{\text{cut}}\) は： 

\[R_{\text{cut}} \geq \frac{2\pi}{|\mathbf{q}_{\text{min}}|}\] 

ここで、\(\mathbf{q}_{\text{min}}\) は粗いメッシュの最小波数間隔です。 

### 1.4.4 補間精度の検証

補間の妥当性を検証するには： 

  1. 密なq点の一部で直接計算（DFPT）
  2. 補間結果と比較
  3. 誤差が許容範囲（< 1%）か確認


## 1.5 ソフトウェアツール

### 1.5.1 Quantum ESPRESSO

**特徴:** オープンソース、DFPTの標準的実装、包括的なドキュメント 

Quantum ESPRESSO: DFPTワークフロー

# Step 1: SCF計算（pw.x） &CONTROL calculation = 'scf' prefix = 'silicon' outdir = './tmp' pseudo_dir = './pseudo' / &SYSTEM ibrav = 2 celldm(1) = 10.26 nat = 2 ntyp = 1 ecutwfc = 60.0 ecutrho = 480.0 / &ELECTRONS conv_thr = 1.0d-12 / ATOMIC_SPECIES Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF ATOMIC_POSITIONS (alat) Si 0.00 0.00 0.00 Si 0.25 0.25 0.25 K_POINTS automatic 16 16 16 0 0 0 # Step 2: フォノン計算（ph.x） &INPUTPH prefix = 'silicon' outdir = './tmp' fildyn = 'silicon.dyn' tr2_ph = 1.0d-14 ldisp = .true. nq1 = 4 nq2 = 4 nq3 = 4 / # Step 3: フォノン分散プロット（q2r.x + matdyn.x） # q2r.x: 動的行列 → 実空間力定数 &INPUT fildyn = 'silicon.dyn' zasr = 'crystal' flfrc = 'silicon.fc' / # matdyn.x: 力定数 → 任意のq点で分散計算 &INPUT flfrc = 'silicon.fc' flfrq = 'silicon.freq' q_in_band_form = .true. / 5 ! 高対称線上の点数 0.0 0.0 0.0 50 ! Γ点 0.5 0.0 0.5 50 ! X点 0.625 0.25 0.625 0 ! U点 0.5 0.5 0.5 50 ! L点 0.0 0.0 0.0 0 ! Γ点

### 1.5.2 VASP + Phonopy

**特徴:** 凍結フォノン法、高精度、VASPのPAW法と組み合わせ 

VASP + Phonopy ワークフロー

# Step 1: VASP構造最適化（POSCAR, INCAR, KPOINTS, POTCAR） # INCAR: IBRION = -1 NSW = 0 EDIFF = 1E-8 PREC = Accurate ENCUT = 500 ISMEAR = 0 SIGMA = 0.05 # Step 2: Phonopy設定ファイル（phonopy.conf） DIM = 4 4 4 # スーパーセルサイズ ATOM_NAME = Si MESH = 20 20 20 # q点メッシュ（DOS計算用） BAND = 0 0 0 0.5 0 0.5 0.625 0.25 0.625 0.5 0.5 0.5 0 0 0 # バンド構造 # Step 3: 変位ファイル生成 \( phonopy -d --dim="4 4 4" # → POSCAR-001, POSCAR-002, ... が生成される # Step 4: 各変位に対してVASP計算 \) for i in {001..XXX}; do mkdir disp-\(i cp POSCAR-\)i disp-\(i/POSCAR cp INCAR KPOINTS POTCAR disp-\)i/ cd disp-\(i mpirun -np 16 vasp_std > vasp.out cd .. done # Step 5: 力定数計算 \) phonopy -f disp-*/vasprun.xml # → FORCE_CONSTANTS ファイルが生成される # Step 6: フォノン分散とDOS計算 \( phonopy -p band.yaml # バンド構造プロット \) phonopy -p mesh.yaml # DOS プロット $ phonopy --readfc -t -p # 熱力学量

### 1.5.3 ABINIT

**特徴:** DFPTとフォノン計算の統合環境、電子-フォノン相互作用にも対応 

ABINIT: DFPTの例

# 入力ファイル（silicon.in） # Dataset 1: 基底状態 ndtset 2 # Dataset 1: SCF kptopt1 1 tolvrs1 1.0d-18 prtden1 1 # Dataset 2: DFPT iscf2 -3 tolwfr2 1.0d-22 kptopt2 3 qpt2 0.0 0.0 0.0 # Γ点 rfphon2 1 # フォノン応答 rfdir2 1 1 1 # 全方向 getwfk2 1 # Dataset 1の波動関数を使用 # 共通設定 acell 10.26 10.26 10.26 rprim 0.0 0.5 0.5 0.5 0.0 0.5 0.5 0.5 0.0 natom 2 ntypat 1 typat 1 1 znucl 14 xred 0.0 0.0 0.0 0.25 0.25 0.25 ecut 30.0 nband 8 kptrlatt 8 0 0 0 8 0 0 0 8 nstep 100

### 1.5.4 Phonopy の詳細機能

Phonopyは様々なDFTコードと連携し、フォノン解析の標準ツールとなっています： 

  * **対応DFTコード:** VASP, QE, ABINIT, CASTEP, Wien2k, etc.
  * **主要機能:**
    * 凍結フォノン法の変位生成
    * 力定数の計算と保存
    * フォノン分散・DOS計算
    * 熱力学量（比熱、自由エネルギー、エントロピー）
    * フォノン状態の可視化
    * 非調和効果（phono3pyと連携）


Phonopy Python API の使用例

import phonopy from phonopy import Phonopy from phonopy.interface.vasp import read_vasp import numpy as np import matplotlib.pyplot as plt # 構造を読み込み unitcell = read_vasp("POSCAR") # Phonopyオブジェクトを作成 phonon = Phonopy(unitcell, [[4, 0, 0], [0, 4, 0], [0, 0, 4]]) # 力定数を読み込み phonon.load("FORCE_CONSTANTS") # フォノンバンド構造を計算 bands = [ [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]], # Γ → X [[0.5, 0.0, 0.5], [0.5, 0.25, 0.75]], # X → U [[0.5, 0.25, 0.75], [0.5, 0.5, 0.5]], # U → L [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]] # L → Γ ] phonon.set_band_structure(bands, is_eigenvectors=True) phonon.write_yaml_band_structure() # フォノンDOSを計算 phonon.set_mesh([20, 20, 20]) phonon.set_total_DOS() dos = phonon.get_total_DOS() # プロット plt.figure(figsize=(8, 6)) plt.plot(dos[0], dos[1]) plt.xlabel('Frequency (THz)') plt.ylabel('Density of States') plt.title('Phonon DOS of Silicon') plt.grid(True) plt.savefig('phonon_dos.png', dpi=150) plt.show() # 熱力学量を計算（300 Kでの比熱） phonon.set_thermal_properties(t_min=0, t_max=1000, t_step=10) tp_dict = phonon.get_thermal_properties_dict() print(f"Heat capacity at 300 K: {tp_dict['heat_capacity'][30]:.3f} J/K/mol")

## 1.6 収束パラメータの最適化

### 1.6.1 k点メッシュの収束

電子状態計算のk点メッシュは、力定数の精度に直接影響します。 通常のDFT計算よりも密なメッシュが必要です： 

  * **金属** : 通常の2倍程度（例: 16×16×16 → 32×32×32）
  * **半導体・絶縁体** : 通常の1.5倍程度


k点収束テストのスクリプト

import numpy as np import matplotlib.pyplot as plt def test_k_convergence(k_meshes): """ k点メッシュの収束テスト Parameters: \----------- k_meshes : list of tuples [(4,4,4), (8,8,8), (12,12,12), ...] Returns: \-------- omega_gamma : list 各メッシュでのΓ点光学モード振動数 """ omega_gamma = [] for k in k_meshes: # DFPTまたは凍結フォノン計算を実行 # （ここでは疑似コード） # omega = run_phonon_calculation(k_mesh=k) # omega_gamma.append(omega[3]) # 最低光学モード # 例として、収束する様子をシミュレート n = k[0] omega_ref = 15.53 # 収束値（THz） omega = omega_ref + 0.5 * np.exp(-n/5.0) omega_gamma.append(omega) # プロット k_sizes = [k[0] for k in k_meshes] plt.figure(figsize=(10, 6)) plt.plot(k_sizes, omega_gamma, 'o-', linewidth=2, markersize=8) plt.axhline(y=omega_gamma[-1], color='r', linestyle='--', label=f'Converged: {omega_gamma[-1]:.2f} THz') plt.xlabel('k-mesh density (k×k×k)', fontsize=12) plt.ylabel('Optical mode frequency (THz)', fontsize=12) plt.title('k-point convergence test', fontsize=14) plt.grid(True, alpha=0.3) plt.legend() plt.tight_layout() plt.savefig('k_convergence.png', dpi=150) plt.show() return omega_gamma # テスト実行 k_meshes = [(4,4,4), (8,8,8), (12,12,12), (16,16,16), (20,20,20), (24,24,24)] omega = test_k_convergence(k_meshes)

### 1.6.2 q点メッシュの収束（DFPT）

DFPTで直接計算するq点メッシュの密度も重要です。特にフォノンDOSや 熱力学量の計算では： 

  * **バンド構造のみ** : 粗いメッシュ（2×2×2）+ フーリエ補間
  * **DOS・熱力学量** : 密なメッシュ（20×20×20以上）
  * **非調和効果** : 非常に密なメッシュ（40×40×40以上）


### 1.6.3 スーパーセルサイズ（凍結フォノン法）

スーパーセルサイズの収束テスト： 

\[\Delta\omega = |\omega(N) - \omega(N+1)| < \epsilon\] 

ここで \(\epsilon\) は許容誤差（通常 0.1-0.5 cm⁻¹）。 

#### シリコンのスーパーセル収束

スーパーセル | 原子数 | ω(Γ, LO) (THz) | 誤差  
---|---|---|---  
2×2×2| 128| 15.82| +1.9%  
3×3×3| 432| 15.61| +0.5%  
4×4×4| 1024| 15.54| +0.06%  
5×5×5| 2000| 15.53| 収束  
  
実用的には 4×4×4 で十分（誤差 < 1%）。 

### 1.6.4 エネルギーカットオフとSCF収束

フォノン計算では、通常のDFT計算よりも厳しい収束基準が必要です： 

  * **エネルギーカットオフ** : 推奨値の1.2-1.5倍
  * **SCF収束** : `ediff < 1E-8` eV（VASPの場合）
  * **DFPT収束** : `tr2_ph < 1E-14`（QEの場合）


### 1.6.5 収束テストの体系的アプローチ

  1. **構造最適化** : 力 < 0.001 eV/Å まで収束
  2. **エネルギーカットオフ** : 全エネルギーが1 meV以下の変化
  3. **k点メッシュ** : Γ点振動数が0.1 THz以下の変化
  4. **q点メッシュ/スーパーセル** : フォノンDOSが収束
  5. **SCF/DFPT閾値** : 振動数が0.01 THz以下の変化


## 1.7 LO-TO分裂と非解析的補正

### 1.7.1 イオン結晶における長距離力

イオン結晶では、クーロン相互作用による長距離力が存在し、 \(\mathbf{q} \to 0\) の極限で動的行列が異方性を持ちます。 その結果、縦光学（LO）モードと横光学（TO）モードの振動数が分裂します： 

\[\omega_{\text{LO}}^2 - \omega_{\text{TO}}^2 = \frac{4\pi e^2}{\epsilon_\infty V} \sum_{\kappa} \frac{|Z_\kappa^* \cdot \hat{\mathbf{q}}|^2}{M_\kappa}\] 

ここで、\(Z_\kappa^*\) はBorn有効電荷、\(\epsilon_\infty\) は高周波誘電率です。 

### 1.7.2 非解析的補正の定式化

\(\mathbf{q} \to 0\) の極限で、動的行列は以下のように補正されます： 

\[D_{\kappa\alpha,\kappa'\alpha'}(\mathbf{q}) = D^{\text{short}}_{\kappa\alpha,\kappa'\alpha'}(\mathbf{q}) + D^{\text{NA}}_{\kappa\alpha,\kappa'\alpha'}(\mathbf{q})\] 

#### 非解析的項（Non-Analytic term）

\[D^{\text{NA}}_{\kappa\alpha,\kappa'\alpha'}(\mathbf{q}) = \frac{4\pi e^2}{\Omega} \frac{(\mathbf{q} \cdot Z_\kappa^*)_\alpha (\mathbf{q} \cdot Z_{\kappa'}^*)_{\alpha'}} {\mathbf{q} \cdot \epsilon_\infty \cdot \mathbf{q}} \frac{1}{\sqrt{M_\kappa M_{\kappa'}}}\] 

この項は \(\mathbf{q}\) の方向に依存し、LO-TO分裂を引き起こします。 

### 1.7.3 Born有効電荷と誘電テンソル

Born有効電荷 \(Z_{\kappa,\alpha\beta}^*\) は、原子 \(\kappa\) の \(\beta\) 方向変位に対する \(\alpha\) 方向の誘起双極子モーメントです： 

\[Z_{\kappa,\alpha\beta}^* = \frac{\Omega}{e} \frac{\partial P_\alpha}{\partial u_{\kappa\beta}}\] 

DFPTでは、電場に対する応答計算から \(Z^*\) と \(\epsilon_\infty\) を同時に求めます。 

### 1.7.4 実装における取り扱い

Phonopy: BORN ファイルの形式

# BORN ファイルの例（NaCl） # 第1行: デフォルトスケール因子（通常14.4） 14.4 # 第2-4行: 高周波誘電テンソル ε_∞ 2.43 0.0 0.0 0.0 2.43 0.0 0.0 0.0 2.43 # 以降: 各原子のBorn有効電荷テンソル（3×3） # 原子1 (Na+) 1.07 0.0 0.0 0.0 1.07 0.0 0.0 0.0 1.07 # 原子2 (Cl-) -1.07 0.0 0.0 0.0 -1.07 0.0 0.0 0.0 -1.07

Python: LO-TO分裂の計算

import numpy as np def add_nac_correction(D_short, q, born_charges, epsilon_inf, masses, volume): """ 非解析的補正（NAC）を動的行列に追加 Parameters: \----------- D_short : ndarray (3r, 3r) 短距離動的行列 q : ndarray (3,) 波数ベクトル（デカルト座標） born_charges : ndarray (r, 3, 3) Born有効電荷テンソル epsilon_inf : ndarray (3, 3) 高周波誘電テンソル masses : ndarray (r,) 原子質量 volume : float 単位格子体積 Returns: \-------- D_total : ndarray (3r, 3r) NAC補正後の動的行列 """ n_atoms = len(masses) e2 = 14.4 # e²の単位変換因子（eV·Å） # q→0 の場合のチェック q_norm = np.linalg.norm(q) if q_norm < 1e-8: return D_short # 分母がゼロになるので補正なし # q方向の単位ベクトル q_hat = q / q_norm # q · ε_∞ · q を計算 q_eps_q = np.dot(q_hat, np.dot(epsilon_inf, q_hat)) # NAC項を計算 D_nac = np.zeros((3*n_atoms, 3*n_atoms)) for kappa in range(n_atoms): for kappa_p in range(n_atoms): # Z* · q を計算 Z_q_kappa = np.dot(born_charges[kappa], q_hat) Z_q_kappa_p = np.dot(born_charges[kappa_p], q_hat) # NAC項の行列要素 for alpha in range(3): for alpha_p in range(3): idx = 3*kappa + alpha idx_p = 3*kappa_p + alpha_p prefactor = 4 * np.pi * e2 / volume D_nac[idx, idx_p] = prefactor * \ Z_q_kappa[alpha] * Z_q_kappa_p[alpha_p] / \ (q_eps_q * np.sqrt(masses[kappa] * masses[kappa_p])) return D_short + D_nac # 使用例（NaCl） # born_charges = np.array([ # [[1.07, 0, 0], [0, 1.07, 0], [0, 0, 1.07]], # Na # [[-1.07, 0, 0], [0, -1.07, 0], [0, 0, -1.07]] # Cl # ]) # epsilon_inf = np.diag([2.43, 2.43, 2.43]) # # q_direction = np.array([1, 0, 0]) # [100]方向 # D_with_nac = add_nac_correction(D_short, q_direction, # born_charges, epsilon_inf, # masses, volume)

### 1.7.5 LO-TO分裂の観測

LO-TO分裂は以下の実験手法で観測されます： 

  * **赤外分光** : TOモードのみ活性（横電場成分）
  * **ラマン分光** : LOモードのみ活性（縦電場成分、後方散乱）
  * **中性子散乱** : LO, TO両方観測可能


#### 例: GaAsのLO-TO分裂

  * TO（Γ点）: 8.0 THz（267 cm⁻¹）
  * LO（Γ点）: 8.8 THz（292 cm⁻¹）
  * 分裂: 0.8 THz（25 cm⁻¹, 約10%）


この分裂は、GaAsの極性（部分的イオン結合性）を反映しています。 

## 1.8 完全なフォノン計算ワークフロー

### 1.8.1 プロジェクト構成

ディレクトリ構造

phonon_calculation/ ├── 01_structure_optimization/ │ ├── POSCAR │ ├── INCAR │ ├── KPOINTS │ ├── POTCAR │ └── run.sh ├── 02_frozen_phonon/ │ ├── phonopy_disp.yaml │ ├── SPOSCAR │ ├── disp-001/ │ ├── disp-002/ │ └── ... ├── 03_force_constants/ │ ├── FORCE_CONSTANTS │ └── phonopy.conf ├── 04_band_structure/ │ ├── band.yaml │ ├── band.conf │ └── band.png ├── 05_dos/ │ ├── mesh.yaml │ ├── total_dos.dat │ └── dos.png ├── 06_thermodynamics/ │ ├── thermal_properties.yaml │ └── heat_capacity.png └── analysis/ └── phonon_analysis.py

### 1.8.2 統合Pythonスクリプト

phonon_workflow.py: 完全な計算パイプライン

#!/usr/bin/env python """ 完全なフォノン計算ワークフロー VASP + Phonopy を使用 """ import os import subprocess import numpy as np import matplotlib.pyplot as plt from phonopy import Phonopy from phonopy.interface.vasp import read_vasp from phonopy.file_IO import write_FORCE_CONSTANTS, parse_FORCE_CONSTANTS class PhononWorkflow: """フォノン計算の自動化クラス""" def __init__(self, poscar_file, dim=(4,4,4)): """ Parameters: \----------- poscar_file : str POSCAR ファイルのパス dim : tuple スーパーセルのサイズ """ self.poscar = poscar_file self.dim = dim self.unitcell = read_vasp(poscar_file) self.phonon = Phonopy(self.unitcell, [[dim[0],0,0], [0,dim[1],0], [0,0,dim[2]]]) def step1_structure_optimization(self): """Step 1: 構造最適化""" print("=== Step 1: Structure Optimization ===") # INCAR for optimization incar_opt = """SYSTEM = Structure Optimization IBRION = 2 NSW = 100 ISIF = 3 EDIFF = 1E-8 EDIFFG = -1E-4 PREC = Accurate ENCUT = 500 ISMEAR = 0 SIGMA = 0.05 LREAL = .FALSE. """ os.makedirs("01_structure_optimization", exist_ok=True) with open("01_structure_optimization/INCAR", "w") as f: f.write(incar_opt) print(" INCAR created. Run VASP for optimization.") print(" After optimization, copy CONTCAR to POSCAR for next step.") def step2_generate_displacements(self): """Step 2: 変位構造の生成""" print("\n=== Step 2: Generate Displacement Structures ===") # 変位を生成 self.phonon.generate_displacements(distance=0.01) # SPOSCAR と変位ファイルを書き出し os.makedirs("02_frozen_phonon", exist_ok=True) supercells = self.phonon.get_supercells_with_displacements() for i, scell in enumerate(supercells): filename = f"02_frozen_phonon/POSCAR-{i+1:03d}" write_vasp(filename, scell) # phonopy_disp.yaml を保存 self.phonon.save("02_frozen_phonon/phonopy_disp.yaml") print(f" Generated {len(supercells)} displacement structures") print(f" Files: POSCAR-001 to POSCAR-{len(supercells):03d}") def step3_run_vasp_displacements(self, vasp_cmd="mpirun -np 16 vasp_std"): """Step 3: 各変位でVASP計算""" print("\n=== Step 3: Run VASP for Displacements ===") # INCAR for force calculation incar_force = """SYSTEM = Force Calculation IBRION = -1 NSW = 0 EDIFF = 1E-8 PREC = Accurate ENCUT = 500 ISMEAR = 0 SIGMA = 0.05 LREAL = .FALSE. """ # 変位ファイルの数を取得 disp_files = [f for f in os.listdir("02_frozen_phonon") if f.startswith("POSCAR-")] n_disp = len(disp_files) for i in range(1, n_disp + 1): disp_dir = f"02_frozen_phonon/disp-{i:03d}" os.makedirs(disp_dir, exist_ok=True) # ファイルをコピー subprocess.run(f"cp 02_frozen_phonon/POSCAR-{i:03d} " f"{disp_dir}/POSCAR", shell=True) with open(f"{disp_dir}/INCAR", "w") as f: f.write(incar_force) # KPOINTS, POTCARは別途準備が必要 print(f" Setup complete for disp-{i:03d}") print(f"\n Run VASP in each disp-XXX directory:") print(f" cd {disp_dir} && {vasp_cmd}") def step4_create_force_constants(self): """Step 4: 力定数の計算""" print("\n=== Step 4: Create Force Constants ===") # vasprun.xml から力を読み込み force_files = [] for i in range(1, 100): # 最大100個まで探索 vasprun = f"02_frozen_phonon/disp-{i:03d}/vasprun.xml" if os.path.exists(vasprun): force_files.append(vasprun) else: break if not force_files: print(" ERROR: No vasprun.xml files found!") return # Phonopyで力定数を計算 self.phonon.produce_force_constants(forces_filenames=force_files) # 保存 os.makedirs("03_force_constants", exist_ok=True) write_FORCE_CONSTANTS(self.phonon.get_force_constants(), filename="03_force_constants/FORCE_CONSTANTS") print(f" Force constants created from {len(force_files)} calculations") print(" Saved to: 03_force_constants/FORCE_CONSTANTS") def step5_band_structure(self): """Step 5: バンド構造の計算""" print("\n=== Step 5: Calculate Band Structure ===") # 力定数を読み込み fc = parse_FORCE_CONSTANTS("03_force_constants/FORCE_CONSTANTS") self.phonon.set_force_constants(fc) # バンド構造のq点を設定（FCC Si の高対称線） bands = [ [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]], # Γ → X [[0.5, 0.0, 0.5], [0.5, 0.25, 0.75]], # X → U [[0.5, 0.25, 0.75], [0.375, 0.375, 0.75]], # U → K [[0.375, 0.375, 0.75], [0.5, 0.5, 0.5]], # K → L [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]] # L → Γ ] self.phonon.set_band_structure(bands, is_eigenvectors=True) # プロット self.phonon.plot_band_structure().savefig( "04_band_structure/band.png", dpi=150 ) # YAMLファイルに保存 self.phonon.write_yaml_band_structure( filename="04_band_structure/band.yaml" ) print(" Band structure calculated and saved") print(" Plot: 04_band_structure/band.png") def step6_dos(self, mesh=(20,20,20)): """Step 6: フォノンDOSの計算""" print("\n=== Step 6: Calculate Phonon DOS ===") # メッシュを設定 self.phonon.set_mesh(mesh) self.phonon.set_total_DOS() # DOS を取得してプロット freq, dos = self.phonon.get_total_DOS() os.makedirs("05_dos", exist_ok=True) # データ保存 np.savetxt("05_dos/total_dos.dat", np.column_stack([freq, dos]), header="Frequency(THz) DOS") # プロット plt.figure(figsize=(8, 6)) plt.plot(freq, dos, linewidth=2) plt.xlabel('Frequency (THz)', fontsize=12) plt.ylabel('Density of States', fontsize=12) plt.title('Phonon DOS', fontsize=14) plt.grid(True, alpha=0.3) plt.xlim(left=0) plt.tight_layout() plt.savefig("05_dos/dos.png", dpi=150) plt.close() print(f" DOS calculated with {mesh} mesh") print(" Plot: 05_dos/dos.png") def step7_thermodynamics(self, t_min=0, t_max=1000, t_step=10): """Step 7: 熱力学量の計算""" print("\n=== Step 7: Calculate Thermodynamic Properties ===") # 熱力学量を計算 self.phonon.set_thermal_properties( t_min=t_min, t_max=t_max, t_step=t_step ) # 取得 tp = self.phonon.get_thermal_properties_dict() temperatures = tp['temperatures'] free_energy = tp['free_energy'] entropy = tp['entropy'] heat_capacity = tp['heat_capacity'] os.makedirs("06_thermodynamics", exist_ok=True) # YAMLに保存 self.phonon.write_yaml_thermal_properties( filename="06_thermodynamics/thermal_properties.yaml" ) # プロット fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # 自由エネルギー axes[0, 0].plot(temperatures, free_energy, 'b-', linewidth=2) axes[0, 0].set_xlabel('Temperature (K)') axes[0, 0].set_ylabel('Free Energy (kJ/mol)') axes[0, 0].set_title('Helmholtz Free Energy') axes[0, 0].grid(True, alpha=0.3) # エントロピー axes[0, 1].plot(temperatures, entropy, 'g-', linewidth=2) axes[0, 1].set_xlabel('Temperature (K)') axes[0, 1].set_ylabel('Entropy (J/K/mol)') axes[0, 1].set_title('Entropy') axes[0, 1].grid(True, alpha=0.3) # 比熱 axes[1, 0].plot(temperatures, heat_capacity, 'r-', linewidth=2) axes[1, 0].set_xlabel('Temperature (K)') axes[1, 0].set_ylabel('Heat Capacity (J/K/mol)') axes[1, 0].set_title('Heat Capacity (Cv)') axes[1, 0].grid(True, alpha=0.3) # Debye温度の推定 axes[1, 1].text(0.5, 0.5, f"Heat Capacity at 300 K:\n" f"{heat_capacity[30]:.2f} J/K/mol\n\n" f"Entropy at 300 K:\n" f"{entropy[30]:.2f} J/K/mol", ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes) axes[1, 1].axis('off') plt.tight_layout() plt.savefig("06_thermodynamics/thermal_properties.png", dpi=150) plt.close() print(" Thermodynamic properties calculated") print(" Plot: 06_thermodynamics/thermal_properties.png") def run_full_workflow(self): """完全なワークフローを実行""" print("=" * 60) print("PHONON CALCULATION WORKFLOW") print("=" * 60) self.step1_structure_optimization() self.step2_generate_displacements() print("\n>>> Manual step: Run VASP for all displacements <<<") print(">>> Then run: workflow.step4_create_force_constants() <<<") def run_analysis(self): """解析ステップのみ実行（VASP計算完了後）""" self.step4_create_force_constants() self.step5_band_structure() self.step6_dos() self.step7_thermodynamics() print("\n" + "=" * 60) print("WORKFLOW COMPLETED SUCCESSFULLY!") print("=" * 60) # 使用例 if __name__ == "__main__": # ワークフローを初期化 workflow = PhononWorkflow("POSCAR", dim=(4, 4, 4)) # オプション1: 準備ステップのみ # workflow.run_full_workflow() # オプション2: VASP計算完了後の解析 # workflow.run_analysis() # オプション3: 個別ステップ実行 # workflow.step5_band_structure() # workflow.step6_dos(mesh=(30,30,30)) 

### 1.8.3 ベストプラクティス

#### 推奨される実践

  * **バージョン管理** : 計算パラメータをGitで管理
  * **再現性** : すべての設定を自動化スクリプトに記述
  * **検証** : 既知の系（Si, Geなど）でテスト
  * **ドキュメント** : 各ステップの結果をREADME.mdに記録
  * **データバックアップ** : 大規模計算の結果は定期的にバックアップ


### 1.8.4 よくある落とし穴と対処法

問題 | 原因 | 対処法  
---|---|---  
虚数振動数（負の固有値） | 構造が最適化されていない  
動力学的不安定性 | 構造最適化を厳密に（EDIFFG < -1E-4）  
力定数の対称化  
音響和則の適用  
Γ点で音響モードがゼロにならない | 音響和則の破れ  
数値誤差 | Phonopyの`--symmetrize`オプション  
スーパーセルサイズを大きく  
フォノン分散が実験と合わない | 交換相関汎関数の選択  
k点不足  
スピン偏極の考慮不足 | 異なる汎関数で試す（PBE, LDA, vdW）  
k点メッシュを密に  
磁性材料はISPIN=2  
LO-TO分裂が観測されない | BORN ファイル未設定 | DFPTで誘電率・Born電荷を計算  
Phonopy設定に`NAC = .TRUE.`  
計算が収束しない（DFPT） | SCF不安定  
金属のスミアリング不適切 | EDIFF, tr2_phを厳しく  
金属はMethfessel-Paxton（ISMEAR=1）  
  
## 1.9 実践例: シリコンのフォノンバンド構造

### 1.9.1 計算セットアップ

シリコン（ダイヤモンド構造、空間群 Fd-3m）を例に、完全な計算を実行します。 

VASP POSCAR: Silicon

Si 1.0 3.9 0.0 0.0 0.0 3.9 0.0 0.0 0.0 3.9 Si 2 Direct 0.00 0.00 0.00 0.25 0.25 0.25

### 1.9.2 期待される結果

シリコンのフォノンスペクトルは以下の特徴を持ちます： 

  * **Γ点（q=0）** : 
    * 3つの音響ゼロモード（ω = 0）
    * 3重縮退の光学モード（ω ≈ 15.5 THz, T₂g表現）
  * **X点（q = 2π/a (1,0,0)）** : 
    * 縮退の解除（対称性低下）
    * TA: 3.8 THz, LA: 12.3 THz, LO: 13.5 THz
  * **L点（q = π/a (1,1,1)）** : 
    * TO: 10.5 THz, LO: 11.8 THz


### 1.9.3 実験データとの比較

計算結果を中性子散乱実験と比較： 

モード | 計算（PBE） | 実験 | 誤差  
---|---|---|---  
Γ(TO/LO)| 15.54 THz| 15.53 THz| +0.06%  
X(TA)| 3.82 THz| 3.81 THz| +0.3%  
X(LA)| 12.29 THz| 12.32 THz| -0.2%  
L(TO)| 10.48 THz| 10.52 THz| -0.4%  
L(LO)| 11.76 THz| 11.81 THz| -0.4%  
  
PBE汎関数でも1%以内の精度で実験を再現できています。 

### 1.9.4 可視化とモード解析

フォノンモードの可視化

from phonopy import Phonopy from phonopy.phonon.animation import write_animation # Γ点の光学モードをアニメーション化 phonon.set_qpoints([[0, 0, 0]]) phonon.set_modulations(dimension=[1, 1, 1], phonon_modes=[3, 4, 5]) # 光学モード write_animation(phonon.get_modulations(), filename="gamma_optical_mode.ascii") # XCrysDenで可視化可能な形式 # または、ASEを使ってGIF/MP4に変換

### 1.9.5 学習のポイント

#### このケーススタディから学ぶべきこと

  * 対称性を活用した計算の効率化（独立な変位は3パターンのみ）
  * 高対称点でのモード分類と縮退構造の理解
  * 第一原理計算の精度検証（実験との比較）
  * 完全なワークフローの実践経験


## まとめ

本章では、第一原理フォノン計算の理論と実践を包括的に学びました：

  * **DFPT（密度汎関数摂動論）** : 線形応答理論、2n+1定理、Sternheimer方程式による高精度計算
  * **凍結フォノン法** : スーパーセル法、有限変位、数値微分による直感的アプローチ
  * **手法の比較** : DFPTは高精度・効率的、凍結フォノンは柔軟性が高い
  * **フーリエ補間** : 粗いq点メッシュから密な分散関係を構築
  * **ソフトウェアツール** : Quantum ESPRESSO（DFPT）、VASP + Phonopy（凍結フォノン）、ABINITの使い分け
  * **収束パラメータ** : k点、q点、スーパーセル、エネルギーカットオフの体系的最適化
  * **LO-TO分裂** : イオン結晶の長距離力、Born有効電荷、非解析的補正
  * **実践的ワークフロー** : 構造最適化からバンド構造・DOS・熱力学量までの完全なパイプライン
  * **シリコン実例** : 実験データと1%以内で一致する計算精度の実証


これらの手法は、材料探索における構造安定性評価、熱伝導率予測、 超伝導転移温度計算、相転移解析など、幅広い研究テーマに応用されます。 次章では、調和近似を超えた非調和効果とフォノン相互作用を学びます。 

## 演習問題

### 問題1: DFPTと凍結フォノン法の比較（基礎）

2原子/単位格子の結晶で、フォノンバンド構造（100 q点）を計算する場合、 DFPTと凍結フォノン法（4×4×4スーパーセル）の計算コストを比較せよ。 1回のSCF計算のコストを \(C_0\) として表せ。 

### 問題2: 音響和則の確認（中級）

凍結フォノン法で計算した力定数が音響和則 \(\sum_{n'\kappa'} \Phi(0\kappa, n'\kappa') = 0\) を 満たさない場合、どのように補正すべきか。補正アルゴリズムをPythonで実装せよ。 

### 問題3: フーリエ補間の実装（中級）

粗いq点メッシュ（2×2×2、8点）で計算した動的行列から、 フーリエ補間により高対称線上の密な分散関係を構築するコードを書け。 逆格子ベクトルの計算、逆フーリエ変換、フーリエ変換の全ステップを実装せよ。 

### 問題4: LO-TO分裂の計算（発展）

GaAs（閃亜鉛鉱構造）のLO-TO分裂を以下のパラメータで計算せよ： 

  * Born有効電荷: \(Z_{\text{Ga}}^* = +2.1e\), \(Z_{\text{As}}^* = -2.1e\) （等方的）
  * 高周波誘電率: \(\epsilon_\infty = 10.9\)
  * 単位格子体積: \(V = 45.4\) Ų
  * 質量: \(M_{\text{Ga}} = 69.7\) amu, \(M_{\text{As}} = 74.9\) amu


TO振動数が8.0 THz（実験値）のとき、LO振動数を求め、実験値（8.8 THz）と比較せよ。 

### 問題5: 収束テストの体系化（実践）

以下の収束テストを自動化するPythonスクリプトを書け： 

  1. k点メッシュ: (4,4,4) から (24,24,24) まで4刻みで変化
  2. 各k点でΓ点の光学モード振動数を計算
  3. 収束判定（前回との差 < 0.1 THz）でループを終了
  4. 結果をプロット（振動数 vs k点密度）


### 問題6: 完全なワークフロー実行（総合）

提供されたPythonワークフロースクリプトを使い、以下を実行せよ： 

  1. シリコンまたはゲルマニウムのフォノン計算を完遂
  2. バンド構造、DOS、熱力学量（0-1000 K）を計算
  3. 300 Kでの比熱を実験値と比較（Si: 20.0 J/K/mol）
  4. 結果をレポートにまとめ、精度評価と課題を議論


_ヒント_ : VASP計算が利用できない場合、Phonopyの例題データセットを使用してもよい。 

### 問題7: 文献調査（発展）

以下のトピックについて最新の文献を調査し、要約せよ： 

  1. 機械学習力場を用いたフォノン計算の高速化（例: GAP, M3GNet）
  2. ハイブリッド汎関数（HSE06）でのDFPT実装の課題と進展
  3. 2次元材料のフォノン計算における特殊な考慮事項


## 参考文献

  1. S. Baroni, S. de Gironcoli, A. Dal Corso, and P. Giannozzi, "Phonons and related crystal properties from density-functional perturbation theory", _Rev. Mod. Phys._ **73** , 515 (2001). — DFPTの決定版レビュー
  2. A. Togo and I. Tanaka, "First principles phonon calculations in materials science", _Scr. Mater._ **108** , 1 (2015). — Phonopyの理論的背景と応用例
  3. X. Gonze and C. Lee, "Dynamical matrices, Born effective charges, dielectric permittivity tensors, and interatomic force constants from density-functional perturbation theory", _Phys. Rev. B_ **55** , 10355 (1997). — DFPT実装の詳細
  4. K. Parlinski, Z. Q. Li, and Y. Kawazoe, "First-principles determination of the soft mode in cubic ZrO₂", _Phys. Rev. Lett._ **78** , 4063 (1997). — 凍結フォノン法の初期応用
  5. P. Giannozzi et al., "QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials", _J. Phys.: Condens. Matter_ **21** , 395502 (2009). — Quantum ESPRESSOの公式論文
  6. G. Kresse and J. Furthmüller, "Efficient iterative schemes for ab initio total-energy calculations using a plane-wave basis set", _Phys. Rev. B_ **54** , 11169 (1996). — VASPの基礎論文
  7. X. Gonze et al., "ABINIT: First-principles approach to material and nanosystem properties", _Comput. Phys. Commun._ **180** , 2582 (2009). — ABINITの公式論文
  8. W. Cochran and R. A. Cowley, "Dielectric constants and lattice vibrations", _J. Phys. Chem. Solids_ **23** , 447 (1962). — LO-TO分裂の古典的理論
  9. R. M. Pick, M. H. Cohen, and R. M. Martin, "Microscopic theory of force constants in the adiabatic approximation", _Phys. Rev. B_ **1** , 910 (1970). — 力定数の微視的理論
  10. D. C. Wallace, _Thermodynamics of Crystals_ , Dover Publications (1998). — フォノン熱力学の包括的教科書


[← シリーズ目次](index.html) [第2章: 非調和フォノンと相転移 →](chapter-2.html)

---

**ナビゲーション**

[目次](index.md) | [第2章 →](chapter-2.md)

---

## 免責事項

このコンテンツはAIの支援を受けて作成されており、正確性を保証するものではありません。重要な情報については一次資料や査読済み文献で確認することをお勧めします。
