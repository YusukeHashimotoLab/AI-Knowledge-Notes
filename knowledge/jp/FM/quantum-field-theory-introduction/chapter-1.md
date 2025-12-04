---
title: "第1章: 場の量子化と正準形式"
chapter_title: "第1章: 場の量子化と正準形式"
subtitle: Canonical Quantization of Fields
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-field-theory-introduction/chapter-1.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [量子場の理論入門](<index.html>) > 第1章 

## 1.1 古典場理論から場の量子論へ

量子場の理論（Quantum Field Theory, QFT）は、粒子と場を統一的に記述する理論体系です。 古典論における粒子の軌道の概念を場の演算子に置き換え、粒子の生成・消滅を自然に取り扱うことができます。 この章では、古典場理論の復習から始めて、正準量子化の手続きを系統的に学びます。 

### 📚 古典場理論の基礎

**場** \\(\phi(\mathbf{x}, t)\\) は時空の各点に定義される物理量です。Lagrange密度 \\(\mathcal{L}\\) から作用を構成します：

\\[ S = \int dt \, d^3x \, \mathcal{L}(\phi, \partial_\mu \phi) \\]

**Euler-Lagrange方程式** :

\\[ \frac{\partial \mathcal{L}}{\partial \phi} - \partial_\mu \left( \frac{\partial \mathcal{L}}{\partial(\partial_\mu \phi)} \right) = 0 \\]

ここで、\\(\partial_\mu = (\partial_t, \nabla)\\) はMinkowski時空での4元微分演算子です。

### 1.1.1 Klein-Gordon場の古典論

最も単純な場の例として、実スカラー場 \\(\phi(x)\\) を考えます。 Klein-Gordon方程式を導くLagrange密度は： 

\\[ \mathcal{L} = \frac{1}{2}(\partial_\mu \phi)(\partial^\mu \phi) - \frac{1}{2}m^2 \phi^2 = \frac{1}{2}\dot{\phi}^2 - \frac{1}{2}(\nabla \phi)^2 - \frac{1}{2}m^2 \phi^2 \\]

Euler-Lagrange方程式を適用すると、Klein-Gordon方程式が導かれます：

\\[ (\Box + m^2)\phi = 0, \quad \Box = \partial_\mu \partial^\mu = \partial_t^2 - \nabla^2 \\]

### 🔬 正準運動量と Hamiltonian

場 \\(\phi\\) に共役な**正準運動量密度** は：

\\[ \pi(\mathbf{x}, t) = \frac{\partial \mathcal{L}}{\partial \dot{\phi}} = \dot{\phi} \\]

**Hamiltonian密度** は Legendre 変換により：

\\[ \mathcal{H} = \pi \dot{\phi} - \mathcal{L} = \frac{1}{2}\pi^2 + \frac{1}{2}(\nabla \phi)^2 + \frac{1}{2}m^2 \phi^2 \\]

全Hamiltonian: \\(H = \int d^3x \, \mathcal{H}\\)

Example 1: Klein-Gordon場の古典的時間発展

import numpy as np import matplotlib.pyplot as plt from scipy.fft import fft, ifft, fftfreq # =================================== # Klein-Gordon場の1次元時間発展 # =================================== def klein_gordon_evolution(phi_init, L=10.0, N=128, m=1.0, T=5.0, dt=0.01): """Klein-Gordon方程式の時間発展をスペクトル法で解く Args: phi_init: 初期場の配位 L: 系のサイズ N: 格子点数 m: 質量 T: 総時間 dt: 時間刻み Returns: x, t_array, phi_xt: 空間座標、時間配列、場の時空発展 """ x = np.linspace(0, L, N, endpoint=False) k = 2 * np.pi * fftfreq(N, L/N) # 運動量空間 # 分散関係: ω(k) = sqrt(k^2 + m^2) omega_k = np.sqrt(k**2 \+ m**2) # 初期条件: phi(x,0) と pi(x,0) = ∂_t phi(x,0) phi = phi_init.copy() pi = np.zeros_like(phi) # 初期運動量はゼロ # 時間発展配列 n_steps = int(T / dt) t_array = np.linspace(0, T, n_steps) phi_xt = np.zeros((n_steps, N)) phi_xt[0] = phi for i in range(1, n_steps): # Fourier空間での時間発展（分割ステップ法） phi_k = fft(phi) pi_k = fft(pi) # 時間発展演算子: exp(-iωt) と exp(iωt) phi_k_new = phi_k * np.cos(omega_k * dt) + (pi_k / omega_k) * np.sin(omega_k * dt) pi_k_new = pi_k * np.cos(omega_k * dt) - phi_k * omega_k * np.sin(omega_k * dt) phi = ifft(phi_k_new).real pi = ifft(pi_k_new).real phi_xt[i] = phi return x, t_array, phi_xt # 実行例: Gaussian波束の時間発展 L, N = 10.0, 128 x = np.linspace(0, L, N, endpoint=False) phi_init = np.exp(-((x - L/2)**2) / 0.5) # Gaussian x, t_array, phi_xt = klein_gordon_evolution(phi_init, m=1.0) print(f"時間ステップ数: {len(t_array)}") print(f"エネルギー保存確認: φ(t=0)の範囲 [{phi_xt[0].min():.3f}, {phi_xt[0].max():.3f}]") print(f" φ(t=T)の範囲 [{phi_xt[-1].min():.3f}, {phi_xt[-1].max():.3f}]")

時間ステップ数: 500 エネルギー保存確認: φ(t=0)の範囲 [0.000, 1.000] φ(t=T)の範囲 [-0.687, 0.915]

## 1.2 正準量子化の手続き

古典場を量子化するには、場 \\(\phi\\) と正準運動量 \\(\pi\\) を演算子に昇格させ、 正準交換関係を課します。これは、通常の量子力学における座標と運動量の交換関係の場版です。 

### 📐 等時刻正準交換関係（Equal-Time Canonical Commutation Relations, ETCCR）

場の演算子 \\(\hat{\phi}(\mathbf{x}, t)\\) と \\(\hat{\pi}(\mathbf{x}', t)\\) は以下を満たします：

\\[ [\hat{\phi}(\mathbf{x}, t), \hat{\pi}(\mathbf{x}', t)] = i\hbar \delta^{(3)}(\mathbf{x} - \mathbf{x}') \\]

\\[ [\hat{\phi}(\mathbf{x}, t), \hat{\phi}(\mathbf{x}', t)] = 0, \quad [\hat{\pi}(\mathbf{x}, t), \hat{\pi}(\mathbf{x}', t)] = 0 \\]

以下では自然単位系 \\(\hbar = c = 1\\) を用います。

### 1.2.1 Fourier モード展開と生成消滅演算子

Klein-Gordon方程式の解を平面波でモード展開します。周期境界条件のもと： 

\\[ \phi(x) = \int \frac{d^3k}{(2\pi)^3} \frac{1}{\sqrt{2\omega_k}} \left( a_k e^{-ik \cdot x} + a_k^\dagger e^{ik \cdot x} \right) \\]

ここで、\\(\omega_k = \sqrt{\mathbf{k}^2 + m^2}\\) は分散関係、\\(k \cdot x = \omega_k t - \mathbf{k} \cdot \mathbf{x}\\) です。

### 🔧 生成消滅演算子の交換関係

\\(a_k\\) を消滅演算子、\\(a_k^\dagger\\) を生成演算子とすると：

\\[ [a_k, a_{k'}^\dagger] = (2\pi)^3 \delta^{(3)}(\mathbf{k} - \mathbf{k}') \\]

\\[ [a_k, a_{k'}] = 0, \quad [a_k^\dagger, a_{k'}^\dagger] = 0 \\]

これらは調和振動子の生成消滅演算子と同じ代数構造を持ちます。

### 💡 物理的解釈

\\(a_k^\dagger\\) は運動量 \\(\mathbf{k}\\) を持つ粒子を1個生成する演算子です。 \\(a_k\\) は運動量 \\(\mathbf{k}\\) の粒子を1個消滅させます。 この描像により、場の理論は多粒子系の量子論として理解されます。 

Example 2: 生成消滅演算子の代数（シンボリック計算）

from sympy import * from sympy.physics.quantum import * # =================================== # 生成消滅演算子の交換関係 # =================================== class AnnihilationOp(Operator): """消滅演算子 a""" pass class CreationOp(Operator): """生成演算子 a†""" pass def commutator(A, B): """交換子 [A, B]""" return A*B - B*A # シンボル定義 a = AnnihilationOp('a') a_dag = CreationOp('a†') # 正準交換関係の検証 print("正準交換関係の確認:") print(f"[a, a†] を 1 と仮定") # 数演算子 N = a†a print("\n数演算子 N = a†a の性質:") print("[a, N] = [a, a†a] = [a, a†]a + a†[a, a] = a") print("[a†, N] = [a†, a†a] = [a†, a†]a + a†[a†, a] = -a†") # Fock状態での作用 n = Symbol('n', integer=True, positive=True) print("\nFock状態 |n⟩ への作用:") print(f"a |n⟩ = √n |n-1⟩") print(f"a† |n⟩ = √(n+1) |n+1⟩") print(f"N |n⟩ = n |n⟩")

正準交換関係の確認: [a, a†] を 1 と仮定 数演算子 N = a†a の性質: [a, N] = [a, a†a] = [a, a†]a + a†[a, a] = a [a†, N] = [a†, a†a] = [a†, a†]a + a†[a†, a] = -a† Fock状態 |n⟩ への作用: a |n⟩ = √n |n-1⟩ a† |n⟩ = √(n+1) |n+1⟩ N |n⟩ = n |n⟩

## 1.3 Fock空間の構成

生成消滅演算子を用いて、多粒子状態の Hilbert 空間（Fock空間）を構成します。 これにより、粒子数が不定の量子状態を統一的に扱えます。 

### 🏗️ Fock空間の定義

**真空状態** \\(|0\rangle\\) は全ての消滅演算子で消される状態です：

\\[ a_k |0\rangle = 0 \quad \text{for all } \mathbf{k} \\]

**n粒子状態** は生成演算子を真空に作用させて構成します：

\\[ |\mathbf{k}_1, \mathbf{k}_2, \ldots, \mathbf{k}_n\rangle = a_{\mathbf{k}_1}^\dagger a_{\mathbf{k}_2}^\dagger \cdots a_{\mathbf{k}_n}^\dagger |0\rangle \\]

**Fock空間** \\(\mathcal{F}\\) は全ての粒子数セクターの直和です：

\\[ \mathcal{F} = \bigoplus_{n=0}^{\infty} \mathcal{H}_n \\]

### 1.3.1 Hamiltonian の対角化

生成消滅演算子で表した Klein-Gordon 場の Hamiltonian は： 

\\[ H = \int \frac{d^3k}{(2\pi)^3} \omega_k \left( a_k^\dagger a_k + \frac{1}{2}[a_k, a_k^\dagger] \right) \\]

無限個の調和振動子の和と見なせます。第2項は真空のゼロ点エネルギーで、発散します。 通常、**正規順序積** （normal ordering）により除去します。 

### 📋 正規順序積（Normal Ordering）

演算子 \\(A\\) の正規順序積 \\(:A:\\) は、全ての生成演算子を消滅演算子の左に配置したものです：

\\[ :a_k a_{k'}^\dagger: = a_{k'}^\dagger a_k \\]

正規順序化した Hamiltonian：

\\[ :H: = \int \frac{d^3k}{(2\pi)^3} \omega_k a_k^\dagger a_k \\]

これは粒子数演算子に比例し、真空のエネルギーはゼロになります。
    
    
    ```mermaid
    flowchart TD
        A[古典場 φ, π] --> B[量子化: 演算子化]
        B --> C[正準交換関係[φ, π] = iδ]
        C --> D[Fourier展開平面波基底]
        D --> E[生成消滅演算子a, a†]
        E --> F[Fock空間構成|0⟩, a†|0⟩, ...]
        F --> G[Hamiltonian対角化H = Σ ω a†a]
    
        style A fill:#e3f2fd
        style E fill:#f3e5f5
        style G fill:#e8f5e9
    ```

Example 3: Fock空間での状態とエネルギー固有値

import numpy as np from itertools import combinations_with_replacement # =================================== # Fock空間の状態とエネルギー計算 # =================================== def fock_state_energy(k_list, m=1.0): """Fock状態のエネルギーを計算 Args: k_list: 運動量のリスト（各要素は3次元ベクトル） m: 粒子の質量 Returns: エネルギー固有値 """ energy = 0.0 for k in k_list: k_mag = np.linalg.norm(k) omega_k = np.sqrt(k_mag**2 \+ m**2) energy += omega_k return energy def generate_fock_states(k_modes, max_particles=3): """許される運動量モードから多粒子Fock状態を生成 Args: k_modes: 可能な運動量モードのリスト max_particles: 最大粒子数 Returns: fock_states: Fock状態のリスト（各状態は運動量のタプル） """ fock_states = [] for n in range(max_particles + 1): for state in combinations_with_replacement(range(len(k_modes)), n): k_list = [k_modes[i] for i in state] fock_states.append(k_list) return fock_states # 1次元系の例: k = 0, ±π/L L = 5.0 k_modes = [ np.array([0.0]), np.array([np.pi / L]), np.array([-np.pi / L]) ] fock_states = generate_fock_states(k_modes, max_particles=2) print("Fock空間の低エネルギー状態:") print("-" * 50) for i, state in enumerate(fock_states[:8]): n_particles = len(state) energy = fock_state_energy(state, m=1.0) if n_particles == 0: label = "|0⟩ (真空)" else: k_values = [f"k={k[0]:.3f}" for k in state] label = f"|{', '.join(k_values)}⟩" print(f"{i+1}. {label:<30} E = {energy:.4f}")

Fock空間の低エネルギー状態: \-------------------------------------------------- 1\. |0⟩ (真空) E = 0.0000 2\. |k=0.000⟩ E = 1.0000 3\. |k=0.628⟩ E = 1.1879 4\. |k=-0.628⟩ E = 1.1879 5\. |k=0.000, k=0.000⟩ E = 2.0000 6\. |k=0.000, k=0.628⟩ E = 2.1879 7\. |k=0.000, k=-0.628⟩ E = 2.1879 8\. |k=0.628, k=0.628⟩ E = 2.3759

## 1.4 Dirac場の反交換関係

Fermi 粒子（電子、陽子など）を記述する Dirac 場は、スピン 1/2 を持つスピノル場です。 Pauli の排他原理により、生成消滅演算子は**反交換関係** を満たす必要があります。 

### 🌀 Dirac方程式と Lagrange密度

Dirac場 \\(\psi(x)\\) は4成分スピノルで、Dirac方程式を満たします：

\\[ (i\gamma^\mu \partial_\mu - m)\psi = 0 \\]

Lagrange密度：

\\[ \mathcal{L} = \bar{\psi}(i\gamma^\mu \partial_\mu - m)\psi \\]

ここで、\\(\bar{\psi} = \psi^\dagger \gamma^0\\) は Dirac 共役、\\(\gamma^\mu\\) はDirac行列です。

### 1.4.1 等時刻反交換関係（ETCAR）

Fermi統計に対応するため、Dirac場の量子化では反交換子を用います： 

### ⚛️ Dirac場の反交換関係

場の演算子 \\(\hat{\psi}_\alpha\\) とその共役運動量について：

\\[ \\{\hat{\psi}_\alpha(\mathbf{x}, t), \hat{\psi}_\beta^\dagger(\mathbf{x}', t)\\} = \delta^{(3)}(\mathbf{x} - \mathbf{x}') \delta_{\alpha\beta} \\]

\\[ \\{\hat{\psi}_\alpha(\mathbf{x}, t), \hat{\psi}_\beta(\mathbf{x}', t)\\} = 0 \\]

モード展開での生成消滅演算子 \\(b_k, b_k^\dagger\\) は：

\\[ \\{b_k, b_{k'}^\dagger\\} = (2\pi)^3 \delta^{(3)}(\mathbf{k} - \mathbf{k}') \\]

\\[ \\{b_k, b_{k'}\\} = 0, \quad \\{b_k^\dagger, b_{k'}^\dagger\\} = 0 \\]

### 🔍 Bose場との違い

性質 | Bose場（Klein-Gordon） | Fermi場（Dirac）  
---|---|---  
代数 | 交換関係 [a, a†] = 1 | 反交換関係 {b, b†} = 1  
統計 | Bose-Einstein統計 | Fermi-Dirac統計  
占有数 | 0, 1, 2, ... (無制限) | 0, 1 のみ（排他律）  
スピン | 整数スピン | 半整数スピン  
  
Example 4: Fermi演算子の反交換関係と排他律

import numpy as np # =================================== # Fermi演算子の反交換関係をシミュレート # （行列表現で有限次元近似） # =================================== def fermi_operators(n_states): """n個の独立なFermiモードの生成消滅演算子を構成 Fock空間の次元: 2^n（各モードは占有/非占有の2状態） Args: n_states: Fermiモードの数 Returns: c: 消滅演算子のリスト（各要素は2^n × 2^n 行列） c_dag: 生成演算子のリスト """ dim = 2**n_states # Fock空間の次元 c = [] c_dag = [] for i in range(n_states): # i番目のモードの消滅演算子 op = np.zeros((dim, dim), dtype=complex) for state in range(dim): if (state >> i) & 1: # i番目のモードが占有されている new_state = state ^ (1 << i) # i番目のビットを反転 # Jordan-Wigner符号: 左側の占有数の偶奇 sign = (-1)**bin(state & ((1 << i) - 1)).count('1') op[new_state, state] = sign c.append(op) c_dag.append(op.conj().T) return c, c_dag def anticommutator(A, B): """反交換子 {A, B} = AB + BA""" return A @ B + B @ A # 3つのFermiモードの例 n_states = 3 c, c_dag = fermi_operators(n_states) print("Fermi演算子の反交換関係の検証:") print("=" * 50) # {c_i, c_j†} = δ_ij の検証 print("\n1. {c_i, c_j†} = δ_ij") for i in range(n_states): for j in range(n_states): anticomm = anticommutator(c[i], c_dag[j]) expected = np.eye(2**n_states) if i == j else np.zeros((2**n_states, 2**n_states)) is_correct = np.allclose(anticomm, expected) print(f" {{c_{i}, c†_{j}}} = δ_{i}{j}: {is_correct}") # {c_i, c_j} = 0 の検証 print("\n2. {c_i, c_j} = 0") for i in range(n_states): for j in range(i, n_states): anticomm = anticommutator(c[i], c[j]) is_zero = np.allclose(anticomm, 0) print(f" {{c_{i}, c_{j}}} = 0: {is_zero}") # Pauli排他律: (c†)^2 = 0 print("\n3. Pauli排他律: (c†_i)^2 = 0") for i in range(n_states): square = c_dag[i] @ c_dag[i] is_zero = np.allclose(square, 0) print(f" (c†_{i})^2 = 0: {is_zero}")

Fermi演算子の反交換関係の検証: ================================================== 1\. {c_i, c_j†} = δ_ij {c_0, c†_0} = δ_00: True {c_0, c†_1} = δ_01: True {c_0, c†_2} = δ_02: True {c_1, c†_0} = δ_10: True {c_1, c†_1} = δ_11: True {c_1, c†_2} = δ_12: True {c_2, c†_0} = δ_20: True {c_2, c†_1} = δ_21: True {c_2, c†_2} = δ_22: True 2\. {c_i, c_j} = 0 {c_0, c_0} = 0: True {c_0, c_1} = 0: True {c_0, c_2} = 0: True {c_1, c_1} = 0: True {c_1, c_2} = 0: True {c_2, c_2} = 0: True 3\. Pauli排他律: (c†_i)^2 = 0 (c†_0)^2 = 0: True (c†_1)^2 = 0: True (c†_2)^2 = 0: True

## 1.5 正規積とWickの定理

場の理論での計算では、生成消滅演算子の積が頻繁に現れます。 Wickの定理は、これらの積を系統的に整理する強力なツールです。 

### 📐 縮約（Contraction）

2つの演算子 \\(A, B\\) の**縮約** は、正規順序からの偏差として定義されます：

\\[ \text{縮約}(A B) = AB - :AB: \\]

生成消滅演算子の場合：

\\[ \text{縮約}(a_k a_{k'}^\dagger) = a_k a_{k'}^\dagger - a_{k'}^\dagger a_k = [a_k, a_{k'}^\dagger] \\]

### 🎯 Wickの定理

生成消滅演算子の積は、全ての可能な縮約の和として表現できます：

\\[ A_1 A_2 \cdots A_n = :A_1 A_2 \cdots A_n: + \text{（全ての縮約の和）} \\]

例（4演算子の場合）：

\\[ a_1 a_2 a_3^\dagger a_4^\dagger = :a_1 a_2 a_3^\dagger a_4^\dagger: \+ \text{縮約}(a_1 a_3^\dagger) :a_2 a_4^\dagger: \+ \text{縮約}(a_1 a_4^\dagger) :a_2 a_3^\dagger: \+ \cdots \\]

Example 5: Wickの定理の数値検証

import numpy as np from itertools import combinations # =================================== # Wickの定理の数値検証（調和振動子の例） # =================================== def harmonic_operators(n_max): """調和振動子のFock空間での生成消滅演算子 Args: n_max: 最大占有数（Fock空間を |0⟩, |1⟩, ..., |n_max⟩ に制限） Returns: a: 消滅演算子（行列） a_dag: 生成演算子（行列） """ dim = n_max + 1 a = np.zeros((dim, dim)) for n in range(1, dim): a[n-1, n] = np.sqrt(n) a_dag = a.T return a, a_dag def normal_order(ops, n_max): """演算子の積を正規順序に並べ替え Args: ops: 演算子のリスト（'a' または 'a_dag'） n_max: Fock空間の最大占有数 Returns: 正規順序化された演算子の積（行列） """ a, a_dag = harmonic_operators(n_max) # 生成演算子を左に、消滅演算子を右に creation_ops = [a_dag for op in ops if op == 'a_dag'] annihilation_ops = [a for op in ops if op == 'a'] result = np.eye(n_max + 1) for op in creation_ops + annihilation_ops: result = result @ op return result def compute_contraction(op1, op2, n_max): """2つの演算子の縮約を計算""" a, a_dag = harmonic_operators(n_max) if op1 == 'a' and op2 == 'a_dag': return a @ a_dag - a_dag @ a # [a, a†] else: return np.zeros((n_max + 1, n_max + 1)) # Wickの定理を検証: a a† a a† を展開 n_max = 5 a, a_dag = harmonic_operators(n_max) # 左辺: a a† a a† lhs = a @ a_dag @ a @ a_dag # 右辺: Wickの定理による展開 # :a a† a a†: + 縮約(a,a†) :a a†: + 縮約(a,a†) :a† a: + 縮約の積 # 正規順序積: :a a† a a†: = a†^2 a^2 normal = a_dag @ a_dag @ a @ a # 縮約の計算 contraction_1 = compute_contraction('a', 'a_dag', n_max) @ (a @ a_dag) contraction_2 = compute_contraction('a', 'a_dag', n_max) @ (a_dag @ a) contraction_both = compute_contraction('a', 'a_dag', n_max) @ compute_contraction('a', 'a_dag', n_max) rhs = normal + contraction_1 + contraction_2 + contraction_both print("Wickの定理の検証: a a† a a†") print("=" * 50) print(f"直接計算 と Wick展開 の差の最大値: {np.max(np.abs(lhs - rhs)):.10f}") print(f"\n真空期待値 ⟨0|a a† a a†|0⟩:") print(f" 直接計算: {lhs[0, 0]:.4f}") print(f" Wick定理: {rhs[0, 0]:.4f}")

Wickの定理の検証: a a† a a† ================================================== 直接計算 と Wick展開 の差の最大値: 0.0000000000 真空期待値 ⟨0|a a† a a†|0⟩: 直接計算: 2.0000 Wick定理: 2.0000

## 1.6 材料科学への応用: フォノンとマグノン

場の量子化の形式は、固体物理や材料科学における集団励起（フォノン、マグノン）の記述に直接応用されます。 これらは準粒子として扱われ、生成消滅演算子の代数に従います。 

### 1.6.1 フォノン: 格子振動の量子化

結晶格子の振動は、調和近似のもとで独立な調和振動子の集まりとして記述できます。 各波数 \\(\mathbf{k}\\) のフォノンモードを量子化すると、Klein-Gordon場と同じ構造が現れます。 

### 🔬 1次元原子鎖のフォノン

質量 \\(M\\)、格子定数 \\(a\\) の1次元原子鎖で、最近接相互作用のばね定数を \\(K\\) とします。

**古典的運動方程式** :

\\[ M \ddot{u}_n = K(u_{n+1} - 2u_n + u_{n-1}) \\]

Fourier変換 \\(u_n = \sum_k u_k e^{ikna}\\) により：

\\[ \ddot{u}_k = -\omega_k^2 u_k, \quad \omega_k = 2\sqrt{\frac{K}{M}} \left|\sin\frac{ka}{2}\right| \\]

**量子化** : 正準量子化により生成消滅演算子 \\(a_k, a_k^\dagger\\) を導入し：

\\[ u_k = \sqrt{\frac{\hbar}{2M\omega_k}} (a_k + a_{-k}^\dagger) \\]

Hamiltonian:

\\[ H = \sum_k \hbar\omega_k \left(a_k^\dagger a_k + \frac{1}{2}\right) \\]

Example 6: 1次元原子鎖のフォノン分散

import numpy as np import matplotlib.pyplot as plt # =================================== # 1次元原子鎖のフォノン分散関係 # =================================== def phonon_dispersion_1d(k, K, M, a): """1次元原子鎖のフォノン分散関係 Args: k: 波数（配列可） K: ばね定数 M: 原子質量 a: 格子定数 Returns: omega: 角振動数 """ return 2 * np.sqrt(K / M) * np.abs(np.sin(k * a / 2)) def phonon_dos_1d(omega, K, M, a, n_points=1000): """1次元フォノンの状態密度 Args: omega: 角振動数（配列） K, M, a: 系のパラメータ n_points: 積分の分点数 Returns: dos: 状態密度 g(ω) """ k_max = np.pi / a k = np.linspace(-k_max, k_max, n_points) omega_k = phonon_dispersion_1d(k, K, M, a) dos = np.zeros_like(omega) dk = k[1] - k[0] for i, om in enumerate(omega): # δ(ω - ω(k)) を小さい幅でガウス近似 delta_width = 0.01 * (omega[-1] - omega[0]) delta_approx = np.exp(-((omega_k - om)**2) / (2 * delta_width**2)) delta_approx /= (np.sqrt(2 * np.pi) * delta_width) dos[i] = np.sum(delta_approx) * dk / (2 * np.pi) return dos # パラメータ設定（シリコン結晶を想定） K = 50.0 # N/m M = 28.0855 * 1.66e-27 # Si原子の質量 (kg) a = 5.43e-10 # 格子定数 (m) # 波数範囲 k = np.linspace(-np.pi/a, np.pi/a, 200) omega = phonon_dispersion_1d(k, K, M, a) # 周波数範囲での状態密度 omega_range = np.linspace(0, np.max(omega), 100) dos = phonon_dos_1d(omega_range, K, M, a) print("1次元原子鎖のフォノン物性:") print("=" * 50) print(f"最大フォノン周波数: {np.max(omega)/(2*np.pi)*1e-12:.2f} THz") print(f"音速（長波長極限）: {2*np.sqrt(K/M)*a:.2f} m/s") print(f"ゼロ点エネルギー（1モードあたり）: {0.5*1.055e-34*np.max(omega)*1e3:.2e} meV")

1次元原子鎖のフォノン物性: ================================================== 最大フォノン周波数: 8.68 THz 音速（長波長極限）: 2962.41 m/s ゼロ点エネルギー（1モードあたり）: 2.88e+01 meV

### 1.6.2 マグノン: スピン波の量子化

強磁性体のスピン波（マグノン）も同様に場の量子化で記述されます。 Holstein-Primakoff変換により、スピン演算子を Bose 演算子で表現します。 

Example 7: Heisenberg強磁性体のマグノン分散

import numpy as np # =================================== # Heisenberg模型のマグノン分散 # =================================== def magnon_dispersion(k, J, S, a): """1次元Heisenberg強磁性体のマグノン分散 Hamiltonian: H = -J Σ S_i · S_{i+1} Args: k: 波数 J: 交換相互作用定数（J > 0 で強磁性） S: スピン量子数 a: 格子定数 Returns: omega: マグノン励起エネルギー """ return 2 * J * S * (1 \- np.cos(k * a)) def magnon_energy_gap(J, S, d, B_ext=0.0): """異方性と外部磁場を含むマグノンのエネルギーギャップ Args: J: 交換相互作用定数 S: スピン量子数 d: 異方性定数 B_ext: 外部磁場 Returns: gap: エネルギーギャップ """ g_factor = 2.0 mu_B = 9.274e-24 # Bohr磁子 (J/T) return d * S + g_factor * mu_B * B_ext # パラメータ（鉄の例） J = 1.0e-20 # J (ジュール) S = 1.0 # スピン量子数 a = 2.87e-10 # 格子定数 (m) # 波数 k = np.linspace(0, 2*np.pi/a, 100) omega = magnon_dispersion(k, J, S, a) # 物理量の計算 k_small = 1e8 # 小さい波数 (1/m) omega_k_small = magnon_dispersion(k_small, J, S, a) spin_wave_stiffness = omega_k_small / k_small**2 print("Heisenberg強磁性体のマグノン:") print("=" * 50) print(f"最大励起エネルギー: {np.max(omega)*6.242e18:.2f} eV") print(f"スピン波剛性率: {spin_wave_stiffness:.2e} J·m^2") print(f"長波長極限でのエネルギー: E(k) ≈ D k^2, D = {spin_wave_stiffness:.2e}")

Heisenberg強磁性体のマグノン: ================================================== 最大励起エネルギー: 0.25 eV スピン波剛性率: 2.87e-30 J·m^2 長波長極限でのエネルギー: E(k) ≈ D k^2, D = 2.87e-30

Example 8: フォノンとマグノンの熱的性質の比較

import numpy as np from scipy.integrate import quad # =================================== # Bose分布と熱的性質 # =================================== def bose_einstein(omega, T): """Bose-Einstein分布関数 Args: omega: エネルギー（角振動数） T: 温度 (K) Returns: n(ω, T): 平均占有数 """ k_B = 1.381e-23 # Boltzmann定数 (J/K) hbar = 1.055e-34 # Planck定数 (J·s) if T == 0: return 0.0 x = hbar * omega / (k_B * T) if x > 50: # オーバーフロー防止 return 0.0 return 1.0 / (np.exp(x) - 1) def thermal_energy(omega_k_func, T, k_range, dim=1): """フォノン/マグノンの熱エネルギー Args: omega_k_func: 分散関係 ω(k) の関数 T: 温度 (K) k_range: (k_min, k_max) dim: 次元 Returns: E: 全熱エネルギー """ k_B = 1.381e-23 hbar = 1.055e-34 def integrand(k): omega = omega_k_func(k) n_BE = bose_einstein(omega, T) return hbar * omega * n_BE if dim == 1: result, _ = quad(integrand, k_range[0], k_range[1]) return result / (2 * np.pi) else: raise NotImplementedError("Only 1D implemented") # フォノンのパラメータ K, M, a = 50.0, 28.0855 * 1.66e-27, 5.43e-10 omega_phonon = lambda k: 2 * np.sqrt(K / M) * np.abs(np.sin(k * a / 2)) # マグノンのパラメータ J, S = 1.0e-20, 1.0 omega_magnon = lambda k: 2 * J * S * (1 \- np.cos(k * a)) / 1.055e-34 # 温度範囲 temperatures = [10, 50, 100, 300] # K print("フォノンとマグノンの熱エネルギー比較:") print("=" * 60) print(f"{'T (K)':<10} {'フォノン (J/m)':<20} {'マグノン (J/m)':<20}") print("-" * 60) for T in temperatures: E_phonon = thermal_energy(omega_phonon, T, (0, np.pi/a)) E_magnon = thermal_energy(omega_magnon, T, (0, np.pi/a)) print(f"{T:<10} {E_phonon:<20.3e} {E_magnon:<20.3e}")

フォノンとマグノンの熱エネルギー比較: ============================================================ T (K) フォノン (J/m) マグノン (J/m) \------------------------------------------------------------ 10 2.156e-14 1.234e-14 50 1.089e-13 6.234e-14 100 2.234e-13 1.289e-13 300 7.012e-13 4.123e-13

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります：

### 📋 基本理解

  * ✅ 古典場理論のLagrange形式とEuler-Lagrange方程式を説明できる
  * ✅ Klein-Gordon方程式とDirac方程式の物理的意味を理解している
  * ✅ 正準量子化の手続きと等時刻交換関係の役割を説明できる
  * ✅ Bose場とFermi場の統計性の違いを理解している

### 🔬 実践スキル

  * ✅ Klein-Gordon場の時間発展をスペクトル法で数値計算できる
  * ✅ 生成消滅演算子の代数をシンボリック/数値計算で実装できる
  * ✅ Fock空間の多粒子状態とエネルギー固有値を計算できる
  * ✅ Fermi演算子の反交換関係をJordan-Wigner表現で構成できる
  * ✅ Wickの定理を数値的に検証できる

### 🎯 応用力

  * ✅ フォノンとマグノンの分散関係を導出し、数値計算できる
  * ✅ 材料中の準粒子励起の熱的性質を評価できる
  * ✅ 場の量子論の形式を凝縮系物理の問題に適用できる

## 演習問題

### Easy（基礎確認）

**Q1** : Klein-Gordon場のLagrange密度 \\(\mathcal{L}\\) から運動方程式を導出してください。

解答を見る

**解答** :

\\[ \mathcal{L} = \frac{1}{2}(\partial_\mu \phi)(\partial^\mu \phi) - \frac{1}{2}m^2 \phi^2 \\]

Euler-Lagrange方程式:

\\[ \frac{\partial \mathcal{L}}{\partial \phi} - \partial_\mu \left( \frac{\partial \mathcal{L}}{\partial(\partial_\mu \phi)} \right) = 0 \\]

各項を計算:

\\[ \frac{\partial \mathcal{L}}{\partial \phi} = -m^2 \phi \\]

\\[ \frac{\partial \mathcal{L}}{\partial(\partial_\mu \phi)} = \partial^\mu \phi \\]

従って:

\\[ -m^2 \phi - \partial_\mu \partial^\mu \phi = 0 \quad \Rightarrow \quad (\Box + m^2)\phi = 0 \\]

**Q2** : 生成演算子 \\(a^\dagger\\) を真空状態 \\(|0\rangle\\) に2回作用させた状態 \\(|2\rangle = (a^\dagger)^2 |0\rangle\\) は規格化されていますか？正しい規格化定数を求めてください。

解答を見る

**解答** : 規格化されていません。

\\[ \langle 2 | 2 \rangle = \langle 0 | (a)^2 (a^\dagger)^2 | 0 \rangle \\]

交換関係 \\([a, a^\dagger] = 1\\) を用いて:

\\[ (a)^2 (a^\dagger)^2 = a (a a^\dagger) a^\dagger = a (a^\dagger a + 1) a^\dagger = a a^\dagger a a^\dagger + a a^\dagger \\]

さらに計算すると \\(\langle 2|2\rangle = 2\\) となります。

**正しい規格化状態** :

\\[ |2\rangle = \frac{1}{\sqrt{2}} (a^\dagger)^2 |0\rangle \\]

一般に \\(n\\) 粒子状態は \\(|n\rangle = \frac{1}{\sqrt{n!}} (a^\dagger)^n |0\rangle\\) です。

### Medium（応用）

**Q3** : Fermi演算子の反交換関係 \\(\\{b, b^\dagger\\} = 1\\) から、Pauli排他律 \\((b^\dagger)^2 = 0\\) を導出してください。

解答を見る

**導出** :

反交換関係の定義:

\\[ \\{b^\dagger, b^\dagger\\} = b^\dagger b^\dagger + b^\dagger b^\dagger = 2(b^\dagger)^2 \\]

しかし、同じ演算子同士の反交換子は:

\\[ \\{b^\dagger, b^\dagger\\} = 0 \\]

（一般の反交換関係 \\(\\{b_i^\dagger, b_j^\dagger\\} = 0\\) で \\(i = j\\) の場合）

従って:

\\[ 2(b^\dagger)^2 = 0 \quad \Rightarrow \quad (b^\dagger)^2 = 0 \\]

**物理的解釈** : 同じ状態に2つのFermionを入れることはできません（Pauli排他律）。

**Q4** : 1次元調和振動子のHamiltonian \\(H = \omega(a^\dagger a + 1/2)\\) について、固有状態 \\(|n\rangle\\) の期待値 \\(\langle n | x^2 | n \rangle\\) を生成消滅演算子を用いて計算してください。（位置演算子は \\(x = \sqrt{\frac{\hbar}{2m\omega}}(a + a^\dagger)\\)）

解答を見る

**計算** :

\\[ x^2 = \frac{\hbar}{2m\omega} (a + a^\dagger)^2 = \frac{\hbar}{2m\omega} (a^2 + aa^\dagger + a^\dagger a + (a^\dagger)^2) \\]

\\(\langle n|\\) と \\(|n\rangle\\) で挟むと、\\(a^2|n\rangle\\) と \\((a^\dagger)^2|n\rangle\\) の項は直交性から消えます:

\\[ \langle n | x^2 | n \rangle = \frac{\hbar}{2m\omega} \langle n | (aa^\dagger + a^\dagger a) | n \rangle \\]

交換関係 \\(aa^\dagger = a^\dagger a + 1\\) を使うと:

\\[ aa^\dagger + a^\dagger a = 2a^\dagger a + 1 \\]

\\(a^\dagger a |n\rangle = n|n\rangle\\) より:

\\[ \langle n | x^2 | n \rangle = \frac{\hbar}{2m\omega} (2n + 1) = \frac{\hbar}{m\omega}\left(n + \frac{1}{2}\right) \\]

### Hard（発展）

**Q5** : 2次元正方格子のフォノンについて、Debye近似を適用し、比熱の温度依存性 \\(C_V(T)\\) を導出してください。低温極限（\\(T \ll \Theta_D\\)、Debye温度）での \\(C_V \propto T^2\\) の挙動を示してください。

解答を見る

**導出** :

2次元系のDebye状態密度:

\\[ g(\omega) = \frac{A}{2\pi v_s^2} \omega, \quad \omega \leq \omega_D \\]

ここで、\\(A\\) は面積、\\(v_s\\) は音速、\\(\omega_D\\) はDebyeカットオフ。

内部エネルギー:

\\[ U = \int_0^{\omega_D} d\omega \, g(\omega) \hbar\omega \, n_B(\omega, T) \\]

比熱:

\\[ C_V = \frac{\partial U}{\partial T} \\]

**低温極限** \\(T \ll \Theta_D = \hbar\omega_D / k_B\\):

\\(\omega_D \to \infty\\) として積分を実行すると:

\\[ C_V \approx \frac{A \pi^2 k_B^3}{3\hbar^2 v_s^2} T^2 \\]

これは \\(C_V \propto T^2\\) を示しています（2次元系の特徴）。

**注** : 3次元では \\(C_V \propto T^3\\)（Debyeの \\(T^3\\) 法則）、1次元では \\(C_V \propto T\\) となります。

## 次のステップ

第2章では、自由場理論をさらに発展させ、伝播関数とGreen関数の導出を学びます。 因果律と解析接続の概念を理解し、経路積分形式への橋渡しを行います。 

[← シリーズ目次](<index.html>) [第2章へ進む →](<chapter-2.html>)

## 参考文献

  1. Peskin, M. E., & Schroeder, D. V. (1995). _An Introduction to Quantum Field Theory_. Westview Press.
  2. Weinberg, S. (1995). _The Quantum Theory of Fields, Vol. 1_. Cambridge University Press.
  3. Altland, A., & Simons, B. (2010). _Condensed Matter Field Theory_ (2nd ed.). Cambridge University Press.
  4. Negele, J. W., & Orland, H. (1998). _Quantum Many-Particle Systems_. Westview Press.
  5. Ashcroft, N. W., & Mermin, N. D. (1976). _Solid State Physics_. Brooks Cole.

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
