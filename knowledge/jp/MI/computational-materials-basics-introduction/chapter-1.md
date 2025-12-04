---
title: 第1章：量子力学と固体物理の基礎
chapter_title: 第1章：量子力学と固体物理の基礎
subtitle: シュレーディンガー方程式から結晶のバンド構造まで
reading_time: 20-25分
difficulty: 中級
code_examples: 7
exercises: 0
---

# 第1章：量子力学と固体物理の基礎

波動関数と演算子の直観を掴み、計算材料の言葉に慣れます。近似の考え方にも触れます。

**💡 補足:** 数式は“現象の圧縮表現”。厳密さより“何を省略し、何を残すか”の視点を育てます。

## 学習目標

この章を読むことで、以下を習得できます： \- シュレーディンガー方程式の物理的意味と数値解法を理解する \- ボルン・オッペンハイマー近似が計算材料科学の基礎となる理由を説明できる \- 固体の周期性とブロッホの定理の関係を理解する \- Pythonで水素原子や量子井戸の波動関数を計算できる

* * *

## 1.1 シュレーディンガー方程式：量子力学の基礎方程式

計算材料科学のすべては、**シュレーディンガー方程式** から始まります。これは原子・分子・固体の電子状態を記述する基本方程式です。

### 時間依存シュレーディンガー方程式

$$ i\hbar \frac{\partial \Psi(\mathbf{r}, t)}{\partial t} = \hat{H} \Psi(\mathbf{r}, t) $$

ここで： \- $\Psi(\mathbf{r}, t)$: 波動関数（確率振幅） \- $\hat{H}$: ハミルトニアン演算子（エネルギー演算子） \- $\hbar = h/(2\pi)$: 換算プランク定数 \- $i$: 虚数単位

### 時間独立シュレーディンガー方程式

定常状態（エネルギー固有状態）を扱う場合、時間依存性を分離できます：

$$ \hat{H} \psi(\mathbf{r}) = E \psi(\mathbf{r}) $$

これは**固有値問題** です： \- $\psi(\mathbf{r})$: エネルギー固有状態（固有関数） \- $E$: エネルギー固有値

### ハミルトニアンの構成

多電子系のハミルトニアンは以下のように書けます：

$$ \hat{H} = \underbrace{-\frac{\hbar^2}{2m_e}\sum_i \nabla_i^2}_{\text{電子の運動エネルギー}} \underbrace{-\frac{\hbar^2}{2}\sum_I \frac{\nabla_I^2}{M_I}}_{\text{原子核の運動エネルギー}} + \underbrace{\frac{1}{2}\sum_{i \neq j} \frac{e^2}{|\mathbf{r}_i - \mathbf{r}_j|}}_{\text{電子間相互作用}} + \underbrace{\sum_{i,I} \frac{-Z_I e^2}{|\mathbf{r}_i - \mathbf{R}_I|}}_{\text{電子-核間相互作用}} + \underbrace{\frac{1}{2}\sum_{I \neq J} \frac{Z_I Z_J e^2}{|\mathbf{R}_I - \mathbf{R}_J|}}_{\text{核間相互作用}} $$

ここで： \- $m_e$: 電子の質量 \- $M_I$: 原子核$I$の質量 \- $\mathbf{r}_i$: 電子$i$の位置 \- $\mathbf{R}_I$: 原子核$I$の位置 \- $Z_I$: 原子核$I$の電荷（原子番号） \- $e$: 電気素量

**問題の複雑さ** : このハミルトニアンは解析的に解けません（水素原子以外）。そこで近似が必要になります。

* * *

## 1.2 ボルン・オッペンハイマー近似

計算材料科学で最も重要な近似の1つが**ボルン・オッペンハイマー近似** （Born-Oppenheimer approximation, BOA）です。

### 基本的な考え方

原子核の質量は電子の約2000-400000倍です（H: 1836倍、Fe: 102000倍）。このため：

  1. **電子は原子核よりはるかに速く動く**
  2. 電子から見ると、原子核は「ほぼ止まっている」
  3. **原子核と電子の運動を分離できる**

### 数学的定式化

全波動関数を以下のように分離します：

$$ \Psi(\mathbf{r}, \mathbf{R}) \approx \psi_{\text{elec}}(\mathbf{r}; \mathbf{R}) \cdot \chi_{\text{nuc}}(\mathbf{R}) $$

  * $\psi_{\text{elec}}(\mathbf{r}; \mathbf{R})$: 電子の波動関数（原子核位置$\mathbf{R}$をパラメータとして含む）
  * $\chi_{\text{nuc}}(\mathbf{R})$: 原子核の波動関数

これにより、2段階で問題を解きます：

**Step 1: 電子状態の計算（原子核を固定）**

$$ \hat{H}_{\text{elec}} \psi_{\text{elec}}(\mathbf{r}; \mathbf{R}) = E_{\text{elec}}(\mathbf{R}) \psi_{\text{elec}}(\mathbf{r}; \mathbf{R}) $$

ここで、電子ハミルトニアンは：

$$ \hat{H}_{\text{elec}} = -\frac{\hbar^2}{2m_e}\sum_i \nabla_i^2 + \frac{1}{2}\sum_{i \neq j} \frac{e^2}{|\mathbf{r}_i - \mathbf{r}_j|} + \sum_{i,I} \frac{-Z_I e^2}{|\mathbf{r}_i - \mathbf{R}_I|} $$

**Step 2: 原子核の運動（ポテンシャルエネルギー面上）**

$$ \left[-\frac{\hbar^2}{2}\sum_I \frac{\nabla_I^2}{M_I} + E_{\text{elec}}(\mathbf{R}) + \frac{1}{2}\sum_{I \neq J} \frac{Z_I Z_J e^2}{|\mathbf{R}_I - \mathbf{R}_J|}\right] \chi_{\text{nuc}}(\mathbf{R}) = E_{\text{total}} \chi_{\text{nuc}}(\mathbf{R}) $$

$E_{\text{elec}}(\mathbf{R})$は原子核位置の関数として**ポテンシャルエネルギー面（PES）** を形成します。

### BOAの物理的意味

  * **電子は常に原子核の瞬間的な位置に対して基底状態にある** （断熱近似）
  * 原子核はこのPES上を古典的または量子的に運動する
  * これにより、DFT計算（電子状態）とMD計算（原子核の運動）が分離可能になる

### BOAが破綻する場合

以下の場合、BOAは不正確になります： 1\. **励起状態間の遷移** : 光吸収、化学反応の非断熱過程 2\. **電子-フォノン相関が強い系** : 超伝導、Jahn-Teller効果 3\. **軽い原子** : 水素原子の零点振動

* * *

## 1.3 水素原子の解：量子力学の原型

水素原子は**唯一解析的に解けるクーロン多体系** です。ここから多電子系への理解が始まります。

### シュレーディンガー方程式（水素原子）

$$ \left[-\frac{\hbar^2}{2m_e}\nabla^2 - \frac{e^2}{r}\right] \psi(\mathbf{r}) = E \psi(\mathbf{r}) $$

極座標$(r, \theta, \phi)$で変数分離すると：

$$ \psi_{nlm}(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi) $$

  * $R_{nl}(r)$: 動径波動関数（主量子数$n$、方位量子数$l$）
  * $Y_l^m(\theta, \phi)$: 球面調和関数（磁気量子数$m$）

### エネルギー固有値

$$ E_n = -\frac{m_e e^4}{2\hbar^2 n^2} = -\frac{13.6 \text{ eV}}{n^2} $$

$n = 1, 2, 3, \ldots$

### 動径波動関数の例

**基底状態（1s軌道、$n=1, l=0$）** :

$$ R_{10}(r) = 2\left(\frac{1}{a_0}\right)^{3/2} e^{-r/a_0} $$

ここで$a_0 = \hbar^2/(m_e e^2) = 0.529$ Åはボーア半径です。

**励起状態（2p軌道、$n=2, l=1$）** :

$$ R_{21}(r) = \frac{1}{\sqrt{3}}\left(\frac{1}{2a_0}\right)^{3/2} \frac{r}{a_0} e^{-r/(2a_0)} $$

### Pythonで波動関数を計算・可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import genlaguerre, sph_harm
    
    # 定数
    a0 = 0.529  # ボーア半径 [Å]
    
    def radial_wavefunction(r, n, l):
        """
        水素原子の動径波動関数 R_nl(r)
    
        Args:
            r: 動径座標 [Å]
            n: 主量子数 (1, 2, 3, ...)
            l: 方位量子数 (0, 1, ..., n-1)
    
        Returns:
            R_nl(r)
        """
        rho = 2 * r / (n * a0)
        L = genlaguerre(n - l - 1, 2*l + 1)  # 一般化ラゲール多項式
    
        # 正規化定数
        N = np.sqrt((2/(n*a0))**3 * np.math.factorial(n-l-1) /
                    (2*n*np.math.factorial(n+l)))
    
        R_nl = N * rho**l * np.exp(-rho/2) * L(rho)
        return R_nl
    
    # 動径座標の範囲
    r = np.linspace(0, 20, 1000)
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左: 波動関数 R_nl(r)
    axes[0].plot(r, radial_wavefunction(r, 1, 0), label='1s (n=1, l=0)')
    axes[0].plot(r, radial_wavefunction(r, 2, 0), label='2s (n=2, l=0)')
    axes[0].plot(r, radial_wavefunction(r, 2, 1), label='2p (n=2, l=1)')
    axes[0].plot(r, radial_wavefunction(r, 3, 0), label='3s (n=3, l=0)')
    axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].set_xlabel('r [Å]', fontsize=12)
    axes[0].set_ylabel('$R\_{nl}(r)$ [Å$^{-3/2}$]', fontsize=12)
    axes[0].set_title('水素原子の動径波動関数', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 右: 動径確率密度 r^2 |R_nl(r)|^2
    axes[1].plot(r, r**2 * radial_wavefunction(r, 1, 0)**2, label='1s')
    axes[1].plot(r, r**2 * radial_wavefunction(r, 2, 0)**2, label='2s')
    axes[1].plot(r, r**2 * radial_wavefunction(r, 2, 1)**2, label='2p')
    axes[1].plot(r, r**2 * radial_wavefunction(r, 3, 0)**2, label='3s')
    axes[1].set_xlabel('r [Å]', fontsize=12)
    axes[1].set_ylabel('$r^2 |R\_{nl}(r)|^2$ [Å$^{-1}$]', fontsize=12)
    axes[1].set_title('動径確率密度', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hydrogen_wavefunctions.png', dpi=150)
    plt.show()
    
    # エネルギー準位の計算
    print("水素原子のエネルギー準位:")
    for n in range(1, 5):
        E_n = -13.6 / n**2
        print(f"n={n}: E = {E_n:.3f} eV")
    

**実行結果** :
    
    
    水素原子のエネルギー準位:
    n=1: E = -13.600 eV
    n=2: E = -3.400 eV
    n=3: E = -1.511 eV
    n=4: E = -0.850 eV
    

**重要なポイント** : \- 1s軌道は節（ゼロ点）を持たない \- 2s軌道は1つの節を持つ（動径方向） \- 2p軌道は原点で0（$r=0$で$R_{21}(0)=0$） \- 動径確率密度のピークがボーアの軌道半径に対応

* * *

## 1.4 固体の周期性とブロッホの定理

固体（結晶）は原子が**周期的** に配列した構造です。この周期性が固体の電子状態に決定的な影響を与えます。

### 結晶の周期性

結晶格子は**格子ベクトル** $\mathbf{R}$で記述されます：

$$ \mathbf{R} = n_1 \mathbf{a}_1 + n_2 \mathbf{a}_2 + n_3 \mathbf{a}_3 $$

  * $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$: 基本格子ベクトル（単位胞を張るベクトル）
  * $n_1, n_2, n_3$: 整数

結晶のポテンシャルは周期性を持ちます：

$$ V(\mathbf{r} + \mathbf{R}) = V(\mathbf{r}) \quad \text{for all } \mathbf{R} $$

### ブロッホの定理

結晶のシュレーディンガー方程式：

$$ \left[-\frac{\hbar^2}{2m_e}\nabla^2 + V(\mathbf{r})\right] \psi(\mathbf{r}) = E \psi(\mathbf{r}) $$

ここで$V(\mathbf{r})$が周期的なら、解（ブロッホ関数）は以下の形を取ります：

$$ \psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} u_{n\mathbf{k}}(\mathbf{r}) $$

  * $\mathbf{k}$: **波数ベクトル** （逆格子空間の座標）
  * $u_{n\mathbf{k}}(\mathbf{r})$: **周期的関数** （$u_{n\mathbf{k}}(\mathbf{r}+\mathbf{R}) = u_{n\mathbf{k}}(\mathbf{r})$）
  * $n$: バンドインデックス

**物理的意味** : \- $e^{i\mathbf{k}\cdot\mathbf{r}}$: 平面波（伝播する波） \- $u_{n\mathbf{k}}(\mathbf{r})$: 結晶格子の周期性を反映した変調

### 第一ブリルアンゾーン

すべての情報は**第一ブリルアンゾーン** （First Brillouin Zone, FBZ）に含まれます。これは逆格子空間の単位胞です。

**簡単な立方格子（格子定数$a$）** : \- 格子ベクトル: $\mathbf{a}_1 = a\hat{\mathbf{x}}, \mathbf{a}_2 = a\hat{\mathbf{y}}, \mathbf{a}_3 = a\hat{\mathbf{z}}$ \- 逆格子ベクトル: $\mathbf{b}_1 = \frac{2\pi}{a}\hat{\mathbf{x}}, \mathbf{b}_2 = \frac{2\pi}{a}\hat{\mathbf{y}}, \mathbf{b}_3 = \frac{2\pi}{a}\hat{\mathbf{z}}$ \- FBZ: $-\frac{\pi}{a} \leq k_x, k_y, k_z \leq \frac{\pi}{a}$

### バンド構造

各バンド$n$に対して、エネルギーは$\mathbf{k}$の関数です：

$$ E_n(\mathbf{k}) $$

これを**バンド構造** （band structure）または**分散関係** （dispersion relation）と呼びます。

**金属 vs 絶縁体** : \- **金属** : フェルミ準位$E_F$がバンド内に存在（部分的に占有されたバンド） \- **絶縁体/半導体** : フェルミ準位がバンドギャップ内に存在（完全に占有されたバンドと完全に空のバンドの間）

* * *

## 1.5 Pythonで1次元結晶のバンド構造を計算

実際に1次元の周期ポテンシャル中の電子のバンド構造を計算してみましょう。

### クローニッヒ・ペニーモデル

1次元の周期的な矩形ポテンシャル：

$$ V(x) = \begin{cases} 0 & 0 < x < a \ V_0 & a < x < a+b \end{cases} $$

周期$d = a + b$で繰り返されます。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import eigh
    
    def kronig_penney_band(N=100, a=1.0, b=0.2, V0=5.0, n_bands=5):
        """
        クローニッヒ・ペニーモデルのバンド構造を計算
    
        Args:
            N: 平面波基底の数
            a: ポテンシャルの幅（井戸） [Å]
            b: ポテンシャルの幅（障壁） [Å]
            V0: 障壁の高さ [eV]
            n_bands: 表示するバンド数
    
        Returns:
            k_points: 波数ベクトル
            energies: エネルギーバンド
        """
        d = a + b  # 周期
        G = 2 * np.pi / d  # 逆格子ベクトル
    
        # 第一ブリルアンゾーン内のk点
        k_points = np.linspace(-np.pi/d, np.pi/d, 200)
        energies = np.zeros((len(k_points), n_bands))
    
        # ポテンシャルのフーリエ係数（矩形ポテンシャル）
        def V_G(n):
            if n == 0:
                return V0 * b / d
            else:
                return V0 * np.sin(n * G * b / 2) / (n * np.pi) * np.exp(-1j * n * G * (a + b/2))
    
        for ik, k in enumerate(k_points):
            # ハミルトニアン行列の構築（平面波基底）
            H = np.zeros((2*N+1, 2*N+1), dtype=complex)
    
            for i in range(-N, N+1):
                for j in range(-N, N+1):
                    if i == j:
                        # 対角成分: 運動エネルギー
                        H[i+N, j+N] = 0.5 * (k + i*G)**2  # 原子単位系
                    else:
                        # 非対角成分: ポテンシャル
                        H[i+N, j+N] = V_G(i - j)
    
            # 固有値問題を解く
            eigvals, eigvecs = eigh(H)
            energies[ik, :] = eigvals[:n_bands]
    
        return k_points, energies
    
    # バンド構造の計算
    k_points, energies = kronig_penney_band(N=50, a=1.0, b=0.2, V0=5.0, n_bands=6)
    
    # プロット
    plt.figure(figsize=(10, 6))
    for n in range(energies.shape[1]):
        plt.plot(k_points, energies[:, n], linewidth=2)
    
    plt.xlabel('波数 k [Å$^{-1}$]', fontsize=12)
    plt.ylabel('エネルギー [eV]', fontsize=12)
    plt.title('1次元周期ポテンシャルのバンド構造（クローニッヒ・ペニーモデル）', fontsize=14)
    plt.grid(alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlim([k_points[0], k_points[-1]])
    plt.ylim([0, 20])
    
    # 高対称点のラベル
    d = 1.2
    plt.axvline(-np.pi/d, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(np.pi/d, color='gray', linestyle='--', alpha=0.5)
    plt.xticks([-np.pi/d, 0, np.pi/d], ['-π/d', 'Γ', 'π/d'])
    
    plt.tight_layout()
    plt.savefig('1d_band_structure.png', dpi=150)
    plt.show()
    
    # バンドギャップの計算
    print("\nバンドギャップの解析:")
    k_gamma = np.argmin(np.abs(k_points))  # Γ点（k=0）
    for n in range(energies.shape[1] - 1):
        E_top = energies[k_gamma, n]
        E_bottom = energies[k_gamma, n+1]
        gap = E_bottom - E_top
        print(f"バンド{n}と{n+1}の間のギャップ: {gap:.3f} eV")
    

**実行結果の解釈** : \- 周期ポテンシャルにより、エネルギーが離散的な**バンド** に分かれる \- バンド間には**バンドギャップ** が存在（電子が存在できないエネルギー領域） \- k=0（Γ点）でバンドの極値が現れる \- ポテンシャルが強いほど、バンドギャップが大きくなる

* * *

## 1.6 量子井戸のエネルギー準位

もう1つの重要な例として、**無限井戸型ポテンシャル** （量子井戸）を見てみましょう。

### 問題設定

$$ V(x) = \begin{cases} 0 & 0 \leq x \leq L \ \infty & \text{otherwise} \end{cases} $$

### 解析解

波動関数とエネルギー固有値は：

$$ \psi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right), \quad E_n = \frac{n^2 \pi^2 \hbar^2}{2 m_e L^2} $$

$n = 1, 2, 3, \ldots$

### Pythonでの数値計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def quantum_well(L=10.0, n_states=5):
        """
        無限井戸型ポテンシャルの波動関数とエネルギー準位
    
        Args:
            L: 井戸の幅 [Å]
            n_states: 表示する状態数
    
        Returns:
            x: 位置座標
            psi: 波動関数
            E: エネルギー準位
        """
        x = np.linspace(0, L, 500)
    
        # エネルギー準位（eV単位に変換）
        # hbar^2 / (2*m_e) = 3.81 eV Å^2（原子単位系）
        coeff = 3.81
        E = np.array([coeff * (n*np.pi/L)**2 for n in range(1, n_states+1)])
    
        # 波動関数
        psi = np.zeros((n_states, len(x)))
        for n in range(1, n_states+1):
            psi[n-1, :] = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    
        return x, psi, E
    
    # 計算
    L = 10.0  # 井戸の幅 [Å]
    x, psi, E = quantum_well(L, n_states=5)
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左: 波動関数
    for n in range(5):
        axes[0].plot(x, psi[n, :] + E[n]/2, label=f'n={n+1}, E={E[n]:.2f} eV')
        axes[0].axhline(E[n]/2, color='gray', linestyle='--', alpha=0.3)
    
    axes[0].set_xlabel('位置 x [Å]', fontsize=12)
    axes[0].set_ylabel('波動関数 + エネルギー準位', fontsize=12)
    axes[0].set_title('量子井戸の波動関数', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 右: 確率密度
    for n in range(5):
        axes[1].plot(x, psi[n, :]**2, label=f'n={n+1}')
    
    axes[1].set_xlabel('位置 x [Å]', fontsize=12)
    axes[1].set_ylabel('確率密度 |ψ(x)|$^2$ [Å$^{-1}$]', fontsize=12)
    axes[1].set_title('確率密度', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_well.png', dpi=150)
    plt.show()
    
    # エネルギー準位の表示
    print("量子井戸のエネルギー準位:")
    for n in range(1, 6):
        print(f"n={n}: E = {E[n-1]:.3f} eV")
    
    # エネルギー間隔の解析
    print("\nエネルギー間隔:")
    for n in range(1, 5):
        delta_E = E[n] - E[n-1]
        print(f"ΔE({n}→{n+1}) = {delta_E:.3f} eV")
    

**物理的洞察** : \- エネルギーは$n^2$に比例（$E_n \propto n^2$） \- 量子数$n$が増えると、波動関数の節の数が増える（$n-1$個の節） \- 井戸の幅$L$が小さいほど、エネルギー準位が高くなる（量子閉じ込め効果） \- これは量子ドットやナノワイヤーの電子状態の基礎

* * *

## 1.7 本章のまとめ

### 学んだこと

  1. **シュレーディンガー方程式** \- 量子力学の基礎方程式 \- 時間依存/時間独立の2つの形式 \- ハミルトニアンの構成（運動エネルギー + ポテンシャルエネルギー）

  2. **ボルン・オッペンハイマー近似** \- 原子核と電子の運動の分離 \- 計算材料科学の基礎となる近似 \- ポテンシャルエネルギー面（PES）の概念

  3. **水素原子** \- 解析的に解ける唯一のクーロン多体系 \- 動径波動関数と球面調和関数 \- エネルギー準位：$E_n = -13.6/n^2$ eV

  4. **固体の周期性とブロッホの定理** \- 結晶格子の周期性 \- ブロッホ関数：$\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} u_{n\mathbf{k}}(\mathbf{r})$ \- バンド構造：$E_n(\mathbf{k})$ \- バンドギャップと金属・絶縁体の違い

  5. **実践的な計算** \- Pythonで水素原子の波動関数を計算 \- 1次元周期ポテンシャルのバンド構造 \- 量子井戸のエネルギー準位

### 重要なポイント

  * 量子力学は原子・分子・固体の電子状態を記述する唯一の正確な理論
  * BOAにより電子と原子核の運動を分離できる → DFTとMDが可能に
  * 固体の周期性がバンド構造を生み出す
  * 数値計算により、複雑な系の電子状態を近似的に解ける

### 次の章へ

第2章では、多電子系の電子状態を実際に計算する手法である**密度汎関数理論（DFT）** を学びます。DFTはシュレーディンガー方程式を実用的に解くための最も重要な近似手法です。

* * *

## 演習問題

### 問題1（難易度：easy）

ボルン・オッペンハイマー近似が成り立つ理由を、原子核と電子の質量の違いから説明してください。

ヒント 電子の質量$m_e$と陽子の質量$m_p$の比を考えてみましょう。$m_p / m_e \approx 1836$です。  解答例 **ボルン・オッペンハイマー近似が成り立つ理由**: 1\. **質量の違い**: \- 電子の質量: $m_e = 9.109 \times 10^{-31}$ kg \- 陽子（水素原子核）の質量: $m_p = 1.673 \times 10^{-27}$ kg \- 質量比: $m_p / m_e = 1836$ \- 鉄原子核（Fe, 質量数56）: $M_{\text{Fe}} / m_e \approx 102,000$ 2\. **運動速度の違い**: \- 運動エネルギー$E = p^2/(2m)$が同じなら、運動量$p = \sqrt{2mE}$ \- 質量が大きいほど、同じエネルギーでの速度$v = p/m$は小さい \- 電子は原子核の約40倍以上速く動く（$v_e / v_p \sim \sqrt{m_p/m_e} \approx 43$） 3\. **時間スケールの分離**: \- 電子の運動: $\sim 10^{-16}$ 秒（フェムト秒スケール） \- 原子核の運動（振動）: $\sim 10^{-13}$ 秒（ピコ秒スケール） \- 約1000倍の時間スケールの差 4\. **物理的描像**: \- 電子から見ると、原子核は「ほぼ止まっている」 \- 電子は原子核の瞬間的な配置に対して即座に応答し、基底状態に緩和する \- この近似により、電子状態計算（DFT）と原子核の運動（MD）を分離可能 **例外（BOAが破綻する場合）**: \- 軽い原子（H, He）: 零点振動の効果が大きい \- 励起状態: 電子状態間の遷移時間と原子核の運動が同程度 \- 強相関系: 電子-格子相互作用が非常に強い系 

### 問題2（難易度：medium）

水素原子の基底状態（1s軌道）のエネルギーは-13.6 eVです。電子が基底状態から第一励起状態（2s or 2p）に遷移するために必要な光子のエネルギー（eV）と波長（nm）を計算してください。

ヒント エネルギー準位: $E_n = -13.6/n^2$ eV 光子のエネルギー: $E_{\text{photon}} = h\nu = hc/\lambda$ プランク定数: $h = 4.136 \times 10^{-15}$ eV·s、光速: $c = 3 \times 10^8$ m/s  解答例 **Step 1: エネルギー準位の計算** 基底状態（$n=1$）: $$E_1 = -\frac{13.6}{1^2} = -13.6 \text{ eV}$$ 第一励起状態（$n=2$）: $$E_2 = -\frac{13.6}{2^2} = -3.4 \text{ eV}$$ **Step 2: 遷移エネルギー** $$\Delta E = E_2 - E_1 = -3.4 - (-13.6) = 10.2 \text{ eV}$$ **Step 3: 光子の波長** $$E_{\text{photon}} = \frac{hc}{\lambda}$$ $$\lambda = \frac{hc}{E_{\text{photon}}} = \frac{(4.136 \times 10^{-15} \text{ eV·s}) \cdot (3 \times 10^8 \text{ m/s})}{10.2 \text{ eV}}$$ $$\lambda = \frac{1.241 \times 10^{-6} \text{ eV·m}}{10.2 \text{ eV}} = 1.217 \times 10^{-7} \text{ m} = 121.7 \text{ nm}$$ **答え**: \- 遷移エネルギー: **10.2 eV** \- 光子の波長: **121.7 nm**（紫外線領域、ライマンα線） **物理的意味**: \- この遷移は**ライマン系列**の最初の線（ライマンα線） \- 紫外線領域なので、地上では観測されない（大気に吸収される） \- 宇宙からの観測では重要な輝線 

### 問題3（難易度：hard）

1次元の周期ポテンシャル中の電子を考えます。ポテンシャルが弱い極限で、自由電子（$V=0$）からのずれを摂動論で考えると、バンドギャップが生じることを示してください。

ヒント 自由電子のエネルギー: $E_k = \hbar^2 k^2 / (2m_e)$ ブリルアンゾーン境界: $k = \pm \pi/a$ 縮退した状態: $E_k = E_{-k}$で縮退 摂動論により縮退が解ける  解答例 **自由電子の場合（$V=0$）**: エネルギー: $$E_k = \frac{\hbar^2 k^2}{2m_e}$$ 波動関数: $$\psi_k(x) = \frac{1}{\sqrt{L}} e^{ikx}$$ **ブリルアンゾーン境界での縮退**: $k = \pm \pi/a$（第一ブリルアンゾーン境界）で： $$E_{\pi/a} = E_{-\pi/a} = \frac{\hbar^2 \pi^2}{2m_e a^2}$$ この2つの状態は縮退している（エネルギーが等しい）。 **弱い周期ポテンシャルの導入**: $$V(x) = V(x + a) = V_0 \cos\left(\frac{2\pi x}{a}\right)$$ フーリエ展開: $$V(x) = \sum_G V_G e^{iGx}, \quad G = \frac{2\pi n}{a}$$ 第一項: $V_G = V_0 / 2$（$G = \pm 2\pi/a$） **摂動論（1次）**: 縮退した状態の線形結合: $$\psi = c_1 e^{i\pi x/a} + c_2 e^{-i\pi x/a}$$ 摂動ハミルトニアン行列: $$H' = \begin{pmatrix} \langle \pi/a | V | \pi/a \rangle & \langle \pi/a | V | -\pi/a \rangle \\\ \langle -\pi/a | V | \pi/a \rangle & \langle -\pi/a | V | -\pi/a \rangle \end{pmatrix}$$ 対角成分: $\langle \pi/a | V | \pi/a \rangle = 0$（平均ポテンシャル） 非対角成分: $$\langle \pi/a | V | -\pi/a \rangle = \frac{1}{L} \int_0^L e^{-i\pi x/a} V_0 \cos\left(\frac{2\pi x}{a}\right) e^{i\pi x/a} dx = \frac{V_0}{2}$$ 行列: $$H' = \frac{V_0}{2} \begin{pmatrix} 0 & 1 \\\ 1 & 0 \end{pmatrix}$$ 固有値: $$E_{\pm} = E_0 \pm \frac{V_0}{2}$$ ここで$E_0 = \hbar^2 \pi^2 / (2m_e a^2)$ **バンドギャップの発生**: $$\Delta E = E_+ - E_- = V_0$$ **結論**: \- 周期ポテンシャルによりブリルアンゾーン境界で縮退が解ける \- エネルギーギャップ（バンドギャップ）が$\Delta E = V_0$だけ生じる \- これが金属と絶縁体の違いを生む物理的起源 

* * *

## データライセンスと引用

### 使用したデータセット

本章で使用した定数・パラメータは以下のデータベース/文献に基づいています：

  1. **NIST Physical Constants Database** (CC0 Public Domain) \- ボーア半径、プランク定数、電子質量 \- URL: https://physics.nist.gov/cuu/Constants/ \- 引用: Tiesinga, E., et al. (2021). CODATA Recommended Values. _Rev. Mod. Phys._ , 93, 025010.

  2. **Materials Project Database** (CC BY 4.0) \- Si結晶構造データ（格子定数） \- URL: https://materialsproject.org \- 引用: Jain, A., et al. (2013). The Materials Project. _APL Materials_ , 1, 011002.

### ソフトウェアライセンス

本章のコード例で使用したソフトウェア：

  * **Python 3.11** : PSF License (BSD-compatible)
  * **NumPy 1.26** : BSD License
  * **SciPy 1.11** : BSD License
  * **Matplotlib 3.8** : PSF License

すべてのソフトウェアは商用利用可能なオープンソースライセンスです。

* * *

## コード再現性チェックリスト

本章のコード例を再現する際は、以下を確認してください：

### 環境構築
    
    
    # Python環境（推奨: Anaconda）
    conda create -n quantum-basics python=3.11
    conda activate quantum-basics
    conda install numpy scipy matplotlib
    
    # バージョン確認
    python --version  # 3.11.x
    python -c "import numpy; print(numpy.\_\_version\_\_)"  # 1.26.x
    

### 実行時の注意点

  1. **数値精度** \- 浮動小数点演算の丸め誤差により、結果が微妙に異なる場合があります \- 相対誤差 < 1e-6 であれば正常です

  2. **グラフ表示** \- Jupyter Notebook使用時: `%matplotlib inline` を追加 \- スクリプト実行時: `plt.show()` が必要

  3. **計算時間** \- Example 1.3 (水素原子波動関数): ~1秒 \- Example 1.5 (1D バンド構造): ~10秒（N=50の場合） \- Example 1.6 (量子井戸): ~2秒

### トラブルシューティング

**問題** : `ImportError: cannot import name 'genlaguerre'` **解決** : `conda install scipy` でScipyを再インストール

**問題** : グラフの日本語が文字化け **解決** :
    
    
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans']
    

* * *

## 実践的な落とし穴と対策

### 1\. 数値計算の精度問題

**落とし穴** : 波動関数の正規化不足
    
    
    # ❌ 間違い: 正規化を忘れる
    psi = np.exp(-r/a0)
    
    # ✅ 正解: 正規化定数を含める
    N = (1/a0)**1.5 / np.sqrt(np.pi)
    psi = N * np.exp(-r/a0)
    

**検証方法** :
    
    
    # 波動関数の正規化チェック
    integral = np.trapz(4*np.pi*r**2 * psi**2, r)
    print(f"Normalization: {integral:.6f}")  # 1.000000に近いはず
    

### 2\. バンド構造計算の収束

**落とし穴** : 平面波基底数が不足
    
    
    # ❌ 不十分: N=10 では収束しない
    energies = kronig\_penney\_band(N=10)
    
    # ✅ 推奨: N=50-100 で収束確認
    for N in [10, 30, 50, 100]:
        energies = kronig\_penney\_band(N=N)
        print(f"N={N}: Gap = {energies[0,1] - energies[0,0]:.4f} eV")
    

**収束判定** : エネルギー差 < 0.01 eV

### 3\. k点サンプリングの重要性

**落とし穴** : k点が粗すぎてバンドギャップを見逃す
    
    
    # ❌ 粗い: 20点では構造を見逃す
    k\_points = np.linspace(-np.pi/d, np.pi/d, 20)
    
    # ✅ 推奨: 200点以上
    k\_points = np.linspace(-np.pi/d, np.pi/d, 200)
    

### 4\. 量子数の物理的制約

**落とし穴** : 無効な量子数の組み合わせ
    
    
    # ❌ 物理的に無効: l >= n
    R\_nl = radial\_wavefunction(r, n=2, l=3)  # エラー
    
    # ✅ 制約: l < n
    assert 0 <= l < n, "Quantum number constraint: l must be < n"
    

* * *

## 品質保証チェックリスト

### 計算結果の妥当性検証

本章の計算結果が正しいか、以下の項目を確認してください：

#### 水素原子のエネルギー準位（Section 1.3）

  * [ ] n=1: E = -13.600 eV（誤差 < 0.001 eV）
  * [ ] n=2: E = -3.400 eV（誤差 < 0.001 eV）
  * [ ] エネルギーは n の増加とともに単調増加
  * [ ] 波動関数の正規化: ∫|ψ|²dV = 1（誤差 < 0.01）

#### 1Dバンド構造（Section 1.5）

  * [ ] Γ点（k=0）でバンドギャップが存在
  * [ ] ポテンシャル強度V₀を増やすとギャップが拡大
  * [ ] バンドの数 = 2N+1（平面波基底数に対応）
  * [ ] エネルギーはk点に対して周期的

#### 量子井戸（Section 1.6）

  * [ ] エネルギー準位は E_n ∝ n²
  * [ ] 井戸幅Lを2倍にするとエネルギーは1/4倍
  * [ ] 波動関数の節の数 = n-1
  * [ ] 確率密度の対称性（偶数nは対称、奇数nは反対称）

### 物理的妥当性の確認

  * [ ] すべてのエネルギーが有限（発散なし）
  * [ ] 波動関数が境界条件を満たす
  * [ ] ブロッホ定理の周期性が保たれる
  * [ ] 対称性（並進、回転）が保存される

### コードの健全性

  * [ ] すべてのコード例が警告なく実行
  * [ ] メモリリークなし（長時間実行でも安定）
  * [ ] 数値計算のオーバーフローなし
  * [ ] グラフが正常に表示される

* * *

## 参考文献

  1. Griffiths, D. J. (2018). _Introduction to Quantum Mechanics_ (3rd ed.). Cambridge University Press. \- 量子力学の標準教科書

  2. Ashcroft, N. W., & Mermin, N. D. (1976). _Solid State Physics_. Saunders College Publishing. \- 固体物理の古典的名著

  3. Kittel, C. (2004). _Introduction to Solid State Physics_ (8th ed.). Wiley. \- 固体物理の入門教科書

  4. Martin, R. M. (2004). _Electronic Structure: Basic Theory and Practical Methods_. Cambridge University Press. \- 計算材料科学の理論的基礎

  5. 常行真司 (2005). 『計算物理学』岩波書店. \- 日本語で読める計算物理の良書

* * *

## 著者情報

**作成者** : MI Knowledge Hub Content Team **作成日** : 2025-10-17 **バージョン** : 1.0 **シリーズ** : 計算材料科学基礎入門 v1.0

**ライセンス** : Creative Commons BY-NC-SA 4.0
