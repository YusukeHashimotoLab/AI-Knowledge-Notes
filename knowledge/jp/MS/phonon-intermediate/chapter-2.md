---
title: "第2章: フォノン相互作用"
chapter_title: "第2章: フォノン相互作用"
subtitle: "非調和性、散乱過程、およびフォノン寿命"
---

[🌐 EN](../../../en/MS/phonon-intermediate/chapter-2.md) | 🇯🇵 JP | Last sync: 2025-12-20

[材料科学道場](../index.html) > [フォノン物理学(中級)](index.md) > 第2章

# 第2章: フォノン相互作用

**非調和性、散乱過程、およびフォノン寿命**

## 学習目標

- 調和近似の限界と非調和性の重要性を理解する
- 三次・四次力定数の物理的意味を把握する
- フェルミの黄金則を用いてフォノン-フォノン散乱率を導出できる
- 通常過程（N過程）とウムクラップ過程（U過程）を区別できる
- 保存則がフォノン散乱に与える制約を理解する
- フォノン寿命と分光線幅の関係を説明できる
- 非調和性の実験的兆候を認識できる
- Pythonでフォノン散乱率を計算できる

## 1. 調和近似を超えて

### 1.1 調和近似の限界

第1章で学んだ調和近似では、格子ポテンシャルエネルギーを変位の二次までで展開しました。この近似下では、フォノンモードは互いに独立であり、エネルギーは完全に保存され、フォノンの寿命は無限大となります。しかし、この描像には重大な物理的問題があります：

- **熱膨張の欠如**: 調和近似では熱膨張係数がゼロになる
- **無限の熱伝導率**: フォノン散乱がないため熱抵抗が生じない
- **温度独立性**: フォノン周波数が温度に依存しない
- **無限の寿命**: 実験で観測される分光線幅を説明できない

> **重要:** これらの観測可能な現象は全て、ポテンシャルの非調和項（三次以上の項）に起因します。

### 1.2 非調和ポテンシャル展開

結晶ポテンシャルエネルギーをより完全に展開すると：

\\[
\begin{align}
V &= V_0 + \underbrace{\sum_{i\alpha,j\beta} \Phi_{i\alpha,j\beta}^{(2)} u_{i\alpha} u_{j\beta}}_{\text{調和項}} \\
&\quad + \underbrace{\sum_{i\alpha,j\beta,k\gamma} \Phi_{i\alpha,j\beta,k\gamma}^{(3)} u_{i\alpha} u_{j\beta} u_{k\gamma}}_{\text{三次非調和項}} \\
&\quad + \underbrace{\sum_{i\alpha,j\beta,k\gamma,l\delta} \Phi_{i\alpha,j\beta,k\gamma,l\delta}^{(4)} u_{i\alpha} u_{j\beta} u_{k\gamma} u_{l\delta}}_{\text{四次非調和項}} + \cdots
\end{align}
\\]

ここで、\\(u_{i\alpha}\\)は原子\\(i\\)の方向\\(\alpha\\)における変位、\\(\Phi^{(n)}\\)は\\(n\\)次力定数です。

### 1.3 三次および四次力定数

力定数は、平衡位置近傍でのポテンシャルの導関数として定義されます：

\\[
\begin{align}
\Phi_{i\alpha,j\beta,k\gamma}^{(3)} &= \left.\frac{\partial^3 V}{\partial u_{i\alpha} \partial u_{j\beta} \partial u_{k\gamma}}\right|_{\text{eq}} \\
\Phi_{i\alpha,j\beta,k\gamma,l\delta}^{(4)} &= \left.\frac{\partial^4 V}{\partial u_{i\alpha} \partial u_{j\beta} \partial u_{k\gamma} \partial u_{l\delta}}\right|_{\text{eq}}
\end{align}
\\]

**物理的解釈:**

- **三次項 \\(\Phi^{(3)}\\)**: 三つのフォノンの相互作用を記述し、一つのフォノンが二つに分裂（崩壊）したり、二つのフォノンが一つに合体したりする過程を可能にします
- **四次項 \\(\Phi^{(4)}\\)**: 四つのフォノン間の相互作用を記述し、二つのフォノンが散乱して別の二つになる過程などを可能にします

## 2. フォノン-フォノン散乱理論

### 2.1 フェルミの黄金則

非調和ハミルトニアンを摂動として扱い、時間依存摂動論を適用すると、初期状態\\(|i\rangle\\)から終状態\\(|f\rangle\\)への遷移率は、フェルミの黄金則で与えられます：

\\[
W_{i \to f} = \frac{2\pi}{\hbar} |\langle f | H_{\text{anh}} | i \rangle|^2 \delta(E_f - E_i)
\\]

ここで、\\(H_{\text{anh}}\\)は非調和ハミルトニアン、\\(\delta(E_f - E_i)\\)はエネルギー保存を表すデルタ関数です。

### 2.2 三フォノン過程

三次非調和項\\(\Phi^{(3)}\\)による主要な散乱過程は、三つのフォノンモードが関与します：

**崩壊過程（Decay Process）**

\\[
\mathbf{q}_0 \to \mathbf{q}_1 + \mathbf{q}_2
\\]

波数\\(\mathbf{q}_0\\)のフォノンが、二つのフォノン（\\(\mathbf{q}_1, \mathbf{q}_2\\)）に分裂します。

**合体過程（Coalescence Process）**

\\[
\mathbf{q}_1 + \mathbf{q}_2 \to \mathbf{q}_0
\\]

二つのフォノンが合体して一つのフォノンになります。

> **注意:** これらは逆過程であり、詳細釣り合い（detailed balance）の原理により関連しています。

### 2.3 散乱率の導出

フォノンモード\\((\mathbf{q}, j)\\)の三フォノン散乱率は、第二量子化された非調和ハミルトニアンから導かれます：

\\[
\begin{align}
\tau_{\mathbf{q}j}^{-1} &= \frac{\hbar}{16\pi N} \sum_{\mathbf{q}'j', \mathbf{q}''j''} \Bigg\{
|V_3^+|^2 (n_{\mathbf{q}'j'} + n_{\mathbf{q}''j''} + 1) \\
&\quad \times \delta(\omega_{\mathbf{q}j} - \omega_{\mathbf{q}'j'} - \omega_{\mathbf{q}''j''}) \delta_{\mathbf{q},\mathbf{q}'+\mathbf{q}''+\mathbf{G}} \\
&\quad + 2|V_3^-|^2 (n_{\mathbf{q}'j'} - n_{\mathbf{q}''j''}) \\
&\quad \times \delta(\omega_{\mathbf{q}j} + \omega_{\mathbf{q}'j'} - \omega_{\mathbf{q}''j''}) \delta_{\mathbf{q}+\mathbf{q}',\mathbf{q}''+\mathbf{G}} \Bigg\}
\end{align}
\\]

ここで、

- \\(n_{\mathbf{q}j} = [\exp(\hbar\omega_{\mathbf{q}j}/k_BT) - 1]^{-1}\\): ボース・アインシュタイン分布
- \\(V_3^{\pm}\\): 三次相互作用行列要素
- \\(\mathbf{G}\\): 逆格子ベクトル
- \\(N\\): 単位胞の数

## 3. 通常過程とウムクラップ過程

### 3.1 保存則

フォノン散乱過程は、次の二つの保存則に従います：

\\[
\begin{align}
&\text{エネルギー保存:} \quad \hbar\omega_{\mathbf{q}j} = \hbar\omega_{\mathbf{q}'j'} + \hbar\omega_{\mathbf{q}''j''} \\
&\text{結晶運動量保存:} \quad \mathbf{q} = \mathbf{q}' + \mathbf{q}'' + \mathbf{G}
\end{align}
\\]

結晶運動量は逆格子ベクトル\\(\mathbf{G}\\)の不定性を持つことに注意が必要です。

### 3.2 通常過程（Normal Process, N-process）

\\(\mathbf{G} = 0\\)の場合、つまり：

\\[
\mathbf{q} = \mathbf{q}' + \mathbf{q}''
\\]

この過程では、全フォノン運動量が保存されます。N過程は：

- フォノンのエネルギー分布を再配分する
- 局所平衡状態（ドリフトした分布）を確立する
- **熱抵抗を直接生じない**（運動量が保存されるため）

### 3.3 ウムクラップ過程（Umklapp Process, U-process）

\\(\mathbf{G} \neq 0\\)の場合、つまり散乱後の波数ベクトルの和が第一ブリルアン域外に出る場合：

\\[
\mathbf{q} = \mathbf{q}' + \mathbf{q}'' + \mathbf{G}, \quad \mathbf{G} \neq 0
\\]

ウムクラップ過程は：

- 全運動量を保存しない（結晶格子が逆向きの運動量を受け取る）
- **熱抵抗の主要な原因**となる
- 高温で重要になる（大きな波数ベクトルを持つフォノンが必要）
- 温度とともに指数関数的に増加する（\\(\propto e^{-\Theta_U/T}\\)）

**物理的直観:**

ウムクラップ過程では、散乱後のフォノンの運動量方向が大きく変わります。これにより、熱流を運ぶフォノンの集団的ドリフトが破壊され、熱抵抗が生じます。一方、N過程では運動量が保存されるため、フォノン気体全体のドリフトは維持され、熱抵抗には直接寄与しません。

### 3.4 温度依存性

**低温（\\(T \ll \Theta_D\\)、デバイ温度）では：**

- 励起されるフォノンは主に小さな\\(\mathbf{q}\\)に集中
- \\(\mathbf{q} + \mathbf{q}' \lesssim 2q_{\max} < |\mathbf{G}|_{\min}\\)となりU過程は稀
- N過程が支配的で、熱伝導率は境界散乱で制限される

**高温（\\(T \gtrsim \Theta_D\\)）では：**

- 大きな\\(\mathbf{q}\\)のフォノンも励起される
- U過程が頻繁に起こり、熱抵抗を支配する
- 熱伝導率は\\(\kappa \propto 1/T\\)の温度依存性を示す

## 4. フォノン寿命と線幅

### 4.1 不確定性原理との関連

フォノンの寿命\\(\tau\\)が有限であることは、エネルギー準位に不確定性を導入します。ハイゼンベルクの不確定性原理より：

\\[
\Delta E \cdot \Delta t \sim \hbar \quad \Rightarrow \quad \Delta E \sim \frac{\hbar}{\tau}
\\]

この不確定性は、分光スペクトルにおいて有限の線幅として現れます。

### 4.2 分光線幅

フォノンモード\\((\mathbf{q}, j)\\)のスペクトル線形は、ローレンツ型となります：

\\[
I(\omega) = \frac{A\Gamma/2\pi}{(\omega - \omega_{\mathbf{q}j})^2 + (\Gamma/2)^2}
\\]

ここで、半値全幅（FWHM）\\(\Gamma\\)は寿命と以下の関係があります：

\\[
\Gamma = \frac{2\hbar}{\tau_{\mathbf{q}j}}
\\]

> **実験的意義:** 中性子散乱やラマン分光で観測される線幅から、直接フォノン寿命を決定できます。鋭い線（小さな\\(\Gamma\\)）は長寿命フォノンを、広い線（大きな\\(\Gamma\\)）は短寿命フォノンを示します。

### 4.3 寿命の温度依存性

フォノン寿命は温度とともに変化します。三フォノン散乱による寿命は、ボース分布関数\\(n(\omega, T)\\)を通じて温度依存性を持ちます：

\\[
\tau^{-1}(T) \propto \sum_{\mathbf{q}'j', \mathbf{q}''j''} |V_3|^2 [n(\omega') + n(\omega'') + 1] \delta(\cdots)
\\]

高温極限（\\(k_BT \gg \hbar\omega\\)）では、\\(n(\omega, T) \approx k_BT/\hbar\omega\\)となり：

\\[
\tau^{-1} \propto T \quad \Rightarrow \quad \Gamma \propto T
\\]

つまり、温度上昇とともに線幅が線形に広がります。

## 5. 熱抵抗との関連

### 5.1 マティーセンの法則

異なる散乱機構は独立であると仮定すると、全散乱率は各機構の散乱率の和となります（マティーセンの法則）：

\\[
\tau_{\mathbf{q}j}^{-1} = \tau_{\text{ph-ph}}^{-1} + \tau_{\text{boundary}}^{-1} + \tau_{\text{defect}}^{-1} + \tau_{\text{electron}}^{-1} + \cdots
\\]

ここで、

- \\(\tau_{\text{ph-ph}}^{-1}\\): フォノン-フォノン散乱（N過程とU過程）
- \\(\tau_{\text{boundary}}^{-1}\\): 境界散乱（\\(\propto L^{-1}\\)、\\(L\\)は試料サイズ）
- \\(\tau_{\text{defect}}^{-1}\\): 点欠陥、不純物散乱
- \\(\tau_{\text{electron}}^{-1}\\): 電子-フォノン散乱

### 5.2 熱伝導率への寄与

ボルツマン輸送方程式（詳細は第3章）から、熱伝導率は：

\\[
\kappa = \frac{1}{3} \sum_{\mathbf{q}j} C_{\mathbf{q}j} v_{\mathbf{q}j}^2 \tau_{\mathbf{q}j}
\\]

ここで、\\(C_{\mathbf{q}j}\\)は比熱、\\(v_{\mathbf{q}j}\\)は群速度です。

**N過程とU過程の役割の違い:**

- **N過程**: 寿命を減少させるが、ドリフト速度を維持するため熱伝導率への直接的な影響は小さい（間接的には平衡化を促進）
- **U過程**: 運動量を破壊するため、熱伝導率を直接制限する主要因
- **境界散乱**: 低温での支配的な散乱機構（U過程が稀な領域）

### 5.3 温度依存性のパターン

| 温度領域 | 支配的機構 | \\(\kappa\\)の温度依存性 | 物理的理由 |
|---------|----------|---------------------|----------|
| 極低温<br>(\\(T \ll \Theta_D\\)) | 境界散乱 | \\(\kappa \propto T^3\\) | デバイ比熱\\(C_V \propto T^3\\)、\\(\tau\\)は一定 |
| 中間温度<br>(\\(T \sim \Theta_D/3\\)) | N過程 + U過程 | \\(\kappa\\)はピーク | \\(C_V\\)増加とU過程増加の競合 |
| 高温<br>(\\(T \gg \Theta_D\\)) | U過程 | \\(\kappa \propto 1/T\\) | \\(\tau \propto 1/T\\)、\\(C_V\\)は一定 |

## 6. Pythonによるフォノン散乱率計算

### 6.1 簡略化されたモデル

実際の散乱率計算は非常に複雑ですが、ここでは教育的な目的で簡略化されたモデルを実装します。デバイモデルを仮定し、単一の音響ブランチを考えます。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 物理定数
hbar = 1.054571817e-34  # J·s
kB = 1.380649e-23      # J/K

class PhononScatteringModel:
    """
    簡略化されたフォノン散乱率計算モデル
    デバイモデルを仮定し、三フォノン過程を考慮
    """

    def __init__(self, theta_D, mass, a, gamma):
        """
        パラメータ:
            theta_D: デバイ温度 [K]
            mass: 平均原子質量 [kg]
            a: 格子定数 [m]
            gamma: グリュナイゼンパラメータ (非調和性の尺度)
        """
        self.theta_D = theta_D
        self.mass = mass
        self.a = a
        self.gamma = gamma

        # デバイ周波数
        self.omega_D = kB * theta_D / hbar

        # 平均音速（簡略化）
        self.v_s = self.omega_D * a / (2 * np.pi)

    def bose_einstein(self, omega, T):
        """ボース・アインシュタイン分布"""
        x = hbar * omega / (kB * T)
        if x < 1e-3:  # 高温極限
            return kB * T / (hbar * omega)
        else:
            return 1.0 / (np.exp(x) - 1.0)

    def umklapp_rate(self, omega, T):
        """
        ウムクラップ散乱率（簡略化されたモデル）

        Klemens の近似式:
        τ_U^(-1) ∝ ω² T exp(-θ_D / 3T)
        """
        # ウムクラップ過程の活性化温度
        theta_U = self.theta_D / 3.0

        # 散乱率の前因子（現象論的）
        A_U = (self.gamma**2 * self.omega_D) / (self.mass * self.v_s**2 * self.a**3)

        # 温度と周波数依存性
        if T < 0.1:  # 数値安定性のため
            return 0.0
        else:
            rate = A_U * (omega / self.omega_D)**2 * T * np.exp(-theta_U / T)
            return rate

    def normal_rate(self, omega, T):
        """
        通常散乱率（簡略化されたモデル）

        N過程は熱抵抗に直接寄与しないが、
        平衡化に重要な役割を果たす
        """
        A_N = (self.gamma**2 * self.omega_D) / (self.mass * self.v_s**2 * self.a**3)

        # N過程はU過程よりも活性化障壁が低い
        rate = A_N * (omega / self.omega_D)**2 * T
        return rate

    def boundary_rate(self, L):
        """
        境界散乱率
        Casimir限界: τ_B^(-1) = v_s / L
        """
        return self.v_s / L

    def total_rate(self, omega, T, L=np.inf):
        """
        全散乱率（マティーセンの法則）
        """
        rate_U = self.umklapp_rate(omega, T)
        rate_N = self.normal_rate(omega, T)
        rate_B = self.boundary_rate(L) if L < np.inf else 0.0

        return rate_U + rate_N + rate_B

    def thermal_conductivity(self, T, L=np.inf, n_points=1000):
        """
        熱伝導率の計算（簡略化されたモデル）

        κ = (1/3) ∫ C(ω) v² τ(ω) g(ω) dω
        """
        # デバイ状態密度を仮定
        omega_max = self.omega_D
        omegas = np.linspace(0.01 * omega_max, omega_max, n_points)

        kappa = 0.0
        for omega in omegas:
            # 比熱（フォノンモードあたり）
            n = self.bose_einstein(omega, T)
            x = hbar * omega / (kB * T)
            if x < 1e-3:
                C_omega = kB
            else:
                C_omega = kB * x**2 * np.exp(x) / (np.exp(x) - 1)**2

            # 状態密度（デバイモデル）
            g_omega = 3 * omega**2 / omega_max**3

            # 緩和時間
            tau = 1.0 / self.total_rate(omega, T, L)

            # 寄与
            kappa += C_omega * self.v_s**2 * tau * g_omega

        # 数値積分（台形則）
        kappa *= (omega_max / n_points)

        return kappa / 3.0  # 因子1/3


# シリコンのパラメータ例
model_Si = PhononScatteringModel(
    theta_D=645,           # K
    mass=28.085 * 1.66054e-27,  # Si原子質量 [kg]
    a=5.43e-10,            # 格子定数 [m]
    gamma=1.0              # グリュナイゼンパラメータ
)

# 散乱率の温度依存性
temperatures = np.linspace(10, 600, 100)
omega_test = 0.5 * model_Si.omega_D  # テスト周波数

rates_U = [model_Si.umklapp_rate(omega_test, T) for T in temperatures]
rates_N = [model_Si.normal_rate(omega_test, T) for T in temperatures]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(temperatures, rates_U, 'r-', label='U-process', linewidth=2)
plt.semilogy(temperatures, rates_N, 'b--', label='N-process', linewidth=2)
plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel('Scattering Rate (s⁻¹)', fontsize=12)
plt.title('Phonon-Phonon Scattering Rates', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# 熱伝導率の温度依存性
kappas_bulk = [model_Si.thermal_conductivity(T, L=np.inf) for T in temperatures]
kappas_1mm = [model_Si.thermal_conductivity(T, L=1e-3) for T in temperatures]

plt.subplot(1, 2, 2)
plt.plot(temperatures, kappas_bulk, 'k-', label='Bulk (L=∞)', linewidth=2)
plt.plot(temperatures, kappas_1mm, 'g--', label='L=1mm', linewidth=2)
plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel('Thermal Conductivity (W/m·K)', fontsize=12)
plt.title('Temperature Dependence of κ', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phonon_scattering_rates.png', dpi=150, bbox_inches='tight')
plt.show()

print("計算完了！")
```

### 6.2 計算結果の解釈

**グラフの読み方:**

- **散乱率の温度依存性**: U過程は低温で指数関数的に抑制され、高温で支配的になる様子が観察できます
- **熱伝導率**: 低温でのサイズ効果、中間温度でのピーク、高温での\\(1/T\\)依存性を再現しています
- **寿命と線幅**: 高周波数フォノンほど散乱率が高く、寿命が短く、線幅が広いことが示されます

> **注意:** このコードは教育目的の簡略化されたモデルです。現実の材料では、複数の分岐、分散関係の詳細、正確な力定数が必要です。研究レベルの計算には、第一原理計算やより洗練されたモデルを使用してください。

## 7. 非調和性の実験的兆候

### 7.1 熱膨張

非調和性の最も直接的な証拠の一つは熱膨張です。体積熱膨張係数は：

\\[
\alpha_V = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_P \propto \gamma C_V
\\]

ここで\\(\gamma\\)はグリュナイゼンパラメータで、非調和性の尺度です。調和系では\\(\gamma = 0\\)となり、熱膨張は起こりません。

### 7.2 フォノン周波数のシフト

温度上昇に伴い、フォノン周波数は変化します。これは主に：

- **体積効果**: 熱膨張による格子定数の変化
- **純粋な非調和効果**: フォノン-フォノン相互作用による繰り込み

周波数の温度依存性は：

\\[
\omega(T) = \omega_0 + \Delta\omega_{\text{vol}}(T) + \Delta\omega_{\text{anh}}(T)
\\]

### 7.3 分光線幅の温度依存性

ラマン散乱や中性子散乱スペクトルにおいて、線幅\\(\Gamma\\)は温度とともに増加します。典型的な依存性は：

\\[
\Gamma(T) = \Gamma_0 + A[1 + 2n(\omega/2, T)] + B[1 + 3n(\omega/3, T) + 3n^2(\omega/3, T)]
\\]

ここで、第二項は三フォノン過程、第三項は四フォノン過程に対応します。

### 7.4 熱伝導率の振る舞い

熱伝導率の温度依存性のパターン自体が、非調和性の証拠となります：

- 高温での\\(\kappa \propto 1/T\\)依存性はU過程の存在を示す
- 中間温度でのピークは、比熱増加とU過程増加の競合を反映
- 低温でのサイズ依存性は、内的散乱（U過程）が抑制されていることを示す

### 7.5 測定技術

| 技術 | 測定量 | 非調和性の情報 |
|-----|--------|--------------|
| X線回折 | 格子定数 vs T | 熱膨張係数 |
| ラマン分光 | 周波数、線幅 vs T | フォノン寿命、非調和シフト |
| 非弾性中性子散乱 | 分散関係、線幅 | 詳細な散乱率 |
| 熱伝導測定 | κ vs T | 散乱機構の総合的効果 |
| 超音波測定 | 音速、減衰 vs T | 弾性定数の温度依存性 |

## まとめ

- 調和近似では説明できない現象（熱膨張、有限の熱伝導率、フォノン寿命）は、ポテンシャルの非調和項に起因する
- 三次力定数\\(\Phi^{(3)}\\)は三フォノン過程を、四次力定数\\(\Phi^{(4)}\\)は四フォノン過程を引き起こす
- フォノン-フォノン散乱は、フェルミの黄金則を用いて量子力学的に記述できる
- 通常過程（N過程）は運動量を保存し、熱抵抗に直接寄与しないが、平衡化に重要
- ウムクラップ過程（U過程）は結晶運動量を保存せず、熱抵抗の主要因となる
- U過程は高温で重要になり、\\(\kappa \propto 1/T\\)の温度依存性を生む
- フォノン寿命\\(\tau\\)は分光線幅\\(\Gamma\\)と\\(\Gamma = 2\hbar/\tau\\)の関係にある
- 非調和性は、熱膨張、周波数シフト、線幅の温度依存性など、多様な実験的兆候を示す
- 簡略化されたモデルでも、フォノン散乱率と熱伝導率の定性的な温度依存性を再現できる

## 演習問題

**問題1: 概念的理解**

なぜ調和近似では熱膨張が起こらないのか説明してください。また、三次非調和項がどのように熱膨張を引き起こすかを、ポテンシャルの非対称性の観点から論じてください。

**問題2: 保存則**

一次元格子（格子定数\\(a\\)）において、波数\\(q_1 = 0.4\pi/a\\)と\\(q_2 = 0.7\pi/a\\)のフォノンが散乱する場合を考えます。散乱後の波数\\(q_3\\)を求め、これがN過程かU過程かを判定してください。（第一ブリルアン域は\\(-\pi/a \leq q \leq \pi/a\\)）

**問題3: 線幅と寿命**

ラマン分光でシリコンの光学フォノン（\\(\omega_0 = 15.5\\) THz）を測定したところ、室温で線幅\\(\Gamma = 0.1\\) THz（半値全幅）が観測されました。このフォノンの寿命\\(\tau\\)を計算してください。（\\(\hbar = 1.055 \times 10^{-34}\\) J·s, 1 THz = \\(2\pi \times 10^{12}\\) rad/s）

**問題4: 温度依存性**

上記のPythonコードを用いて、異なるグリュナイゼンパラメータ\\(\gamma = 0.5, 1.0, 2.0\\)に対する熱伝導率の温度依存性を計算し、\\(\gamma\\)が非調和性の強さにどのように対応するかを考察してください。

**問題5: 発展課題**

低温（\\(T \ll \Theta_D\\)）での熱伝導率が境界散乱で制限されるとき、\\(\kappa \propto T^3\\)の依存性を示すことを、デバイモデルを用いて導出してください。（ヒント: デバイ比熱\\(C_V \propto T^3\\)、緩和時間\\(\tau = L/v_s\\)は温度に依存しない）

**問題6: プログラミング課題**

提供されたPythonコードを拡張して、境界散乱の効果を含む熱伝導率の試料サイズ依存性（\\(L = 10\\) μm, 100 μm, 1 mm, バルク）を低温（10-100 K）で計算し、グラフ化してください。

## 参考文献

1. Ziman, J. M. "Electrons and Phonons" (Oxford University Press, 1960) - フォノン-フォノン散乱の古典的教科書
2. Klemens, P. G. "Anharmonic Decay of Optical Phonons" Physical Review 148, 845 (1966) - 非調和崩壊過程の基礎理論
3. Ashcroft, N. W. & Mermin, N. D. "Solid State Physics" Chapter 25-26 (Saunders, 1976) - 格子力学と非調和性の教科書的扱い
4. Dove, M. T. "Introduction to Lattice Dynamics" (Cambridge, 1993) - 格子力学の現代的入門書
5. Srivastava, G. P. "The Physics of Phonons" (Taylor & Francis, 1990) - フォノン物理の包括的教科書
6. Debernardi, A., Baroni, S., & Molinari, E. "Anharmonic Phonon Lifetimes in Semiconductors from Density-Functional Perturbation Theory" Physical Review Letters 75, 1819 (1995) - 第一原理計算によるフォノン寿命
7. Ward, A. & Broido, D. A. "Intrinsic Phonon Relaxation Times from First-Principles Studies of the Thermal Conductivities of Si and Ge" Physical Review B 81, 085205 (2010) - 最新の第一原理熱伝導率計算
8. Lindsay, L., Broido, D. A., & Mingo, N. "Diameter Dependence of Carbon Nanotube Thermal Conductivity and Extension to the Graphene Limit" Physical Review B 82, 161402(R) (2010) - ナノ構造における散乱機構

---

[← 第1章: 格子力学の理論](chapter-1.md) | [目次](index.md) | [第3章: 熱伝導 →](chapter-3.md)

---

## 免責事項

この教育コンテンツは、橋本研究室のナレッジベース用にAIの支援を受けて作成されました。正確性を期していますが、重要な情報については一次資料や教科書で確認することをお勧めします。

© 2025 東北大学 橋本研究室. All rights reserved.
