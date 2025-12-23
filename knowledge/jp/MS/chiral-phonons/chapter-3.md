---
title: "第3章：カイラルフォノンの実験的検出"
chapter_title: "第3章"
subtitle: "円偏光ラマン分光法から超高速分光法まで"
---

[🌐 EN](../../../en/MS/chiral-phonons/chapter-3.md) | 🇯🇵 JP | Last sync: 2025-12-19

[材料科学道場](../index.html) > [カイラルフォノン](index.md) > 第3章

---

# 第3章：カイラルフォノンの実験的検出
**円偏光ラマン分光法から超高速分光法まで**

⏱️ 90-120分 | 💻 4つのコード例 | 📊 上級

---

## 学習目標

- 円偏光ラマン分光法の実験セットアップと測定原理を理解する
- カイラルフォノンのラマン選択則を導出・適用できる
- σ⁺/σ⁺とσ⁺/σ⁻配置の物理的意味を説明する
- フォノン吸収の円二色性とフォノン角運動量の関係を理解する
- ヘリシティ分解フォトルミネッセンスによるバレー偏極検出の原理を学ぶ
- 超高速分光法によるカイラルフォノンのダイナミクス測定を習得する
- 非弾性中性子・X線散乱の可能性と課題を評価する
- 主要な実験結果（WSe₂、α-石英）を解釈できる
- Pythonで偏光依存性をシミュレーションできる

---

## 3.1 円偏光ラマン分光法

### 3.1.1 実験セットアップ

**円偏光ラマン分光法**は、カイラルフォノンを検出する最も強力で広く使われている実験技術です。この手法の鍵は、入射光と散乱光の両方の偏光状態を制御することにあります。

#### 基本的な光学配置

典型的な円偏光ラマンセットアップには以下の構成要素が含まれます：

- **レーザー光源**：通常532 nm（緑）または633 nm（赤）連続波レーザー
- **直線偏光子**：レーザー出力を直線偏光にする
- **1/4波長板（QWP）**：直線偏光を円偏光（σ⁺またはσ⁻）に変換
- **対物レンズ**：試料への集光と散乱光の収集
- **分析用1/4波長板**：散乱光の円偏光を直線偏光に戻す
- **分析用偏光子**：特定の偏光成分を選択
- **分光器**：ラマン散乱光の波長分析
- **検出器**：CCD（電荷結合素子）またはPMT（光電子増倍管）

```mermaid
graph LR
    A[レーザー] --> B[直線偏光子]
    B --> C[QWP1: σ⁺/σ⁻選択]
    C --> D[対物レンズ]
    D --> E[試料]
    E --> F[対物レンズ]
    F --> G[QWP2: 分析]
    G --> H[分析用偏光子]
    H --> I[分光器]
    I --> J[CCD検出器]
```

### 3.1.2 1/4波長板の動作原理

1/4波長板（QWP）は、直線偏光を円偏光に変換する複屈折光学素子です。その動作は以下のように理解できます：

#### 直線偏光から円偏光へ

QWPの速軸（fast axis）に対して45°で入射する直線偏光を考えます。電場は2つの成分に分解されます：

```
E_入射 = E₀(cos 45° x̂ + sin 45° ŷ) = (E₀/√2)(x̂ + ŷ)
```

QWP内部では、2つの偏光成分が異なる速度で伝播し、位相差δ = π/2（90°）が生じます：

```
E_出射 = (E₀/√2)(x̂ + e^(iπ/2) ŷ) = (E₀/√2)(x̂ + i ŷ)
```

これは**右円偏光（σ⁺）**を表します。QWPを90°回転させると左円偏光（σ⁻）が得られます：

```
E_出射 = (E₀/√2)(x̂ - i ŷ)  → σ⁻
```

### 3.1.3 σ⁺/σ⁺とσ⁺/σ⁻配置

カイラルフォノンの測定では、2つの標準的な配置が使用されます：

| 配置 | 入射光 | 散乱光分析 | 検出対象 | 物理的意味 |
|------|--------|------------|----------|------------|
| **σ⁺/σ⁺** | 右円偏光 | 右円偏光成分 | 同じヘリシティ | 角運動量保存（ΔL = 0） |
| **σ⁺/σ⁻** | 右円偏光 | 左円偏光成分 | 反対ヘリシティ | 角運動量移行（ΔL = 2ℏ） |
| **σ⁻/σ⁻** | 左円偏光 | 左円偏光成分 | 同じヘリシティ | 角運動量保存（ΔL = 0） |
| **σ⁻/σ⁺** | 左円偏光 | 右円偏光成分 | 反対ヘリシティ | 角運動量移行（ΔL = -2ℏ） |

#### 記法の説明

「σ⁺/σ⁺」という記法では：
- 最初のσ⁺：入射光の円偏光状態
- 2番目のσ⁺：分析（検出）する散乱光の偏光状態

時計回り（試料から見て）の電場回転がσ⁺、反時計回りがσ⁻です。

### 3.1.4 カイラルフォノンのラマン選択則

カイラルフォノンのラマン散乱選択則は、**角運動量保存則**から導出されます。光子とフォノンの角運動量を考えます：

#### 角運動量収支

```
L_入射光 + L_フォノン = L_散乱光 + L_電子励起
```

バレー励起を無視すると（または基底状態での測定）：

```
L_入射光 - L_散乱光 = L_フォノン
```

#### 各配置での選択則

| 測定配置 | 光子角運動量変化 | 検出されるフォノン | 例（TMDの場合） |
|----------|------------------|--------------------|--------------------|
| σ⁺/σ⁺ | ΔL_光 = 0 | L_ph = 0（アキラル） | A₁モード、E₂gモード |
| σ⁺/σ⁻ | ΔL_光 = +2ℏ | L_ph = -2ℏ（左カイラル） | E'モード（K'谷） |
| σ⁻/σ⁺ | ΔL_光 = -2ℏ | L_ph = +2ℏ（右カイラル） | E'モード（K谷） |
| σ⁻/σ⁻ | ΔL_光 = 0 | L_ph = 0（アキラル） | A₁モード、E₂gモード |

#### 重要な実験的示唆

カイラルフォノンの存在は、σ⁺/σ⁻とσ⁻/σ⁺配置で異なる強度を示すことで確認できます：
- アキラルフォノン：σ⁺/σ⁺ = σ⁻/σ⁻、σ⁺/σ⁻ = σ⁻/σ⁺ = 0（または弱い）
- カイラルフォノン：σ⁺/σ⁻ ≠ σ⁻/σ⁺、谷依存性を示す

### 3.1.5 円偏光度の測定

**円偏光度（degree of circular polarization, DCP）**は、カイラルフォノンの強さを定量化する重要な指標です：

```
DCP = (I_σ⁺/σ⁺ - I_σ⁺/σ⁻) / (I_σ⁺/σ⁺ + I_σ⁺/σ⁻)
```

ここで、I_σ⁺/σ⁺とI_σ⁺/σ⁻は、それぞれの配置でのラマン散乱強度です。

#### DCPの物理的解釈

- **DCP = +1**：完全な右円偏光（純粋な右カイラルフォノン）
- **DCP = -1**：完全な左円偏光（純粋な左カイラルフォノン）
- **DCP = 0**：非偏光またはアキラル
- **|DCP| < 1**：部分的なカイラリティ（混合状態）

---

## 3.2 フォノン吸収の円二色性

### 3.2.1 赤外円二色性（IR-CD）

**赤外円二色性**は、物質が左円偏光と右円偏光の赤外光を異なる程度に吸収する現象です。カイラルフォノンを持つ材料では、フォノン共鳴周波数でIR-CDが観測されます。

#### 測定原理

円二色性シグナルは、2つの円偏光状態での吸収係数の差として定義されます：

```
ΔA(ω) = A_σ⁺(ω) - A_σ⁻(ω)
```

または規格化された形式：

```
CD(ω) = (A_σ⁺(ω) - A_σ⁻(ω)) / (A_σ⁺(ω) + A_σ⁻(ω))
```

### 3.2.2 フォノン角運動量との関連

IR-CDとフォノン角運動量の関係は、電磁場と格子の相互作用ハミルトニアンから導出できます。カイラルフォノンモードνでは：

```
ΔA_ν ∝ Im[ε_xy(ω_ν)] ∝ L_ph,ν
```

ここで、ε_xy(ω)は誘電率テンソルの非対角成分で、フォノン角運動量L_ph,νに比例します。

#### α-石英でのIR-CD観測

Zhu et al. (2018)は、α-石英の光学フォノンで顕著なIR-CDを報告しました：
- 周波数：~1100 cm⁻¹（Si-O伸縮モード）
- CD強度：~5%（左右円偏光での吸収差）
- 温度依存性：低温でCDシグナルが増大

この結果は、フォノン角運動量の直接的な光学的測定を初めて実証しました。

### 3.2.3 IR-CDとラマン散乱の相補性

| 側面 | 円偏光ラマン | 赤外円二色性 |
|------|--------------|--------------|
| **測定対象** | 非弾性散乱過程 | 光吸収過程 |
| **選択則** | ラマン活性モード | IR活性モード |
| **情報** | フォノン分散、寿命 | 吸収強度、ダンピング |
| **感度** | 高感度（単層可能） | バルクに適する |
| **空間分解能** | 高い（~1 μm） | 低い（数十μm） |

---

## 3.3 ヘリシティ分解フォトルミネッセンス

### 3.3.1 バレー偏極検出の原理

2次元TMDにおいて、**ヘリシティ分解フォトルミネッセンス（HR-PL）**は、バレー物理とカイラルフォノンの結合を調べる強力な手法です。

#### 測定手順

1. **バレー選択励起**：σ⁺（またはσ⁻）円偏光でK（またはK'）谷を選択的に励起
2. **フォノン支援過程**：励起子-フォノン相互作用が谷間散乱を引き起こす可能性
3. **偏光分解検出**：σ⁺とσ⁻成分のPL強度を分離測定
4. **バレー偏極度計算**：P_v = (I_σ⁺ - I_σ⁻)/(I_σ⁺ + I_σ⁻)

### 3.3.2 フォノン支援過程

カイラルフォノンは、谷間散乱において重要な役割を果たします：

```
|K, σ⁺⟩ + カイラルフォノン(L_ph = -2ℏ) → |K', σ⁻⟩
```

この過程は角運動量保存を満たします：

```
L_初期 = +ℏ (K谷) + (-2ℏ) (フォノン) = -ℏ (K'谷) = L_最終
```

#### WSe₂における実験結果（2018–2019年）

複数の研究グループは、単層WSe₂でバレー偏極度の温度依存性を測定し、カイラルフォノンの役割を明らかにしました（Zhu et al. 2018; Chen et al. 2019）：
- 4 Kで：P_v ≈ 90%（高いバレー偏極保持）
- 室温で：P_v ≈ 30%（カイラルフォノンによる谷間散乱増加）
- E'フォノン（~250 cm⁻¹）が主要な谷間散乱チャネル

---

## 3.4 超高速分光法

### 3.4.1 カイラルフォノンの時間分解検出

**超高速分光法**は、カイラルフォノンのダイナミクスを実時間で観測できる強力な技術です。典型的なセットアップは**ポンプ-プローブ配置**を使用します。

#### 実験配置

1. **ポンプパルス**：フェムト秒（10⁻¹⁵秒）の円偏光パルスで試料を励起
2. **遅延制御**：機械的遅延ステージでプローブパルスのタイミングを制御
3. **プローブパルス**：弱い円偏光パルスで時間発展を観測
4. **ヘテロダイン/ホモダイン検出**：微弱な信号変化を高感度検出

### 3.4.2 コヒーレントフォノン生成

**コヒーレントフォノン**は、パルス励起によって生成される位相が揃ったフォノン振動です。カイラルフォノンの場合、円偏光ポンプが重要です：

#### 誘導ラマン散乱機構（ISRS）

強いポンプパルスと弱いプローブパルスが仮想中間状態を経由してフォノンを生成：

```
ポンプ(σ⁺) + ポンプ(σ⁺) → カイラルフォノン(L = 0)
ポンプ(σ⁺) + ポンプ(σ⁻) → カイラルフォノン(L = ±2ℏ)
```

#### 測定可能な量

時間分解測定から以下の情報が得られます：
- **振動周波数**：フォノンエネルギー ω_ph
- **減衰時間**：フォノン寿命 τ_ph = 1/Γ_ph
- **初期位相**：励起過程の詳細
- **カップリング強度**：電子-フォノン相互作用

### 3.4.3 ポンプ-プローブ技術

差分反射率の時間発展は以下のように表されます：

```
ΔR/R(t, ω_probe) = A₀ · cos(ω_ph t + φ₀) · exp(-t/τ_ph) + 非振動成分
```

ここで：
- A₀：振幅（電子-フォノンカップリングに比例）
- ω_ph：フォノン角周波数
- φ₀：初期位相
- τ_ph：減衰時間

---

## 3.5 非弾性中性子・X線散乱

### 3.5.1 偏極中性子散乱

**非弾性中性子散乱（INS）**は、フォノン分散を測定する古典的な手法ですが、カイラルフォノンの検出には課題があります。

#### 原理

中性子がフォノンと相互作用してエネルギーと運動量を交換：

```
ℏω = E_入射 - E_散乱
ℏq = k_入射 - k_散乱
```

#### 偏極中性子の利点

スピン偏極した中性子ビームを使用すると、磁気相互作用を通じてフォノンの角運動量成分に感度を持つ可能性があります：

- 中性子スピン：s = 1/2ℏ
- フォノン角運動量：L_ph = ±2ℏ
- 角運動量移行：ΔL = L_ph - Δs

#### 主な課題

- **弱い結合**：中性子-フォノン角運動量相互作用は非常に弱い
- **試料サイズ**：十分な散乱強度を得るには大きな結晶が必要（cm³オーダー）
- **エネルギー分解能**：低エネルギーフォノン（meV）の分解が困難
- **コスト**：中性子源（原子炉または加速器）へのアクセスが限定的

### 3.5.2 共鳴非弾性X線散乱（RIXS）

**RIXS**は、円偏光X線を使用してフォノン励起を測定する新興技術です：

#### 利点

- 高いエネルギー分解能（~100 meV → 数meV）
- 小さな試料サイズで測定可能（μmオーダー）
- 元素選択性（特定の原子のフォノンに焦点）
- 運動量分解能

#### カイラルフォノン検出への応用

円偏光軟X線RIXSは、原理的にカイラルフォノンの角運動量を検出できます：

```
入射X線(σ⁺) → 内殻励起 → フォノン放出 → 散乱X線(σ⁺ or σ⁻)
```

しかし、実用化にはX線光学系の改良と検出器の高感度化が必要です。

---

## 3.6 主要な実験結果

### 3.6.1 WSe₂実験（Zhu et al. 2018）

この画期的な研究は、単層WSe₂でカイラルフォノンを初めて実験的に観測しました。

#### 実験手法

- **試料**：CVD成長単層WSe₂、SiO₂/Si基板上
- **レーザー**：532 nm、出力 < 1 mW（過熱防止）
- **温度**：77 K（液体窒素冷却）
- **配置**：σ⁺/σ⁺、σ⁺/σ⁻、σ⁻/σ⁺、σ⁻/σ⁻の4配置で測定

#### 主要な発見

1. **E'モード（~250 cm⁻¹）のカイラリティ**：
   - σ⁺/σ⁺配置：強いピーク
   - σ⁺/σ⁻配置：ほぼゼロ
   - σ⁻/σ⁺配置：ほぼゼロ
   - σ⁻/σ⁻配置：強いピーク
   - → 完璧な谷選択性（DCP ≈ 1）

2. **A₁モード（~250 cm⁻¹）の非カイラル性**：
   - すべての配置で同じ強度
   - DCP ≈ 0

3. **温度依存性**：
   - 室温ではE'モードのDCPが減少（~0.3）
   - フォノン-フォノン散乱による谷間混合

### 3.6.2 α-石英実験（Zhu et al. 2018）

Zhuらは、3次元カイラル結晶α-石英でフォノン角運動量を光学的に検出しました。

#### 実験アプローチ

- **試料**：高品質α-石英単結晶、c軸方向測定
- **技術**：フーリエ変換赤外（FTIR）円二色性分光法
- **温度範囲**：10 K - 300 K
- **周波数範囲**：400 - 1400 cm⁻¹

#### 主要な結果

1. **強いIR-CDシグナル**：
   - ~1100 cm⁻¹（Si-O伸縮モード）で最大
   - CD ≈ 5%（α-石英の螺旋対称性を反映）

2. **フォノン角運動量の定量化**：
   - 第一原理計算と比較
   - L_ph ≈ 0.3ℏ（格子単位胞あたり）

3. **エナンチオマー依存性**：
   - 左旋性石英：正のCD
   - 右旋性石英：負のCD
   - → 構造カイラリティとフォノンカイラリティの一対一対応

#### 両実験の重要性

これら2つの実験は、異なる材料系（2D TMDと3Dカイラル結晶）でカイラルフォノンの普遍性を示しました：
- 2D系：バレーフォノン結合を介したカイラリティ
- 3D系：構造カイラリティに由来するカイラリティ
- 両者とも円偏光光学技術で検出可能

---

## 3.7 Pythonコード：ラマン偏光依存性シミュレーション

### 3.7.1 カイラルフォノンのラマン強度計算

```python
import numpy as np
import matplotlib.pyplot as plt

# ラマン選択則に基づくカイラルフォノンのシミュレーション

class ChiralPhononRaman:
    """カイラルフォノンのラマン散乱強度を計算するクラス"""

    def __init__(self, omega_phonon, gamma, chirality='right'):
        """
        パラメータ:
        -----------
        omega_phonon : float
            フォノン周波数 (cm⁻¹)
        gamma : float
            減衰定数 (cm⁻¹)
        chirality : str
            'right' (L = +2ℏ) または 'left' (L = -2ℏ)
        """
        self.omega = omega_phonon
        self.gamma = gamma
        self.chirality = chirality

    def raman_tensor(self):
        """
        カイラルフォノンのラマンテンソルを返す

        E'モード（二重縮退）の場合:
        右カイラル: R ∝ [[1, i], [-i, 1]]
        左カイラル: R ∝ [[1, -i], [i, 1]]
        """
        if self.chirality == 'right':
            # 右カイラル (K谷)
            R = np.array([[1, 1j], [-1j, 1]], dtype=complex)
        else:
            # 左カイラル (K'谷)
            R = np.array([[1, -1j], [1j, 1]], dtype=complex)
        return R

    def intensity(self, omega_laser, polarization_in, polarization_out):
        """
        ラマン散乱強度を計算

        パラメータ:
        -----------
        omega_laser : float
            レーザー周波数 (cm⁻¹)
        polarization_in : str
            入射偏光 ('sigma+', 'sigma-', 'linear_x', 'linear_y')
        polarization_out : str
            散乱偏光 ('sigma+', 'sigma-', 'linear_x', 'linear_y')

        戻り値:
        -------
        intensity : float
            ラマン散乱強度
        """
        # 偏光ベクトルの定義
        pol_vectors = {
            'sigma+': np.array([1, 1j]) / np.sqrt(2),
            'sigma-': np.array([1, -1j]) / np.sqrt(2),
            'linear_x': np.array([1, 0]),
            'linear_y': np.array([0, 1])
        }

        e_in = pol_vectors[polarization_in]
        e_out = pol_vectors[polarization_out].conj()

        # ラマンテンソル
        R = self.raman_tensor()

        # ラマン散乱振幅: e_out · R · e_in
        amplitude = np.dot(e_out, np.dot(R, e_in))

        # ローレンツ型共鳴
        omega_stokes = omega_laser - self.omega
        resonance = 1 / ((omega_stokes - self.omega)**2 + self.gamma**2)

        # 強度 ∝ |振幅|² × 共鳴因子
        intensity = np.abs(amplitude)**2 * resonance

        return intensity

    def spectrum(self, omega_laser, omega_range, polarization_in, polarization_out):
        """
        ラマンスペクトルを計算（ストークスシフトの関数）

        パラメータ:
        -----------
        omega_laser : float
            レーザー周波数 (cm⁻¹)
        omega_range : array
            ラマンシフト範囲 (cm⁻¹)
        polarization_in, polarization_out : str
            偏光配置

        戻り値:
        -------
        spectrum : array
            ラマン強度スペクトル
        """
        spectrum = np.zeros_like(omega_range)

        for i, omega_shift in enumerate(omega_range):
            # ローレンツ型ピーク
            spectrum[i] = 1 / ((omega_shift - self.omega)**2 + self.gamma**2)

        # 偏光依存性の振幅因子
        pol_vectors = {
            'sigma+': np.array([1, 1j]) / np.sqrt(2),
            'sigma-': np.array([1, -1j]) / np.sqrt(2),
            'linear_x': np.array([1, 0]),
            'linear_y': np.array([0, 1])
        }

        e_in = pol_vectors[polarization_in]
        e_out = pol_vectors[polarization_out].conj()
        R = self.raman_tensor()

        amplitude = np.dot(e_out, np.dot(R, e_in))
        intensity_factor = np.abs(amplitude)**2

        return spectrum * intensity_factor


# シミュレーション実行
# WSe₂のE'モードをモデル化
omega_E_prime = 250  # cm⁻¹
gamma = 2  # cm⁻¹
omega_laser = 18797  # 532 nm → cm⁻¹

# ラマンシフト範囲
omega_shift = np.linspace(200, 300, 500)

# 右カイラルフォノン（K谷）
phonon_K = ChiralPhononRaman(omega_E_prime, gamma, chirality='right')

# 左カイラルフォノン（K'谷）
phonon_Kp = ChiralPhononRaman(omega_E_prime, gamma, chirality='left')

# 4つの偏光配置でスペクトルを計算
configs = [
    ('sigma+', 'sigma+', 'σ⁺/σ⁺'),
    ('sigma+', 'sigma-', 'σ⁺/σ⁻'),
    ('sigma-', 'sigma+', 'σ⁻/σ⁺'),
    ('sigma-', 'sigma-', 'σ⁻/σ⁻')
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (pol_in, pol_out, label) in enumerate(configs):
    # K谷（右カイラル）とK'谷（左カイラル）の寄与
    spec_K = phonon_K.spectrum(omega_laser, omega_shift, pol_in, pol_out)
    spec_Kp = phonon_Kp.spectrum(omega_laser, omega_shift, pol_in, pol_out)

    # プロット
    axes[i].plot(omega_shift, spec_K, 'b-', linewidth=2, label='K谷 (右カイラル)')
    axes[i].plot(omega_shift, spec_Kp, 'r--', linewidth=2, label="K'谷 (左カイラル)")
    axes[i].fill_between(omega_shift, 0, spec_K, alpha=0.3, color='blue')
    axes[i].fill_between(omega_shift, 0, spec_Kp, alpha=0.3, color='red')

    axes[i].set_xlabel('ラマンシフト (cm⁻¹)', fontsize=11)
    axes[i].set_ylabel('ラマン強度 (任意単位)', fontsize=11)
    axes[i].set_title(f'配置: {label}', fontsize=12, fontweight='bold')
    axes[i].legend(fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim(200, 300)

plt.tight_layout()
plt.savefig('chiral_phonon_raman_polarization.png', dpi=300, bbox_inches='tight')
plt.show()

# 円偏光度（DCP）の計算
I_pp = phonon_K.spectrum(omega_laser, np.array([omega_E_prime]), 'sigma+', 'sigma+')[0]
I_pm = phonon_K.spectrum(omega_laser, np.array([omega_E_prime]), 'sigma+', 'sigma-')[0]
DCP_K = (I_pp - I_pm) / (I_pp + I_pm)

I_mm = phonon_Kp.spectrum(omega_laser, np.array([omega_E_prime]), 'sigma-', 'sigma-')[0]
I_mp = phonon_Kp.spectrum(omega_laser, np.array([omega_E_prime]), 'sigma-', 'sigma+')[0]
DCP_Kp = (I_mm - I_mp) / (I_mm + I_mp)

print(f"\n円偏光度（DCP）:")
print(f"K谷（右カイラル）: DCP = {DCP_K:.3f}")
print(f"K'谷（左カイラル）: DCP = {DCP_Kp:.3f}")
print(f"\n理想的なカイラルフォノンではDCP ≈ ±1を示します")
```

#### コード出力の解釈

このシミュレーションは以下を示します：
- σ⁺/σ⁺配置：K谷（右カイラル）フォノンのみ検出
- σ⁺/σ⁻配置：両谷からの信号がほぼゼロ（角運動量非保存）
- σ⁻/σ⁺配置：両谷からの信号がほぼゼロ
- σ⁻/σ⁻配置：K'谷（左カイラル）フォノンのみ検出
- DCP ≈ 1：完全な谷選択性

### 3.7.2 円二色性スペクトルのシミュレーション

```python
import numpy as np
import matplotlib.pyplot as plt

# フォノン円二色性（CD）スペクトルのシミュレーション

def phonon_cd_spectrum(omega, omega_phonon, gamma, L_phonon, oscillator_strength):
    """
    フォノンモードの円二色性スペクトルを計算

    パラメータ:
    -----------
    omega : array
        周波数範囲 (cm⁻¹)
    omega_phonon : float
        フォノン共鳴周波数 (cm⁻¹)
    gamma : float
        減衰定数 (cm⁻¹)
    L_phonon : float
        フォノン角運動量 (ℏ単位)
    oscillator_strength : float
        振動子強度

    戻り値:
    -------
    CD : array
        円二色性シグナル ΔA = A_σ⁺ - A_σ⁻
    """
    # ローレンツ型吸収（虚部）
    absorption = gamma / ((omega - omega_phonon)**2 + gamma**2)

    # CD ∝ L_phonon × 吸収の微分
    # 簡略化：CD ∝ L_phonon × Im[ε_xy]
    CD = L_phonon * oscillator_strength * absorption

    return CD


# α-石英のパラメータ（Zhu et al. 2018に基づく）
phonon_modes = [
    {'omega': 400, 'gamma': 5, 'L': 0.1, 'f': 0.3},   # 弱いカイラルモード
    {'omega': 700, 'gamma': 8, 'L': 0.2, 'f': 0.5},   # 中程度
    {'omega': 1100, 'gamma': 10, 'L': 0.3, 'f': 1.0}, # 強いSi-O伸縮
]

omega_range = np.linspace(300, 1400, 2000)

# CDスペクトル計算
CD_total = np.zeros_like(omega_range)

for mode in phonon_modes:
    CD_mode = phonon_cd_spectrum(
        omega_range,
        mode['omega'],
        mode['gamma'],
        mode['L'],
        mode['f']
    )
    CD_total += CD_mode

# プロット
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 上パネル：個別モード
axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.8)
for i, mode in enumerate(phonon_modes):
    CD_mode = phonon_cd_spectrum(
        omega_range,
        mode['omega'],
        mode['gamma'],
        mode['L'],
        mode['f']
    )
    axes[0].plot(omega_range, CD_mode, linewidth=2,
                label=f"ω = {mode['omega']} cm⁻¹, L = {mode['L']}ℏ")
    axes[0].fill_between(omega_range, 0, CD_mode, alpha=0.3)

axes[0].set_xlabel('周波数 (cm⁻¹)', fontsize=12)
axes[0].set_ylabel('円二色性 ΔA (任意単位)', fontsize=12)
axes[0].set_title('個別フォノンモードのCD寄与', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(300, 1400)

# 下パネル：全CDスペクトル
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.8)
axes[1].plot(omega_range, CD_total, 'b-', linewidth=2.5, label='全CDスペクトル')
axes[1].fill_between(omega_range, 0, CD_total, alpha=0.4, color='blue')

# 強度の最大値を示す
max_idx = np.argmax(np.abs(CD_total))
axes[1].plot(omega_range[max_idx], CD_total[max_idx], 'ro', markersize=10,
            label=f'最大CD: {omega_range[max_idx]:.0f} cm⁻¹')

axes[1].set_xlabel('周波数 (cm⁻¹)', fontsize=12)
axes[1].set_ylabel('円二色性 ΔA (任意単位)', fontsize=12)
axes[1].set_title('α-石英の全フォノンCDスペクトル（シミュレーション）', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(300, 1400)

plt.tight_layout()
plt.savefig('phonon_circular_dichroism.png', dpi=300, bbox_inches='tight')
plt.show()

# CDの温度依存性シミュレーション
temperatures = [10, 100, 200, 300]  # K
fig2, ax2 = plt.subplots(figsize=(12, 7))

for T in temperatures:
    # 温度によるブロードニング: γ(T) = γ₀(1 + 2n_B(T))
    # ボース分布: n_B = 1/(exp(ℏω/k_BT) - 1)
    k_B = 0.695  # cm⁻¹/K（ボルツマン定数）

    CD_T = np.zeros_like(omega_range)
    for mode in phonon_modes:
        # 温度依存ブロードニング
        n_B = 1 / (np.exp(mode['omega'] / (k_B * T)) - 1) if T > 0 else 0
        gamma_T = mode['gamma'] * (1 + 2 * n_B)

        CD_mode = phonon_cd_spectrum(
            omega_range,
            mode['omega'],
            gamma_T,
            mode['L'],
            mode['f']
        )
        CD_T += CD_mode

    ax2.plot(omega_range, CD_T, linewidth=2, label=f'T = {T} K')

ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
ax2.set_xlabel('周波数 (cm⁻¹)', fontsize=12)
ax2.set_ylabel('円二色性 ΔA (任意単位)', fontsize=12)
ax2.set_title('フォノンCDスペクトルの温度依存性', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(300, 1400)

plt.tight_layout()
plt.savefig('phonon_cd_temperature.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nフォノン円二色性の主要特性:")
print(f"最大CDピーク: {omega_range[max_idx]:.1f} cm⁻¹")
print(f"最大CD強度: {CD_total[max_idx]:.3f} (任意単位)")
print(f"\n温度上昇に伴い:")
print(f"- ピークがブロードニング（フォノン-フォノン散乱）")
print(f"- CD強度が減少（熱的乱れ）")
```

---

## まとめ

### 重要なポイント

- **円偏光ラマン分光法**は、カイラルフォノン検出の最も強力で広く使用される技術である
- **1/4波長板**を用いて直線偏光を円偏光（σ⁺/σ⁻）に変換し、入射・散乱光の偏光状態を制御する
- **σ⁺/σ⁺とσ⁺/σ⁻配置**は、角運動量保存則に基づき異なるフォノンモードを選択的に検出する
- **ラマン選択則**：ΔL_光 = -L_phの関係から、カイラルフォノンは特定の偏光配置でのみ観測される
- **円偏光度（DCP）**は、カイラルフォノンの強さを定量化する重要な指標（理想的に±1）
- **赤外円二色性（IR-CD）**は、フォノン角運動量の直接的な光学測定を可能にする
- **ヘリシティ分解PL**は、バレー-フォノン結合を介したカイラルフォノンの役割を明らかにする
- **超高速分光法**は、カイラルフォノンのダイナミクスを実時間で観測できる（フェムト秒分解能）
- **中性子・X線散乱**は、原理的には可能だが、技術的課題が大きい
- **WSe₂実験**は、2D TMDにおけるバレー選択的カイラルフォノンを初めて実証した
- **α-石英実験**は、3Dカイラル結晶におけるフォノン角運動量の光学測定を初めて達成した

---

## 演習問題

### 問題1：偏光変換の理解

1/4波長板の速軸に対して以下の角度で入射する直線偏光を考えます：
a) 0°（速軸に平行）
b) 45°
c) 90°（遅軸に平行）

それぞれの場合の出射光の偏光状態を記述してください。σ⁺またはσ⁻円偏光が得られるのはどの場合ですか？

### 問題2：ラマン選択則の導出

カイラルフォノン（フォノン角運動量L_ph = +2ℏ）を考えます。以下の偏光配置でラマン散乱が許容されるか、角運動量保存則に基づいて判定してください：
a) σ⁺入射、σ⁺散乱
b) σ⁺入射、σ⁻散乱
c) σ⁻入射、σ⁺散乱
d) σ⁻入射、σ⁻散乱

光子のスピン角運動量：σ⁺ = +ℏ、σ⁻ = -ℏを使用してください。

### 問題3：実験データの解釈

ある材料で円偏光ラマン測定を行い、以下の相対強度を得ました（任意単位）：
- I(σ⁺/σ⁺) = 100
- I(σ⁺/σ⁻) = 10
- I(σ⁻/σ⁺) = 90
- I(σ⁻/σ⁻) = 100

a) 円偏光度（DCP）を計算してください
b) このフォノンはカイラルですか？その理由を説明してください
c) σ⁺/σ⁻とσ⁻/σ⁺の強度差は何を意味しますか？

### 問題4：超高速分光のデータ解析

ポンプ-プローブ測定で以下の時間依存信号が観測されました：

```
ΔR/R(t) = 0.05 × cos(250 × 2πc × t) × exp(-t/2 ps)
```

ここで、cは光速（cm/s）、tは時間（秒）です。

a) フォノンのエネルギー（meV）を求めてください（1 cm⁻¹ ≈ 0.124 meV）
b) フォノン寿命（ps）を求めてください
c) 線幅（FWHM、cm⁻¹）を推定してください（Γ = ℏ/τ）

### 問題5（発展）：円二色性の定量的解析

α-石英のフォノンモード（ω = 1100 cm⁻¹）で、以下の吸収係数が測定されました：
- A_σ⁺ = 0.525
- A_σ⁻ = 0.475

a) 規格化CD = (A_σ⁺ - A_σ⁻)/(A_σ⁺ + A_σ⁻)を計算してください
b) このCDがフォノン角運動量L_ph = 0.3ℏに対応すると仮定して、L_phとCDの比例定数を推定してください
c) 対掌体（鏡像異性体）の石英では、CDシグナルはどうなりますか？

---

[← 第2章](chapter-2.md) | [第4章 →](chapter-4.md)

---

## 免責事項

本記事は教育目的で作成されており、学術的な正確性を目指していますが、研究の進展により内容が更新される可能性があります。実際の研究や応用においては、最新の文献を参照してください。

**著者**: MS知識ハブコンテンツチーム
**バージョン**: 1.0 | **最終更新**: 2025-12-19
**ライセンス**: クリエイティブ・コモンズ BY 4.0
