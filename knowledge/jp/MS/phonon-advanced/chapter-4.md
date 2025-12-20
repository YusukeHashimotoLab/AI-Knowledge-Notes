---
title: "第4章: トポロジカルフォノン"
subtitle: ""
series: "フォノン物理学(上級)"
chapter: 4
language: "ja"
---

[🌐 EN](../../../en/MS/phonon-advanced/chapter-4.md) | 🇯🇵 JP | Last sync: 2025-12-20

[材料科学道場](../index.html) > [フォノン物理学(上級)](index.md) > 第4章

# 第4章: トポロジカルフォノン

# 第4章: トポロジカルフォノン

上級 トポロジカル物性 幾何学的位相 バンドトポロジー

## 学習目標

  * フォノンのベリー位相とベリー曲率の物理的意味を理解する
  * トポロジカル不変量（チャーン数、Z2不変量）の計算方法を習得する
  * ワイルフォノンとディラックフォノンの特徴を理解する
  * トポロジカル保護されたフォノンエッジ状態の性質を学ぶ
  * 実材料におけるトポロジカルフォノンの実験的兆候を理解する
  * フォノンホール効果と熱輸送特性との関連を理解する
  * Pythonを用いたベリー位相計算の実装方法を習得する


## 1\. トポロジカルフォノンとは

トポロジカルフォノンは、電子系のトポロジカル絶縁体に類似した概念をフォノン（格子振動）に適用したものです。フォノンバンド構造がトポロジカルに非自明な性質を持つことで、バルク-エッジ対応により保護された表面・エッジ状態や、異常な輸送現象が現れます。

### 1.1 電子系とフォノン系の類似性

性質 | 電子系 | フォノン系  
---|---|---  
量子統計 | フェルミオン | ボソン  
バンド構造 | 電子バンド | フォノンバンド  
トポロジカル不変量 | チャーン数、Z2 | チャーン数、Z2  
ワイル点 | ワイル半金属 | ワイルフォノン  
エッジ状態 | スピンホール効果 | フォノンホール効果  
時間反転対称性 | スピン軌道相互作用で破れる | 回転系、磁場で破れる  
  
### 1.2 トポロジカルフォノンの分類

graph TD A[トポロジカルフォノン] --> B[時間反転対称性保存] A --> C[時間反転対称性破れ] B --> D[Z2トポロジカル絶縁体型] B --> E[ディラック/ワイルフォノン] C --> F[チャーン絶縁体型] C --> G[アノマラスホール効果] D --> H[エッジ状態: クラマース対] E --> I[ノードポイント: ワイル点] F --> J[エッジ状態: カイラルモード] G --> K[横熱流: フォノンホール効果] 

**重要概念:**

  * **バルク-エッジ対応:** バルクのトポロジカル不変量が非零のとき、境界に保護されたエッジ状態が現れる
  * **トポロジカル保護:** 対称性が保たれる限り、エッジ状態は散乱に対して頑強
  * **ベリー曲率:** 波動関数の幾何学的位相が運動量空間で生成する「磁場」


## 2\. ベリー位相とベリー曲率

### 2.1 ベリー位相の定義

パラメータ空間（運動量空間）\\(\mathbf{k}\\)を断熱的に変化させたときに波動関数が獲得する幾何学的位相をベリー位相と呼びます。

\[\gamma_n = i \oint_C \langle u_{n\mathbf{k}} | \nabla_{\mathbf{k}} | u_{n\mathbf{k}} \rangle \cdot d\mathbf{k}\] 

ここで、

  * \\(|u_{n\mathbf{k}}\rangle\\): バンド\\(n\\)のブロッホ波動関数の周期部分
  * \\(C\\): 運動量空間内の閉曲線
  * \\(\gamma_n\\): ベリー位相


### 2.2 ベリー接続とベリー曲率

**ベリー接続** （ベリーポテンシャル）:

\[\mathcal{A}_{n}(\mathbf{k}) = i \langle u_{n\mathbf{k}} | \nabla_{\mathbf{k}} | u_{n\mathbf{k}} \rangle\] 

**ベリー曲率** :

\[\boldsymbol{\Omega}_n(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathcal{A}_n(\mathbf{k})\] 

成分表示では:

\[\Omega_{n,\mu\nu}(\mathbf{k}) = \partial_{k_\mu} \mathcal{A}_{n,\nu} - \partial_{k_\nu} \mathcal{A}_{n,\mu}\] 

### 2.3 フォノンのベリー曲率

フォノンの場合、固有ベクトル\\(|\mathbf{e}_{n\mathbf{k}}\rangle\\)（偏極ベクトル）を用いて:

\[\Omega_{n,xy}(\mathbf{k}) = -2\text{Im}\sum_{m \neq n} \frac{\langle \mathbf{e}_{n\mathbf{k}} | \partial_{k_x} D(\mathbf{k}) | \mathbf{e}_{m\mathbf{k}} \rangle \langle \mathbf{e}_{m\mathbf{k}} | \partial_{k_y} D(\mathbf{k}) | \mathbf{e}_{n\mathbf{k}} \rangle}{[\omega_n(\mathbf{k}) - \omega_m(\mathbf{k})]^2}\] 

ここで、\\(D(\mathbf{k})\\)は動力学行列です。

**物理的解釈:** ベリー曲率は運動量空間の「磁場」に対応し、フォノンの異常速度や横方向の熱流（フォノンホール効果）を引き起こします。 

### 2.4 ベリー位相の計算例

Python コピー
    
    
    import numpy as np
    from scipy.linalg import eigh
    
    def compute_berry_curvature(dynamical_matrix_func, kx, ky, band_index, dk=1e-5):
        """
        フォノンのベリー曲率を数値微分で計算
    
        Parameters:
        -----------
        dynamical_matrix_func : callable
            k点を引数にとり動力学行列を返す関数
        kx, ky : float
            運動量空間の座標
        band_index : int
            対象バンドのインデックス
        dk : float
            数値微分の刻み幅
    
        Returns:
        --------
        omega_z : float
            z方向のベリー曲率成分
        """
        k_points = [
            [kx + dk, ky],
            [kx - dk, ky],
            [kx, ky + dk],
            [kx, ky - dk]
        ]
    
        eigenvectors = []
        for kpt in k_points:
            D = dynamical_matrix_func(kpt[0], kpt[1])
            eigenvals, eigvecs = eigh(D)
            eigenvectors.append(eigvecs[:, band_index])
    
        # 数値微分による接続の計算
        Ax_plus = 1j * np.log(np.vdot(eigenvectors[0], eigenvectors[0]))
        Ax_minus = 1j * np.log(np.vdot(eigenvectors[1], eigenvectors[1]))
        Ay_plus = 1j * np.log(np.vdot(eigenvectors[2], eigenvectors[2]))
        Ay_minus = 1j * np.log(np.vdot(eigenvectors[3], eigenvectors[3]))
    
        # ベリー曲率の計算
        dAy_dx = (Ay_plus - Ay_minus) / (2 * dk)
        dAx_dy = (Ax_plus - Ax_minus) / (2 * dk)
    
        omega_z = dAy_dx - dAx_dy
    
        return np.real(omega_z)
    
    # 使用例: 2Dハニカム格子
    def honeycomb_dynamical_matrix(kx, ky):
        """ハニカム格子の簡易動力学行列"""
        a = 1.0  # 格子定数
        K = 1.0  # ばね定数
    
        # 最近接相互作用のみの簡易モデル
        delta_1 = np.array([0, 0])
        delta_2 = np.array([a*np.sqrt(3)/2, a/2])
        delta_3 = np.array([a*np.sqrt(3)/2, -a/2])
    
        D = np.zeros((6, 6), dtype=complex)
    
        # 動力学行列の構築（簡略化）
        phase_1 = np.exp(1j * (kx * delta_1[0] + ky * delta_1[1]))
        phase_2 = np.exp(1j * (kx * delta_2[0] + ky * delta_2[1]))
        phase_3 = np.exp(1j * (kx * delta_3[0] + ky * delta_3[1]))
    
        # 対角成分
        D[0:3, 0:3] = 3 * K * np.eye(3)
        D[3:6, 3:6] = 3 * K * np.eye(3)
    
        # 非対角成分（サブラティス間相互作用）
        coupling = K * (phase_1 + phase_2 + phase_3)
        D[0:3, 3:6] = -coupling * np.eye(3)
        D[3:6, 0:3] = -coupling * np.eye(3)
    
        return D
    
    # K点近傍でのベリー曲率マップ
    kx_range = np.linspace(-0.2, 0.2, 20)
    ky_range = np.linspace(-0.2, 0.2, 20)
    
    berry_curvature_map = np.zeros((len(kx_range), len(ky_range)))
    
    for i, kx in enumerate(kx_range):
        for j, ky in enumerate(ky_range):
            berry_curvature_map[i, j] = compute_berry_curvature(
                honeycomb_dynamical_matrix, kx, ky, band_index=3
            )
    
    print("ベリー曲率の最大値:", np.max(np.abs(berry_curvature_map)))
    print("ベリー曲率の積分 (チャーン数の近似):",
          np.sum(berry_curvature_map) * (kx_range[1] - kx_range[0]) * (ky_range[1] - ky_range[0]) / (2 * np.pi))

## 3\. トポロジカル不変量

### 3.1 チャーン数

2次元系において、第1ブリルアンゾーンでのベリー曲率の積分がトポロジカル不変量（チャーン数）を与えます:

\[C_n = \frac{1}{2\pi} \int_{\text{BZ}} \Omega_{n,xy}(\mathbf{k}) \, d^2\mathbf{k}\] 

チャーン数は整数値をとり、バンドのトポロジカルな性質を特徴づけます。

**チャーン数の物理的意味:**

  * \\(C_n \neq 0\\): トポロジカルに非自明、エッジ状態が存在
  * \\(C_n = 0\\): トポロジカルに自明、通常の絶縁体
  * 異なるチャーン数を持つバンド間の遷移にはギャップの閉じが必要


### 3.2 Z2不変量

時間反転対称性がある3次元系では、Z2トポロジカル不変量\\(\nu_0; (\nu_1\nu_2\nu_3)\\)を定義できます:

\[(-1)^{\nu_0} = \prod_{i=1}^{8} \sqrt{\frac{\det[w(\Lambda_i)]}{\text{Pf}[w(\Lambda_i)]}}\] 

ここで、\\(\Lambda_i\\)は時間反転不変運動量（TRIM）点、\\(w(\Lambda_i)\\)はそこでのワイル行列です。

### 3.3 ワイル点のカイラリティ

ワイル点（ディラック点が分裂したもの）周りのベリー曲率の積分はカイラリティ\\(\chi = \pm 1\\)を与えます:

\[\chi = \frac{1}{2\pi} \oint_S \boldsymbol{\Omega} \cdot d\mathbf{S}\] 

ここで、\\(S\\)はワイル点を囲む閉曲面です。

graph LR A[ディラック点] -->|対称性の破れ| B[ワイル点 +] A -->|対称性の破れ| C[ワイル点 -] B --> D[カイラリティ +1] C --> E[カイラリティ -1] D --> F[フェルミアーク] E --> F style B fill:#ffcccc style C fill:#ccccff 

### 3.4 ミラーチャーン数

ミラー対称性がある系では、ミラー平面ごとにチャーン数を定義できます:

\[C_{\mathcal{M}} = \frac{1}{2\pi} \int_{\text{BZ}_{\mathcal{M}}} \Omega_{xy}(\mathbf{k}) \, d^2\mathbf{k}\] 

ここで、積分はミラー不変部分空間に制限されます。

## 4\. ワイル・ディラックフォノン

### 4.1 ディラックフォノン

ディラック点は、2つの縮退したバンドが線形に交差する点です。ディラック点近傍のハミルトニアンは:

\[H(\mathbf{q}) = \hbar v_x q_x \sigma_x + \hbar v_y q_y \sigma_y + \hbar v_z q_z \sigma_z\] 

ここで、\\(\mathbf{q}\\)はディラック点からの運動量のずれ、\\(\sigma_i\\)はパウリ行列、\\(v_i\\)は速度パラメータです。

### 4.2 ワイルフォノン

対称性が破れるとディラック点が分裂し、ワイル点が現れます。ワイル点は運動量空間における「モノポール」として振る舞います。

**ワイルフォノンの特徴:**

  * 時間反転対称性または空間反転対称性の少なくとも一方が破れている
  * 必ず対で現れる（カイラリティが逆）
  * ワイル点間をつなぐフォノンアーク（表面状態）が存在
  * 異常な熱輸送特性を示す


### 4.3 フォノンアーク

ワイルフォノンの表面状態は、異なるカイラリティのワイル点をつなぐ「アーク」状の分散を示します。

graph TD subgraph "表面ブリルアンゾーン" A[ワイル点 +1] -->|フォノンアーク| B[ワイル点 -1] end subgraph "バルク" C[ワイル点対] end A -.投影.- C B -.投影.- C style A fill:#ffcccc style B fill:#ccccff 

### 4.4 実験的検出

ワイル・ディラックフォノンの実験的検出方法:

  * **中性子散乱:** 角度分解測定でフォノンアークを直接観測
  * **ラマン散乱:** トポロジカル相転移に伴うピーク変化
  * **熱伝導測定:** 異常な温度依存性
  * **第一原理計算:** バンド構造とベリー曲率の計算


## 5\. エッジ・表面状態

### 5.1 バルク-エッジ対応

トポロジカルフォノン系における基本原理:

\[N_{\text{edge}} = C_{\text{bulk,L}} - C_{\text{bulk,R}}\] 

ここで、\\(N_{\text{edge}}\\)はエッジモードの数、\\(C_{\text{bulk,L/R}}\\)は左右のバルクのチャーン数です。

### 5.2 トポロジカル保護

エッジ状態は以下の性質を持ちます:

  * 対称性が保たれる限り、散乱や不純物に対して頑強
  * バックスキャッタリングが禁止される
  * 一方向伝播（カイラルエッジモード）または反対方向の対（ヘリカルエッジモード）


### 5.3 カイラルエッジフォノン

時間反転対称性が破れた2次元系（チャーン絶縁体型）では、エッジに一方向伝播するカイラルモードが現れます。

graph LR subgraph "上エッジ" A1[左端] -->|カイラルモード| A2[右端] end subgraph "バルク C=1" B[ギャップ] end subgraph "下エッジ" C1[右端] -->|カイラルモード| C2[左端] end style A1 fill:#ffcccc style C2 fill:#ffcccc 

### 5.4 ヘリカルエッジフォノン

時間反転対称性が保たれたZ2トポロジカル絶縁体型では、スピン偏極した対のエッジモード（クラマース対）が現れます。

Python コピー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import eigh
    
    def compute_edge_states_1d_chain(N_sites, hopping_t, on_site_energy, boundary='open'):
        """
        1次元鎖のエッジ状態を計算（SSH模型の類似）
    
        Parameters:
        -----------
        N_sites : int
            サイト数
        hopping_t : array-like [t1, t2]
            交互のホッピングパラメータ
        on_site_energy : float
            オンサイトエネルギー
        boundary : str
            'open' または 'periodic'
    
        Returns:
        --------
        eigenvalues, eigenvectors
        """
        t1, t2 = hopping_t
        H = np.zeros((N_sites, N_sites))
    
        # オンサイト項
        for i in range(N_sites):
            H[i, i] = on_site_energy
    
        # ホッピング項（交互）
        for i in range(N_sites - 1):
            if i % 2 == 0:
                H[i, i+1] = H[i+1, i] = t1
            else:
                H[i, i+1] = H[i+1, i] = t2
    
        # 周期境界条件
        if boundary == 'periodic':
            H[0, N_sites-1] = H[N_sites-1, 0] = t2 if (N_sites-1) % 2 == 1 else t1
    
        eigenvalues, eigenvectors = eigh(H)
        return eigenvalues, eigenvectors
    
    # トポロジカル相のパラメータ
    N = 100
    t1, t2 = 1.0, 0.5  # t1 > t2: トポロジカル相
    E_onsite = 0.0
    
    eigenvals_open, eigvecs_open = compute_edge_states_1d_chain(
        N, [t1, t2], E_onsite, boundary='open'
    )
    
    # エッジ状態の可視化
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # エネルギースペクトル
    axes[0].scatter(range(N), eigenvals_open, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', label='ゼロエネルギー')
    axes[0].set_xlabel('状態インデックス')
    axes[0].set_ylabel('エネルギー')
    axes[0].set_title('エネルギースペクトル（開境界）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ゼロエネルギー近傍の状態の空間分布
    zero_energy_states = np.where(np.abs(eigenvals_open) < 0.1)[0]
    if len(zero_energy_states) >= 2:
        # 左エッジ状態
        left_edge_state = eigvecs_open[:, zero_energy_states[0]]
        # 右エッジ状態
        right_edge_state = eigvecs_open[:, zero_energy_states[-1]]
    
        axes[1].plot(np.abs(left_edge_state)**2, 'b-', label='左エッジ状態', linewidth=2)
        axes[1].plot(np.abs(right_edge_state)**2, 'r-', label='右エッジ状態', linewidth=2)
        axes[1].set_xlabel('サイト位置')
        axes[1].set_ylabel('確率密度 |ψ|²')
        axes[1].set_title('エッジ状態の空間分布')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('edge_states_ssh.png', dpi=150, bbox_inches='tight')
    print(f"ゼロエネルギー状態数: {len(zero_energy_states)}")
    print(f"トポロジカル相: {'Yes' if t1 > t2 else 'No'}")

## 6\. 対称性の破れとトポロジー

### 6.1 時間反転対称性の破れ

フォノン系における時間反転対称性の破れは以下の方法で実現できます:

  * **回転系:** コリオリ力により実効的に破れる
  * **外部磁場:** 磁性イオンを含む系で破れる
  * **循環熱流:** 非平衡状態で破れる
  * **フォノン-マグノン結合:** スピン-格子相互作用により破れる


\[\mathcal{T}: \mathbf{u}(\mathbf{r}, t) \to \mathbf{u}(\mathbf{r}, -t)\] 

時間反転対称性が破れると、チャーン絶縁体型のトポロジカル相が実現できます。

### 6.2 空間反転対称性の破れ

空間反転対称性の破れは:

  * 非対称結晶構造（非中心対称空間群）
  * 圧電材料
  * 表面や界面


で自然に実現します。空間反転対称性が破れると、ワイルフォノンが現れやすくなります。

### 6.3 対称性とトポロジカル分類

対称性クラス | 時間反転 | 空間反転 | トポロジカル相 | 不変量  
---|---|---|---|---  
A（無対称性） | × | × | チャーン絶縁体 | チャーン数 \\(C \in \mathbb{Z}\\)  
AI（時間反転+空間反転） | ○ | ○ | Z2トポロジカル絶縁体 | Z2不変量 \\(\nu \in \\{0,1\\}\\)  
半金属（ノード） | ○/× | × | ワイル/ディラック | カイラリティ \\(\chi = \pm 1\\)  
ミラー対称性 | ○ | 部分的 | ミラーチャーン絶縁体 | ミラーチャーン数  
  
### 6.4 対称性指示子（Symmetry Indicator）

空間群の対称性から、高対称点での固有値の組み合わせがトポロジカル相を示唆する場合があります。これを対称性指示子と呼びます。

\[\text{SI} = \sum_{i \in \text{TRIM}} n_i \cdot \text{irrep}_i \mod 2\] 

ここで、\\(n_i\\)はTRIM点\\(i\\)での占有バンド数、\\(\text{irrep}_i\\)は既約表現です。

## 7\. フォノンホール効果

### 7.1 横熱流と熱ホール効果

温度勾配に垂直な方向に熱流が流れる現象をフォノンホール効果と呼びます。これはベリー曲率が原因で生じます。

\[\mathbf{j}_Q = -\kappa_{xx} \nabla T \hat{x} - \kappa_{xy} \nabla T \hat{y}\] 

熱ホール伝導率\\(\kappa_{xy}\\)は:

\[\kappa_{xy} = \frac{k_B^2 T}{\hbar} \sum_n \int_{\text{BZ}} \frac{d^2\mathbf{k}}{(2\pi)^2} c_2\left(\frac{\hbar\omega_n}{k_B T}\right) \Omega_{n,xy}(\mathbf{k})\] 

ここで、\\(c_2(x) = (1+x)\left(\frac{x}{e^x-1}\right)^2 - \left(\frac{x}{e^x-1}\right)\\)はボソン分布関数から導かれる熱因子です。

### 7.2 異常速度

フォノンの群速度にベリー曲率由来の異常項が加わります:

\[\mathbf{v}_n = \nabla_{\mathbf{k}} \omega_n(\mathbf{k}) - \dot{\mathbf{k}} \times \boldsymbol{\Omega}_n(\mathbf{k})\] 

第二項が異常速度で、外力（温度勾配による実効的な力）と垂直な運動を引き起こします。

### 7.3 実験的観測

フォノンホール効果の測定:

  * **横方向熱測定:** 温度勾配に垂直な方向の温度差を測定
  * **磁場依存性:** 時間反転対称性の破れを磁場で制御
  * **温度依存性:** 低温でのピーク構造


**実験例:**

  * パラ磁性絶縁体Tb3Ga5O12: \\(\kappa_{xy} \sim 0.1\\) W/K·m at 5K
  * 強磁性絶縁体Lu2V2O7: 磁場依存的な熱ホール効果
  * 常磁性体SrTiO3: 磁場誘起熱ホール効果


### 7.4 計算例

Python コピー
    
    
    import numpy as np
    from scipy.integrate import simps
    
    def thermal_factor_c2(x):
        """熱ホール効果の熱因子 c2(x)"""
        # x = ℏω / (k_B T)
        exp_x = np.exp(x)
        bose = x / (exp_x - 1)
        c2 = (1 + x) * bose**2 - bose
        return c2
    
    def compute_thermal_hall_conductivity(omega_bands, berry_curvature, temperature, k_mesh):
        """
        熱ホール伝導率の計算
    
        Parameters:
        -----------
        omega_bands : ndarray, shape (n_bands, nkx, nky)
            フォノン分散 (rad/s)
        berry_curvature : ndarray, shape (n_bands, nkx, nky)
            ベリー曲率 (Å²)
        temperature : float
            温度 (K)
        k_mesh : tuple (kx_array, ky_array)
            運動量メッシュ
    
        Returns:
        --------
        kappa_xy : float
            熱ホール伝導率 (W/K·m)
        """
        hbar = 1.054571817e-34  # J·s
        kB = 1.380649e-23       # J/K
    
        kx, ky = k_mesh
        dk = (kx[1] - kx[0]) * (ky[1] - ky[0])  # k空間の面積要素
    
        kappa_xy = 0.0
    
        for band_idx in range(omega_bands.shape[0]):
            omega = omega_bands[band_idx]
            Omega = berry_curvature[band_idx]
    
            # x = ℏω / (k_B T)
            x = hbar * omega / (kB * temperature)
    
            # 熱因子の計算
            c2_factor = thermal_factor_c2(x)
    
            # 積分（シンプソン則）
            integrand = c2_factor * Omega
            integral = simps(simps(integrand, ky), kx)
    
            kappa_xy += integral
    
        # 係数
        prefactor = (kB**2 * temperature) / (hbar * (2 * np.pi)**2)
        kappa_xy *= prefactor * dk
    
        return kappa_xy
    
    # 使用例: 2Dハニカム格子
    nk = 50
    kx = np.linspace(-np.pi, np.pi, nk)
    ky = np.linspace(-np.pi, np.pi, nk)
    KX, KY = np.meshgrid(kx, ky)
    
    # 簡易的なフォノン分散（2バンド）
    v_sound = 5000  # m/s (音速)
    hbar = 1.054571817e-34
    omega_acoustic = v_sound * np.sqrt(KX**2 + KY**2) / 1e-10  # rad/s
    
    # 簡易的なベリー曲率（ディラック点周りにピーク）
    berry_curv = np.zeros_like(KX)
    K_point_x, K_point_y = 4*np.pi/3, 0
    distance = np.sqrt((KX - K_point_x)**2 + (KY - K_point_y)**2)
    berry_curv = 0.5 * np.exp(-distance**2 / 0.1)  # Å²単位
    
    omega_bands = np.array([omega_acoustic])
    berry_curvature = np.array([berry_curv])
    
    # 温度依存性
    temperatures = np.logspace(0, 2, 20)  # 1K to 100K
    kappa_xy_values = []
    
    for T in temperatures:
        kappa_xy = compute_thermal_hall_conductivity(
            omega_bands, berry_curvature, T, (kx, ky)
        )
        kappa_xy_values.append(kappa_xy)
    
    # 結果のプロット
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.semilogx(temperatures, kappa_xy_values, 'o-', linewidth=2)
    plt.xlabel('温度 (K)', fontsize=12)
    plt.ylabel('熱ホール伝導率 κ_xy (W/K·m)', fontsize=12)
    plt.title('熱ホール伝導率の温度依存性', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('thermal_hall_conductivity.png', dpi=150)
    
    print(f"最大値: {np.max(kappa_xy_values):.2e} W/K·m at {temperatures[np.argmax(kappa_xy_values)]:.1f} K")

## 8\. 2次元材料のトポロジカルフォノン

### 8.1 グラフェンのディラックフォノン

グラフェンは、K点とK'点にディラックフォノンを持ちます。低エネルギー有効ハミルトニアンは:

\[H_{\text{Dirac}} = \hbar v_{\text{ph}} (\tau q_x \sigma_x + q_y \sigma_y)\] 

ここで、\\(\tau = \pm 1\\)は谷の指標、\\(v_{\text{ph}}\\)はフォノンの速度です。

**グラフェンのフォノン特性:**

  * 面外光学モード（ZO）がディラック点を形成
  * 谷自由度による擬スピン偏極
  * 高い熱伝導率（室温で5000 W/m·K）
  * ひずみによるトポロジカル相転移の可能性


### 8.2 遷移金属ダイカルコゲナイド（TMD）

MoS2、WS2などのTMDは、空間反転対称性が破れており、谷依存的なベリー曲率を持ちます。

  * **谷ホール効果:** K谷とK'谷で逆符号のベリー曲率
  * **スピン-谷結合:** スピン軌道相互作用により谷とスピンが結合
  * **層数依存性:** 単層はトポロジカル、二層は自明な絶縁体


### 8.3 h-BN（六方晶窒化ホウ素）

h-BNは大きなバンドギャップを持つ絶縁体ですが、フォノンバンドにトポロジカルな性質があります。

\[C_{\text{phonon}} = +2 \quad (\text{lower optical bands})\] 

これにより、エッジに一方向伝播するフォノンモードが現れます。

### 8.4 2D材料のベリー曲率計算

Python コピー
    
    
    import numpy as np
    from scipy.linalg import eigh
    
    def graphene_phonon_hamiltonian(kx, ky, a=1.42e-10, K_spring=100):
        """
        グラフェンの簡易フォノンハミルトニアン
    
        Parameters:
        -----------
        kx, ky : float
            運動量 (1/m)
        a : float
            炭素間距離 (m)
        K_spring : float
            ばね定数 (N/m)
    
        Returns:
        --------
        H : ndarray, shape (6, 6)
            動力学行列
        """
        # 最近接ベクトル
        delta1 = np.array([a, 0])
        delta2 = np.array([-a/2, a*np.sqrt(3)/2])
        delta3 = np.array([-a/2, -a*np.sqrt(3)/2])
    
        # 構造因子
        f_k = (np.exp(1j * (kx * delta1[0] + ky * delta1[1])) +
               np.exp(1j * (kx * delta2[0] + ky * delta2[1])) +
               np.exp(1j * (kx * delta3[0] + ky * delta3[1])))
    
        mass = 12 * 1.66054e-27  # 炭素の質量 (kg)
    
        # 動力学行列（簡略版: x-y平面内のみ）
        H = np.zeros((6, 6), dtype=complex)
    
        # サブラティスA
        H[0:3, 0:3] = (3 * K_spring / mass) * np.eye(3)
        # サブラティスB
        H[3:6, 3:6] = (3 * K_spring / mass) * np.eye(3)
    
        # サブラティス間相互作用
        coupling = -(K_spring / mass) * f_k
        H[0, 3] = H[3, 0] = coupling
        H[1, 4] = H[4, 1] = coupling
        H[2, 5] = H[5, 2] = coupling
    
        return H
    
    def compute_berry_curvature_graphene(kx_range, ky_range, band_idx=2):
        """グラフェンK点近傍のベリー曲率マップ"""
        nkx, nky = len(kx_range), len(ky_range)
        berry_map = np.zeros((nkx, nky))
    
        dk = 1e-5  # 数値微分の刻み
    
        for i, kx in enumerate(kx_range):
            for j, ky in enumerate(ky_range):
                # 4点での固有ベクトル
                H_plus_x = graphene_phonon_hamiltonian(kx + dk, ky)
                H_minus_x = graphene_phonon_hamiltonian(kx - dk, ky)
                H_plus_y = graphene_phonon_hamiltonian(kx, ky + dk)
                H_minus_y = graphene_phonon_hamiltonian(kx, ky - dk)
    
                _, v_px = eigh(H_plus_x)
                _, v_mx = eigh(H_minus_x)
                _, v_py = eigh(H_plus_y)
                _, v_my = eigh(H_minus_y)
    
                # ベリー接続の計算
                vec_px = v_px[:, band_idx]
                vec_mx = v_mx[:, band_idx]
                vec_py = v_py[:, band_idx]
                vec_my = v_my[:, band_idx]
    
                # 位相の連続性を保つ
                if np.vdot(vec_px, vec_mx) < 0:
                    vec_mx = -vec_mx
                if np.vdot(vec_py, vec_my) < 0:
                    vec_my = -vec_my
    
                # ベリー接続（簡易計算）
                Ax = np.imag(np.log(np.vdot(vec_px, vec_mx) + 1e-10))
                Ay = np.imag(np.log(np.vdot(vec_py, vec_my) + 1e-10))
    
                # ベリー曲率（回転）
                berry_map[i, j] = (Ay - Ax) / (2 * dk)
    
        return berry_map
    
    # K点近傍での計算
    a = 1.42e-10
    K_point = 4 * np.pi / (3 * a)
    kx_range = np.linspace(K_point - 1e9, K_point + 1e9, 30)
    ky_range = np.linspace(-1e9, 1e9, 30)
    
    berry_curv_graphene = compute_berry_curvature_graphene(kx_range, ky_range)
    
    # 可視化
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.contourf(kx_range / 1e9, ky_range / 1e9, berry_curv_graphene.T,
                 levels=20, cmap='RdBu_r')
    plt.colorbar(label='ベリー曲率 (m²)')
    plt.xlabel('kx (×10⁹ m⁻¹)', fontsize=12)
    plt.ylabel('ky (×10⁹ m⁻¹)', fontsize=12)
    plt.title('グラフェンK点近傍のベリー曲率', fontsize=14)
    plt.tight_layout()
    plt.savefig('graphene_berry_curvature.png', dpi=150)
    
    print(f"ベリー曲率の積分（チャーン数）: {np.sum(berry_curv_graphene) * (kx_range[1]-kx_range[0]) * (ky_range[1]-ky_range[0]) / (2*np.pi):.3f}")

## 9\. フォノニック結晶とメタマテリアル

### 9.1 フォノニック結晶

周期的な構造により人工的にバンドギャップを作り出した材料です。トポロジカル設計により、頑強なエッジモードを実現できます。

**フォノニック結晶の利点:**

  * バンド構造の人工的制御が可能
  * トポロジカル相転移を幾何パラメータで調整
  * マクロスケールでの実証実験が容易
  * 振動絶縁、導波路、エネルギー収穫への応用


### 9.2 量子スピンホール型フォノニック結晶

時間反転対称性を保ちながら、擬スピン自由度によりZ2トポロジカル相を実現:

  * **二重ディラック錐:** 高対称点で縮退したディラック点
  * **対称性の低下:** 幾何変形により質量項が誘起され、ギャップが開く
  * **ヘリカルエッジモード:** 反対方向に伝播する対のモード


### 9.3 フロッケトポロジカルフォノン

時間周期的な駆動（例: 周期的変形、回転）により、実効的なトポロジカル相を誘起できます。

\[H_{\text{eff}} = H_0 + \sum_n H_n e^{in\Omega t}\] 

フロッケ理論により、時間平均された実効ハミルトニアンがトポロジカルな性質を持ちます。

### 9.4 メカニカルメタマテリアル

負のポアソン比、負の質量密度などの異常な機械的性質を示す人工材料です。トポロジカル設計により:

  * **ゼロエネルギーモード:** 内部変形なしの大変形モード
  * **エッジソフトニング:** エッジに局在した柔らかいモード
  * **一方向波動伝播:** カイラルエッジ波


graph TD A[フォノニック結晶/メタマテリアル] --> B[周期構造設計] A --> C[時間変調] B --> D[バンドギャップエンジニアリング] B --> E[トポロジカル相転移] C --> F[フロッケトポロジー] C --> G[動的調整] D --> H[エッジ導波路] E --> I[頑強な伝播] F --> J[非平衡トポロジー] G --> K[適応材料] 

## 10\. 実験的兆候と検出方法

### 10.1 中性子散乱

中性子非弾性散乱は、フォノン分散を直接測定できる最も強力な手法です。

  * **角度分解測定:** フォノンバンド構造全体をマッピング
  * **ワイル点の検出:** バンド交差点の直接観測
  * **表面状態:** 表面に敏感な測定によりフォノンアークを検出


**中性子散乱の特徴:**

  * エネルギー範囲: 0.1-100 meV（フォノンに最適）
  * 運動量分解能: ~0.01 Å⁻¹
  * 大型施設が必要（J-PARC, ISIS, SNSなど）
  * 単結晶試料が必要（通常 > 1 cm³）


### 10.2 ラマン散乱

光学フォノンのエネルギーと対称性を調べるのに適しています。

  * **対称性解析:** 既約表現の同定
  * **相転移の検出:** ピーク分裂や新規モードの出現
  * **応力・ひずみ効果:** トポロジカル相転移の誘起


### 10.3 熱輸送測定

トポロジカルフォノンの間接的な証拠を熱伝導から得られます。

測定量 | トポロジカルな兆候 | 典型的な温度範囲  
---|---|---  
熱ホール伝導率 κxy | ゼロ磁場で非ゼロ | 1-50 K  
熱伝導率 κxx | 異常な温度依存性 | 0.1-300 K  
熱電効果 | エッジ寄与の増強 | 室温付近  
磁場依存性 | 線形または飽和 | 低温  
  
### 10.4 第一原理計算

密度汎関数摂動論（DFPT）により、実材料のフォノン分散とベリー曲率を計算できます。

Python コピー
    
    
    # Quantum ESPRESSOでの計算フロー（疑似コード）
    
    """
    1. SCF計算（自己無撞着場）
       pw.x < scf.in > scf.out
    
    2. DFPT計算（動力学行列）
       ph.x < ph.in > ph.out
    
    3. フォノン分散の計算
       q2r.x < q2r.in > q2r.out
       matdyn.x < matdyn.in > matdyn.out
    
    4. ベリー曲率の計算（WannierTools等）
    """
    
    # Pythonでの解析例
    import numpy as np
    import matplotlib.pyplot as plt
    
    def load_phonon_bands(filename):
        """Quantum ESPRESSOの出力からフォノンバンドを読み込み"""
        data = np.loadtxt(filename)
        k_path = data[:, 0]
        frequencies = data[:, 1:]
        return k_path, frequencies
    
    def identify_weyl_points(k_mesh, frequencies, threshold=0.1):
        """
        バンド交差点（ワイル点候補）を検出
    
        Parameters:
        -----------
        k_mesh : ndarray
            k点メッシュ
        frequencies : ndarray
            フォノン周波数
        threshold : float
            バンドギャップの閾値 (THz)
    
        Returns:
        --------
        weyl_candidates : list
            ワイル点候補のk座標
        """
        weyl_candidates = []
    
        # 隣接バンド間のギャップをチェック
        for i in range(frequencies.shape[1] - 1):
            gap = frequencies[:, i+1] - frequencies[:, i]
            crossing_indices = np.where(gap < threshold)[0]
    
            for idx in crossing_indices:
                weyl_candidates.append(k_mesh[idx])
    
        return weyl_candidates
    
    # 使用例
    k_path, phonon_freqs = load_phonon_bands('phonon_bands.dat')
    
    # バンド構造のプロット
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for band_idx in range(phonon_freqs.shape[1]):
        ax.plot(k_path, phonon_freqs[:, band_idx], 'b-', linewidth=1.5)
    
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('波数ベクトル', fontsize=12)
    ax.set_ylabel('周波数 (THz)', fontsize=12)
    ax.set_title('フォノンバンド構造', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # ワイル点候補をマーク
    weyl_points = identify_weyl_points(k_path, phonon_freqs, threshold=0.1)
    if weyl_points:
        for wp in weyl_points:
            ax.axvline(x=wp, color='r', linestyle=':', alpha=0.5)
        print(f"ワイル点候補数: {len(weyl_points)}")
    
    plt.tight_layout()
    plt.savefig('phonon_bands_with_weyl.png', dpi=150)

### 10.5 走査トンネル顕微鏡（STM）

表面に敏感な手法で、エッジ状態の空間分布を可視化できます（電子-フォノン結合を介して間接的に）。

## 11\. 熱輸送特性との関連

### 11.1 ボルツマン輸送方程式

フォノンの熱輸送は、半古典的ボルツマン方程式で記述されます:

\[\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla_{\mathbf{r}} f + \mathbf{F} \cdot \nabla_{\mathbf{k}} f = \left(\frac{\partial f}{\partial t}\right)_{\text{coll}}\] 

トポロジカルフォノンでは、ベリー曲率により速度\\(\mathbf{v}\\)に異常項が加わります。

### 11.2 熱伝導率テンソル

熱伝導率は、フォノンの寄与の和として:

\[\kappa_{\alpha\beta} = \frac{1}{V} \sum_{\mathbf{k},n} C_n(\mathbf{k}) v_{n,\alpha}(\mathbf{k}) v_{n,\beta}(\mathbf{k}) \tau_n(\mathbf{k})\] 

ここで、

  * \\(C_n(\mathbf{k})\\): 比熱
  * \\(v_{n,\alpha}(\mathbf{k})\\): 群速度（異常速度を含む）
  * \\(\tau_n(\mathbf{k})\\): 緩和時間


### 11.3 トポロジカル効果による熱伝導の増強

トポロジカル保護されたエッジ状態は、散乱に強いため高い熱伝導率をもたらす可能性があります。

\[\kappa_{\text{edge}} \propto N_{\text{edge}} \times v_{\text{edge}} \times \ell_{\text{edge}}\] 

ここで、\\(N_{\text{edge}}\\)はエッジチャネル数、\\(\ell_{\text{edge}}\\)は平均自由行程（散乱に頑強なため長い）です。

### 11.4 量子化熱伝導

理想的な1次元エッジチャネルでは、熱伝導率が量子化される可能性があります:

\[\kappa_{\text{quantum}} = N_{\text{channel}} \times \frac{\pi^2 k_B^2 T}{3h}\] 

これは電子系のランダウアー公式の熱版に相当します。

### 11.5 計算例: 緩和時間近似

Python コピー
    
    
    import numpy as np
    from scipy.constants import hbar, k as kB
    
    def bose_einstein(omega, T):
        """ボース・アインシュタイン分布"""
        x = hbar * omega / (kB * T)
        return 1 / (np.exp(x) - 1)
    
    def heat_capacity_phonon(omega, T):
        """フォノンの比熱"""
        x = hbar * omega / (kB * T)
        n_BE = bose_einstein(omega, T)
        C = kB * x**2 * np.exp(x) / (np.exp(x) - 1)**2
        return C
    
    def thermal_conductivity_RTA(omega_bands, group_velocity, relaxation_time,
                                   k_mesh, temperature, volume):
        """
        緩和時間近似（RTA）での熱伝導率計算
    
        Parameters:
        -----------
        omega_bands : ndarray, shape (n_bands, n_k)
            フォノン角周波数 (rad/s)
        group_velocity : ndarray, shape (n_bands, n_k, 3)
            群速度 (m/s)
        relaxation_time : ndarray, shape (n_bands, n_k)
            緩和時間 (s)
        k_mesh : ndarray, shape (n_k, 3)
            k点メッシュ
        temperature : float
            温度 (K)
        volume : float
            単位胞体積 (m³)
    
        Returns:
        --------
        kappa : ndarray, shape (3, 3)
            熱伝導率テンソル (W/m·K)
        """
        n_bands, n_k = omega_bands.shape
        dk = (2 * np.pi)**3 / (volume * n_k)  # k空間の体積要素
    
        kappa = np.zeros((3, 3))
    
        for band_idx in range(n_bands):
            for k_idx in range(n_k):
                omega = omega_bands[band_idx, k_idx]
                v = group_velocity[band_idx, k_idx]
                tau = relaxation_time[band_idx, k_idx]
    
                if omega < 1e-6:  # 音響ブランチのゼロ点を除外
                    continue
    
                # 比熱
                C = heat_capacity_phonon(omega, temperature)
    
                # 熱伝導率テンソルへの寄与
                for alpha in range(3):
                    for beta in range(3):
                        kappa[alpha, beta] += C * v[alpha] * v[beta] * tau * dk / volume
    
        return kappa
    
    # 使用例: 簡単な等方的モデル
    n_k = 1000
    n_bands = 3
    
    # 簡易的なフォノン分散（デバイモデル）
    k_magnitude = np.linspace(0, np.pi / 3e-10, n_k)
    omega_debye = 3e13  # rad/s
    omega_bands_1d = np.zeros((n_bands, n_k))
    
    # 音響モード
    omega_bands_1d[0] = omega_debye * k_magnitude / k_magnitude[-1] * 0.3
    omega_bands_1d[1] = omega_debye * k_magnitude / k_magnitude[-1] * 0.5
    # 光学モード
    omega_bands_1d[2] = omega_debye * 0.8 * np.ones(n_k)
    
    # 群速度（簡易: 分散の微分）
    group_vel_1d = np.zeros((n_bands, n_k, 3))
    dk_spacing = k_magnitude[1] - k_magnitude[0]
    for band in range(n_bands):
        dw_dk = np.gradient(omega_bands_1d[band], dk_spacing)
        group_vel_1d[band, :, 0] = dw_dk  # x方向のみ
    
    # 緩和時間（簡易: ウムクラッププロセス）
    T_ref = 300  # K
    tau_0 = 1e-11  # s
    relaxation_time_1d = tau_0 * (T_ref / 100)**2 * np.ones((n_bands, n_k))
    
    # 温度依存性の計算
    temperatures = np.linspace(10, 400, 40)
    kappa_values = []
    
    volume = (3e-10)**3  # 典型的な単位胞体積
    
    for T in temperatures:
        kappa_tensor = thermal_conductivity_RTA(
            omega_bands_1d, group_vel_1d, relaxation_time_1d,
            k_magnitude, T, volume
        )
        kappa_values.append(kappa_tensor[0, 0])
    
    # プロット
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.plot(temperatures, kappa_values, 'o-', linewidth=2)
    plt.xlabel('温度 (K)', fontsize=12)
    plt.ylabel('熱伝導率 (W/m·K)', fontsize=12)
    plt.title('熱伝導率の温度依存性（RTA）', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('thermal_conductivity_RTA.png', dpi=150)
    
    print(f"室温（300K）での熱伝導率: {kappa_values[np.argmin(np.abs(temperatures - 300))]:.2f} W/m·K")

## 12\. 計算手法とPython実装

### 12.1 動力学行列の対角化

フォノンバンド構造を計算するには、各k点で動力学行列を対角化します:

\[D_{\alpha\beta}^{ij}(\mathbf{k}) = \frac{1}{\sqrt{M_i M_j}} \sum_{\mathbf{R}} \Phi_{\alpha\beta}^{ij}(0, \mathbf{R}) e^{i\mathbf{k} \cdot \mathbf{R}}\] 

ここで、\\(\Phi\\)は力定数行列、\\(M_i\\)は原子\\(i\\)の質量です。

### 12.2 ウィルソンループ法

Z2不変量やチャーン数を数値的に計算する方法です。ブリルアンゾーン上の閉路でのベリー接続の積分を計算します。

Python コピー
    
    
    import numpy as np
    from scipy.linalg import eigh, det
    
    def wilson_loop_1d(dynamical_matrix_func, k_loop, occupied_bands):
        """
        1次元ウィルソンループの計算
    
        Parameters:
        -----------
        dynamical_matrix_func : callable
            k点を引数にとり動力学行列を返す関数
        k_loop : ndarray, shape (n_k, 2)
            ブリルアンゾーン上の閉路
        occupied_bands : list
            占有バンドのインデックスリスト
    
        Returns:
        --------
        wilson_eigenvalues : ndarray
            ウィルソンループの固有値（位相）
        """
        n_k = len(k_loop)
        n_occ = len(occupied_bands)
    
        # 初期化
        F = np.eye(n_occ, dtype=complex)
    
        # 閉路に沿って積分
        for i in range(n_k):
            k_current = k_loop[i]
            k_next = k_loop[(i + 1) % n_k]
    
            # 現在と次のk点での固有ベクトル
            D_current = dynamical_matrix_func(k_current[0], k_current[1])
            D_next = dynamical_matrix_func(k_next[0], k_next[1])
    
            _, v_current = eigh(D_current)
            _, v_next = eigh(D_next)
    
            # 占有バンドの部分空間
            U_current = v_current[:, occupied_bands]
            U_next = v_next[:, occupied_bands]
    
            # 重なり行列
            overlap = U_next.conj().T @ U_current
    
            # ウィルソンループ行列の更新
            F = overlap @ F
    
        # 固有値（位相）を計算
        wilson_eigenvalues = np.linalg.eigvals(F)
        wilson_phases = np.angle(wilson_eigenvalues)
    
        return wilson_phases
    
    def compute_chern_number_wilson(dynamical_matrix_func, k_mesh_2d, occupied_bands):
        """
        2次元系のチャーン数をウィルソンループ法で計算
    
        Parameters:
        -----------
        dynamical_matrix_func : callable
            動力学行列を返す関数
        k_mesh_2d : tuple (kx_array, ky_array)
            2次元k点メッシュ
        occupied_bands : list
            占有バンドのインデックス
    
        Returns:
        --------
        chern_number : float
            チャーン数
        """
        kx_array, ky_array = k_mesh_2d
        n_kx = len(kx_array)
        n_ky = len(ky_array)
    
        # ky方向のウィルソンループ固有値を計算
        wilson_phases = np.zeros((n_kx, len(occupied_bands)))
    
        for i, kx in enumerate(kx_array):
            # kyループを構築
            k_loop = np.array([[kx, ky] for ky in ky_array])
            # ウィルソンループ計算
            phases = wilson_loop_1d(dynamical_matrix_func, k_loop, occupied_bands)
            wilson_phases[i] = phases
    
        # kx方向でウィルソンループ固有値の巻き数を計算
        total_winding = 0
        for band_idx in range(len(occupied_bands)):
            phase_evolution = wilson_phases[:, band_idx]
            # 位相の変化を追跡（アンラップ）
            phase_unwrapped = np.unwrap(phase_evolution)
            winding = (phase_unwrapped[-1] - phase_unwrapped[0]) / (2 * np.pi)
            total_winding += winding
    
        chern_number = np.round(total_winding)
        return chern_number
    
    # 使用例
    def example_haldane_phonon(kx, ky, t1=1.0, t2=0.3, phi=np.pi/2):
        """Haldane模型風のトポロジカルフォノン"""
        # 簡略化されたハルデン型モデル
        H = np.zeros((6, 6), dtype=complex)
    
        # 最近接ホッピング
        delta_1 = [1, 0]
        delta_2 = [-0.5, np.sqrt(3)/2]
        delta_3 = [-0.5, -np.sqrt(3)/2]
    
        phase_1 = np.exp(1j * (kx * delta_1[0] + ky * delta_1[1]))
        phase_2 = np.exp(1j * (kx * delta_2[0] + ky * delta_2[1]))
        phase_3 = np.exp(1j * (kx * delta_3[0] + ky * delta_3[1]))
    
        # オンサイト
        H[0:3, 0:3] = 3 * t1 * np.eye(3)
        H[3:6, 3:6] = 3 * t1 * np.eye(3)
    
        # サブラティス間
        coupling = t1 * (phase_1 + phase_2 + phase_3)
        H[0:3, 3:6] = -coupling * np.eye(3)
        H[3:6, 0:3] = -np.conj(coupling) * np.eye(3)
    
        # 次近接ホッピング（時間反転対称性を破る）
        nnn_phase = np.exp(1j * phi)
        H[0:3, 0:3] += t2 * nnn_phase * np.eye(3)
        H[3:6, 3:6] += t2 * np.conj(nnn_phase) * np.eye(3)
    
        return H
    
    # チャーン数の計算
    kx_mesh = np.linspace(-np.pi, np.pi, 20)
    ky_mesh = np.linspace(-np.pi, np.pi, 20)
    
    chern_num = compute_chern_number_wilson(
        example_haldane_phonon,
        (kx_mesh, ky_mesh),
        occupied_bands=[0, 1, 2]
    )
    
    print(f"計算されたチャーン数: {chern_num:.0f}")

### 12.3 WannierToolsの利用

WannierToolsは、トポロジカル物性の計算に特化したツールです。第一原理計算の結果からベリー曲率、チャーン数、表面状態を計算できます。

**WannierToolsの主な機能:**

  * ベリー曲率とチャーン数の計算
  * Z2不変量の計算
  * 表面グリーン関数法による表面状態の計算
  * フェルミアーク、フォノンアークの可視化
  * 異常ホール伝導度、熱ホール伝導度の計算


## 13\. 実例: α-石英とNa3Bi

### 13.1 α-石英（α-SiO2）

α-石英は、最初に発見されたトポロジカルフォノンを持つ材料の一つです。

#### 構造的特徴

  * 空間群: P3121（非中心対称、らせん対称性）
  * 六方晶系、a = 4.916 Å, c = 5.405 Å
  * Si-O四面体がらせん状につながる構造


#### トポロジカル性質

  * **ワイルフォノン点:** ブリルアンゾーンのH-K線上に存在
  * **カイラリティ:** \\(\chi = \pm 1\\)のワイル点対
  * **フォノンアーク:** (0001)表面で観測
  * **周波数範囲:** 約10-15 THz


**実験的証拠:**

  * 中性子散乱によるフォノン分散の測定（Li et al., Science 2018）
  * 第一原理計算によるベリー曲率の確認
  * ワイル点近傍での線形分散の観測


### 13.2 Na3Bi

Na3Biは、ディラック半金属として知られていますが、フォノンもトポロジカルな性質を示します。

#### 構造的特徴

  * 空間群: P63/mmc（六方晶）
  * 層状構造: Na-Bi-Na積層
  * 空間反転対称性と時間反転対称性を両方持つ


#### トポロジカル性質

  * **ディラックフォノン:** Γ点とA点にディラック錐
  * **Z2不変量:** ν0 = 1（非自明なトポロジカル絶縁体相）
  * **ヘリカルエッジ状態:** (001)表面で予測
  * **電子-フォノン結合:** トポロジカル電子状態との相互作用


graph TD A[Na3Bi] --> B[電子系] A --> C[フォノン系] B --> D[ディラック半金属] B --> E[表面フェルミアーク] C --> F[ディラックフォノン] C --> G[Z2トポロジカル] D -.電子-フォノン結合.- F E -.表面状態相互作用.- G style D fill:#ffe6e6 style F fill:#e6f3ff 

### 13.3 その他の実例

材料 | 空間群 | トポロジカル特徴 | 実験的検証  
---|---|---|---  
α-SiO2 | P3121 | ワイル点、フォノンアーク | 中性子散乱（2018）  
Na3Bi | P63/mmc | ディラック点、Z2 = 1 | 第一原理計算（2019）  
FeSi | P213 | 多重ワイル点 | 中性子散乱（2020）  
ScZn | Pm-3m | ノードライン、ドラムヘッド表面状態 | 理論予測（2021）  
Bi2Se3 | R-3m | Z2トポロジカル絶縁体 | ラマン散乱（2020）  
  
### 13.4 計算例: α-石英のワイル点解析

Python コピー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def analyze_weyl_point_structure(k_weyl, omega_func, berry_curv_func,
                                       delta_k=0.05, n_points=20):
        """
        ワイル点近傍の構造解析
    
        Parameters:
        -----------
        k_weyl : array-like
            ワイル点のk座標 [kx, ky, kz]
        omega_func : callable
            フォノン周波数を返す関数
        berry_curv_func : callable
            ベリー曲率を返す関数
        delta_k : float
            ワイル点からの距離範囲
        n_points : int
            各方向のサンプル点数
    
        Returns:
        --------
        analysis : dict
            解析結果（線形分散、カイラリティ等）
        """
        kx_w, ky_w, kz_w = k_weyl
    
        # ワイル点周りのメッシュ
        kx_range = np.linspace(kx_w - delta_k, kx_w + delta_k, n_points)
        ky_range = np.linspace(ky_w - delta_k, ky_w + delta_k, n_points)
    
        # 線形分散の検証（kz固定、kx-ky平面）
        omega_map = np.zeros((n_points, n_points))
        berry_flux = 0
    
        for i, kx in enumerate(kx_range):
            for j, ky in enumerate(ky_range):
                omega_map[i, j] = omega_func(kx, ky, kz_w)
    
                # ベリー曲率の積分（カイラリティ）
                if i > 0 and j > 0:
                    dkx = kx_range[1] - kx_range[0]
                    dky = ky_range[1] - ky_range[0]
                    berry_curv = berry_curv_func(kx, ky, kz_w)
                    berry_flux += berry_curv * dkx * dky
    
        # カイラリティの計算
        chirality = np.round(berry_flux / (2 * np.pi))
    
        # 線形フィット（ワイル点からの距離 vs エネルギー）
        distances = []
        energies = []
    
        for i, kx in enumerate(kx_range):
            for j, ky in enumerate(ky_range):
                dist = np.sqrt((kx - kx_w)**2 + (ky - ky_w)**2)
                energy = omega_map[i, j]
                if dist > 0:
                    distances.append(dist)
                    energies.append(energy)
    
        # 線形フィット
        coeffs = np.polyfit(distances, energies, 1)
        velocity = coeffs[0]  # フォノン速度
    
        analysis = {
            'chirality': chirality,
            'velocity': velocity,
            'omega_map': omega_map,
            'k_mesh': (kx_range, ky_range)
        }
    
        return analysis
    
    # α-石英の簡易モデル（実際のDFPT結果の近似）
    def quartz_omega_near_weyl(kx, ky, kz):
        """ワイル点近傍のフォノン周波数（THz）"""
        # H点近傍のワイル点: k_weyl ≈ (1/3, 1/3, 1/2) in reduced coordinates
        k_weyl = np.array([2*np.pi/(3*4.916e-10), 2*np.pi/(3*4.916e-10), np.pi/5.405e-10])
    
        delta_k = np.array([kx, ky, kz]) - k_weyl
    
        # 線形分散（簡易）
        v_ph = 5000  # m/s
        omega = v_ph * np.linalg.norm(delta_k) / (2 * np.pi)  # Hz
        omega_THz = omega / 1e12
    
        return omega_THz + 12.5  # 12.5 THz付近にワイル点
    
    def quartz_berry_curvature(kx, ky, kz):
        """ベリー曲率のモデル（簡易）"""
        k_weyl = np.array([2*np.pi/(3*4.916e-10), 2*np.pi/(3*4.916e-10), np.pi/5.405e-10])
        delta_k = np.array([kx, ky, kz]) - k_weyl
    
        # ワイル点周りの1/r²依存性
        r = np.linalg.norm(delta_k[:2])  # kx-ky平面内
        if r < 1e8:
            return 0
    
        # モノポール構造
        Omega = 1e-18 / r**2  # 簡易的な形
        return Omega
    
    # ワイル点解析
    k_weyl_quartz = [2*np.pi/(3*4.916e-10), 2*np.pi/(3*4.916e-10), np.pi/5.405e-10]
    
    analysis = analyze_weyl_point_structure(
        k_weyl_quartz,
        quartz_omega_near_weyl,
        quartz_berry_curvature,
        delta_k=5e8,  # 1/m
        n_points=30
    )
    
    print(f"カイラリティ: {analysis['chirality']:.0f}")
    print(f"フォノン速度: {analysis['velocity']:.2e} m/s")
    
    # 分散の可視化
    kx_mesh, ky_mesh = analysis['k_mesh']
    omega_map = analysis['omega_map']
    
    fig = plt.figure(figsize=(12, 5))
    
    # 2Dマップ
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(kx_mesh / 1e9, ky_mesh / 1e9, omega_map.T,
                            levels=20, cmap='viridis')
    ax1.plot([k_weyl_quartz[0] / 1e9], [k_weyl_quartz[1] / 1e9],
             'r*', markersize=15, label='ワイル点')
    ax1.set_xlabel('kx (×10⁹ m⁻¹)')
    ax1.set_ylabel('ky (×10⁹ m⁻¹)')
    ax1.set_title('ワイル点近傍のフォノン分散')
    plt.colorbar(contour, ax=ax1, label='周波数 (THz)')
    ax1.legend()
    
    # 3D表示
    ax2 = fig.add_subplot(122, projection='3d')
    KX, KY = np.meshgrid(kx_mesh / 1e9, ky_mesh / 1e9)
    surf = ax2.plot_surface(KX, KY, omega_map.T, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('kx (×10⁹ m⁻¹)')
    ax2.set_ylabel('ky (×10⁹ m⁻¹)')
    ax2.set_zlabel('周波数 (THz)')
    ax2.set_title('3D分散（ワイル錐）')
    
    plt.tight_layout()
    plt.savefig('quartz_weyl_phonon.png', dpi=150)
    print("分散図を保存しました: quartz_weyl_phonon.png")

## 14\. まとめ

本章では、トポロジカルフォノンの理論と応用について学びました。

### 重要なポイント

#### 理論的基礎

  * **ベリー位相:** 波動関数の幾何学的位相がフォノンの異常な運動を引き起こす
  * **ベリー曲率:** 運動量空間の「磁場」として作用し、横方向の運動を誘起
  * **トポロジカル不変量:** チャーン数、Z2不変量がバンド構造の位相的性質を特徴づける
  * **バルク-エッジ対応:** バルクのトポロジーが表面・エッジ状態の存在を保証


#### トポロジカル相の種類

  * **チャーン絶縁体型:** 時間反転対称性が破れ、カイラルエッジモードが現れる
  * **Z2トポロジカル絶縁体型:** 時間反転対称性が保たれ、ヘリカルエッジモードが現れる
  * **ワイル・ディラック半金属型:** バンド交差点（ノード）とフォノンアークが特徴


#### 物理現象

  * **フォノンホール効果:** 温度勾配に垂直な熱流（ベリー曲率由来）
  * **トポロジカル保護:** エッジ状態の散乱に対する頑強性
  * **異常速度:** 外力と垂直な運動成分
  * **熱輸送の増強:** トポロジカルエッジチャネルによる高効率熱伝導


#### 実材料と応用

  * **α-石英:** ワイルフォノンの実証例、中性子散乱で確認
  * **Na3Bi:** ディラックフォノン、電子系との相互作用
  * **2D材料:** グラフェン、TMDでの谷依存的トポロジー
  * **フォノニック結晶:** 人工的なトポロジカル設計と制御


#### 計算手法

  * **第一原理計算:** DFPT によるフォノン分散とベリー曲率
  * **ウィルソンループ法:** トポロジカル不変量の数値計算
  * **緩和時間近似:** 熱伝導率の計算
  * **WannierTools:** トポロジカル物性の包括的解析


### 今後の展望

  * **新材料探索:** 高スループット計算とデータベースによるトポロジカルフォノン材料の発見
  * **実験技術の進展:** 高分解能中性子散乱、時間分解ラマン分光
  * **デバイス応用:** トポロジカル熱ダイオード、フォノニック導波路
  * **量子効果:** 低温でのフォノンホール効果の量子化
  * **非平衡トポロジー:** フロッケ状態、駆動系でのトポロジカル相
  * **多体効果:** フォノン-フォノン相互作用の影響


### 関連分野とのつながり

graph LR A[トポロジカルフォノン] --> B[トポロジカル絶縁体] A --> C[熱電材料] A --> D[フォノニック結晶] A --> E[スピントロニクス] B --> F[バンドトポロジー理論] C --> G[熱輸送制御] D --> H[メタマテリアル設計] E --> I[スピン-格子結合] F --> J[材料探索] G --> J H --> J I --> J style A fill:#e6f3ff style J fill:#ffe6e6 

## 演習問題

### 演習 4.1: ベリー曲率の計算

**問題:** 2バンド系のハミルトニアン \\(H(\mathbf{k}) = d_x(\mathbf{k})\sigma_x + d_y(\mathbf{k})\sigma_y + d_z(\mathbf{k})\sigma_z\\) に対して、下側バンドのベリー曲率を導出せよ。ただし、\\(\mathbf{d}(\mathbf{k}) = (d_x, d_y, d_z)\\)とする。

ヒント

ベリー曲率は \\(\Omega_{xy} = \mathbf{d} \cdot (\partial_{k_x}\mathbf{d} \times \partial_{k_y}\mathbf{d}) / (2|\mathbf{d}|^3)\\) で与えられます。

### 演習 4.2: チャーン数の計算

**問題:** 提供されたPythonコードを用いて、ハルデン模型のパラメータ\\(\phi\\)を0から\\(2\pi\\)まで変化させたときのチャーン数の変化を計算し、プロットせよ。位相図を描け。

### 演習 4.3: エッジ状態の可視化

**問題:** SSH模型（1次元鎖）において、ホッピングパラメータの比\\(t_1/t_2\\)をパラメータとして、エッジ状態の数とエネルギーの関係を調べよ。どの値でトポロジカル相転移が起こるか?

### 演習 4.4: 熱ホール伝導率の温度依存性

**問題:** 熱因子\\(c_2(x)\\)の関数形から、熱ホール伝導率\\(\kappa_{xy}\\)が低温でどのような温度依存性を示すか理論的に導出せよ。高温極限についても議論せよ。

ヒント

低温(\\(T \to 0\\))では\\(x = \hbar\omega/(k_B T) \to \infty\\)となり、\\(c_2(x) \sim x^2 e^{-x}\\)と振る舞います。高温では\\(x \to 0\\)、\\(c_2(x) \sim 1\\)となります。

### 演習 4.5: ワイル点の探索

**問題:** グラフェンのフォノンバンド構造において、K点とK'点近傍でディラック点を探索するPythonコードを作成せよ。線形分散を確認し、フォノン速度を計算せよ。

### 演習 4.6: フォノニック結晶の設計

**問題:** 1次元フォノニック結晶（質量\\(m_1, m_2\\)とばね定数\\(K_1, K_2\\)が交互に配列）のバンド構造を計算せよ。バンドギャップが開く条件を導出し、トポロジカル相が現れるパラメータ領域を同定せよ。

### 演習 4.7: Z2不変量の理解

**問題:** 3次元系において、8つのTRIM点での偏極\\(P_i\\)から\\(Z_2\\)不変量\\(\nu_0; (\nu_1\nu_2\nu_3)\\)を計算する手順を説明せよ。各不変量の物理的意味を述べよ。

### 演習 4.8: 実験データ解析

**問題:** α-石英の中性子散乱データ（仮想データ）からワイル点の位置を同定し、そのカイラリティを推定する解析プログラムを作成せよ。

## 参考文献

  1. Zhang, T., et al. "Double-Weyl Phonons in Transition-Metal Monosilicides." _Physical Review Letters_ 120, 016401 (2018).
  2. Li, J., et al. "Computation and data driven discovery of topological phononic materials." _Nature Communications_ 12, 1204 (2021).
  3. Liu, Y., et al. "Floquet topological phases with high Chern numbers in periodically driven systems." _Physical Review B_ 101, 174301 (2020).
  4. Murakami, S., et al. "Phonon Hall Effect." _Physical Review Letters_ 101, 035301 (2008).
  5. Xia, B. W., et al. "Observation of chiral phonons." _Science_ 359, 579-582 (2018).
  6. Süsstrunk, R., and Huber, S. D. "Observation of phononic helical edge states in a mechanical topological insulator." _Science_ 349, 47-50 (2015).
  7. Chen, H., et al. "Topological phononic insulator with robust pseudospin-dependent transport." _Physical Review B_ 96, 020202(R) (2017).
  8. Strohm, C., et al. "Phonon Hall viscosity from gauge invariance." _Physical Review B_ 95, 024302 (2017).
  9. Qian, K., et al. "Observation of phonon Hall effect." _Science Bulletin_ 68, 783-787 (2023).
  10. Armitage, N. P., et al. "Weyl and Dirac semimetals in three-dimensional solids." _Reviews of Modern Physics_ 90, 015001 (2018).


### 推奨テキスト

  * Bernevig, B. A., and Hughes, T. L. _Topological Insulators and Topological Superconductors_ (Princeton University Press, 2013).
  * Shen, S.-Q. _Topological Insulators: Dirac Equation in Condensed Matter_ , 2nd ed. (Springer, 2017).
  * Vanderbilt, D. _Berry Phases in Electronic Structure Theory_ (Cambridge University Press, 2018).


### オンラインリソース

  * [WannierTools 公式サイト](http://www.wanniertools.org/) \- トポロジカル物性計算ツール
  * [Quantum ESPRESSO](https://www.quantum-espresso.org/) \- 第一原理計算パッケージ
  * [Phonopy](https://phonopy.github.io/phonopy/) \- フォノン計算ツール
  * [Materials Project](https://materialsproject.org/) \- 材料データベース

---

**ナビゲーション**

[← 第3章](chapter-3.md) | [目次](index.md) | [第5章 →](chapter-5.md)

---

## 免責事項

このコンテンツはAIの支援を受けて作成されており、正確性を保証するものではありません。重要な情報については一次資料や査読済み文献で確認することをお勧めします。
