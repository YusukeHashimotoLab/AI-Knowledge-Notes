---
title: "第5章: X線光電子分光法（XPS）"
chapter_title: "第5章: X線光電子分光法（XPS）"
---

## ビデオ講義

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/so3riEt-4Ew"
    title="分光分析入門 第5章: X線光電子分光法（XPS）"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> このビデオは以下のテキストと同じ内容をカバーしています。お好みの学習形式をお選びください。

JP | [EN](<../../../en/MS/spectroscopy-introduction/chapter-5.html>) | Last sync: 2025-12-26

[AI寺子屋 Top](<../../index.html>)>[材料科学](<../index.html>)>[分光分析入門](<index.html>)>第5章

# 第5章: X線光電子分光法（XPS）

**この章で学ぶこと:** X線光電子分光法（XPS: X-ray Photoelectron Spectroscopy）は、材料表面の元素組成と化学状態を高感度で分析する表面分析手法です。光電効果の原理に基づき、X線照射により放出される光電子の運動エネルギーを測定することで、元素同定、酸化状態の決定、定量分析を実現します。本章では、XPSの物理的原理からVoigt関数によるピークフィッティング、感度係数を用いた定量分析、深さ方向分析まで、Pythonによる実践的なデータ解析手法を学びます。

## 5.1 光電効果と結合エネルギー

### 5.1.1 光電効果の基本原理

XPSは、Albert Einsteinが1905年に理論化した光電効果に基づく分析手法です。試料表面にX線を照射すると、X線光子のエネルギーが原子の内殻電子に吸収され、電子が真空中に放出されます。この放出された電子を光電子（photoelectron）と呼びます。

**アインシュタインの光電効果の式:**

\\[ E_K = h\nu - E_B - \phi \\] 

ここで:

  * \\( E_K \\): 光電子の運動エネルギー（Kinetic Energy）
  * \\( h\nu \\): 入射X線の光子エネルギー
  * \\( E_B \\): 電子の結合エネルギー（Binding Energy）
  * \\( \phi \\): 分光器の仕事関数

**結合エネルギーの算出:**

\\[ E_B = h\nu - E_K - \phi \\] 

結合エネルギーは元素固有の値を持ち、さらに化学状態によって数eVの範囲でシフトします。これが化学シフト（chemical shift）であり、XPSによる化学状態分析の基礎となります。

### 5.1.2 結合エネルギー計算のPython実装

以下のコードでは、運動エネルギーから結合エネルギーを計算し、XPSの基本原理を可視化します。
    
    
    # コード例1: 光電効果と結合エネルギー計算
    import numpy as np
    import matplotlib.pyplot as plt
    
    # X線源のエネルギー（eV）
    X_RAY_SOURCES = {
        'Al_Ka': 1486.6,      # Al Kalpha（単色化）
        'Mg_Ka': 1253.6,      # Mg Kalpha
        'Ag_La': 2984.3,      # Ag Lalpha
    }
    
    # 典型的な仕事関数（分光器により異なる）
    WORK_FUNCTION = 4.5  # eV
    
    def calculate_binding_energy(kinetic_energy, photon_energy, work_function=WORK_FUNCTION):
        """
        運動エネルギーから結合エネルギーを計算
    
        Parameters
        ----------
        kinetic_energy : float or array
            光電子の運動エネルギー (eV)
        photon_energy : float
            X線光子エネルギー (eV)
        work_function : float
            分光器の仕事関数 (eV)
    
        Returns
        -------
        binding_energy : float or array
            結合エネルギー (eV)
        """
        return photon_energy - kinetic_energy - work_function
    
    def calculate_kinetic_energy(binding_energy, photon_energy, work_function=WORK_FUNCTION):
        """
        結合エネルギーから運動エネルギーを計算
        """
        return photon_energy - binding_energy - work_function
    
    # 主要元素の結合エネルギー（文献値）
    BINDING_ENERGIES = {
        'C_1s': 284.8,   # C-C/C-H（炭素汚染の標準）
        'O_1s': 531.0,   # 金属酸化物
        'N_1s': 399.0,   # 有機窒素
        'Si_2p': 99.3,   # Si元素
        'Si_2p_SiO2': 103.5,  # SiO2
        'Fe_2p3/2': 707.0,    # Fe金属
        'Fe_2p3/2_Fe2O3': 710.8,  # Fe2O3
        'Au_4f7/2': 84.0,     # Au金属（標準）
    }
    
    # Al Ka線での運動エネルギー計算
    photon_energy = X_RAY_SOURCES['Al_Ka']
    print("Al Ka線（1486.6 eV）での各元素の運動エネルギー:")
    print("-" * 50)
    for element, be in BINDING_ENERGIES.items():
        ke = calculate_kinetic_energy(be, photon_energy)
        print(f"{element:15s}: BE = {be:6.1f} eV, KE = {ke:7.1f} eV")
    
    # エネルギーダイアグラムの可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左: エネルギー準位図
    ax1 = axes[0]
    elements = ['C 1s', 'O 1s', 'Si 2p\n(Si)', 'Si 2p\n(SiO2)', 'Fe 2p3/2\n(Fe)', 'Au 4f7/2']
    be_values = [284.8, 531.0, 99.3, 103.5, 707.0, 84.0]
    ke_values = [calculate_kinetic_energy(be, photon_energy) for be in be_values]
    
    y_pos = np.arange(len(elements))
    ax1.barh(y_pos, be_values, height=0.4, label='結合エネルギー', color='#f093fb', alpha=0.8)
    ax1.barh(y_pos + 0.4, ke_values, height=0.4, label='運動エネルギー', color='#f5576c', alpha=0.8)
    ax1.set_yticks(y_pos + 0.2)
    ax1.set_yticklabels(elements)
    ax1.set_xlabel('エネルギー (eV)')
    ax1.set_title('光電効果: 結合エネルギーと運動エネルギー')
    ax1.legend(loc='upper right')
    ax1.grid(axis='x', alpha=0.3)
    
    # 右: 光電効果の模式図
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    
    # 原子核
    circle = plt.Circle((3, 5), 0.5, color='#2c3e50', label='原子核')
    ax2.add_patch(circle)
    
    # 電子軌道
    for r, label in [(1.5, '1s'), (2.5, '2s'), (3.5, '2p')]:
        orbit = plt.Circle((3, 5), r, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax2.add_patch(orbit)
        ax2.annotate(label, (3 + r, 5), fontsize=8, color='gray')
    
    # 電子（1s軌道）
    electron = plt.Circle((3 + 1.5, 5), 0.15, color='#f5576c', label='内殻電子')
    ax2.add_patch(electron)
    
    # X線入射
    ax2.annotate('', xy=(3, 5), xytext=(-1, 5),
                arrowprops=dict(arrowstyle='->', color='#f093fb', lw=2))
    ax2.text(-0.5, 5.5, r'X線 ($h\nu$)', fontsize=10, color='#f093fb')
    
    # 光電子放出
    ax2.annotate('', xy=(8, 7), xytext=(4.5, 5),
                arrowprops=dict(arrowstyle='->', color='#f5576c', lw=2))
    ax2.text(6.5, 7.5, r'光電子 ($E_K$)', fontsize=10, color='#f5576c')
    
    # 式の表示
    ax2.text(5, 2, r'$E_K = h\nu - E_B - \phi$', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_title('光電効果の模式図')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('xps_photoelectric_effect.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n図を 'xps_photoelectric_effect.png' に保存しました")
    

## 5.2 XPS装置の構成

### 5.2.1 X線源

XPSで使用される代表的なX線源とその特性を以下に示します。単色化X線源を使用することで、エネルギー分解能が向上し、より精密な化学状態分析が可能になります。

X線源 | 光子エネルギー (eV) | 固有線幅 (eV) | 特徴  
---|---|---|---  
Mg Ka | 1253.6 | 0.70 | 非単色化、汎用  
Al Ka | 1486.6 | 0.85（非単色化） | 最も一般的  
Al Ka（単色化） | 1486.6 | 0.25-0.30 | 高分解能測定用  
Ag La | 2984.3 | 2.6 | 深い内殻準位用  
  
### 5.2.2 半球型エネルギー分析器

XPSでは、半球型静電分析器（Hemispherical Analyzer）が広く使用されています。光電子は入口スリットから入射し、内外球間の電界により偏向され、特定のエネルギーを持つ電子のみが検出器に到達します。
    
    
    ```mermaid
    flowchart LR
            A[X線源<br/>Al Ka: 1486.6 eV] --> B[試料<br/>表面]
            B --> C[電子レンズ系<br/>集束・減速]
            C --> D[半球型分析器<br/>エネルギー選別]
            D --> E[検出器<br/>マルチチャンネル]
            E --> F[XPSスペクトル<br/>BE vs. 強度]
    
            style A fill:#e3f2fd
            style B fill:#fff3e0
            style C fill:#fce4ec
            style D fill:#e8f5e9
            style E fill:#f3e5f5
            style F fill:#ffe0b2
        
    
    分析器のパラメータ
    
    パスエネルギー (PE): 分析器を通過する電子のエネルギー。低いPEほど高分解能だが感度が低下
    ワイドスキャン: PE = 100-200 eV（サーベイスペクトル、元素同定用）
    ナロースキャン: PE = 10-50 eV（高分解能スペクトル、化学状態分析用）
    分解能: 通常 0.5-1.0 eV（単色化X線源 + 低PE）
    
    
    # コード例2: XPSスペクトルのシミュレーション
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import voigt_profile
    
    def gaussian(x, center, sigma, amplitude):
        """ガウス関数"""
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    def lorentzian(x, center, gamma, amplitude):
        """ローレンツ関数"""
        return amplitude * gamma**2 / ((x - center)**2 + gamma**2)
    
    def voigt(x, center, sigma, gamma, amplitude):
        """
        Voigt関数（ガウス関数とローレンツ関数の畳み込み）
        XPSピークのフィッティングに使用
        """
        # scipy.special.voigt_profile を使用
        # sigma: ガウス成分の標準偏差
        # gamma: ローレンツ成分の半値半幅
        return amplitude * voigt_profile(x - center, sigma, gamma)
    
    def simulate_xps_spectrum(binding_energies, peaks_info, noise_level=0.02):
        """
        XPSスペクトルをシミュレーション
    
        Parameters
        ----------
        binding_energies : array
            結合エネルギーの配列 (eV)
        peaks_info : list of dict
            各ピークの情報 {center, sigma, gamma, amplitude, label}
        noise_level : float
            ノイズレベル（最大強度に対する割合）
    
        Returns
        -------
        intensity : array
            スペクトル強度
        """
        intensity = np.zeros_like(binding_energies)
    
        for peak in peaks_info:
            intensity += voigt(
                binding_energies,
                peak['center'],
                peak['sigma'],
                peak['gamma'],
                peak['amplitude']
            )
    
        # ノイズ追加
        max_intensity = np.max(intensity)
        noise = np.random.normal(0, noise_level * max_intensity, len(binding_energies))
        intensity += noise
    
        return np.maximum(intensity, 0)  # 負の値を防止
    
    # C 1sスペクトルのシミュレーション（複数の化学状態）
    be_range = np.linspace(280, 295, 500)
    
    c1s_peaks = [
        {'center': 284.8, 'sigma': 0.4, 'gamma': 0.2, 'amplitude': 1.0, 'label': 'C-C/C-H'},
        {'center': 286.3, 'sigma': 0.4, 'gamma': 0.2, 'amplitude': 0.3, 'label': 'C-O'},
        {'center': 287.8, 'sigma': 0.4, 'gamma': 0.2, 'amplitude': 0.15, 'label': 'C=O'},
        {'center': 289.0, 'sigma': 0.4, 'gamma': 0.2, 'amplitude': 0.1, 'label': 'O-C=O'},
    ]
    
    c1s_spectrum = simulate_xps_spectrum(be_range, c1s_peaks, noise_level=0.01)
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左: ワイドスキャン（サーベイスペクトル）
    ax1 = axes[0]
    wide_be = np.linspace(0, 1200, 2000)
    
    # 主要元素のピークを追加
    survey_peaks = [
        {'center': 284.8, 'sigma': 2, 'gamma': 1, 'amplitude': 0.8, 'label': 'C 1s'},
        {'center': 531.0, 'sigma': 2, 'gamma': 1, 'amplitude': 1.0, 'label': 'O 1s'},
        {'center': 399.0, 'sigma': 2, 'gamma': 1, 'amplitude': 0.3, 'label': 'N 1s'},
        {'center': 103.0, 'sigma': 2, 'gamma': 1, 'amplitude': 0.5, 'label': 'Si 2p'},
        {'center': 153.0, 'sigma': 2, 'gamma': 1, 'amplitude': 0.4, 'label': 'Si 2s'},
        {'center': 711.0, 'sigma': 3, 'gamma': 2, 'amplitude': 0.4, 'label': 'Fe 2p'},
    ]
    
    survey_spectrum = simulate_xps_spectrum(wide_be, survey_peaks, noise_level=0.03)
    ax1.plot(wide_be, survey_spectrum, 'k-', linewidth=0.8)
    ax1.set_xlim(1200, 0)  # XPSは高結合エネルギーが左
    ax1.set_xlabel('結合エネルギー (eV)')
    ax1.set_ylabel('強度 (任意単位)')
    ax1.set_title('ワイドスキャン（サーベイスペクトル）')
    
    # ピーク位置にラベル
    for peak in survey_peaks:
        idx = np.argmin(np.abs(wide_be - peak['center']))
        ax1.annotate(peak['label'],
                    xy=(peak['center'], survey_spectrum[idx]),
                    xytext=(peak['center'], survey_spectrum[idx] + 0.15),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
    
    ax1.grid(True, alpha=0.3)
    
    # 右: C 1sナロースキャン
    ax2 = axes[1]
    ax2.plot(be_range, c1s_spectrum, 'k-', linewidth=1, label='Total')
    
    # 各成分をプロット
    colors = ['#f093fb', '#f5576c', '#667eea', '#48bb78']
    for i, peak in enumerate(c1s_peaks):
        component = voigt(be_range, peak['center'], peak['sigma'], peak['gamma'], peak['amplitude'])
        ax2.fill_between(be_range, 0, component, alpha=0.3, color=colors[i], label=peak['label'])
        ax2.axvline(peak['center'], color=colors[i], linestyle='--', alpha=0.5)
    
    ax2.set_xlim(295, 280)
    ax2.set_xlabel('結合エネルギー (eV)')
    ax2.set_ylabel('強度 (任意単位)')
    ax2.set_title('C 1s ナロースキャン（高分解能）')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xps_spectrum_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    5.3 化学状態分析とピークフィッティング
    5.3.1 化学シフトの起源
    原子の化学環境が変化すると、内殻電子の結合エネルギーがシフトします。これを化学シフト（chemical shift）と呼びます。
    
    化学シフトの規則
    
    酸化状態が高い: 結合エネルギーが増加（高BE側へシフト）
    電気陰性度が高い原子と結合: 結合エネルギーが増加
    電子密度が増加: 結合エネルギーが減少（低BE側へシフト）
    
    例: C 1sの化学シフト
    
    C-C/C-H: 284.8 eV（標準）
    C-O (エーテル、アルコール): 286.3 eV (+1.5 eV)
    C=O (カルボニル): 287.8 eV (+3.0 eV)
    O-C=O (カルボキシル、エステル): 289.0 eV (+4.2 eV)
    
    
    5.3.2 Voigt関数によるピークフィッティング
    XPSピークは、装置由来のガウス成分（エネルギー分解能）と物理由来のローレンツ成分（正孔の有限寿命）の畳み込みであるVoigt関数で記述されます。
    
    Voigt関数:
            \[
            V(x; \sigma, \gamma) = \int_{-\infty}^{\infty} G(x'; \sigma) L(x - x'; \gamma) dx'
            \]
            ここで \( G \) はガウス関数、\( L \) はローレンツ関数です。
    ピーク面積: ピーク面積は信号強度に比例し、定量分析に使用されます。
    
    # コード例3: Voigt関数によるピークフィッティング
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.special import voigt_profile
    
    def multi_voigt(x, *params):
        """
        複数のVoigtピークの和
        params: [center1, sigma1, gamma1, amp1, center2, sigma2, gamma2, amp2, ...]
        """
        n_peaks = len(params) // 4
        result = np.zeros_like(x, dtype=float)
    
        for i in range(n_peaks):
            center = params[i * 4]
            sigma = params[i * 4 + 1]
            gamma = params[i * 4 + 2]
            amplitude = params[i * 4 + 3]
            result += amplitude * voigt_profile(x - center, sigma, gamma)
    
        return result
    
    def shirley_background(x, y, tol=1e-5, max_iter=50):
        """
        Shirleyバックグラウンド補正
    
        XPSスペクトルで最も一般的なバックグラウンド補正法
        非弾性散乱電子によるバックグラウンドを除去
    
        Parameters
        ----------
        x : array
            結合エネルギー
        y : array
            強度
        tol : float
            収束判定の許容誤差
        max_iter : int
            最大反復回数
    
        Returns
        -------
        background : array
            Shirleyバックグラウンド
        """
        # 端点の強度
        y_left = np.mean(y[:10])   # 高BE側
        y_right = np.mean(y[-10:]) # 低BE側
    
        # 初期バックグラウンド
        background = np.full_like(y, y_right, dtype=float)
    
        for iteration in range(max_iter):
            background_old = background.copy()
    
            # 累積積分（右から左へ）
            y_subtracted = y - background
            cumulative = np.zeros_like(y)
            for i in range(len(y) - 2, -1, -1):
                cumulative[i] = cumulative[i + 1] + y_subtracted[i + 1]
    
            # 正規化
            if cumulative[0] > 0:
                background = y_right + (y_left - y_right) * cumulative / cumulative[0]
    
            # 収束判定
            if np.max(np.abs(background - background_old)) < tol:
                break
    
        return background
    
    def fit_xps_peaks(x, y, initial_params, peak_labels=None):
        """
        XPSスペクトルのピークフィッティング
    
        Parameters
        ----------
        x : array
            結合エネルギー
        y : array
            バックグラウンド補正後の強度
        initial_params : list
            初期パラメータ [center, sigma, gamma, amp, ...]
        peak_labels : list, optional
            ピークのラベル
    
        Returns
        -------
        popt : array
            最適化されたパラメータ
        pcov : array
            共分散行列
        """
        # パラメータの境界設定
        n_peaks = len(initial_params) // 4
        lower_bounds = []
        upper_bounds = []
    
        for i in range(n_peaks):
            center = initial_params[i * 4]
            lower_bounds.extend([center - 2, 0.1, 0.1, 0])  # center, sigma, gamma, amp
            upper_bounds.extend([center + 2, 2.0, 2.0, np.inf])
    
        popt, pcov = curve_fit(
            multi_voigt, x, y,
            p0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )
    
        return popt, pcov
    
    # サンプルデータ生成（Si 2pスペクトル: Si金属 + SiO2）
    np.random.seed(42)
    be = np.linspace(96, 108, 300)
    
    # 真のパラメータ
    true_params = [
        99.3, 0.5, 0.3, 0.8,    # Si 2p3/2 (Si)
        99.9, 0.5, 0.3, 0.53,   # Si 2p1/2 (Si) - スピン軌道分裂
        103.5, 0.6, 0.35, 1.0,  # Si 2p3/2 (SiO2)
        104.1, 0.6, 0.35, 0.67, # Si 2p1/2 (SiO2) - スピン軌道分裂
    ]
    
    # スペクトル生成
    true_spectrum = multi_voigt(be, *true_params)
    noise = 0.02 * np.random.randn(len(be))
    spectrum_with_noise = true_spectrum + noise + 0.1  # バックグラウンド追加
    
    # 線形バックグラウンド（簡略化）
    linear_bg = 0.1 + 0.005 * (be - be.min())
    spectrum_with_bg = true_spectrum + noise + linear_bg
    
    # Shirleyバックグラウンド補正
    background = shirley_background(be, spectrum_with_bg)
    spectrum_corrected = spectrum_with_bg - background
    
    # ピークフィッティング
    initial_guess = [
        99.3, 0.5, 0.3, 0.7,
        99.9, 0.5, 0.3, 0.5,
        103.5, 0.6, 0.3, 0.9,
        104.1, 0.6, 0.3, 0.6,
    ]
    
    popt, pcov = fit_xps_peaks(be, spectrum_corrected, initial_guess)
    fitted_spectrum = multi_voigt(be, *popt)
    
    # 結果の表示
    print("Si 2p ピークフィッティング結果:")
    print("=" * 60)
    peak_labels = ['Si 2p3/2 (Si)', 'Si 2p1/2 (Si)', 'Si 2p3/2 (SiO2)', 'Si 2p1/2 (SiO2)']
    for i, label in enumerate(peak_labels):
        center = popt[i * 4]
        sigma = popt[i * 4 + 1]
        gamma = popt[i * 4 + 2]
        amp = popt[i * 4 + 3]
    
        # FWHM計算（Voigt関数の近似式）
        fwhm_g = 2 * np.sqrt(2 * np.log(2)) * sigma
        fwhm_l = 2 * gamma
        fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)
    
        print(f"{label:20s}: BE = {center:.2f} eV, FWHM = {fwhm:.2f} eV, Amp = {amp:.3f}")
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上: 生データとバックグラウンド
    ax1 = axes[0, 0]
    ax1.plot(be, spectrum_with_bg, 'k-', linewidth=1, label='生データ')
    ax1.plot(be, background, 'r--', linewidth=1.5, label='Shirleyバックグラウンド')
    ax1.set_xlim(108, 96)
    ax1.set_xlabel('結合エネルギー (eV)')
    ax1.set_ylabel('強度 (任意単位)')
    ax1.set_title('バックグラウンド補正')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右上: バックグラウンド補正後
    ax2 = axes[0, 1]
    ax2.plot(be, spectrum_corrected, 'ko', markersize=2, alpha=0.5, label='データ')
    ax2.plot(be, fitted_spectrum, 'r-', linewidth=2, label='フィッティング')
    ax2.set_xlim(108, 96)
    ax2.set_xlabel('結合エネルギー (eV)')
    ax2.set_ylabel('強度 (任意単位)')
    ax2.set_title('ピークフィッティング結果')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 左下: 成分分離
    ax3 = axes[1, 0]
    ax3.plot(be, spectrum_corrected, 'ko', markersize=2, alpha=0.3, label='データ')
    
    colors = ['#f093fb', '#f5576c', '#667eea', '#48bb78']
    for i, (label, color) in enumerate(zip(peak_labels, colors)):
        center = popt[i * 4]
        sigma = popt[i * 4 + 1]
        gamma = popt[i * 4 + 2]
        amp = popt[i * 4 + 3]
        component = amp * voigt_profile(be - center, sigma, gamma)
        ax3.fill_between(be, 0, component, alpha=0.4, color=color, label=label)
    
    ax3.plot(be, fitted_spectrum, 'k-', linewidth=1.5, label='合計')
    ax3.set_xlim(108, 96)
    ax3.set_xlabel('結合エネルギー (eV)')
    ax3.set_ylabel('強度 (任意単位)')
    ax3.set_title('成分分離（デコンボリューション）')
    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 右下: 残差
    ax4 = axes[1, 1]
    residual = spectrum_corrected - fitted_spectrum
    ax4.plot(be, residual, 'b-', linewidth=1)
    ax4.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax4.fill_between(be, residual, 0, alpha=0.3)
    ax4.set_xlim(108, 96)
    ax4.set_xlabel('結合エネルギー (eV)')
    ax4.set_ylabel('残差')
    ax4.set_title(f'残差 (RMS = {np.sqrt(np.mean(residual**2)):.4f})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xps_peak_fitting.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    5.3.3 化学状態の自動識別
    # コード例4: 化学状態データベースと自動識別
    import numpy as np
    from dataclasses import dataclass
    from typing import List, Tuple, Optional
    
    @dataclass
    class ChemicalState:
        """化学状態の情報を格納するデータクラス"""
        element: str
        orbital: str
        compound: str
        binding_energy: float
        be_range: Tuple[float, float]
        oxidation_state: str
    
    # XPS化学状態データベース
    XPS_DATABASE = [
        # Carbon 1s
        ChemicalState('C', '1s', 'C-C/C-H (adventitious)', 284.8, (284.6, 285.0), '0'),
        ChemicalState('C', '1s', 'C-N', 285.5, (285.3, 285.8), '+'),
        ChemicalState('C', '1s', 'C-O (ether, alcohol)', 286.3, (286.0, 286.6), '+'),
        ChemicalState('C', '1s', 'C=O (carbonyl)', 287.8, (287.5, 288.2), '+2'),
        ChemicalState('C', '1s', 'O-C=O (carboxyl)', 289.0, (288.7, 289.5), '+3'),
        ChemicalState('C', '1s', 'CF2', 291.0, (290.5, 291.5), '+2'),
        ChemicalState('C', '1s', 'CF3', 293.0, (292.5, 293.5), '+3'),
    
        # Oxygen 1s
        ChemicalState('O', '1s', 'Metal oxide (M-O)', 530.0, (529.5, 530.5), '-2'),
        ChemicalState('O', '1s', 'Hydroxide (M-OH)', 531.5, (531.0, 532.0), '-2/-1'),
        ChemicalState('O', '1s', 'Organic C-O', 532.5, (532.0, 533.0), '-2'),
        ChemicalState('O', '1s', 'Adsorbed H2O', 533.5, (533.0, 534.0), '-2'),
    
        # Silicon 2p
        ChemicalState('Si', '2p', 'Si (elemental)', 99.3, (99.0, 99.6), '0'),
        ChemicalState('Si', '2p', 'Si suboxide (SiOx)', 101.5, (100.5, 102.5), '+1/+2'),
        ChemicalState('Si', '2p', 'SiO2', 103.5, (103.0, 104.0), '+4'),
        ChemicalState('Si', '2p', 'Si3N4', 101.8, (101.5, 102.2), '+3'),
    
        # Iron 2p3/2
        ChemicalState('Fe', '2p3/2', 'Fe (metallic)', 707.0, (706.5, 707.5), '0'),
        ChemicalState('Fe', '2p3/2', 'FeO', 709.5, (709.0, 710.0), '+2'),
        ChemicalState('Fe', '2p3/2', 'Fe2O3', 710.8, (710.3, 711.3), '+3'),
        ChemicalState('Fe', '2p3/2', 'FeOOH', 711.5, (711.0, 712.0), '+3'),
    
        # Nitrogen 1s
        ChemicalState('N', '1s', 'Pyridinic N', 398.5, (398.0, 399.0), '-'),
        ChemicalState('N', '1s', 'Amino N (NH2)', 399.5, (399.0, 400.0), '-3'),
        ChemicalState('N', '1s', 'Pyrrolic N', 400.3, (400.0, 400.7), '-'),
        ChemicalState('N', '1s', 'Graphitic N', 401.2, (400.8, 401.6), '-'),
        ChemicalState('N', '1s', 'Oxidized N', 402.5, (402.0, 403.0), '+'),
    ]
    
    def identify_chemical_state(element: str, orbital: str,
                               binding_energy: float) -> List[ChemicalState]:
        """
        結合エネルギーから可能な化学状態を識別
    
        Parameters
        ----------
        element : str
            元素記号
        orbital : str
            軌道（'1s', '2p'など）
        binding_energy : float
            測定された結合エネルギー (eV)
    
        Returns
        -------
        matches : List[ChemicalState]
            マッチする化学状態のリスト（近い順）
        """
        matches = []
    
        for state in XPS_DATABASE:
            if state.element == element and state.orbital == orbital:
                if state.be_range[0] <= binding_energy <= state.be_range[1]:
                    distance = abs(binding_energy - state.binding_energy)
                    matches.append((state, distance))
    
        # 距離でソート
        matches.sort(key=lambda x: x[1])
        return [m[0] for m in matches]
    
    def analyze_spectrum_peaks(peaks: List[dict]) -> dict:
        """
        複数のピークを解析し、化学状態を推定
    
        Parameters
        ----------
        peaks : List[dict]
            ピーク情報のリスト {element, orbital, binding_energy, area}
    
        Returns
        -------
        analysis : dict
            解析結果
        """
        analysis = {
            'peaks': [],
            'summary': {}
        }
    
        for peak in peaks:
            states = identify_chemical_state(
                peak['element'],
                peak['orbital'],
                peak['binding_energy']
            )
    
            peak_analysis = {
                'input': peak,
                'possible_states': [],
            }
    
            for state in states:
                peak_analysis['possible_states'].append({
                    'compound': state.compound,
                    'expected_be': state.binding_energy,
                    'oxidation_state': state.oxidation_state,
                    'shift': peak['binding_energy'] - state.binding_energy
                })
    
            analysis['peaks'].append(peak_analysis)
    
        return analysis
    
    # 使用例
    sample_peaks = [
        {'element': 'C', 'orbital': '1s', 'binding_energy': 284.8, 'area': 1000},
        {'element': 'C', 'orbital': '1s', 'binding_energy': 286.4, 'area': 300},
        {'element': 'C', 'orbital': '1s', 'binding_energy': 289.0, 'area': 150},
        {'element': 'O', 'orbital': '1s', 'binding_energy': 532.3, 'area': 800},
        {'element': 'Si', 'orbital': '2p', 'binding_energy': 103.4, 'area': 500},
    ]
    
    print("XPS化学状態分析結果")
    print("=" * 70)
    
    result = analyze_spectrum_peaks(sample_peaks)
    
    for peak_result in result['peaks']:
        peak = peak_result['input']
        print(f"\n{peak['element']} {peak['orbital']} @ {peak['binding_energy']:.1f} eV:")
        print("-" * 50)
    
        if peak_result['possible_states']:
            for state in peak_result['possible_states']:
                print(f"  - {state['compound']}")
                print(f"    期待値: {state['expected_be']:.1f} eV, "
                      f"シフト: {state['shift']:+.1f} eV, "
                      f"酸化状態: {state['oxidation_state']}")
        else:
            print("  該当する化学状態がデータベースにありません")
    
    5.4 定量分析と感度係数
    5.4.1 定量分析の原理
    XPSによる定量分析では、ピーク面積を相対感度係数（RSF: Relative Sensitivity Factor）で補正し、元素の原子濃度を算出します。
    
    原子濃度の計算式:
            \[
            C_i = \frac{I_i / S_i}{\sum_j (I_j / S_j)} \times 100\%
            \]
            ここで:
    
    \( C_i \): 元素 \( i \) の原子濃度 (at%)
    \( I_i \): 元素 \( i \) のピーク面積
    \( S_i \): 元素 \( i \) の相対感度係数（RSF）
    
    
    5.4.2 相対感度係数
    RSFは光電子断面積、非弾性平均自由行程（IMFP）、装置のトランスミッション関数などを考慮した係数です。装置や測定条件により異なるため、装置メーカーが提供する値を使用するのが一般的です。
    
    
    
    元素/軌道
    RSF (Wagner)
    RSF (Scofield)
    備考
    
    
    
    
    C 1s
    0.25
    1.00
    標準（Scofieldでは参照）
    
    
    O 1s
    0.66
    2.93
    
    
    
    N 1s
    0.42
    1.80
    
    
    
    Si 2p
    0.27
    0.82
    
    
    
    Fe 2p
    2.69
    10.82
    2p3/2 + 2p1/2
    
    
    Au 4f
    4.95
    17.12
    4f7/2 + 4f5/2
    
    
    
    # コード例5: XPS定量分析
    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from typing import Dict, List
    
    # Scofield相対感度係数（C 1s = 1.00基準）
    RSF_SCOFIELD = {
        'C_1s': 1.00,
        'O_1s': 2.93,
        'N_1s': 1.80,
        'Si_2p': 0.82,
        'Si_2s': 0.96,
        'Fe_2p': 10.82,  # 2p3/2 + 2p1/2
        'Fe_2p3/2': 6.77,
        'Cu_2p': 16.73,
        'Cu_2p3/2': 10.67,
        'Au_4f': 17.12,  # 4f7/2 + 4f5/2
        'Au_4f7/2': 9.58,
        'Ag_3d': 18.04,
        'Ti_2p': 7.81,
        'Zn_2p': 18.92,
        'Al_2p': 0.54,
        'Na_1s': 2.85,
        'K_2p': 2.84,
        'Ca_2p': 5.07,
        'Cl_2p': 2.29,
        'F_1s': 4.43,
        'S_2p': 1.68,
        'P_2p': 1.19,
    }
    
    def calculate_atomic_concentration(peak_areas: Dict[str, float],
                                      rsf: Dict[str, float] = RSF_SCOFIELD) -> Dict[str, float]:
        """
        ピーク面積から原子濃度を計算
    
        Parameters
        ----------
        peak_areas : Dict[str, float]
            ピーク面積 {element_orbital: area}
        rsf : Dict[str, float]
            相対感度係数
    
        Returns
        -------
        concentrations : Dict[str, float]
            原子濃度 (at%)
        """
        # 正規化強度の計算
        normalized = {}
        for element, area in peak_areas.items():
            if element in rsf:
                normalized[element] = area / rsf[element]
            else:
                print(f"警告: {element} のRSFが見つかりません。RSF=1.0を使用します。")
                normalized[element] = area
    
        # 合計
        total = sum(normalized.values())
    
        # 原子濃度 (at%)
        concentrations = {elem: (norm / total) * 100 for elem, norm in normalized.items()}
    
        return concentrations
    
    def quantitative_analysis_report(peak_areas: Dict[str, float],
                                    sample_name: str = "Sample") -> None:
        """
        定量分析レポートを生成
        """
        concentrations = calculate_atomic_concentration(peak_areas)
    
        print(f"\nXPS定量分析レポート: {sample_name}")
        print("=" * 60)
        print(f"{'元素/軌道':<15} {'ピーク面積':<12} {'RSF':<8} {'原子濃度 (at%)':<15}")
        print("-" * 60)
    
        total_conc = 0
        for element in sorted(peak_areas.keys()):
            area = peak_areas[element]
            rsf_val = RSF_SCOFIELD.get(element, 1.0)
            conc = concentrations[element]
            total_conc += conc
            print(f"{element:<15} {area:<12.1f} {rsf_val:<8.2f} {conc:<15.2f}")
    
        print("-" * 60)
        print(f"{'合計':<15} {'':<12} {'':<8} {total_conc:<15.2f}")
    
        return concentrations
    
    # 使用例: SiO2薄膜上の有機汚染層の分析
    sample_peaks = {
        'C_1s': 1500,
        'O_1s': 3200,
        'Si_2p': 1800,
        'N_1s': 200,
    }
    
    concentrations = quantitative_analysis_report(sample_peaks, "SiO2/有機汚染層")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左: ピーク面積の比較
    ax1 = axes[0]
    elements = list(sample_peaks.keys())
    areas = list(sample_peaks.values())
    colors = ['#f093fb', '#f5576c', '#667eea', '#48bb78']
    
    bars1 = ax1.bar(elements, areas, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('ピーク面積 (任意単位)')
    ax1.set_title('XPSピーク面積')
    ax1.grid(axis='y', alpha=0.3)
    
    # 各バーに値を表示
    for bar, area in zip(bars1, areas):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{area:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 右: 原子濃度の円グラフ
    ax2 = axes[1]
    conc_values = [concentrations[e] for e in elements]
    explode = [0.02] * len(elements)
    
    wedges, texts, autotexts = ax2.pie(
        conc_values, labels=elements, autopct='%1.1f%%',
        colors=colors, explode=explode,
        shadow=True, startangle=90
    )
    ax2.set_title('原子濃度 (at%)')
    
    # パーセンテージのフォントサイズ
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig('xps_quantitative_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 化学量論比の計算
    print("\n化学量論比の推定:")
    print("-" * 40)
    si_conc = concentrations['Si_2p']
    o_conc = concentrations['O_1s']
    c_conc = concentrations['C_1s']
    
    # Si:O比
    si_o_ratio = si_conc / o_conc
    print(f"Si:O = 1:{o_conc/si_conc:.2f}")
    if abs(o_conc/si_conc - 2) < 0.3:
        print("  -> SiO2と整合")
    elif abs(o_conc/si_conc - 1) < 0.3:
        print("  -> SiOと整合")
    
    # 有機汚染の推定
    if c_conc > 10:
        print(f"\n有機汚染: C = {c_conc:.1f} at%")
        print("  -> 表面に有機汚染層が存在")
    
    5.5 深さ方向分析と表面敏感性
    5.5.1 非弾性平均自由行程（IMFP）
    光電子が試料内を移動する際、非弾性散乱によりエネルギーを失います。非弾性平均自由行程（IMFP: Inelastic Mean Free Path）は、光電子がエネルギーを失わずに移動できる平均距離を表します。
    
    XPSの検出深さ:
            \[
            I(d) = I_0 \exp\left(-\frac{d}{\lambda \cos\theta}\right)
            \]
            ここで:
    
    \( I(d) \): 深さ \( d \) からの信号強度
    \( I_0 \): 表面からの信号強度
    \( \lambda \): 非弾性平均自由行程（IMFP）
    \( \theta \): 検出器への放出角度（表面法線からの角度）
    
    95%検出深さ: \( d_{95\%} = 3\lambda \cos\theta \)（約 3-10 nm）
    
    5.5.2 角度分解XPS（AR-XPS）
    検出角度を変えることで、深さ方向の情報を得ることができます。表面に対して浅い角度（grazing angle）で測定すると、より表面敏感な情報が得られます。
    # コード例6: 深さ方向分析とAR-XPS
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_imfp(kinetic_energy: float, material: str = 'organic') -> float:
        """
        非弾性平均自由行程（IMFP）を計算（TPP-2M式の簡略版）
    
        Parameters
        ----------
        kinetic_energy : float
            光電子の運動エネルギー (eV)
        material : str
            材料タイプ ('organic', 'inorganic', 'metal')
    
        Returns
        -------
        imfp : float
            IMFP (nm)
        """
        # 簡略化されたTPP-2M近似
        # 実際の計算はより複雑
        if material == 'organic':
            # 有機材料: rho = 1 g/cm3 程度
            imfp = 0.0449 * (kinetic_energy ** 0.5) + 0.5
        elif material == 'inorganic':
            # 無機材料（酸化物など）: rho = 3 g/cm3 程度
            imfp = 0.039 * (kinetic_energy ** 0.5) + 0.3
        elif material == 'metal':
            # 金属: rho = 8 g/cm3 程度
            imfp = 0.028 * (kinetic_energy ** 0.5) + 0.2
        else:
            imfp = 0.04 * (kinetic_energy ** 0.5) + 0.4
    
        return imfp
    
    def signal_from_depth(depth: np.ndarray, imfp: float, angle: float = 0) -> np.ndarray:
        """
        深さからの信号強度を計算
    
        Parameters
        ----------
        depth : array
            深さ (nm)
        imfp : float
            非弾性平均自由行程 (nm)
        angle : float
            検出角度（表面法線から、度）
    
        Returns
        -------
        intensity : array
            相対信号強度
        """
        theta_rad = np.radians(angle)
        effective_imfp = imfp * np.cos(theta_rad)
        return np.exp(-depth / effective_imfp)
    
    def depth_profile_simulation(layers: list, imfp: float, angles: list = [0, 45, 75]):
        """
        多層膜のAR-XPSシミュレーション
    
        Parameters
        ----------
        layers : list of dict
            層の情報 [{thickness, composition, name}, ...]
            最初の要素が最表面
        imfp : float
            非弾性平均自由行程 (nm)
        angles : list
            測定角度（度）
        """
        max_depth = sum(layer['thickness'] for layer in layers) + 5
        depth = np.linspace(0, max_depth, 500)
    
        results = {}
    
        for angle in angles:
            signal_weights = signal_from_depth(depth, imfp, angle)
    
            # 各層からの寄与を計算
            layer_signals = {}
            current_depth = 0
    
            for layer in layers:
                layer_mask = (depth >= current_depth) & (depth < current_depth + layer['thickness'])
                weighted_signal = np.sum(signal_weights[layer_mask])
    
                for element, fraction in layer['composition'].items():
                    if element not in layer_signals:
                        layer_signals[element] = 0
                    layer_signals[element] += weighted_signal * fraction
    
                current_depth += layer['thickness']
    
            # 正規化
            total = sum(layer_signals.values())
            results[angle] = {elem: sig/total * 100 for elem, sig in layer_signals.items()}
    
        return results
    
    # IMFPのエネルギー依存性を可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上: IMFPのエネルギー依存性
    ax1 = axes[0, 0]
    ke_range = np.linspace(100, 1400, 100)
    
    for material, style in [('organic', '-'), ('inorganic', '--'), ('metal', ':')]:
        imfp_values = [calculate_imfp(ke, material) for ke in ke_range]
        ax1.plot(ke_range, imfp_values, style, linewidth=2, label=material)
    
    ax1.set_xlabel('運動エネルギー (eV)')
    ax1.set_ylabel('IMFP (nm)')
    ax1.set_title('非弾性平均自由行程（IMFP）のエネルギー依存性')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(100, 1400)
    
    # 右上: 深さからの信号強度
    ax2 = axes[0, 1]
    depth = np.linspace(0, 15, 200)
    imfp = 2.5  # 典型的な値 (nm)
    
    for angle in [0, 30, 45, 60, 75]:
        intensity = signal_from_depth(depth, imfp, angle)
        ax2.plot(depth, intensity, linewidth=2, label=f'{angle} deg')
    
    ax2.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='5%閾値')
    ax2.set_xlabel('深さ (nm)')
    ax2.set_ylabel('相対信号強度')
    ax2.set_title(f'深さからの信号強度（IMFP = {imfp} nm）')
    ax2.legend(title='検出角度')
    ax2.grid(True, alpha=0.3)
    
    # 左下: 多層膜のAR-XPSシミュレーション
    ax3 = axes[1, 0]
    
    # サンプル構造: 有機汚染層/SiO2/Si基板
    layers = [
        {'thickness': 1.5, 'composition': {'C': 0.8, 'O': 0.2}, 'name': '有機汚染'},
        {'thickness': 3.0, 'composition': {'Si': 0.33, 'O': 0.67}, 'name': 'SiO2'},
        {'thickness': 10.0, 'composition': {'Si': 1.0}, 'name': 'Si基板'},
    ]
    
    angles = [0, 15, 30, 45, 60, 75]
    imfp_si = 2.8  # Si 2pの場合
    
    ar_results = depth_profile_simulation(layers, imfp_si, angles)
    
    elements = ['C', 'O', 'Si']
    colors = {'C': '#f093fb', 'O': '#f5576c', 'Si': '#667eea'}
    
    x = np.arange(len(angles))
    width = 0.25
    
    for i, element in enumerate(elements):
        values = [ar_results[angle].get(element, 0) for angle in angles]
        ax3.bar(x + i*width, values, width, label=element, color=colors[element], alpha=0.8)
    
    ax3.set_xlabel('検出角度 (度)')
    ax3.set_ylabel('原子濃度 (at%)')
    ax3.set_title('AR-XPS: 角度依存性（有機層/SiO2/Si）')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(angles)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 右下: 層構造の模式図
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(-2, 16)
    
    # 層を描画
    layer_colors = ['#ffeb3b', '#90caf9', '#a5d6a7']
    layer_names = ['有機汚染層\n(1.5 nm)', 'SiO2\n(3.0 nm)', 'Si基板']
    y_positions = [0, 1.5, 4.5]
    heights = [1.5, 3.0, 10]
    
    for i, (y, h, name, color) in enumerate(zip(y_positions, heights, layer_names, layer_colors)):
        rect = plt.Rectangle((1, y), 6, h, facecolor=color, edgecolor='black', linewidth=1.5)
        ax4.add_patch(rect)
        ax4.text(4, y + h/2, name, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 検出深さの矢印
    for angle, x_pos in zip([0, 45, 75], [7.5, 8.5, 9.5]):
        theta_rad = np.radians(angle)
        d_95 = 3 * imfp_si * np.cos(theta_rad)
        ax4.annotate('', xy=(x_pos, 0), xytext=(x_pos, d_95),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax4.text(x_pos, d_95 + 0.5, f'{angle}deg\n({d_95:.1f}nm)',
                 ha='center', va='bottom', fontsize=8)
    
    ax4.set_title('試料構造と検出深さ')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('xps_depth_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nAR-XPS解析結果:")
    print("=" * 60)
    print(f"{'角度':<10} {'C (at%)':<15} {'O (at%)':<15} {'Si (at%)':<15}")
    print("-" * 60)
    for angle in angles:
        c = ar_results[angle].get('C', 0)
        o = ar_results[angle].get('O', 0)
        si = ar_results[angle].get('Si', 0)
        print(f"{angle:<10} {c:<15.1f} {o:<15.1f} {si:<15.1f}")
    
    print("\n解釈:")
    print("- 高角度(grazing)ではC濃度が増加 -> 表面に有機汚染層")
    print("- 低角度(法線方向)ではSi濃度が増加 -> より深い領域からの信号")
    
    5.5.3 スパッタリングによる深さプロファイル
    # コード例7: スパッタリング深さプロファイル分析
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import erf
    
    def simulate_sputter_depth_profile(layers: list, sputter_rate: float,
                                       total_time: float, time_step: float = 1.0,
                                       interface_width: float = 0.5):
        """
        スパッタリング深さプロファイルをシミュレーション
    
        Parameters
        ----------
        layers : list of dict
            層の情報 [{thickness, composition, name}, ...]
        sputter_rate : float
            スパッタリングレート (nm/min)
        total_time : float
            総スパッタリング時間 (min)
        time_step : float
            時間ステップ (min)
        interface_width : float
            界面の広がり（原子混合、深さ分解能）(nm)
    
        Returns
        -------
        times : array
            スパッタリング時間
        depths : array
            深さ
        profiles : dict
            各元素の濃度プロファイル
        """
        times = np.arange(0, total_time + time_step, time_step)
        depths = times * sputter_rate
    
        # 理想的な濃度プロファイルを生成
        elements = set()
        for layer in layers:
            elements.update(layer['composition'].keys())
    
        profiles = {elem: np.zeros_like(depths) for elem in elements}
    
        # 各層の深さ範囲を計算
        layer_boundaries = [0]
        for layer in layers:
            layer_boundaries.append(layer_boundaries[-1] + layer['thickness'])
    
        # 各深さでの組成を計算（界面の広がりを考慮）
        for i, d in enumerate(depths):
            for layer_idx, layer in enumerate(layers):
                d_start = layer_boundaries[layer_idx]
                d_end = layer_boundaries[layer_idx + 1]
    
                # 誤差関数による界面の広がり
                # 層内の寄与率を計算
    
                # 上界面
                contrib_upper = 0.5 * (1 + erf((d - d_start) / (interface_width * np.sqrt(2))))
                # 下界面
                contrib_lower = 0.5 * (1 + erf((d_end - d) / (interface_width * np.sqrt(2))))
    
                layer_contrib = contrib_upper * contrib_lower
    
                for elem, frac in layer['composition'].items():
                    profiles[elem][i] += frac * layer_contrib
    
        # 正規化
        total = np.zeros_like(depths)
        for elem in profiles:
            total += profiles[elem]
    
        for elem in profiles:
            profiles[elem] = np.where(total > 0, profiles[elem] / total * 100, 0)
    
        return times, depths, profiles
    
    # サンプル: TiN/SiO2/Si多層膜
    layers = [
        {'thickness': 2.0, 'composition': {'Ti': 0.5, 'N': 0.5}, 'name': 'TiN'},
        {'thickness': 10.0, 'composition': {'Si': 0.33, 'O': 0.67}, 'name': 'SiO2'},
        {'thickness': 50.0, 'composition': {'Si': 1.0}, 'name': 'Si基板'},
    ]
    
    sputter_rate = 2.0  # nm/min（SiO2換算）
    total_time = 40     # min
    
    times, depths, profiles = simulate_sputter_depth_profile(
        layers, sputter_rate, total_time, interface_width=1.0
    )
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上: 時間 vs 濃度
    ax1 = axes[0, 0]
    colors = {'Ti': '#f093fb', 'N': '#f5576c', 'Si': '#667eea', 'O': '#48bb78'}
    for elem, profile in profiles.items():
        ax1.plot(times, profile, linewidth=2, label=elem, color=colors.get(elem, 'gray'))
    
    ax1.set_xlabel('スパッタリング時間 (min)')
    ax1.set_ylabel('原子濃度 (at%)')
    ax1.set_title('深さプロファイル（時間軸）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, total_time)
    
    # 右上: 深さ vs 濃度
    ax2 = axes[0, 1]
    for elem, profile in profiles.items():
        ax2.plot(depths, profile, linewidth=2, label=elem, color=colors.get(elem, 'gray'))
    
    # 界面位置をマーク
    layer_boundaries = [0, 2, 12]  # TiN終端、SiO2終端
    for boundary, name in zip(layer_boundaries[1:], ['TiN/SiO2界面', 'SiO2/Si界面']):
        ax2.axvline(boundary, color='gray', linestyle='--', alpha=0.5)
        ax2.annotate(name, (boundary, 90), rotation=90, fontsize=8, alpha=0.7)
    
    ax2.set_xlabel('深さ (nm)')
    ax2.set_ylabel('原子濃度 (at%)')
    ax2.set_title('深さプロファイル（深さ軸）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 25)
    
    # 左下: 層構造の模式図（断面図）
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(-5, 65)
    
    layer_colors = ['#ffcdd2', '#bbdefb', '#c8e6c9']
    layer_y_start = [0, 2, 12]
    layer_heights = [2, 10, 50]
    layer_labels = ['TiN\n(Ti:N = 1:1)', 'SiO2\n(Si:O = 1:2)', 'Si基板']
    
    for y, h, label, color in zip(layer_y_start, layer_heights, layer_labels, layer_colors):
        if h > 20:
            display_h = 20
            ax3.plot([1, 9], [y + display_h, y + display_h], 'k-', linewidth=2)
            ax3.annotate('', xy=(5, y + display_h + 2), xytext=(5, y + display_h + 5),
                        arrowprops=dict(arrowstyle='->', color='black'))
        else:
            display_h = h
    
        rect = plt.Rectangle((1, y), 8, display_h, facecolor=color,
                              edgecolor='black', linewidth=1.5)
        ax3.add_patch(rect)
    
        text_y = y + min(display_h, 10) / 2
        ax3.text(5, text_y, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('深さ (nm)')
    ax3.set_title('試料構造（断面図）')
    ax3.set_aspect('equal')
    ax3.invert_yaxis()
    ax3.set_xticks([])
    
    # 右下: Ti/Si比とO/(Si+Ti)比
    ax4 = axes[1, 1]
    
    # 比の計算
    with np.errstate(divide='ignore', invalid='ignore'):
        ti_si_ratio = np.where(profiles['Si'] > 1,
                               profiles['Ti'] / profiles['Si'],
                               np.nan)
        o_metal_ratio = np.where((profiles['Si'] + profiles['Ti']) > 1,
                                  profiles['O'] / (profiles['Si'] + profiles['Ti']),
                                  np.nan)
    
    ax4.plot(depths, ti_si_ratio, 'b-', linewidth=2, label='Ti/Si')
    ax4.plot(depths, o_metal_ratio, 'r--', linewidth=2, label='O/(Si+Ti)')
    
    ax4.axhline(1, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(2, color='gray', linestyle=':', alpha=0.5)
    
    ax4.set_xlabel('深さ (nm)')
    ax4.set_ylabel('原子比')
    ax4.set_title('原子比の深さ依存性')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 25)
    ax4.set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig('xps_sputter_depth_profile.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n深さプロファイル解析結果:")
    print("=" * 50)
    print(f"スパッタリングレート: {sputter_rate} nm/min")
    print(f"総スパッタリング時間: {total_time} min")
    print(f"総エッチング深さ: {total_time * sputter_rate} nm")
    print("\n界面位置の推定:")
    print("-" * 50)
    for i, (boundary, name) in enumerate(zip([2, 12], ['TiN/SiO2', 'SiO2/Si'])):
        time_at_boundary = boundary / sputter_rate
        print(f"{name}: 深さ = {boundary} nm, 時間 = {time_at_boundary:.1f} min")
    
    5.6 実践演習
    
    演習1: 結合エネルギーの計算
    Al Ka線（1486.6 eV）を使用して、運動エネルギー 1380 eV で光電子が検出されました。仕事関数を 4.5 eV とすると、この電子の結合エネルギーはいくらですか？また、どの元素のどの軌道に由来する可能性がありますか？
    
    解答を見る
    
    結合エネルギーの計算:
    $$E_B = h\nu - E_K - \phi = 1486.6 - 1380 - 4.5 = 102.1 \text{ eV}$$
    元素の同定:
    BE = 102.1 eV は Si 2p（SiO2の場合 103.5 eV、Siサブオキサイドの場合 100-102 eV）に該当する可能性が高いです。
    # 検証コード
    photon_energy = 1486.6
    kinetic_energy = 1380
    work_function = 4.5
    
    binding_energy = photon_energy - kinetic_energy - work_function
    print(f"結合エネルギー: {binding_energy:.1f} eV")
    print("推定: Si 2p（サブオキサイド SiOx, x < 2）")
    
    
    
    
    
    演習2: Voigt関数によるピークフィッティング
    以下のFe 2p3/2スペクトルデータに対して、2成分（Fe金属とFe2O3）のVoigt関数フィッティングを行い、各成分の割合を求めてください。
    
    解答を見る
    
    # Fe 2p3/2のピークフィッティング
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.special import voigt_profile
    from scipy.integrate import simpson
    import matplotlib.pyplot as plt
    
    def dual_voigt(x, c1, s1, g1, a1, c2, s2, g2, a2, bg):
        """2成分Voigt関数 + バックグラウンド"""
        v1 = a1 * voigt_profile(x - c1, s1, g1)
        v2 = a2 * voigt_profile(x - c2, s2, g2)
        return v1 + v2 + bg
    
    # サンプルデータ生成
    be = np.linspace(704, 716, 200)
    true_spectrum = dual_voigt(be, 707.0, 0.5, 0.3, 1.0,  # Fe metal
                                    710.8, 0.6, 0.4, 0.8,  # Fe2O3
                                    0.1)
    noise = 0.02 * np.random.randn(len(be))
    data = true_spectrum + noise
    
    # フィッティング
    p0 = [707, 0.5, 0.3, 0.8, 711, 0.6, 0.4, 0.6, 0.1]
    popt, _ = curve_fit(dual_voigt, be, data, p0=p0)
    
    # 成分の面積計算
    fe_metal = popt[3] * voigt_profile(be - popt[0], popt[1], popt[2])
    fe2o3 = popt[7] * voigt_profile(be - popt[4], popt[5], popt[6])
    
    area_metal = simpson(fe_metal, x=be)
    area_oxide = simpson(fe2o3, x=be)
    total_area = area_metal + area_oxide
    
    print(f"Fe金属: {area_metal/total_area*100:.1f}%")
    print(f"Fe2O3: {area_oxide/total_area*100:.1f}%")
    
    答え: Fe金属 約55%、Fe2O3 約45%（ノイズにより変動）
    
    
    
    
    演習3: 定量分析
    以下のXPSピーク面積から、表面の原子組成を計算してください。
    
    C 1s: 面積 = 2500
    O 1s: 面積 = 4500
    Ti 2p: 面積 = 6000
    
    RSF（Scofield）: C 1s = 1.00, O 1s = 2.93, Ti 2p = 7.81
    
    解答を見る
    
    # 定量分析計算
    peak_areas = {'C_1s': 2500, 'O_1s': 4500, 'Ti_2p': 6000}
    rsf = {'C_1s': 1.00, 'O_1s': 2.93, 'Ti_2p': 7.81}
    
    # 正規化
    normalized = {elem: area/rsf[elem] for elem, area in peak_areas.items()}
    total = sum(normalized.values())
    
    # 原子濃度
    concentrations = {elem: norm/total*100 for elem, norm in normalized.items()}
    
    for elem, conc in concentrations.items():
        print(f"{elem}: {conc:.1f} at%")
    
    # Ti:O比の確認
    ti_o_ratio = concentrations['Ti_2p'] / concentrations['O_1s']
    print(f"\nTi:O = 1:{1/ti_o_ratio:.2f}")
    
    答え:
    
    C 1s: 52.7 at%
    O 1s: 32.4 at%
    Ti 2p: 16.2 at%
    
    Ti:O = 1:2.0 -> TiO2と整合（表面に炭素汚染あり）
    
    
    
    
    演習4: AR-XPSによる薄膜厚さの推定
    Si基板上のSiO2薄膜について、0度と60度でのSi 2pスペクトルから以下のデータが得られました。酸化膜の厚さを推定してください。
    
    0度: SiO2 (103.5 eV) = 70%, Si (99.3 eV) = 30%
    60度: SiO2 (103.5 eV) = 85%, Si (99.3 eV) = 15%
    
    IMFP (Si 2p in SiO2) = 2.8 nm
    
    解答を見る
    
    # 薄膜厚さの推定
    import numpy as np
    
    # データ
    ratio_0deg = 70 / 30   # SiO2/Si at 0 deg
    ratio_60deg = 85 / 15  # SiO2/Si at 60 deg
    
    imfp = 2.8  # nm
    
    # 薄膜厚さ d の推定
    # I_oxide / I_substrate = [1 - exp(-d/lambda)] / exp(-d/lambda)
    # R = exp(d/lambda) - 1
    
    # 0度の場合
    lambda_0 = imfp * np.cos(np.radians(0))
    d_0 = lambda_0 * np.log(1 + ratio_0deg)
    
    # 60度の場合
    lambda_60 = imfp * np.cos(np.radians(60))
    d_60 = lambda_60 * np.log(1 + ratio_60deg)
    
    print(f"0度から推定した膜厚: {d_0:.2f} nm")
    print(f"60度から推定した膜厚: {d_60:.2f} nm")
    print(f"平均膜厚: {(d_0 + d_60)/2:.2f} nm")
    
    答え: 0度から 3.37 nm、60度から 2.66 nm、平均 約 3.0 nm（角度により異なる値が得られるのは、実際には汚染層や表面ラフネスの影響を受けるため）
    
    
    
    
    演習5: 化学状態の同定
    C 1sスペクトルで以下のピークが観測されました。各ピークに対応する化学状態を同定し、ポリエチレンテレフタレート（PET）の構造と比較してください。
    
    ピーク1: 284.8 eV（強度 100）
    ピーク2: 286.4 eV（強度 40）
    ピーク3: 288.9 eV（強度 20）
    
    
    解答を見る
    
    化学状態の同定:
    
    284.8 eV: C-C/C-H（芳香環の炭素）
    286.4 eV: C-O（エステル結合の C-O）
    288.9 eV: O-C=O（カルボキシルエステル）
    
    PET構造との比較:
    PET: [-O-CH2-CH2-O-CO-C6H4-CO-]n
    
    芳香環C: 4個
    C-O (エチレン): 2個
    O-C=O: 2個
    
    理論比 = 4:2:2 = 2:1:1
    測定比 = 100:40:20 = 5:2:1
    -> 表面に炭化水素系の汚染があると推測
    
    
    
    
    学習目標の確認
    以下の項目について、自己評価してください：
    レベル1: 基本理解
    
    光電効果とアインシュタインの式を理解している
    結合エネルギーと化学シフトの関係を説明できる
    XPS装置の基本構成（X線源、分析器）を理解している
    相対感度係数（RSF）の意味を理解している
    
    レベル2: 実践スキル
    
    Shirleyバックグラウンド補正ができる
    Voigt関数によるピークフィッティングができる
    定量分析により原子濃度を計算できる
    化学状態データベースを用いてピークを同定できる
    
    レベル3: 応用力
    
    AR-XPSによる深さ方向分析ができる
    スパッタリング深さプロファイルを解析できる
    多成分系のデコンボリューションができる
    薄膜厚さを推定できる
    
    
    
    参考文献
    
    Briggs, D., Seah, M.P. (1990). Practical Surface Analysis, Volume 1: Auger and X-ray Photoelectron Spectroscopy (2nd ed.). Wiley, pp. 26-31 (photoionization cross-sections), pp. 85-105 (quantification methods), pp. 201-215 (chemical shifts), pp. 312-335 (depth profiling). - XPSの原理、定量分析法、実践的測定技術の包括的解説
    Shirley, D.A. (1972). High-resolution X-ray photoemission spectrum of the valence bands of gold. Physical Review B, 5(12), 4709-4714. DOI: 10.1103/PhysRevB.5.4709 - Shirleyバックグラウンド減算法の原論文
    Scofield, J.H. (1976). Hartree-Slater subshell photoionization cross-sections at 1254 and 1487 eV. Journal of Electron Spectroscopy and Related Phenomena, 8(2), 129-137. DOI: 10.1016/0368-2048(76)80015-1 - XPS相対感度係数の理論計算の基礎論文
    Hufner, S. (2003). Photoelectron Spectroscopy: Principles and Applications (3rd ed.). Springer, pp. 1-28 (basic principles), pp. 45-65 (chemical shifts), pp. 350-380 (surface analysis). - 光電子分光の量子力学的基礎、化学シフトの理論
    Moulder, J.F., Stickle, W.F., Sobol, P.E., Bomben, K.D. (1992). Handbook of X-ray Photoelectron Spectroscopy. Physical Electronics, pp. 40-42 (C 1s), pp. 82-84 (O 1s), pp. 181-183 (Si 2p), pp. 230-232 (Fe 2p). - XPSスペクトルデータベース、標準ピーク位置集
    Powell, C.J., Jablonski, A. (2010). NIST Electron Inelastic-Mean-Free-Path Database, Version 1.2. National Institute of Standards and Technology. DOI: 10.18434/T48C78 - 非弾性平均自由行程（IMFP）データベース
    Tanuma, S., Powell, C.J., Penn, D.R. (2011). Calculations of electron inelastic mean free paths. IX. Data for 41 elemental solids over the 50 eV to 30 keV range. Surface and Interface Analysis, 43(3), 689-713. DOI: 10.1002/sia.3522 - TPP-2M式によるIMFP計算
    SciPy 1.11 documentation. scipy.special.voigt_profile, scipy.optimize.curve_fit. https://docs.scipy.org/doc/scipy/ - Voigt関数、非線形最小二乗フィッティング
    
    
    
    
    
    免責事項
    
    本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言（法律・会計・技術的保証など）を提供するものではありません。
    本コンテンツおよび付随するコード例は「現状有姿（AS IS）」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
    外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
    本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
    本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
    本コンテンツの著作権・ライセンスは明記された条件（例: CC BY 4.0）に従います。当該ライセンスは通常、無保証条項を含みます。
    ```
