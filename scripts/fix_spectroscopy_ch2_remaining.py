#!/usr/bin/env python3
"""
Fix remaining Japanese text in spectroscopy chapter 2
"""

import re

def fix_remaining_japanese():
    """Fix all remaining Japanese text"""

    file_path = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/spectroscopy-introduction/chapter-2.html"

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix hybrid words and remaining Japanese
    fixes = {
        'コード例1: 調和Vibration子のエネルギー準位とVibration周波数計算':
            'Code Example 1: Energy Levels and Vibrational Frequency Calculation for Harmonic Oscillator',
        '2原子分子のVibrational frequency (Hz)とWavenumber (cm^-1)を計算':
            'Calculate vibrational frequency (Hz) and wavenumber (cm^-1) for diatomic molecule',
        '調和Vibration子のエネルギー準位':
            'Energy levels of harmonic oscillator',
        'VibrationModeは': 'Vibrational modes are classified into',
        'StretchVibration（stretching）': 'stretching vibrations',
        '変角Vibration（bending）': 'bending vibrations',
        'StretchVibration': 'Stretching vibrations',
        '対称Stretch（symmetric stretch, νₛ）、非対称Stretch（asymmetric stretch, νₐₛ）':
            'symmetric stretch (νₛ), asymmetric stretch (νₐₛ)',
        '変角Vibration': 'Bending vibrations',
        'はさみVibration（scissoring, δ）、横揺れVibration（rocking, ρ）、縦揺れVibration（wagging, ω）、ねじれVibration（twisting, τ）':
            'scissoring (δ), rocking (ρ), wagging (ω), twisting (τ)',
        'コード例2: H₂O分子の3つのVibrationModeシミュレーション':
            'Code Example 2: Simulation of Three Vibrational Modes of H₂O Molecule',
        'H2O分子の3つのVibrationMode（対称Stretch、非対称Stretch、変角）の可視化':
            'Visualization of three vibrational modes of H₂O (symmetric stretch, asymmetric stretch, bending)',
        'VibrationModeの図': 'Vibrational mode diagram',
        'VibrationMode': 'Vibrational mode',
        '変角Vibration (δ)': 'Bending (δ)',
        'Vibrationの変位（拡大表示）': 'Vibrational displacement (magnified)',
        'VibrationModeの特徴': 'Characteristics of vibrational modes',
        'H₂O分子の3つの基本VibrationMode': 'Three Fundamental Vibrational Modes of H₂O Molecule',
        '両方のO-HBondが同時にStretch': 'Both O-H bonds stretch simultaneously',
        '変角Vibration (δ): 1595 cm⁻¹': 'Bending (δ): 1595 cm⁻¹',
        '最も低いVibration数（Weakい力の定数）': 'Lowest frequency (weak force constant)',
        'IRでVibrationがActive（IR active）であるためには、Vibrationに伴って':
            'For a vibration to be IR active, there must be a change during the vibration of',
        'Vibrationの規準座標': 'normal coordinate of the vibration',
        '対称StretchVibrationはIRInactiveですが、非対称Stretchや変角VibrationはIR Activeです':
            'symmetric stretching vibration is IR inactive, but asymmetric stretching and bending vibrations are IR active',
        '非常にStrong': 'Very strong',
        '透過率 (%)': 'Transmittance (%)',
        'IR spectrumのシミュレーション': 'IR spectrum simulation',
        '各波数のIntensity': 'Intensity at each wavenumber',
        'IR spectrumからInterferogramを生成（簡略化版）':
            'Generate interferogram from IR spectrum (simplified version)',
        'Interferogram（各波数成分の干渉パターンの和）':
            'Interferogram (sum of interference patterns for each wavenumber component)',
        'Interferogramを Fourier transformしてスペクトルを復元':
            'Fourier transform interferogram to restore spectrum',
        'Interferogram生成': 'Generate interferogram',
        'Fourier transformでスペクトル復元': 'Restore spectrum by Fourier transform',
        'FTIRInterferogram（時間領域）': 'FTIR Interferogram (Time Domain)',
        'Fourier transform後のスペクトル': 'Spectrum after Fourier transform',
        '干渉パターン（Interferogram）を記録': 'Record interference pattern (interferogram)',
        'Fourier transformで周波数領域のスペクトルを復元': 'Restore frequency-domain spectrum by Fourier transform',
        '分子のVibrational energy（周波数 $\\nu_m$）だけシフトした散乱光として観測される現象':
            'scattered light is observed with a frequency shift by the molecular vibrational energy (frequency $\\nu_m$)',
        '分子が励起': 'molecule is excited',
        '既に励起状態にある分子が基底状態へ、Weak': 'excited molecule returns to ground state, weak',
        'VibrationMode': 'Vibrational mode',
        'Stokesピーク（正のシフト）': 'Stokes peak (positive shift)',
        'Anti-Stokesピーク（負のシフト）': 'Anti-Stokes peak (negative shift)',
        'Boltzmann因子でIntensityが減少': 'Intensity decreases by Boltzmann factor',
        'Rayleigh散乱（中心、非常にStrong）': 'Rayleigh scattering (center, very strong)',
        '全スペクトル（Rayleigh含む）': 'Full spectrum (including Rayleigh)',
        'Stokes領域の拡大（実用的なMeasurement範囲）': 'Enlarged Stokes region (practical measurement range)',
        'ピーク帰属': 'Peak assignment',
        'Boltzmann比の温度依存性': 'Temperature dependence of Boltzmann ratio',
        '結晶ピークと非晶ピークの2成分モデル': 'Two-component model for crystalline and amorphous peaks',
        '結晶ピーク': 'Crystalline peak',
        '非晶ピーク': 'Amorphous peak',
        '初期推定値': 'Initial guess',
        'フィッティング': 'Fitting',
        '個別ピーク': 'Individual peaks',
        '結晶化度（ピーク面積比）': 'Crystallinity (peak area ratio)',
        '合成データ（半結晶性Polymerのc-c stretch領域）':
            'Synthetic data (C-C stretching region of semicrystalline polymer)',
        '結晶化度解析': 'Crystallinity analysis',
        '実験データ': 'Experimental data',
        '結晶成分': 'Crystalline component',
        '非晶成分': 'Amorphous component',
        'ピーク分離による結晶化度解析': 'Crystallinity Analysis by Peak Deconvolution',
        '結晶化度の表示': 'Display crystallinity',
        '結果の出力': 'Output results',
        'H₂O分子（c2v点群）のVibrationModeと選択則':
            'Vibrational modes and selection rules of H₂O molecule (C2v point group)',
        'VibrationModeの情報': 'Vibrational mode information',
        'h2o分子の3つの基本VibrationMode': 'Three fundamental vibrational modes of H₂O',
        '両O-HBondが同時にStretch、対称': 'Both O-H bonds stretch simultaneously, symmetric',
        'H-O-H角が変化': 'H-O-H angle changes',
        '一方のO-Hが伸びる時、他方が縮む': 'When one O-H stretches, the other contracts',
        '表形式で表示': 'Display in table format',
        'モード': 'Mode',
        '説明': 'Description',
        'c₂v点群の指標表（Character Table）': 'Character Table of C₂v Point Group',
        '基底関数': 'Basis Functions',
        '選択則:': 'Selection rules:',
        'h₂oの場合、a₁とb₁はいずれもirとraman両方でActive':
            'For H₂O, both A₁ and B₁ are active in both IR and Raman',
        '可視化：エネルギーレベル図': 'Visualization: Energy level diagram',
        '基底状態と励起状態': 'Ground state and excited states',
        'スケーリング': 'Scaling',
        '遷移矢印': 'Transition arrows',
        '基底状態 (v=0)': 'Ground state (v=0)',
        '相対エネルギー (cm⁻¹ / 100)': 'Relative Energy (cm⁻¹ / 100)',
        'h₂o分子のVibration励起エネルギー準位': 'Vibrational Excitation Energy Levels of H₂O Molecule',
        'IR・Ramanスペクトルの統合解析クラス': 'Integrated IR and Raman spectral analysis class',
        '官能基データベース（簡略版）': 'Functional group database (simplified)',
        'スペクトルから官能基を同定': 'Identify functional groups from spectrum',
        'ピーク検出の閾値（最大値に対する相対値）': 'Threshold for peak detection (relative to maximum)',
        '同定された官能基のリスト': 'List of identified functional groups',
        'ピーク検出': 'Peak detection',
        '官能基データベースと照合': 'Match against functional group database',
        'IRとRamanの相補的解析': 'Complementary analysis of IR and Raman',
        '統合解析結果': 'Integrated analysis results',
        'IRで検出された官能基': 'Functional groups detected in IR',
        'Ramanで検出された官能基': 'Functional groups detected in Raman',
        '統合': 'Integration',
        '実行例：Acetone（CH₃COCH₃）のIR・Raman統合解析':
            'Example: Integrated IR and Raman analysis of acetone (CH₃COCH₃)',
        '合成IRスペクトル': 'Synthetic IR spectrum',
        'ノイズ': 'Noise',
        '合成Ramanスペクトル': 'Synthetic Raman spectrum',
        '統合解析': 'Integrated analysis',
        'IRスペクトル': 'IR spectrum',
        '吸光度 (a.u.)': 'Absorbance (a.u.)',
        'AcetoneのIRスペクトル': 'IR Spectrum of Acetone',
        'Ramanスペクトル': 'Raman spectrum',
        'AcetoneのRamanスペクトル': 'Raman Spectrum of Acetone',
        '解析結果の表示': 'Display analysis results',
        'IRとRamanの統合解析結果（Acetone）': 'Integrated IR and Raman Analysis Results (Acetone)',
        'IRのみで検出:': 'Detected only in IR:',
        'Ramanのみで検出:': 'Detected only in Raman:',
        '両方で検出:': 'Detected in both:',
        '結論:': 'Conclusions:',
        'この章で学んだ内容を振り返り、以下の項目を確認してください。':
            'Review what you learned in this chapter and check the following items.',
        '基本理解': 'Basic Understanding',
        '実践スキル': 'Practical Skills',
        '応用力': 'Application Skills',
        '第2章では、赤外・ラマン分光法の原理、選択則、官能基同定、群論による対称性解析を学びました。調和Vibration子モデル、ftirの原理、stokes/anti-stokes散乱、結晶化度評価など、実践的なData解析スキルも習得しました。':
            'In Chapter 2, we learned the principles of infrared and Raman spectroscopy, selection rules, functional group identification, and symmetry analysis using group theory. We also acquired practical data analysis skills including the harmonic oscillator model, FTIR principles, Stokes/Anti-Stokes scattering, and crystallinity evaluation.',
    }

    # Apply all fixes
    for jp, en in fixes.items():
        content = content.replace(jp, en)

    # Additional regex-based fixes for common patterns
    content = re.sub(r'Vibration子', 'oscillator', content)
    content = re.sub(r'VibrationMode', 'vibrational mode', content)
    content = re.sub(r'Vibration', 'vibration', content)
    content = re.sub(r'Stretch', 'stretch', content)
    content = re.sub(r'Bond', 'bond', content)
    content = re.sub(r'Weak', 'weak', content)
    content = re.sub(r'Strong', 'strong', content)
    content = re.sub(r'Active', 'active', content)
    content = re.sub(r'Inactive', 'inactive', content)
    content = re.sub(r'Interferogram', 'interferogram', content)
    content = re.sub(r'Intensity', 'intensity', content)
    content = re.sub(r'Measurement', 'measurement', content)
    content = re.sub(r'Polymer', 'polymer', content)
    content = re.sub(r'Data', 'data', content)
    content = re.sub(r'Acetone', 'acetone', content)

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Fixed remaining Japanese text")
    print(f"File size: {len(content)} bytes")

if __name__ == "__main__":
    fix_remaining_japanese()
    print("\nVerifying Japanese character count...")
