#!/usr/bin/env python3
"""
Complete translation of spectroscopy-introduction chapter-2.html
Translates all Japanese content to English while preserving HTML structure
"""

import re

def translate_content(content: str) -> str:
    """Complete translation of chapter 2"""
    
    # Title and metadata
    content = content.replace('<html lang="ja">', '<html lang="en">')
    content = content.replace('<title>第2章:赤外・ラマン分光法 - MS Terakoya</title>', 
                             '<title>Chapter 2: Infrared and Raman Spectroscopy - MS Terakoya</title>')
    
    # Breadcrumb
    content = content.replace('AI寺子屋トップ', 'AI Terakoya Top')
    content = content.replace('材料科学', 'Materials Science')
    
    # Header
    content = content.replace('<h1>第2章：赤外・ラマン分光法</h1>', 
                             '<h1>Chapter 2: Infrared and Raman Spectroscopy</h1>')
    content = content.replace('<p class="subtitle">振動分光で探る分子構造と化学結合</p>', 
                             '<p class="subtitle">Probing Molecular Structure and Chemical Bonds with Vibrational Spectroscopy</p>')
    content = content.replace('📚 シリーズ: 分光分析入門', '📚 Series: Introduction to Spectroscopy')
    content = content.replace('⏱️ 学習時間: 100分', '⏱️ Study Time: 100 minutes')
    content = content.replace('🎯 難易度: 初級〜中級', '🎯 Difficulty: Beginner to Intermediate')
    
    # Introduction section
    content = content.replace('<h2>イントロダクション</h2>', '<h2>Introduction</h2>')
    
    intro_jp = '赤外分光（Infrared Spectroscopy, IR）とラマン分光（Raman Spectroscopy）は、分子の振動情報を通じて化学結合、官能基、結晶構造を解明する相補的な手法です。IRは赤外光の吸収を測定し、Ramanは散乱光の周波数シフトを観測します。両者は異なる選択則に従うため、IRで活性な振動がRamanで不活性、またはその逆という相補性を持ちます。'
    intro_en = 'Infrared (IR) spectroscopy and Raman spectroscopy are complementary techniques for elucidating chemical bonds, functional groups, and crystal structures through molecular vibrational information. IR measures absorption of infrared light, while Raman observes frequency shifts in scattered light. Because they follow different selection rules, vibrations that are IR-active may be Raman-inactive, and vice versa, providing complementary information.'
    content = content.replace(intro_jp, intro_en)
    
    # Info box
    content = content.replace('<strong>IRとRamanの使い分け</strong>', 
                             '<strong>When to Use IR vs Raman</strong>')
    
    ir_use_jp = '<li><strong>IR</strong>: 極性基（C=O, O-H, N-H）の検出、有機物の官能基同定、固体・液体・気体すべてに適用可能</li>'
    ir_use_en = '<li><strong>IR</strong>: Detection of polar groups (C=O, O-H, N-H), identification of functional groups in organic compounds, applicable to solids, liquids, and gases</li>'
    content = content.replace(ir_use_jp, ir_use_en)
    
    raman_use_jp = '<li><strong>Raman</strong>: 対称振動（C=C, S-S）の検出、水溶液試料、結晶性評価（低波数領域）、非破壊・非接触測定</li>'
    raman_use_en = '<li><strong>Raman</strong>: Detection of symmetric vibrations (C=C, S-S), aqueous samples, crystallinity assessment (low-frequency region), non-destructive and contactless measurements</li>'
    content = content.replace(raman_use_jp, raman_use_en)
    
    # Section 1
    content = content.replace('<h2>1. 分子振動の基礎</h2>', '<h2>1. Fundamentals of Molecular Vibrations</h2>')
    content = content.replace('<h3>1.1 調和振動子モデル</h3>', '<h3>1.1 Harmonic Oscillator Model</h3>')
    
    harmonic_jp = '2原子分子の振動は調和振動子で近似できます。ポテンシャルエネルギーはHookeの法則に従います：'
    harmonic_en = 'The vibration of diatomic molecules can be approximated by a harmonic oscillator. The potential energy follows Hooke\'s law:'
    content = content.replace(harmonic_jp, harmonic_en)
    
    force_const_jp = 'ここで、$k$ は力の定数（N/m）、$r_e$ は平衡核間距離です。振動周波数 $\\nu$ は以下で与えられます：'
    force_const_en = 'where $k$ is the force constant (N/m) and $r_e$ is the equilibrium internuclear distance. The vibrational frequency $\\nu$ is given by:'
    content = content.replace(force_const_jp, force_const_en)
    
    reduced_mass_jp = '$\\mu = \\frac{m_1 m_2}{m_1 + m_2}$ は換算質量です。振動エネルギー準位は量子化され、'
    reduced_mass_en = 'where $\\mu = \\frac{m_1 m_2}{m_1 + m_2}$ is the reduced mass. The vibrational energy levels are quantized:'
    content = content.replace(reduced_mass_jp, reduced_mass_en)
    
    selection_jp = '調和振動子近似では、選択則は $\\Delta v = \\pm 1$ です（基本振動のみ許容）。実際の分子では非調和性により $\\Delta v = \\pm 2, \\pm 3, \\ldots$（倍音）も弱く観測されます。'
    selection_en = 'In the harmonic oscillator approximation, the selection rule is $\\Delta v = \\pm 1$ (only fundamental vibrations are allowed). In real molecules, anharmonicity allows weak observation of $\\Delta v = \\pm 2, \\pm 3, \\ldots$ (overtones).'
    content = content.replace(selection_jp, selection_en)
    
    # Code Example 1
    content = content.replace('<h4>コード例1: 調和振動子のエネルギー準位と振動周波数計算</h4>',
                             '<h4>Code Example 1: Calculation of Harmonic Oscillator Energy Levels and Vibrational Frequencies</h4>')
    
    # Python code comments
    content = content.replace('# 物理定数', '# Physical constants')
    content = content.replace('    2原子分子の振動周波数（Hz）と波数（cm^-1）を計算', 
                             '    Calculate vibrational frequency (Hz) and wavenumber (cm^-1) for diatomic molecules')
    content = content.replace('        力の定数（N/m）', '        Force constant (N/m)')
    content = content.replace('        原子の質量（amu）', '        Atomic masses (amu)')
    content = content.replace('        振動周波数（Hz）', '        Vibrational frequency (Hz)')
    content = content.replace('        波数（cm^-1）', '        Wavenumber (cm^-1)')
    content = content.replace('    # 換算質量', '    # Reduced mass')
    content = content.replace('    # 振動周波数', '    # Vibrational frequency')
    content = content.replace('    # 波数に変換', '    # Convert to wavenumber')
    content = content.replace('    調和振動子のエネルギー準位', 
                             '    Harmonic oscillator energy levels')
    content = content.replace('        最大振動量子数', '        Maximum vibrational quantum number')
    content = content.replace('        振動周波数（Hz）', '        Vibrational frequency (Hz)')
    content = content.replace('        振動量子数', '        Vibrational quantum number')
    content = content.replace('        エネルギー（eV）', '        Energy (eV)')
    content = content.replace('# 典型的な化学結合の計算', '# Calculations for typical chemical bonds')
    content = content.replace('print("典型的な化学結合の振動周波数")', 
                             'print("Vibrational Frequencies of Typical Chemical Bonds")')
    content = content.replace("print(f\"{'結合':<8} {'力の定数 (N/m)':<18} {'周波数 (Hz)':<18} {'波数 (cm⁻¹)':<15}\")",
                             "print(f\"{'Bond':<8} {'Force Constant (N/m)':<18} {'Frequency (Hz)':<18} {'Wavenumber (cm⁻¹)':<15}\")")
    content = content.replace('# C=O伸縮振動のエネルギー準位図', 
                             '# Energy level diagram for C=O stretching vibration')
    
    # Chart labels
    content = content.replace("ax1.set_ylabel('エネルギー (eV)', fontsize=12)",
                             "ax1.set_ylabel('Energy (eV)', fontsize=12)")
    content = content.replace("ax1.set_title('C=O伸縮振動のエネルギー準位', fontsize=14, fontweight='bold')",
                             "ax1.set_title('Energy Levels of C=O Stretching Vibration', fontsize=14, fontweight='bold')")
    content = content.replace("# 遷移の矢印", "# Transition arrows")
    content = content.replace("# 同位体効果", "# Isotope effect")
    content = content.replace("ax2.set_ylabel('波数 (cm⁻¹)', fontsize=12)",
                             "ax2.set_ylabel('Wavenumber (cm⁻¹)', fontsize=12)")
    content = content.replace("ax2.set_title('同位体効果：C=O伸縮振動', fontsize=14, fontweight='bold')",
                             "ax2.set_title('Isotope Effect: C=O Stretching Vibration', fontsize=14, fontweight='bold')")
    
    # Print statements
    content = content.replace('print(f"\\nC=O伸縮振動の波数: {wn_CO:.1f} cm⁻¹")',
                             'print(f"\\nWavenumber of C=O stretch: {wn_CO:.1f} cm⁻¹")')
    content = content.replace('print(f"基底状態(v=0)のゼロ点エネルギー: {E[0]:.4f} eV")',
                             'print(f"Zero-point energy of ground state (v=0): {E[0]:.4f} eV")')
    content = content.replace('print(f"v=0 → v=1 遷移エネルギー: {E[1] - E[0]:.4f} eV")',
                             'print(f"Transition energy v=0 → v=1: {E[1] - E[0]:.4f} eV")')
    
    return content

# Read the Japanese source file
with open('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/spectroscopy-introduction/chapter-2.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Translate
translated = translate_content(content)

# Write to English file
with open('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/spectroscopy-introduction/chapter-2.html', 'w', encoding='utf-8') as f:
    f.write(translated)

print("✅ Initial translation complete - Phase 1 done")
print("Run Phase 2 script next for remaining sections")
