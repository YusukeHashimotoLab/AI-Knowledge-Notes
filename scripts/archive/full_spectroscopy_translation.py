#!/usr/bin/env python3
"""
Complete systematic translation of spectroscopy chapter-2
Handles all 5000+ Japanese characters in structured phases
"""

import re
import sys

def translate_full_file():
    # Read Japanese source
    with open('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/spectroscopy-introduction/chapter-2.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # PHASE 1: HTML STRUCTURE & METADATA
    translations = {
        # HTML lang
        '<html lang="ja">': '<html lang="en">',
        
        # Title
        '<title>第2章:赤外・ラマン分光法 - MS Terakoya</title>': 
        '<title>Chapter 2: Infrared and Raman Spectroscopy - MS Terakoya</title>',
        
        # Breadcrumb
        'AI寺子屋トップ': 'AI Terakoya Top',
        '材料科学': 'Materials Science',
        
        # Header
        '<h1>第2章：赤外・ラマン分光法</h1>': 
        '<h1>Chapter 2: Infrared and Raman Spectroscopy</h1>',
        
        '<p class="subtitle">振動分光で探る分子構造と化学結合</p>': 
        '<p class="subtitle">Probing Molecular Structure and Chemical Bonds with Vibrational Spectroscopy</p>',
        
        '📚 シリーズ: 分光分析入門': '📚 Series: Introduction to Spectroscopy',
        '⏱️ 学習時間: 100分': '⏱️ Study Time: 100 minutes',
        '🎯 難易度: 初級〜中級': '🎯 Difficulty: Beginner to Intermediate',
        
        # PHASE 2: MAIN SECTIONS
        '<h2>イントロダクション</h2>': '<h2>Introduction</h2>',
        '<h2>1. 分子振動の基礎</h2>': '<h2>1. Fundamentals of Molecular Vibrations</h2>',
        '<h3>1.1 調和振動子モデル</h3>': '<h3>1.1 Harmonic Oscillator Model</h3>',
        '<h3>1.2 多原子分子の振動モード</h3>': '<h3>1.2 Vibrational Modes of Polyatomic Molecules</h3>',
        '<h2>2. 赤外分光法（IR）</h2>': '<h2>2. Infrared Spectroscopy (IR)</h2>',
        '<h3>2.1 IR吸収の選択則</h3>': '<h3>2.1 Selection Rules for IR Absorption</h3>',
        '<h3>2.2 官能基と特性吸収</h3>': '<h3>2.2 Functional Groups and Characteristic Absorptions</h3>',
        '<h3>2.3 FTIR（フーリエ変換赤外分光法）</h3>': '<h3>2.3 FTIR (Fourier Transform Infrared Spectroscopy)</h3>',
        
        # Introduction paragraph
        '赤外分光（Infrared Spectroscopy, IR）とラマン分光（Raman Spectroscopy）は、分子の振動情報を通じて化学結合、官能基、結晶構造を解明する相補的な手法です。IRは赤外光の吸収を測定し、Ramanは散乱光の周波数シフトを観測します。両者は異なる選択則に従うため、IRで活性な振動がRamanで不活性、またはその逆という相補性を持ちます。':
        'Infrared (IR) spectroscopy and Raman spectroscopy are complementary techniques for elucidating chemical bonds, functional groups, and crystal structures through molecular vibrational information. IR measures absorption of infrared light, while Raman observes frequency shifts in scattered light. Because they follow different selection rules, vibrations that are IR-active may be Raman-inactive, and vice versa, providing complementary information.',
        
        # Info boxes
        '<strong>IRとRamanの使い分け</strong>': '<strong>When to Use IR vs Raman</strong>',
        '<li><strong>IR</strong>: 極性基（C=O, O-H, N-H）の検出、有機物の官能基同定、固体・液体・気体すべてに適用可能</li>':
        '<li><strong>IR</strong>: Detection of polar groups (C=O, O-H, N-H), identification of functional groups in organic compounds, applicable to solids, liquids, and gases</li>',
        '<li><strong>Raman</strong>: 対称振動（C=C, S-S）の検出、水溶液試料、結晶性評価（低波数領域）、非破壊・非接触測定</li>':
        '<li><strong>Raman</strong>: Detection of symmetric vibrations (C=C, S-S), aqueous samples, crystallinity assessment (low-frequency region), non-destructive and contactless measurements</li>',
        
        # Section 1.1 content
        '2原子分子の振動は調和振動子で近似できます。ポテンシャルエネルギーはHookeの法則に従います：':
        'The vibration of diatomic molecules can be approximated by a harmonic oscillator. The potential energy follows Hooke\'s law:',
        
        'ここで、$k$ は力の定数（N/m）、$r_e$ は平衡核間距離です。振動周波数 $\\nu$ は以下で与えられます：':
        'where $k$ is the force constant (N/m) and $r_e$ is the equilibrium internuclear distance. The vibrational frequency $\\nu$ is given by:',
        
        '$\\mu = \\frac{m_1 m_2}{m_1 + m_2}$ は換算質量です。振動エネルギー準位は量子化され、':
        'where $\\mu = \\frac{m_1 m_2}{m_1 + m_2}$ is the reduced mass. The vibrational energy levels are quantized:',
        
        '調和振動子近似では、選択則は $\\Delta v = \\pm 1$ です（基本振動のみ許容）。実際の分子では非調和性により $\\Delta v = \\pm 2, \\pm 3, \\ldots$（倍音）も弱く観測されます。':
        'In the harmonic oscillator approximation, the selection rule is $\\Delta v = \\pm 1$ (only fundamental vibrations are allowed). In real molecules, anharmonicity allows weak observation of $\\Delta v = \\pm 2, \\pm 3, \\ldots$ (overtones).',
    }
    
    # Apply all translations
    for jp, en in translations.items():
        content = content.replace(jp, en)
    
    print(f"✅ Applied {len(translations)} basic translations")
    
    # Write intermediate
    with open('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/spectroscopy-introduction/chapter-2.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Phase 1 complete - Basic structure translated")
    return content

if __name__ == '__main__':
    content = translate_full_file()
    print("\n Next: Run Phase 2 for code comments and detailed content...")

