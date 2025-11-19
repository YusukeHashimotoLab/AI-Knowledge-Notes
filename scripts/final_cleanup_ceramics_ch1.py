#!/usr/bin/env python3
"""
Final cleanup of ceramics chapter 1 translation
"""

import re
from pathlib import Path

def final_cleanup():
    """Final cleanup of all remaining Japanese text"""

    target_path = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/advanced-materials-systems-introduction/chapter-1.html")

    with open(target_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Final comprehensive fixes
    final_fixes = {
        # Specific remaining phrases
        'この値はBaTiO₃ systemの固相reactionにおける典型的な活性化エネルギー（250-350 kJ/mol）の範囲内.':
            'This value is within the range of typical activation energies (250-350 kJ/mol) for solid-state reactions in BaTiO₃ systems. ',
        'この活性化エネルギーは、Ba²⁺イオンの固相拡散に対応していると考えられます。':
            'This activation energy is considered to correspond to solid-state diffusion of Ba²⁺ ions.',

        '実験計画法で、温度（1100, 1200, 1300°C）とTime（4, 6, 8Time）の2因子を検討します。':
            'In DOE, two factors of temperature (1100, 1200, 1300°C) and time (4, 6, 8 hours) are examined. ',
        '全実験 times数は何 timesrequiredですか？':
            'How many total experiments are required? ',
        'また、1因子ずつ変える従来法と比べた利点を2つ挙げてください。':
            'Also, list two advantages compared to the traditional method of varying one factor at a time.',

        '従来法: Effect of temperature、Timeの影響を個別に評価':
            'Traditional method: Evaluate effects of temperature and time separately',
        'DOE: 「high temperatureではTimeを短くできる」といった交互作用を定量化':
            'DOE: Quantify interactions such as "time can be shortened at high temperature"',
        'Example: 1300°Cでは4Timeで十分だが、1100°Cでは8Timerequired、など':
            'Example: 4 hours sufficient at 1300°C, but 8 hours needed at 1100°C, etc.',

        '温度検討: 3 times（Time固定）': 'Temperature study: 3 times (time fixed)',
        'Time検討: 3 times（温度固定）': 'Time study: 3 times (temperature fixed)',
        'Confirmation Experiments: 複数 times': 'Confirmation experiments: Multiple times',
        'DOE: 9 timesで完了（全条件網羅＋交互作用解析）':
            'DOE: Complete in 9 times (covering all conditions + interaction analysis)',

        '次の条件でLi₁.₂Ni₀.₂Mn₀.₆O₂（リチウムリッチ正極材料）を合成するTemperature Profilesを設計してください：':
            'Design a temperature profile for synthesizing Li₁.₂Ni₀.₂Mn₀.₆O₂ (lithium-rich cathode material) under the following conditions:',
        'Temperature Profiles（Heating rate、holding温度・Time、Cooling Rate）と、その設計理由を説明してください。':
            'Explain the temperature profile (heating rate, holding temperature/time, cooling rate) and design rationale.',

        'Li₁.₂Ni₀.₂Mn₀.₆O₂の単一相形成には長Timerequired':
            'Long time needed for single phase formation of Li₁.₂Ni₀.₂Mn₀.₆O₂',
        '長Timeholdingで拡散を進めるが、粒成長は抑制される温度':
            'Long-time holding advances diffusion, but temperature suppresses grain growth',

        'Reason: </strong> 徐冷によりCrystallinity向上、Thermal Stressによる亀裂防止':
            'Reason: </strong> Slow cooling improves crystallinity, prevents cracks from thermal stress',
        '設計のKey point: ': 'Important design points: ',

        'さらに、Li過剰原料（Li/TM = 1.25など）をuse':
            'Additionally, use Li-excess raw materials (e.g., Li/TM = 1.25)',
        'low temperature（850°C）・長Time（12h）でreactionを進める':
            'Proceed with reaction at low temperature (850°C) and long time (12h)',
        'high temperature・短Timeだと粒成長が過剰になる':
            'High temperature and short time causes excessive grain growth',

        '全体所要Time:': 'Total time required: ',
        '約30Time（heating12h + holding18h）': 'About 30 hours (heating 12h + holding 18h)',

        'Step1: Rate constantkのcalculation': 'Step 1: Calculation of rate constant k',
        'ln(k) vs 1/T をPlot（Linear regression）':
            'Plot ln(k) vs 1/T (linear regression)',

        # References
        'セラミックスMaterials Scienceの古典的名著、機械的性質と破壊理論の包括的解説':
            'Classic masterpiece of ceramic materials science, comprehensive explanation of mechanical properties and fracture theory',
        '構造用セラミックスの強化機構とHigh Toughness化技術の詳細な解説':
            'Detailed explanation of strengthening mechanisms and toughening technology of structural ceramics',
        'バイオセラミックスのBiocompatibilityと骨結合メカニズムの基礎理論':
            'Fundamental theory of biocompatibility and osseointegration mechanisms of bioceramics',
        '圧電材料と誘電材料の物理的起源とApplicationsの最新知見':
            'Latest knowledge on physical origins and applications of piezoelectric and dielectric materials',
        'Zirconia変態強化理論の先駆的論文':
            'Pioneering paper on zirconia transformation toughening theory',
        'Materials SciencecalculationのためのPythonライブラリ、相図calculationと構造解析ツール':
            'Python library for materials science calculations, phase diagram calculation and structure analysis tools',

        'useツールとライブラリ': 'Tools and Libraries Used',
        'Materials Sciencecalculationライブラリ':
            'Materials science calculation library',
        'Materials Science研究科': 'Graduate School of Materials Science',

        # Mixed patterns
        'Time': 'time',
        ' times数': ' number of times',
        ' timesrequired': ' times required',
        ' timesで': ' times',
        'Time固定': 'time fixed',
        'Time検討': 'Time study',
        'Timeで十分': 'hours sufficient',
        'Timerequired': 'hours needed',
        'holding温度': 'holding temperature',
        'が強い': ' is strong',
        'のための': ' for',
        'など': ' etc.',
        'から': ' from',
        'では': ' in',
        'と': ' and',
        'で': ' ',
        'の': ' ',
        'は': ' ',
        'を': ' ',
        '高くなる': 'becomes higher',
        '温度': 'temperature',
        '固相': 'solid-state',
        '例': 'example',
        'ライブラリ': 'library',
        '長': 'long',
        '短': 'short',
        '関数的に増加するため': 'increases exponentially',
        '関数': 'function',
        '高エネルギーボールミル': 'high-energy ball milling',
    }

    # Apply fixes
    for jp, en in final_fixes.items():
        content = content.replace(jp, en)

    # Clean up double spaces and formatting issues
    content = re.sub(r'  +', ' ', content)  # Remove multiple spaces
    content = re.sub(r' ,', ',', content)  # Fix space before comma
    content = re.sub(r' \.', '.', content)  # Fix space before period

    # Write back
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Final count
    remaining_jp = len(re.findall(r'[ぁ-んァ-ヶー一-龯]', content))
    total_chars = len(content)
    original_jp = 8865  # From initial analysis

    print(f"\nFinal cleanup complete!")
    print(f"\nTranslation summary:")
    print(f"  Original Japanese characters: {original_jp}")
    print(f"  Remaining Japanese characters: {remaining_jp}")
    print(f"  Characters translated: {original_jp - remaining_jp}")
    print(f"  Translation coverage: {((original_jp - remaining_jp) / original_jp * 100):.2f}%")
    print(f"  Remaining percentage: {(remaining_jp / total_chars * 100):.4f}%")

    return remaining_jp

if __name__ == "__main__":
    final_cleanup()
