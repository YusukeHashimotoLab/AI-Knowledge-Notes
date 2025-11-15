#!/usr/bin/env python3
"""
Fix remaining Japanese text in ceramics chapter 1
"""

import re
from pathlib import Path

def fix_remaining_japanese():
    """Fix all remaining Japanese text"""

    target_path = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/advanced-materials-systems-introduction/chapter-1.html")

    with open(target_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Additional translations for remaining text
    fixes = {
        # Mixed text cleanup
        '酸化物セラミックスの代表格。': 'Representative of oxide ceramics. ',
        '優れた耐摩耗性、Biocompatibilityにより、切削工具・人工関節にuse。':
            'excellent wear resistance, and biocompatibility, used in cutting tools and artificial joints. ',
        '製造コストが低く最も広く普及。': 'Most widely used due to low manufacturing cost.',

        '共有結合性が強く、1400°Cまで高強度を維持。':
            'Strong covalent bonding maintains high strength up to 1400°C. ',
        'ガスタービン部品・ベアリングなどの高温構造材料としてuse。':
            'Used as high-temperature structural material for gas turbine components and bearings. ',
        '熱衝撃抵抗性も優れる。': 'Also exhibits excellent thermal shock resistance.',

        'セラミックスは高強度・High Hardnessを持つ一方で、':
            'While ceramics possess high strength and high hardness, ',
        'が最大の欠点.': ' is the major drawback. ',
        '微小な欠陥（気孔、亀裂）が応力集中点となり、突発的な破壊を引き起こします（Griffith理論）。':
            'Microscopic defects (pores, cracks) become stress concentration points, causing catastrophic fracture (Griffith theory). ',
        '破壊靭性は金属の1/10以下.': 'Fracture toughness is less than 1/10 that of metals. ',
        'このため、High Toughness化技術が重要な研究課題となっています。':
            'Therefore, toughening technology is an important research topic.',

        'High Toughness化メカニズム': 'Toughening Mechanisms',

        'PZTは鉛（Pb）を60wt%以上含むため、欧州RoHS規制でuse制限があります。':
            'PZT contains more than 60 wt% lead (Pb), subject to usage restrictions under European RoHS regulations. ',
        '鉛フリー代替材料として、BaTiO₃系、(K,Na)NbO₃系、BiFeO₃系が研究されていますが、PZTの性能には及びません（d₃₃ = 100-300 pC/N）。':
            'Lead-free alternatives such as BaTiO₃-based, (K,Na)NbO₃-based, and BiFeO₃-based materials are being researched, but do not match PZT performance (d₃₃ = 100-300 pC/N). ',
        '圧電デバイスは医療機器等の適用除外品目ですが、長期的には代替材料開発が必要.':
            'While piezoelectric devices are exempt items for medical equipment, alternative material development is necessary in the long term.',

        'としてuseされます。': '.',
        'Spontaneous Polarizationが外部電場により反転可能な性質':
            'Property where spontaneous polarization can be reversed by external electric field',
        'Dielectric constant peaks at this temperature':
            'Dielectric constant peaks at this temperature',
        'BaTiO₃ベースのMLCCは電子機器の小型化・高性能化の鍵となる材料.':
            'BaTiO₃-based MLCCs are key materials for miniaturization and performance enhancement of electronic devices.',

        'を持つため、トランスフォーマー・インダクタ・電波吸収体に広くuseされます。':
            ', widely used in transformers, inductors, and electromagnetic wave absorbers.',
        'フェライトの種類とApplications': 'Types and Applications of Ferrites',
        'High-Frequency Characteristicsに優れる（GHz帯）、EMI対策部品用':
            'Excellent high-frequency characteristics (GHz band), for EMI countermeasure components',
        'モーター、スピーカー、磁気記録媒体にuse':
            'Used in motors, speakers, magnetic recording media',
        'することで発現します（フェリ磁性）。':
            ' in the spinel structure (AB₂O₄) (ferrimagnetism). ',
        'Mn-Zn FerriteではMn²⁺とFe³⁺の磁気モーメントが部分的に打ち消し合うため、全体としての磁化は小さくなりますが、高透磁率が実現されます。':
            'In Mn-Zn ferrites, the magnetic moments of Mn²⁺ and Fe³⁺ partially cancel each other, resulting in small overall magnetization but achieving high permeability.',

        # Code comments
        '# Example 1: Arrhenius式シミュレーション':
            '# Example 1: Arrhenius Equation Simulation',
        '# 対数Plot（ArrheniusPlot）':
            '# Logarithmic plot (Arrhenius plot)',
        '# 1/T vs ln(D) Plot（直線関係）':
            '# 1/T vs ln(D) plot (linear relationship)',

        '# Example 2: Jander式によるConversion計算':
            '# Example 2: Conversion Calculation using Jander Equation',
        '"""Jander式': '"""Jander equation',
        '# Time配列（0-50Time）': '# Time array (0-50 hours)',
        '# 50%反応に要するTimeを計算':
            '# Calculate time required for 50% conversion',
        'print("\\n50%反応に要するTime:")':
            'print("\\nTime required for 50% conversion:")',
        '# 50%反応に要するTime:':
            '# Time required for 50% conversion:',

        '# Example 3: Kissinger法によるCalculate activation energy':
            '# Example 3: Calculate Activation Energy using Kissinger Method',
        '# Tp: ピークTemperature [K]':
            '# Tp: Peak temperature [K]',
        'ピークTemperature [K]': 'Peak temperature [K]',

        '固相反応におけるTemperature Profilesは、反応の成功を左右する最も重要な制御パラメータ.':
            'The temperature profile in solid-state reactions is the most important control parameter determining reaction success. ',
        '以下の3要素を適切に設計する必要があります：':
            'The following three elements must be properly designed:',

        '保持Time<br/>Holding Time': 'Holding Time',
        '試料内部と表面の温度差が大きいとThermal Stressが発生し、亀裂の原因に':
            'Large temperature differences between sample interior and surface generate thermal stress, causing cracks',

        '⚠️ 実例: BaCO₃のDecomposition Reactions':
            '⚠️ Example: Decomposition Reaction of BaCO₃',
        'BaTiO₃合成では800-900°Cで BaCO₃ → BaO + CO₂ の分解が起こります。':
            'In BaTiO₃ synthesis, decomposition BaCO₃ → BaO + CO₂ occurs at 800-900°C. ',
        'Heating rateが20°C/min以上だと、CO₂が急激に放出され、試料が破裂することがあります。':
            'At heating rates above 20°C/min, CO₂ is released rapidly and samples may rupture. ',
        '推奨Heating rateは5°C/min以下.':
            'Recommended heating rate is 5°C/min or below.',

        '保持Time（Holding Time）': 'Holding Time',
        '必要な保持Timeは以下の式で推定できます：':
            'Required holding time can be estimated from the following equation:',
        '典型的な保持Time：': 'Typical holding times:',
        '低温反応（<1000°C）: 12-24Time':
            'Low-temperature reactions (<1000°C): 12-24 hours',
        '中温反応（1000-1300°C）: 4-8Time':
            'Medium-temperature reactions (1000-1300°C): 4-8 hours',
        '高温反応（>1300°C）: 2-4Time':
            'High-temperature reactions (>1300°C): 2-4 hours',

        'Heating rateより遅め）': 'slower than heating rate)',
        '徐冷はCrystallinityを向上': 'Slow cooling improves crystallinity',

        '# Example 4: Temperature Profiles最適化':
            '# Example 4: Temperature Profile Optimization',
        'Time配列 [min]': 'Time array [min]',
        '保持Time [min]': 'Holding time [min]',
        '加熱Time': 'Heating time',
        'Room temperature以下にはならない':
            'Does not go below room temperature',
        '簡易積分（微小TimeでのReaction Progress）':
            'Simple integration (reaction progress in small time steps)',

        '異なるHeating rateでの比較':
            'Comparison at different heating rates',
        '各Heating rateでの95%反応到達Timeを計算':
            'Calculate time to reach 95% conversion at each heating rate',
        '95%反応到達Timeの比較:':
            'Comparison of time to reach 95% conversion:',

        # More specific patterns
        'にuse': ' used',
        'use制限': ' usage restrictions',
        '式': ' equation',
        '配列': ' array',
        '保持': 'holding',
        '反応到達': ' to reach conversion',
        '反応に要する': ' required for reaction',
        'を計算': ' calculate',
        '例:': 'Example:',
        'ステップ': 'Step',
        '高温': 'high temperature',
        '低温': 'low temperature',
        '中温': 'medium temperature',
        '短時間': 'short time',
        '長時間': 'long time',
        '系': ' system',
        '温度と': 'temperature and',
        '最大': 'maximum',
        '推奨': 'recommended',
        '必要': 'required',
        '実際の': 'actual',
        '回': ' times',
        '反応速度モデルの': ' of reaction kinetics models',
        '反応': 'reaction',
        '加熱': 'heating',
        '計算': 'calculation',

        # Table headers that might have been missed
        'β (K/min)': 'β (K/min)',
        'Tp (K)': 'Tp (K)',
        'ln(β/Tp²)': 'ln(β/Tp²)',
        '1000/Tp (K⁻¹)': '1000/Tp (K⁻¹)',
    }

    # Apply fixes
    for jp, en in fixes.items():
        content = content.replace(jp, en)

    # Write back
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Count remaining Japanese
    remaining_jp = len(re.findall(r'[ぁ-んァ-ヶー一-龯]', content))
    total_chars = len(content)

    print(f"Fix complete!")
    print(f"Remaining Japanese characters: {remaining_jp}")
    print(f"Remaining Japanese percentage: {(remaining_jp / total_chars * 100):.2f}%")

    return remaining_jp

if __name__ == "__main__":
    fix_remaining_japanese()
