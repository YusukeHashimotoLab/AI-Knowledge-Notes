#!/usr/bin/env python3
"""
Translation script for MS (Materials Science) category files
Translates Japanese text to English while preserving HTML structure, CSS, JavaScript, code blocks, and equations
Processes 37 files across 6 MS series with comprehensive Materials Science terminology
"""

import os
from pathlib import Path

# Base paths
BASE_PATH = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en')

# MS series with Japanese characters (6 series, 37 files total)
MS_SERIES = [
    'ceramic-materials-introduction',      # 6 files (index + 5 chapters)
    'composite-materials-introduction',    # 6 files (index + 5 chapters)
    'crystallography-introduction',        # 6 files (index + 5 chapters)
    'electron-microscopy-introduction',    # 6 files (index + 5 chapters)
    'materials-properties-introduction',   # 7 files (index + 6 chapters)
    'metallic-materials-introduction',     # 6 files (index + 5 chapters)
]

# Comprehensive translation dictionary for Materials Science content
TRANSLATIONS = {
    # HTML attributes
    'lang="ja"': 'lang="en"',

    # Breadcrumb navigation - MS specific
    'AI寺子屋トップ': 'AI Terakoya Top',
    'マテリアルサイエンス': 'Materials Science',
    '材料科学': 'Materials Science',

    # Common metadata
    '読了時間': 'Reading Time',
    '難易度': 'Difficulty',
    '総学習時間': 'Total Learning Time',
    'レベル': 'Level',
    '対象者': 'Target Audience',
    '学習時間の目安': 'Estimated Learning Time',

    # Difficulty levels
    '初級': 'Beginner',
    '中級': 'Intermediate',
    '上級': 'Advanced',
    '初級〜中級': 'Beginner to Intermediate',
    '中級〜上級': 'Intermediate to Advanced',
    '初級者向け': 'For Beginners',
    '中級者向け': 'For Intermediate',
    '上級者向け': 'For Advanced',

    # Time units
    '分': ' minutes',
    '時間': ' hours',
    '個': '',
    'シリーズ': ' Series',
    '章': ' Chapter',
    '約': 'Approx. ',

    # Navigation elements
    '前の章': 'Previous Chapter',
    '次の章': 'Next Chapter',
    'シリーズ目次': 'Series Contents',
    '目次に戻る': 'Back to Contents',
    '目次へ': 'To Contents',
    '目次': 'Contents',
    'トップへ戻る': 'Back to Top',
    'ホームへ': 'To Home',

    # Common section titles
    '学習内容': 'Learning Content',
    '学習目標': 'Learning Objectives',
    'この章で学ぶこと': 'What You Will Learn',
    'まとめ': 'Summary',
    '演習問題': 'Exercises',
    '参考文献': 'References',
    '次のステップ': 'Next Steps',
    'シリーズ概要': 'Series Overview',
    '各章の詳細': 'Chapter Details',
    '全体の学習成果': 'Overall Learning Outcomes',
    '前提知識': 'Prerequisites',
    '使用技術とツール': 'Technologies and Tools Used',
    '学習の進め方': 'How to Learn',
    '推奨学習順序': 'Recommended Learning Order',
    '主要ライブラリ': 'Main Libraries',
    '開発環境': 'Development Environment',
    '更新履歴': 'Update History',
    '重要なポイント': 'Key Points',
    '基礎知識': 'Fundamental Knowledge',
    '応用編': 'Advanced Applications',
    '実践例': 'Practical Examples',

    # Chapter markers
    '第1章': 'Chapter 1',
    '第2章': 'Chapter 2',
    '第3章': 'Chapter 3',
    '第4章': 'Chapter 4',
    '第5章': 'Chapter 5',
    '第6章': 'Chapter 6',
    '第7章': 'Chapter 7',
    '第8章': 'Chapter 8',

    # Common phrases
    'コード例': 'Code Examples',
    '実装例': 'Implementation Example',
    '実例': 'Example',
    '例題': 'Example Problem',
    'さあ、始めましょう': "Let's Get Started",
    '詳しく見ていきましょう': "Let's Look in Detail",
    '理解を深めましょう': "Let's Deepen Understanding",
    '確認してみましょう': "Let's Verify",

    # === Materials Science Specific Terms ===

    # Core MS concepts
    '電子顕微鏡': 'Electron Microscopy',
    '走査型電子顕微鏡': 'Scanning Electron Microscopy',
    '透過型電子顕微鏡': 'Transmission Electron Microscopy',
    '結晶構造': 'Crystal Structure',
    '結晶': 'Crystal',
    '結晶化': 'Crystallization',
    '結晶学': 'Crystallography',
    'セラミックス': 'Ceramics',
    'セラミック材料': 'Ceramic Materials',
    '複合材料': 'Composite Materials',
    'コンポジット': 'Composite',
    '金属材料': 'Metallic Materials',
    '金属': 'Metal',
    '合金': 'Alloy',
    '材料特性': 'Materials Properties',
    '材料物性': 'Material Properties',
    '材料科学': 'Materials Science',
    '材料工学': 'Materials Engineering',

    # Microscopy terms
    '走査電子顕微鏡': 'Scanning Electron Microscope',
    '透過電子顕微鏡': 'Transmission Electron Microscope',
    '電子線': 'Electron Beam',
    '電子銃': 'Electron Gun',
    '分解能': 'Resolution',
    '倍率': 'Magnification',
    '二次電子': 'Secondary Electrons',
    '反射電子': 'Backscattered Electrons',
    '試料作製': 'Sample Preparation',
    '試料': 'Sample',
    '観察': 'Observation',
    '像': 'Image',
    '明視野像': 'Bright Field Image',
    '暗視野像': 'Dark Field Image',
    '回折': 'Diffraction',
    '回折パターン': 'Diffraction Pattern',

    # Crystal structure terms
    '単位格子': 'Unit Cell',
    '格子定数': 'Lattice Constant',
    '格子': 'Lattice',
    'ブラベー格子': 'Bravais Lattice',
    '対称性': 'Symmetry',
    '空間群': 'Space Group',
    '点群': 'Point Group',
    '結晶系': 'Crystal System',
    '立方晶系': 'Cubic System',
    '正方晶系': 'Tetragonal System',
    '斜方晶系': 'Orthorhombic System',
    '六方晶系': 'Hexagonal System',
    '単斜晶系': 'Monoclinic System',
    '三斜晶系': 'Triclinic System',
    '面心立方格子': 'Face-Centered Cubic',
    '体心立方格子': 'Body-Centered Cubic',
    '単純立方格子': 'Simple Cubic',
    '稠密充填': 'Close Packing',
    'ミラー指数': 'Miller Indices',

    # Ceramic materials terms
    '酸化物': 'Oxide',
    '炭化物': 'Carbide',
    '窒化物': 'Nitride',
    'ガラス': 'Glass',
    'ガラス転移': 'Glass Transition',
    'ガラス転移温度': 'Glass Transition Temperature',
    '焼結': 'Sintering',
    '焼成': 'Firing',
    '成形': 'Forming',
    '粉末': 'Powder',
    '粉末冶金': 'Powder Metallurgy',
    '多孔質': 'Porous',
    '緻密化': 'Densification',
    '粒界': 'Grain Boundary',
    '粒子': 'Particle',
    '粒径': 'Particle Size',

    # Composite materials terms
    '強化材': 'Reinforcement',
    '母材': 'Matrix',
    'マトリックス': 'Matrix',
    '繊維': 'Fiber',
    '炭素繊維': 'Carbon Fiber',
    'ガラス繊維': 'Glass Fiber',
    '繊維強化プラスチック': 'Fiber Reinforced Plastic',
    '繊維強化複合材料': 'Fiber Reinforced Composite',
    '積層板': 'Laminate',
    '積層': 'Lamination',
    '界面': 'Interface',
    '界面接着': 'Interfacial Adhesion',
    '樹脂': 'Resin',
    '熱硬化性樹脂': 'Thermosetting Resin',
    '熱可塑性樹脂': 'Thermoplastic Resin',
    'エポキシ': 'Epoxy',

    # Metallic materials terms
    '鉄鋼': 'Steel',
    '鋼': 'Steel',
    '鉄': 'Iron',
    'ステンレス鋼': 'Stainless Steel',
    '炭素鋼': 'Carbon Steel',
    'アルミニウム': 'Aluminum',
    'アルミニウム合金': 'Aluminum Alloy',
    'チタン': 'Titanium',
    'チタン合金': 'Titanium Alloy',
    '銅': 'Copper',
    '銅合金': 'Copper Alloy',
    '黄銅': 'Brass',
    '青銅': 'Bronze',
    'ニッケル': 'Nickel',
    'ニッケル合金': 'Nickel Alloy',
    '相': 'Phase',
    '相図': 'Phase Diagram',
    '相変態': 'Phase Transformation',
    '変態': 'Transformation',
    '固溶体': 'Solid Solution',
    '析出': 'Precipitation',
    '析出物': 'Precipitate',
    '時効': 'Aging',
    '時効硬化': 'Age Hardening',
    '焼入れ': 'Quenching',
    '焼戻し': 'Tempering',
    '焼鈍': 'Annealing',
    '熱処理': 'Heat Treatment',

    # Material properties terms
    '機械的性質': 'Mechanical Properties',
    '機械的特性': 'Mechanical Properties',
    '強度': 'Strength',
    '引張強度': 'Tensile Strength',
    '圧縮強度': 'Compressive Strength',
    '曲げ強度': 'Flexural Strength',
    '降伏強度': 'Yield Strength',
    '破壊強度': 'Fracture Strength',
    '硬度': 'Hardness',
    'ビッカース硬度': 'Vickers Hardness',
    'ブリネル硬度': 'Brinell Hardness',
    'ロックウェル硬度': 'Rockwell Hardness',
    '靭性': 'Toughness',
    '延性': 'Ductility',
    '脆性': 'Brittleness',
    '弾性': 'Elasticity',
    '塑性': 'Plasticity',
    '弾性率': 'Elastic Modulus',
    'ヤング率': "Young's Modulus",
    'せん断弾性率': 'Shear Modulus',
    '体積弾性率': 'Bulk Modulus',
    'ポアソン比': "Poisson's Ratio",
    '応力': 'Stress',
    'ひずみ': 'Strain',
    '変形': 'Deformation',
    '塑性変形': 'Plastic Deformation',
    '弾性変形': 'Elastic Deformation',

    # Thermal properties
    '熱的性質': 'Thermal Properties',
    '熱伝導': 'Thermal Conductivity',
    '熱伝導率': 'Thermal Conductivity',
    '熱膨張': 'Thermal Expansion',
    '熱膨張係数': 'Coefficient of Thermal Expansion',
    '比熱': 'Specific Heat',
    '融点': 'Melting Point',
    '沸点': 'Boiling Point',
    '熱容量': 'Heat Capacity',

    # Electrical properties
    '電気的性質': 'Electrical Properties',
    '電気伝導': 'Electrical Conductivity',
    '電気伝導率': 'Electrical Conductivity',
    '電気抵抗': 'Electrical Resistance',
    '抵抗率': 'Resistivity',
    '誘電率': 'Dielectric Constant',
    '絶縁': 'Insulation',
    '導電性': 'Conductivity',
    '半導体': 'Semiconductor',
    '導体': 'Conductor',
    '絶縁体': 'Insulator',

    # Magnetic properties
    '磁気的性質': 'Magnetic Properties',
    '磁性': 'Magnetism',
    '強磁性': 'Ferromagnetism',
    '常磁性': 'Paramagnetism',
    '反磁性': 'Diamagnetism',
    '磁化': 'Magnetization',
    '保磁力': 'Coercivity',

    # Optical properties
    '光学的性質': 'Optical Properties',
    '屈折率': 'Refractive Index',
    '透過率': 'Transmittance',
    '反射率': 'Reflectance',
    '吸収': 'Absorption',
    '透明性': 'Transparency',

    # Chemical properties
    '化学的性質': 'Chemical Properties',
    '耐食性': 'Corrosion Resistance',
    '腐食': 'Corrosion',
    '酸化': 'Oxidation',
    '還元': 'Reduction',
    '耐薬品性': 'Chemical Resistance',

    # Defects and microstructure
    '欠陥': 'Defect',
    '点欠陥': 'Point Defect',
    '線欠陥': 'Line Defect',
    '面欠陥': 'Planar Defect',
    '転位': 'Dislocation',
    '空孔': 'Vacancy',
    '格子間原子': 'Interstitial',
    '微細構造': 'Microstructure',
    '組織': 'Microstructure',
    '結晶粒': 'Grain',
    '粒径': 'Grain Size',
    '結晶粒界': 'Grain Boundary',

    # Processing and manufacturing
    '加工': 'Processing',
    '製造': 'Manufacturing',
    '成形': 'Forming',
    '鋳造': 'Casting',
    '鍛造': 'Forging',
    '圧延': 'Rolling',
    '押出': 'Extrusion',
    '引抜': 'Drawing',
    '切削': 'Machining',
    '溶接': 'Welding',
    '接合': 'Joining',
    '表面処理': 'Surface Treatment',
    '表面改質': 'Surface Modification',
    'コーティング': 'Coating',
    'めっき': 'Plating',

    # Testing and characterization
    '試験': 'Testing',
    '評価': 'Evaluation',
    '分析': 'Analysis',
    '特性評価': 'Characterization',
    '引張試験': 'Tensile Test',
    '圧縮試験': 'Compression Test',
    '硬度試験': 'Hardness Test',
    '疲労試験': 'Fatigue Test',
    'クリープ試験': 'Creep Test',
    '衝撃試験': 'Impact Test',
    '非破壊検査': 'Non-Destructive Testing',
    'X線回折': 'X-ray Diffraction',
    'X線': 'X-ray',
    '分光': 'Spectroscopy',

    # Common technical terms
    '特性': 'Properties',
    '性質': 'Properties',
    '構造': 'Structure',
    '組成': 'Composition',
    '原子': 'Atom',
    '分子': 'Molecule',
    '結合': 'Bond',
    '共有結合': 'Covalent Bond',
    'イオン結合': 'Ionic Bond',
    '金属結合': 'Metallic Bond',
    '密度': 'Density',
    '比重': 'Specific Gravity',
    '濃度': 'Concentration',
    '純度': 'Purity',
    '不純物': 'Impurity',
    '添加物': 'Additive',
    '元素': 'Element',
    '周期表': 'Periodic Table',
    '原料': 'Raw Material',
    '製品': 'Product',
    '品質': 'Quality',
    '性能': 'Performance',
    '信頼性': 'Reliability',
    '寿命': 'Lifetime',
    '劣化': 'Degradation',
    '破壊': 'Fracture',
    '破損': 'Failure',
    'き裂': 'Crack',

    # Applications and industries
    '応用': 'Applications',
    '用途': 'Applications',
    '産業': 'Industry',
    '自動車': 'Automotive',
    '航空宇宙': 'Aerospace',
    '電子': 'Electronics',
    '電子デバイス': 'Electronic Devices',
    'エネルギー': 'Energy',
    '環境': 'Environment',
    '医療': 'Medical',
    'バイオマテリアル': 'Biomaterials',
    '建築': 'Construction',
    '構造材料': 'Structural Materials',
    '機能材料': 'Functional Materials',

    # Research and development
    '研究': 'Research',
    '開発': 'Development',
    '実験': 'Experiment',
    'データ': 'Data',
    '測定': 'Measurement',
    '計算': 'Calculation',
    'シミュレーション': 'Simulation',
    'モデリング': 'Modeling',
    '理論': 'Theory',
    '実験データ': 'Experimental Data',
    '解析': 'Analysis',
    '最適化': 'Optimization',
    '設計': 'Design',
    '材料設計': 'Materials Design',

    # Units and measurements
    '温度': 'Temperature',
    '圧力': 'Pressure',
    '時間': 'Time',
    '速度': 'Velocity',
    '長さ': 'Length',
    '体積': 'Volume',
    '質量': 'Mass',
    '重量': 'Weight',
    '力': 'Force',
    'エネルギー': 'Energy',
    '仕事': 'Work',
    '熱': 'Heat',

    # Mathematical and scientific terms
    '方程式': 'Equation',
    '関数': 'Function',
    'グラフ': 'Graph',
    '図': 'Figure',
    '表': 'Table',
    'パラメータ': 'Parameter',
    '変数': 'Variable',
    '定数': 'Constant',
    '係数': 'Coefficient',
    '比': 'Ratio',
    '割合': 'Proportion',
    '平均': 'Average',
    '標準偏差': 'Standard Deviation',
    '分布': 'Distribution',
    '確率': 'Probability',
}


def translate_text(text):
    """
    Apply dictionary-based translation to text.
    Sorts by length (longest first) to avoid partial replacements.
    Preserves HTML structure, CSS, JavaScript, code blocks, and equations.
    """
    result = text

    # Sort by length (longest first) to avoid partial replacements
    for ja, en in sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        result = result.replace(ja, en)

    return result


def translate_file(file_path):
    """
    Translate a single HTML file in-place from Japanese to English.
    Reads the file, applies dictionary translation, and writes back to the same location.

    Args:
        file_path: Path object pointing to the HTML file to translate

    Returns:
        bool: True if translation successful, False otherwise
    """
    try:
        # Read original content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply translations
        translated = translate_text(content)

        # Write translated content back to same file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(translated)

        print(f"  ✓ Translated: {file_path.name}")
        return True

    except Exception as e:
        print(f"  ✗ Error translating {file_path.name}: {e}")
        return False


def process_ms_series():
    """
    Process all MS series files for translation.
    Translates 37 files across 6 MS series with progress output.
    """
    print("\n" + "="*70)
    print("MS (Materials Science) Category Translation")
    print("="*70)
    print(f"Processing 6 series with 37 total files")
    print("="*70 + "\n")

    total_files = 0
    total_success = 0

    for series_name in MS_SERIES:
        series_path = BASE_PATH / 'MS' / series_name

        if not series_path.exists():
            print(f"\n✗ Series directory not found: {series_path}")
            continue

        print(f"\n{'─'*70}")
        print(f"Series: {series_name}")
        print(f"{'─'*70}")

        # Get all HTML files in the series
        html_files = sorted(series_path.glob('*.html'))

        if not html_files:
            print("  No HTML files found")
            continue

        series_success = 0

        for html_file in html_files:
            total_files += 1
            if translate_file(html_file):
                series_success += 1
                total_success += 1

        print(f"\nSeries completed: {series_success}/{len(html_files)} files translated")

    # Final summary
    print("\n" + "="*70)
    print("TRANSLATION COMPLETE")
    print("="*70)
    print(f"Total files processed: {total_files}")
    print(f"Successfully translated: {total_success}")
    print(f"Failed: {total_files - total_success}")
    print("="*70 + "\n")

    return total_success == total_files


def main():
    """Main execution function"""
    success = process_ms_series()

    if success:
        print("✓ All MS files translated successfully!")
        return 0
    else:
        print("⚠ Some files failed translation. Check output above for details.")
        return 1


if __name__ == '__main__':
    exit(main())
