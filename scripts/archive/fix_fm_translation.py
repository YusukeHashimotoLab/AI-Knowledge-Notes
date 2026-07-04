#!/usr/bin/env python3
"""
Enhanced comprehensive translation script for FM (Fundamentals of Mathematics) category
Handles Japanese particles, verbs, and mixed-language patterns
Includes post-processing for clean output
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# Base paths
BASE_DIR = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en')
FILE_LIST_PATH = BASE_DIR / 'JAPANESE_CHARACTER_FILES_LIST.txt'

# FM series names
FM_SERIES = [
    'quantum-field-theory-introduction',
    'quantum-mechanics',
    'probability-stochastic-processes',
    'classical-statistical-mechanics',
]

# Comprehensive translation dictionary for FM content
# IMPORTANT: Longer phrases FIRST to avoid partial replacements
TRANSLATIONS = {
    # HTML attributes
    'lang="ja"': 'lang="en"',

    # ===== LONGER PHRASES FIRST (Mixed Language Patterns) =====

    # Common mixed-language patterns with particles
    'Statistical Mechanicsの基礎': 'Foundations of Statistical Mechanics',
    'Quantum Mechanicsの': 'Quantum Mechanics',
    'Field Theoryの': 'Field Theory',
    'Probabilityの': 'Probability',
    'Stochastic Processesの': 'Stochastic Processes',

    # Verb endings in mixed contexts
    'を実行します': 'executes',
    'を計算します': 'calculates',
    'を示します': 'shows',
    'を表します': 'represents',
    'を得ます': 'obtains',
    'を使います': 'uses',
    'を考えます': 'considers',
    'を定義します': 'defines',
    'を求めます': 'finds',
    'を導きます': 'derives',
    'を解きます': 'solves',

    # Common phrase patterns
    'することができます': 'can be done',
    'することが可能です': 'is possible',
    'する必要があります': 'is necessary',
    'する場合': 'when doing',
    'した場合': 'when done',
    'したとき': 'when',
    'するとき': 'when',
    'するために': 'in order to',
    'したがって': 'therefore',
    'に対して': 'for',
    'について': 'about',
    'に関して': 'regarding',
    'によって': 'by',
    'において': 'in',
    'における': 'in',
    'に基づいて': 'based on',
    'に応じて': 'according to',
    'のような': 'like',
    'のように': 'as',
    'のため': 'because of',
    'のもと': 'under',
    'のとき': 'when',

    # Numerical patterns with particles
    'の状態': ' states',
    'の場合': ' case',
    'の値': ' value',
    'の数': ' number',
    'の式': ' equation',
    'の関数': ' function',
    'の演算子': ' operator',
    'の行列': ' matrix',
    'の確率': ' probability',
    'の分布': ' distribution',
    'の平均': ' average',
    'の和': ' sum',
    'の積': ' product',
    'の差': ' difference',
    'の比': ' ratio',
    'の次元': ' dimension',
    'の成分': ' component',
    'の要素': ' element',
    'の項': ' term',
    'の係数': ' coefficient',

    # Adjective patterns
    'な状態': ' state',
    'な場合': ' case',
    'な形': ' form',
    'な方法': ' method',
    'な関数': ' function',
    'な解': ' solution',
    'な条件': ' condition',
    'な性質': ' property',
    'な特徴': ' feature',

    # ===== BREADCRUMB NAVIGATION =====
    'AI寺子屋トップ': 'AI Terakoya Top',
    '基礎数学': 'Fundamentals of Mathematics',
    '数学・物理道場': 'Mathematics & Physics Dojo',

    # ===== COMMON METADATA =====
    '読了時間': 'Reading Time',
    '難易度': 'Difficulty',
    '総学習時間': 'Total Learning Time',
    'レベル': 'Level',

    # ===== DIFFICULTY LEVELS =====
    '初級': 'Beginner',
    '中級': 'Intermediate',
    '上級': 'Advanced',
    '初級〜中級': 'Beginner to Intermediate',
    '中級〜上級': 'Intermediate to Advanced',

    # ===== TIME UNITS =====
    '分': ' minutes',
    '時間': ' hours',
    '個': '',
    'シリーズ': ' Series',
    '章': ' Chapter',

    # ===== NAVIGATION ELEMENTS =====
    '前の章': 'Previous Chapter',
    '次の章': 'Next Chapter',
    'シリーズ目次': 'Series Contents',
    '目次に戻る': 'Back to Contents',
    '目次へ': 'To Contents',
    '前のセクション': 'Previous Section',
    '次のセクション': 'Next Section',

    # ===== COMMON SECTION TITLES =====
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

    # ===== CHAPTER MARKERS =====
    '第1章': 'Chapter 1',
    '第2章': 'Chapter 2',
    '第3章': 'Chapter 3',
    '第4章': 'Chapter 4',
    '第5章': 'Chapter 5',
    '第6章': 'Chapter 6',

    # ===== COMMON PHRASES =====
    'コード例': 'Code Examples',
    '実装例': 'Implementation Example',
    '実例': 'Example',
    '例題': 'Example Problem',
    'さあ、始めましょう': "Let's Get Started",

    # ===== PHYSICS TERMS - QUANTUM MECHANICS =====
    '量子': 'Quantum',
    '量子力学': 'Quantum Mechanics',
    '量子力学入門': 'Introduction to Quantum Mechanics',
    '波動力学': 'Wave Mechanics',
    '波動力学の基礎': 'Foundations of Wave Mechanics',
    '波動関数': 'Wave Function',
    'シュレディンガー方程式': 'Schrödinger Equation',
    'エルミート演算子': 'Hermitian Operator',
    '固有値': 'Eigenvalue',
    '固有状態': 'Eigenstate',
    '固有関数': 'Eigenfunction',
    '固有値問題': 'Eigenvalue Problem',
    '調和振動子': 'Harmonic Oscillator',
    'トンネル効果': 'Tunneling Effect',
    '水素原子': 'Hydrogen Atom',
    '原子軌道': 'Atomic Orbital',
    '角運動量': 'Angular Momentum',
    'スピン': 'Spin',
    '摂動論': 'Perturbation Theory',
    '変分法': 'Variational Method',
    '演算子': 'Operator',
    '期待値': 'Expectation Value',
    '不確定性原理': 'Uncertainty Principle',
    '交換関係': 'Commutation Relation',
    '生成演算子': 'Creation Operator',
    '消滅演算子': 'Annihilation Operator',
    'はしご演算子': 'Ladder Operator',
    'ラゲール多項式': 'Laguerre Polynomial',
    'エルミート多項式': 'Hermite Polynomial',
    '球面調和関数': 'Spherical Harmonics',
    '動径方程式': 'Radial Equation',
    '動径関数': 'Radial Function',
    '電子雲': 'Electron Cloud',
    '透過係数': 'Transmission Coefficient',
    '反射係数': 'Reflection Coefficient',
    '井戸型ポテンシャル': 'Square-Well Potential',
    '障壁透過': 'Barrier Penetration',
    '走査トンネル顕微鏡': 'Scanning Tunneling Microscope',
    '基底状態': 'Ground State',
    '励起状態': 'Excited State',
    'パウリ行列': 'Pauli Matrix',
    'スピン軌道相互作用': 'Spin-Orbit Interaction',
    '磁性': 'Magnetism',
    '磁性材料': 'Magnetic Materials',
    '縮退': 'Degeneracy',
    '非縮退': 'Non-degenerate',
    '縮退系': 'Degenerate System',
    '変分原理': 'Variational Principle',
    '分子軌道': 'Molecular Orbital',
    'ハートリー・フォック法': 'Hartree-Fock Method',

    # ===== PHYSICS TERMS - QUANTUM FIELD THEORY =====
    '場の量子論': 'Quantum Field Theory',
    '場の理論': 'Field Theory',
    'ラグランジアン': 'Lagrangian',
    'ハミルトニアン': 'Hamiltonian',
    'ラグランジュ形式': 'Lagrangian Formalism',
    'ハミルトン形式': 'Hamiltonian Formalism',
    '作用': 'Action',
    '正準量子化': 'Canonical Quantization',
    '経路積分': 'Path Integral',
    'ファインマン図': 'Feynman Diagram',
    '相互作用': 'Interaction',
    'ゲージ理論': 'Gauge Theory',
    'ゲージ対称性': 'Gauge Symmetry',
    '対称性': 'Symmetry',
    '対称性の破れ': 'Symmetry Breaking',
    '自発的対称性の破れ': 'Spontaneous Symmetry Breaking',
    'ノーザーの定理': "Noether's Theorem",
    '保存則': 'Conservation Law',
    '場の演算子': 'Field Operator',
    '生成場': 'Creation Field',
    '消滅場': 'Annihilation Field',
    '真空状態': 'Vacuum State',
    '励起': 'Excitation',
    '粒子': 'Particle',
    '反粒子': 'Antiparticle',
    '散乱': 'Scattering',
    '散乱振幅': 'Scattering Amplitude',
    '断面積': 'Cross Section',
    '繰り込み': 'Renormalization',
    '繰り込み群': 'Renormalization Group',

    # ===== MATHEMATICS TERMS - PROBABILITY & STOCHASTIC PROCESSES =====
    '確率': 'Probability',
    '確率論': 'Probability Theory',
    '確率過程': 'Stochastic Process',
    '確率・確率過程入門': 'Introduction to Probability & Stochastic Processes',
    '確率変数': 'Random Variable',
    '確率分布': 'Probability Distribution',
    '確率密度': 'Probability Density',
    '確率密度関数': 'Probability Density Function',
    '累積分布関数': 'Cumulative Distribution Function',
    '期待値': 'Expected Value',
    '分散': 'Variance',
    '標準偏差': 'Standard Deviation',
    '共分散': 'Covariance',
    '相関': 'Correlation',
    '相関係数': 'Correlation Coefficient',
    '独立': 'Independent',
    '独立性': 'Independence',
    '同時分布': 'Joint Distribution',
    '周辺分布': 'Marginal Distribution',
    '条件付き確率': 'Conditional Probability',
    '条件付き期待値': 'Conditional Expectation',
    'ベイズの定理': "Bayes' Theorem",
    '大数の法則': 'Law of Large Numbers',
    '中心極限定理': 'Central Limit Theorem',
    'マルコフ過程': 'Markov Process',
    'マルコフ連鎖': 'Markov Chain',
    'ポアソン過程': 'Poisson Process',
    'ブラウン運動': 'Brownian Motion',
    '拡散過程': 'Diffusion Process',
    'ウィナー過程': 'Wiener Process',
    '伊藤の補題': "Itô's Lemma",
    '確率微分方程式': 'Stochastic Differential Equation',
    '定常過程': 'Stationary Process',
    'エルゴード性': 'Ergodicity',
    '自己相関': 'Autocorrelation',
    'スペクトル': 'Spectrum',
    'スペクトル密度': 'Spectral Density',
    'モンテカルロ法': 'Monte Carlo Method',
    'マルコフ連鎖モンテカルロ法': 'Markov Chain Monte Carlo',
    'メトロポリス法': 'Metropolis Algorithm',
    'ギブスサンプリング': 'Gibbs Sampling',

    # ===== MATHEMATICS TERMS - STATISTICAL MECHANICS =====
    '統計力学': 'Statistical Mechanics',
    '古典統計力学': 'Classical Statistical Mechanics',
    '古典統計力学入門': 'Introduction to Classical Statistical Mechanics',
    '統計': 'Statistics',
    '熱力学': 'Thermodynamics',
    '熱平衡': 'Thermal Equilibrium',
    '平衡状態': 'Equilibrium State',
    'ミクロカノニカル集団': 'Microcanonical Ensemble',
    'カノニカル集団': 'Canonical Ensemble',
    'グランドカノニカル集団': 'Grand Canonical Ensemble',
    '分配関数': 'Partition Function',
    'ボルツマン分布': 'Boltzmann Distribution',
    'ボルツマン因子': 'Boltzmann Factor',
    'ギブス分布': 'Gibbs Distribution',
    'エントロピー': 'Entropy',
    '自由エネルギー': 'Free Energy',
    'ヘルムホルツ自由エネルギー': 'Helmholtz Free Energy',
    'ギブス自由エネルギー': 'Gibbs Free Energy',
    '化学ポテンシャル': 'Chemical Potential',
    '相転移': 'Phase Transition',
    '臨界現象': 'Critical Phenomena',
    '臨界点': 'Critical Point',
    '秩序パラメータ': 'Order Parameter',
    'イジングモデル': 'Ising Model',
    '平均場理論': 'Mean Field Theory',
    '相関関数': 'Correlation Function',
    '応答関数': 'Response Function',
    '揺動散逸定理': 'Fluctuation-Dissipation Theorem',
    '統計集団': 'Statistical Ensemble',
    '状態密度': 'Density of States',
    '理想気体': 'Ideal Gas',
    'フェルミ気体': 'Fermi Gas',
    'ボース気体': 'Bose Gas',
    'フェルミ分布': 'Fermi Distribution',
    'ボース分布': 'Bose Distribution',
    'フェルミ準位': 'Fermi Level',
    'ボース・アインシュタイン凝縮': 'Bose-Einstein Condensation',

    # ===== GENERAL MATHEMATICS TERMS =====
    '微積分': 'Calculus',
    '微分': 'Derivative',
    '積分': 'Integral',
    '偏微分': 'Partial Derivative',
    '偏微分方程式': 'Partial Differential Equation',
    '微分方程式': 'Differential Equation',
    '常微分方程式': 'Ordinary Differential Equation',
    'ベクトル': 'Vector',
    'ベクトル解析': 'Vector Calculus',
    '行列': 'Matrix',
    '線形代数': 'Linear Algebra',
    '線形': 'Linear',
    '非線形': 'Nonlinear',
    'テンソル': 'Tensor',
    '複素数': 'Complex Number',
    '複素関数': 'Complex Function',
    'フーリエ変換': 'Fourier Transform',
    'フーリエ級数': 'Fourier Series',
    'ラプラス変換': 'Laplace Transform',
    '関数': 'Function',
    '解析': 'Analysis',
    '数値計算': 'Numerical Computation',
    '数値解析': 'Numerical Analysis',
    '近似': 'Approximation',
    '級数': 'Series',
    '級数展開': 'Series Expansion',
    'テイラー展開': 'Taylor Expansion',
    '収束': 'Convergence',
    '発散': 'Divergence',
    '極限': 'Limit',
    '連続': 'Continuous',
    '離散': 'Discrete',

    # ===== TECHNICAL TERMS - GENERAL =====
    '理論': 'Theory',
    '原理': 'Principle',
    '定理': 'Theorem',
    '公式': 'Formula',
    '方程式': 'Equation',
    '境界条件': 'Boundary Condition',
    '初期条件': 'Initial Condition',
    '解': 'Solution',
    '厳密解': 'Exact Solution',
    '近似解': 'Approximate Solution',
    '数値解': 'Numerical Solution',
    '解析解': 'Analytical Solution',
    '物理': 'Physics',
    '物理学': 'Physics',
    '材料科学': 'Materials Science',
    '電子構造': 'Electronic Structure',
    '化学結合': 'Chemical Bond',
    '光学特性': 'Optical Properties',
    '第一原理計算': 'First-Principles Calculation',
    '可視化': 'Visualization',

    # ===== SPECIAL TECHNICAL TERMS =====
    'Born解釈': 'Born Interpretation',
    'Born の確率解釈': "Born's Probability Interpretation",
    'STM原理': 'STM Principle',
    'STM': 'Scanning Tunneling Microscope (STM)',

    # ===== JAPANESE-SPECIFIC SECTION HEADERS =====
    '基本原理': 'Basic Principles',
    '基礎': 'Fundamentals',
    '基本': 'Basic',
    '応用': 'Application',
    '実装': 'Implementation',
    '詳細': 'Details',
    '概要': 'Overview',

    # ===== SERIES-SPECIFIC NAMES =====
    '量子力学入門': 'Introduction to Quantum Mechanics',
    '場の量子論入門': 'Introduction to Quantum Field Theory',
    '確率・確率過程入門': 'Introduction to Probability & Stochastic Processes',
    '古典統計力学入門': 'Introduction to Classical Statistical Mechanics',

    # ===== TECHNICAL TERMS (KATAKANA) =====
    'ミクロ': 'micro',
    'マクロ': 'macro',
    'エネルギー': 'energy',

    # ===== COMMON WORDS =====
    '状態': 'state',
    '実行': 'execution',
    '最大': 'maximum',
    '最小': 'minimum',
    '計算': 'calculation',
    '結果': 'result',
    '方法': 'method',
    '性質': 'property',
    '特性': 'characteristic',
    '特徴': 'feature',
    '条件': 'condition',
    '問題': 'problem',
    '解法': 'solution method',
    '手法': 'technique',
    '系': 'system',

    # ===== VERB ENDINGS (suru verbs) =====
    'します': '',
    'する': '',
    'された': 'ed',
    'して': 'ing',
    'しない': 'not',

    # ===== COPULA =====
    'です': 'is',
    'である': 'is',
    'でした': 'was',
    'だった': 'was',
    'だ': '',

    # ===== COMMON PARTICLES (STANDALONE) =====
    'など': 'etc.',
    'から': 'from',
    'まで': 'to',
    'より': 'than',
    'ため': 'for',
    'もの': 'thing',
    'こと': 'thing',
    'とき': 'when',
    'ところ': 'place',
    'ほか': 'other',
    'ほど': 'degree',
    'くらい': 'about',
    'ぐらい': 'about',
    'ばかり': 'only',
    'だけ': 'only',
    'しか': 'only',
    'さえ': 'even',
    'まで': 'until',
    'ほど': 'as',
}

# Additional particle cleanup patterns (applied after main translation)
PARTICLE_CLEANUP = [
    # Remove trailing particles at end of English words
    (r'([A-Za-z]+)の([^a-zA-Z]|$)', r'\1\2'),  # "Wordの " -> "Word "
    (r'([A-Za-z]+)を([^a-zA-Z]|$)', r'\1\2'),  # "Wordを " -> "Word "
    (r'([A-Za-z]+)が([^a-zA-Z]|$)', r'\1\2'),  # "Wordが " -> "Word "
    (r'([A-Za-z]+)は([^a-zA-Z]|$)', r'\1\2'),  # "Wordは " -> "Word "
    (r'([A-Za-z]+)に([^a-zA-Z]|$)', r'\1\2'),  # "Wordに " -> "Word "
    (r'([A-Za-z]+)で([^a-zA-Z]|$)', r'\1\2'),  # "Wordで " -> "Word "
    (r'([A-Za-z]+)と([^a-zA-Z]|$)', r'\1\2'),  # "Wordと " -> "Word "
    (r'([A-Za-z]+)な([^a-zA-Z]|$)', r'\1\2'),  # "Wordな " -> "Word "
    (r'([A-Za-z]+)や([^a-zA-Z]|$)', r'\1\2'),  # "Wordや " -> "Word "
    (r'([A-Za-z]+)へ([^a-zA-Z]|$)', r'\1\2'),  # "Wordへ " -> "Word "

    # Numerical patterns
    (r'(\d+)章', r'\1 Chapters'),  # N章 -> N Chapters
    (r'(\d+)個', r'\1'),  # N個 -> N (remove counter)
]

# Post-processing cleanup patterns
POST_PROCESSING = [
    # Clean up double spaces
    (r'  +', ' '),  # Multiple spaces -> single space

    # Clean up orphaned articles and prepositions at line/tag boundaries
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</h', r'</h'),  # Remove orphaned articles before closing heading tags
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</p', r'</p'),  # Remove orphaned articles before closing paragraph tags
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</div', r'</div'),  # Remove orphaned articles before closing div tags
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*</span', r'</span'),  # Remove orphaned articles before closing span tags
    (r'\s+(of|the|a|an|in|on|at|to|for|with|by)\s*$', ''),  # Remove orphaned articles at end of line

    # Clean up double punctuation
    (r'\s*,\s*,', ','),  # Double comma
    (r'\s*\.\s*\.', '.'),  # Double period (but preserve ellipsis ...)

    # Clean up spaces before punctuation
    (r'\s+([,\.!?;:])', r'\1'),  # Remove space before punctuation

    # Clean up spaces around parentheses
    (r'\(\s+', '('),  # Remove space after opening paren
    (r'\s+\)', ')'),  # Remove space before closing paren
]


def translate_text(text: str) -> str:
    """
    Apply comprehensive dictionary-based translation to text with post-processing.

    Args:
        text: Original text with Japanese characters

    Returns:
        Translated text with Japanese replaced by English
    """
    result = text

    # First pass: Direct dictionary replacements (longest first to avoid partial matches)
    for ja, en in sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        result = result.replace(ja, en)

    # Second pass: Particle cleanup patterns
    for pattern, replacement in PARTICLE_CLEANUP:
        result = re.sub(pattern, replacement, result)

    # Third pass: Post-processing cleanup
    for pattern, replacement in POST_PROCESSING:
        result = re.sub(pattern, replacement, result)

    return result


def get_fm_files(file_list_path: Path) -> List[Tuple[Path, str]]:
    """
    Read file list and extract FM files.

    Args:
        file_list_path: Path to JAPANESE_CHARACTER_FILES_LIST.txt

    Returns:
        List of (absolute_path, relative_path) tuples for FM files
    """
    if not file_list_path.exists():
        print(f"✗ File list not found: {file_list_path}")
        return []

    fm_files = []

    with open(file_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Filter for FM files only
            if line.startswith('./FM/'):
                # Convert relative path to absolute
                rel_path = line[2:]  # Remove './'
                abs_path = BASE_DIR / rel_path
                fm_files.append((abs_path, rel_path))

    return fm_files


def translate_file(file_path: Path, rel_path: str) -> bool:
    """
    Translate a single HTML file in-place.

    Args:
        file_path: Absolute path to the file
        rel_path: Relative path for display purposes

    Returns:
        True if successful, False otherwise
    """
    if not file_path.exists():
        print(f"  ✗ File not found: {file_path}")
        return False

    try:
        # Read original content
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Apply translations
        translated_content = translate_text(original_content)

        # Check if any changes were made
        if original_content == translated_content:
            print(f"  ⊙ No changes needed: {rel_path}")
            return True

        # Write translated content back to the same file (in-place)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(translated_content)

        print(f"  ✓ Translated: {rel_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error translating {rel_path}: {e}")
        return False


def get_series_from_path(rel_path: str) -> str:
    """
    Extract series name from relative path.

    Args:
        rel_path: Relative path (e.g., 'FM/quantum-mechanics/chapter-1.html')

    Returns:
        Series name (e.g., 'quantum-mechanics')
    """
    parts = rel_path.split('/')
    if len(parts) >= 2:
        return parts[1]
    return 'unknown'


def main():
    """Main translation process for FM category."""
    print("\n" + "="*70)
    print("FM Category Translation - Enhanced Comprehensive Fix")
    print("Includes: Particles, Verbs, Mixed Patterns, Post-processing")
    print("="*70)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"File list: {FILE_LIST_PATH}")
    print(f"\nTarget series: {', '.join(FM_SERIES)}\n")

    # Get FM files from the list
    fm_files = get_fm_files(FILE_LIST_PATH)

    if not fm_files:
        print("✗ No FM files found in the file list")
        return

    print(f"Found {len(fm_files)} FM files to process\n")

    # Group files by series for organized output
    files_by_series = {}
    for abs_path, rel_path in fm_files:
        series = get_series_from_path(rel_path)
        if series not in files_by_series:
            files_by_series[series] = []
        files_by_series[series].append((abs_path, rel_path))

    # Process files series by series
    total_files = 0
    total_success = 0

    for series in FM_SERIES:
        if series not in files_by_series:
            print(f"\n{'='*70}")
            print(f"Series: {series}")
            print(f"{'='*70}")
            print("  ⊙ No files found for this series")
            continue

        files = files_by_series[series]
        print(f"\n{'='*70}")
        print(f"Series: {series}")
        print(f"{'='*70}")
        print(f"Files to process: {len(files)}\n")

        series_success = 0

        for abs_path, rel_path in sorted(files, key=lambda x: x[1]):
            if translate_file(abs_path, rel_path):
                series_success += 1
                total_success += 1
            total_files += 1

        print(f"\nSeries completed: {series_success}/{len(files)} files translated")

    # Final summary
    print("\n" + "="*70)
    print("TRANSLATION COMPLETE")
    print("="*70)
    print(f"Total files processed: {total_files}")
    print(f"Successfully translated: {total_success}")
    print(f"Failed: {total_files - total_success}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
