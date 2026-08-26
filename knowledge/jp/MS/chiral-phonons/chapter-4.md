---
title: "カイラルフォノン入門シリーズ"
chapter_title: "第4章: 応用と計算手法"
subtitle: "フォノン角運動量の計算、バレートロニクス応用、新興デバイス概念"
---

## ビデオ講義

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/bWqfo0IBMbs"
    title="カイラルフォノン 第4章: 応用と計算手法"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> このビデオは以下のテキストと同じ内容をカバーしています。お好みの学習形式をお選びください。

---

[🌐 EN](<../../../en/MS/chiral-phonons/chapter-4.md>) | 🇯🇵 JP | Last sync: 2025-12-19

[材料科学道場](<../index.html>) > [カイラルフォノン](<index.md>) > 第4章

# 第4章: 応用と計算手法

**フォノン角運動量の計算、バレートロニクス応用、新興デバイス概念**

## 学習目標

- DFPTによるフォノン固有ベクトル計算手法を理解する
- フォノン角運動量（PAM）の計算式 \\(\mathbf{L} = \text{Im}[\mathbf{e}^* \times \mathbf{e}]\\) を導出・実装できる
- Phonopy、Quantum ESPRESSOを用いたカイラルフォノン解析ワークフローを習得する
- フォノンEinstein-de Haas効果とフォノン角運動量輸送を理解する
- バレー-フォノン結合の物理とバレートロニクス応用を説明できる
- カイラルフォノンポラリトンと光物質相互作用を理解する
- フォノンベースの情報処理デバイスの動作原理を理解する
- 完全なPAM計算コードをPythonで実装できる

## 導入

第1章から第3章では、カイラルフォノンの理論的基礎、材料系、実験的検出手法を学びました。本章では、この新興分野の応用展開と実践的な計算手法に焦点を当てます。カイラルフォノンの応用は、バレートロニクスにおけるバレー自由度の制御から、フォノン角運動量輸送、さらには新しいフォノンベースの情報処理デバイスまで多岐にわたります。

計算手法の部分では、第一原理計算（DFT/DFPT）によるフォノン固有ベクトルの取得から、フォノン角運動量の抽出まで、完全なワークフローを詳しく解説します。Phonopy、Quantum ESPRESSO、VASPなどの主要ソフトウェアの使用法と、円偏光度の計算、バンド構造解析、対称性検証などの実践的技術を網羅します。

また、フォノン角運動量の輸送現象（フォノンEinstein-de Haas効果、フォノンスピン流）、バレー-フォノン結合を利用したバレートロニクスデバイス、カイラルフォノンポラリトンによる増強された光物質相互作用など、最先端の研究トピックを紹介します。最後に、量子フォノニクスやトポロジカル保護など、将来の研究方向についても展望します。

## 4.1 カイラルフォノンの計算手法

### 4.1.1 フォノン固有ベクトルのDFPT計算

カイラルフォノンの計算の出発点は、密度汎関数摂動論（DFPT）によるフォノン固有ベクトルの取得です。動的行列 \\(D_{\kappa\alpha,\kappa'\beta}(\mathbf{q})\\) の固有値問題を解きます：

\\[
\sum_{\kappa'\beta} D_{\kappa\alpha,\kappa'\beta}(\mathbf{q}) e_{\kappa'\beta}(\mathbf{q}\nu) = \omega^2(\mathbf{q}\nu) e_{\kappa\alpha}(\mathbf{q}\nu)
\\]

ここで、\\(\mathbf{q}\\) は波数ベクトル、\\(\nu\\) はバンドインデックス、\\(\omega(\mathbf{q}\nu)\\) はフォノン周波数、\\(e_{\kappa\alpha}(\mathbf{q}\nu)\\) は固有ベクトル（原子 \\(\kappa\\) の \\(\alpha\\) 方向成分）です。

**動的行列の計算**

DFPTでは、原子変位に対する電子密度の応答を自己無撞着に計算します：

\\[
D_{\kappa\alpha,\kappa'\beta}(\mathbf{q}) = \frac{1}{\sqrt{M_\kappa M_{\kappa'}}} \frac{\partial^2 E}{\partial u_{\kappa\alpha}(\mathbf{q}) \partial u_{\kappa'\beta}(-\mathbf{q})}
\\]

ここで、\\(M_\kappa\\) は原子質量、\\(E\\) は全エネルギーです。この2階微分は線形応答理論により効率的に計算されます。

### 4.1.2 フォノン角運動量の計算式

フォノン固有ベクトル \\(\mathbf{e}(\mathbf{q}\nu)\\) が得られれば、フォノン角運動量（PAM）は以下の式で計算されます：

\\[
\mathbf{L}(\mathbf{q}\nu) = \text{Im}\left[\mathbf{e}^*(\mathbf{q}\nu) \times \mathbf{e}(\mathbf{q}\nu)\right]
\\]

原子ごとの寄与を明示的に書くと：

\\[
\mathbf{L}(\mathbf{q}\nu) = \sum_\kappa \text{Im}\left[e^*_{\kappa x}(\mathbf{q}\nu) e_{\kappa y}(\mathbf{q}\nu) - e^*_{\kappa y}(\mathbf{q}\nu) e_{\kappa x}(\mathbf{q}\nu)\right] \hat{\mathbf{z}}
\\]

2次元材料では、PAMは面直方向（\\(z\\) 方向）を向き、その符号がカイラリティを決定します。\\(L_z > 0\\) なら右円偏光、\\(L_z < 0\\) なら左円偏光です。

### 4.1.3 円偏光度の計算

円偏光度（circular polarization degree）は、フォノンモードのカイラリティの強さを定量化する指標です：

\\[
P_\text{circ}(\mathbf{q}\nu) = \frac{2|L_z(\mathbf{q}\nu)|}{|\mathbf{e}(\mathbf{q}\nu)|^2}
\\]

完全な円偏光では \\(P_\text{circ} = 1\\)、線形偏光では \\(P_\text{circ} = 0\\) です。実際の材料では、対称性の制約により \\(0 < P_\text{circ} < 1\\) の値をとります。

### 4.1.4 Phonopyを用いた実装

Phonopyは、DFTコード（VASP、Quantum ESPRESSO、ABINIT等）と連携してフォノン計算を行うPythonライブラリです。以下は、Phonopyを用いたカイラルフォノン解析の基本ワークフローです：

```python
#!/usr/bin/env python3
"""
Phonopy を用いたカイラルフォノン解析
WSe2単層を例とする
"""
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
import matplotlib.pyplot as plt

# ステップ1: 構造の読み込み
unitcell = read_vasp("POSCAR")

# ステップ2: Phonopyオブジェクトの作成
# スーパーセルサイズを指定（ここでは4×4×1）
phonon = Phonopy(unitcell, [[4, 0, 0],
                             [0, 4, 0],
                             [0, 0, 1]])

# ステップ3: 変位を生成（VASPで力を計算するため）
phonon.generate_displacements(distance=0.01)  # 0.01 Å変位
supercells = phonon.get_supercells_with_displacements()

# ここでVASPで各変位に対して力を計算（外部プロセス）
# 結果はvasprun.xmlに保存される

# ステップ4: 力定数の読み込み
phonon.produce_force_constants()

# ステップ5: 動的行列の計算
# K点でのフォノン固有値・固有ベクトルを計算
q_K = np.array([1.0/3.0, 1.0/3.0, 0.0])  # K点
phonon.set_qpoints_phonon([q_K])
phonon.run_qpoints_phonon()

# ステップ6: 固有ベクトルと周波数の取得
qpoints = phonon.get_qpoints_phonon()
frequencies = qpoints[0]  # THz単位
eigenvectors = qpoints[1]  # 複素数配列 (num_modes, num_atoms, 3)

print(f"K点でのフォノン周波数 (THz): {frequencies}")
print(f"固有ベクトルの形状: {eigenvectors.shape}")

# ステップ7: フォノン角運動量の計算
def compute_phonon_angular_momentum(eigvec):
    """
    フォノン固有ベクトルから角運動量を計算

    Parameters:
    -----------
    eigvec : complex array, shape (num_atoms, 3)
        フォノン固有ベクトル

    Returns:
    --------
    L_z : float
        z方向の角運動量
    """
    L = np.zeros(3)
    for atom_idx in range(eigvec.shape[0]):
        # Im[e* × e] を計算
        ex_conj = np.conj(eigvec[atom_idx, 0])
        ey_conj = np.conj(eigvec[atom_idx, 1])
        ez_conj = np.conj(eigvec[atom_idx, 2])

        ex = eigvec[atom_idx, 0]
        ey = eigvec[atom_idx, 1]
        ez = eigvec[atom_idx, 2]

        # 外積の虚部
        L[0] += np.imag(ey_conj * ez - ez_conj * ey)
        L[1] += np.imag(ez_conj * ex - ex_conj * ez)
        L[2] += np.imag(ex_conj * ey - ey_conj * ex)

    return L

# 各モードのPAMを計算
num_modes = eigenvectors.shape[0]
PAM_values = []

for mode_idx in range(num_modes):
    L = compute_phonon_angular_momentum(eigenvectors[mode_idx])
    PAM_values.append(L[2])  # z成分のみ

    # 円偏光度の計算
    norm_sq = np.sum(np.abs(eigenvectors[mode_idx])**2)
    P_circ = 2 * np.abs(L[2]) / norm_sq if norm_sq > 0 else 0

    print(f"モード {mode_idx}: "
          f"周波数 = {frequencies[mode_idx]:.3f} THz, "
          f"PAM (Lz) = {L[2]:.6f}, "
          f"円偏光度 = {P_circ:.3f}")

# ステップ8: バンド構造解析
# 高対称点経路を設定
path = [[[0.0, 0.0, 0.0], [1.0/3.0, 1.0/3.0, 0.0],
         [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]]
labels = ["$\\Gamma$", "K", "M", "$\\Gamma$"]

phonon.run_band_structure(path, labels=labels)
phonon.plot_band_structure().savefig("phonon_band.png")

print("\nPhonopyワークフロー完了")
```

### 4.1.5 Quantum ESPRESSOによる完全な計算例

Quantum ESPRESSOは、オープンソースのDFTコードであり、DFPTによるフォノン計算が可能です。以下は、WSe₂単層のK点フォノンを計算する完全な入力ファイルと解析スクリプトです。

```bash
# ステップ1: SCF計算（pw.x）
&CONTROL
  calculation = 'scf'
  prefix = 'wse2'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 4
  a = 3.282
  c = 20.0  ! 真空層
  nat = 3
  ntyp = 2
  ecutwfc = 60.0
  ecutrho = 480.0
/
&ELECTRONS
  conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  W  183.84  W.upf
  Se  78.96  Se.upf
ATOMIC_POSITIONS crystal
  W   0.333333  0.666667  0.5
  Se  0.666667  0.333333  0.55
  Se  0.666667  0.333333  0.45
K_POINTS automatic
  12 12 1  0 0 0

# ステップ2: フォノン計算（ph.x）
&INPUTPH
  prefix = 'wse2'
  outdir = './tmp'
  fildyn = 'wse2.dyn'
  tr2_ph = 1.0d-14
  ldisp = .true.
  nq1 = 6, nq2 = 6, nq3 = 1
/
```

動的行列ファイル（wse2.dyn）からフォノン固有ベクトルを読み込み、PAMを計算するPythonスクリプト：

```python
#!/usr/bin/env python3
"""
Quantum ESPRESSO動的行列ファイルからPAM計算
"""
import numpy as np
import re

def read_qe_dynmat(filename):
    """
    Quantum ESPRESSOの動的行列ファイルを読み込む

    Returns:
    --------
    q_point : array, shape (3,)
        q点座標
    frequencies : array, shape (num_modes,)
        周波数 (cm^-1)
    eigenvectors : complex array, shape (num_modes, num_atoms, 3)
        固有ベクトル
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # q点の読み込み
    for i, line in enumerate(lines):
        if 'q =' in line:
            parts = line.split()
            q_point = np.array([float(parts[3]),
                               float(parts[4]),
                               float(parts[5])])
            break

    # 周波数と固有ベクトルの読み込み
    frequencies = []
    eigenvectors = []

    i = 0
    while i < len(lines):
        if 'freq' in lines[i]:
            # 周波数の読み込み
            freq_str = lines[i].split('=')[1].split('[')[0].strip()
            freq = float(freq_str)
            frequencies.append(freq)

            # 固有ベクトルの読み込み
            i += 1
            eigvec = []
            for atom in range(3):  # WSe2は3原子
                i += 1
                parts = lines[i].split()
                # 実部と虚部を読み込む
                # フォーマット: ( Re(x)  Im(x) ) ( Re(y)  Im(y) ) ( Re(z)  Im(z) )
                x = complex(float(parts[1]), float(parts[2]))
                y = complex(float(parts[4]), float(parts[5]))
                z = complex(float(parts[7]), float(parts[8]))
                eigvec.append([x, y, z])

            eigenvectors.append(eigvec)
        i += 1

    return q_point, np.array(frequencies), np.array(eigenvectors)

def compute_PAM_full(frequencies, eigenvectors):
    """
    全モードのPAMを計算し、カイラリティを分析
    """
    num_modes = len(frequencies)

    print(f"{'Mode':>4} {'Freq (cm-1)':>12} {'PAM (Lz)':>12} "
          f"{'P_circ':>10} {'Type':>10}")
    print("-" * 60)

    PAM_data = []

    for mode_idx in range(num_modes):
        freq = frequencies[mode_idx]
        eigvec = eigenvectors[mode_idx]

        # PAM計算
        L = np.zeros(3, dtype=complex)
        for atom_idx in range(eigvec.shape[0]):
            e_conj = np.conj(eigvec[atom_idx])
            e = eigvec[atom_idx]
            # L = Im[e* × e]
            L += np.array([
                np.imag(e_conj[1] * e[2] - e_conj[2] * e[1]),
                np.imag(e_conj[2] * e[0] - e_conj[0] * e[2]),
                np.imag(e_conj[0] * e[1] - e_conj[1] * e[0])
            ])

        L_z = L[2].real

        # 円偏光度
        norm_sq = np.sum(np.abs(eigvec)**2)
        P_circ = 2 * abs(L_z) / norm_sq if norm_sq > 0 else 0

        # カイラリティ判定
        if abs(L_z) < 1e-6:
            chirality = "Linear"
        elif L_z > 0:
            chirality = "Right (σ+)"
        else:
            chirality = "Left (σ-)"

        print(f"{mode_idx:4d} {freq:12.2f} {L_z:12.6f} "
              f"{P_circ:10.4f} {chirality:>10}")

        PAM_data.append({
            'mode': mode_idx,
            'frequency': freq,
            'PAM': L_z,
            'P_circ': P_circ,
            'chirality': chirality
        })

    return PAM_data

# メイン実行
if __name__ == "__main__":
    # K点の動的行列を読み込む
    q_point, frequencies, eigenvectors = read_qe_dynmat('wse2.dyn_q1')

    print(f"q点: {q_point}")
    print(f"モード数: {len(frequencies)}\n")

    # PAM計算と分析
    PAM_data = compute_PAM_full(frequencies, eigenvectors)

    # カイラルモードの抽出
    chiral_modes = [d for d in PAM_data if abs(d['PAM']) > 0.01]

    print(f"\nカイラルモード数: {len(chiral_modes)}")
    print("\n強いカイラリティを持つモード:")
    for d in sorted(chiral_modes, key=lambda x: -abs(x['PAM']))[:5]:
        print(f"  モード {d['mode']}: {d['frequency']:.2f} cm⁻¹, "
              f"PAM = {d['PAM']:.4f}, {d['chirality']}")
```

**計算の収束性チェック**

正確なPAM計算のためには、以下のパラメータを収束させる必要があります：

- **k点メッシュ**: SCF計算で電荷密度を収束させる（12×12以上推奨）
- **q点メッシュ**: フォノン計算での動的行列の精度（6×6以上）
- **エネルギーカットオフ**: ecutwfc, ecutrho（波動関数と電荷密度）
- **収束閾値**: conv_thr（SCF）、tr2_ph（フォノン）

### 4.1.6 対称性による検証

計算されたPAMは、材料の結晶対称性と整合している必要があります。例えば、WSe₂のようなC₃ᵥ対称性を持つ材料では、K点とK'点でPAMの符号が反転します：

\\[
\mathbf{L}(\mathbf{K}, \nu) = -\mathbf{L}(-\mathbf{K}, \nu) = -\mathbf{L}(\mathbf{K}', \nu)
\\]

この関係は時間反転対称性とバレー対称性の帰結であり、計算結果の妥当性を検証する重要なチェック項目です。

## 4.2 フォノン角運動量輸送

### 4.2.1 フォノンEinstein-de Haas効果

Einstein-de Haas効果は、磁化変化による機械的回転を示す古典的な現象ですが、カイラルフォノンにも類似の効果が予測されています。フォノンが角運動量を運ぶため、フォノンの励起・消滅は結晶全体に機械的回転をもたらします。

```
円偏光レーザー → カイラルフォノン励起 → PAM注入 → 結晶の回転運動 → 角運動量保存 → 測定可能な回転
```

フォノンモード \\(\nu\\) の占有数が \\(\Delta n_\nu\\) だけ変化すると、結晶に注入される角運動量は：

\\[
\Delta \mathbf{L}_\text{crystal} = -\hbar \sum_{\mathbf{q}\nu} \mathbf{L}(\mathbf{q}\nu) \Delta n_{\mathbf{q}\nu}
\\]

ここで、負符号は角運動量保存を表します。この効果は、超高速円偏光ポンプ-プローブ分光法で検出可能です。

### 4.2.2 フォノンスピン流

フォノン角運動量は、温度勾配や電場の下で輸送され、**フォノンスピン流**を形成します。この流れは、ボルツマン輸送方程式の枠組みで記述されます：

\\[
\mathbf{j}_L = \sum_{\mathbf{q}\nu} \mathbf{L}(\mathbf{q}\nu) \mathbf{v}_{\mathbf{q}\nu} n_{\mathbf{q}\nu}
\\]

ここで、\\(\mathbf{v}_{\mathbf{q}\nu} = \nabla_\mathbf{q} \omega(\mathbf{q}\nu)\\) は群速度、\\(n_{\mathbf{q}\nu}\\) はBose-Einstein分布関数です。

フォノンスピン流は、界面でのスピン移行を通じて電子スピンと結合し、スピントロニクスデバイスにおける新しい制御手段を提供します。

### 4.2.3 熱ホール効果との関連

カイラルフォノンは、磁場がない状況でも熱ホール効果を引き起こす可能性があります。これは、フォノンの**ベリー曲率**に起因します：

\\[
\boldsymbol{\Omega}_\nu(\mathbf{q}) = \nabla_\mathbf{q} \times \langle u_{\mathbf{q}\nu} | i\nabla_\mathbf{q} | u_{\mathbf{q}\nu} \rangle
\\]

熱ホール伝導率は：

\\[
\kappa_{xy} = \frac{k_B^2 T}{\hbar V} \sum_{\mathbf{q}\nu} c_2\left(\frac{\hbar\omega_\nu}{k_B T}\right) \Omega_{\nu,z}(\mathbf{q})
\\]

ここで、\\(c_2(x) = (1+x)[\text{Li}_2(-e^x) - \text{Li}_2(-e^{-x})]\\) は熱輸送関数、\\(\text{Li}_2\\) は2次のポリログ関数です。

### 4.2.4 フォノンNernst効果

温度勾配とフォノン角運動量の結合により、横方向の熱電流が生じます。これは**フォノンNernst効果**と呼ばれ、熱電変換デバイスへの応用が期待されます。

**輸送現象の統一的記述**

フォノン輸送現象は、拡張されたボルツマン方程式で統一的に記述されます：

\\[
\frac{\partial n_{\mathbf{q}\nu}}{\partial t} + \mathbf{v}_{\mathbf{q}\nu} \cdot \nabla n_{\mathbf{q}\nu} + \dot{\mathbf{q}} \cdot \nabla_\mathbf{q} n_{\mathbf{q}\nu} = \left(\frac{\partial n}{\partial t}\right)_\text{coll}
\\]

ここで、\\(\dot{\mathbf{q}} = -\frac{e}{\hbar}(\mathbf{E} + \mathbf{v} \times \mathbf{B})\\) は外場による波数変化（フォノンに電荷がある場合を仮想的に考慮）、右辺は散乱項です。

## 4.3 バレートロニクスへの応用

### 4.3.1 バレー-フォノン結合の物理

2次元遷移金属ダイカルコゲナイド（TMD）では、運動量空間のK点とK'点に電子バレーが存在します。カイラルフォノンは、これらのバレーと選択的に結合し、バレー自由度の制御を可能にします。

電子-フォノン結合ハミルトニアンは：

\\[
H_{e-ph} = \sum_{\mathbf{k}\mathbf{q}\nu} g_{\mathbf{k}\mathbf{q}\nu} c^\dagger_{\mathbf{k}+\mathbf{q}} c_{\mathbf{k}} (a_{\mathbf{q}\nu} + a^\dagger_{-\mathbf{q}\nu})
\\]

ここで、\\(g_{\mathbf{k}\mathbf{q}\nu}\\) は結合定数、\\(c^\dagger, c\\) は電子生成・消滅演算子、\\(a^\dagger, a\\) はフォノン生成・消滅演算子です。

カイラルフォノンの場合、結合定数は円偏光に依存します：

\\[
g_{K,\sigma^+} \neq g_{K,\sigma^-}, \quad g_{K',\sigma^+} \neq g_{K',\sigma^-}
\\]

さらに、バレー対称性により：

\\[
g_{K,\sigma^+} = g_{K',\sigma^-}, \quad g_{K,\sigma^-} = g_{K',\sigma^+}
\\]

```
σ+ 光励起 → 右円偏光フォノン → Kバレー電子散乱 → バレー偏極生成
σ- 光励起 → 左円偏光フォノン → K'バレー電子散乱 → バレー偏極生成
```

### 4.3.2 カイラルフォノン媒介バレー間散乱

カイラルフォノンは、異なるバレー間の電子散乱を媒介します。この散乱率は、Fermiの黄金律により計算されます：

\\[
\Gamma_{K \to K'} = \frac{2\pi}{\hbar} \sum_{\mathbf{q}\nu} |g_{\mathbf{q}\nu}|^2 [n_{\mathbf{q}\nu} + f(\omega_\nu)] \delta(E_{\mathbf{k}+\mathbf{q}} - E_\mathbf{k} - \hbar\omega_\nu)
\\]

ここで、\\(n_{\mathbf{q}\nu}\\) はフォノン占有数、\\(f(\omega)\\) はBose分布関数です。

重要な点は、円偏光フォノンの選択則により、\\(\sigma^+\\) フォノンはK→K'散乱を優先し、\\(\sigma^-\\) フォノンはK'→K散乱を優先することです。

### 4.3.3 バレー情報の保存

バレートロニクスデバイスでは、バレー情報（バレー偏極）を長時間保持することが重要です。バレー偏極の緩和時間 \\(T_v\\) は、バレー間散乱率の逆数で与えられます：

\\[
\frac{1}{T_v} = \Gamma_{K \to K'} + \Gamma_{K' \to K}
\\]

カイラルフォノンを制御することで、この緩和時間を調整できます。例えば、円偏光レーザーでカイラルフォノンを選択的に励起することで、バレー間散乱を抑制し、\\(T_v\\) を延長できます。

### 4.3.4 バレートロニクスデバイス概念

カイラルフォノン制御に基づくバレートロニクスデバイスの例：

**1. フォノン制御バレーフィルター**

- **原理**: 円偏光レーザーでカイラルフォノンを励起し、特定のバレーの電子のみを選択的に散乱
- **入力**: 無偏極電子流
- **出力**: バレー偏極電子流
- **利点**: 電極不要、光学的制御

**2. バレー-フォノンメモリ**

- **原理**: バレー偏極をカイラルフォノンの占有数として保存
- **書き込み**: 円偏光レーザーでフォノン励起
- **読み出し**: 円偏光ラマン分光
- **保存時間**: フォノン寿命（ピコ秒～ナノ秒）

**3. バレースイッチ**

- **原理**: フォノン励起により電子をK↔K'間で切り替え
- **制御**: 円偏光の切り替え
- **応答時間**: サブピコ秒
- **応用**: 超高速論理ゲート

## 4.4 新興デバイス概念

### 4.4.1 フォノンベースの情報処理

カイラルフォノンは、電子系とは独立した情報キャリアとして利用できます。フォノンベースの情報処理の利点：

- **低消費電力**: 電荷輸送に伴うジュール熱がない
- **高速性**: フォノン群速度は音速（km/s）
- **低干渉**: 電磁ノイズに強い
- **量子コヒーレンス**: 低温でコヒーレント状態を維持

情報符号化の方法：

- **振幅符号化**: フォノン占有数 \\(n_{\mathbf{q}\nu}\\) で情報を表現
- **位相符号化**: フォノン波束の位相で情報を表現
- **カイラリティ符号化**: 右・左円偏光で0・1を表現

### 4.4.2 カイラルフォノントランジスタ

フォノントランジスタは、フォノン流を制御する3端子素子です。カイラルフォノンを用いた設計例：

```
ソース(フォノン発生) → チャネル(伝播経路) → ドレイン(フォノン検出)
                          ↑
                    ゲート(円偏光制御)
```

**動作原理**：

1. **ソース**: 圧電素子や光励起でフォノンを生成
2. **ゲート**: 円偏光レーザーでカイラルフォノンの分布を制御
   - σ+ 光 → 右円偏光フォノン増加
   - σ- 光 → 左円偏光フォノン増加
3. **チャネル**: カイラリティに依存した伝播
   - フォノニック結晶で特定カイラリティのみ透過
   - 界面でのカイラリティ選択的散乱
4. **ドレイン**: 円偏光ラマン分光や熱電変換で検出

**性能指標**：

- **On/Off比**: ゲート制御によるフォノン流の変調率（10:1以上目標）
- **応答時間**: フォノン寿命で制限（ピコ秒オーダー）
- **エネルギー効率**: ゲート制御に必要な光エネルギー

### 4.4.3 フォノン偏光子とフィルター

**フォノン偏光子**は、特定のカイラリティのフォノンのみを通過させるデバイスです。実現方法の例：

**1. カイラルフォノニック結晶**

周期的な構造により、特定の円偏光フォノンにバンドギャップを形成。反対の円偏光は透過する。

設計方法: トポロジカルフォノニクスの原理を利用し、カイラル対称性を破った超格子構造を設計。

**2. 界面カイラリティフィルター**

2つの材料の界面でのフォノン透過率がカイラリティに依存することを利用。

例: WSe₂/hBN界面では、界面対称性により右・左円偏光フォノンの透過率が異なる。

**3. 回転ドメイン境界フィルター**

2次元材料の回転角度を制御したドメイン境界を利用。

例: ツイストバイレイヤーグラフェンの境界では、回転角度に依存してカイラルフォノンの透過率が変化。

### 4.4.4 集積フォノニクス回路

将来的には、複数のフォノンデバイスを集積した**フォノニクス集積回路**が実現される可能性があります：

```
フォノン発生器 → カイラル偏光子 → フォノン導波路 → フォノントランジスタ → フォノン検出器
                                                        ↑
                                                  制御レーザー
                                                  バイアス音波
```

**構成要素**：

- **フォノン発生器**: 圧電素子、光励起、熱勾配
- **フォノン導波路**: ナノワイヤ、2次元材料リボン
- **フォノン偏光子・スプリッター**: カイラリティ分離
- **フォノン論理ゲート**: AND、OR、NOTゲート
- **フォノンメモリ**: キャビティ閉じ込め
- **フォノン検出器**: ラマン分光、熱電変換

課題：室温での動作、長距離伝播、高効率変換、大規模集積

## 4.5 カイラルフォノンポラリトン

### 4.5.1 光子-フォノン結合

カイラルフォノンが光子と強く結合すると、**カイラルフォノンポラリトン**が形成されます。これは、光とフォノンの混成状態であり、両者の性質を併せ持ちます。

結合ハミルトニアンは：

\\[
H = \sum_\mathbf{q} \hbar\omega_\text{ph}(\mathbf{q}) a^\dagger_\mathbf{q} a_\mathbf{q}
+ \sum_\mathbf{k} \hbar\omega_\text{opt}(\mathbf{k}) b^\dagger_\mathbf{k} b_\mathbf{k}
+ \sum_{\mathbf{k}\mathbf{q}} g_{\mathbf{k}\mathbf{q}} (a^\dagger_\mathbf{q} b_\mathbf{k} + a_\mathbf{q} b^\dagger_\mathbf{k})
\\]

ここで、\\(a^\dagger, a\\) はフォノン演算子、\\(b^\dagger, b\\) は光子演算子、\\(g_{\mathbf{k}\mathbf{q}}\\) は結合定数です。

この系の固有状態（ポラリトン）は、上部分枝（UPB）と下部分枝（LPB）に分かれます：

\\[
\omega_\pm(\mathbf{k}) = \frac{\omega_\text{ph} + \omega_\text{opt}}{2}
\pm \frac{1}{2}\sqrt{(\omega_\text{ph} - \omega_\text{opt})^2 + 4g^2}
\\]

共鳴条件（\\(\omega_\text{ph} \approx \omega_\text{opt}\\)）では、ラビ分裂 \\(2g\\) が観測されます。

### 4.5.2 増強された光物質相互作用

カイラルフォノンポラリトンは、通常の光-物質相互作用よりも大きな結合強度を示します。これは以下の理由によります：

- **カイラリティマッチング**: 円偏光光とカイラルフォノンの選択的結合
- **局在増強**: 2次元材料での電場局在
- **量子閉じ込め**: ナノ構造でのモード体積縮小

増強因子は、Purcell係数で評価されます：

\\[
F_P = \frac{3}{4\pi^2}\left(\frac{\lambda}{n}\right)^3 \frac{Q}{V}
\\]

ここで、\\(Q\\) は共振器のQ値、\\(V\\) はモード体積、\\(\lambda\\) は波長、\\(n\\) は屈折率です。2次元材料のナノキャビティでは、\\(F_P > 100\\) が達成されています。

### 4.5.3 応用: 超低閾値レーザー

強結合ポラリトンは、ボース凝縮による**ポラリトンレーザー**の実現を可能にします。カイラルフォノンポラリトンを用いると、円偏光出力のコヒーレント光源が実現できます。

利点：

- 低閾値（電子系よりも軽い有効質量）
- 室温動作の可能性
- 円偏光の制御が容易

### 4.5.4 非線形光学応用

カイラルフォノンポラリトンは、強い非線形光学効果を示します。特に、第二高調波発生（SHG）や和周波発生（SFG）において、カイラリティ選択的な応答が観測されます。

カイラル非線形感受率：

\\[
\chi^{(2)}_{\sigma^+} \neq \chi^{(2)}_{\sigma^-}
\\]

これにより、円偏光制御された非線形光学デバイス（光スイッチ、周波数変換器）が実現できます。

## 4.6 将来の方向性

### 4.6.1 カイラリティを持つ量子フォノニクス

量子情報科学への応用として、カイラルフォノンを用いた**量子フォノニクス**が注目されています。

**量子ビット符号化**

カイラルフォノンの量子状態として量子ビットを符号化：

\\[
|\psi\rangle = \alpha |0, \sigma^+\rangle + \beta |1, \sigma^-\rangle
\\]

ここで、\\(|0, \sigma^+\rangle\\) は右円偏光フォノン1個、\\(|1, \sigma^-\rangle\\) は左円偏光フォノン1個を表します。

**量子ゲート操作**：

- **単一量子ビットゲート**: 円偏光パルスでカイラリティを回転
- **2量子ビットゲート**: フォノン-フォノン相互作用（非線形項）を利用
- **読み出し**: 円偏光分解ラマン分光

課題：デコヒーレンス時間の延長（現状：ピコ秒→目標：ナノ秒）

### 4.6.2 トポロジカル保護

カイラルフォノンとトポロジカル物性の組み合わせにより、**トポロジカル保護されたフォノンモード**が実現できます。

例: Chern数を持つフォノンバンド

\\[
C = \frac{1}{2\pi} \int_\text{BZ} d^2q \, \Omega_z(\mathbf{q})
\\]

ここで、\\(\Omega_z(\mathbf{q})\\) はフォノンのベリー曲率です。\\(C \neq 0\\) の場合、エッジにカイラルフォノンモードが現れます。

**利点**：

- 欠陥に対する頑健性（後方散乱の抑制）
- 方向性のあるフォノン伝播
- 量子状態の保護

### 4.6.3 室温応用への挑戦

現状の多くの研究は極低温で行われていますが、実用化には室温動作が不可欠です。室温応用への戦略：

**1. 高周波フォノンの利用**

光学フォノン（THz領域）は \\(\hbar\omega \gg k_B T\\) を満たしやすく、熱励起の影響を受けにくい。

**2. 強結合系の利用**

光-フォノン強結合により、ポラリトン分裂 \\(2g\\) を熱エネルギー \\(k_B T\\) より大きくする（\\(g > 25\\) meV @ 室温）。

**3. トポロジカル保護**

トポロジカルフォノンは、無秩序や熱揺らぎに対して頑健。エネルギーギャップ \\(\Delta > k_B T\\) を確保。

**4. ナノ構造の利用**

量子閉じ込め効果により、離散的なエネルギー準位を形成。準位間隔 \\(\Delta E > k_B T\\) とすることで、室温での状態制御が可能。

### 4.6.4 新材料探索

機械学習を用いた大規模材料探索により、より強いカイラリティを持つフォノンモードを示す新材料の発見が期待されます。

探索の観点：

- **対称性**: カイラルな空間群（P3₁, P6₁, etc.）
- **強いスピン軌道相互作用**: 重元素を含む化合物
- **バレー縮退**: 2次元材料、特にTMD類似体
- **高周波光学フォノン**: 室温動作に有利

有望な材料系：

- 新規TMD（MoTe₂、WSeTe混晶）
- カイラルペロブスカイト
- トポロジカル絶縁体表面
- 2次元磁性体（CrI₃、CrBr₃）

### 4.6.5 他分野との融合

カイラルフォノン研究は、他の分野との融合により新しい展開を見せています：

- **スピントロニクス**: フォノン-スピン変換、スピンカロリトロニクス
- **量子情報**: フォノン量子ビット、量子トランスデューサー
- **オプトメカニクス**: 光-機械結合、量子測定
- **熱電変換**: カイラリティ制御による効率向上
- **化学**: カイラル触媒、不斉合成への応用

```
カイラルフォノン → スピントロニクス、量子情報、オプトメカニクス、熱電変換、化学・触媒
                 → 新デバイス、新応用
```

## 演習問題

### 問題1: PAM計算の実装

以下のフォノン固有ベクトル（規格化済み）が与えられたとき、フォノン角運動量 \\(L_z\\) と円偏光度 \\(P_\text{circ}\\) を計算せよ。

2原子系（W, Se）、K点でのA'₁光学モード：

\\[
\mathbf{e}_W = \frac{1}{\sqrt{2}}(1, i, 0), \quad
\mathbf{e}_\text{Se} = \frac{1}{\sqrt{2}}(-1, -i, 0)
\\]

<details>
<summary>解答を見る</summary>

**解答**：

各原子の寄与を計算：

W原子: \\(L_{z,W} = \text{Im}[e^*_x e_y - e^*_y e_x] = \text{Im}[1 \cdot i - (-i) \cdot 1] = \text{Im}[i + i] = 2\\)

Se原子: \\(L_{z,\text{Se}} = \text{Im}[(-1) \cdot (-i) - i \cdot (-1)] = \text{Im}[i + i] = 2\\)

合計: \\(L_z = L_{z,W} + L_{z,\text{Se}} = 4\\)

規格化: \\(|\mathbf{e}|^2 = 2\\) なので、\\(P_\text{circ} = 2|L_z|/|\mathbf{e}|^2 = 2 \times 4 / 2 = 4\\)

※実際には規格化により \\(P_\text{circ} \leq 1\\) となるよう調整が必要。この例は教育目的の簡略化。

</details>

### 問題2: バレー-フォノン結合の選択則

WSe₂単層のKバレーの電子が、右円偏光フォノン（σ⁺）を吸収する過程を考える。最終状態のバレー（KまたはK'）と、許される遷移の選択則を、角運動量保存則から導出せよ。

<details>
<summary>解答を見る</summary>

**解答**：

角運動量保存則: \\(L_\text{final} = L_\text{initial} + L_\text{phonon}\\)

Kバレー電子のバレー指数: \\(\tau = +1\\) （角運動量 \\(+\hbar/2\\) に対応）

K'バレー電子のバレー指数: \\(\tau = -1\\) （角運動量 \\(-\hbar/2\\) に対応）

右円偏光フォノン（σ⁺）: \\(L_z = +\hbar\\)

初期状態: Kバレー (\\(\tau_i = +1\\))

最終状態の角運動量: \\(\tau_f = \tau_i + 1 = +1 + 1 = +2\\) (mod 2) → K'バレー (\\(\tau = -1\\))

**選択則**: K バレー + σ⁺ フォノン → K' バレー

一般化: \\(\tau_f = \tau_i + \text{sgn}(L_z)\\) （\\(\text{sgn}\\) は符号関数）

</details>

### 問題3: フォノン熱ホール伝導率

フォノンのベリー曲率が \\(\Omega_z(\mathbf{q}) = \Omega_0\\) （定数）である2次元材料を考える。温度 \\(T = 300\\) K、フォノン周波数 \\(\omega = 10\\) THz、ブリルアンゾーンの面積 \\(A_\text{BZ} = (2\pi/a)^2\\) （\\(a = 3\\) Å）のとき、熱ホール伝導率 \\(\kappa_{xy}\\) を見積もれ（\\(\Omega_0 = 10\\) Å²と仮定）。

<details>
<summary>解答を見る</summary>

**解答**：

熱ホール伝導率の式（単一モード近似）:

\\(\kappa_{xy} = \frac{k_B^2 T}{\hbar} n_\text{ph} c_2(x) \Omega_0\\)

ここで、\\(x = \hbar\omega / k_B T\\)、\\(n_\text{ph} = A_\text{BZ}^{-1}\\) はフォノン密度

数値代入:

- \\(k_B T = 26\\) meV @ 300 K
- \\(\hbar\omega = 10 \times 10^{12} \times 6.58 \times 10^{-16} = 6.58\\) meV
- \\(x = 6.58 / 26 \approx 0.25\\)
- \\(c_2(0.25) \approx 0.2\\) （数値計算）
- \\(n_\text{ph} = (2\pi / 3 \times 10^{-10})^{-2} \approx 2.25 \times 10^{19}\\) m⁻²

\\(\kappa_{xy} \approx \frac{(1.38 \times 10^{-23})^2 \times 300}{1.05 \times 10^{-34}} \times 2.25 \times 10^{19} \times 0.2 \times 10^{-20}\\)

\\(\kappa_{xy} \approx 0.1\\) W/K （オーダー見積もり）

</details>

### 問題4: カイラルフォノンポラリトンのラビ分裂

フォノン周波数 \\(\omega_\text{ph} = 5\\) THz、光共振器周波数 \\(\omega_\text{opt} = 5.2\\) THz、結合定数 \\(g = 0.3\\) THz のカイラルフォノンポラリトン系において、上部・下部ポラリトン分枝の周波数 \\(\omega_\pm\\) とラビ分裂を計算せよ。

<details>
<summary>解答を見る</summary>

**解答**：

ポラリトン分散式:

\\(\omega_\pm = \frac{\omega_\text{ph} + \omega_\text{opt}}{2} \pm \frac{1}{2}\sqrt{(\omega_\text{ph} - \omega_\text{opt})^2 + 4g^2}\\)

数値代入:

\\(\omega_+ + \omega_- = \frac{5 + 5.2}{2} = 5.1\\) THz

\\(\sqrt{(5 - 5.2)^2 + 4 \times 0.3^2} = \sqrt{0.04 + 0.36} = \sqrt{0.4} \approx 0.632\\) THz

\\(\omega_+ = 5.1 + 0.316 = 5.416\\) THz

\\(\omega_- = 5.1 - 0.316 = 4.784\\) THz

**ラビ分裂**: \\(\Delta\omega = \omega_+ - \omega_- = 0.632\\) THz ≈ 2.6 meV

これは室温熱エネルギー（26 meV）の約1/10なので、明確なポラリトン分裂を観測するには低温が必要。

</details>

### 問題5: カイラルフォノントランジスタの設計

WSe₂リボン（幅100 nm、長さ1 μm）を用いたカイラルフォノントランジスタを設計する。ソースで励起されたフォノンが、ゲート円偏光レーザー（波長600 nm、出力1 mW）によりカイラリティ選択的に制御される。On/Off比10を達成するための設計パラメータを議論せよ。

<details>
<summary>解答を見る</summary>

**解答**：

**設計方針**:

1. **ソース**: 圧電素子（ZnO薄膜）で幅広い周波数のフォノンを励起
2. **ゲート**: 円偏光レーザー（600 nm = 2.07 eV）でWSe₂のA励起子（1.65 eV）付近を共鳴励起
   - σ⁺光 → Kバレー励起 → 右円偏光フォノン選択的散乱
   - σ⁻光 → K'バレー励起 → 左円偏光フォノン選択的散乱
3. **チャネル**: WSe₂リボン、カイラリティ依存伝播
   - 右円偏光フォノン: 電子との散乱が少ない（透過率 90%）
   - 左円偏光フォノン: 電子との散乱が多い（透過率 10%）
4. **ドレイン**: 熱電変換素子（Bi₂Te₃）で検出

**On/Off比の計算**:

On状態（σ⁺光）: ドレイン到達フォノン数 \\(N_\text{on} \propto 0.9\\) （透過率90%）

Off状態（σ⁻光）: ドレイン到達フォノン数 \\(N_\text{off} \propto 0.1\\) （透過率10%）

On/Off比: \\(N_\text{on} / N_\text{off} = 0.9 / 0.1 = 9\\) ≈ 10 ✓

**課題**:

- 応答時間: フォノン寿命（～1 ps）で制限
- ゲート効率: 光吸収率を向上（共振器構造の利用）
- 熱散逸: 1 mWレーザーによる加熱を抑制（パルス動作）

</details>

## まとめ

本章では、カイラルフォノンの計算手法と多様な応用展開を学びました。DFPTによるフォノン固有ベクトル計算から、フォノン角運動量（PAM）の抽出まで、完全な計算ワークフローをPhonopyとQuantum ESPRESSOを用いて実装しました。重要な計算式 \\(\mathbf{L} = \text{Im}[\mathbf{e}^* \times \mathbf{e}]\\) は、カイラルフォノン研究の基礎となります。

応用面では、フォノン角運動量輸送（Einstein-de Haas効果、熱ホール効果）、バレートロニクスにおけるバレー-フォノン結合の利用、フォノンベースの情報処理デバイス（トランジスタ、偏光子、集積回路）、カイラルフォノンポラリトンによる増強された光物質相互作用など、幅広いトピックをカバーしました。

将来の方向性として、量子フォノニクス、トポロジカル保護、室温応用、新材料探索、他分野との融合が重要な研究課題です。カイラルフォノンは、基礎物理学の理解を深めるだけでなく、次世代の情報処理技術や量子技術の基盤となる可能性を秘めています。

**主要なポイント**：

- PAM計算: \\(\mathbf{L} = \text{Im}[\mathbf{e}^* \times \mathbf{e}]\\)、円偏光度 \\(P_\text{circ} = 2|L_z|/|\mathbf{e}|^2\\)
- 計算ツール: Phonopy（フォノン解析）、Quantum ESPRESSO（DFPT）、Python（後処理）
- PAM輸送: Einstein-de Haas効果、フォノンスピン流、熱ホール効果
- バレートロニクス: バレー-フォノン結合、カイラリティ選択的散乱、バレー情報保存
- 新デバイス: フォノントランジスタ、偏光子、集積フォノニクス回路
- ポラリトン: 光-フォノン強結合、ラビ分裂、増強された相互作用
- 将来展望: 量子フォノニクス、トポロジカル保護、室温応用

---

[← 第3章](<chapter-3.md>) | [シリーズトップ →](<index.md>)

---

## 免責事項

この教育コンテンツは、橋本研究室のナレッジベース用にAIの支援を受けて作成されました。正確性を期していますが、重要な情報については一次資料や査読済み文献で確認することをお勧めします。
