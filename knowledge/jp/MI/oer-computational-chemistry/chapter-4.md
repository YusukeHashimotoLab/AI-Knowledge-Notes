---
title: "第4章: ハンズオン: Pythonによるスクリーニング"
chapter_title: "第4章: ハンズオン: Pythonによるスクリーニング"
subtitle: "5つの架空の触媒、それぞれ4つの段階、そして火山型プロットが厳密に再現するランキング"
---

## ビデオ講義

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/QMPBJhZdoWY"
    title="OERの計算化学 第4章: ハンズオン: Pythonによるスクリーニング"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> このビデオは以下のテキストと同じ内容をカバーしています。お好みの学習形式をお選びください。

---

🌐 JP | [🇬🇧 EN](<../../../en/MI/oer-computational-chemistry/chapter-4.html>) | Last sync: 2026-08-18

[マテリアルズ・インフォマティクス道場](<../index.html>) > [OERの計算化学](<index.html>) > 第4章

3章分の道具立てには、それを動かす機械がふさわしいでしょう。本章では、実際のスクリーニングが行う論理的な手数をすべて備えたまま、可能な限り小さく完結したスクリーニングのパイプラインを組み立てます。候補を定義し、その中間体を計算し、自由エネルギー図を構築し、性能を表す1つの数値を取り出し、順位をつけ、そしてそのランキングを第3章の安価な記述子モデルと突き合わせる — という手順です。

NumPyで、1秒とかからずに、存在しない5つの触媒の上で走ります。

最後の点は本章の設計であって、限界ではありません。**以下の吸着自由エネルギーは、Catalyst AからCatalyst Eと呼ばれる架空の表面についての、説明用の教育的な値です。** これらは、5つが火山型プロット上の互いに異なる示唆的な5つの位置に着地するように選ばれています — 酸素を強く吸着しすぎるものが1つ、弱すぎるものが1つ、頂点に近いものが1つ、その中間が2つ。DFTの結果でもなければ測定値でもなく、いかなる実在の材料にも帰属させてはなりません。*本物* なのは、そこから下流のすべてです。本章に登場する各段階の自由エネルギー変化、各図、各過電圧、各ランキングは、明示された入力と第3章の2つの定数に対してコードが実行した算術であり、印字されたあらゆる数値はコードの実際の出力です。

入力が正直な教育用の値であり、論理が厳密であるスクリーニングのパイプラインは、実データを持ちながら手順が隠されているものよりも、良い教師です。DFTのエネルギーに差し替えても、コードは何ひとつ変わりません。

> **これらは第2章の触媒ではありません。** 第2章では、同じくアルファベットで名付けられた別の架空の表面の組を使っており、そちらは4つの段階の自由エネルギー変化によって直接定義されていました。ここでの5つは *吸着* エネルギーによって定義される、無関係な顔ぶれです。名前は章の境界でリセットされます。引き継がれるのは定数だけです。

## 4.1 計画

5つのステージがあり、それぞれが1つ前のステージの産物を消費します。

| ステージ | 入力 | 出力 |
|---|---|---|
| 1. データセットを定義する | 明示された5組の \\((\Delta G_{\ast\text{OH}}, \Delta G_{\ast\text{O}})\\) | 記述子 \\(x = \Delta G_{\ast\text{O}} - \Delta G_{\ast\text{OH}}\\) |
| 2. 各段階を構築する | データセット + スケーリング則 | \\(\Delta G_{\ast\text{OOH}}\\) と4つの \\(\Delta G_i\\) すべて（総和チェック済み） |
| 3. 自由エネルギー図 | 各段階の自由エネルギー変化 | \\(U = 0\\) と \\(U = 1.23\\) V における累積プロファイル |
| 4. 過電圧 | 各段階の自由エネルギー変化 | \\(U_L\\)、\\(\eta\\)、電位決定段階、ランキング |
| 5. 火山型プロットへの配置 | 記述子のみ | 予測された \\(\eta\\)（ステージ4と照合） |

ステージ5の狙いは、新しい何かを計算することではありません。**触媒あたり1つの数** を使うモデルが、**4つ** を使った計算を厳密に再現することを確かめることです。両者が食い違えば記述子は仕事をしていないことになり、一致すれば、第3章の次元の潰れは本物だったということになります。

## 4.2 ステージ1: 教育用データセット

スクリーニング研究は、候補の表と、それぞれについて計算できた量から始まります。私たちの表は5行、計算された列が2つです。

2つの定数は、それぞれ異なる資格で使われており、コードもそのことをコメントで述べています。\\(1.23\\) V は **定義** — 4電子反応の標準平衡電位 — であり、\\(4 \times 1.23 = 4.92\\) eV はそれに対する算術です。\\(3.2\\) eV というスケーリングのオフセットは **経験的で近似的な規則性** であり、第3章で導入し留保もつけたものです。そして、ここで興味深い仕方で誤りうる唯一の量でもあります。

```python
import numpy as np

# ------------------------------------------------------------------
# 定数。2つは定義とそれに対する算術、1つは経験則である。
# ------------------------------------------------------------------
E_EQ = 1.23                    # V, OERの平衡電位（定義）
N_STEPS = 4
SUM_TOTAL = N_STEPS * E_EQ     # eV, 算術: 4 x 1.23
SCALING = 3.2                  # eV, dG_OOH - dG_OH; 経験的かつ近似的

# ------------------------------------------------------------------
# 教育用データセット。
#
# 触媒A〜Eは架空のものである。以下の吸着自由エネルギーは、5つの表面を
# 火山型プロット上の5つの異なる位置に配置するために、本章のために
# 作られた説明用の値である。測定されたものでもなく、DFTから計算された
# ものでもなく、いかなる実在の材料にも帰属させてはならない。これらより
# 下流のすべては、これらの数値と上記の2つの定数に対して行われた算術
# である。
# ------------------------------------------------------------------
CATALYSTS = [
    # 名前, dG_OH (eV), dG_O (eV), 意図した役割
    ("Catalyst A", 0.60, 1.60, "binds oxygen too strongly"),
    ("Catalyst B", 0.80, 2.20, "mid, strong-binding side"),
    ("Catalyst C", 1.00, 2.65, "near the apex"),
    ("Catalyst D", 1.10, 3.00, "mid, weak-binding side"),
    ("Catalyst E", 1.30, 3.60, "binds oxygen too weakly"),
]

names = [c[0] for c in CATALYSTS]
dG_OH = np.array([c[1] for c in CATALYSTS])
dG_O = np.array([c[2] for c in CATALYSTS])
roles = [c[3] for c in CATALYSTS]
descriptor = dG_O - dG_OH

print("Stage 1: the teaching dataset (ILLUSTRATIVE values, fictitious surfaces)")
print(f"{'catalyst':<12} {'dG_OH':>7} {'dG_O':>7} {'x = dG_O - dG_OH':>18}   role")
for k, name in enumerate(names):
    print(f"{name:<12} {dG_OH[k]:7.2f} {dG_O[k]:7.2f} {descriptor[k]:18.2f}   {roles[k]}")
print()
print(f"descriptor range : {descriptor.min():.2f} to {descriptor.max():.2f} eV")
print(f"all distinct     : {len(set(np.round(descriptor, 6))) == len(names)}")
print(f"dG_O > dG_OH for every surface (O is the later intermediate): "
      f"{bool(np.all(dG_O > dG_OH))}")
```

**出力:**

```
Stage 1: the teaching dataset (ILLUSTRATIVE values, fictitious surfaces)
catalyst       dG_OH    dG_O   x = dG_O - dG_OH   role
Catalyst A      0.60    1.60               1.00   binds oxygen too strongly
Catalyst B      0.80    2.20               1.40   mid, strong-binding side
Catalyst C      1.00    2.65               1.65   near the apex
Catalyst D      1.10    3.00               1.90   mid, weak-binding side
Catalyst E      1.30    3.60               2.30   binds oxygen too weakly

descriptor range : 1.00 to 2.30 eV
all distinct     : True
dG_O > dG_OH for every surface (O is the later intermediate): True
```

**結果を読み解く。** 記述子は \\(1.00\\) から \\(2.30\\) eV に広がり、第3章が導いた頂点の位置 \\(1.60\\) eV をまたいでいます — AとBは強吸着側、Cは頂点をわずかに越えたところ、DとEは弱吸着側です。この広がりはデータセット設計における意図的な行為であり、それを自覚しておく価値があります。火山型プロットの片方の斜面だけをたまたま標本にしてしまったスクリーニングの集合は、記述子と活性の間に単調に見える関係を生み出し、そのプロットを読んだ人は、強く吸着するほど常に良いのだと結論するでしょう。そうではありません。良いのは頂点までの話です。

末尾にある2行の健全性チェックはコストゼロで、打ち間違いを捕まえてくれます。記述子がすべて異なるということは、後のランキングを同順位が煩わせないということであり、\\(\Delta G_{\ast\text{O}} > \Delta G_{\ast\text{OH}}\\) は、中間体が機構の要求する順に並んでいることを確かめてくれます — \\(\ast\text{O}\\) は反応座標上で \\(\ast\text{OH}\\) より先にあるので、差が負になったなら、それは風変わりな表面ではなく入力の誤りの合図です。

## 4.3 ステージ2: 2つの数から4つの段階へ

ここでスケーリング則が働きに見合う対価を得ます。私たちは \\(\Delta G_{\ast\text{OH}}\\) と \\(\Delta G_{\ast\text{O}}\\) を知っていますが、\\(\Delta G_{\ast\text{OOH}}\\) は知りません。3つ目の中間体を計算する代わりに、それを *予測* します — そしてその予測こそが、記述子1つの火山型プロットが存在する理由のすべてです。

4つの段階は、累積エネルギーの差として得られます。

\\[ \Delta G_1 = \Delta G_{\ast\text{OH}}, \quad \Delta G_2 = \Delta G_{\ast\text{O}} - \Delta G_{\ast\text{OH}}, \quad \Delta G_3 = \Delta G_{\ast\text{OOH}} - \Delta G_{\ast\text{O}}, \quad \Delta G_4 = 4.92 - \Delta G_{\ast\text{OOH}} \\]

最後の1つは総和則によって決まってしまうため、真っ先に思いつくチェック — 4つを足すと \\(4.92\\) になるか — は、構成上そもそも循環しています。そこでコードは、循環しない2つ目のチェックを行います。独立な代数的経路に沿って \\(\Delta G_4\\) を組み立て直し、突き合わせるのです。

```python
# ------------------------------------------------------------------
# ステージ2: スケーリング則がdG_OOHを供給し、4つの段階はその差として
# 得られる。最後の段階は総和則によって決まってしまい、まさにそれゆえ
# 下の総和チェックは自明にしか満たされえない -- そこで段階4を独立に
# 組み立て直し、両者を比較する。
# ------------------------------------------------------------------
dG_OOH = dG_OH + SCALING

dG1 = dG_OH - 0.0                 # H2O + *   -> *OH  + H+ + e-
dG2 = dG_O - dG_OH                # *OH       -> *O   + H+ + e-
dG3 = dG_OOH - dG_O               # *O + H2O  -> *OOH + H+ + e-
dG4 = SUM_TOTAL - dG_OOH          # *OOH      -> * + O2 + H+ + e-

steps = np.vstack([dG1, dG2, dG3, dG4]).T          # 形状は (触媒5, 段階4)
STEP_LABELS = ["1  H2O -> *OH", "2  *OH -> *O",
               "3  *O -> *OOH", "4  *OOH -> O2"]

print("Stage 2: intermediates and the four step energies (eV)")
print(f"{'catalyst':<12} {'dG_OH':>7} {'dG_O':>7} {'dG_OOH':>8} "
      f"{'dG1':>7} {'dG2':>7} {'dG3':>7} {'dG4':>7} {'sum':>8}")
for k, name in enumerate(names):
    print(f"{name:<12} {dG_OH[k]:7.2f} {dG_O[k]:7.2f} {dG_OOH[k]:8.2f} "
          f"{dG1[k]:7.2f} {dG2[k]:7.2f} {dG3[k]:7.2f} {dG4[k]:7.2f} "
          f"{steps[k].sum():8.4f}")
print()

# 相互チェック1: 総和の制約を、浮動小数点の許容誤差いっぱいまで確認する。
sums = steps.sum(axis=1)
print(f"sum constraint: every catalyst sums to {SUM_TOTAL:.2f} eV : "
      f"{bool(np.allclose(sums, SUM_TOTAL, atol=1e-12))}")
print(f"  largest deviation from {SUM_TOTAL:.2f} eV : "
      f"{np.max(np.abs(sums - SUM_TOTAL)):.2e} eV")

# 相互チェック2: 総和則を使わずに、中間体だけからdG4を組み立て直し、
# 2つの経路が一致することを確認する。
dG4_independent = SUM_TOTAL - SCALING - dG_OH
print(f"  dG4 rebuilt as SUM_TOTAL - SCALING - dG_OH agrees : "
      f"{bool(np.allclose(dG4, dG4_independent, atol=1e-12))}")

# 相互チェック3: スケーリング則が中央のペアを潰していること。
print(f"  dG2 + dG3 equals the scaling constant {SCALING:.2f} eV for all five : "
      f"{bool(np.allclose(dG2 + dG3, SCALING, atol=1e-12))}")
print(f"  dG1 + dG4 equals {SUM_TOTAL - SCALING:.2f} eV for all five        : "
      f"{bool(np.allclose(dG1 + dG4, SUM_TOTAL - SCALING, atol=1e-12))}")
print()
print("Every step is positive (no spontaneous step at U = 0) : "
      f"{bool(np.all(steps > 0.0))}")
```

**出力:**

```
Stage 2: intermediates and the four step energies (eV)
catalyst       dG_OH    dG_O   dG_OOH     dG1     dG2     dG3     dG4      sum
Catalyst A      0.60    1.60     3.80    0.60    1.00    2.20    1.12   4.9200
Catalyst B      0.80    2.20     4.00    0.80    1.40    1.80    0.92   4.9200
Catalyst C      1.00    2.65     4.20    1.00    1.65    1.55    0.72   4.9200
Catalyst D      1.10    3.00     4.30    1.10    1.90    1.30    0.62   4.9200
Catalyst E      1.30    3.60     4.50    1.30    2.30    0.90    0.42   4.9200

sum constraint: every catalyst sums to 4.92 eV : True
  largest deviation from 4.92 eV : 0.00e+00 eV
  dG4 rebuilt as SUM_TOTAL - SCALING - dG_OH agrees : True
  dG2 + dG3 equals the scaling constant 3.20 eV for all five : True
  dG1 + dG4 equals 1.72 eV for all five        : True

Every step is positive (no spontaneous step at U = 0) : True
```

**結果を読み解く。** \\(\Delta G_2\\) と \\(\Delta G_3\\) の列を並べて眺めると、溶接の跡が肉眼で見えます。5つの表面にわたって \\(\Delta G_2\\) が \\(1.00\\) から \\(2.30\\) eV へと登っていくあいだ、\\(\Delta G_3\\) は \\(2.20\\) から \\(0.90\\) へと落ちていきます — 一方の増加は、もう一方のちょうど同じだけの減少です。両者の和が \\(3.20\\) eV に釘付けにされているからです。5つの異なる材料があって、この2つの数の組が持つ自由度は、それら全体を通じてわずか *1つ* しかありません。

同じことが外側のペアでも起きており、こちらは見落としやすいものです。どの触媒でも \\(\Delta G_1 + \Delta G_4 = 4.92 - 3.20 = 1.72\\) eV となります。やはりスケーリング則が \\(\Delta G_{\ast\text{OOH}}\\) を \\(\Delta G_{\ast\text{OH}}\\) に縛りつけているからです。4つの段階、2つの溶接されたペア、2つの自由パラメータ。これらの表面が生きている設計空間は、それがすべてです。

3つの相互チェックについては一言述べておく価値があります。その非対称性こそが教訓だからです。チェック1 — 総和 — は \\(\Delta G_4\\) の構成の仕方によって **保証** されているので、`True` しか印字しえず、物理については何も検証しません。その価値は狭いながらも本物です。壊れた配列の形や、転置の向きを誤った `vstack` を捕まえてくれます。意味があるのはチェック2のほうで、こちらは別の代数的経路で \\(\Delta G_4\\) に辿り着きます。これほど小さなパイプラインではこの区別は衒学的ですが、候補が1000あってモジュールが6つあるパイプラインでは、これはテストスイートと飾りの違いになります。

## 4.4 ステージ3: 自由エネルギー図

自由エネルギー図は電極触媒機構の標準的な描像であり、そこから取り出されるどんな単一の数値よりも多くを語ります。

累積自由エネルギーを反応の進行に対してプロットします。清浄表面をゼロとして出発し、\\(\Delta G_1\\) を足して \\(\ast\text{OH}\\) に到達し、\\(\Delta G_2\\) を足して \\(\ast\text{O}\\) に到達し、以下同様です。印加電位 \\(U\\) のもとでは、すでに移動した電子1つごとに現在の準位が \\(eU\\) だけ下がるので、\\(i\\) 段階後の準位は次のようになります。

\\[ G_i(U) = \sum_{j \le i} \Delta G_j - i\,eU \\]

印加電位は階段全体を、一段ずつ一様に、下向きへ傾けます。上を向いた段が1つもなくなったとき触媒作用が始まります — それが、ステージ4で取り出す制限電位の定義です。

各触媒について図を2回描きます。\\(U = 0\\) では図は純粋な化学であり、\\(U = 1.23\\) V では熱力学的な最小限が供給済みで、それでもなお上り坂であるものは触媒自身の責任です。

```python
# ------------------------------------------------------------------
# ステージ3: 自由エネルギー図。
#
# 印加電位Uでは、移動した電子1つごとに状態がeUだけ下がるので、
# i段階後の準位は
#     G_i(U) = sum(dG_1..dG_i) - i * e * U
# となる。図はこの準位の列であり、隣り合う差がすべて0以下のとき、
# 触媒は「すべて下り坂」である。
# ------------------------------------------------------------------
def profile(step_row, U):
    """電位Uにおける累積自由エネルギー準位。0から出発する。"""
    return np.concatenate([[0.0], np.cumsum(step_row) - U * np.arange(1, 5)])

LEVEL_LABELS = ["*", "*OH", "*O", "*OOH", "* + O2"]

for U in (0.0, E_EQ):
    print(f"Stage 3: free-energy profile at U = {U:.2f} V  (levels in eV)")
    print(f"{'catalyst':<12} " + " ".join(f"{lab:>9}" for lab in LEVEL_LABELS)
          + "   largest uphill step")
    for k, name in enumerate(names):
        lv = profile(steps[k], U)
        rises = np.diff(lv)
        j = int(np.argmax(rises))
        print(f"{name:<12} " + " ".join(f"{v:9.2f}" for v in lv)
              + f"   {STEP_LABELS[j]}  (+{rises[j]:.2f} eV)")
    print()

# U = E_EQでの終点は、総和則を反対側から見たチェックになっている。
# 1.23 Vにおける4個の電子が、ちょうど4.92 eVを支払い切る。
final_at_eq = np.array([profile(steps[k], E_EQ)[-1] for k in range(len(names))])
print(f"at U = {E_EQ:.2f} V every profile ends at 0.00 eV : "
      f"{bool(np.allclose(final_at_eq, 0.0, atol=1e-12))}")
print(f"  largest final level : {np.max(np.abs(final_at_eq)):.2e} eV")

# U = E_EQでまだ上り坂なのはどの段階か。その段階こそがボトルネックである。
print()
print(f"{'catalyst':<12} " + " ".join(f"{lab:>16}" for lab in STEP_LABELS))
for k, name in enumerate(names):
    rises = np.diff(profile(steps[k], E_EQ))
    print(f"{name:<12} " + " ".join(f"{r:16.2f}" for r in rises))
print()
n_uphill = [int(np.sum(np.diff(profile(steps[k], E_EQ)) > 1e-12))
            for k in range(len(names))]
print(f"steps still uphill at U = {E_EQ:.2f} V, per catalyst : {n_uphill}")
print("every catalyst has at least one uphill step at the equilibrium "
      f"potential : {all(n > 0 for n in n_uphill)}")
```

**出力:**

```
Stage 3: free-energy profile at U = 0.00 V  (levels in eV)
catalyst             *       *OH        *O      *OOH    * + O2   largest uphill step
Catalyst A        0.00      0.60      1.60      3.80      4.92   3  *O -> *OOH  (+2.20 eV)
Catalyst B        0.00      0.80      2.20      4.00      4.92   3  *O -> *OOH  (+1.80 eV)
Catalyst C        0.00      1.00      2.65      4.20      4.92   2  *OH -> *O  (+1.65 eV)
Catalyst D        0.00      1.10      3.00      4.30      4.92   2  *OH -> *O  (+1.90 eV)
Catalyst E        0.00      1.30      3.60      4.50      4.92   2  *OH -> *O  (+2.30 eV)

Stage 3: free-energy profile at U = 1.23 V  (levels in eV)
catalyst             *       *OH        *O      *OOH    * + O2   largest uphill step
Catalyst A        0.00     -0.63     -0.86      0.11      0.00   3  *O -> *OOH  (+0.97 eV)
Catalyst B        0.00     -0.43     -0.26      0.31      0.00   3  *O -> *OOH  (+0.57 eV)
Catalyst C        0.00     -0.23      0.19      0.51      0.00   2  *OH -> *O  (+0.42 eV)
Catalyst D        0.00     -0.13      0.54      0.61      0.00   2  *OH -> *O  (+0.67 eV)
Catalyst E        0.00      0.07      1.14      0.81      0.00   2  *OH -> *O  (+1.07 eV)

at U = 1.23 V every profile ends at 0.00 eV : True
  largest final level : 0.00e+00 eV

catalyst        1  H2O -> *OH     2  *OH -> *O    3  *O -> *OOH    4  *OOH -> O2
Catalyst A              -0.63            -0.23             0.97            -0.11
Catalyst B              -0.43             0.17             0.57            -0.31
Catalyst C              -0.23             0.42             0.32            -0.51
Catalyst D              -0.13             0.67             0.07            -0.61
Catalyst E               0.07             1.07            -0.33            -0.81

steps still uphill at U = 1.23 V, per catalyst : [1, 2, 2, 2, 2]
every catalyst has at least one uphill step at the equilibrium potential : True
```

**結果を読み解く。** 4つあり、持ち帰るべきは3つ目です。

  * **\\(U = 0\\) では、どのプロファイルも同じ場所で終わる上り階段です。** 5つすべてが \\(4.92\\) eV まで登ります。全体の反応は、どの表面がそれを触媒したかを気にしないからです。触媒どうしの違いは、その登りを *どう配分するか* に尽きます — これは設計問題を言い換えたものでもあります。
  * **\\(U = 1.23\\) V では、どのプロファイルもちょうど \\(0.00\\) eV で終わります**。浮動小数点の誤差はゼロです。これは総和則を電気化学の側から見たものです。平衡電位における4個の電子が、ちょうど \\(4.92\\) eV を支払い切ります。同時にこれは `profile` 関数の本物のテストでもあります。この関数は、1つずれやすい添字の算術を行っているからです。
  * **火山型プロットの2つの斜面が、2種類の異なるボトルネックとして現れます。** AとBは段階3、\\(\ast\text{O} \rightarrow \ast\text{OOH}\\) でまだ上り坂です。\\(\ast\text{O}\\) を強く掴みすぎているのです — Catalyst Aを見てください。その \\(\ast\text{O}\\) 準位は \\(-0.86\\) eV と表の中で最も深い井戸であり、反応はそこから這い上がらねばなりません。C、D、Eは段階2、\\(\ast\text{OH} \rightarrow \ast\text{O}\\) で上り坂です。\\(\ast\text{O}\\) の安定化が足りないために、2つ目のプロトンを安価に引き抜けないのです。同じ溶接されたペアで、正反対の破綻の仕方。そして図は、過電圧という単一の数値には決してできない仕方で、この区別を一目瞭然にします。
  * **\\(1.23\\) V で完全に下り坂になっているものは1つもありません**。そして第3章を経た今、スケーリング則に従う表面がそうなりうることは決してないと分かっています。Catalyst Eはまた、平衡電位で *最初の* 段階が上り坂（\\(+0.07\\) eV）になっている唯一の触媒でもあります — \\(\ast\text{OH}\\) の吸着が弱すぎて、最初の中間体を作ることすら代償を伴うのです。これは弱吸着の2つ目の、独立した症状であり、\\(\eta\\) しか報告しないスクリーニングのパイプラインなら捨ててしまう類の細部です。

## 4.5 ステージ4: 過電圧とランキング

ここで各図を1つの数値へ圧縮します。

**制限電位** \\(U_L\\) は、どの段階も上り坂にならない最小の電位です。すべての段階が同じ \\(eU\\) だけ傾けられるので、これは単に最大の段階の自由エネルギー変化を電位として表したものになります。

\\[ U_L = \max_i \Delta G_i / e, \qquad \eta = U_L - 1.23\ \text{V} \\]

最大値を与える段階が **電位決定段階** です。これを \\(\eta\\) と並べて報告するのは1行のコストで、点数を診断へと変えます。材料をどちらの方向へ押すべきかを教えてくれるからです。

```python
# ------------------------------------------------------------------
# ステージ4: 理論過電圧。
#
#     U_L  = max_i dG_i / e        (制限電位: どの段階も上り坂に
#                                   ならないような最小の
#                                   U)
#     eta  = U_L - E_EQ
# ------------------------------------------------------------------
U_L = steps.max(axis=1)
eta = U_L - E_EQ
pds = [STEP_LABELS[int(np.argmax(steps[k]))] for k in range(len(names))]

print("Stage 4: limiting potential and overpotential")
print(f"{'catalyst':<12} {'U_L (V)':>9} {'eta (V)':>9}   potential-determining step")
for k, name in enumerate(names):
    print(f"{name:<12} {U_L[k]:9.3f} {eta[k]:9.3f}   {pds[k]}")
print()

order = list(np.argsort(eta, kind="stable"))
print("Ranking, best (lowest overpotential) first")
for rank, k in enumerate(order, start=1):
    print(f"  {rank}. {names[k]:<12} eta = {eta[k]:.3f} V   "
          f"limited by {pds[k]:<16} x = {descriptor[k]:.2f} eV   ({roles[k]})")
print()

eta_floor = SCALING / 2.0 - E_EQ
print(f"floor implied by the scaling relation : {eta_floor:.3f} V")
print(f"best catalyst in this set             : {names[order[0]]}, "
      f"eta = {eta[order[0]]:.3f} V")
print(f"gap between the two                   : "
      f"{eta[order[0]] - eta_floor:.3f} V")
print(f"no catalyst beats the floor           : "
      f"{bool(np.all(eta >= eta_floor - 1e-12))}")
print(f"spread across the set                 : "
      f"{eta.max() - eta.min():.3f} V")
```

**出力:**

```
Stage 4: limiting potential and overpotential
catalyst       U_L (V)   eta (V)   potential-determining step
Catalyst A       2.200     0.970   3  *O -> *OOH
Catalyst B       1.800     0.570   3  *O -> *OOH
Catalyst C       1.650     0.420   2  *OH -> *O
Catalyst D       1.900     0.670   2  *OH -> *O
Catalyst E       2.300     1.070   2  *OH -> *O

Ranking, best (lowest overpotential) first
  1. Catalyst C   eta = 0.420 V   limited by 2  *OH -> *O     x = 1.65 eV   (near the apex)
  2. Catalyst B   eta = 0.570 V   limited by 3  *O -> *OOH    x = 1.40 eV   (mid, strong-binding side)
  3. Catalyst D   eta = 0.670 V   limited by 2  *OH -> *O     x = 1.90 eV   (mid, weak-binding side)
  4. Catalyst A   eta = 0.970 V   limited by 3  *O -> *OOH    x = 1.00 eV   (binds oxygen too strongly)
  5. Catalyst E   eta = 1.070 V   limited by 2  *OH -> *O     x = 2.30 eV   (binds oxygen too weakly)

floor implied by the scaling relation : 0.370 V
best catalyst in this set             : Catalyst C, eta = 0.420 V
gap between the two                   : 0.050 V
no catalyst beats the floor           : True
spread across the set                 : 0.650 V
```

**結果を読み解く。** 頂点に近い触媒が勝ち、そして勝つ *理由* がそのすぐ隣に印字されています。

Catalyst Cの制限段階は \\(1.650\\) eV — \\(\Delta G_2\\) であり、\\(1.55\\) というその \\(\Delta G_3\\) をわずか \\(0.10\\) eV 上回っているだけです。溶接された2つの段階がほぼ等しく、これはまさに第3章が最適と見定めた条件です。和が \\(3.20\\) eV に固定されている以上、最大値が最小になるのは両者が釣り合ったときだからです。Cが優れているのは、酸素を強く吸着するからでも弱く吸着するからでもありません。酸素を *均等に* 吸着し、溶接されたペアをほぼ真ん中で分けているから優れているのです。その \\(\eta = 0.420\\) V は下限の \\(0.370\\) V のわずか \\(0.050\\) V 上にあり、この \\(0.050\\) V はまさにその不均衡 — 中央の2段階の差 \\(0.10\\) eV の半分 — に等しい値です。

このランキングは、直観への警告でもあります。Catalyst B（\\(x = 1.40\\)）とCatalyst D（\\(x = 1.90\\)）は反対側の斜面に位置し、勝つのはBです — 化学的な理由からではなく、\\(|1.40 - 1.60| = 0.20\\) が \\(|1.90 - 1.60| = 0.30\\) より小さいからです。重要なのは頂点からの距離だけであり、どちら側から近づくかは問われません。一方、強く吸着するCatalyst Aと弱く吸着するCatalyst Eは、\\(0.970\\) V と \\(1.070\\) V で最下位に沈みます — 吸着エネルギーの違いが1電子ボルトを大きく下回る集合の中で、最良と最悪の間に \\(0.650\\) V もの開きが生じているのです。吸着の小さな違いが、性能の大きな違いになる。この感度こそがスクリーニングを行う価値を生み、同時にスクリーニングの誤りを高くつくものにしています。

## 4.6 ステージ5: 火山型プロットの上へ

ステージ4は各触媒について4つの段階の自由エネルギー変化をすべて使いました。第3章は、1つの数値で足りると主張していました。ステージ5はその主張を、新しい物理を一切持ち込まずに検証します。各記述子を火山型プロットの曲線上に写し、予測される過電圧を読み取り、ステージ4と要素ごとに突き合わせるのです。

これが失敗するなら、火山型プロットが間違っているか、パイプラインが間違っているかのどちらかです。成功するなら、4次元の問題は初めから本当に1次元だったということになります。

```python
# ------------------------------------------------------------------
# ステージ5: 5つの触媒を第3章の火山型プロット上に配置する。
#
# 火山型プロットが知っているのは触媒あたり1つの数、記述子
# x = dG_O - dG_OH だけであり、そこから次を予測する。
#     eta_volcano(x) = max(x, SCALING - x) - E_EQ
# ステージ4は4つの段階エネルギーすべてを使った。両者が一致すれば、
# 記述子は本当に重要なものすべてを吸収していたことになる。
# ------------------------------------------------------------------
X_APEX = SCALING / 2.0

def eta_volcano(x):
    return np.maximum(x, SCALING - x) - E_EQ

eta_pred = eta_volcano(descriptor)
leg = ["strong-binding (left)" if x < X_APEX else
       ("weak-binding (right)" if x > X_APEX else "apex") for x in descriptor]

print("Stage 5: volcano placement")
print(f"apex at x = {X_APEX:.2f} eV, floor eta = {eta_floor:.3f} V")
print(f"{'catalyst':<12} {'x':>6} {'|x - apex|':>11} {'eta_volcano':>12} "
      f"{'eta_stage4':>11} {'activity -eta':>14}   leg")
for k, name in enumerate(names):
    print(f"{name:<12} {descriptor[k]:6.2f} {abs(descriptor[k] - X_APEX):11.2f} "
          f"{eta_pred[k]:12.3f} {eta[k]:11.3f} {-eta[k]:14.3f}   {leg[k]}")
print()

# 相互チェック1: 記述子による予測がステージ4を厳密に再現すること。
print(f"eta_volcano == eta_stage4 for all five : "
      f"{bool(np.allclose(eta_pred, eta, atol=1e-12))}")
print(f"  largest discrepancy : {np.max(np.abs(eta_pred - eta)):.2e} V")

# 相互チェック2: 2つのランキングが要素ごとに一致すること。
order_volcano = list(np.argsort(eta_pred, kind="stable"))
print(f"  stage 4 ranking : {[names[k] for k in order]}")
print(f"  stage 5 ranking : {[names[k] for k in order_volcano]}")
print(f"  identical       : {order == order_volcano}")

# 相互チェック3: 頂点からの距離だけで損失が予測できること。
print(f"  eta == floor + |x - apex| for all five : "
      f"{bool(np.allclose(eta, eta_floor + np.abs(descriptor - X_APEX), atol=1e-12))}")
print()

# 一致が自動的でない理由: 段階1と4が邪魔をしない位置に留まっている
# 必要があった。どれだけの余裕でそうなっていたかを示す。
margin = U_L - np.maximum(dG1, dG4)
print("Margin by which steps 1 and 4 stayed below the limiting step (eV)")
for k, name in enumerate(names):
    print(f"  {name:<12} max(dG1, dG4) = {max(dG1[k], dG4[k]):.2f}   "
          f"dG_max = {U_L[k]:.2f}   margin = {margin[k]:.2f}")
print(f"  all margins positive : {bool(np.all(margin > 0.0))}")
print("  (had any margin gone negative, the one-descriptor volcano would "
      "have mis-ranked that catalyst)")
```

**出力:**

```
Stage 5: volcano placement
apex at x = 1.60 eV, floor eta = 0.370 V
catalyst          x  |x - apex|  eta_volcano  eta_stage4  activity -eta   leg
Catalyst A     1.00        0.60        0.970       0.970         -0.970   strong-binding (left)
Catalyst B     1.40        0.20        0.570       0.570         -0.570   strong-binding (left)
Catalyst C     1.65        0.05        0.420       0.420         -0.420   weak-binding (right)
Catalyst D     1.90        0.30        0.670       0.670         -0.670   weak-binding (right)
Catalyst E     2.30        0.70        1.070       1.070         -1.070   weak-binding (right)

eta_volcano == eta_stage4 for all five : True
  largest discrepancy : 2.22e-16 V
  stage 4 ranking : ['Catalyst C', 'Catalyst B', 'Catalyst D', 'Catalyst A', 'Catalyst E']
  stage 5 ranking : ['Catalyst C', 'Catalyst B', 'Catalyst D', 'Catalyst A', 'Catalyst E']
  identical       : True
  eta == floor + |x - apex| for all five : True

Margin by which steps 1 and 4 stayed below the limiting step (eV)
  Catalyst A   max(dG1, dG4) = 1.12   dG_max = 2.20   margin = 1.08
  Catalyst B   max(dG1, dG4) = 0.92   dG_max = 1.80   margin = 0.88
  Catalyst C   max(dG1, dG4) = 1.00   dG_max = 1.65   margin = 0.65
  Catalyst D   max(dG1, dG4) = 1.10   dG_max = 1.90   margin = 0.80
  Catalyst E   max(dG1, dG4) = 1.30   dG_max = 2.30   margin = 1.00
  all margins positive : True
  (had any margin gone negative, the one-descriptor volcano would have mis-ranked that catalyst)
```

**結果を読み解く。** 3点あります。

  * **近道と長い計算は \\(2 \times 10^{-16}\\) V まで一致し**、2つのランキングは同一のリストになります。これは第3章の見返りを、主張としてではなく実行可能な表明として述べたものです。\\(\Delta G_{\ast\text{O}} - \Delta G_{\ast\text{OH}}\\) しか知らないモデルが、3つの吸着エネルギーと4つの段階の自由エネルギー変化を使った計算を再現します。これらの仮定のもとでは、残りの情報は本当に冗長だったのです。
  * **\\(\eta = \eta_{\text{floor}} + |x - x_{\text{apex}}|\\) が5つすべてで成り立ちます。** Catalyst Cは頂点から \\(0.05\\) eV のところにあり \\(0.05\\) V を支払い、Catalyst Eは \\(0.70\\) eV 離れていて \\(0.70\\) V を支払います。交換レートは1対1であり、頂上に平坦部はありません。CがBと反対の斜面にありながらBを打ち負かす理由もこれです。\\(0.05 < 0.20\\)、それ以外は何も入ってきません。
  * **この一致は自動的なものではなく、最後のブロックがその理由を語っています。** 段階1と段階4は、記述子からは見えません。どちらかが電位決定段階になるほど大きくなっていたら、火山型プロットは \\(\eta\\) を過小に予測し、ランキングの順序を入れ替えてしまっていたかもしれません。印字された余裕 — \\(0.65\\) から \\(1.08\\) eV — は、それらが問題になるまでにどれだけの距離があったかを示しています。これは第3章が計算した妥当性の窓と同じものを、いま触媒ごとに評価したものです。**実際のスクリーニングでは、必ずこの余裕を計算してください。** 3行で済み、そしてそれが、信頼できる記述子と、そうであってほしいと願うだけの記述子との違いになります。

## 4.7 実際のスクリーニングが加えるもの

このパイプラインは論理において完結しており、入力において空洞です。これを研究の道具にするものはすべて、「説明用の値」と「計算されたエネルギー」の間の隙間に収まっています。ですから、その隙間を何が埋めるのかを具体的に述べておく価値があります。

**教育用の値ではなくDFTのエネルギー。** \\(\Delta G_{\ast\text{OH}}\\) と \\(\Delta G_{\ast\text{O}}\\) のそれぞれがスラブ計算になります。表面を作り、ファセットと終端を選び、被覆率を決め、構造を緩和し、電子エネルギーを自由エネルギーに変えるために第2章で導入した零点補正とエントロピー補正を加える。これは候補1つあたり数時間の計算と十数個の人間の判断であり、しかもその判断 — どのファセットか、どの終端か、どの被覆率か — は、汎関数の選択よりも結果を大きく動かすことがしばしばあります。

**誤差棒と、それを伝播させる誠実さ。** 4.5節では、結合エネルギーの違いが1電子ボルトに満たない触媒群にわたって \\(0.650\\) V の開きが現れました。ここで、DFTの吸着エネルギーが日常的に数十分の1電子ボルトの不確かさを伴うこと、そしてスケーリング則自体が \\(3.2\\) eV というオフセットのまわりにばらつきを持つことに注意してください。これらを4.6節の1対1の交換レートと合わせると、居心地の悪い結論が出てきます。**\\(\eta\\) の不確かさは、あなたが順位づけに使っている差と同程度の大きさである** ということです。ここではCatalyst CがCatalyst Bを \\(0.15\\) V 上回りました。実際の研究では、その差は誤差棒との接触を生き延びないかもしれません。そして誤差棒なしにランキングを報告するスクリーニングは、擁護できないランキングを報告しているのです。実際のパイプラインの正しい出力は「Cが最良である」ではなく、「C、B、Dは区別できない。AとEは明らかに劣る」です。

**安定性のフィルタ。たいていはこちらが本当のボトルネックです。** 本章では、これらの表面のいずれかが実運転の電位と酸性度を生き延びるかどうかを、一度も問いませんでした。計算上もっとも有望なOER候補の多くは、運転条件下で溶解し、意図した状態を通り越して酸化され、あるいはまったく別のものへ再構成されます。活性のスクリーニングはリストを作り、安定性のスクリーニング — プールベ図の解析、溶解電位、表面再構成 — がそれを短くします。そして後者は前者よりはるかに多くの候補を脱落させます。\\(\eta\\) だけで順位づけするパイプラインは、問いの半分にしか答えていません。

**速度論。枠組み全体が静かに無視し続けてきたものです。** 理論過電圧は熱力学的な量です。各段階が下り坂かどうかは問いますが、どれだけ速いかは決して問いません。現実の活性化障壁、被覆率の効果、電解液、物質輸送は、いずれもモデルの外にあります。同じ \\(\eta\\) を持つ2つの触媒が、測定される電流では桁違いに異なりうるのです。

そして、そのすべての下敷きになっている仮定があります。本章のあらゆる数値はスケーリング則の下流にあり、私たちはそれを近似的な経験則として注意深く留保をつけたうえで、まるで厳密であるかのように使いました — スクリーニングのパイプラインとはそういうものだからです。5つの触媒は、構成上それに完璧に従いました。現実の表面はそのまわりにばらつき、そして興味深いのは外れ値のほうなのです。

第5章はそこから始まります。この問題を機械学習モデルに手渡したとき何が起こるのかを問うのです。学習された記述子が、導出された記述子に対して何を買い足すのか。スケーリング則に従順な表面で訓練されたモデルが、なぜ自らには問い直しようのない火山型プロットを自信たっぷりに再現してしまうのか。そして計算水素電極の限界 — 第2章で受け入れ、以来ずっと積み重ねてきたもの — が、このようにして作られたデータからモデルが下すあらゆる予測へどう伝播していくのか。それは、他の4章を監査する章です。

### 🎯 演習問題

  1. **6番目の候補。** \\(\Delta G_{\ast\text{OH}} = 0.95\\) eV、\\(\Delta G_{\ast\text{O}} = 2.55\\) eV を持つCatalyst Fを加えてください。コードを走らせる前にその過電圧と順位を予測し、そのうえで確認してください。下限にどれだけ近づくでしょうか。また、ちょうど頂点に位置するためには \\(\Delta G_{\ast\text{O}}\\) はいくつでなければならないでしょうか。
  2. **記述子を壊す。** \\(x = 1.60\\) eV — 頂点そのもの — でありながら \\(\Delta G_{\ast\text{OH}} = 1.85\\) eV である候補を構成してください。それに対してステージ2から5までを走らせてください。ステージ5のどの相互チェックが最初に失敗し、余裕の計算は何を報告するでしょうか。
  3. **経験的な定数への感度。** スケーリングのオフセットを \\(3.0\\) eV と \\(3.4\\) eV に設定して、パイプライン全体を走らせ直してください。5つのうちどの触媒が順位を変えるでしょうか。また、電位決定段階の正体が、一部の触媒では変わり他では変わらないのはなぜでしょうか。
  4. **誤差棒を、粗く。** 各 \\(\Delta G_{\ast\text{OH}}\\) と \\(\Delta G_{\ast\text{O}}\\) に標準偏差 \\(0.15\\) eV の独立なガウス雑音を加え、ランキングを10,000回繰り返して、各触媒が1位になる頻度を報告してください。本章の結論を、順序ではなく確率として述べ直してください。
  5. **走らせなかったチェック。** ステージ3は、Catalyst Eが \\(U = 1.23\\) V で最初の段階が上り坂になっていると報告しました。\\(\Delta G_1\\) または \\(\Delta G_4\\) がその制限段階の \\(0.20\\) eV 以内に迫っている候補に印をつける診断を書き、記述子モデルが利用できる場合であっても、そうした候補が4段階すべてによる扱いに値する理由を説明してください。

## まとめ

本章では、NumPyで5ステージのOERスクリーニングのパイプラインを組み立て、第3章の火山型プロット上の5つの位置を占めるように選ばれた、**説明用の吸着自由エネルギーを持つ5つの架空の触媒** の上で端から端まで走らせました。ステージ1はデータセットと、その記述子 \\(x = \Delta G_{\ast\text{O}} - \Delta G_{\ast\text{OH}}\\) を定義し、記述子は頂点 \\(1.60\\) eV のまわりで \\(1.00\\) から \\(2.30\\) eV に広がりました。ステージ2はスケーリング則を適用して \\(\Delta G_{\ast\text{OOH}}\\) を予測し、4つの段階の自由エネルギー変化をすべて構築して、\\(4.92\\) eV の総和が浮動小数点誤差ゼロで成り立つことを検証し、\\(\Delta G_4\\) を独立な経路で組み立て直し、2つの溶接を直接に示しました。すなわち、どの表面についても \\(\Delta G_2 + \Delta G_3 = 3.20\\) eV かつ \\(\Delta G_1 + \Delta G_4 = 1.72\\) eV です。ステージ3は \\(U = 0\\) と \\(U = 1.23\\) V における累積自由エネルギーのプロファイルを描き — 平衡電位ではそのすべてがちょうど \\(0.00\\) eV で終わり、これは総和則を電気化学の側から見たチェックです — 火山型プロットの2つの斜面が2種類の異なるボトルネックとして現れることを示しました。強く吸着するAとBでは段階3、C、D、Eでは段階2です。ステージ4は制限電位と過電圧を取り出し、**C（0.420 V）< B（0.570）< D（0.670）< A（0.970）< E（1.070）** と順位づけました。頂点に近い触媒Cが勝ったのは、溶接された2つの段階がほぼ等しく（\\(1.65\\) 対 \\(1.55\\) eV）、\\(0.370\\) V の下限のわずか \\(0.050\\) V 上に位置していたからです。ステージ5は記述子だけを使って5つを火山型プロット上に写し、ステージ4の過電圧を \\(2 \times 10^{-16}\\) V まで、そしてそのランキングを厳密に再現し、5つすべてで \\(\eta = \eta_{\text{floor}} + |x - x_{\text{apex}}|\\) が成り立つことを確認し、段階1と4が制限段階から離れて留まっていた触媒ごとの余裕（\\(0.65\\) から \\(1.08\\) eV）を印字しました — 記述子1つの近道を、幸運ではなく妥当なものにしていた条件です。最後に、実際のスクリーニングが何を加えるのかを名指ししました。ファセット・終端・被覆率の判断を伴うDFTのエネルギー、順位づけの対象である差と同程度の大きさを持つ誤差棒、活性よりも多くの候補を脱落させる安定性のフィルタ、そして、この枠組みがまったくモデル化しなかった速度論です。

第5章は、この構成全体に監査の目を向けます。これほど制約された問題に機械学習が何を加えるのか、そして何を暗黙のうちに受け継いでしまうのか — 計算水素電極から、スケーリング則から、そして行儀の良い表面で訓練されたモデルには、本当に重要であるはずの外れ値を認識する手立てがないという事実から。

[← 第3章: スケーリング則と火山型プロット](<chapter-3.html>) [第5章: 機械学習スクリーニングとCHEの限界 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
