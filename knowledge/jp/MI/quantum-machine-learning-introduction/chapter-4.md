---
title: "第4章: 変分量子回路によるML"
chapter_title: "第4章: 変分量子回路によるML"
subtitle: 学習する回路、それを訓練する勾配、そして同規模ネットワークとの誠実な比較
reading_time: 45-50分
difficulty: 上級
code_examples: 7
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/MI/quantum-machine-learning-introduction/chapter-4.html>) | Last sync: 2026-08-13

[マテリアルズ・インフォマティクス道場](<../index.html>) > [量子機械学習入門](<index.html>) > 第4章

第2章と第3章の量子モデルは一発で作れました。符号化を選び、それが誘導するカーネルを書き下し、線形方程式を解く。訓練するものは何もありませんでした。本章では訓練します。変分量子回路（VQC）は自分自身のパラメータを持ち、そのパラメータはニューラルネットワークの重みとまったく同じように勾配降下でデータに当てはめられます。

そのためVQCは、量子機械学習のなかで最も見慣れた対象であり、この分野の活動と過剰な主張がもっとも集中している場所でもあります。道具はすでに揃っています。姉妹コース[量子コンピューティング入門](<../../FM/quantum-computing-introduction/index.html>)は第3章で変分固有値ソルバを作りましたが、VQCはコスト関数がエネルギーではなく残差の和になったVQEにすぎません。回路も勾配則も古典最適化器も同じです。

ですから本章は、重要でありながら文献では飛ばされがちな2つの問いに労力を割けます。第一に、動くのか。答えはここでは、同じパラメータ数のNumPy製ニューラルネットワークを相手に、同じ最適化器で、同じ選択手順で、同じ40点の上で測られ、どちらに転んでも公表されます。第二に、規模を大きくすると何が壊れるのか。姉妹コースの barren plateau が再登場します。しかも新しい成分が1つ加わり — コストがパラメータだけでなくデータについての平均にもなる — ノートPC上で有効数字3桁まで測定できます。

## 学習目標

本章を読み終えると、次のことができるようになります。

  * 変分量子モデルを3つの独立した設計判断（符号化、変分ブロック、読み出し）に分解し、それぞれがモデルのどの性質を支配するかを述べられる
  * VQEとVQCの対応を、姉妹コースの道具をそのまま再利用できる精度で書き下し、両者が本質的に異なる唯一の箇所を指摘できる
  * 1パラメータの期待値が純粋な第1高調波であるという事実から parameter-shift 則を導き、実機では有限差分と等価ではなく明確に優れている理由を説明できる
  * *沈黙*パラメータ（勾配が恒等的に零になる回路の角度）を検出し、読み出し観測量のClifford引き戻しからその存在を説明できる
  * 敵対的な査読に耐える古典比較を実行できる。パラメータ数を揃え、最適化器を1つに固定し、選択手順を1つに固定し、全ての初期値を報告する
  * 勾配分散の幅・深さ・読み出しの局所性・エンタングルメント依存性を測定し、測定した減衰率を30量子ビットと50量子ビットでのショット予算に換算できる
  * VQCの容量と表現力を独立に調整できない理由と、それが正則化に何を意味するかを説明できる

### 記法

**量子ビットの順序、ゲート、シミュレータ**は姉妹コースに完全に従います。量子ビット0が左端かつ最上位ビットであり、姉妹コース第2章のシミュレータのうち本章が必要とする関数を Code Example 1 で逐語再掲して本章を自己完結させます。`sample()` は意図的に省いています。ここでは測定結果を1つも引かず、ショット数はすべてシミュレーションではなく算術だからです。回転は半角の規約 $R_y(\theta) = \exp(-i\theta Y/2)$ を用いるので、$R_y(\theta)$ はBlochベクトルを $\theta$ ラジアン回し、振幅には $\cos(\theta/2)$ が現れます。

**データセット**は第1章で定義した合成組成記述子のデータで、Code Example 1 にバイト単位で再掲します。60点、$[0,1]$ の記述子4個、なめらかな非線形の目的変数に弱いノイズ、訓練は先頭40行、テストは末尾20行です。本章のすべての数値はこのデータとこの分割から出ています。

**記述子の周波数**は $x$ の単位あたりのサイクル数で表します。符号化ゲート $R_y(\pi x)$ は振幅に $\cos(\pi x/2)$ を寄与し、これは単位区間で4分の1サイクルです。期待値は振幅の*積*なので、高調波は $1/2$ の倍数で現れます。この2倍の落とし穴は姉妹コースが警告しているものと同じであり、以下のコードで $\pi$ の因子をすべて明示的に書いているのも同じ理由です。

* * *

## 4.1 基底状態からデータセットへ

### VQEを3つの部品として思い出す

姉妹コースの変分固有値ソルバには、動く部品がちょうど3つあります。

  1. **パラメータ付き状態** $|\psi(\boldsymbol\theta)\rangle = U(\boldsymbol\theta)|0\cdots0\rangle$。ゲート角がパラメータになっている回路が作ります。
  2. **固定した観測量** $H$。その状態上で測って実数1つ $E(\boldsymbol\theta) = \langle\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)\rangle$ を得ます。
  3. **古典最適化器**。量子デバイスをブラックボックスの関数評価器として呼びながら $E$ を $\boldsymbol\theta$ について最小化します。

変分量子モデルはここで1点だけ変えます。状態がパラメータに加えて入力にも依存することを許し、コストが1つの期待値ではなく多数の入力にわたる和になります。

$$
\text{VQE:}\quad E(\boldsymbol\theta) = \langle\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)\rangle \cr
\text{VQC:}\quad f(\mathbf{x};\boldsymbol\theta) = \sum_q w_q\,\langle\psi(\mathbf{x},\boldsymbol\theta)|Z_q|\psi(\mathbf{x},\boldsymbol\theta)\rangle + b,\qquad
\mathcal{L}(\boldsymbol\theta,\mathbf{w},b) = \frac{1}{N}\sum_{i=1}^{N}\bigl(f(\mathbf{x}_i;\boldsymbol\theta) - y_i\bigr)^2
$$

対応は表にする価値があるほど近いものです。左の列にあるものはすべて既に作ってあります。

| 量子コンピューティング入門のVQEでは | VQCでは |
| --- | --- |
| Ansatz $U(\boldsymbol\theta)$ | 同じ回路に符号化ゲートを挟んだもの |
| ハミルトニアン $H$（Pauli文字列の和） | 読み出し観測量、通常は数個の1量子ビット $Z_q$ |
| パラメータ1組につきエネルギー1つ | データ点1つにつき期待値1つ、コスト1回に $N$ 個 |
| コスト = エネルギー | コスト = データ全体の平均二乗残差 |
| parameter-shift勾配 | 同一。ただしデータの和の内側で使う |
| 収束の目標: chemical accuracy | 収束の目標: 未知データへの汎化 |
| 失敗の形: barren plateau、測定の壁 | それら2つに加えて過学習 |

最後の行が唯一の本質的な違いであり、それは大きな違いです。VQEにはテストセットがありません。最小化しているエネルギーがそのまま関心のある量なので、数値が小さいことは無条件に良いことです。VQCの訓練損失は関心のある量では*なく*、訓練損失が小さいことはしばしば悪いことです。4.5節はまさにそれを測ります。

### 3つの設計判断、3つの異なる役割

VQCは3つの部品から作られており、その役割の混同はこの分野で労力を無駄にする最大の原因です。

**符号化**は*そもそもモデルがどの関数を表現できるか*を決めます。第2章で確立したように、data re-uploading 回路は記述子についての打ち切りFourier級数を作り、その周波数の台は符号化ゲートと再アップロード回数だけで決まります。変分パラメータをどう選んでも、符号化が供給していない周波数を作ることはできません。目的関数が台の外の周波数に構造を持つなら、モデルはそれに当てはめられず、いくら訓練しても解決しません。

**変分ブロック**は*その張る空間の内側でどの関数に到達できるか*と、到達可能集合がどうパラメータ化されるかを決めます。訓練される角度はここにあり、最適化器が動かすのもここです。浅すぎるブロックは張る空間の薄い部分集合にしか届かず、深すぎるブロックは全体に届く代わりに訓練できません。その理由は4.4節が測定します。

**読み出し**は*訓練可能性とショットコスト*を決めます。量子ビットごとに局所観測量を1つ測るのは安価で、勾配は1のオーダーになります。1つの基底状態への射影演算子 — 大域観測量 — を測ると、勾配の大きさ、したがってショット数に量子ビット数の指数がかかります。これは些細な実装の詳細ではなく、訓練できるモデルと訓練できないモデルの違いです。

本章で用いる回路は、この3つを1層ずつ交互に並べます。記述子を再アップロードし、CNOTのリングでエンタングルさせ、量子ビットごとに訓練可能な回転を2つ置く。4量子ビット3層で角度は24個、読み出し重み4個とオフセット1個を加えて訓練可能パラメータは29個になります。

### Code Example 1: シミュレータとデータセット

以下はすべて1つのPythonセッションで、NumPyだけを使って動きます。シミュレータは姉妹コース第2章で構築したもので、本章が必要とする関数を無変更で再掲して本章を自立させています（`sample()` は測定結果を引かないので除いています）。データセットは本コース第1章で定義したものをシードごと厳密に再現しています。

```python
"""第4章の準備を1ブロックで: 姉妹コースのミニシミュレータの逐語再掲（sample() は
ここでは不要なので除く）と、
第1章で定義した合成材料データセット。

シミュレータは『量子コンピューティング入門』第2章で構築したものです
（ビッグエンディアン: 量子ビット0が左端＝最上位）。NumPyのみを使います。
"""
import numpy as np

# ---- 1量子ビットゲート --------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    e = np.exp(-1j * theta / 2)
    return np.array([[e, 0], [0, np.conj(e)]], dtype=complex)


# ---- 状態 ---------------------------------------------------------------
def ket(bits: str) -> np.ndarray:
    """'01' -> 4次元の基底状態 |01>（ビッグエンディアン）"""
    n = len(bits)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[int(bits, 2)] = 1.0
    return psi


def apply_gate(state, U, targets, n):
    """n量子ビット状態の targets に 2^k x 2^k ユニタリ U を作用させる"""
    k = len(targets)
    psi = state.reshape([2] * n)          # 1. n添字テンソルとして見る
    psi = np.moveaxis(psi, targets, range(k))   # 2. 標的軸を先頭へ
    rest = psi.shape[k:]
    psi = psi.reshape(2 ** k, -1)         # 3. 平坦化して行列積
    psi = U @ psi
    psi = psi.reshape(list((2,) * k) + list(rest))
    psi = np.moveaxis(psi, range(k), targets)   # 4. 軸を元に戻す
    return psi.reshape(-1)


CNOT4 = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=complex)


def cnot(state, control, target, n):
    """任意の量子ビット対・任意の向きのCNOT"""
    return apply_gate(state, CNOT4, [control, target], n)


def probs(state):
    """Born則による全 2^n 通りの確率"""
    return np.abs(state) ** 2


PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}


def expval(state, pauli, coeff_map=None):
    """'ZZ' や 'XI' のようなPauli文字列（1量子ビット1文字）の期待値。

    coeff_map を与えると結果に coeff_map[pauli] を掛けるので、ハミルトニアン
    全体が1行で書ける:  sum(expval(psi, p, terms) for p in terms)
    """
    n = len(pauli)
    phi = state.copy()
    for q, ch in enumerate(pauli):
        if ch != 'I':
            phi = apply_gate(phi, PAULI[ch], [q], n)
    val = np.vdot(state, phi).real
    if coeff_map is not None:
        val *= coeff_map.get(pauli, 1.0)
    return val


# ---- 本コース全章で共通のデータセット ------------------------------------
def make_materials_dataset(n=60, seed=7):
    """組成記述子から形成エネルギー風の物性値への合成回帰データ。
    記述子は [0,1] の4次元。滑らかな非線形ターゲット＋弱いノイズ。決定的。"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (n, 4))
    y = (np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])
         + 0.5 * X[:, 2]**2 - 0.3 * X[:, 3]
         + 0.05 * rng.standard_normal(n))
    return X, y


Xall, yall = make_materials_dataset()
Xtr, ytr = Xall[:40], yall[:40]      # 訓練 = 先頭40行
Xte, yte = Xall[40:], yall[40:]      # テスト = 末尾20行

print("合成材料データセット（本コース全章で共通）")
print("-" * 70)
print(f"  記述子の数            {Xall.shape[1]}")
print(f"  訓練 / テスト         {len(ytr)} / {len(yte)}")
print(f"  y の範囲              [{yall.min():.4f}, {yall.max():.4f}]")
print(f"  y の平均 / 標準偏差   {yall.mean():.4f} / {yall.std():.4f}")
print(f"  訓練 y 平均 / 標準偏差 {ytr.mean():.4f} / {ytr.std():.4f}")
print(f"  テスト y 平均 / 標準偏差 {yte.mean():.4f} / {yte.std():.4f}")

print("\n以降の全ての数値が超えるべき2つのベースライン")
print("-" * 70)
mse_const = np.mean((yte - ytr.mean()) ** 2)
A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
coef = np.linalg.lstsq(A, ytr, rcond=None)[0]
pred_lin = np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef
mse_lin = np.mean((yte - pred_lin) ** 2)
print(f"  訓練データの平均を返すだけ    テストMSE = {mse_const:.4f}")
print(f"  最小二乗法（線形回帰）        テストMSE = {mse_lin:.4f}")
print(f"  除去できないノイズの下限      MSE      = {0.05**2:.4f}")

print("\n先頭3行を材料の表として表示")
print("-" * 70)
print(f"  {'x1':>8}{'x2':>8}{'x3':>8}{'x4':>8}{'y':>10}")
for i in range(3):
    print("  " + "".join(f"{v:8.4f}" for v in Xall[i]) + f"{yall[i]:10.4f}")
```

```text
合成材料データセット（本コース全章で共通）
----------------------------------------------------------------------
  記述子の数            4
  訓練 / テスト         40 / 20
  y の範囲              [-1.2198, 1.2142]
  y の平均 / 標準偏差   -0.0336 / 0.5446
  訓練 y 平均 / 標準偏差 -0.0883 / 0.5585
  テスト y 平均 / 標準偏差 0.0759 / 0.4978

以降の全ての数値が超えるべき2つのベースライン
----------------------------------------------------------------------
  訓練データの平均を返すだけ    テストMSE = 0.2747
  最小二乗法（線形回帰）        テストMSE = 0.0465
  除去できないノイズの下限      MSE      = 0.0025

先頭3行を材料の表として表示
----------------------------------------------------------------------
        x1      x2      x3      x4         y
    0.6251  0.8972  0.7757  0.2252   -0.6472
    0.3002  0.8736  0.0053  0.8212   -0.9934
    0.7971  0.4679  0.3030  0.2784   -0.0502
```

**注目すべき点。** 量子モデルを1行も書く前に2つのベースラインを確立します。本章が誠実な結論に到達できるのはこのおかげです。訓練データの平均を返すだけのテストMSEは 0.2747 で、これがモデルが「何かを学習した」と言うために超えるべき数値です。最小二乗法は5パラメータで 0.0465、6倍近く良い数値です。この2番目の数値は居心地が悪く、そうであるべきものです。目的変数は $\sin(\pi x_1)\cos(\pi x_2)$ を含んでおり強く非線形ですが、$\cos(\pi x_2)$ は $[0,1]$ 上でほぼ線形であり、$\sin(\pi x_1)$ はほぼ平均だけを寄与するため、線形当てはめが分散の大部分を捉えてしまいます。線形ベースラインを省いた比較は2つの非線形モデルを実際より良く見せますし、その省略はよく行われます。

ノイズの下限は 0.0025 です。これより良くできるモデルはなく、0.0465 と 0.0025 の間の距離が、本章の残りが争う全空間です。

### Code Example 2: VQC回帰器

```python
"""回帰モデルとしての変分量子回路。
Code Example 1 の続き（同一セッション）。
"""
N_QUBITS = 4          # 記述子1つにつき量子ビット1つ
N_LAYERS = 3          # 符号化 + エンタングル + 変分回転 を3回
LOCAL_Z = [''.join('Z' if i == q else 'I' for i in range(N_QUBITS))
           for q in range(N_QUBITS)]


def vqc_state(x, theta, layers=N_LAYERS):
    """data re-uploading型VQC。各層で x を再アップロード、エンタングル、回転。"""
    n = len(x)
    psi = ket('0' * n)
    k = 0
    for _ in range(layers):
        for q in range(n):                                  # 符号化層
            psi = apply_gate(psi, ry(np.pi * x[q]), [q], n)
        for q in range(n):                                  # エンタングルのリング
            psi = cnot(psi, q, (q + 1) % n, n)
        for q in range(n):                                  # 変分層
            psi = apply_gate(psi, rz(theta[k]), [q], n); k += 1
            psi = apply_gate(psi, ry(theta[k]), [q], n); k += 1
    return psi


def n_theta(layers=N_LAYERS, n=N_QUBITS):
    """回路の角度パラメータ数: 1層あたり1量子ビット2回転。"""
    return 2 * n * layers


def vqc_features(x, theta, layers=N_LAYERS):
    """測定量: 各量子ビットの <Z_q>。いずれも [-1, +1] に収まる。"""
    psi = vqc_state(x, theta, layers)
    return np.array([expval(psi, p) for p in LOCAL_Z])


def unpack(params, n=N_QUBITS):
    """params = [回路の角度 ..., w_0..w_{n-1}, b]。"""
    return params[:-(n + 1)], params[-(n + 1):-1], params[-1]


def vqc_predict(X, params, layers=N_LAYERS):
    """モデル出力: y_hat = sum_q w_q <Z_q> + b。"""
    theta, w, b = unpack(params)
    return np.array([w @ vqc_features(x, theta, layers) + b for x in X])


rng = np.random.default_rng(11)
theta0 = rng.uniform(0, 2 * np.pi, n_theta())
params0 = np.concatenate([theta0, np.full(N_QUBITS, 0.5), [0.0]])

print("VQC回帰器: 構造とパラメータ数")
print("-" * 74)
print(f"  量子ビット数                        {N_QUBITS}")
print(f"  層数（符号化 + エンタングル + 回転） {N_LAYERS}")
print(f"  回路の角度パラメータ                {n_theta()}")
print(f"  読み出し重み w_q とオフセット b     {N_QUBITS + 1}")
print(f"  訓練可能パラメータ合計              {len(params0)}")
gates = N_LAYERS * (N_QUBITS + N_QUBITS + 2 * N_QUBITS)
print(f"  1回の順伝播あたりのゲート数         {gates}"
      f"  ({N_LAYERS} x ({N_QUBITS} Ry符号化 + {N_QUBITS} CNOT"
      f" + {2*N_QUBITS} 回転))")
print(f"  状態ベクトルの次元                  {2**N_QUBITS}")
print(f"  1回の順伝播で測る観測量             {N_QUBITS} 個（すべて局所、weight 1）")

print("\n同じものをVQE（量子コンピューティング入門 第3章）として見る")
print("-" * 74)
print("  VQE:  E(theta)   = <psi(theta)| H |psi(theta)>,  H は固定、theta で最小化")
print("  VQC:  f(x,theta) = sum_q w_q <psi(x,theta)| Z_q |psi(x,theta)> + b。")
print("        この期待値がデータ点ごとに1つあり、損失はデータ全体の和になる。")
print("  回路もparameter-shift勾配も古典最適化器も同一。変わったのはコスト関数だけ")
print("  で、1つのエネルギーが40個の残差になった。")

print("\n全ての角度が出力を動かすか: 沈黙パラメータの検査")
print("-" * 74)
probe = np.random.default_rng(3).uniform(0, 1, (6, N_QUBITS))
sens = np.zeros(n_theta())
for xq in probe:
    for k in range(n_theta()):
        tp = theta0.copy(); tp[k] += np.pi / 2
        tm = theta0.copy(); tm[k] -= np.pi / 2
        d = 0.5 * (vqc_features(xq, tp) - vqc_features(xq, tm))
        sens[k] = max(sens[k], np.abs(d).max())
print(f"  どの <Z_q> にも全く影響しない角度: "
      f"{int(np.sum(sens < 1e-12))} / {n_theta()}")
print(f"  感度の最小 / 最大: {sens.min():.4f} / {sens.max():.4f}")
print("  ここでは順序が効く。回転してからエンタングルし、読み出しを1つの大域観測量")
print("  にすると、5つの角度が原理的に沈黙する。Cliffordのリングは Pauli-Z 列を")
print("  台がより小さい別の Pauli-Z 列へ引き戻すからである。")

print("\nモデルは記述子の有界な三角関数である")
print("-" * 74)
print(f"  {'x1':>6}{'<Z0>':>11}{'<Z1>':>11}{'<Z2>':>11}{'<Z3>':>11}{'y_hat':>11}")
base = np.array([0.5, 0.5, 0.5, 0.5])
for xv in np.linspace(0.0, 1.0, 6):
    xq = base.copy(); xq[0] = xv
    z = vqc_features(xq, theta0)
    print(f"  {xv:6.2f}" + "".join(f"{v:11.6f}" for v in z)
          + f"{params0[-5:-1] @ z + params0[-1]:11.6f}")

pred = vqc_predict(Xtr[:5], params0)
print("\n未訓練の出力（訓練データ先頭5行）")
print("-" * 74)
print(f"  {'y':>10}{'y_hat':>10}{'residual':>12}")
for a, b_ in zip(ytr[:5], pred):
    print(f"  {a:10.4f}{b_:10.4f}{a-b_:12.4f}")
print(f"\n  未訓練の訓練MSE = {np.mean((ytr - vqc_predict(Xtr, params0))**2):.4f}")
print(f"  |<Z_q>| <= 1 が常に成り立つので、出力の到達範囲は w_q が決める。すべての")
print(f"  w_q = 0.5 では絶対値2.0を超えられないが、目的変数は {yall.max():.4f} に達する。")
print(f"  読み出し重みは飾りではなく必須である。")
```

```text
VQC回帰器: 構造とパラメータ数
--------------------------------------------------------------------------
  量子ビット数                        4
  層数（符号化 + エンタングル + 回転） 3
  回路の角度パラメータ                24
  読み出し重み w_q とオフセット b     5
  訓練可能パラメータ合計              29
  1回の順伝播あたりのゲート数         48  (3 x (4 Ry符号化 + 4 CNOT + 8 回転))
  状態ベクトルの次元                  16
  1回の順伝播で測る観測量             4 個（すべて局所、weight 1）

同じものをVQE（量子コンピューティング入門 第3章）として見る
--------------------------------------------------------------------------
  VQE:  E(theta)   = <psi(theta)| H |psi(theta)>,  H は固定、theta で最小化
  VQC:  f(x,theta) = sum_q w_q <psi(x,theta)| Z_q |psi(x,theta)> + b。
        この期待値がデータ点ごとに1つあり、損失はデータ全体の和になる。
  回路もparameter-shift勾配も古典最適化器も同一。変わったのはコスト関数だけ
  で、1つのエネルギーが40個の残差になった。

全ての角度が出力を動かすか: 沈黙パラメータの検査
--------------------------------------------------------------------------
  どの <Z_q> にも全く影響しない角度: 0 / 24
  感度の最小 / 最大: 0.0216 / 0.7969
  ここでは順序が効く。回転してからエンタングルし、読み出しを1つの大域観測量
  にすると、5つの角度が原理的に沈黙する。Cliffordのリングは Pauli-Z 列を
  台がより小さい別の Pauli-Z 列へ引き戻すからである。

モデルは記述子の有界な三角関数である
--------------------------------------------------------------------------
      x1       <Z0>       <Z1>       <Z2>       <Z3>      y_hat
    0.00   0.257407  -0.051145  -0.077532   0.089633   0.109182
    0.20   0.132006  -0.036863  -0.035678  -0.126665  -0.033600
    0.40   0.015148  -0.188992   0.068246  -0.276723  -0.191161
    0.60   0.014283  -0.307693   0.123274  -0.354712  -0.262424
    0.80   0.064278  -0.146170   0.040597  -0.297564  -0.169430
    1.00   0.057372   0.239214  -0.054612  -0.212931   0.014522

未訓練の出力（訓練データ先頭5行）
--------------------------------------------------------------------------
           y     y_hat    residual
     -0.6472   -0.1992     -0.4480
     -0.9934    0.2215     -1.2149
     -0.0502   -0.2025      0.1522
      0.0615    0.0270      0.0344
     -0.0772   -0.2147      0.1375

  未訓練の訓練MSE = 0.3553
  |<Z_q>| <= 1 が常に成り立つので、出力の到達範囲は w_q が決める。すべての
  w_q = 0.5 では絶対値2.0を超えられないが、目的変数は 1.2142 に達する。
  読み出し重みは飾りではなく必須である。
```

**注目すべき点。** 3つあります。

**読み出し重みは省略可能ではありません。** どの $\langle Z_q\rangle$ も $[-1,+1]$ に収まるので、それを重み1で4つ読み出すモデルは $[-4,4]$ の外を出力できず、初期値 $w_q = 0.5$ では絶対値2を超えられません。目的変数は 1.2142 に達します。出力スケールを訓練しないVQCは出力範囲を固定したモデルであり、範囲の異なる目的関数に当てはめることは原理的に不可能です。スケールとオフセットは29パラメータのうちの2つで、VQCについて報告されるパラメータ数はこれらを含むべきです。

**沈黙パラメータの検査は形式的な作業ではありません。** この回路では24個の角度のうち沈黙するものは0個ですが、これは1つの順序の決定の結果です。順序を逆にして — 先に回転してからエンタングルし、局所読み出し4つの代わりに $\langle Z_0\rangle$ 1つを読み出す — と、24個のうち5個が原理的に沈黙し、任意の入力・任意のパラメータ値で勾配が恒等的に零になります。理由は一般化するので理解する価値があります。CNOTのリングはClifford回路なので、Pauli文字列を通して共役をとると別のPauli文字列になります。リング `cnot(0,1), cnot(1,2), cnot(2,3), cnot(3,0)` を通して $Z_0$ を引き戻すと $Z_1Z_2Z_3$ になり、これは量子ビット0上では恒等演算子なので、最終層の量子ビット0の回転2つは何もしません。さらに対角観測量は $R_z$ と可換なので、最終層の各量子ビットの $R_z$ も何もしません。合計5つの角度が、パラメータ数には数えられながらモデルに何も寄与しません。演習2でこの数を再現します。

**モデルは有界な三角関数**であり、押しつぶし非線形性を備えた万能近似器ではありません。$x_1$ の走査では4つの読み出しがなめらかに動き、$[-1,1]$ の十分内側に留まります。未訓練モデルの出力範囲は目的変数の範囲の一部にすぎません。これはtanhネットワークとは別の関数クラスであり、それが容量にとって何を意味するかは4.5節に戻ります。

* * *

## 4.2 逆伝播なしで訓練する

### なぜ逆伝播が使えないのか

ニューラルネットワークの勾配が安価なのは、順伝播が痕跡を残すからです。すべての中間活性化が保存され、連鎖律がそれを1回さかのぼるだけで済み、コストは順伝播そのものと同程度です。この技には中間状態を読むことが必要です。量子ハードウェアには読める中間状態がありません。量子ビットを測ればつぶれてしまい、回路の途中で $|\psi\rangle$ を覗いてから続きを実行する方法はありません。VQCが使える勾配則は、完全な回路評価だけから組み立てられていなければならず、その各評価は最初から実行されます。

もう1つ、より微妙な点があります。*シミュレータ*上なら回路を通した逆伝播は完全に可能です。状態ベクトルはメモリ上にあり、線形代数を通した自動微分は動きます。どのフレームワークもそれを提供しており、代替手段よりずっと高速です。しかしそうして計算した勾配は実機では得られないので、シミュレータの逆伝播で得た訓練曲線を報告する論文は、訓練可能な量子モデルを示したのではなく、量子の形をしたパラメータ化を持つ古典モデルを示したにすぎません。以下の規則は実機に移しても生き残るものであり、遅いにもかかわらず本章がそれを使うのはこの理由です。

### parameter-shift則

1つのゲート $R_y(\theta) = \exp(-i\theta Y/2)$ に現れる $\theta$ 以外のパラメータをすべて固定します。$Y^2 = I$ なので

$$
R_y(\theta) = \cos(\theta/2)\,I - i\sin(\theta/2)\,Y
$$

となり、得られる状態上での任意の観測量 $O$ の期待値は $\cos(\theta/2)$ と $\sin(\theta/2)$ の二次形式、すなわち $1$、$\cos\theta$、$\sin\theta$ の線形結合になります。

$$
f(\theta) = \langle O\rangle_\theta = a + b\cos\theta + c\sin\theta
$$

ここで $a, b, c$ は $\theta$ 以外のすべてに依存する定数です。これがこの規則の全内容です。この形の関数はどれも厳密な2点恒等式に従います。$\pm\pi/2$ ずらして差をとると

$$
\frac{f(\theta + \pi/2) - f(\theta - \pi/2)}{2}
= \frac{(a - b\sin\theta + c\cos\theta) - (a + b\sin\theta - c\cos\theta)}{2}
= -b\sin\theta + c\cos\theta
= f'(\theta)
$$

微分は同じ回路をずらした角度で2回評価するだけで*厳密に*得られます。極限をとらず、小さな数で割らず、恒等式は漸近的にではなくすべての $\theta$ で成立します。同じ議論は $G^2 = I$ を満たす任意の生成子について成立します。$R_x$、$R_z$、2量子ビットの $\exp(-i\theta\, Z\otimes Z/2)$ はすべて同一の規則に従い、演習1で検証します。相異なる固有値が3つ以上ある生成子には項が増え、文献の4項規則も同じように高い高調波成分から導かれます。

読み出し重み $w_q$ とオフセット $b$ は回路の外にある古典パラメータなので、その微分は通常のものです。

$$
\frac{\partial\mathcal{L}}{\partial w_q} = \frac{2}{N}\sum_i r_i\,\langle Z_q\rangle_i,\qquad
\frac{\partial\mathcal{L}}{\partial b} = \frac{2}{N}\sum_i r_i,\qquad
r_i = f(\mathbf{x}_i;\boldsymbol\theta) - y_i
$$

回路の角度は読み出しを通した連鎖律を拾います。

$$
\frac{\partial\mathcal{L}}{\partial\theta_k} = \frac{2}{N}\sum_i r_i \sum_q w_q\,
\frac{\langle Z_q\rangle_i^{\theta_k + \pi/2} - \langle Z_q\rangle_i^{\theta_k - \pi/2}}{2}
$$

### Code Example 3: parameter-shift則の検証と、勾配1回のコスト

```python
"""parameter-shift則の検証と、勾配1回の真のコスト。
Code Example 1、2 の続き（同一セッション）。
"""


def mse_loss(params, X, y, layers=N_LAYERS):
    return float(np.mean((vqc_predict(X, params, layers) - y) ** 2))


def vqc_grad(params, X, y, layers=N_LAYERS):
    """MSE損失の厳密な勾配。回路の角度はparameter-shift、読み出し重みは解析的。
    戻り値は (勾配, 使用した回路評価回数)。"""
    theta, w, b = unpack(params)
    Zf = np.array([vqc_features(x, theta, layers) for x in X])   # (N, n_qubits)
    resid = (Zf @ w + b) - y
    g = np.zeros_like(params)
    evals = len(X)
    for k in range(len(theta)):
        tp = theta.copy(); tp[k] += np.pi / 2
        tm = theta.copy(); tm[k] -= np.pi / 2
        Zp = np.array([vqc_features(x, tp, layers) for x in X])
        Zm = np.array([vqc_features(x, tm, layers) for x in X])
        evals += 2 * len(X)
        dZ = 0.5 * (Zp - Zm)                    # d<Z_q>/dtheta_k を厳密に与える
        g[k] = np.mean(2.0 * resid * (dZ @ w))
    g[-(N_QUBITS + 1):-1] = 2.0 * (resid @ Zf) / len(X)
    g[-1] = np.mean(2.0 * resid)
    return g, evals


g_ps, n_evals = vqc_grad(params0, Xtr, ytr)

print("実際の損失に対する parameter-shift と有限差分の比較")
print("-" * 78)
print(f"  {'k':>3}{'parameter shift':>18}{'central diff h=1e-2':>22}"
      f"{'central diff h=1e-6':>22}")
for k in [0, 7, 15, 23, 24, 28]:
    row = []
    for h in (1e-2, 1e-6):
        pp = params0.copy(); pp[k] += h
        pm = params0.copy(); pm[k] -= h
        row.append((mse_loss(pp, Xtr, ytr) - mse_loss(pm, Xtr, ytr)) / (2 * h))
    tag = {24: " (w_0)", 28: " (b)"}.get(k, "")
    print(f"  {k:>3}{g_ps[k]:18.9f}{row[0]:22.9f}{row[1]:22.9f}{tag}")

err = []
for k in range(len(params0)):
    pp = params0.copy(); pp[k] += 1e-6
    pm = params0.copy(); pm[k] -= 1e-6
    err.append(abs(g_ps[k]
                   - (mse_loss(pp, Xtr, ytr) - mse_loss(pm, Xtr, ytr)) / 2e-6))
print(f"\n  全 {len(params0)} パラメータでの max |shift則 - 中心差分(1e-6)|:"
      f" {max(err):.3e}")
print("  shift則は近似ではない。2回の評価は pi/2 離れているので、小さな数で割る操作")
print("  がなく、ノイズが増幅されない。")

print("\n勾配1ステップのコスト")
print("-" * 78)
print(f"  訓練点数 N                            {len(Xtr)}")
print(f"  回路の角度パラメータ数                 {n_theta()}")
print(f"  1ステップあたりの回路評価回数         {n_evals}"
      f"   ( N + 2 N n_theta )")
print(f"  1ステップあたりのゲート数             {n_evals * 48}")
print("  同規模の古典ネットワークは逆伝播1回で全勾配を得る（順伝播と同程度のコスト）。")
print("  係数 2 n_theta は parameter-shift に固有であり、実機では各評価が多数のショット")
print("  になる:")
for shots in (100, 1000, 10000):
    print(f"    期待値1つあたり {shots:5d} ショット -> "
          f"1ステップ {n_evals * shots:>12,} ショット")
print("\n  これは量子コンピューティング入門3.6節の測定の壁が機械学習の装いで現れたもの")
print("  である。本章の比較はすべて勾配*ステップ数*を揃え、ショットコストは別に報告")
print("  する。2つのモデルはステップ数は同じだが、コストは同じではない。")
```

```text
実際の損失に対する parameter-shift と有限差分の比較
------------------------------------------------------------------------------
    k   parameter shift   central diff h=1e-2   central diff h=1e-6
    0       0.057748778           0.057748283           0.057748778
    7       0.029954645           0.029954150           0.029954645
   15       0.054176427           0.054175479           0.054176427
   23      -0.056156170          -0.056155160          -0.056156170
   24       0.117487432           0.117487432           0.117487432 (w_0)
   28       0.119945446           0.119945446           0.119945446 (b)

  全 29 パラメータでの max |shift則 - 中心差分(1e-6)|: 6.525e-11
  shift則は近似ではない。2回の評価は pi/2 離れているので、小さな数で割る操作
  がなく、ノイズが増幅されない。

勾配1ステップのコスト
------------------------------------------------------------------------------
  訓練点数 N                            40
  回路の角度パラメータ数                 24
  1ステップあたりの回路評価回数         1960   ( N + 2 N n_theta )
  1ステップあたりのゲート数             94080
  同規模の古典ネットワークは逆伝播1回で全勾配を得る（順伝播と同程度のコスト）。
  係数 2 n_theta は parameter-shift に固有であり、実機では各評価が多数のショット
  になる:
    期待値1つあたり   100 ショット -> 1ステップ      196,000 ショット
    期待値1つあたり  1000 ショット -> 1ステップ    1,960,000 ショット
    期待値1つあたり 10000 ショット -> 1ステップ   19,600,000 ショット

  これは量子コンピューティング入門3.6節の測定の壁が機械学習の装いで現れたもの
  である。本章の比較はすべて勾配*ステップ数*を揃え、ショットコストは別に報告
  する。2つのモデルはステップ数は同じだが、コストは同じではない。
```

**注目すべき点。** shift則は $h = 10^{-6}$ の中心差分と全29パラメータで $6.5\times10^{-11}$ まで一致します。これは規則の誤差ではなく有限差分の丸め誤差の蓄積です。$h = 10^{-2}$ では有限差分は既に小数第7位で誤ります。無雑音シミュレータ上ではこれはほとんど問題にならず、$h$ を小さくとれば済みます。実機では決定的です。各期待値は1000ショットで $10^{-2}$ 程度の統計誤差を持ち、それを $2h = 2\times10^{-6}$ で割ると誤差が $5\times10^5$ 倍されます。parameter-shift則は何も割らないので、勾配のノイズは期待値2つ分のノイズであり、それより悪くなりません。

コストの表が記憶すべき数値です。全バッチの勾配1ステップに $N + 2 N n_\theta = 1960$ 回の回路評価が必要です。各1000ショット — 化学の実験としては控えめ、機械学習としては大盤振る舞い — なら、勾配1ステップで196万ショット、4.3節の120ステップで2億3520万ショットになります。同規模の古典ネットワークは逆伝播1回で全勾配を得ます。この非対称性はより良いソフトウェアで消せる細部ではなく、手法の構造そのものです。ミニバッチ化は $N$ を減らし、SPSAのような確率近似法は $2n_\theta$ の因子を定数に置き換えますが、その代償は勾配方向のノイズです。量子勾配が別々の実験から組み立てられ、古典勾配がそうでないという事実は、何をしても消えません。

* * *

## 4.3 比較と、それが従うべき規則

本コースがこの節のために存在しているので、数値より先に規則を書きます。

**規則1 — データと分割を同一にする。** 両モデルは同じ40行の訓練データを見て、同じ20行のテストデータで評価される。再標本化なし、再シャッフルなし、結果を見てから分割を選ぶこともなし。

**規則2 — 最適化器は1つ。** 同じAdam実装、同じ $\beta_1, \beta_2, \epsilon$、同じステップ数。「ネットワークにはAdam、回路には手調整のスケジュール」は不可。

**規則3 — パラメータ数を揃え、揃わない分は明記する。** VQCの訓練可能パラメータは29個。2つのネットワークは25個と31個でこれを挟む。VQCが25個に勝ち31個に負けたなら、それは勝利ではなく曖昧な結果として報告する。

**規則4 — ハイパーパラメータ探索を同一にする。** どのモデルも同じ学習率グリッドと同じ早期停止規則を与えられ、選択は*訓練*データから切り出した検証分割の上で行う。テストデータに触れるのは最後の1回だけ、すべてのモデルについて同時に。

**規則5 — すべての初期値を報告する。** モデルごとにランダム初期値3通り、つまり訓練実行9回、そのすべてを4つの数値 — 選ばれた停止ステップとステップ120における訓練損失とテスト損失 — とともに出力する。中央値を引用し、個々の実行も見えるようにして、読者が広がりを確認し、中央値が幸運な1回に支えられていないことを検査できるようにする。初期化は回路角について $[0,2\pi)$ 上の一様分布であり、これは標準的な選択であると同時に barren plateau が語る対象そのものです。恒等ブロック初期化は本章のどこでも試しておらず、以下の数値はしたがってランダムに初期化された回路の数値です。

規則3には1つ正直な留保が必要です。パラメータ数がここでの通貨として妥当なのは読者が検算できる量だからですが、容量の完全な尺度ではなく、どちらの側にも一貫して有利ではありません。ネットワークの重みは非線形関数に入る有界でない実数であり、回路の角度は周期変数で出力への影響は構成上有界です。両者の間に交換レートはありません。言えるのは、調整可能な数の個数が同じで、最適化器が同じで、選択手順が同じ2つのモデルがテスト誤差で大きく違うなら、その違いは会計上の問題ではなくモデルクラスの性質だ、ということです。

### Code Example 4: 古典側の対戦相手と、両者共通の手順

```python
"""古典側の対戦相手、両者共通の最適化器、そして両者共通の選択手順。
Code Example 1、2、3 の続き（同一セッション）。
"""


# ---- 同規模の古典ネットワーク。NumPyのみ --------------------------------
def mlp_shapes(h, d=N_QUBITS):
    return [(d, h), (h,), (h, 1), (1,)]


def mlp_nparams(h, d=N_QUBITS):
    return d * h + h + h + 1


def mlp_unpack(params, h, d=N_QUBITS):
    i, out = 0, []
    for shape in mlp_shapes(h, d):
        size = int(np.prod(shape))
        out.append(params[i:i + size].reshape(shape))
        i += size
    return out


def mlp_predict(X, params, h):
    W1, b1, W2, b2 = mlp_unpack(params, h)
    return (np.tanh(X @ W1 + b1) @ W2 + b2).ravel()


def mlp_loss(params, X, y, h):
    return float(np.mean((mlp_predict(X, params, h) - y) ** 2))


def mlp_grad(params, X, y, h):
    """逆伝播による厳密な勾配。順伝播1回と逆伝播1回のみ。"""
    W1, b1, W2, b2 = mlp_unpack(params, h)
    Zh = np.tanh(X @ W1 + b1)
    pred = (Zh @ W2 + b2).ravel()
    r = 2.0 * (pred - y) / len(y)
    gW2 = Zh.T @ r[:, None]
    gb2 = np.array([r.sum()])
    dZ = r[:, None] @ W2.T * (1.0 - Zh ** 2)
    return np.concatenate([(X.T @ dZ).ravel(), dZ.sum(axis=0),
                           gW2.ravel(), gb2.ravel()])


# ---- 両モデルが共有する最適化器 ------------------------------------------
def adam(grad_fn, params, steps, lr, record=None):
    """素のAdam。record を与えると record(step, params) が毎ステップ呼ばれる。"""
    p = params.copy()
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    if record is not None:
        record(0, p)
    for t in range(1, steps + 1):
        g = grad_fn(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        p = p - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if record is not None:
            record(t, p)
    return p


def init_vqc(seed, layers=N_LAYERS, y0=None):
    # 角度は [0, 2 pi) 上の一様分布。これは標準的な選択であると同時に plateau に
    # 陥りやすい選択でもある。barren plateau は*ランダムな*パラメータについての
    # 主張だからである。恒等ブロック初期化 — 変分層を恒等にして plateau の外から
    # 訓練を始める通常の緩和策 — は本章のどこでも試していない。したがって以下の
    # VQCの数値はすべてランダムに初期化された回路の数値であり、これが比較の
    # 誠実な適用範囲である（最良の回路についての主張ではない）。
    r = np.random.default_rng(seed)
    return np.concatenate([r.uniform(0, 2 * np.pi, n_theta(layers)),
                           r.normal(0, 0.5, N_QUBITS),
                           [ytr.mean() if y0 is None else y0]])


def init_mlp(seed, h, y0=None):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, 1.0 / np.sqrt(N_QUBITS), (N_QUBITS, h))
    W2 = r.normal(0, 1.0 / np.sqrt(h), (h, 1))
    return np.concatenate([W1.ravel(), np.zeros(h), W2.ravel(),
                           [ytr.mean() if y0 is None else y0]])


# ---- 全モデルに同一に適用するモデル選択手順 ------------------------------
H_SMALL, H_LARGE = 4, 5
MODELS = ('VQC', f'MLP h={H_SMALL}', f'MLP h={H_LARGE}')
MAX_STEPS = 120
GRID = (0.02, 0.05, 0.10, 0.20, 0.40)
VAL = np.arange(30, 40)                      # 訓練データ末尾10行を検証に回す
FIT = np.arange(0, 30)


def train_curve(name, params0, Xa, ya, Xb, yb, lr, steps=MAX_STEPS):
    """(Xa, ya) で訓練し、毎ステップ (Xa, ya) と (Xb, yb) の損失を記録する。

    ホールドアウト側の予測値も毎ステップ保存する。こうすればプロトコルが後で選ぶ
    停止ステップにおいて、再訓練なしでR6の対応のあるブートストラップを計算できる。
    """
    hist, preds = [], []
    if name == 'VQC':
        def rec(t, p):
            q = vqc_predict(Xb, p)
            hist.append((t, mse_loss(p, Xa, ya), float(np.mean((q - yb) ** 2))))
            preds.append(q)
        pf = adam(lambda q: vqc_grad(q, Xa, ya)[0], params0, steps, lr, record=rec)
    else:
        h = int(name.split('=')[1])

        def rec(t, p):
            q = mlp_predict(Xb, p, h)
            hist.append((t, mlp_loss(p, Xa, ya, h), float(np.mean((q - yb) ** 2))))
            preds.append(q)
        pf = adam(lambda q: mlp_grad(q, Xa, ya, h), params0, steps, lr, record=rec)
    return pf, np.array(hist), np.array(preds)


def init_of(name, seed, y0):
    return (init_vqc(seed, y0=y0) if name == 'VQC'
            else init_mlp(seed, int(name.split('=')[1]), y0=y0))


print("パラメータ数: 古典側に有利を与えていないことの確認")
print("-" * 78)
print(f"  {'VQC, ' + str(N_LAYERS) + ' layers, ' + str(N_QUBITS) + ' qubits':<34}"
      f"{len(params0):>4} parameters")
for h in (H_SMALL, H_LARGE):
    print(f"  {'MLP 4-' + str(h) + '-1 with tanh':<34}{mlp_nparams(h):>4} parameters")
print(f"  2つのネットワークがVQCを挟む（{mlp_nparams(H_SMALL)} < {len(params0)}"
      f" < {mlp_nparams(H_LARGE)}）ので、")
print("  結論がパラメータ数の細工で決まることはない。")

print("\n3モデルに同一に適用する選択手順")
print("-" * 78)
print(f"  訓練行1-30で当てはめ、行31-40で検証。テストデータには触れない")
print(f"  学習率は {GRID} から、停止ステップは 0..{MAX_STEPS} から選ぶ")
print(f"  {'model':<12}{'lr':>7}{'best val MSE':>15}{'at step':>9}"
      f"{'val MSE at 120':>16}")
chosen = {}
for name in MODELS:
    rows = []
    for lr in GRID:
        y0 = ytr[FIT].mean()
        _, hist, _ = train_curve(name, init_of(name, 101, y0), Xtr[FIT], ytr[FIT],
                                 Xtr[VAL], ytr[VAL], lr)
        i = int(np.argmin(hist[:, 2]))
        rows.append((lr, hist[i, 2], int(hist[i, 0]), hist[-1, 2]))
    j = int(np.argmin([r[1] for r in rows]))
    chosen[name] = (rows[j][0], rows[j][2])
    for k, (lr, vb, st, vend) in enumerate(rows):
        mark = "  <- chosen" if k == j else ""
        print(f"  {name if k == 0 else '':<12}{lr:7.2f}{vb:15.4f}{st:9d}"
              f"{vend:16.4f}{mark}")
print("\n  停止ステップは後付けではなく手順の一部である。最終列を第3列と読み比べると、")
print("  3モデルのうち2つは最良ステップよりステップ120で明確に悪化しており、その度合い")
print("  が最も大きいのはVQCである。")
```

```text
パラメータ数: 古典側に有利を与えていないことの確認
------------------------------------------------------------------------------
  VQC, 3 layers, 4 qubits             29 parameters
  MLP 4-4-1 with tanh                 25 parameters
  MLP 4-5-1 with tanh                 31 parameters
  2つのネットワークがVQCを挟む（25 < 29 < 31）ので、
  結論がパラメータ数の細工で決まることはない。

3モデルに同一に適用する選択手順
------------------------------------------------------------------------------
  訓練行1-30で当てはめ、行31-40で検証。テストデータには触れない
  学習率は (0.02, 0.05, 0.1, 0.2, 0.4) から、停止ステップは 0..120 から選ぶ
  model            lr   best val MSE  at step  val MSE at 120
  VQC            0.02         0.1917       91          0.2056
                 0.05         0.2378       22          0.4983
                 0.10         0.2206       25          0.4655
                 0.20         0.1657       22          0.1962  <- chosen
                 0.40         0.2741        2          0.6451
  MLP h=4        0.02         0.0518       50          0.0603
                 0.05         0.0509       24          0.0567
                 0.10         0.0273      120          0.0273  <- chosen
                 0.20         0.0438       43          0.0780
                 0.40         0.0839       56          0.1673
  MLP h=5        0.02         0.0498       36          0.0758
                 0.05         0.0470       20          0.0824  <- chosen
                 0.10         0.0520       13          0.1078
                 0.20         0.0511       29          0.0615
                 0.40         0.0722       41          0.0915

  停止ステップは後付けではなく手順の一部である。最終列を第3列と読み比べると、
  3モデルのうち2つは最良ステップよりステップ120で明確に悪化しており、その度合い
  が最も大きいのはVQCである。
```

**注目すべき点。** 選択の表は、そこから出てくる結論よりも情報量があります。「best val MSE」の列を「val MSE at 120」の列と読み比べてください。VQCの $\text{lr} = 0.05$ では最良ステップの検証誤差が 0.2378、ステップ120では 0.4983 で、予算の後半をまるごと使って2倍悪化しています。$\text{lr} = 0.40$ では最良の検証誤差がステップ2で訪れ、終了時には2.4倍悪くなっています。ネットワークも同じ傾向をより穏やかに示し、$\text{lr} = 0.10$ では予算内で過学習に至らないため、その設定が選ばれます。

VQCのどの設定でも到達する最良の検証MSEは 0.1657、ネットワークの最良は 0.0273 で、6倍の差です。この差はテストデータを見る前に、モデルを選ぶのに使ったデータの上で、両者を同一に扱う手順の下で測られています。次にテストデータが何を言おうと、6倍を覆すことはありません。

### Code Example 5: 学習曲線と結論

```python
"""正直な学習曲線による直接対決。
Code Example 1、2、3、4 の続き（同一セッション）。
"""
SEEDS = (101, 202, 303)
curves, finals, runs_by = {}, {}, {}
for name in MODELS:
    lr, stop = chosen[name]
    runs = [train_curve(name, init_of(name, s, ytr.mean()), Xtr, ytr, Xte, yte, lr)
            for s in SEEDS]
    runs_by[name] = runs
    curves[name] = np.mean([r[1] for r in runs], axis=0)
    finals[name] = [(r[1][stop, 1], r[1][stop, 2], r[1][-1, 1], r[1][-1, 2])
                    for r in runs]

print(f"学習曲線: Adam {MAX_STEPS} ステップ、初期値 {len(SEEDS)} 通りの平均")
print("-" * 92)
head = f"  {'step':>5}"
for name in MODELS:
    head += f"{name + ' tr':>15}{name + ' te':>15}"
print(head)
for t in (0, 5, 10, 20, 30, 40, 60, 80, 100, 120):
    row = f"  {t:>5}"
    for name in MODELS:
        row += f"{curves[name][t, 1]:15.4f}{curves[name][t, 2]:15.4f}"
    print(row)
print("\n  選ばれた停止ステップ: " + ",  ".join(
    f"{name} at {chosen[name][1]} (lr {chosen[name][0]:g})" for name in MODELS))

print("\n初期値ごとの全結果。幸運な初期値の陰に何も隠さないため")
print("-" * 92)
print(f"  {'model':<12}{'seed':>6}{'train@stop':>12}{'test@stop':>12}"
      f"{'train@120':>12}{'test@120':>12}")
for name in MODELS:
    for s, row in zip(SEEDS, finals[name]):
        print(f"  {name:<12}{s:>6}" + "".join(f"{v:12.4f}" for v in row))
    med = np.median(np.array(finals[name]), axis=0)
    print(f"  {name:<12}{'median':>6}" + "".join(f"{v:12.4f}" for v in med))

print("\nCode Example 1 のベースラインと並べた結論")
print("-" * 92)
print(f"  {'model':<28}{'params':>8}{'train MSE':>12}{'test MSE':>12}{'vs OLS':>10}")
rows = [("predict the training mean", 1, np.mean((ytr - ytr.mean()) ** 2), mse_const),
        ("ordinary least squares", 5, np.mean((A @ coef - ytr) ** 2), mse_lin)]
for name in MODELS:
    npar = len(params0) if name == 'VQC' else mlp_nparams(int(name.split('=')[1]))
    med = np.median(np.array(finals[name]), axis=0)
    rows.append((name + " (early stopped)", npar, med[0], med[1]))
for nm, npar, tr, te in rows:
    print(f"  {nm:<28}{npar:>8}{tr:12.4f}{te:12.4f}{te/mse_lin:10.2f}x")
print(f"  {'irreducible noise floor':<28}{'-':>8}{0.05**2:12.4f}{0.05**2:12.4f}"
      f"{0.05**2/mse_lin:10.2f}x")

vq = np.median(np.array(finals['VQC']), axis=0)
cl = min((np.median(np.array(finals[m]), axis=0) for m in MODELS if m != 'VQC'),
         key=lambda r: r[1])
print("\nこの結果が語ること")
print("-" * 92)
if vq[1] < cl[1]:
    print(f"  ここではVQCの汎化が優れている: テストMSE {vq[1]:.4f} 対 {cl[1]:.4f}。")
else:
    print(f"  VQCは負ける。テストMSE {vq[1]:.4f} 対 同規模ネットワークの最良 {cl[1]:.4f}、")
    print(f"  すなわち {vq[1]/cl[1]:.2f} 倍悪い。データも最適化器も選択手順も同一で、")
    print(f"  パラメータ数はVQCの方がわずかに多い条件下での結果である。")
print(f"  2つの非線形モデルは5パラメータの線形回帰（{mse_lin:.4f}）とも比較しており、")
print(f"  VQCはこれにも負けている。この行を省けばずっと見栄えの良い話になり、実際")
print("  そうした省略はよく行われる。")
print(f"  {MAX_STEPS} ステップのコスト: VQCは {MAX_STEPS * 1960:,} 回の回路評価、")
print(f"  各ネットワークは {MAX_STEPS} 回の逆伝播。演算量で約 1,960:1 であり、実機の")
print("  実時間ではこれよりはるかに大きな差になる。")
print("\n  訓練側の列も読むこと。VQCはステップ120で3モデル中*最小の訓練損失*かつ*最大の")
print("  テスト損失*である。これは学習不足ではなく、40点を24個の三角関数的角度で当て")
print("  はめて汎化に失敗している状態である。古典的な表形式データに対するNISQ期の")
print("  想定どおりの結果であり、第5章の集中と脱量子化の議論が予測するとおりである。")


# ---- R6: 同じ20行で対応をとった差の区間 ------------------------------------
def paired_bootstrap_mse(y_true, pred_a, pred_b, B=10000, seed=0, alpha=0.05):
    """MSE(a) - MSE(b) の95%区間。両者に同じテスト行を再標本化して用いる。

    第1章のR6を、約束ではなくここで実行する。対応をとることで「どの行がテスト集合
    に入ったか」に由来する分散が除かれ、20行ではその項が他のすべてを支配する。
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pa, pb = np.asarray(pred_a), np.asarray(pred_b)
    m = len(y_true)
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, m, m)
        d[b] = (np.mean((y_true[i] - pa[i]) ** 2)
                - np.mean((y_true[i] - pb[i]) ** 2))
    return (float(d.mean()), float(np.quantile(d, alpha / 2)),
            float(np.quantile(d, 1 - alpha / 2)))


def median_run_pred(name):
    """停止ステップでのテストMSEが3回の中央値になる実行のテスト予測値。
    判定表が引用するのと同じ実行である。"""
    stop = chosen[name][1]
    te = [r[1][stop, 2] for r in runs_by[name]]
    return runs_by[name][int(np.argsort(te)[len(te) // 2])][2][stop]


PRED = {name: median_run_pred(name) for name in MODELS}
PRED['ordinary least squares'] = pred_lin
PRED['predict the training mean'] = np.full(len(yte), ytr.mean())

print("\nR6: 同じ差に対応のあるブートストラップ区間をつける")
print("-" * 92)
print("  各対の両側について同じ20テスト行を10,000回再標本化。実行は各モデルの中央値。")
print(f"  {'MSE(A) - MSE(B)':<48}{'mean':>10}{'95% interval':>21}{'verdict':>12}")
for a, b in [('VQC', f'MLP h={H_SMALL}'),
             ('VQC', f'MLP h={H_LARGE}'),
             ('VQC', 'ordinary least squares'),
             (f'MLP h={H_SMALL}', 'ordinary least squares'),
             ('VQC', 'predict the training mean')]:
    m_, lo, hi = paired_bootstrap_mse(yte, PRED[a], PRED[b])
    v = "A better" if hi < 0.0 else ("B better" if lo > 0.0 else "no call")
    print(f"  {a + '  -  ' + b:<48}{m_:+10.4f}   [{lo:+.4f}, {hi:+.4f}]{v:>12}")
print("  3.87倍は20行で測った2つの点推定値の比である。R6が求めるのは区間の方であり、")
print("  判定はそれに基づかなければならない。")
```

```text
学習曲線: Adam 120 ステップ、初期値 3 通りの平均
--------------------------------------------------------------------------------------------
   step         VQC tr         VQC te     MLP h=4 tr     MLP h=4 te     MLP h=5 tr     MLP h=5 te
      0         0.3552         0.2712         0.9507         0.8026         0.4322         0.3717
      5         0.1135         0.1375         0.2753         0.2938         0.1991         0.2424
     10         0.0694         0.1333         0.1568         0.1385         0.1137         0.1247
     20         0.0370         0.1476         0.0944         0.0938         0.0684         0.0771
     30         0.0238         0.1691         0.0606         0.0590         0.0563         0.0572
     40         0.0142         0.2027         0.0519         0.0531         0.0528         0.0530
     60         0.0080         0.2503         0.0468         0.0443         0.0487         0.0484
     80         0.0070         0.2768         0.0413         0.0389         0.0462         0.0463
    100         0.0065         0.2853         0.0332         0.0295         0.0433         0.0429
    120         0.0059         0.2939         0.0269         0.0245         0.0397         0.0391

  選ばれた停止ステップ: VQC at 22 (lr 0.2),  MLP h=4 at 120 (lr 0.1),  MLP h=5 at 20 (lr 0.05)

初期値ごとの全結果。幸運な初期値の陰に何も隠さないため
--------------------------------------------------------------------------------------------
  model         seed  train@stop   test@stop   train@120    test@120
  VQC            101      0.0232      0.0893      0.0074      0.1375
  VQC            202      0.0445      0.2490      0.0054      0.3992
  VQC            303      0.0376      0.0925      0.0047      0.3450
  VQC         median      0.0376      0.0925      0.0054      0.3450
  MLP h=4        101      0.0212      0.0164      0.0212      0.0164
  MLP h=4        202      0.0135      0.0239      0.0135      0.0239
  MLP h=4        303      0.0461      0.0333      0.0461      0.0333
  MLP h=4     median      0.0212      0.0239      0.0212      0.0239
  MLP h=5        101      0.0733      0.0794      0.0420      0.0450
  MLP h=5        202      0.0684      0.0773      0.0388      0.0363
  MLP h=5        303      0.0635      0.0747      0.0383      0.0362
  MLP h=5     median      0.0684      0.0773      0.0388      0.0363

Code Example 1 のベースラインと並べた結論
--------------------------------------------------------------------------------------------
  model                         params   train MSE    test MSE    vs OLS
  predict the training mean          1      0.3120      0.2747      5.91x
  ordinary least squares             5      0.0521      0.0465      1.00x
  VQC (early stopped)               29      0.0376      0.0925      1.99x
  MLP h=4 (early stopped)           25      0.0212      0.0239      0.51x
  MLP h=5 (early stopped)           31      0.0684      0.0773      1.66x
  irreducible noise floor            -      0.0025      0.0025      0.05x

この結果が語ること
--------------------------------------------------------------------------------------------
  VQCは負ける。テストMSE 0.0925 対 同規模ネットワークの最良 0.0239、
  すなわち 3.87 倍悪い。データも最適化器も選択手順も同一で、
  パラメータ数はVQCの方がわずかに多い条件下での結果である。
  2つの非線形モデルは5パラメータの線形回帰（0.0465）とも比較しており、
  VQCはこれにも負けている。この行を省けばずっと見栄えの良い話になり、実際
  そうした省略はよく行われる。
  120 ステップのコスト: VQCは 235,200 回の回路評価、
  各ネットワークは 120 回の逆伝播。演算量で約 1,960:1 であり、実機の
  実時間ではこれよりはるかに大きな差になる。

  訓練側の列も読むこと。VQCはステップ120で3モデル中*最小の訓練損失*かつ*最大の
  テスト損失*である。これは学習不足ではなく、40点を24個の三角関数的角度で当て
  はめて汎化に失敗している状態である。古典的な表形式データに対するNISQ期の
  想定どおりの結果であり、第5章の集中と脱量子化の議論が予測するとおりである。

R6: 同じ差に対応のあるブートストラップ区間をつける
--------------------------------------------------------------------------------------------
  各対の両側について同じ20テスト行を10,000回再標本化。実行は各モデルの中央値。
  MSE(A) - MSE(B)                                       mean         95% interval     verdict
  VQC  -  MLP h=4                                    +0.0687   [+0.0200, +0.1256]    B better
  VQC  -  MLP h=5                                    +0.0154   [-0.0265, +0.0620]     no call
  VQC  -  ordinary least squares                     +0.0463   [+0.0012, +0.1011]    B better
  MLP h=4  -  ordinary least squares                 -0.0224   [-0.0431, -0.0038]    A better
  VQC  -  predict the training mean                  -0.1823   [-0.3516, -0.0361]    A better
  3.87倍は20行で測った2つの点推定値の比である。R6が求めるのは区間の方であり、
  判定はそれに基づかなければならない。
```

**注目すべき点。** 学習曲線は同じことを3通りに述べています。

**VQCはよく訓練され、汎化に失敗します。** 120ステップの予算の終わりでは、3モデル中で最小の訓練損失と最大のテスト損失になります。それを報告する表は2つあり、統計量が違うので混ぜないでおきます。学習曲線の表は初期値3通りの*平均*で、訓練は 0.0059 対 0.0269 と 0.0397、テストは 0.2939 対 0.0245 と 0.0391 です。初期値ごとの表が引用するのは*中央値*で、VQCは 0.0054 と 0.3450 です。判定表が使うのは中央値の方です。これは学習不足でもなく最適化器の失敗でもありません。回路は40点を24個の三角関数的角度と4成分の線形読み出しで当てはめており、ノイズまで当てはめています。訓練損失が競合の4.6分の1で、テスト損失が12倍の値であるモデルは、関数ではなく訓練データを学習したのです。早期停止はその一部を救うので結論の表は早期停止後の数値を使いますが、差は埋まりません。

**判定がパラメータ数の問いに対して頑健なのは半分だけで、それを示すのが対応のある区間です。** VQCの25パラメータのネットワークに対する負けは解像できます。MSE(VQC) $-$ MSE(MLP $h{=}4$) の区間は $[+0.020, +0.126]$ で完全にゼロより上です。31パラメータのネットワークに対する負けは解像できません。$[-0.027, +0.062]$ で*no call*です。つまり規則3の曖昧な場合は結局生じており、しかもR6を適用したときにだけ生じます。R6が礼儀ではなく規則である理由がまさにこれです。さらに2つの区間が位置を確定させます。VQCは自明なベースラインには勝っており $[-0.352, -0.036]$、何も学んでいないわけではありません。そして5パラメータの線形回帰には、ぎりぎりゼロを超える差 $[+0.001, +0.101]$ で負けています。したがって正確な見出しは比ではなく連言です。*VQCは閉形式の線形フィットと、同一予算の2つのネットワークのうち小さい方に負け、大きい方とは区別できない。*これは「3.87倍」より弱い主張であり、データが支持するのはこちらです。

**線形回帰は誠実な物差しであり、恥をかいていません。** 5パラメータ、閉形式、反復なし、テストMSE 0.0465。3つの反復的モデルのうちこれに勝つのは25パラメータのネットワーク（0.0239）だけです。VQCは 0.0925 で勝てず、大きい方のネットワークも 0.0773 で勝てません。後者は検証分割がたまたま悪い点で早期停止させた結果です。この実験をVQCと大きい方のネットワークだけで見せれば「同じパラメータ数で同等の性能」と表現できたはずで、それは真であり無意味です。線形回帰の行がその読み方を止めます。

これは量子モデルが学習できないという証拠ではありません。1つのモデルクラス、1種類のデータ、1つの規模についての証拠であり、その証拠は1年をどこに使うか決める材料研究者がまさに必要とするものです。結論が変わりうる状況は具体的で、第5章がそれを1つずつ扱います。目的関数の構造に周波数の台が合っている符号化、回路が共有する群対称性を持つ目的関数、あるいは最初から量子であるデータです。

* * *

## 4.4 barren plateau、こんどはデータ付きで

### 姉妹コースが確立したこと

[量子コンピューティング入門](<../../FM/quantum-computing-introduction/chapter-3.html>)の3.6節はこの効果を直接測定しました。$n$ 量子ビット上の深く構造のないansatzにランダムなパラメータを入れると、勾配1成分の分散は $n$ について幾何級数的に減衰します。典型的な大きさ $g$ の勾配を分解するには $O(1/g^2)$ ショットが必要なので、指数的に小さな勾配は指数的に大きな測定コストです。最適化は派手に失敗しません。損失が動かなくなるだけで、どのステップもノイズと区別できなくなります。

同じ回路を機械学習に使うと3つのことが変わり、そのすべてが測定できます。

**平均がパラメータだけでなくデータについても走ります。** VQEのコストは数値1つなので、plateauの主張はランダムな $\boldsymbol\theta$ 上での $\partial E/\partial\theta_k$ の分布についてのものです。VQCのコストは入力についての平均なので、関係する主張はランダムな $\boldsymbol\theta$ *と*ランダムな $\mathbf{x}$ 上の分布についてのものになります。符号化ゲート自身が回転であり、入力をランダム化すればそれもランダム化されるので、回路がスクランブリングに近づく方向にしか働きません。実際には追加の平均が何かを救うことはなく、Code Example 6 の減衰率は姉妹コースのものと同じオーダーです。

**読み出しは自由に選べ、しかも結果に大きく効きます。** VQEの観測量は物理が決めます。ハミルトニアンはそういうものです。VQCの読み出しはモデル作成者が選び、局所的な $\langle Z_0\rangle$ と大域的な射影演算子の選択は減衰率で1量子ビットあたり2倍分に相当します。つまり全部です。

**表現力が手段ではなく目的になります。** VQEは1つの特定の状態、基底状態に到達したいので、物理的に関係する部分空間だけに届く構造化ansatzは利点です。VQCは豊かな関数クラスが欲しいので、より深く、よりエンタングルする回路 — すなわちplateauの真ん中 — へ押されます。この二律背反は化学より鋭いのです。

### Code Example 6: 深さ、読み出しの局所性、エンタングルメントを測る

```python
"""QMLにおける barren plateau: 深さ、読み出しの局所性、エンタングルメント。
Code Example 1 の続き（同一セッション）。
"""


def qml_state(x, theta, n, layers, entangle=True):
    """Code Example 2 の回路を任意の幅と深さで。"""
    psi = ket('0' * n)
    k = 0
    for _ in range(layers):
        for q in range(n):
            psi = apply_gate(psi, ry(np.pi * x[q]), [q], n)
        if entangle:
            for q in range(n):
                psi = cnot(psi, q, (q + 1) % n, n)
        for q in range(n):
            psi = apply_gate(psi, rz(theta[k]), [q], n); k += 1
            psi = apply_gate(psi, ry(theta[k]), [q], n); k += 1
    return psi


def readout(psi, n, kind):
    """2種類の読み出し。'local' は <Z_0>。'global' は |00...0> の確率、すなわち
    全n量子ビットに台をもつ射影演算子の期待値。"""
    if kind == 'local':
        return expval(psi, 'Z' + 'I' * (n - 1))
    return float(probs(psi)[0])


def shot_var(r, kind):
    """読み出し1ショットの分散。ショットコストを決めるのはこの量である。

    <Z_0> は ±1 の平均なので1ショットの分散は 1 - <Z>^2。全ゼロ確率は1つのビット列
    を数えて推定するので1ショットの分散はBernoulliの p(1 - p) であり、スクランブル
    する回路では p ~ 2^-n なのでこの分散も指数的に小さくなる。
    """
    return 1.0 - r * r if kind == 'local' else r * (1.0 - r)


def grad_sample(rng, n, layers, kind, entangle=True, samples=120):
    """勾配の1成分と読み出し1ショットの分散を、ランダムな角度とランダムな入力に
    ついて標本抽出する。

    成分は常に中間層・量子ビット0のRy角に固定する。こうすると以下の4条件が
    意図した点でのみ異なる。
    """
    npar = 2 * n * layers
    k = (layers // 2) * (2 * n) + 1
    g = np.empty(samples)
    v = np.empty(2 * samples)
    for s in range(samples):
        theta = rng.uniform(0, 2 * np.pi, npar)
        x = rng.uniform(0.0, 1.0, n)
        tp = theta.copy(); tp[k] += np.pi / 2
        tm = theta.copy(); tm[k] -= np.pi / 2
        rp = readout(qml_state(x, tp, n, layers, entangle), n, kind)
        rm = readout(qml_state(x, tm, n, layers, entangle), n, kind)
        g[s] = 0.5 * (rp - rm)
        v[2 * s], v[2 * s + 1] = shot_var(rp, kind), shot_var(rm, kind)
    return g, float(v.mean())


VARIANTS = [
    ("shallow L=2, local, ring", lambda n: 2,     'local',  True),
    ("deep L=3n, local, ring",   lambda n: 3 * n, 'local',  True),
    ("deep L=3n, global, ring",    lambda n: 3 * n, 'global', True),
    ("deep L=3n, local, no ring",      lambda n: 3 * n, 'local',  False),
]
NS = list(range(2, 9))

print("データ依存コストの勾配分散（角度と入力の両方について平均）")
print("-" * 100)
print(f"  {'variant':<26}" + "".join(f"{f'n={n}':>10}" for n in NS))
table, shotvar = {}, {}
for label, depth, kind, ent in VARIANTS:
    row, vrow = [], []
    for n in NS:
        g, v = grad_sample(np.random.default_rng(1000 + n), n, depth(n), kind, ent)
        row.append(g.var()); vrow.append(v)
    table[label] = np.array(row)
    shotvar[label] = np.array(vrow)
    print(f"  {label:<26}" + "".join(f"{v:10.3e}" for v in row))

print("\n量子ビット1つ増やしたときの比 — フィットではなく減衰の形そのもの")
print("-" * 100)
print(f"  {'variant':<26}" + "".join(f"{f'{n}->{n+1}':>10}" for n in NS[:-1]))
for label, v in table.items():
    print(f"  {label:<26}" + "".join(f"{v[i+1]/v[i]:10.3f}"
                                    for i in range(len(v) - 1)))

print("\n指数フィット Var ~ c r^n")
print("-" * 100)
ns = np.array(NS, dtype=float)
print(f"  {'variant':<26}{'r (n=2..8)':>13}{'r (n=5..8)':>13}"
      f"{'decay/qubit (n=5..8)':>22}")
for label, v in table.items():
    r_all = np.exp(np.polyfit(ns, np.log(v), 1)[0])
    r_tail = np.exp(np.polyfit(ns[3:], np.log(v[3:]), 1)[0])
    print(f"  {label:<26}{r_all:13.4f}{r_tail:13.4f}{1/r_tail:22.2f}x")

print("\n読み出し自身の1ショット分散 v — コストのもう半分")
print("-" * 100)
print(f"  {'variant':<26}" + "".join(f"{f'n={n}':>10}" for n in NS))
for label, v in shotvar.items():
    print(f"  {label:<26}" + "".join(f"{x:10.3e}" for x in v))

print("\nショットノイズを std(g) まで下げるのに要する1評価あたりショット数:  S = v / Var(g)")
print("-" * 100)
print(f"  {'variant':<26}" + "".join(f"{f'n={n}':>13}" for n in (2, 4, 6, 8))
      + f"{'growth/qubit':>15}")
for label, v in table.items():
    S = shotvar[label] / v
    r = np.exp(np.polyfit(ns[3:], np.log(S[3:]), 1)[0])
    print(f"  {label:<26}"
          + "".join(f"{S[n-2]:13,.0f}" for n in (2, 4, 6, 8)) + f"{r:14.2f}x")

print("\nこの実験が測っているもの")
print("-" * 100)
print("  深さ。浅い回路の比はnが増えると1に近づく。量子ビット0の光円錐が広がらなく")
print("  なるので、レジスタを広げても効かなくなる。深い回路の比は1より十分小さい定数")
print("  付近に留まり、これが真の幾何級数的減衰である。")
print("  読み出し。大域的な射影演算子の勾配分散は局所的なものの2倍の速さで1量子ビット")
print("  あたり減衰する — しかし1ショット分散 p(1-p) ~ 2^-n も同じく減衰し、ショット")
print("  コストは 1/Var ではなく v/Var である。そう測ると2つの読み出しは同じオーダーの")
print("  ショット数を要し、増加率もどちらも1量子ビットあたり約2倍である。")
print("  L = 3n ではどちらの回路も自力で集中するほど深いので、この実験は局所性の効果を")
print("  分離できない。局所読み出しを推奨する定理が扱うのは**浅い**回路であり、")
print("  ここの2条件はどちらも浅くない。")
print("  エンタングルメント。CNOTリングを除くと回路は独立な1量子ビットモデルの積に")
print("  なり、分散はnに依らない。同時に表現力も1変数関数の集まりに落ちる。")
print("  訓練可能性と表現力は積み上がるのではなく、互いに取引されている。")
```

```text
データ依存コストの勾配分散（角度と入力の両方について平均）
----------------------------------------------------------------------------------------------------
  variant                          n=2       n=3       n=4       n=5       n=6       n=7       n=8
  shallow L=2, local, ring   1.989e-01 1.098e-01 4.585e-02 2.817e-02 9.398e-03 7.665e-03 6.333e-03
  deep L=3n, local, ring     1.303e-01 7.367e-02 3.176e-02 1.599e-02 8.622e-03 3.928e-03 1.846e-03
  deep L=3n, global, ring    1.701e-02 6.777e-03 1.479e-03 5.499e-04 1.088e-04 2.963e-05 8.121e-06
  deep L=3n, local, no ring  2.367e-01 2.548e-01 2.469e-01 2.486e-01 2.621e-01 2.285e-01 2.229e-01

量子ビット1つ増やしたときの比 — フィットではなく減衰の形そのもの
----------------------------------------------------------------------------------------------------
  variant                         2->3      3->4      4->5      5->6      6->7      7->8
  shallow L=2, local, ring       0.552     0.418     0.614     0.334     0.816     0.826
  deep L=3n, local, ring         0.566     0.431     0.503     0.539     0.456     0.470
  deep L=3n, global, ring        0.398     0.218     0.372     0.198     0.272     0.274
  deep L=3n, local, no ring      1.077     0.969     1.007     1.054     0.872     0.975

指数フィット Var ~ c r^n
----------------------------------------------------------------------------------------------------
  variant                      r (n=2..8)   r (n=5..8)  decay/qubit (n=5..8)
  shallow L=2, local, ring         0.5401       0.6262                  1.60x
  deep L=3n, local, ring           0.4907       0.4837                  2.07x
  deep L=3n, global, ring          0.2724       0.2479                  4.03x
  deep L=3n, local, no ring        0.9880       0.9546                  1.05x

読み出し自身の1ショット分散 v — コストのもう半分
----------------------------------------------------------------------------------------------------
  variant                          n=2       n=3       n=4       n=5       n=6       n=7       n=8
  shallow L=2, local, ring   8.003e-01 8.892e-01 9.537e-01 9.718e-01 9.906e-01 9.923e-01 9.936e-01
  deep L=3n, local, ring     7.969e-01 8.708e-01 9.280e-01 9.716e-01 9.831e-01 9.919e-01 9.965e-01
  deep L=3n, global, ring    1.531e-01 9.810e-02 5.369e-02 3.349e-02 1.517e-02 7.867e-03 3.969e-03
  deep L=3n, local, no ring  6.541e-01 6.315e-01 6.426e-01 6.431e-01 6.440e-01 6.734e-01 6.653e-01

ショットノイズを std(g) まで下げるのに要する1評価あたりショット数:  S = v / Var(g)
----------------------------------------------------------------------------------------------------
  variant                             n=2          n=4          n=6          n=8   growth/qubit
  shallow L=2, local, ring              4           21          105          157          1.61x
  deep L=3n, local, ring                6           29          114          540          2.09x
  deep L=3n, global, ring               9           36          139          489          1.99x
  deep L=3n, local, no ring             3            3            2            3          1.06x

この実験が測っているもの
----------------------------------------------------------------------------------------------------
  深さ。浅い回路の比はnが増えると1に近づく。量子ビット0の光円錐が広がらなく
  なるので、レジスタを広げても効かなくなる。深い回路の比は1より十分小さい定数
  付近に留まり、これが真の幾何級数的減衰である。
  読み出し。大域的な射影演算子の勾配分散は局所的なものの2倍の速さで1量子ビット
  あたり減衰する — しかし1ショット分散 p(1-p) ~ 2^-n も同じく減衰し、ショット
  コストは 1/Var ではなく v/Var である。そう測ると2つの読み出しは同じオーダーの
  ショット数を要し、増加率もどちらも1量子ビットあたり約2倍である。
  L = 3n ではどちらの回路も自力で集中するほど深いので、この実験は局所性の効果を
  分離できない。局所読み出しを推奨する定理が扱うのは**浅い**回路であり、
  ここの2条件はどちらも浅くない。
  エンタングルメント。CNOTリングを除くと回路は独立な1量子ビットモデルの積に
  なり、分散はnに依らない。同時に表現力も1変数関数の集まりに落ちる。
  訓練可能性と表現力は積み上がるのではなく、互いに取引されている。
```

**注目すべき点。** 同じ回路族から4つの条件、4つの異なる物語が出ます。

**plateauを作るのは深さであり、幅だけではありません。** 浅い回路の量子ビット間の比は 0.5 付近から始まり $n = 8$ では 0.83 まで漂います。量子ビット0の光円錐はCNOTリング2層で広がるのをやめるので、そこから遠い9番目の量子ビットを足してもほとんど何も変わりません。裾のフィットは1量子ビットあたり $1.60\times$ で、なお1に向かって落ちています。深い回路の比は全域で 0.48 付近に留まり、1量子ビットあたり $2.07\times$ の真の幾何級数的減衰で、飽和の兆しがありません。

**大域的な読み出しは勾配を2倍速く縮めますが、ショットは余分にかかりません。** $\langle Z_0\rangle$ を全ビットに台をもつ射影演算子 — 全ゼロ列の確率 — に替えると、*分散*の減衰は1量子ビットあたり $2.07\times$ から $4.03\times$ になります。これが普通引用される数値ですが、会計の半分にすぎません。ショットコストは $v/\mathrm{Var}$ であり、$v$ は1ショットの分散です。全ゼロ確率では1ショットが返すのは*ビット*なので $v = p(1-p)$ で、スクランブリング回路では $p \sim 2^{-n}$ です。したがって分子も1量子ビットあたり約2倍で縮み、真ん中の表がまさにそれを測っています。$v$ は2量子ビットの $0.15$ から8量子ビットの $4.0\times10^{-3}$ へ落ちる一方、局所読み出しの $v$ は $\langle Z_0\rangle$ がゼロに集中するにつれて $0.80$ から $1.00$ へ*上がり*ます。両者を割ると、8量子ビットでの1評価あたりショット数は局所540に対し大域489 — 227倍高いのではなく、大域の方がわずかに*安い* — であり、増加率もどちらも1量子ビットあたり約2倍です。

「局所的に読み出せ」という設計規則が反証されたのではありません。適用範囲が正されたのであり、その範囲こそが要点です。この規則の背後にある定理（コスト関数依存のbarren plateauに関するCerezoらの結果）が扱うのは**浅い**回路であり、そこでは局所コストの勾配が多項式的にしか減衰しないのに大域コストは指数的に集中します。上の深い2条件はどちらも $L = 3n$ 層で、回路が自力で集中するほど深く、そうなると読み出しはもはや領域を決めません。この実験は浅い回路と大域読み出しの組み合わせを一度も走らせないので、2つの効果を分離できないのです。局所読み出しは、効くと分かっている浅く構造化された回路で使い、深く構造のないansatzを救うことを期待しない — これが測定から支持されることであり、通常の言い方より弱い主張です。

**エンタングルメントこそが支払っている対価です。** CNOTリングを消すと分散は $n$ に依らなくなります。1量子ビットあたり $1.05\times$、どの幅でも分解に3ショット。回路はいま独立な1量子ビットモデルの積であり、完璧に訓練可能で、独立な1変数関数4つとちょうど同じだけ表現力があります。つまり相互作用する何かのモデルではありません。訓練可能性と表現力はここで積み上がるのではなく取引されています。第5章はこの観察を形式的な問題に変えます。訓練できるほど浅く局所的な回路は、たいてい古典的にシミュレートできるほど単純でもあり、安価な古典サロゲートを持つモデルには主張できる量子優位がありません。

### 実際には何をするのか

緩和策は実在しますが、どれにも代償があり、何を諦めているかを明示する価値があります。

  * **局所読み出し。** 大域射影演算子ではなく1量子ビット観測量を測る。これは無料であり、浅い回路や構造化された回路では勾配が多項式的に減衰するか指数的に減衰するかの分かれ目になります。Code Example 6 が示すのは、それが買えて*いない*ものです。回路が自力で集中するほど深くなると、大域射影演算子の小さな1ショット分散が速い勾配減衰を打ち消し、1成分あたりのショットコストは同じになります。
  * **浅く構造化した回路。** 深さを制限するか、問題に合った対称性のセクターに回路を制限する（equivariant回路の路線）。これは効きますが、量子モデルを動機づけていた表現力を代償にします。すると誠実な問いは、制限されたクラスがまだ古典的にシミュレートしにくいかどうかになります。
  * **情報のある初期化。** 恒等回路の近く、あるいは既知の良解の近くから始めて、最適化をplateauの外側で開始する。これは化学コースのHartree-Fockの技と同じ理由で効きます。plateauは*ランダムな*パラメータについての主張であり、すべてのパラメータについての主張ではありません。機械学習の問題では、Hartree-Fockに相当する出発点が普通ありません。
  * **層ごとの訓練。** 残りを固定して1層ずつ訓練する。発現を遅らせますが、消しません。

このリストに何が無いかに注意してください。より良いハードウェアです。Code Example 6 の分散は厳密で無雑音のシミュレータで計算しました。barren plateau は回路とコスト関数の性質であってデバイスの性質ではないので、誤り率やコヒーレンス時間の改善はあの表のどの数値も動かしません。ノイズはplateauを悪化させます — 姉妹コースの[第5章](<../../FM/quantum-computing-introduction/chapter-5.html>)がノイズ誘起barren plateauを扱います — が、ここで測った効果は完璧な機械の中にも存在します。

* * *

## 4.5 過学習と、容量と表現力が溶接されている理由

### 周波数の絵と、その代償

第2章はこの規則を自分の単位で確立しました。生成子がPauliの2分の1である符号化ゲートが $L$ 個あれば（第2章は $Z/2$、ここは $Y/2$ で、効くのは固有値間隔だけです）、*回転角*についての周波数集合は $\Omega = \lbrace -L, \ldots, L\rbrace$ です。ここでは角度が $\pi x_q$ なので、同じ主張はこう読めます。$R_y(\pi x_q)$ 符号化を $L$ 層置くと、読み出しに使える期待値は $x_q$ について記述子の単位あたり $1/2$ サイクルの倍数の高調波を $L/2$ まで含む。演習4がスペクトルを直接測り、上限より上のパワーが $10^{-16}$ であることまで含めて、まさにそうなることを確認します。目的関数の $\sin(\pi x_1)$ は $1/2$ サイクルに位置するので、重要な構造を表現するには1層で足ります。

ここに問題があります。ニューラルネットワークでは容量と関数クラスが別々のつまみで調整されます。層を広げればネットワークが表現できる関数の族を変えずにパラメータが増え、活性化関数を変えれば個数を変えずに族が変わります。この形のVQCにはつまみが1つしかありません。層を1つ足すとパラメータが8個増え、*同時に*最大高調波が半サイクル上がります。「パラメータは増やすが滑らかさは同じ」と言う方法がなく、モデルの自然な軸が両者を結びつけています。

この結合は正則化に直接影響します。古典の標準的な対策には明白な量子版がありません。角度に重み減衰はかけられません。角度は周期的で $\theta = 0$ は特別な点ではないからです。ユニタリであるゲートにドロップアウトもできません。残るのは早期停止、深さの制限、そして検証 — つまり4.3節の手順そのものであり、モデルを1つも訓練する前にあの手順を書き下した理由でもあります。

### Code Example 7: 容量と汎化

```python
"""容量と汎化の関係を、両方のモデル族について。
Code Example 1、2、3、4、5 の続き（同一セッション）。
"""
SWEEP_STEPS = 60
SWEEP_SEEDS = (101, 202)


def vqc_sweep(layers):
    tr, te = [], []
    for s in SWEEP_SEEDS:
        p0 = init_vqc(s, layers=layers, y0=ytr.mean())
        pf = adam(lambda q: vqc_grad(q, Xtr, ytr, layers)[0], p0,
                  SWEEP_STEPS, chosen['VQC'][0])
        tr.append(mse_loss(pf, Xtr, ytr, layers))
        te.append(mse_loss(pf, Xte, yte, layers))
    return np.mean(tr), np.mean(te)


def mlp_sweep(h):
    tr, te = [], []
    for s in SWEEP_SEEDS:
        pf = adam(lambda q: mlp_grad(q, Xtr, ytr, h), init_mlp(s, h, ytr.mean()),
                  SWEEP_STEPS, chosen[f'MLP h={H_SMALL}'][0])
        tr.append(mlp_loss(pf, Xtr, ytr, h))
        te.append(mlp_loss(pf, Xte, yte, h))
    return np.mean(tr), np.mean(te)


print(f"容量の走査: {SWEEP_STEPS} ステップ、初期値 {len(SWEEP_SEEDS)} 通りの平均、"
      f"訓練40点")
print("-" * 82)
print(f"  {'model':<24}{'params':>8}{'max harmonic':>14}{'train MSE':>12}"
      f"{'test MSE':>12}{'test/train':>12}")
vq = []
for L in (1, 2, 3, 4):
    a, b_ = vqc_sweep(L)
    npar = n_theta(L) + N_QUBITS + 1
    vq.append((L, npar, a, b_))
    print(f"  {'VQC, L = ' + str(L):<24}{npar:>8}{L/2:>14.1f}"
          f"{a:12.4f}{b_:12.4f}{b_/a:12.2f}")
print()
ml = []
for h in (2, 3, 4, 6, 8, 12):
    a, b_ = mlp_sweep(h)
    ml.append((h, mlp_nparams(h), a, b_))
    print(f"  {'MLP 4-' + str(h) + '-1':<24}{mlp_nparams(h):>8}{'-':>14}"
          f"{a:12.4f}{b_:12.4f}{b_/a:12.2f}")

print("\n各モデル族が到達した最良のテストMSEと、その位置 -- これはORACLEの行である。")
print("ここでは容量をテスト集合上で選んでいるので、上限であって選択ではない")
print("-" * 82)
iv = int(np.argmin([r[3] for r in vq]))
im = int(np.argmin([r[3] for r in ml]))
print(f"  VQC        {vq[iv][0]} layers, {vq[iv][1]:>2} parameters "
      f"-> test MSE {vq[iv][3]:.4f}")
print(f"  MLP        h = {ml[im][0]}, {ml[im][1]:>2} parameters "
      f"-> test MSE {ml[im][3]:.4f}")
print(f"  OLS        5 parameters              -> test MSE {mse_lin:.4f}")
print(f"  mean of y  1 parameter               -> test MSE {mse_const:.4f}")

print("\n容量への感度を数値で述べる")
print("-" * 82)
vt = np.array([r[3] for r in vq]); vr = np.array([r[2] for r in vq])
mt = np.array([r[3] for r in ml]); mr = np.array([r[2] for r in ml])
print(f"  {'family':<12}{'params spanned':>16}{'train MSE range':>20}"
      f"{'test MSE range':>20}{'test/train max':>16}")
print(f"  {'VQC':<12}{f'{vq[0][1]}-{vq[-1][1]}':>16}"
      f"{f'{vr.max():.4f}-{vr.min():.4f}':>20}"
      f"{f'{vt.min():.4f}-{vt.max():.4f}':>20}{max(t/r for t, r in zip(vt, vr)):16.1f}")
print(f"  {'MLP':<12}{f'{ml[0][1]}-{ml[-1][1]}':>16}"
      f"{f'{mr.max():.4f}-{mr.min():.4f}':>20}"
      f"{f'{mt.min():.4f}-{mt.max():.4f}':>20}{max(t/r for t, r in zip(mt, mr)):16.1f}")
print(f"  走査全体でのVQCのテストMSEの振れ幅: {vt.max()/vt.min():.2f}x")
print(f"  走査全体でのMLPのテストMSEの振れ幅: {mt.max()/mt.min():.2f}x")
print("\nこの走査が示すこと")
print("-" * 82)
print("  2つの族の容量への応答はまったく同じではない。ネットワークのテスト誤差は")
print("  パラメータ数5.6倍の範囲にわたって1.6倍以内で平坦であり、test/train比は常に1の")
print("  近傍を離れない。このデータ量では、tanhネットワークの幅を増やすのはほぼ無料で")
print("  ある。VQCのテスト誤差は2層で最良となり、その後は訓練誤差が下がり続ける一方で")
print("  単調に悪化し、test/train比は70に達する。")
print("  量子側の曲線に固有の点が2つある。第一に max harmonic の列である。層を1つ足す")
print("  とパラメータと周波数成分が同時に増えるので、ネットワークの幅と深さのように")
print("  容量と表現力を別々に調整できない。第二に、VQCが訓練損失を買う際のテスト損失")
print("  での交換レートははるかに悪い。これは問題に合った帰納バイアスではなく、過学習")
print("  の定義そのものである。したがって検証の予算を割くべきハイパーパラメータは深さ")
print("  であり、1つの深さを報告することは曲線の1点を報告することにすぎない。")
```

```text
容量の走査: 60 ステップ、初期値 2 通りの平均、訓練40点
----------------------------------------------------------------------------------
  model                     params  max harmonic   train MSE    test MSE  test/train
  VQC, L = 1                    13           0.5      0.2204      0.4775        2.17
  VQC, L = 2                    21           1.0      0.0234      0.1661        7.11
  VQC, L = 3                    29           1.5      0.0088      0.2212       25.12
  VQC, L = 4                    37           2.0      0.0049      0.3433       69.70

  MLP 4-2-1                     13             -      0.0543      0.0525        0.97
  MLP 4-3-1                     19             -      0.0477      0.0390        0.82
  MLP 4-4-1                     25             -      0.0459      0.0488        1.06
  MLP 4-6-1                     37             -      0.0372      0.0333        0.89
  MLP 4-8-1                     49             -      0.0475      0.0399        0.84
  MLP 4-12-1                    73             -      0.0453      0.0433        0.96

各モデル族が到達した最良のテストMSEと、その位置 -- これはORACLEの行である。
ここでは容量をテスト集合上で選んでいるので、上限であって選択ではない
----------------------------------------------------------------------------------
  VQC        2 layers, 21 parameters -> test MSE 0.1661
  MLP        h = 6, 37 parameters -> test MSE 0.0333
  OLS        5 parameters              -> test MSE 0.0465
  mean of y  1 parameter               -> test MSE 0.2747

容量への感度を数値で述べる
----------------------------------------------------------------------------------
  family        params spanned     train MSE range      test MSE range  test/train max
  VQC                    13-37       0.2204-0.0049       0.1661-0.4775            69.7
  MLP                    13-73       0.0543-0.0372       0.0333-0.0525             1.1
  走査全体でのVQCのテストMSEの振れ幅: 2.88x
  走査全体でのMLPのテストMSEの振れ幅: 1.58x

この走査が示すこと
----------------------------------------------------------------------------------
  2つの族の容量への応答はまったく同じではない。ネットワークのテスト誤差は
  パラメータ数5.6倍の範囲にわたって1.6倍以内で平坦であり、test/train比は常に1の
  近傍を離れない。このデータ量では、tanhネットワークの幅を増やすのはほぼ無料で
  ある。VQCのテスト誤差は2層で最良となり、その後は訓練誤差が下がり続ける一方で
  単調に悪化し、test/train比は70に達する。
  量子側の曲線に固有の点が2つある。第一に max harmonic の列である。層を1つ足す
  とパラメータと周波数成分が同時に増えるので、ネットワークの幅と深さのように
  容量と表現力を別々に調整できない。第二に、VQCが訓練損失を買う際のテスト損失
  での交換レートははるかに悪い。これは問題に合った帰納バイアスではなく、過学習
  の定義そのものである。したがって検証の予算を割くべきハイパーパラメータは深さ
  であり、1つの深さを報告することは曲線の1点を報告することにすぎない。
```

**注目すべき点。** 2つの族の容量への応答はまったく同じではなく、その違いは最良スコアの差より大きいものです。パラメータ数5.6倍の範囲にわたってネットワークのテスト誤差は 0.0333 から 0.0525 まで1.58倍しか動かず、test/train比は常に1の近傍を離れません。40点では、tanhネットワークの幅を増やすのはほぼ無料です。VQCのテスト誤差は2層の 0.1661 が最良で、そこから4層の 0.3433 まで単調に悪化します。*その間も訓練誤差は下がり続けて* 0.0049 に達し、test/train比は 69.7 になります。VQCはここで単に劣っているだけでなく、ハイパーパラメータの選択に対してはるかに敏感なのです。

`max harmonic` の列がその一因を説明します。VQCの4行は1つのモデルクラスの4つの容量ではなく、それぞれ異なる滑らかさの事前分布を持つ4つの*別の関数クラス*であり、パラメータ数はその滑らかさと連動して動きます。当てはめる関数族とは独立にモデルの複雑さを選ぶことに慣れた材料研究者は、それを手放す必要があります。実務上これは、層数がVQCの最も重要なハイパーパラメータであり、検証データ上で選ばなければならず、走査なしに1つの深さを報告する論文は、形を測っていない曲線の1点を報告しただけだということを意味します。

### VQCがここで勝つには何が真である必要があるか

「量子モデルはこのベンチマークで負けた」は、この分野の熱狂も批判もふつう認めるよりずっと弱い主張なので、条件を肯定形で述べておく価値があります。

  * **目的関数の構造が回路の構造と一致していること。** 符号化が周波数の台を固定し、エンタングルのパターンがどの交差項が安価かを固定します。ちょうどそれらの項から作った目的関数はVQCに有利で、そうした目的関数の構成が文献の肯定的な結果の大半が作られる方法です。原理実証としては正当で、材料データについての証拠としては無価値です。
  * **対称性が共有されていること。** 物理の問題がある群のもとで不変で、回路が同じ群のもとで同変なら、モデルは汎用ネットワークが欠く真の帰納バイアスを得ます。これは分野で最も有望な方向であり、道具ではなく現役の研究領域です。
  * **データがパラメータよりはるかに多いこと。** 40点と29パラメータではすべてが過学習し、比較はモデルクラスよりも正則化に支配されます。本章は $10^4$ 点で何が起きるかを何も確立しておらず、正直な理由は parameter-shift の予算がその実験を高価にするからです。1ステップ1960回の回路評価が49万回になります。
  * **データが量子であること。** これは程度ではなく議論の種類を変える唯一の条件であり、第5章の主題です。

* * *

## 演習

#### 演習1: $R_y$ を超えたshift則

  1. $f(\theta) = a + b\cos\theta + c\sin\theta$ を用いて $\bigl[f(\theta+\pi/2) - f(\theta-\pi/2)\bigr]/2 = f'(\theta)$ を代数的に確かめ、$G^2 = I$ の仮定がどこで入ったかを述べてください。
  2. $R_y(\theta)|0\rangle$ 上の $\langle Z\rangle$ について、$\theta = 0.7$ で2項則が厳密であることを数値的に確認してください。
  3. 2量子ビットゲート $\exp(-i\theta\,Z\otimes Z/2)$ でも同じ2項則が成立することを示し、その理由を説明してください。
  4. 固有値が $\lbrace 0, 1, 2\rbrace$ の生成子 $G$ から作られるゲートには2回の評価では足りません。係数を導かずに、$f(\theta)$ に現れる相異なる周波数の数と、したがって必要なずらした評価の回数を述べてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(f(\theta \pm \pi/2) = a \mp b\sin\theta \pm c\cos\theta\) なので差の半分は \(-b\sin\theta + c\cos\theta\)、これはちょうど \(f'(\theta)\) です。\(G^2 = I\) の仮定は、\(f\) が周波数 \(0\) と \(1\) だけを含むと主張したところで入りました。\(G^2 = I\) ならゲートは \(\cos(\theta/2)I - i\sin(\theta/2)G\) なので \(\langle O\rangle\) は \(\cos(\theta/2), \sin(\theta/2)\) の二次式であり、倍角公式でそれが \(1, \cos\theta, \sin\theta\) に潰れてそれ以上の高調波が出ません。</p>

<p><strong>2.</strong> \(\langle Z\rangle = \cos\theta\) が厳密に成立するので微分は \(-\sin(0.7) = -0.644217687238\)、shift則は同じ12桁を返します。</p>

<p><strong>3.</strong> \((Z\otimes Z)^2 = I\) なので1の議論がそのまま適用され、生成子が作用する量子ビット数は無関係です。下のコードは中心差分と比べて \(7\times10^{-11}\) で一致しており、これは有限差分の丸め誤差です。</p>

<p><strong>4.</strong> 固有値差 \(\lbrace 0, \pm 1, \pm 2\rbrace\) は \(\theta\) について周波数 \(0, 1, 2\)、すなわち5つの実係数 \((a, b_1, c_1, b_2, c_2)\) を与えます。微分にはずらした評価4回で足ります。これが制御回転など固有値が3つある生成子に用いられる4項shift則です。</p>

```python
"""演習1。Code Example 1 の続き（同一セッション）。"""
th = 0.7


def f_ry(t):
    return expval(apply_gate(ket('0'), ry(t), [0], 1), 'Z')


print(f"Ry:  shift rule {0.5*(f_ry(th+np.pi/2) - f_ry(th-np.pi/2)):.12f}"
      f"   厳密値 -sin(theta) {-np.sin(th):.12f}")


def rzz(t):
    """exp(-i t ZZ / 2)。ZZ が対角なのでこれも対角。"""
    return np.diag(np.exp(-1j * t / 2 * np.array([1, -1, -1, 1])))


def f_zz(t):
    psi = apply_gate(apply_gate(ket('00'), H, [0], 2), H, [1], 2)
    return expval(apply_gate(psi, rzz(t), [0, 1], 2), 'XI')


print(f"ZZ:  shift rule {0.5*(f_zz(th+np.pi/2) - f_zz(th-np.pi/2)):.12f}"
      f"   中心差分      {(f_zz(th+1e-6) - f_zz(th-1e-6))/2e-6:.12f}")
```

```text
Ry:  shift rule -0.644217687238   厳密値 -sin(theta) -0.644217687238
ZZ:  shift rule -0.644217687238   中心差分      -0.644217687307
```

</details>

#### 演習2: 沈黙する角度を数える

Code Example 2 の回路で、各層を*符号化、回転、エンタングル* — リングを最後 — の順序にし、読み出しを $\langle Z_0\rangle$ だけにした変種を作ってください。

  1. 24個の角度のうち、勾配が恒等的に零になるものはいくつありますか。
  2. それぞれを (層, 量子ビット, ゲート種別) として同定してください。
  3. リング `cnot(0,1), cnot(1,2), cnot(2,3), cnot(3,0)` を通した $Z_0$ のClifford引き戻しから、それぞれを説明してください。
  4. 2つの順序のどちらを使いますか。また、その答えは文献で報告されるパラメータ数について何を意味しますか。

<details>
<summary>解答</summary>

<p><strong>1-2.</strong> 5個、すべて最終層です。\((2,0,R_y)\)、\((2,0,R_z)\)、\((2,1,R_z)\)、\((2,2,R_z)\)、\((2,3,R_z)\)。</p>

<p><strong>3.</strong> 観測量に対してリングを逆順に打ち消します。<code>cnot(3,0)</code> は \(Z_0 \to Z_0Z_3\)、<code>cnot(2,3)</code> は \(Z_3 \to Z_2Z_3\) なので \(Z_0Z_2Z_3\)、<code>cnot(1,2)</code> で \(Z_0Z_1Z_2Z_3\)、<code>cnot(0,1)</code> は \(Z_1 \to Z_0Z_1\) で2つの \(Z_0\) が打ち消し合い \(Z_1Z_2Z_3\) が残ります。この実効観測量は量子ビット0上で恒等演算子なので、最終層の量子ビット0の回転2つは見えません。さらにこれは対角なので、\(R_z\) は \(Z\) の任意の積と可換であり、他の3量子ビットの最終層 \(R_z\) も見えません。合計5個です。</p>

<p><strong>4.</strong> 沈黙する角度がない Code Example 2 の順序（符号化、エンタングル、回転）を量子ビットごとの読み出しと組み合わせて使います。含意は、報告されたパラメータ数はVQCの容量の上界にすぎず、実質を大きく上回りうる（ここでは21%）ということ、そして検査はパラメータについてのループ1つで済むので、新しいansatzのパラメータ数を引用する前に実行すべきだということです。</p>

```python
"""演習2。Code Example 1 の続き（同一セッション）。"""


def state_rotate_then_entangle(x, theta, layers=3):
    """各層で符号化、回転、*その後*エンタングル — 避けるべき順序。"""
    n = len(x)
    psi = ket('0' * n)
    k = 0
    for _ in range(layers):
        for q in range(n):
            psi = apply_gate(psi, ry(np.pi * x[q]), [q], n)
        for q in range(n):
            psi = apply_gate(psi, ry(theta[k]), [q], n); k += 1
            psi = apply_gate(psi, rz(theta[k]), [q], n); k += 1
        for q in range(n):
            psi = cnot(psi, q, (q + 1) % n, n)
    return psi


rg = np.random.default_rng(5)
th0 = rg.uniform(0, 2 * np.pi, 24)
silent = []
for k in range(24):
    biggest = 0.0
    for xq in rg.uniform(0, 1, (8, 4)):
        tp = th0.copy(); tp[k] += np.pi / 2
        tm = th0.copy(); tm[k] -= np.pi / 2
        biggest = max(biggest,
                      abs(expval(state_rotate_then_entangle(xq, tp), 'ZIII')
                          - expval(state_rotate_then_entangle(xq, tm), 'ZIII')) / 2)
    if biggest < 1e-12:
        silent.append(k)
print(f"沈黙する角度: 24 個中 {len(silent)} 個 -> 添字 {silent}")
print("(layer, qubit, gate) 表記: " + ", ".join(
    f"({k//8}, {(k%8)//2}, {'Ry' if k % 2 == 0 else 'Rz'})" for k in silent))
```

```text
沈黙する角度: 24 個中 5 個 -> 添字 [16, 17, 19, 21, 23]
(layer, qubit, gate) 表記: (2, 0, Ry), (2, 0, Rz), (2, 1, Rz), (2, 2, Rz), (2, 3, Rz)
```

</details>

#### 演習3: ショットノイズが勾配を食べるとき

各期待値は $S$ ショットから推定されるので、$\langle Z_q\rangle$ の分散は高々 $1/S$ です。

  1. 回路角1成分の勾配のショットノイズ分散が $\dfrac{4}{N^2}\Bigl(\sum_i r_i^2\Bigr)\Bigl(\sum_q w_q^2\Bigr)\dfrac{1}{2S}$ であることを示し、用いた独立性の仮定を述べてください。
  2. Code Example 2 の未訓練パラメータについて定数を評価し、典型的な成分の信号対雑音比が1と10になる $S$ を求めてください。
  3. 訓練中に残差は小さくなります。それは必要な $S$ に何をし、固定したショット予算が到達できる損失について何を意味しますか。
  4. SNR $= 10$ の $S$ に1ステップ1960回の回路評価と120ステップを掛け、毎秒 $10^4$ 回実行できるデバイスと比べてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> ずらした各評価は独立な実験なので \(\mathrm{Var}[\tfrac{1}{2}(z^+ - z^-)] = \tfrac{1}{4}(\sigma^2 + \sigma^2) = \sigma^2/2\)、ここで \(\sigma^2 \le 1/S\) です。連鎖律 \(g_k = \frac{2}{N}\sum_i r_i \sum_q w_q \,\mathrm{d}z_{iq}\) を独立なデータ点と量子ビットについて足すと式が出ます。仮定は \(2Nn_\theta\) 回の評価を独立とみなすことで、ショットを再利用して相関させれば分散はやや下がります。</p>

<p><strong>2.</strong> 定数は 0.017765 なので \(\sigma(g) = \sqrt{0.017765/S}\)、典型的な \(|g| = 0.031468\) に対して SNR 1 は \(S \approx 18\)、SNR 10 は \(S \approx 1800\) です。</p>

<p><strong>3.</strong> \(\sum_i r_i^2\) は訓練MSEの \(N\) 倍なので \(\sigma(g) \propto \sqrt{L/S}\)、勾配のノイズは損失の平方根としてしか下がりません。機構のもう半分が結論を効かせます。\(L\) は0まで下がらず、データ自身のノイズ床 \(L_{\min}\) で底を打つので \(\sigma(g)\) も \(\sqrt{L_{\min}/S}\) で底を打ちます。一方で真の勾配は極小までの距離に比例して消え、超過損失 \(L - L_{\min}\) はその距離の2乗で消えます。\(|g| = \sigma(g)\) とおくと、極小のまわりにすべてのステップがノイズと区別できない近傍が残り、その端での超過損失は \(L_{\min}/S\) のオーダーになります。つまり到達できる訓練損失は、ステップをいくら重ねても \(1/S\) より速く床へ近づきません。シミュレータでは再現されないのに実機のVQC訓練で見られる停滞は、この機構によるものです。</p>

<p><strong>4.</strong> \(1800 \times 1960 \times 120 = 4.23\times10^{8}\) 回の回路実行で、毎秒 \(10^4\) 回なら連続で11.8時間のデバイス時間です。データ40点、パラメータ29個、線形回帰に負けるモデルのために。共用ハードウェアでの実時間は、待ち行列と校正を含めるとかなり長くなります。</p>

```python
"""演習3。Code Example 1、2、3 の続き（同一セッション）。"""
g, _ = vqc_grad(params0, Xtr, ytr)
typical = np.abs(g[:n_theta()]).mean()
theta_, w_, b_ = unpack(params0)
Zf = np.array([vqc_features(x, theta_) for x in Xtr])
resid = (Zf @ w_ + b_) - ytr
const = 4.0 / len(Xtr) ** 2 * np.sum(resid ** 2) * np.sum(w_ ** 2) * 0.5
print(f"{n_theta()} 個の角度についての mean |dL/dtheta|: {typical:.6f}")
print(f"Var[g] = (4/N^2) (sum_i r_i^2)(sum_q w_q^2)/(2S) = {const:.6f} / S")
print(f"  {'S':>9}{'sigma(g)':>12}{'SNR':>9}")
for S in (10, 100, 1000, 10000, 100000):
    sg = np.sqrt(const / S)
    print(f"  {S:>9}{sg:12.6f}{typical/sg:9.2f}")
S1 = int(np.ceil(const / typical ** 2))
print(f"SNR = 1 は期待値1つあたり S ~ {S1:,} ショット、SNR = 10 には"
      f" {100*S1:,} ショット")
print(f"SNR = 10 での訓練1回: {100*S1:,} x 1960 x 120 ="
      f" {100*S1*1960*120:,} 回の回路実行")
print(f"毎秒1e4回の実行なら {100*S1*1960*120/1e4/3600:.1f} 時間")
```

```text
24 個の角度についての mean |dL/dtheta|: 0.031468
Var[g] = (4/N^2) (sum_i r_i^2)(sum_q w_q^2)/(2S) = 0.017765 / S
          S    sigma(g)      SNR
         10    0.042148     0.75
        100    0.013328     2.36
       1000    0.004215     7.47
      10000    0.001333    23.61
     100000    0.000421    74.66
SNR = 1 は期待値1つあたり S ~ 18 ショット、SNR = 10 には 1,800 ショット
SNR = 10 での訓練1回: 1,800 x 1960 x 120 = 423,360,000 回の回路実行
毎秒1e4回の実行なら 11.8 時間
```

</details>

#### 演習4: モデルから周波数の台を読み取る

  1. $L = 1, 2, 3$ について $\langle Z_0\rangle$ を $x_1$ のみの関数として2周期分標本化し、FFTをとってください。どの高調波が存在しますか。
  2. $L$ と最大高調波を結ぶ規則を述べ、半角の規約から説明してください。
  3. 目的関数は $\sin(\pi x_1)$ を含みます。それはどの高調波にあり、それを表現できる最小の $L$ はいくつですか。
  4. 3の結果を踏まえると、なぜ Code Example 7 の走査で $L = 1$ が勝たないのですか。

<details>
<summary>解答</summary>

<p><strong>1-2.</strong> \(L = 1\) は高調波 0 と 0.5、\(L = 2\) は 1.0 を追加、\(L = 3\) は 1.5 を追加します。最大高調波は単位 \(x\) あたり \(L/2\) サイクルです。各符号化ゲート \(R_y(\pi x)\) は振幅に \(\cos(\pi x/2), \sin(\pi x/2)\)、つまり4分の1サイクルを寄与します。期待値は振幅の二次式なので1層で半サイクルに達し、\(L\) 層はそうした因子を \(L\) 個掛けます。</p>

<p><strong>3.</strong> \(\sin(\pi x_1)\) は \(x_1\) について周期2なので単位あたり 0.5 サイクルにあり、\(L = 1\) で既に含まれています。</p>

<p><strong>4.</strong> ある周波数を含むことは、目的関数全体に当てはめられることと同じではありません。1層ではモデルの角度は8個で、どの記述子についても最低次の高調波しか持たないため、\(0.5x_3^2\) の項、積構造 \(\sin(\pi x_1)\cos(\pi x_2)\)、そしてオフセットを同時に表現できません。この走査では2層が最良の妥協です。教訓は、周波数の台は必要条件であって十分条件ではなく、変分ブロックがその張る空間の正しい点に到達しなければならないということです。</p>

```python
"""演習4。Code Example 1、2 の続き（同一セッション）。"""
base = np.array([0.4, 0.55, 0.2, 0.7])
for L in (1, 2, 3):
    grid = np.arange(64) / 64.0 * 2.0          # x1 の2周期分
    th = np.random.default_rng(4).uniform(0, 2 * np.pi, n_theta(L))
    vals = np.array([vqc_features(np.where(np.arange(4) == 0, t, base), th, L)[0]
                     for t in grid])
    amp = np.abs(np.fft.rfft(vals)) / len(grid)
    present = "  ".join(f"{j/2:.1f}:{amp[j]:.4f}" for j in range(6) if amp[j] > 1e-10)
    print(f"L = {L}:  存在する高調波 -> {present}"
          f"   {L/2:.1f} より上の成分: {np.sum(amp[L+1:]**2)**0.5:.2e}")
print("目的関数の sin(pi x1) は x1 について 0.5 cycles per unit に位置する")
```

```text
L = 1:  存在する高調波 -> 0.0:0.0742  0.5:0.0329   0.5 より上の成分: 1.09e-16
L = 2:  存在する高調波 -> 0.0:0.2490  0.5:0.0273  1.0:0.0336   1.0 より上の成分: 9.42e-17
L = 3:  存在する高調波 -> 0.0:0.0708  0.5:0.1185  1.0:0.1030  1.5:0.0291   1.5 より上の成分: 1.08e-16
目的関数の sin(pi x1) は x1 について 0.5 cycles per unit に位置する
```

</details>

#### 演習5: plateauを外挿する

Code Example 6 の裾のフィット、局所読み出しで分散 $\sim 0.4837^n$、大域読み出しで $\sim 0.2479^n$、1ショット分散はそれぞれ $v \approx 1$ と $v \approx 2^{-n}$ を用います。

  1. $S = v/\mathrm{Var}$ を使い、$n = 10, 20, 30, 50$ で勾配1成分を分解するのに必要なショット数を各読み出しについて見積もってください。
  2. 毎秒 $10^4$ 回の回路実行として、年単位の実時間に換算してください。
  3. 各読み出しで、勾配1成分あたり実機時間1時間を超えるのはどの $n$ ですか。
  4. ある提案が「誤り率が10倍改善すれば50量子ビットのVQCを訓練する」と述べています。この論法の誤りはどこですか。

<details>
<summary>解答</summary>

<p><strong>1-2.</strong> 表を参照。局所読み出しは \(n=10\) で \(1.4\times10^{3}\) ショット、\(n=30\) で \(2.9\times10^{9}\)、\(n=50\) で \(5.9\times10^{15}\)。最後は毎秒 \(10^4\) ショットで約 \(1.9\times10^{4}\) 年です。大域読み出しは同じ幅で \(1.1\times10^{3}\)、\(1.4\times10^{9}\)、\(1.7\times10^{15}\)、すなわち50量子ビットで \(5.4\times10^{3}\) 年です。両者はどこでも4倍以内におさまり、しかも大域の方が安いのです。\(v = 2^{-n}\) の因子を落とすと \(n = 50\) で大域読み出しが \(10^{15}\) 倍悪く見えてしまいます。それが避けるべき算術です。</p>

<p><strong>3.</strong> 1時間は \(3.6\times10^{7}\) ショットです。ショット数は局所で \((1/0.4837)^n = 2.067^n\)、大域で \((2\times0.2479)^{-n} = 2.017^n\) と増えるので、到達点は \(n = 24.0\) と \(n = 24.8\) — 1量子ビット以内で同じ場所です。しかもこれは勾配の1<em>成分</em>であり、全勾配にはその \(2n_\theta\) 倍が必要です。</p>

<p><strong>4.</strong> 誤り率はこの障害に無関係です。上のすべての数値は厳密で無雑音のシミュレータから出ています。分散の減衰は回路のランダムさと観測量の台の性質であり、ハードウェアの性質ではありません。誤り率が10倍改善すればより長い回路が使えますが、構造のないansatzではそれはplateauを浅くするどころか深くします。この論法はハードウェアの制約と数学的な制約を混同しており、この領域で最もよくある誤りです。</p>

```python
"""演習5。NumPyのみ。2つの減衰率は Code Example 6 の裾のフィット値。"""
import numpy as np
SEC_PER_YEAR = 3.15576e7
# S = v / Var(g)。<Z_0> の1ショット分散 v は約1。全ゼロ射影演算子では p(1-p) ~ 2^-n
# であり、1量子ビットあたり2倍の因子を1つ打ち消す。
for r, vfun, name in ((0.4837, lambda n: 1.0, "local read-out "),
                      (0.2479, lambda n: 2.0 ** -n, "global read-out")):
    print(f"{name}: Var ~ {r:.4f}^n,  v = {'1' if vfun(1) == 1.0 else '2^-n'}")
    print(f"  {'n':>4}{'Var':>12}{'v':>12}{'shots':>12}"
          f"{'seconds at 1e4/s':>19}{'years':>12}")
    for n in (10, 20, 24, 30, 50):
        var = r ** n
        shots = vfun(n) / var
        print(f"  {n:>4}{var:12.2e}{vfun(n):12.2e}{shots:12.2e}{shots/1e4:19.2e}"
              f"{shots/1e4/SEC_PER_YEAR:12.2e}")
    eff = r if vfun(1) == 1.0 else r / vfun(1)     # shots ~ (1/eff)^n
    n_hour = np.log(3.6e7) / np.log(1.0 / eff)
    print(f"  ショット数は {1/eff:.4f}^n で増え、実機時間1時間に達するのは"
          f" n = {n_hour:.1f}（勾配の*1成分*について）")
```

```text
local read-out : Var ~ 0.4837^n,  v = 1
     n         Var           v       shots   seconds at 1e4/s       years
    10    7.01e-04    1.00e+00    1.43e+03           1.43e-01    4.52e-09
    20    4.91e-07    1.00e+00    2.03e+06           2.03e+02    6.45e-06
    24    2.69e-08    1.00e+00    3.72e+07           3.72e+03    1.18e-04
    30    3.45e-10    1.00e+00    2.90e+09           2.90e+05    9.20e-03
    50    1.69e-16    1.00e+00    5.90e+15           5.90e+11    1.87e+04
  ショット数は 2.0674^n で増え、実機時間1時間に達するのは n = 24.0（勾配の*1成分*について）
global read-out: Var ~ 0.2479^n,  v = 2^-n
     n         Var           v       shots   seconds at 1e4/s       years
    10    8.77e-07    9.77e-04    1.11e+03           1.11e-01    3.53e-09
    20    7.68e-13    9.54e-07    1.24e+06           1.24e+02    3.93e-06
    24    2.90e-15    5.96e-08    2.05e+07           2.05e+03    6.51e-05
    30    6.73e-19    9.31e-10    1.38e+09           1.38e+05    4.38e-03
    50    5.17e-31    8.88e-16    1.72e+15           1.72e+11    5.44e+03
  ショット数は 2.0169^n で増え、実機時間1時間に達するのは n = 24.8（勾配の*1成分*について）
```

</details>

* * *

## まとめ

### 要点

**1\. VQCはコストがデータ依存になったVQEである**

  * 回路も parameter-shift 勾配も古典最適化器も同じです。観測量が読み出しになり、エネルギーが残差の和になります。
  * 唯一の本質的な違いは、VQEのコストが目的そのものである一方、VQCの訓練損失はそうではないことです。訓練損失が低いほど悪いことがあり、Code Example 5 では実際にそうなりました。
  * 符号化が表現できるものを決め、変分ブロックがその中で到達できるものを決め、読み出しが訓練可能性を決めます。3つの判断、3つの役割であり、これは日常的に混同されています。

**2\. parameter-shiftは厳密で、そして高価である**

  * $G^2 = I$ を満たす任意の生成子について期待値は $a + b\cos\theta + c\sin\theta$ であり、$\pi/2$ 離れた2回の評価が微分を厳密に与えます。有限差分とは $6.5\times10^{-11}$ で一致しました。
  * コストは1ステップあたり $N + 2Nn_\theta$ 回の回路評価で、ここでは1960回です。120ステップを各1000ショットで走らせると2億3520万ショットになり、古典側は逆伝播120回で済みます。
  * シミュレータの逆伝播は代替になりません。実機が作れない勾配を与えるので、そうして得た訓練曲線は訓練可能な量子モデルを示していないのです。

**3\. パラメータ数は嘘をつきうるし、検査は安い**

  * 回路の2行を入れ替えると24個の角度のうち5個が沈黙します（任意の入力で勾配が恒等的に零になります）。Cliffordリングが $Z_0$ を $Z_1Z_2Z_3$ に引き戻し、$R_z$ が対角観測量と可換だからです。
  * 引用されたパラメータ数は容量の上界です。引用する前に Code Example 2 の感度ループを実行してください。
  * 読み出し重みとオフセットもパラメータであり、それを持たないVQCは出力範囲が固定されたモデルです。

**4\. このデータでは量子モデルが負け、その理由はハードウェアではない**

  * パラメータ数を揃え（29対25と31）、最適化器を1つにし、選択手順を1つにし、初期値3通りを報告した条件で、早期停止したVQCのテストMSEの中央値は 0.0925、同規模ネットワークの最良は 0.0239 で、3.87倍の差でした。対応のあるブートストラップ区間は $[+0.020, +0.126]$ なので、この負けは解像できます。
  * 一方*大きい*方の同規模ネットワーク（31パラメータ、0.0773）に対する同じ区間は $[-0.027, +0.062]$ で引き分けです。R6は「両方のネットワークに負ける」を「一方に負け、他方と引き分ける」に変えます。これが結果の誠実な形です。
  * 予算の終わりでは3モデル中で*最小の訓練損失*かつ*最大のテスト損失*（0.0059 対 0.2939）になります。40点を過学習しており、古典的な表形式データに対するNISQ期の想定どおりの結果です。
  * VQCは5パラメータの線形回帰 0.0465 にも負け、これに勝つのは25パラメータのネットワークだけです。線形回帰の行を落とせば、結果は引き分けのように見えたでしょう。

**5\. barren plateauは回路の性質であってデバイスの性質ではない**

  * 量子ビット1個あたりの勾配分散の減衰、実測値: 浅い回路 $1.60\times$、深い回路＋局所読み出し $2.07\times$、深い回路＋大域読み出し $4.03\times$、エンタングルのリングを除去 $1.05\times$。
  * 大域読み出しの余分な因子は*ショット*の増加を買いません。全ゼロ確率自身の1ショット分散 $p(1-p)\sim2^{-n}$ が一緒に減衰するからです。8量子ビットでの実測ショットコストは局所540に対し大域489で、どちらも1量子ビットあたり約2倍で増えます。「局所的に読み出せ」という規則は浅い回路のものであり、$L = 3n$ では回路が自力で集中して読み出しはもはや領域を決めません。
  * 深さがplateauを作り、エンタングルメントこそが支払っている対価です。完璧に訓練できる積回路は1変数モデルの積にすぎません。
  * これらの数値はすべて厳密で無雑音のシミュレータから出ています。誤り率の改善はどれ1つも変えません。

**6\. 容量と表現力はつまみを共有している**

  * $L$ 層は記述子あたり $L/2$ サイクルまでの高調波*と* $8L$ 個のパラメータを与えます。「パラメータは増やすが滑らかさは同じ」という方向が存在しません。
  * 古典の正則化は移植できません。周期的な角度に重み減衰はなく、ユニタリにドロップアウトもありません。残るのは深さの選択、早期停止、検証分割です。
  * したがって深さはVQCの最重要ハイパーパラメータであり、単一深さの結果は形を測っていない曲線の1点にすぎません。

**実務への含意**

  * 何かを訓練する前に手順を書き下してください。分割、最適化器、パラメータ予算、ハイパーパラメータのグリッド、初期値の本数です。後から書き下すのが、誠実な人が不誠実なベンチマークを作る道筋です。
  * 線形ベースラインと平均予測ベースラインを常に含めてください。量子モデルがその両方に勝てないなら、比較の他の部分はどうでもよくなります。
  * 精度と並べてショット数と回路評価回数を報告してください。優位性は比であり、ここでの分母は5〜6桁です。
  * 「ハードウェアが良くなれば動く」という形の主張は、障害がハードウェアのものだという証明を要すると見なしてください。plateauと集中については、そうではありません。

第5章は議論をひっくり返します。量子モデルの関数クラスを明示的に書き下せるなら — 浅く局所的な回路については書き下せます — 同じ関数クラスを持つ古典モデルはそれに並ぶはずです。次章はそのようなサロゲートを2つ作り、どこまで近づくかを測り、優位性の主張を読むためのチェックリストを組み立て、答えが変わるには何が変わらなければならないかを本コースにできる限りの精度で述べます。

[← 第3章: 量子カーネル法](<chapter-3.html>) [第5章: 誠実な評価と展望 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 本章の性能比較はすべて、1つの合成60点データセット・1つのアーキテクチャ・1つの最適化器で測定したものです。当該実験を特徴づけるものであり、量子機械学習と古典機械学習の一般的な優劣として読んではなりません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
