---
title: "第5章: 誠実な評価と展望"
chapter_title: "第5章: 誠実な評価と展望"
subtitle: 脱量子化、古典サロゲート、ベンチマークの衛生、そして答えを変える唯一のもの
reading_time: 45-50分
difficulty: 上級
code_examples: 6
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/MI/quantum-machine-learning-introduction/chapter-5.html>) | Last sync: 2026-08-13

[マテリアルズ・インフォマティクス道場](<../index.html>) > [量子機械学習入門](<index.html>) > 第5章

4つの章で量子機械学習のモデルを作り、測ってきました。どれも勝ちませんでした。第3章のエンタングル型忠実度カーネルは、同一手順の下で調整済みの動径基底関数（RBF）カーネルより1.16倍悪く、そこでRBFに並んだ唯一の量子カーネルはエンタングルメントを全く含まない*積*角度符号化 — 同章が述べたとおり、量子の衣をまとった古典カーネル — でした。第4章の変分回路は同規模のNumPy製ネットワークに3.9倍の差で負けました。対応のあるブートストラップ区間は、この負けを同一予算の2つのネットワークのうち小さい方に対しては解像し、大きい方に対しては引き分けと判定します。そして5パラメータの線形回帰にも負けました。どの敗北もハードウェアが原因ではありません。本コースのすべての数値は厳密で無雑音のシミュレータから出ているので、誤り率もコヒーレンス時間も量子ビット数も一度も邪魔をしていないのです。

本章はなぜそうなったのかを説明し、より有用なことに、そうならないためには何が必要かを述べます。説明には*脱量子化（dequantization）*という名前があり、それは標語ではありません。量子モデルを受け取って、同等の性能と多項式コストを持つ古典モデルを作り出す構成の一族です。5.2節ではそのうち2つをゼロから作り、どこまで近づくかを測ります。とても近くまで来ますし、一方は模倣する相手の量子モデルを上回ります。

その結果は注意して読む必要があります。どちらの方向にも誘惑が強いからです。「量子機械学習は動かない」も「量子機械学習は材料探索に革命を起こす」も、この証拠からは同程度に正当化されません。証拠が支持するのはもっと狭く、もっと有用な主張です。材料データベースに入っているような*古典的な*データに対して、本コースのモデルには提供できる優位性がなく、障害は技術的というより数学的であり、議論を変える具体的なものは良い機械ではなく別種のデータである、というものです。5.3節と5.4節がその論を立て、材料研究者が今なにをすべきかを述べます。

## 学習目標

本章を読み終えると、次のことができるようになります。

  * 脱量子化が何を証明し何を証明しないかを述べ、計算量理論の定理としての意味と、通常使われる緩い意味を区別できる
  * 量子カーネルの低weight Pauli（古典シャドウ）サロゲートを構成し、量子ビット数に対する特徴数を数え、カーネルのどれだけを再現できるかを測定できる
  * 符号化自身の周波数スペクトルから打ち切りFourierサロゲートを構成し、スペクトルが有限であることを数値的に確認し、サロゲートのテスト誤差を量子モデルと比較できる
  * 1つのデータセット上で、算術的に真でありながら互いに矛盾する3通りの要約を示し、それぞれの背後にある操作を名指しできる
  * 量子優位の主張に8項目の衛生チェックリストを適用し、その主張がどの項目に答えていないかを特定できる
  * なぜ量子*データ*が古典データの場合に支配的だった入力問題を取り除くのか、その代わりにどんなコストが生じるのか、材料科学のどの測定がこれに該当しうるのかを説明できる
  * 特定の研究計画について、この分野のどの部分を今学ぶ価値があり、どの部分は待つ価値があるかを判断できる

* * *

## 5.1 脱量子化が実際に述べていること

### 1つの語をまとう2つの主張

技術文献における*脱量子化*とは、ある種の定理を指します。ある精度をある資源量で達成する量子アルゴリズムが与えられたとき、同等の精度を同じパラメータの多項式コストで達成する古典アルゴリズムを提示するのが脱量子化の結果です。ただし入力の与えられ方について対等な仮定 — **sample-and-query access**、すなわち成分の2乗に比例する確率で添字を引ける能力と任意の成分を読める能力であり、量子アルゴリズムが仮定する効率的な状態準備の古典的対応物 — の下で成立します。この仮定が仕事をしており、見落としやすいもう1つの仮定も同じです。これらの結果が適用されるのは、関係する行列が**低ランク**であるか、他の意味で小さな記述をもつ場合です。最もよく知られた例は、多くのQML提案の土台になっていた量子線形代数アルゴリズム、すなわち推薦システム、主成分分析、低ランク線形方程式に関するものです。いずれの場合も、指数的高速化は入力モデルに依存しており、同じ入力モデルを古典アルゴリズムにも等しく与えると古典側も速くなってしまいました。

**脱量子化されていない**ものを同じ息で名指しておく価値があります。「量子線形代数は脱量子化された」は日常的に読み過ごされるからです。**疎で条件のよい**行列に対する元来の量子線形方程式アルゴリズム — 入力が低ランクの標本構造ではなく疎アクセスのオラクルである場合 — には既知の古典的対応物がなく、脱量子化の技法はそこに届きません。疎で条件のよい系が持たない低ランク構造を必要とするからです。したがって正しい要約は、特定の*行列クラス*について特定の*入力モデル*が対等化されたということであり、量子線形代数が反証されたということではありません。

日常的な用法では「脱量子化された」はもっと弱く、もっと実用的な意味を持ちます。*誰かが同じくらいよく働く古典モデルを作った*、ということです。それは定理ではなく実験であり、本章がやることです。研究の1年をどこに使うかの判断にとって重要なのは弱い方の主張です。古典的な構成が証明可能に一般的である必要はなく、目の前の問題で動けばよいからです。

どちらの意味も1つの教訓を共有しており、コードの前に述べておく価値があります。量子モデルの優位性が量子モデル単独の性質であることは決してありません。それは*差*であり、差には2つの側があります。この分野で報告された差の大半は、古典側から埋められてきました。量子的な構成が暗黙に記述していた古典モデルを、誰かが書き下したのです。

### Code Example 1: 量子カーネル、共通手順、そしてベースライン

本章のすべてはNumPyだけの1つのPythonセッションで動きます。シミュレータは姉妹コース[量子コンピューティング入門](<../../FM/quantum-computing-introduction/index.html>)第2章のもののうち本章が必要とする関数を逐語再掲し（`sample()` は測定結果を1つも引かないので省いています）、データセットは第1章のものをシードごと再現し、カーネルは第3章の忠実度カーネル $k(\mathbf{x},\mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2$ です。手順 — すべてのハイパーパラメータを訓練データ上の5分割交差検証で選び、テストデータは最後に1度だけ見る — は本章のすべてのモデルに例外なく適用します。

第3章から意図的に1点変えています。ここでの特徴写像は、第3章のZZ写像ではなく、再アップロード2層とCNOTリングを備えた素の角度符号化です。後で都合のよい選択に見えないよう、理由を先に述べます。第3章の写像は各2量子ビット回転に*対*の角度 $(\pi - \pi x_j)(\pi - \pi x_k)$ を入れており、これは記述子について2次なので、得られるカーネルは $\mathbf{x}$ について帯域制限されず、周波数集合を列挙できません。ここで使う写像は帯域制限されており、スペクトルは有限で測定可能です。5.2節で2つのサロゲートの一方を作るにはそれが必要です。もう一方 — Pauliサロゲート — は第3章の写像を含むどんな特徴写像にも使えます。

```python
"""第5章の準備を1ブロックで: 本章が必要とするミニシミュレータの関数を逐語再掲
（sample() は無し。ここでは測定結果を1つも引かない）、データセット、
第3章の量子カーネル、そして閉形式のカーネルリッジ回帰。
NumPyのみを使い、前章の結果は一切前提としません。
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

# ---- 第3章の量子カーネル -------------------------------------------------
K_LAYERS = 2                          # 特徴写像の data re-uploading 深さ


def feature_state(x, layers=K_LAYERS):
    """|phi(x)>: 角度符号化を layers 回再アップロードし、CNOTリングを挟む。"""
    n = len(x)
    psi = ket('0' * n)
    for _ in range(layers):
        for q in range(n):
            psi = apply_gate(psi, ry(np.pi * x[q]), [q], n)
        for q in range(n):
            psi = cnot(psi, q, (q + 1) % n, n)
    return psi


def feature_matrix(Xs, layers=K_LAYERS):
    return np.array([feature_state(x, layers) for x in Xs])


def quantum_kernel(XA, XB, layers=K_LAYERS):
    """k(x, x') = |<phi(x)|phi(x')>|^2、第3章の忠実度カーネル。"""
    SA, SB = feature_matrix(XA, layers), feature_matrix(XB, layers)
    return np.abs(SA.conj() @ SB.T) ** 2


def rbf_kernel(XA, XB, gamma):
    d2 = ((XA[:, None, :] - XB[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2)


# ---- カーネルリッジ回帰の閉形式解 ----------------------------------------
def krr_fit(K, y, lam):
    """alpha = (K + lam I)^{-1} (y - 平均)。平均は別に扱う。"""
    mu = y.mean()
    return np.linalg.solve(K + lam * np.eye(len(y)), y - mu), mu


def krr_predict(Kcross, alpha, mu):
    return Kcross @ alpha + mu


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def cv_select(kernel_fn, X, y, folds=5, lambdas=LAMBDAS):
    """*訓練データのみ*でk分割交差検証によりlamを選ぶ。戻り値は (lam, CV MSE)。"""
    K = kernel_fn(X, X)
    idx = np.arange(len(y))
    cut = np.array_split(idx, folds)
    best = (None, np.inf)
    for lam in lambdas:
        errs = []
        for f in range(folds):
            va = cut[f]
            tr = np.setdiff1d(idx, va)
            a, mu = krr_fit(K[np.ix_(tr, tr)], y[tr], lam)
            errs.append(mse(krr_predict(K[np.ix_(va, tr)], a, mu), y[va]))
        m = float(np.mean(errs))
        if m < best[1]:
            best = (lam, m)
    return best


print("60点の材料データに対する量子カーネル")
print("-" * 76)
Kqq = quantum_kernel(Xall, Xall)
off = Kqq[~np.eye(len(Kqq), dtype=bool)]
print(f"  特徴写像              再アップロード {K_LAYERS} 層、4量子ビット、"
      f"状態次元 {2**4}")
print(f"  対角成分              min {np.diag(Kqq).min():.6f}"
      f"  max {np.diag(Kqq).max():.6f}   （厳密に1でなければならない）")
print(f"  非対角の平均          {off.mean():.6f}")
print(f"  非対角の標準偏差      {off.std():.6f}")
print(f"  非対角の範囲          [{off.min():.6f}, {off.max():.6f}]")
ev = np.linalg.eigvalsh(Kqq)[::-1]
print(f"  固有値（上位6個）     " + " ".join(f"{v:.4f}" for v in ev[:6]))
print(f"  実効ランク            {ev.sum()**2 / (ev**2).sum():.2f} / {len(ev)}")

print("\n本章の全モデルに課す共通手順")
print("-" * 88)
GAMMAS = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 10.0)
lam_q, cvq = cv_select(lambda a, b: quantum_kernel(a, b), Xtr, ytr)
Kt = quantum_kernel(Xtr, Xtr)
aq, muq = krr_fit(Kt, ytr, lam_q)
mse_q = mse(krr_predict(quantum_kernel(Xte, Xtr), aq, muq), yte)


def rbf_run(g):
    """バンド幅1つについて (lam, CV MSE, test MSE)。lamはCVで選ぶ。"""
    lam, cv = cv_select(lambda a, b, g=g: rbf_kernel(a, b, g), Xtr, ytr)
    a, mu = krr_fit(rbf_kernel(Xtr, Xtr, g), ytr, lam)
    return lam, cv, mse(krr_predict(rbf_kernel(Xte, Xtr, g), a, mu), yte)


rbf_table = {g: rbf_run(g) for g in GAMMAS}
g_star = min(GAMMAS, key=lambda g: rbf_table[g][1])          # 正当な選択
lam_r, cvr, mse_r = rbf_table[g_star]
g_orc = min(GAMMAS, key=lambda g: rbf_table[g][2])           # 上界（選択ではない）
lam_o, cvo, mse_orc = rbf_table[g_orc]
A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
coef = np.linalg.lstsq(A, ytr, rcond=None)[0]
mse_ols = mse(np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef, yte)
mse_mean = mse(np.full(len(yte), ytr.mean()), yte)
print(f"  {'model':<40}{'selected by 5-fold CV':>24}{'CV MSE':>9}"
      f"{'test MSE':>10}{'test RMSE':>11}")
rows = [("quantum fidelity kernel ridge", f"lam={lam_q:g}", cvq, mse_q),
        ("RBF kernel ridge", f"gamma={g_star:g}, lam={lam_r:g}", cvr, mse_r),
        ("RBF kernel ridge, best gamma on test", f"gamma={g_orc:g} (not a selection)",
         cvo, mse_orc),
        ("ordinary least squares", "-", None, mse_ols),
        ("predict the training mean", "-", None, mse_mean),
        ("irreducible noise floor", "-", None, 0.05 ** 2)]
for nm, sel, cv, te in rows:
    c = f"{cv:9.4f}" if cv is not None else f"{'-':>9}"
    print(f"  {nm:<40}{sel:>24}{c}{te:10.4f}{te**0.5:11.4f}")
print("  3行目は誰も選んではいけないモデルであり、この分割で古典側が到達できる上界を")
print("  示すものである。以下のすべての結論は、正当な2行目に加えてこの上界にも照らして")
print("  検査する。")
print("  RMSEの列は、第1章と第3章のアンカー（線形リッジ 0.2146、最良の古典RBF 0.1453）と")
print("  直接読み比べられるようにするためにある。")
```

```text
60点の材料データに対する量子カーネル
----------------------------------------------------------------------------
  特徴写像              再アップロード 2 層、4量子ビット、状態次元 16
  対角成分              min 1.000000  max 1.000000   （厳密に1でなければならない）
  非対角の平均          0.153000
  非対角の標準偏差      0.191101
  非対角の範囲          [0.000000, 0.977952]
  固有値（上位6個）     11.8973 5.7725 5.0626 3.7772 3.4988 3.0388
  実効ランク            13.23 / 60

本章の全モデルに課す共通手順
----------------------------------------------------------------------------------------
  model                                      selected by 5-fold CV   CV MSE  test MSE  test RMSE
  quantum fidelity kernel ridge                           lam=0.01   0.0722    0.0680     0.2607
  RBF kernel ridge                            gamma=0.3, lam=1e-06   0.0232    0.0422     0.2055
  RBF kernel ridge, best gamma on test    gamma=2 (not a selection)   0.0308    0.0208     0.1443
  ordinary least squares                                         -        -    0.0465     0.2157
  predict the training mean                                      -        -    0.2747     0.5242
  irreducible noise floor                                        -        -    0.0025     0.0500
  3行目は誰も選んではいけないモデルであり、この分割で古典側が到達できる上界を
  示すものである。以下のすべての結論は、正当な2行目に加えてこの上界にも照らして
  検査する。
  RMSEの列は、第1章と第3章のアンカー（線形リッジ 0.2146、最良の古典RBF 0.1453）と
  直接読み比べられるようにするためにある。
```

**注目すべき点。** 4つの読みどころが本章全体を設定します。

**実効ランクは60のうち13.23** です。量子特徴空間は $2^4 = 16$ 次元なので、60点のGram行列は原理上ランク16まで取れますが、参加率はスペクトルが約13方向に支配されていると言っています。それは小さく、古典的な対象です。このカーネルが何をしているにせよ、書き下せないほど大きな空間を活用してはいません。

**量子カーネルはRBFカーネルに負けます。** 0.0680 対 0.0422、両者を同じ分割の同じ交差検証で調整した手順の下でです。5パラメータの線形回帰 0.0465 にも負け、この分割で使える最強のRBF 0.0208 には3.3倍の差で負けます。最後の数値は立ち止まる価値があります。RMSEにすると 0.1443 で、第3章の最良の古典アンカー 0.1453 をほぼそのまま再現しており、最小二乗の行の 0.2157 は第1章の線形リッジのアンカー 0.2146 を再現しています。本章の古典側は、本コースの他の章が測ったのと同じ古典側です。

**交差検証誤差とテスト誤差は食い違います。** しかも大きさだけの問題ではありません。40点での5分割CVは $\gamma = 0.3$ を選びますが、テストデータが好むのは $\gamma = 2$ です。3行目が存在する理由がこれです。量子カーネルのCV誤差はCVが選んだRBFの 0.0232 に対して 0.0722 で3倍ですが、その対のテストの差は1.6倍です。訓練40点・テスト20点では、どちらの数値も有効数字3桁の意味を持ちません。それ自身が5.3節の発見の1つであり、あの節が1つではなく20の分割を走らせる理由です。

### 特徴空間には宣伝ほどのものが入っていない

量子カーネルを動機づける直観は、$n$ 量子ビット上の状態が $2^n$ 次元空間に住むので、その空間への符号化は指数的に豊かな特徴写像を与えるというものです。この直観は正しく、そして無関係です。理由は2行の計算です。

$d$ 次元空間の一様（Haar）測度から独立に取った2つの状態 $|\phi\rangle, |\phi'\rangle$ を考えます。すると $|\langle\phi|\phi'\rangle|^2$ の平均は $1/d$、標準偏差も $1/d$ のオーダーです。$d = 2^n$ が大きくなるにつれ、異なる2状態の忠実度は指数的に0へ集中します。そうした状態の上の忠実度カーネル行列は、単位行列に消えかけの非対角が付いたものになります。そして単位行列上のカーネルリッジ回帰は、訓練データを記憶して他ではすべて平均を返す引き当て表です。

実際の符号化はHaarランダムではないので集中はもっと遅いですが、それでも存在し、それでも幾何級数的であり、測定できます。

### Code Example 2: 指数的集中を測る

```python
"""指数的集中: レジスタを広げるとカーネルはどうなるか。
Code Example 1 の続き（同一セッション）。
"""


def widen(Xs, n):
    """n量子ビットのレジスタを4個の記述子で埋める。必要なら繰り返す。"""
    reps = int(np.ceil(n / Xs.shape[1]))
    return np.tile(Xs, (1, reps))[:, :n]


print("レジスタ幅に対するカーネルの統計量（データは同じ60点）")
print("-" * 92)
print(f"  {'n':>3}{'dim':>7}{'off-diag mean':>15}{'off-diag std':>14}"
      f"{'max off-diag':>14}{'eff. rank':>11}{'test MSE':>10}")
NS = list(range(2, 13))
stats = {}
for n in NS:
    Xw = widen(Xall, n)
    Kw = quantum_kernel(Xw, Xw)
    off = Kw[~np.eye(len(Kw), dtype=bool)]
    ev = np.linalg.eigvalsh(Kw)[::-1]
    er = ev.sum() ** 2 / (ev ** 2).sum()
    kf = lambda a, b, nn=n: quantum_kernel(a, b)
    lam, _ = cv_select(kf, Xw[:40], ytr)
    a, mu = krr_fit(Kw[:40, :40], ytr, lam)
    te = mse(krr_predict(Kw[40:, :40], a, mu), yte)
    stats[n] = (off.mean(), off.std(), off.max(), er, te)
    print(f"  {n:>3}{2**n:>7}{off.mean():15.6f}{off.std():14.6f}"
          f"{off.max():14.6f}{er:11.2f}{te:10.4f}")

print("\n減衰は幾何級数的であり、2^-n が基準線になる")
print("-" * 92)
print(f"  {'n':>3}{'mean':>13}{'2^-n':>13}{'mean/2^-n':>12}"
      f"{'std':>13}{'std ratio n-1 -> n':>21}")
prev = None
for n in NS:
    m, s, _, _, _ = stats[n]
    r = "-" if prev is None else f"{s/prev:.3f}"
    print(f"  {n:>3}{m:13.6f}{2.0**-n:13.6f}{m/2.0**-n:12.3f}{s:13.6f}{r:>21}")
    prev = s

print("\n集中がショット数として何を要求するか")
print("-" * 92)
print("  忠実度カーネルの各成分は、inversion test での全ゼロ列の計数という")
print("  Bernoulli試行から推定する。したがって1ショットの分散は 1 ではなく k(1-k)")
print("  であり、S = k(1-k)/eps^2 となる。広がりの10分の1で構造を分解するには")
print("  eps = sigma/10 とすればよく（第3章の規約）、S = 100 k(1-k)/sigma^2 となる。")
print("  N点のGram行列には N(N-1)/2 成分ある。")
print(f"  {'n':>3}{'mean k':>12}{'std of entries':>16}{'shots per entry':>18}"
      f"{'shots for 60x60 Gram':>22}")
for n in NS:
    m, s = stats[n][0], stats[n][1]
    per = 100.0 * m * (1.0 - m) / s ** 2
    print(f"  {n:>3}{m:12.6f}{s:16.6f}{per:18,.0f}{per * 60 * 59 / 2:22,.0f}")
print("  この列はほぼ横ばいであり、それは忠実度カーネル一般の性質ではなくこの符号化")
print("  の性質である。ここでは k が sigma より速く落ちるので、分子が分母と一緒に")
print("  縮む。代わりに 1/sigma^2 を引用すれば n = 12 で190ショットとなり、外挿すると")
print("  指数的になる。写像が k ~ sigma ~ 2^-n となるほど表現力を持ったときコストが")
print("  実際にどうなるかは、演習1が示す。")

print("\n正直な読み方")
print("-" * 92)
print("  ここにハードウェアの制約は1つも入っていない。シミュレータは厳密かつ無雑音")
print("  である。2^n 次元空間にある2状態の忠実度は典型的に 2^-n の大きさなのでカーネル")
print("  は平坦化し、単位行列に消えかけの非対角が付いただけの行列はデータについて何の")
print("  情報も持たない。その上でカーネルリッジ回帰をしても訓練データの平均しか再現")
print("  できない。")
print("  test MSE の列の向きに注意すること。レジスタを広げると良くなるどころか悪くなる。")
print("  量子カーネルを動機づけた「指数的に大きな特徴空間」という直観の逆である。")
```

```text
レジスタ幅に対するカーネルの統計量（データは同じ60点）
--------------------------------------------------------------------------------------------
    n    dim  off-diag mean  off-diag std  max off-diag  eff. rank  test MSE
    2      4       0.359838      0.309325      0.998156       4.20    0.0362
    3      8       0.246025      0.262934      0.986398       6.94    0.0284
    4     16       0.153000      0.191101      0.977952      13.23    0.0680
    5     32       0.109725      0.167800      0.974733      17.80    0.0628
    6     64       0.077986      0.137245      0.880540      24.29    0.0831
    7    128       0.056518      0.119306      0.867474      29.58    0.0838
    8    256       0.044784      0.105990      0.857991      33.69    0.1064
    9    512       0.036139      0.094839      0.856278      37.32    0.1235
   10   1024       0.029538      0.085575      0.773427      40.44    0.1243
   11   2048       0.024784      0.078636      0.767840      42.82    0.1301
   12   4096       0.021315      0.072502      0.759562      44.88    0.1431

減衰は幾何級数的であり、2^-n が基準線になる
--------------------------------------------------------------------------------------------
    n         mean         2^-n   mean/2^-n          std   std ratio n-1 -> n
    2     0.359838     0.250000       1.439     0.309325                    -
    3     0.246025     0.125000       1.968     0.262934                0.850
    4     0.153000     0.062500       2.448     0.191101                0.727
    5     0.109725     0.031250       3.511     0.167800                0.878
    6     0.077986     0.015625       4.991     0.137245                0.818
    7     0.056518     0.007812       7.234     0.119306                0.869
    8     0.044784     0.003906      11.465     0.105990                0.888
    9     0.036139     0.001953      18.503     0.094839                0.895
   10     0.029538     0.000977      30.246     0.085575                0.902
   11     0.024784     0.000488      50.757     0.078636                0.919
   12     0.021315     0.000244      87.308     0.072502                0.922

集中がショット数として何を要求するか
--------------------------------------------------------------------------------------------
  忠実度カーネルの各成分は、inversion test での全ゼロ列の計数という
  Bernoulli試行から推定する。したがって1ショットの分散は 1 ではなく k(1-k)
  であり、S = k(1-k)/eps^2 となる。広がりの10分の1で構造を分解するには
  eps = sigma/10 とすればよく（第3章の規約）、S = 100 k(1-k)/sigma^2 となる。
  N点のGram行列には N(N-1)/2 成分ある。
    n      mean k  std of entries   shots per entry  shots for 60x60 Gram
    2    0.359838        0.309325               241               426,127
    3    0.246025        0.262934               268               474,916
    4    0.153000        0.191101               355               628,088
    5    0.109725        0.167800               347               614,066
    6    0.077986        0.137245               382               675,667
    7    0.056518        0.119306               375               663,082
    8    0.044784        0.105990               381               674,010
    9    0.036139        0.094839               387               685,470
   10    0.029538        0.085575               391               692,844
   11    0.024784        0.078636               391               691,819
   12    0.021315        0.072502               397               702,448
  この列はほぼ横ばいであり、それは忠実度カーネル一般の性質ではなくこの符号化
  の性質である。ここでは k が sigma より速く落ちるので、分子が分母と一緒に
  縮む。代わりに 1/sigma^2 を引用すれば n = 12 で190ショットとなり、外挿すると
  指数的になる。写像が k ~ sigma ~ 2^-n となるほど表現力を持ったときコストが
  実際にどうなるかは、演習1が示す。

正直な読み方
--------------------------------------------------------------------------------------------
  ここにハードウェアの制約は1つも入っていない。シミュレータは厳密かつ無雑音
  である。2^n 次元空間にある2状態の忠実度は典型的に 2^-n の大きさなのでカーネル
  は平坦化し、単位行列に消えかけの非対角が付いただけの行列はデータについて何の
  情報も持たない。その上でカーネルリッジ回帰をしても訓練データの平均しか再現
  できない。
  test MSE の列の向きに注意すること。レジスタを広げると良くなるどころか悪くなる。
  量子カーネルを動機づけた「指数的に大きな特徴空間」という直観の逆である。
```

**注目すべき点。** 非対角の平均は2量子ビットの 0.360 から12量子ビットの 0.021 へ、標準偏差は 0.309 から 0.073 へ落ちます。1量子ビットあたり 0.73〜0.92 倍で、後半は0.9に近づきます。2層の符号化はランダム回路から程遠いのでHaarの予測 0.5 より遅いものの、やはり幾何級数的です。演習1がHaarとの比較を扱います。

重要なのは最後の列です。**レジスタを広げるとテスト誤差は悪化します。** 2量子ビットで 0.036、12量子ビットで 0.143。これは指数的特徴空間の直観が予測することの逆であり、微妙な効果でもありません。厳密なシミュレータ上で、各幅でハイパーパラメータを再調整して、10量子ビットにわたって4倍です。量子ビットを足すことは、有用な意味でのモデル容量を足すことではありません。カーネルがデータ点の区別をつけられなくなるまで、すべての対の間に距離を足しているのです。

ショットの列はこれを実験コストに換算しますが、正しく計算しなければ真実の逆を語ってしまいます。忠実度カーネルの成分は inversion test で全ゼロが出る確率として推定されるので、1ショットは分散 $k(1-k)$ の*ビット*であり、分散1の有界な観測量ではありません。したがって広がりの10分の1で構造を分解するコストは成分あたり $S = 100\,k(1-k)/\sigma^2$ ショットであり、この符号化ではその列は $n$ についてほぼ横ばいです。2量子ビットで約240、12量子ビットで約400です。$k$ が $\sigma$ より速く落ち、分子が分母と一緒に縮むからです。60点のGram行列には1770成分あるので、12量子ビットで約 $7\times10^5$ ショット — そしてこの数値はレジスタを広げてもほとんど増えません。

素朴な $1/\sigma^2$ なら12量子ビットで190ショットと読め、幾何級数的に増えたはずです。より劇的な物語ですが、誤っています。指数的なショットコストが本当に住んでいるのは、この符号化が決して到達しない領域です。特徴写像が $k$ と $\sigma$ の両方が $2^{-n}$ で落ちるほど表現力を持てば、比 $k(1-k)/\sigma^2$ 自身が $2^{n}$ のオーダーになり、*そのとき*30量子ビットの忠実度カーネルは誤り率とはまったく無関係な理由で測定不能になります。演習1がHaar予測に対してその算術を行い、第3章の3.6節がその領域にある写像で指数を測ります。*この*表の誠実な要約はふつう言われるものより狭いのです。レジスタ幅は、測定予算を壊すよりずっと前にモデルの精度を壊します。

* * *

## 5.2 2つのサロゲートを作って測る

古典サロゲートとは、特定の量子モデルを模倣するために作った古典モデルです。一般的なレシピはありませんが、本コースで使うほとんどの形のカーネルに適用できる構成が2つあり、それらは反対の方向から攻めます。

**Pauli（古典シャドウ）サロゲート**は、忠実度カーネルが密度行列の内積であることを利用します。$\rho(\mathbf{x}) = |\phi(\mathbf{x})\rangle\langle\phi(\mathbf{x})|$ をPauli基底で書くと

$$
\rho(\mathbf{x}) = \frac{1}{2^n}\sum_{P} c_P(\mathbf{x})\,P,\qquad c_P(\mathbf{x}) = \mathrm{Tr}\bigl[\rho(\mathbf{x})P\bigr] = \langle\phi(\mathbf{x})|P|\phi(\mathbf{x})\rangle
$$

したがって

$$
k(\mathbf{x},\mathbf{x}') = \bigl|\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle\bigr|^2 = \mathrm{Tr}\bigl[\rho(\mathbf{x})\rho(\mathbf{x}')\bigr] = \frac{1}{2^n}\sum_P c_P(\mathbf{x})\,c_P(\mathbf{x}')
$$

となります。和は $4^n$ 項で、これが指数的コストです。Pauli weight $w$ で打ち切り — 非単位因子が $w$ 個以下の文字列だけを残す — と、特徴ベクトルの成分数は $\sum_{k\le w}\binom{n}{k}3^k$、すなわち $O(n^w)$ です。残る各成分は低weight Pauli演算子の期待値であり、それはまさに*古典シャドウ*が $n$ に依らないショット数でランダム化測定から推定する量です。したがってこのサロゲートは数学的な抽象ではなく、測定プロトコルの記述であり、特徴が手に入ればモデルは古典的なリッジ回帰です。

**Fourierサロゲート**は代わりに符号化の構造を利用します。第2章が確立した規則はこうです。生成子がPauliの2分の1である符号化ゲートが $L$ 個あれば（第2章は $Z/2$、ここは $Y/2$ で、効くのは固有値間隔だけです）、回転角についての周波数集合は $\Omega = \lbrace -L,\ldots,L\rbrace$ になります。したがって期待値は記述子についての*有限*個の周波数を持つ三角多項式であり、ここでは角度が $\pi x_q$ なので記述子の単位あたり $1/2$ サイクルの倍数になります。その集合が列挙できるなら、まさにその基底関数を持つ古典モデルは量子モデルと同じ関数空間を張り、しかも量子デバイスに一度も触れません。訓練でも推論でも、特徴を作るためにも触れません。

最後の一文には「列挙できる」の上限が必要で、その上限はPauliサロゲートが抱えるものと同じです。2層では各記述子が帯域制限された関数を5つ寄与します — $1$、$\cos\pi x_q$、$\sin\pi x_q$、$\cos 2\pi x_q$、$\sin 2\pi x_q$ — なので、$d = 4$ 記述子の*完全な*帯域制限積基底は $5^4 = 625$ 個、一般には $5^d$ 個です。量子ビット数について $4^n$ が指数的なのと同じで、記述子数について指数的です。以下で作るサロゲートはその625個のうち41個、すなわち*総次数が2以下*の積だけを保ちます。この打ち切りは無料でも中立でもありません。**目的関数に記述子間の高次相互作用がないという賭け**であり、weight $w$ のPauli打ち切りが高weight相関について行うのと同じ賭けです。このターゲットではその賭けが当たり、Code Example 4 がどれだけ当たるかを示します。4体相互作用から作ったターゲットでは外れます。この手法の誠実な言い方は「帯域制限基底を列挙し、相互作用の次数で打ち切る」であって、「帯域制限基底を列挙する」ではありません。

### Code Example 3: Pauliサロゲート

```python
"""カーネル自身の構造から作る古典サロゲート: 低weightのPauli。
Code Example 1 の続き（同一セッション）。
"""
from itertools import product


def pauli_strings(n, max_weight):
    """n量子ビット上で非単位因子が max_weight 個以下のPauli文字列すべて。"""
    out = []
    for letters in product('IXYZ', repeat=n):
        if sum(c != 'I' for c in letters) <= max_weight:
            out.append(''.join(letters))
    return out


def shadow_features(Xs, max_weight, n=4, layers=K_LAYERS):
    """低weightの各Pauli P について Phi(x)_P = <phi(x)| P |phi(x)> / 2^(n/2)。

    4^n 個すべてのPauliを使えば忠実度カーネルを厳密に再現する。
    rho = 2^-n sum_P <P> P であり k(x,x') = Tr[rho(x) rho(x')] だからである。
    和を weight w で打ち切ると、大きさ O(n^w) の古典的な特徴写像が残る。しかも
    各特徴は古典シャドウが推定する量そのものである。
    """
    ps = pauli_strings(n, max_weight)
    S = feature_matrix(Xs, layers)
    Phi = np.empty((len(Xs), len(ps)))
    for i, psi in enumerate(S):
        Phi[i] = [expval(psi, p) for p in ps]
    return Phi / np.sqrt(2.0 ** n), ps


Kq_all = quantum_kernel(Xall, Xall)
print("量子カーネルの厳密なPauli展開を打ち切る（4量子ビット）")
print("-" * 90)
off_mask = ~np.eye(len(Kq_all), dtype=bool)
print(f"  {'max weight':>11}{'features':>10}{'rel. error (all)':>18}"
      f"{'rel. error (off-diag)':>23}{'alignment':>12}")
surrogates = {}
for w in range(0, 5):
    Phi, ps = shadow_features(Xall, w)
    Kw = Phi @ Phi.T
    err = np.linalg.norm(Kq_all - Kw) / np.linalg.norm(Kq_all)
    erro = (np.linalg.norm(Kq_all[off_mask] - Kw[off_mask])
            / np.linalg.norm(Kq_all[off_mask]))
    align = (np.sum(Kq_all * Kw)
             / (np.linalg.norm(Kq_all) * np.linalg.norm(Kw)))
    surrogates[w] = Phi
    print(f"  {w:>11}{len(ps):>10}{err:18.6f}{erro:23.6f}{align:12.6f}")
print("  weight 4 は展開全体なので最終行は恒等性の検査になる。サロゲートと量子カーネル")
print("  は機械精度で一致する。全成分の列は対角成分に支配されており、打ち切りは対角を")
print("  必ず1より小さくする。データを担っているのは非対角の列である。")

print("\n各打ち切りが実際の課題でどれだけ働くか")
print("-" * 90)


def ridge_features(Phi, ytr_, lam):
    """明示的な特徴写像に対する主形式リッジ: w = (Phi'Phi + lam I)^-1 Phi'(y-mu)。"""
    mu = ytr_.mean()
    G = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    return np.linalg.solve(G, Phi.T @ (ytr_ - mu)), mu


def cv_features(Phi, y, folds=5, lambdas=LAMBDAS):
    idx = np.arange(len(y))
    cut = np.array_split(idx, folds)
    best = (None, np.inf)
    for lam in lambdas:
        errs = []
        for f in range(folds):
            va = cut[f]
            tr = np.setdiff1d(idx, va)
            wv, mu = ridge_features(Phi[tr], y[tr], lam)
            errs.append(mse(Phi[va] @ wv + mu, y[va]))
        m = float(np.mean(errs))
        if m < best[1]:
            best = (lam, m)
    return best


print(f"  {'model':<38}{'features':>10}{'lam':>9}{'CV MSE':>10}{'test MSE':>11}")
for w in range(0, 5):
    Phi = surrogates[w]
    lam, cv = cv_features(Phi[:40], ytr)
    wv, mu = ridge_features(Phi[:40], ytr, lam)
    te = mse(Phi[40:] @ wv + mu, yte)
    print(f"  {'Pauli surrogate, weight <= ' + str(w):<38}"
          f"{Phi.shape[1]:>10}{lam:9g}{cv:10.4f}{te:11.4f}")
print(f"  {'quantum fidelity kernel ridge':<38}{'2^4 dim':>10}{lam_q:9g}"
      f"{cvq:10.4f}{mse_q:11.4f}")
print(f"  {'RBF kernel ridge (tuned)':<38}{'-':>10}{lam_r:9g}{cvr:10.4f}"
      f"{mse_r:11.4f}")

print("\n脱量子化の論点を一段落で")
print("-" * 90)
print("  weight <= 2 に必要な特徴数は 1 + 3n + 9 n(n-1)/2 で、4^n ではなく O(n^2) で")
print("  ある。4量子ビットならデータ点あたり67個の数になる。Gram行列を精密に再現する")
print("  わけではなく、非対角の相対誤差は数十パーセント残る。それでもテスト誤差は完全な")
print("  量子カーネルに既に並び、weight 3 では上回る。予測性能はカーネル忠実度と同じ量")
print("  ではなく、優位性の主張が問題にしているのは予測性能の方である。")
print("  67個の数はいずれも低weight Pauliの期待値であり、それはまさに古典シャドウが")
print("  n に依らないショット数で測定する量である。したがって特徴が手に入れば、モデル")
print("  の構築・訓練・運用は古典的に行える。そして特徴そのものが古典的に計算できるなら")
print("  量子計算機は登場すらしない。")
print("  実務上の「脱量子化」とはこれである。量子モデルが無用だという証明ではなく、")
print("  ある特定の量子モデルの優位性が、その古典シャドウを書き下した時点で消えることの")
print("  実証である。")
```

```text
量子カーネルの厳密なPauli展開を打ち切る（4量子ビット）
------------------------------------------------------------------------------------------
   max weight  features  rel. error (all)  rel. error (off-diag)   alignment
            0         1          0.880535               0.863743    0.607811
            1        13          0.834832               0.819586    0.689051
            2        67          0.673329               0.666768    0.822302
            3       175          0.311714               0.318940    0.968583
            4       256          0.000000               0.000000    1.000000
  weight 4 は展開全体なので最終行は恒等性の検査になる。サロゲートと量子カーネル
  は機械精度で一致する。全成分の列は対角成分に支配されており、打ち切りは対角を
  必ず1より小さくする。データを担っているのは非対角の列である。

各打ち切りが実際の課題でどれだけ働くか
------------------------------------------------------------------------------------------
  model                                   features      lam    CV MSE   test MSE
  Pauli surrogate, weight <= 0                   1    1e-06    0.3167     0.2747
  Pauli surrogate, weight <= 1                  13      0.1    0.2089     0.2655
  Pauli surrogate, weight <= 2                  67      0.1    0.1058     0.0642
  Pauli surrogate, weight <= 3                 175     0.01    0.0768     0.0379
  Pauli surrogate, weight <= 4                 256     0.01    0.0722     0.0680
  quantum fidelity kernel ridge            2^4 dim     0.01    0.0722     0.0680
  RBF kernel ridge (tuned)                       -    1e-06    0.0232     0.0422

脱量子化の論点を一段落で
------------------------------------------------------------------------------------------
  weight <= 2 に必要な特徴数は 1 + 3n + 9 n(n-1)/2 で、4^n ではなく O(n^2) で
  ある。4量子ビットならデータ点あたり67個の数になる。Gram行列を精密に再現する
  わけではなく、非対角の相対誤差は数十パーセント残る。それでもテスト誤差は完全な
  量子カーネルに既に並び、weight 3 では上回る。予測性能はカーネル忠実度と同じ量
  ではなく、優位性の主張が問題にしているのは予測性能の方である。
  67個の数はいずれも低weight Pauliの期待値であり、それはまさに古典シャドウが
  n に依らないショット数で測定する量である。したがって特徴が手に入れば、モデル
  の構築・訓練・運用は古典的に行える。そして特徴そのものが古典的に計算できるなら
  量子計算機は登場すらしない。
  実務上の「脱量子化」とはこれである。量子モデルが無用だという証明ではなく、
  ある特定の量子モデルの優位性が、その古典シャドウを書き下した時点で消えることの
  実証である。
```

**注目すべき点。** weight 4 の行は恒等性の検査です。256個すべてのPauliを残せばカーネルは機械精度で再現され、量子カーネルとちょうど同じテスト誤差 0.0680 を与えます。展開が正しいことの確認です。

その上の行は、混同しやすい2つのものを分離しています。**カーネル忠実度と予測性能は別の量です。** weight 2 のサロゲートはGram行列の再現は下手で、非対角成分の相対誤差は67%、alignmentは 0.822 です。それでもテスト誤差 0.0642 は完全な量子カーネルの 0.0680 より既にわずかに*良い*のです。weight 3 のサロゲートは 0.0379 に達し、余裕をもって上回ります。ここでの打ち切りはコスト削減であるだけでなく正則化でもあり、忠実度カーネルが40点では支えきれない重みを置いている高weight方向を除いています。

優位性の主張にとって関係するのは予測の列だけです。「われわれのカーネルはFrobeniusノルムで $\epsilon$ 以内に古典近似できない」という形の議論は、古典モデルが同じくらいよく予測するかどうかについて何も言っておらず、両者は単調に関係していません。

特徴数が議論のもう半分です。4量子ビットで67個、$n$ 量子ビットで $1 + 3n + 9n(n-1)/2$ なので、$n = 50$ では 11,176 個。$4^{50} \approx 1.3\times10^{30}$ に対してです。演習2が表にします。

### Code Example 4: 量子計算機を必要としないFourierサロゲート

```python
"""量子計算機に一度も触れないサロゲート: 打ち切りFourier特徴。
Code Example 1 と 3 の続き（同一セッション）。
"""

print("まず、量子カーネルが実際に含んでいる周波数を確かめる")
print("-" * 90)
xref = np.array([0.31, 0.62, 0.17, 0.85])
grid = np.arange(64) / 64.0 * 2.0                # x1 の2周期分（周期は2）
Kline = np.array([quantum_kernel(np.array([[t, xref[1], xref[2], xref[3]]]),
                                 xref[None, :])[0, 0] for t in grid])
amp = np.abs(np.fft.rfft(Kline)) / len(grid)
print(f"  k(x, x_ref) を x1 のみの関数として x1 in [0,2) の64点で標本化")
print(f"  {'harmonic (cycles per unit x1)':<34}{'amplitude':>12}")
for j in range(0, 7):
    print(f"  {j/2.0:<34.1f}{amp[j]:12.6f}")
print(f"  高調波 1.0 より上の全パワー: {np.sum(amp[3:]**2)**0.5:.3e}")
print("  スペクトルは厳密に打ち切られている。Ry(pi x) の再アップロード2層は x1 について")
print("  0、1/2、1 cycles per unit の高調波のみを生み、それ以上を生まない。したがって")
print("  同じ基底関数を持つ古典モデルは同じ関数空間を張る。第2章の周波数の議論を、")
print("  そのまま攻撃手段として使うことになる。")
print("  記述子あたり5つの関数（1, cos, sin, cos2, sin2）なので、*完全な*帯域制限積基底は")
print(f"  5^4 = {5**4} 個、一般には 5^d 個ある。以下のサロゲートが保つのは総次数 2 以下の")
print("  41個だけであり、これは目的関数に高次相互作用がないという賭けである。タダ飯ではない。")


def fourier_features(Xs, dmax, n=4):
    """総次数 dmax 以下の cos/sin 基底。周波数は j * pi * x_q。"""
    cols = [np.ones(len(Xs))]
    names = ['1']

    def rec(q, degs):
        if q == n:
            if sum(degs) == 0:
                return
            for signs in product((0, 1), repeat=n):
                if any(s and d == 0 for s, d in zip(signs, degs)):
                    continue
                col = np.ones(len(Xs))
                nm = []
                for qq, (d, s) in enumerate(zip(degs, signs)):
                    if d == 0:
                        continue
                    ang = d * np.pi * Xs[:, qq]
                    col = col * (np.sin(ang) if s else np.cos(ang))
                    nm.append(('sin' if s else 'cos') + f'({d}pi x{qq+1})')
                cols.append(col)
                names.append('*'.join(nm))
            return
        for d in range(0, dmax + 1):
            if sum(degs) + d <= dmax:
                rec(q + 1, degs + [d])

    rec(0, [])
    return np.column_stack(cols), names


print("\nサロゲートと、それが模倣している当のモデル")
print("-" * 90)
print(f"  {'model':<40}{'features':>10}{'lam':>9}{'CV MSE':>10}"
      f"{'test MSE':>11}{'align':>8}")
for dmax in (1, 2, 3):
    Phi, names = fourier_features(Xall, dmax)
    Kf = Phi @ Phi.T / Phi.shape[1]
    align = (np.sum(Kq_all * Kf)
             / (np.linalg.norm(Kq_all) * np.linalg.norm(Kf)))
    lam, cv = cv_features(Phi[:40], ytr)
    wv, mu = ridge_features(Phi[:40], ytr, lam)
    te = mse(Phi[40:] @ wv + mu, yte)
    print(f"  {'Fourier surrogate, total degree <= ' + str(dmax):<40}"
          f"{Phi.shape[1]:>10}{lam:9g}{cv:10.4f}{te:11.4f}{align:8.3f}")
Phi2, _ = shadow_features(Xall, 2)
lam2, cv2 = cv_features(Phi2[:40], ytr)
w2, mu2 = ridge_features(Phi2[:40], ytr, lam2)
te2 = mse(Phi2[40:] @ w2 + mu2, yte)
print(f"  {'Pauli surrogate, weight <= 2':<40}{Phi2.shape[1]:>10}{lam2:9g}"
      f"{cv2:10.4f}{te2:11.4f}{'-':>8}")
print(f"  {'quantum fidelity kernel ridge':<40}{'2^4 dim':>10}{lam_q:9g}"
      f"{cvq:10.4f}{mse_q:11.4f}{1.0:8.3f}")
print(f"  {'RBF kernel ridge (tuned)':<40}{'-':>10}{lam_r:9g}{cvr:10.4f}"
      f"{mse_r:11.4f}{'-':>8}")
print(f"  {'ordinary least squares':<40}{'5':>10}{'-':>9}{'-':>10}"
      f"{mse_ols:11.4f}{'-':>8}")

print("\n差を数値で述べる")
print("-" * 90)
Phi, _ = fourier_features(Xall, 2)
lam, _ = cv_features(Phi[:40], ytr)
wv, mu = ridge_features(Phi[:40], ytr, lam)
pred_f = Phi[40:] @ wv + mu
te_f = mse(pred_f, yte)
print(f"  quantum kernel ridge                     test MSE {mse_q:.4f}")
print(f"  degree-2 Fourier surrogate, no quantum    test MSE {te_f:.4f}"
      f"   ({te_f/mse_q:.2f}x)")
print(f"  weight-2 Pauli surrogate                  test MSE {te2:.4f}"
      f"   ({te2/mse_q:.2f}x)")
print(f"  tuned RBF                                 test MSE {mse_r:.4f}"
      f"   ({mse_r/mse_q:.2f}x)")


# ---- R6: 上の比はどれも20行での点推定である。対応をとる。 -------------------
def paired_bootstrap_mse(y_true, pred_a, pred_b, B=10000, seed=0, alpha=0.05):
    """MSE(a) - MSE(b) の95%区間。両者に同じテスト行を再標本化して用いる。

    第1章のR6。上で印字した比はすべて20行で測った2つの点推定値の比であり、
    その差が実在するかを述べるのはこの統計量である。
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


aq2, muq2 = krr_fit(quantum_kernel(Xtr, Xtr), ytr, lam_q)
pred_q = krr_predict(quantum_kernel(Xte, Xtr), aq2, muq2)
ar2, mur2 = krr_fit(rbf_kernel(Xtr, Xtr, g_star), ytr, lam_r)
pred_r = krr_predict(rbf_kernel(Xte, Xtr, g_star), ar2, mur2)
pred_p = Phi2[40:] @ w2 + mu2
P = {'quantum fidelity kernel': pred_q, 'Fourier surrogate d<=2': pred_f,
     'Pauli surrogate w<=2': pred_p, 'tuned RBF': pred_r}

print("\n  R6: 同じ差を、同じ20行で対応をとったブートストラップ区間とともに")
print(f"  {'MSE(A) - MSE(B)':<50}{'mean':>10}{'95% interval':>21}{'verdict':>10}")
for a, b in [('Fourier surrogate d<=2', 'quantum fidelity kernel'),
             ('Pauli surrogate w<=2', 'quantum fidelity kernel'),
             ('tuned RBF', 'quantum fidelity kernel'),
             ('Fourier surrogate d<=2', 'tuned RBF')]:
    m_, lo, hi = paired_bootstrap_mse(yte, P[a], P[b])
    v = "A better" if hi < 0.0 else ("B better" if lo > 0.0 else "no call")
    print(f"  {a + '  -  ' + b:<50}{m_:+10.4f}   [{lo:+.4f}, {hi:+.4f}]{v:>10}")
print("  実験の誤差幅の内側で量子モデルを再現できるサロゲートは、その量子モデルを走らせる")
print("  理由を取り除くサロゲートである。")
print("  この議論に答えるためにハードウェアの改善は要らない。必要なのは、カーネルに小さな")
print("  古典的記述が存在しないような問題である。")
```

```text
まず、量子カーネルが実際に含んでいる周波数を確かめる
------------------------------------------------------------------------------------------
  k(x, x_ref) を x1 のみの関数として x1 in [0,2) の64点で標本化
  harmonic (cycles per unit x1)        amplitude
  0.0                                   0.375000
  0.5                                   0.250000
  1.0                                   0.062500
  1.5                                   0.000000
  2.0                                   0.000000
  2.5                                   0.000000
  3.0                                   0.000000
  高調波 1.0 より上の全パワー: 1.227e-16
  スペクトルは厳密に打ち切られている。Ry(pi x) の再アップロード2層は x1 について
  0、1/2、1 cycles per unit の高調波のみを生み、それ以上を生まない。したがって
  同じ基底関数を持つ古典モデルは同じ関数空間を張る。第2章の周波数の議論を、
  そのまま攻撃手段として使うことになる。
  記述子あたり5つの関数（1, cos, sin, cos2, sin2）なので、*完全な*帯域制限積基底は
  5^4 = 625 個、一般には 5^d 個ある。以下のサロゲートが保つのは総次数 2 以下の
  41個だけであり、これは目的関数に高次相互作用がないという賭けである。タダ飯ではない。

サロゲートと、それが模倣している当のモデル
------------------------------------------------------------------------------------------
  model                                     features      lam    CV MSE   test MSE   align
  Fourier surrogate, total degree <= 1             9        1    0.0841     0.0546   0.759
  Fourier surrogate, total degree <= 2            41     0.01    0.0236     0.0172   0.846
  Fourier surrogate, total degree <= 3           129    1e-06    0.0337     0.0363   0.844
  Pauli surrogate, weight <= 2                    67      0.1    0.1058     0.0642       -
  quantum fidelity kernel ridge              2^4 dim     0.01    0.0722     0.0680   1.000
  RBF kernel ridge (tuned)                         -    1e-06    0.0232     0.0422       -
  ordinary least squares                           5        -         -     0.0465       -

差を数値で述べる
------------------------------------------------------------------------------------------
  quantum kernel ridge                     test MSE 0.0680
  degree-2 Fourier surrogate, no quantum    test MSE 0.0172   (0.25x)
  weight-2 Pauli surrogate                  test MSE 0.0642   (0.94x)
  tuned RBF                                 test MSE 0.0422   (0.62x)

  R6: 同じ差を、同じ20行で対応をとったブートストラップ区間とともに
  MSE(A) - MSE(B)                                         mean         95% interval   verdict
  Fourier surrogate d<=2  -  quantum fidelity kernel   -0.0507   [-0.1127, -0.0107]  A better
  Pauli surrogate w<=2  -  quantum fidelity kernel     -0.0039   [-0.0464, +0.0387]   no call
  tuned RBF  -  quantum fidelity kernel                -0.0257   [-0.0976, +0.0362]   no call
  Fourier surrogate d<=2  -  tuned RBF                 -0.0250   [-0.0613, +0.0015]   no call
  実験の誤差幅の内側で量子モデルを再現できるサロゲートは、その量子モデルを走らせる
  理由を取り除くサロゲートである。
  この議論に答えるためにハードウェアの改善は要らない。必要なのは、カーネルに小さな
  古典的記述が存在しないような問題である。
```

**注目すべき点。** スペクトルの検査が先に来て、それは厳密です。$k(\mathbf{x},\mathbf{x}_\mathrm{ref})$ を1つの記述子に沿って標本化してFFTをとると、高調波 0、1/2、1 サイクル毎単位に振幅 0.375、0.250、0.0625 が現れ、それより上のパワーは $1.2\times10^{-16}$ です。カーネルは記述子あたり3項の三角多項式です。高周波に何も隠れていません。高周波が存在しないからです。

これが分かれば、サロゲートは2行の構成です。総次数2まで $\cos(j\pi x_q)$ と $\sin(j\pi x_q)$ を列挙し、他のすべてと同じ交差検証で $\lambda$ を選んでリッジ回帰を当てはめる。41特徴、量子デバイスなし、テストMSE **0.0172** に対して量子カーネルは **0.0680**。4倍*良く*、調整済みRBFの 0.0422 よりも良いのです。総次数そのものも選ばれたのではなく交差検証されています。次数1のCVは0.0841、次数3は0.0337で、次数2の0.0236に対して負けるので、勝った行は訓練データが選んだ行です。

この数値の意味には2つの限界があり、どちらも末尾の留保ではなくここに属します。第一は相互作用の次数への賭けです。41特徴は $5^4 = 625$ 個の帯域制限基底の総次数 $\le 2$ の切片であり、完全な基底は記述子数について $5^d$ で増えます。このサロゲートが安いのは、ターゲットが対の構造に支配されているからです — 実際そうであり、生成子が $\sin(\pi x_0)\cos(\pi x_1) + \tfrac12 x_2^2 - 0.3 x_3$ である以上、構成上そうなっています。第二はR6の点です。0.0680 に対する 0.0172 は20テスト行で測った2つの点推定値の比であり、それと並べて印字される対応のある区間が、この差が実在するかを決めます。ここではそれが生き残ります — MSE(Fourier) $-$ MSE(量子) の区間は $[-0.113, -0.011]$ で完全にゼロより下 — であり、これが本章の荷重を担う結果です。隣接する比較のうち2つは生き残らないので、同じ息で述べておきます。weight 2 のPauliサロゲート対量子カーネルは $[-0.046, +0.039]$ で引き分け、調整済みRBF対量子カーネルは $[-0.098, +0.036]$ なので、Code Example 1 が報告した $1.61\times$ の差はこの単一分割では**解像できません**。その最後の区間こそ、5.3節が1つではなく20の分割を走らせる理由であり、その判定が差の大きさではなく20回すべてに勝つことに基づく理由です。

この結果は逆方向にも正確に読む必要があります。過大にも過小にも主張しやすいからです。量子カーネルが無用であることを示しては**いません**。*この*量子カーネルが*このデータ*上で、古典モデルが列挙してより効率よく使える関数クラスを持っていたことを示しており、符号化が浅く、記述子が少なく、相互作用が低次だったので列挙が容易だったことを示しています。悲観の方向での一般的な主張は、第4章が反対側から到達したものです。訓練できるほど浅く、勾配が消えないほど局所的な回路は、古典的に書き下せるほど単純でもある傾向があります。この緊張が、ハードウェアの限界ではなく、この分野の中心的な難所です。

### サロゲートが証明することと、しないこと

  * **証明すること。** ある報告された優位性が優位性ではないこと。サロゲートが並ぶなら、その問題で量子資源は何も買っていません。
  * **証明しないこと。** その問題でどんな量子モデルも役に立たないこと。別の符号化は小さな古典的記述を持たないカーネルを誘導しうる。
  * **自動的にスケールしないこと。** Fourier構成は周波数の台が列挙できることを要し、深い回路では破れます。第3章のZZ写像でも破れます。対の角度が記述子について2次であり帯域制限されないからです。列挙できる場合でも、この2層写像では記述子数について $5^d$ で増えるので、実際に作られるのは相互作用の次数による打ち切り — ここでは625個のうち41個 — であり、その打ち切りは真に高次の相互作用をもつターゲットでは失敗します。Pauli構成は別の変数について対応する制約を持ちます。低weight特徴が信号を担うことを要し、その個数は量子ビット数について $O(n^w)$ で増えます。どちらも低次構造への賭けであり、どちらもどんな特徴写像にも適用でき、どちらの破れ方も興味深い場合です — そしてそこが第4章の訓練可能性の代償が効く場所でもあります。
  * **安価であり、したがって必須であること。** 本コースのモデルについて両方のサロゲートを作るのに数百行と1秒未満の計算しか要りませんでした。古典データ上の量子優位の主張が、これらに照らして検査されていないなら、それは検査されていないのです。

* * *

## 5.3 量子優位の主張の読み方

2つのモデル、1つのデータセット、そして何を報告するかについての大きな自由。本節はその自由を意図的に使い、それが何を買うかを示します。

### Code Example 5: 1つの実験についての、真である3つの要約

```python
"""ベンチマークの衛生を定量化する: 優位性の主張はどう作られるか。
Code Example 1、3、4 の続き（同一セッション）。
"""
print("衛生項目 1 — 分割1つは結果ではない")
print("-" * 92)
print("  同じ60点をランダムに40/20へ分ける操作を20回。両モデルは各分割の訓練データ内で")
print("  5分割交差検証によりlambdaを再選択する。それ以外は何も変えない。")
diffs, qs, rs = [], [], []
for rep in range(20):
    perm = np.random.default_rng(500 + rep).permutation(len(yall))
    tr, te = perm[:40], perm[40:]
    Xa, ya, Xb, yb = Xall[tr], yall[tr], Xall[te], yall[te]
    lq, _ = cv_select(lambda a, b: quantum_kernel(a, b), Xa, ya)
    aq_, mq_ = krr_fit(quantum_kernel(Xa, Xa), ya, lq)
    eq = mse(krr_predict(quantum_kernel(Xb, Xa), aq_, mq_), yb)
    best = (None, np.inf)
    for g in GAMMAS:
        lr_, c_ = cv_select(lambda a, b, g=g: rbf_kernel(a, b, g), Xa, ya)
        if c_ < best[1]:
            best = ((g, lr_), c_)
    (gs, ls), _ = best
    ar_, mr_ = krr_fit(rbf_kernel(Xa, Xa, gs), ya, ls)
    er = mse(krr_predict(rbf_kernel(Xb, Xa, gs), ar_, mr_), yb)
    qs.append(eq); rs.append(er); diffs.append(eq - er)
qs, rs, diffs = np.array(qs), np.array(rs), np.array(diffs)
print(f"  {'statistic':<44}{'quantum':>12}{'RBF':>12}")
print(f"  {'mean test MSE over 20 splits':<44}{qs.mean():12.4f}{rs.mean():12.4f}")
print(f"  {'median test MSE':<44}{np.median(qs):12.4f}{np.median(rs):12.4f}")
print(f"  {'best single split':<44}{qs.min():12.4f}{rs.min():12.4f}")
print(f"  {'worst single split':<44}{qs.max():12.4f}{rs.max():12.4f}")
print(f"  {'splits won':<44}{int((diffs < 0).sum()):12d}"
      f"{int((diffs > 0).sum()):12d}")
se = diffs.std(ddof=1) / np.sqrt(len(diffs))
print(f"\n  対応のある差（quantum - RBF）: 平均 {diffs.mean():+.4f}"
      f"  標準誤差 {se:.4f}  t = {diffs.mean()/se:+.2f}")
print("  この t は妥当な t 統計量ではなく、本コースはそのふりをしない。20分割は同じ60点")
print("  の再標本化なので訓練集合が大きく重なり、20個の差は正に相関する。素朴な標準誤差")
print("  は小さすぎ、t は膨らむ。これは再標本化 t 検定に対するDietterichの批判であり、")
print("  「ランダム分割の繰り返し」を載せるすべての表に当てはまる。")
print("  ここで結論を担うのは分布によらない事実の方である。量子側は")
print(f"  {len(diffs)} 分割のうち {int((diffs > 0).sum())} 敗であり、2つのモデルが同等だとする")
print(f"  どんな帰無仮説の下でも、符号検定によればその確率は 2^-{len(diffs)} = {2.0**-len(diffs):.1e} である。")
j = int(np.argmin(diffs))
print(f"  量子側に最も有利な分割: #{j}、quantum {qs[j]:.4f} 対 RBF {rs[j]:.4f}")
print(f"\n  この1つの実験について、算術的にはすべて真である3つの文:")
print(f"    「量子カーネルは test MSE {qs.min():.4f} に達し、RBFベースラインの"
      f" {rs.max():.4f} を")
print(f"     上回った」 — 量子の最良分割と古典の最悪分割の比較、{rs.max()/qs.min():.2f} 倍の勝ち")
print(f"    「標準の分割では量子カーネルが {mse_q:.4f}、RBFが {mse_r:.4f}」 —"
      f" {mse_q/mse_r:.2f} 倍の負け")
print(f"    「対応のある20分割で量子カーネルは全敗、平均 {qs.mean():.4f} 対"
      f" {rs.mean():.4f}」")
print("  正直な要約は3番目だけであり、実験の再現によって反証できるのも3番目だけである。")
print("  比較の2つの腕をまたいで分割を選ぶことが、結果を作り出す最も効果的な方法である。")

print("\n衛生項目 2 — 量子側を固定したままベースラインの強さを変える")
print("-" * 92)
print("  同じ量子側の結果を、努力量の異なるRBFベースラインと比べる。")
print(f"  {'baseline':<44}{'test MSE':>12}{'quantum looks':>16}")
Kq_tr = quantum_kernel(Xtr, Xtr)
for label, g, lam in [("RBF, gamma=10, lambda=1 (barely tuned)", 10.0, 1.0),
                      ("RBF, gamma=10, lambda=1e-3", 10.0, 1e-3),
                      ("RBF, gamma=1, lambda=1e-3 (a guess)", 1.0, 1e-3),
                      ("RBF, gamma and lambda by 5-fold CV", g_star, lam_r)]:
    a_, m_ = krr_fit(rbf_kernel(Xtr, Xtr, g), ytr, lam)
    e = mse(krr_predict(rbf_kernel(Xte, Xtr, g), a_, m_), yte)
    verdict = f"{e/mse_q:.2f}x better" if e > mse_q else f"{mse_q/e:.2f}x worse"
    print(f"  {label:<44}{e:12.4f}{verdict:>16}")
print(f"  quantum fidelity kernel, lambda by the same CV: {mse_q:.4f}")
print("  この4行の間で量子モデルは何も変わっていない。それでも結論は「2.25倍良い」から")
print("  「3.50倍悪い」まで動き、動かしたのは対戦相手の選び方だけである。")
print(f"  さらに正直な補足として、CVが選んだRBFはここでは最良のRBFではない。gamma = {g_orc:g}")
print(f"  はこの分割で {mse_orc:.4f} に達する。40点はgammaを信頼して同定するには少なすぎる。")
print("  これは両側に効く話であり、対応のある誤差幅を伴わない単一の数値が証拠にならない")
print("  理由でもある。上に並べた数値自身にも当てはまる。")

print("\n衛生項目 3 — データ量")
print("-" * 92)
print(f"  {'train size':>11}{'quantum':>10}{'RBF':>10}{'q/RBF':>8}{'OLS':>10}"
      f"{'Fourier d2':>12}{'mean of y':>11}")
Phi_all, _ = fourier_features(Xall, 2)
for ntr in (10, 15, 20, 30, 40):
    Xa, ya = Xall[:ntr], yall[:ntr]
    lq, _ = cv_select(lambda a, b: quantum_kernel(a, b), Xa, ya, folds=5)
    aq_, mq_ = krr_fit(quantum_kernel(Xa, Xa), ya, lq)
    eq = mse(krr_predict(quantum_kernel(Xte, Xa), aq_, mq_), yte)
    best = (None, np.inf)
    for g in GAMMAS:
        lr_, c_ = cv_select(lambda a, b, g=g: rbf_kernel(a, b, g), Xa, ya, folds=5)
        if c_ < best[1]:
            best = ((g, lr_), c_)
    (gs, ls), _ = best
    ar_, mr_ = krr_fit(rbf_kernel(Xa, Xa, gs), ya, ls)
    er = mse(krr_predict(rbf_kernel(Xte, Xa, gs), ar_, mr_), yte)
    Aa = np.hstack([Xa, np.ones((ntr, 1))])
    ca = np.linalg.lstsq(Aa, ya, rcond=None)[0]
    eo = mse(np.hstack([Xte, np.ones((len(yte), 1))]) @ ca, yte)
    lf, _ = cv_features(Phi_all[:ntr], ya, folds=5)
    wf, mf = ridge_features(Phi_all[:ntr], ya, lf)
    ef = mse(Phi_all[40:] @ wf + mf, yte)
    print(f"  {ntr:>11}{eq:10.4f}{er:10.4f}{eq/er:8.2f}{eo:10.4f}{ef:12.4f}"
          f"{mse(np.full(len(yte), ya.mean()), yte):11.4f}")
print("  絶対値ではなく比の列を読むこと。どのモデルもデータが増えれば良くなるので絶対差")
print("  は 0.08 から 0.03 へ縮み、都合よく書けば「古典側の優位はスケールとともに消える」")
print("  と表現できてしまう。比は 1.6 から 3.2 の間をさまようだけで傾向がない。量子カーネル")
print("  はどの規模でも一定倍率だけ悪く、ふらつきはテスト20点の標本誤差である。")
print("  点ではなく曲線を出し、そこに誤差幅を付けること。")

print("\n量子優位の主張を読むためのチェックリスト")
print("-" * 92)
CHECKLIST = [
    ("データ量",
     "数百点を下回ると、2つのモデルの比較はそもそも分離できない。",
     "孤立した2つの点推定ではなく、対応のある標準誤差を要求すること。"),
    ("ベースラインの強さ",
     "調整済みRBF、勾配ブースティング木、素の線形回帰が最低ラインである。",
     "「未調整のSVMに勝った」は何も測定していない。"),
    ("調整の対等性",
     "両側で探索したハイパーパラメータの数と費やした計算量を数えること。",
     "探索努力の不均衡は、天秤に置かれる最も一般的な無言の親指である。"),
    ("選択の回数",
     "分割・乱数種・初期値・特徴写像・量子ビット数をいくつ試し、報告値は",
     "その最良かを問うこと。最大値ではなく分布を要求する。"),
    ("データの出自",
     "特徴が表計算や古典シミュレーションから来るなら、入力問題と脱量子化の",
     "議論がそのまま全面的に当てはまる。"),
    ("サロゲートの排除",
     "低weight Pauli特徴と符号化自身のFourier基底はどちらも安価である。",
     "どちらかが量子モデルに並べば、説明すべきものは残っていない。"),
    ("資源の報告",
     "ショット数、回路評価回数、実時間、そしてどうせ必要になる古典側の前処理・",
     "後処理。優位性は比であって勝敗ではない。"),
    ("ノイズより大きいか",
     "主張された差をデータ自身のラベルノイズと比べること。",
     "報告される差の多くは、それを測る際のノイズより小さい。"),
]
for i, (head, l1, l2) in enumerate(CHECKLIST, start=1):
    print(f"  {i}. {head}")
    print(f"     {l1}")
    print(f"     {l2}")
```

```text
衛生項目 1 — 分割1つは結果ではない
--------------------------------------------------------------------------------------------
  同じ60点をランダムに40/20へ分ける操作を20回。両モデルは各分割の訓練データ内で
  5分割交差検証によりlambdaを再選択する。それ以外は何も変えない。
  statistic                                        quantum         RBF
  mean test MSE over 20 splits                      0.0843      0.0242
  median test MSE                                   0.0777      0.0238
  best single split                                 0.0375      0.0108
  worst single split                                0.1833      0.0431
  splits won                                             0          20

  対応のある差（quantum - RBF）: 平均 +0.0600  標準誤差 0.0080  t = +7.54
  この t は妥当な t 統計量ではなく、本コースはそのふりをしない。20分割は同じ60点
  の再標本化なので訓練集合が大きく重なり、20個の差は正に相関する。素朴な標準誤差
  は小さすぎ、t は膨らむ。これは再標本化 t 検定に対するDietterichの批判であり、
  「ランダム分割の繰り返し」を載せるすべての表に当てはまる。
  ここで結論を担うのは分布によらない事実の方である。量子側は
  20 分割のうち 20 敗であり、2つのモデルが同等だとする
  どんな帰無仮説の下でも、符号検定によればその確率は 2^-20 = 9.5e-07 である。
  量子側に最も有利な分割: #4、quantum 0.0487 対 RBF 0.0302

  この1つの実験について、算術的にはすべて真である3つの文:
    「量子カーネルは test MSE 0.0375 に達し、RBFベースラインの 0.0431 を
     上回った」 — 量子の最良分割と古典の最悪分割の比較、1.15 倍の勝ち
    「標準の分割では量子カーネルが 0.0680、RBFが 0.0422」 — 1.61 倍の負け
    「対応のある20分割で量子カーネルは全敗、平均 0.0843 対 0.0242」
  正直な要約は3番目だけであり、実験の再現によって反証できるのも3番目だけである。
  比較の2つの腕をまたいで分割を選ぶことが、結果を作り出す最も効果的な方法である。

衛生項目 2 — 量子側を固定したままベースラインの強さを変える
--------------------------------------------------------------------------------------------
  同じ量子側の結果を、努力量の異なるRBFベースラインと比べる。
  baseline                                        test MSE   quantum looks
  RBF, gamma=10, lambda=1 (barely tuned)            0.1528    2.25x better
  RBF, gamma=10, lambda=1e-3                        0.1091    1.61x better
  RBF, gamma=1, lambda=1e-3 (a guess)               0.0194     3.50x worse
  RBF, gamma and lambda by 5-fold CV                0.0422     1.61x worse
  quantum fidelity kernel, lambda by the same CV: 0.0680
  この4行の間で量子モデルは何も変わっていない。それでも結論は「2.25倍良い」から
  「3.50倍悪い」まで動き、動かしたのは対戦相手の選び方だけである。
  さらに正直な補足として、CVが選んだRBFはここでは最良のRBFではない。gamma = 2
  はこの分割で 0.0208 に達する。40点はgammaを信頼して同定するには少なすぎる。
  これは両側に効く話であり、対応のある誤差幅を伴わない単一の数値が証拠にならない
  理由でもある。上に並べた数値自身にも当てはまる。

衛生項目 3 — データ量
--------------------------------------------------------------------------------------------
   train size   quantum       RBF   q/RBF       OLS  Fourier d2  mean of y
           10    0.1977    0.1197    1.65    0.1234      0.0704     0.2497
           15    0.1475    0.0495    2.98    0.1272      0.0527     0.2964
           20    0.1682    0.0549    3.06    0.0548      0.0739     0.2763
           30    0.0977    0.0305    3.21    0.0464      0.0251     0.2742
           40    0.0680    0.0422    1.61    0.0465      0.0172     0.2747
  絶対値ではなく比の列を読むこと。どのモデルもデータが増えれば良くなるので絶対差
  は 0.08 から 0.03 へ縮み、都合よく書けば「古典側の優位はスケールとともに消える」
  と表現できてしまう。比は 1.6 から 3.2 の間をさまようだけで傾向がない。量子カーネル
  はどの規模でも一定倍率だけ悪く、ふらつきはテスト20点の標本誤差である。
  点ではなく曲線を出し、そこに誤差幅を付けること。

量子優位の主張を読むためのチェックリスト
--------------------------------------------------------------------------------------------
  1. データ量
     数百点を下回ると、2つのモデルの比較はそもそも分離できない。
     孤立した2つの点推定ではなく、対応のある標準誤差を要求すること。
  2. ベースラインの強さ
     調整済みRBF、勾配ブースティング木、素の線形回帰が最低ラインである。
     「未調整のSVMに勝った」は何も測定していない。
  3. 調整の対等性
     両側で探索したハイパーパラメータの数と費やした計算量を数えること。
     探索努力の不均衡は、天秤に置かれる最も一般的な無言の親指である。
  4. 選択の回数
     分割・乱数種・初期値・特徴写像・量子ビット数をいくつ試し、報告値は
     その最良かを問うこと。最大値ではなく分布を要求する。
  5. データの出自
     特徴が表計算や古典シミュレーションから来るなら、入力問題と脱量子化の
     議論がそのまま全面的に当てはまる。
  6. サロゲートの排除
     低weight Pauli特徴と符号化自身のFourier基底はどちらも安価である。
     どちらかが量子モデルに並べば、説明すべきものは残っていない。
  7. 資源の報告
     ショット数、回路評価回数、実時間、そしてどうせ必要になる古典側の前処理・
     後処理。優位性は比であって勝敗ではない。
  8. ノイズより大きいか
     主張された差をデータ自身のラベルノイズと比べること。
     報告される差の多くは、それを測る際のノイズより小さい。
```

**注目すべき点。** 3つのブロックはそれぞれ結果を作り出す別の方法であり、どれも印刷物でよく見かけます。

**比較の腕をまたぐこと。** ランダムな分割20回、各分割内で両モデルを再調整すると、量子カーネルは**20回すべて負けます**。それでも量子側の最良分割 0.0375 は古典側の最悪分割 0.0431 より良いので、その2つの数値を比べる文は算術的に真であり $1.15\times$ の勝利を記述します。その文に捏造は1つもありません。操作はまるごと対応付けの中にあります。

このブロックは対応のある差 $+0.0600$、名目標準誤差 0.0080、$t = 7.5$ も印字し、そのうえで読者にその $t$ を信じないよう告げます。これは意図的です。同じ欠陥がこの分野の「ランダム分割の繰り返し」の表のほとんどに潜んでいるからです。同じ60点の分割20回は訓練行の大半を共有するので、20個の分割ごとの差は正に相関し、素朴な標準誤差は小さすぎ、$t$ は未知の倍率で膨らみます — 再標本化 $t$ 検定に対するDietterichの批判であり、他人の表と同じだけ本コース自身の表にも当てはまります。結論を担うのはその隣にある分布によらない言明です。20回中20敗であり、2つのモデルが同等な帰無仮説の下では符号検定によって確率 $2^{-20} \approx 10^{-6}$ です。符号検定は大きさを捨てて向きだけを残しますが、重なりが壊さない情報はまさにその向きです。引用すべきは $t$ ではなくこちらです。

**対戦相手を選ぶこと。** 量子側を完全に固定したまま、古典ベースラインの努力量だけを変えます。$\gamma = 10, \lambda = 1$ のRBFに対して量子カーネルは $2.25\times$ 良く見え、交差検証したRBFに対しては $1.61\times$ 悪く見え、$\gamma = 1, \lambda = 10^{-3}$ という当たりの良い当て推量に対しては $3.50\times$ 悪く見えます。量子モデルは変えずに、報告される結論の幅が8倍です。「標準的なSVMと比較した」という文が情報を持たないのはこのためです。問うべきは、提案に対してベースラインにどれだけの努力が払われたかであり、その答えはほとんど報告されません。

そのブロックには正直な複雑さもあります。交差検証で選ばれたRBFはここでは*最良の*RBFではありません。40点は5分割CVが $\gamma$ を信頼して同定するには少なすぎるので、手順は当て推量より悪い設定を選びました。これは両方向に効きますし、対応のある誤差幅を伴わない単一の点推定が証拠にならない理由でもあります。本コース自身の数値にも当てはまる教訓です。

**規模を選ぶこと。** 訓練10点から40点にかけて、量子カーネルのRBFに対する*絶対*差は 0.078 から 0.026 に縮み、都合よく書けば「優位はスケールとともに現れる」と記述できてしまいます。*比*は 1.6 から 3.2 の間をさまよい傾向がないので、正しい読み方はこうです。量子カーネルはどこでも一定倍率だけ悪く、ふらつきはテスト20点の標本誤差である。曲線を出し、そこに誤差幅を付けること。

### チェックリスト

Code Example 5 が出力する8項目が実務的な蒸留です。うち2つは、最も欠けがちで最も決定的なので強調しておきます。

**「データセットは古典的か？」** 特徴が表計算、DFT計算、あるいは装置のデータ縮約パイプラインから届くなら、それを量子状態に載せること自体がボトルネック — 第1章の入力問題 — であり、本章の脱量子化の議論がそのまま全面的に当てはまります。公表されたQMLベンチマークのほぼすべてが古典データを使っています。

**「古典サロゲートは排除されたか？」** 5.2節が雛形です。2つのサロゲートを作るのは、量子モデルを走らせられる人にとって1日の仕事であり、どちらかが並べば説明すべきものは残りません。それらに照らして検査していない主張は、何も測定していません。

読むときの有用な習慣として、報告された数値を見る前に*ベースライン*がどうあるべきかと*対応のある*統計量がどうあるべきかを書き下し、それから論文にそれらがあるかを見ることです。その不在は数値より情報量があります。

* * *

## 5.4 量子データが話を変えるとき

本コースの否定的な結果はすべて1つの前提を共有しています。データが古典的であることです。材料あたり4つの実数が表の中にあり、量子モデルが見る前に量子レジスタへ載せなければならない。この前提がすべての働きをしており、それがどう働いているかを正確に見る価値があります。

### 入力問題と、それが消える理由

古典ベクトルを量子状態に載せるには何かがかかります。角度符号化は深さでは安価ですが特徴あたり量子ビット1つを使い、制約された特徴写像を与えます。振幅符号化はコンパクトですが、次元とともに深くなる状態準備回路を要します。いずれにせよ載せる操作は*古典から量子への*変換であり、古典アルゴリズムは一度も実行する必要がないもので、そのコストはまるごと比較の量子側に請求されます。さらに悪いことに、符号化が課す構造はいまやモデルの帰納バイアスであり、5.2節はそのバイアスがいかに容易に書き下され、代わりに古典的に使われうるかを示しました。

ではデータが既に量子状態であるとしましょう。パラメトリック下方変換源から出るもつれ光子対、パルス列の途中にある核スピン集団の状態、クエンチ後の冷却原子アレイの多体状態、候補触媒の量子シミュレーションの出力。これらの場合、載せるものは何もありません。量子計算機は量子状態を直接受け取り、古典の競合相手はそれを受け取ることすらできません。誰かが抽出することを選んだ古典的な測定記録 — 状態の損失を伴う射影 — しか受け取れないのです。

これは第1章の4象限の絵におけるCQ/QQの区別であり、非対称性が逆向きに走る唯一の場所です。この方向の結果のいくつかは真に厳密であり、その主張の正確な形が重要です。これらは**サンプル複雑度**の分離であって計算量の分離ではありません。量子状態上に定義された学習課題で、状態の単一コピーに制限されたどんな戦略も — 古典的な後処理がどれほど巧妙でも、計算量が無制限でも — 指数的に多くの実験を要するのに対し、2コピー以上をコヒーレントに処理できる戦略は多項式回で済むものがあります。限界がかかるのは実験回数であって実行時間ではなく、しかも誰かが思いついた古典アルゴリズムに対してではなくあらゆる単一コピー測定方式に対して成り立ちます。機構は、コヒーレントなプロセッサが*アンサンブル*の性質 — たとえば異なる実験設定から来た状態の間の重なり — を測れることであり、それは独立な単一コピー測定をいくら並べても効率よく再構成できません。

### 代わりに何がかかるか

入力問題は3つの新しい問題に置き換わり、量子データの場合を障害のないものとして提示するのは不誠実です。

  * **状態は保存できません。** 古典データセットは無限に再利用できますが、量子状態は測定で消費されコピーもできません。「訓練例1万個」はエポックごとに実験を1万回走らせることを意味し、訓練点を再訪するアルゴリズムはそれを再準備しなければなりません。
  * **実験をプロセッサと同じ場所に置く必要があります。** 状態はコヒーレントに届かなければなりません。実際にはセンサやシミュレーションと量子プロセッサが同一の装置であるか、量子チャネルで結ばれていることを意味します。これは重い工学的制約であり、既存の装置の大半を除外します。
  * **古典ベースラインは弱くありません。** 古典シャドウ — 5.2節のサロゲート特徴と同じ構成 — は、ランダム化測定から古典アルゴリズムが多くを学べるようにし、低weight観測量については標本複雑度が量子ビット数に依りません。提案された量子データの優位性のいくつかは、まさにこの経路で脱量子化されました。生き残る分離は、関心のある量が低weight観測量ではない場合です。

### 材料研究にとって具体的に何を意味するか

誠実な立場は、量子データが議論を程度ではなく種類において変える方向であり、材料科学における近い将来の実例は狭いが実在するというものです。

  * **訓練データとしての量子シミュレーション出力。** 誤り耐性のある機械が将来、強相関物質の基底状態を計算するなら、抽出された数個の数値ではなく状態そのものから性質を学ぶのは、古典的な載せ込み段階のないQQ象限の課題です。
  * **量子センシングと計量。** 信号が古典的な読み出しではなく量子プローブのコヒーレンスに宿る測定は、この意味での量子データを生みます。そうしたプローブからの識別や推定を学ぶことは活発な領域であり、その扱いは量子センシングの講座に属するもので、本コースには属しません。
  * **量子デバイス自身の特性評価。** プロセッサのノイズモデルやハミルトニアンをその状態から学ぶことは、最も直ちに実用的な実例であり、これは材料の問題です。答えは欠陥、界面、2準位揺動子についての言明になります。姉妹コース[量子ハードウェア入門](<../../FM/quantum-hardware-introduction/index.html>)が物理を扱い、学習問題はその上に載ります。

これらはどれも、材料情報学のグループが今年構築できるパイプラインではありません。しかしどれも、この分野を切り捨てるのではなく理解しておく価値がある理由です。

* * *

## 5.5 議論全体を1つの走査で

### Code Example 6: 集中と脱量子化を同時に

```python
"""総括: 集中と脱量子化を1つの走査で。
Code Example 1、2、3、4、5 の続き（同一セッション）。
"""


def widen(Xs, n):
    """n量子ビットのレジスタを4個の記述子で埋める。必要なら繰り返す。"""
    reps = int(np.ceil(n / Xs.shape[1]))
    return np.tile(Xs, (1, reps))[:, :n]


def krr_test_mse(kernel_fn, Xa, ya, Xb, yb):
    lam, cv = cv_select(kernel_fn, Xa, ya)
    a, mu = krr_fit(kernel_fn(Xa, Xa), ya, lam)
    return mse(krr_predict(kernel_fn(Xb, Xa), a, mu), yb), lam, cv


print("集中とその古典サロゲートを並べて見る")
print("-" * 96)
print("  shots/entry = 100 k(1-k)/sigma^2、すなわちBernoulli成分に eps = sigma/10")
print(f"  {'n':>3}{'off-diag std':>14}{'quantum test':>14}"
      f"{'Pauli w<=2':>12}{'features':>10}{'shots/entry':>13}{'quantum/Pauli':>15}")
NSW = list(range(2, 9))
sweep = {}
for n in NSW:
    Xw = widen(Xall, n)
    Kw = quantum_kernel(Xw, Xw)
    off = Kw[~np.eye(len(Kw), dtype=bool)]
    kf = lambda a, b: quantum_kernel(a, b)
    te_q, lam_qn, _ = krr_test_mse(kf, Xw[:40], ytr, Xw[40:], yte)
    Phi, ps = shadow_features(Xw, 2, n=n)
    lam_s, _ = cv_features(Phi[:40], ytr)
    wv, mu = ridge_features(Phi[:40], ytr, lam_s)
    te_s = mse(Phi[40:] @ wv + mu, yte)
    sweep[n] = (off.std(), te_q, te_s, Phi.shape[1])
    kb = off.mean()
    print(f"  {n:>3}{off.std():14.6f}{te_q:14.4f}{te_s:12.4f}"
          f"{Phi.shape[1]:>10}{100 * kb * (1 - kb) / off.std()**2:13,.0f}"
          f"{te_q/te_s:15.2f}")

print("\nn に全く依存しない、純粋に古典的な参照値")
print("-" * 96)
Phi_f, _ = fourier_features(Xall, 2)
lam_f, cv_f = cv_features(Phi_f[:40], ytr)
wf, mf = ridge_features(Phi_f[:40], ytr, lam_f)
te_fourier = mse(Phi_f[40:] @ wf + mf, yte)
print(f"  {'Fourier surrogate, total degree <= 2':<44}{Phi_f.shape[1]:>10} features"
      f"   test MSE {te_fourier:.4f}")
print(f"  {'RBF kernel ridge, gamma and lambda by CV':<44}{'-':>10}          "
      f"   test MSE {mse_r:.4f}")
print(f"  {'RBF kernel ridge, best gamma on test (bound)':<44}{'-':>10}          "
      f"   test MSE {mse_orc:.4f}")
print(f"  {'ordinary least squares':<44}{'5':>10} features"
      f"   test MSE {mse_ols:.4f}")
print(f"  {'best quantum result anywhere in the sweep':<44}{'-':>10}          "
      f"   test MSE {min(v[1] for v in sweep.values()):.4f}"
      f"  (n = {min(sweep, key=lambda k: sweep[k][1])})")

print("\n本コース全体の総括を数値で")
print("-" * 96)
rows = [
    ("predict the training mean", 1, mse_mean),
    ("ordinary least squares", 5, mse_ols),
    ("RBF kernel ridge, tuned by CV", 40, mse_r),
    ("RBF kernel ridge, best gamma on test (bound)", 40, mse_orc),
    ("Fourier surrogate of the quantum feature map", Phi_f.shape[1], te_fourier),
    ("weight-2 Pauli surrogate, 4 qubits", 67, sweep[4][2]),
    ("quantum fidelity kernel ridge, 4 qubits", 40, sweep[4][1]),
    ("quantum fidelity kernel ridge, best n", 40,
     min(v[1] for v in sweep.values())),
]
print(f"  {'model':<46}{'params/features':>17}{'test MSE':>11}{'vs best':>10}")
best = min(r[2] for r in rows)
for nm, npar, e in rows:
    print(f"  {nm:<46}{npar:>17}{e:11.4f}{e/best:10.2f}x")
print(f"  {'irreducible noise floor':<46}{'-':>17}{0.05**2:11.4f}"
      f"{0.05**2/best:10.2f}x")

print("\nこの走査が確立すること")
print("-" * 96)
print("  1. レジスタを広げると量子モデルは悪くなる。単調ではない — n = 3 が最良、n = 8 が")
print("     最悪で、その間の凹凸はテスト20点では分解できない — が、走査全体の傾向は")
print("     明確である。カーネルが集中し、Gram行列は単位行列へ近づき、単位行列上の")
print("     カーネルリッジ回帰は汎化のない引き当て表になる。")
print("     量子ビットを増やすことはモデルを増やすことではなく、古典データに対しては")
print("     モデルを減らすことである。")
print("  2. O(n^2) 個の特徴をもつ古典サロゲートは、どの幅でも量子モデルを両方向に")
print("     2倍程度の範囲で追随し、量子側に系統的な優位はない。量子カーネルが集中した")
print("     幅では打ち切りが助けにさえなる。忠実度カーネルが欠いている正則化として")
print("     働くからである。")
print("  3. 表の中の最良値は、量子符号化自身の周波数リストだけから作った41特徴の古典")
print("     モデルに属する。量子モデルが表現できたものはすべて、それなしで表現できた。")
print("  4. ここには1行もハードウェアの話がない。シミュレータは厳密かつ無雑音であり、")
print("     誤り率もコヒーレンス時間も量子ビット数もどの数値も変えない。障害は問題の側に")
print("     あって機械の側にはない。だからこそ、より良い量子ハードウェアではなく量子")
print("     データを持つ問題を探すべきなのである。")
```

```text
集中とその古典サロゲートを並べて見る
------------------------------------------------------------------------------------------------
  shots/entry = 100 k(1-k)/sigma^2、すなわちBernoulli成分に eps = sigma/10
    n  off-diag std  quantum test  Pauli w<=2  features  shots/entry  quantum/Pauli
    2      0.309325        0.0362      0.0362        16          241           1.00
    3      0.262934        0.0284      0.0345        37          268           0.82
    4      0.191101        0.0680      0.0642        67          355           1.06
    5      0.167800        0.0628      0.0735       106          347           0.85
    6      0.137245        0.0831      0.1955       154          382           0.43
    7      0.119306        0.0838      0.0506       211          375           1.65
    8      0.105990        0.1064      0.0667       277          381           1.60

n に全く依存しない、純粋に古典的な参照値
------------------------------------------------------------------------------------------------
  Fourier surrogate, total degree <= 2                41 features   test MSE 0.0172
  RBF kernel ridge, gamma and lambda by CV             -             test MSE 0.0422
  RBF kernel ridge, best gamma on test (bound)         -             test MSE 0.0208
  ordinary least squares                               5 features   test MSE 0.0465
  best quantum result anywhere in the sweep            -             test MSE 0.0284  (n = 3)

本コース全体の総括を数値で
------------------------------------------------------------------------------------------------
  model                                           params/features   test MSE   vs best
  predict the training mean                                     1     0.2747     15.94x
  ordinary least squares                                        5     0.0465      2.70x
  RBF kernel ridge, tuned by CV                                40     0.0422      2.45x
  RBF kernel ridge, best gamma on test (bound)                 40     0.0208      1.21x
  Fourier surrogate of the quantum feature map                 41     0.0172      1.00x
  weight-2 Pauli surrogate, 4 qubits                           67     0.0642      3.73x
  quantum fidelity kernel ridge, 4 qubits                      40     0.0680      3.94x
  quantum fidelity kernel ridge, best n                        40     0.0284      1.65x
  irreducible noise floor                                       -     0.0025      0.15x

この走査が確立すること
------------------------------------------------------------------------------------------------
  1. レジスタを広げると量子モデルは悪くなる。単調ではない — n = 3 が最良、n = 8 が
     最悪で、その間の凹凸はテスト20点では分解できない — が、走査全体の傾向は
     明確である。カーネルが集中し、Gram行列は単位行列へ近づき、単位行列上の
     カーネルリッジ回帰は汎化のない引き当て表になる。
     量子ビットを増やすことはモデルを増やすことではなく、古典データに対しては
     モデルを減らすことである。
  2. O(n^2) 個の特徴をもつ古典サロゲートは、どの幅でも量子モデルを両方向に
     2倍程度の範囲で追随し、量子側に系統的な優位はない。量子カーネルが集中した
     幅では打ち切りが助けにさえなる。忠実度カーネルが欠いている正則化として
     働くからである。
  3. 表の中の最良値は、量子符号化自身の周波数リストだけから作った41特徴の古典
     モデルに属する。量子モデルが表現できたものはすべて、それなしで表現できた。
  4. ここには1行もハードウェアの話がない。シミュレータは厳密かつ無雑音であり、
     誤り率もコヒーレンス時間も量子ビット数もどの数値も変えない。障害は問題の側に
     あって機械の側にはない。だからこそ、より良い量子ハードウェアではなく量子
     データを持つ問題を探すべきなのである。
```

**注目すべき点。** これは本コースの総括実験なので、ゆっくり読む価値があります。

**レジスタを広げると量子モデルは悪くなります。** 3量子ビットが最良、8量子ビットが最悪。単調ではありません — テスト20点では凹凸が分解できません — が、走査全体の傾向は明白で、機構は第2列にあります。非対角の広がりは $n=2$ から $n=8$ で3分の1になり、Gram行列は単位行列へ漂い、回帰に使える情報が順に減っていきます。

**古典サロゲートはどの幅でも量子モデルに追随します。** どちらの方向にも2倍程度の範囲で、量子側に系統的な優位はありません。その特徴数は $O(n^2)$ で、4量子ビットで67、8量子ビットで277、厳密な展開の $4^n$ に対してです。

**表の中の最良値は古典モデルに属します。** 41特徴のFourierサロゲート 0.0172 で、量子符号化自身の周波数リストだけから作ったものです。走査中のどこでも最良の量子結果は3量子ビットの 0.0284 で、それすら量子優位を理由に誰も提案しない構成であり、しかも最強のRBFの 0.0208 に及びません。上界の行はここで効きます。交差検証したRBFを弱い相手だと疑う読者は代わりに上界の行を使えばよく、本章のすべての結論はその置き換えに耐えます。

**どれ1つもハードウェアの話ではありません。** シミュレータは厳密かつ無雑音です。誤り率もコヒーレンス時間も量子ビット数もどの項目も変えません。これは本コースで最も重要な1文です。この種の結果に対する標準的な反応が「ハードウェアが良くなれば動く」であり、ここではそれが単純に当てはまらないからです。

* * *

## 5.6 実践指針と、このシリーズが何のためにあったか

### いま学ぶ価値があること

否定的な結果5章の結論は「この分野を無視せよ」ではありません。学ぶ*理由*が通常挙げられるものとは違い、その別の理由は長持ちする、ということです。

**カーネルの視点を学ぶ。持ち運べるからです。** 第3章の内容 — 特徴写像、誘導される内積、閉形式のカーネルリッジ回帰、Gram行列の幾何、実効ランク — は、量子的な問いによって研ぎ澄まされた古典機械学習です。量子計算が20年停滞しても役に立ち続けます。「自分の表現はどんな内積を誘導しており、その内積は物理を尊重しているか」と問う習慣は、材料情報学の実務者にとってどんな回路よりも価値があります。

**周波数の視点を学ぶ。診断に使えるからです。** 第2章の絵 — モデルは打ち切りFourier級数であり、その周波数の台は当てはめ手続きではなく表現が決める — は、特徴表現が何を表現できて何を表現できないかを考えるための、利用可能な最も明快な方法です。記述子の設計、古典カーネルのFourier特徴、Transformerの位置符号化に直接移せます。

**ベンチマークの規律を学ぶ。分野がそれを必要としているからです。** パラメータ数を揃える、最適化器を1つにする、選択手順を1つにする、対応のある統計量を出す、全初期値を報告する、自明なものを含むベースラインを置く。4.3節と5.3節は、材料情報学のグループが行うあらゆる比較 — 量子であれそうでなくとも — に適用できる雛形です。本コースで最も移転可能な技能は、報告された差を見て8項目のどれを満たしていないかを問える能力です。

**集中とplateauの議論を理解する。利用可能な最速のフィルタだからです。** 30量子ビットの忠実度カーネルや、大域読み出しを持つ深く構造のないansatzを含む提案は、何も走らせずに1行で答えられます。会議の場でできると便利なことです。

### 待つ価値があること

  * **SDKの習熟。** フレームワークのAPIは毎年変わり、技能の移転性は低いです。走らせるべき具体的な計算ができたときに1つ学べばよく、その前にではありません。
  * **ハードウェアへのアクセス。** 本コースの何にも必要ありませんでしたし、あったとしても本コースの何も変わりませんでした。デバイス上の時間は、ボトルネックがデバイスであるときに価値があります。
  * **表形式の材料データ向けQMLパイプラインの構築。** 証拠が反対している具体的な活動がこれです。モデルは負け、サロゲートは並び、障害は数学的で、コスト比は5桁です。

### 判断のための表

| 問題が | 関係する問いは | 近い将来の答えは |
| --- | --- | --- |
| データベースの記述子からの物性予測 | 量子モデルは調整済み古典モデルに勝つか | 勝たない。理由はハードウェアではない |
| 相関の強い物質の基底状態エネルギー | 量子計算機はそもそもデータを作れるか | 姉妹コースのVQEとその後継。まだ学習問題ではない |
| 量子センサや量子シミュレータからの学習 | データは量子か、プロセッサは同じ場所にあるか | 真の非対称性がある唯一の方向で、最も未成熟 |
| 量子デバイスの振る舞いからの特性評価 | ノイズは材料について何を語るか | いま実用的で、しかも材料の問題 |
| MLの装いをした組合せ最適化 | 古典ソルバは本当に強いか | ふつう強く、ふつう比較されていない |

### シリーズを1ページで

第1章は4象限を示し、本コースがそのうち1つ — 古典データと量子処理 — にほぼ完全に住むことを予告しました。分野の活動と過剰主張がそこに集中しているからです。

第2章は、量子モデルの表現力が変分部分ではなく符号化から来ることと、再アップロード回路が周波数の台を列挙できる打ち切りFourier級数であることを示しました。その列挙が本章で最も効果的な古典側の攻撃になりました。

第3章は忠実度カーネルを作り、カーネルリッジ回帰につなぎ、指数的集中を測りました。量子ビットが増えるとカーネルは平坦になり、平坦なカーネルは学習を減らします。同章のエンタングル型カーネルは調整済みRBFより1.16倍悪く、古典ベースラインに並んだ唯一の量子カーネルはエンタングルメントを含みませんでした。

第4章は変分回路を同規模の古典ネットワークと、弁護できるよう設計した手順の下で戦わせ、負けました。3モデル中で最小の訓練損失と最大のテスト損失、すなわち学習不足ではなく過学習です。対応のある区間は25パラメータのネットワークに対する負けを解像し、31パラメータとの比較は引き分けと判定します。このデータサイズでの誠実な結果はたいていこの形になります。QML版の barren plateau も測りました。局所読み出しで1量子ビットあたり分散 $2.07\times$ 減衰、大域読み出しで $4.03\times$、エンタングルメントを除くと $1.05\times$。

第5章は古典サロゲートを2つ作り、両方が量子カーネルに並び一方が上回ることを見出し、1つの実験について矛盾しつつ真である3つの要約を示し、議論を変える唯一の前提を特定しました。古典データであることです。

本コースに1つのメッセージがあるとすれば、**誠実な数値が有用な数値である**ということです。持ち帰るべき数値は、雑音の大きい量の1回の引きにすぎない単一分割の1.6倍ではありません。より検出力の高い方 — 対応のある20分割で量子カーネルは全敗し、テストMSEの平均は交差検証したRBFの 0.0242 に対して 0.0843、3.5倍の差 — と、この分割で使える最強のRBFに対する単一分割の3.3倍です。適切に調整された古典カーネルに量子カーネルが約3倍負けることとその理由を知っている材料研究者は、量子機械学習が指数的高速化をもたらすと読んだ研究者よりはるかに良い位置にいます。前者は計画でき、後者は待つことしかできません。そしてその数値を生んだ規律 — 先に手順を固定し、自明なベースラインを含め、全実行を報告し、点ではなく曲線を出す — は、本コースの具体的な結果がすべて置き換わった後も長く働き続ける部分です。

* * *

## 演習

#### 演習1: Haarの予測と実測の比較

$d$ 次元の独立なHaarランダム純粋状態2つについて、忠実度 $|\langle\phi|\phi'\rangle|^2$ はBeta$(1, d-1)$ 分布に従います。

  1. その平均と分散を $d$ で書き下してください。
  2. $n = 2$ から $12$ 量子ビットについて予測される平均と標準偏差を表にし、Code Example 2 の実測値と比べてください。
  3. 実測の平均はHaarの予測を $n=2$ の1.44倍から $n=12$ の87倍まで上回ります。これは符号化について何を語りますか。
  4. この食い違いは実務者にとって集中問題を良くしますか悪くしますか。カーネル成分がBernoulli変数であり $S = k(1-k)/\varepsilon^2$ であることを踏まえ、ショット数で答えてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 平均 \(1/d\)、分散 \((d-1)/\bigl(d^2(d+1)\bigr)\) なので標準偏差は \(\sqrt{d-1}/\bigl(d\sqrt{d+1}\bigr) \approx 1/d\)（大きな \(d\)）。どちらも \(2^{-n}\) でスケールします。</p>

<p><strong>2.</strong> コード中の表を参照。\(n = 12\)、\(d = 4096\) でHaarの平均は \(2.44\times10^{-4}\)、実測は \(2.13\times10^{-2}\) です。</p>

<p><strong>3.</strong> 2層の角度符号化はHaarランダムから程遠いということです。浅く、エンタングルの構造は層あたりCNOTリング1本だけで、符号化された状態は4つの実数でパラメータ化されたHilbert空間の低次元多様体を占めます。したがって状態同士はランダム状態よりはるかに似ており、カーネルはHaarの見積りが許すより多くの構造を保っています。</p>

<p><strong>4.</strong> 良くします。そして改善の大きさは正しく計算する価値があります。カーネル成分はBernoulli変数なので \(S = k(1-k)/\varepsilon^2\) であり、10倍の余裕 \(\varepsilon = \sigma/10\) では \(100\,k(1-k)/\sigma^2\) です。Haar領域では \(k \approx \sigma \approx 1/d\) なので \(S \approx 100\,d\)、すなわち12量子ビットで \(4.1\times10^{5}\)、余裕を落としてショットノイズを広がり全体に合わせるだけなら \(d = 4096\) です。実測の符号化に必要なのは約 \(4\times10^{2}\) で、1000分の1です。\(\sigma\) がHaar値よりはるかに大きいからです。なお誤った式 \(1/\sigma^2\) はHaarの要求を \(d^2 = 1.7\times10^{7}\) と見積もり、さらに \(d\) 倍過大評価します。しかしショットを節約してくれるその同じ事実は、優位性の議論には悪い知らせです。回路が浅いおかげで古典的な構造を保っているカーネルは、小さな古典的記述を持つカーネルであり、それはまさに Code Example 3 と 4 が利用したものです。ショット問題からの脱出と脱量子化からの脱出は逆方向に引き合います。</p>

```python
"""演習1。Code Example 1、2 の続き（同一セッション）。"""
print(f"  {'n':>3}{'d':>7}{'Haar mean':>12}{'Haar std':>12}{'measured mean':>15}"
      f"{'measured std':>14}{'mean ratio':>12}")
for n in NS:
    d = 2 ** n
    hm = 1.0 / d
    hs = np.sqrt((d - 1) / (d ** 2 * (d + 1)))
    m, s = stats[n][0], stats[n][1]
    print(f"  {n:>3}{d:>7}{hm:12.6f}{hs:12.6f}{m:15.6f}{s:14.6f}{m/hm:12.2f}")
# S = k(1-k)/eps^2、eps = sigma/10。カーネル成分は有界な観測量ではなくビットなので
# 分子は 1 ではなく k(1-k) である。
d = 2 ** 12
hm, hs = 1.0 / d, np.sqrt((d - 1) / (d ** 2 * (d + 1)))
mm, ms = stats[12][0], stats[12][1]
print(f"\n  n = 12、eps = sigma/10 でのカーネル成分1つあたりのショット数")
print(f"    Haar:   {100*hm*(1-hm)/hs**2:,.0f}   (~ 100 d = {100*d:,d})")
print(f"    実測:   {100*mm*(1-mm)/ms**2:,.0f}")
print(f"    誤った式 1/sigma^2 では Haar {1/hs**2:,.0f}"
      f"、実測 {1/ms**2:,.0f} となる")
```

```text
    n      d   Haar mean    Haar std  measured mean  measured std  mean ratio
    2      4    0.250000    0.193649       0.359838      0.309325        1.44
    3      8    0.125000    0.110240       0.246025      0.262934        1.97
    4     16    0.062500    0.058709       0.153000      0.191101        2.45
    5     32    0.031250    0.030288       0.109725      0.167800        3.51
    6     64    0.015625    0.015383       0.077986      0.137245        4.99
    7    128    0.007812    0.007752       0.056518      0.119306        7.23
    8    256    0.003906    0.003891       0.044784      0.105990       11.46
    9    512    0.001953    0.001949       0.036139      0.094839       18.50
   10   1024    0.000977    0.000976       0.029538      0.085575       30.25
   11   2048    0.000488    0.000488       0.024784      0.078636       50.76
   12   4096    0.000244    0.000244       0.021315      0.072502       87.31

  n = 12、eps = sigma/10 でのカーネル成分1つあたりのショット数
    Haar:   409,700   (~ 100 d = 409,600)
    実測:   397
    誤った式 1/sigma^2 では Haar 16,785,410、実測 190 となる
```

</details>

#### 演習2: 多項式と指数を数値で

  1. $n$ 量子ビット上で weight が高々 $w$ のPauli文字列の個数を与えてください。
  2. $w = 1, 2, 3$ と $n = 4, 10, 20, 50$ について $4^n$ と並べて表にしてください。
  3. $n = 50$、$w = 2$ で、完全な展開のうちどれだけの割合を保持していますか。
  4. 査読者が「打ち切ったサロゲートは同じカーネルではない」と反論します。Code Example 3 の数値を使って答えてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(\sum_{k=0}^{w}\binom{n}{k}3^k\)。</p>

<p><strong>2-3.</strong> 表を参照。\(n = 50, w = 2\) では 11,176 特徴に対して \(4^{50} = 1.27\times10^{30}\)、割合は \(8.8\times10^{-27}\) です。</p>

<p><strong>4.</strong> 反論は正しく、そして無関係です。打ち切ったサロゲートは実証的に別のカーネルであり、Code Example 3 はその差 — weight 2 で非対角成分の相対誤差67% — を測っています。それでもテスト誤差は厳密なカーネルと等しいか良く（weight 2 で 0.0642 対 0.0680、weight 3 で 0.0379）なっています。優位性の主張とは、固定した手順の下での予測性能についての主張であり、作用素ノルムについての主張ではありません。サロゲートが同じくらいよく予測するなら、両者の距離がいくらであろうと厳密なカーネルは何も買っていません。</p>

```python
"""演習2。NumPy と math.comb のみ。"""
import numpy as np
from math import comb


def npauli(n, w):
    """n量子ビット上で weight が w 以下のPauli文字列の個数。"""
    return sum(comb(n, k) * 3 ** k for k in range(w + 1))


print(f"  {'n':>4}{'w=1':>10}{'w=2':>12}{'w=3':>14}{'4^n':>14}{'w=2 fraction':>16}")
for n in (4, 10, 20, 50):
    total = 4.0 ** n
    print(f"  {n:>4}{npauli(n,1):>10,}{npauli(n,2):>12,}{npauli(n,3):>14,}"
          f"{total:>14.3e}{npauli(n,2)/total:>16.2e}")
```

```text
     n       w=1         w=2           w=3           4^n    w=2 fraction
     4        13          67           175     2.560e+02        2.62e-01
    10        31         436         3,676     1.049e+06        4.16e-04
    20        61       1,771        32,551     1.100e+12        1.61e-09
    50       151      11,176       540,376     1.268e+30        8.82e-27
```

</details>

#### 演習3: cherry-pickingはどれくらいの頻度で成功するか

2つのモデルが真に同等で、分割ごとのテスト誤差の違いがノイズだけだとします。

  1. 両腕が同じ平均と現実的な広がりを持つ帰無モデルから対応のある分割20個を模擬し、$\max_j(\text{classical}_j)/\min_j(\text{quantum}_j)$ — 「自分の最良と相手の最悪」の比 — の分布を求めてください。
  2. その経路で $1.5\times$ 以上の改善を主張できる帰無実験の割合はいくらですか。
  3. 正しい対応のある統計量で繰り返してください。$|t| > 2$ になる帰無実験の割合はいくらですか。
  4. そこから従う報告の規則を述べてください。

<details>
<summary>解答</summary>

<p><strong>1-2.</strong> 分割20個では、最良対最悪の比は帰無実験のほぼすべてで \(1.5\) を超えます。試した最小の広がりで0.998、それより大きい2つでは1.000です。比の中央値は変動係数 0.15、0.30、0.45 に対して 2.18、4.78、10.47 であり、中段の行は Code Example 5 で実測された腕内4.9倍の範囲を再現します。理由は、この比が反対の裾から取った2つの順序統計量を比べており、平均が同一でも分割数とともに期待値が増えるからです。</p>

<p><strong>3.</strong> 対応のある \(t\) 統計量が絶対値で2を超えるのは帰無実験の約5%で、それが設計どおりの振る舞いです。</p>

<p><strong>4.</strong> 全分割にわたる対応のある差とその標準誤差を報告し、走らせた分割数を報告すること。一方の腕の極値と他方の腕の極値を比べては決していけません。論文が最良ケースの数値を報告しているなら、誠実な読者の最初の問いは「ケースはいくつあったのか」です。</p>

```python
"""演習3。NumPyのみ。"""
import numpy as np
rng = np.random.default_rng(21)
trials, splits = 20000, 20
print(f"  帰無モデル: 両腕は同一分布、対応のある分割 {splits} 個、"
      f"試行 {trials:,} 回")
print("  Code Example 5 の量子側は20分割で 0.0375 から 0.1833 まで広がり、")
print("  範囲は4.9倍だった。これを再現するのが下表の cv = 0.30 の行である")
print(f"  {'cv':>6}{'range within one arm':>22}{'median best/worst':>19}"
      f"{'P(ratio>=1.5)':>15}{'P(|t|>2)':>10}")
for cv in (0.15, 0.30, 0.45):
    ratio = np.empty(trials)
    tstat = np.empty(trials)
    within = np.empty(trials)
    for t in range(trials):
        base = rng.lognormal(0.0, cv, splits)        # 分割そのものの難しさ
        a = base * rng.lognormal(0.0, cv, splits)    # 腕A
        b = base * rng.lognormal(0.0, cv, splits)    # 腕B、Aと同一分布
        ratio[t] = b.max() / a.min()                 # Aの最良とBの最悪を比べる
        within[t] = a.max() / a.min()
        d = a - b
        tstat[t] = d.mean() / (d.std(ddof=1) / np.sqrt(splits))
    print(f"  {cv:6.2f}{np.median(within):22.2f}{np.median(ratio):19.2f}"
          f"{np.mean(ratio >= 1.5):15.3f}{np.mean(np.abs(tstat) > 2):10.3f}")
```

```text
  帰無モデル: 両腕は同一分布、対応のある分割 20 個、試行 20,000 回
  Code Example 5 の量子側は20分割で 0.0375 から 0.1833 まで広がり、
  範囲は4.9倍だった。これを再現するのが下表の cv = 0.30 の行である
      cv  range within one arm  median best/worst  P(ratio>=1.5)  P(|t|>2)
    0.15                  2.18               2.18          0.998     0.060
    0.30                  4.79               4.78          1.000     0.058
    0.45                 10.49              10.47          1.000     0.055
```

</details>

#### 演習4: 主張を監査する

ある要旨にこうあります。*「形成エネルギー予測のための量子カーネルを提案する。80個の合金組成のベンチマークにおいて、本手法はRMSE 42 meV/atomを達成し、サポートベクトル回帰（58 meV/atom）を28%上回った。結果は12量子ビットの特徴写像で得られた。」*

  1. この要旨は8項目のチェックリストのうちどれに答えていませんか。
  2. 改善はRMSEで28%と書かれています。MSEに換算し、点あたり二乗誤差の変動係数を1.5と仮定して、その差が $t = 2$ で分解できるテスト点数を見積もってください。
  3. 主張を最も安く決着させる追加実験を1つ挙げてください。
  4. Code Example 2 の集中の測定は12量子ビットの忠実度カーネルについて何を予測し、それは報告された結果について何を意味しますか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> データ量は書かれていますが少なすぎます。ベースラインはSVR1つで調整努力の記述がありません。調整の対等性、試した特徴写像や量子ビット数の個数、分割や乱数種、古典サロゲート、ショット数や実時間について言及がありません。データの出自は古典（合金組成）ですが、それが関係することを要旨は明示していません。8項目のうち6項目が未回答で、答えられている2項目は不利な形で答えられています。</p>

<p><strong>2.</strong> RMSE 42 対 58 meV/atom はMSEで \(1764\) 対 \(3364\)（meV/atom）\(^2\)、MSEの相対差 0.476 です。点あたり二乗誤差の変動係数を \(c = 1.5\) とし対応のある比較なら、\(t \approx \sqrt{N}\,\Delta/(c\,\bar{\mu})\) より \(N \approx (2 \times 1.5/0.476)^2 \approx 40\) テスト点。80点を60/20に分けると際どいものの絶望的ではありませんが、比較が対応付けられており分割が事後に選ばれていないことが条件です。特徴写像にわたる選択を無視しているので、この見積りは楽観的です。</p>

<p><strong>3.</strong> 同じ特徴写像の weight-2 Pauliサロゲートを作り、同じ手順で走らせること。1日の仕事で量子デバイスを要さず、主張を取り除くか大幅に強化するかのどちらかになります。</p>

<p><strong>4.</strong> Code Example 2 は12量子ビットで非対角の標準偏差 0.0725 と、2量子ビットのほぼ4倍悪いテスト誤差を測っています。80点上の12量子ビット忠実度カーネルは集中した領域にあるので、良い結果が報告された最もありそうな説明は、カーネルが実際には全幅で評価されていない（たとえば符号化が少数の実効特徴上の積である）か、比較が1の理由のいずれかで有利になっていたか、です。</p>

```python
"""演習4。NumPyのみ。"""
import numpy as np
rmse_q, rmse_c = 42.0, 58.0
mse_q, mse_c = rmse_q ** 2, rmse_c ** 2
rel = (mse_c - mse_q) / mse_c
print(f"  RMSE {rmse_q:.0f} 対 {rmse_c:.0f} meV/atom  ->  MSE {mse_q:.0f} 対 {mse_c:.0f}")
print(f"  MSEでの相対差: {rel:.3f}")
for c in (1.0, 1.5, 2.0):
    print(f"  変動係数 {c:.1f}  ->  t = 2 で分解するには対応のあるテスト点が N ~ "
          f"{(2*c/rel)**2:6.0f} 必要")
print("  80点を60/20に分けるとテストは20点である")
```

```text
  RMSE 42 対 58 meV/atom  ->  MSE 1764 対 3364
  MSEでの相対差: 0.476
  変動係数 1.0  ->  t = 2 で分解するには対応のあるテスト点が N ~     18 必要
  変動係数 1.5  ->  t = 2 で分解するには対応のあるテスト点が N ~     40 必要
  変動係数 2.0  ->  t = 2 で分解するには対応のあるテスト点が N ~     71 必要
  80点を60/20に分けるとテストは20点である
```

</details>

#### 演習5: 量子データのショット予算

量子センサが実験設定 $i$ でラベル付けされた状態 $\rho_i$ を1回の実行につき1コピー生成します。そうした状態 $N$ 個から性質を学びたいとします。

  1. $N$ 個の状態上の忠実度カーネルについて、各成分に $S$ ショット、各ショットが2状態を1コピーずつ消費するとして、Gram行列は状態準備を何回要しますか。
  2. $N = 100$、$S = 1000$ で評価し、各データ点を記述子ベクトルから必要なだけ読み込める古典の場合と比べてください。
  3. 古典シャドウの手法は代わりに状態あたり単一コピーのランダム化観測量を $M$ 個測ります。コストはどうスケールし、それは何を買いますか。
  4. 最初の実験にどちらの経路を選びますか。その判断は何に依存しますか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(N(N-1)/2\) 成分、各 \(S\) ショット、1ショットあたり2コピーなので \(N(N-1)S\) 回の状態準備です。</p>

<p><strong>2.</strong> \(100 \times 99 \times 1000 = 9.9\times10^{6}\) 回の実験実行です。古典データでは対応する数は0で、記述子ベクトルは必要なだけ無料でメモリから読み直せます。これは入力問題が逆向きに走ったものであり、量子データの設定の代価です。</p>

<p><strong>3.</strong> 合計 \(NM\) 回の準備で、\(N\) について2次ではなく1次です。しかもシャドウ特徴は、実験を再実行せずに下流の任意個数のモデル、カーネル、交差検証の分割に再利用できます。買えるものはまさに5.2節の古典サロゲートであり、量子ビット数に依らない標本複雑度の低weight観測量です。手放すものは低weight観測量でない任意の性質で、そこにこそ証明可能な量子データの分離が住んでいます。</p>

<p><strong>4.</strong> 最初の実験にはシャドウを選びます。理由は2つで、コストが \(N\) について2次ではなく1次であること、そして得られる特徴が1つのモデルのGram行列ではなく再利用可能なデータセットであることです。状態のコヒーレントな処理に移るのは、低weight観測量では証明可能に捉えられない関心量が具体的にあるときだけにします。2次のコストと同一場所の要求が、古典的な後処理では持ち得ないものを買うのはその状況だけだからです。</p>

```python
"""演習5。NumPyのみ。"""
N, S, M = 100, 1000, 500
print(f"  忠実度Gram行列: {N*(N-1)//2:,} 成分 x {S:,} ショット x 状態2コピー")
print(f"    -> 実験 {N*(N-1)*S:,} 回")
print(f"  古典的な記述子: 0 回（ベクトルはメモリから読み直せる）")
print(f"  古典シャドウ: N x M = {N*M:,} 回。下流のどのモデルにも再利用できる")
print(f"  比（コヒーレントなカーネル 対 シャドウ）: {N*(N-1)*S/(N*M):,.0f}x")
```

```text
  忠実度Gram行列: 4,950 成分 x 1,000 ショット x 状態2コピー
    -> 実験 9,900,000 回
  古典的な記述子: 0 回（ベクトルはメモリから読み直せる）
  古典シャドウ: N x M = 50,000 回。下流のどのモデルにも再利用できる
  比（コヒーレントなカーネル 対 シャドウ）: 198x
```

</details>

* * *

## まとめ

### 要点

**1\. 脱量子化は2つの主張であり、研究予算を決めるのは弱い方である**

  * 定理は入力モデルについてのものです。古典アルゴリズムに同じ sample-and-query access を与えると、称賛された指数的高速化のいくつかは消えました。そして低ランク構造も同じくらい効いています。疎で条件のよい線形系は脱量子化されていません。
  * 実務版は実験です。量子的構成が暗黙に記述している古典モデルを作るだけで、コストは1日です。
  * 優位性は差であり、差には2つの側があります。この分野で報告された差の大半は古典側から埋められました。

**2\. 指数的な特徴空間は実在し、そして役に立たない**

  * Haarランダムな忠実度は $2^{-n}$ で集中します。この符号化での実測では非対角の広がりが1量子ビットあたり 0.73〜0.92 倍、2量子ビットの 0.309 から12量子ビットの 0.073 まで落ちました。
  * テスト誤差はレジスタ幅とともに*上がります*。厳密なシミュレータ上で各幅で正則化を再調整して 0.036 から 0.143 です。古典データでは、量子ビットが増えることはモデルが減ることです。
  * ここで爆発しない唯一のものがショットコストです。1成分はBernoulli計数なので10倍の余裕をとって $S = 100\,k(1-k)/\sigma^2$ となり、2量子ビットで241ショット、12量子ビットで397ショット — 横ばいです。$k$ が $\sigma$ より速く落ちるからです。12量子ビットで60点のGram行列全体が約 $7\times10^5$ ショットで買えます。指数的なショットコストは $k \approx \sigma \approx 2^{-n}$ となるHaar領域のものであり、そこでは $S \approx 100\,d$ です。誤った式 $1/\sigma^2$ はそれをレジスタ1つ分早く捏造してしまいます。

**3\. サロゲート2つ、どちらも安く、どちらも十分**

  * 低weight Pauli打ち切り: $O(n^w)$ 特徴で、各特徴は古典シャドウの観測量です。weight 2 ではGram行列の再現は下手（非対角の誤差67%）ですが、テスト誤差は厳密なものより良くなります（0.0642 対 0.0680）。
  * 符号化自身のFourier基底: スペクトルは証明可能に有限（高調波 0、1/2、1 でそれ以上は $10^{-16}$）で、41個の古典特徴が 0.0172 を出します。量子カーネルの4倍良く、調整済みRBFよりも良い値です。総次数自体も交差検証で選ばれており、CV誤差で2が1と3に勝ちます。本章で対応のあるブートストラップが解像できる差はこれだけで、区間は $[-0.113, -0.011]$ です。量子カーネルが調整済みRBFに単一分割で負ける差は解像できず、それが5.3節が20分割を走らせる理由です。
  * どちらのサロゲートもタダ飯ではなく、留保はどちらにも共通です。41個のFourier特徴は $5^4 = 625$ 個からなる帯域制限基底の総次数 $\le2$ の切片であり、基底は $5^d$ で増えます。Pauli特徴は $4^n$ のweight $\le2$ の切片です。どちらもターゲットの構造が低次であるという賭けであり、そうでないターゲットではどちらも失敗します。
  * カーネル忠実度と予測性能は別の量です。優位性の主張が問題にしているのは2番目だけであり、しかもそれは20行で測った2つの推定値の差なので、対応のある区間を伴わなければ何も伴っていないことになります。

**4\. 1つの実験、真である3つの要約**

  * 量子の最良分割対古典の最悪分割で $1.15\times$ の勝ち。標準分割で $1.61\times$ の負け。対応のある20分割では全敗で、平均 0.0843 対 0.0242、3.5倍の差です。
  * その20分割が出す $t = 7.5$ は使えません。訓練集合が重なるため差が相関し、$t$ が膨らみます（Dietterich）。使えるのは符号検定です。20回中20敗は帰無仮説の下で $2^{-20}$ であり、結論はそこに基づきます。
  * 量子側を固定してベースラインだけを変えると、結論は「$2.25\times$ 良い」から「$3.50\times$ 悪い」まで動きます。
  * 絶対差はデータとともに縮みますが比は動かないので、1つの規模での絶対差ではなく比と曲線を出してください。

**5\. 量子データこそが問題の前提である**

  * ここでの否定的な結果はすべて古典データを仮定しており、したがって量子側に請求される載せ込み段階を仮定しています。
  * 状態が直接供給されれば載せ込み段階は消え、厳密な分離が存在します — それは*サンプル複雑度*の分離であり、既知の古典アルゴリズムに対してではなく単一コピー測定のあらゆる戦略に対して成り立つもので、実行時間についての主張ではありません。代価はショットごとに状態を1つ消費すること、実験をプロセッサと同じ場所に置くこと、そして見た目より強い古典シャドウのベースラインに勝つことです。
  * 材料科学における近い将来の実例は狭いものです。量子シミュレーションの出力、量子プローブによる測定、そして材料としての量子デバイスの特性評価です。

**6\. 否定的な結果5章から持ち帰るもの**

  * カーネルの視点と周波数の視点は、量子的な問いで研ぎ澄まされた古典機械学習であり、移転可能です。
  * ベンチマークの規律 — 手順の固定、予算を揃える、自明なベースライン、対応のある統計量、全実行の報告 — が本コースで最も価値ある輸出可能な技能です。
  * 「ハードウェアが良くなれば動く」には答えられます。ここでの否定的な数値はすべて厳密で無雑音のシミュレータから出ています。

**実務への含意**

  * 表形式の材料データに量子モデルを提案する前に、2つのサロゲートを作ってください。どちらかが並べば提案は終わりであり、それを安く終わらせることは成功です。
  * どんな優位性の主張にも問うてください。点数はいくつか、ベースラインはどれだけ強いか、調整はどれだけ対等か、いくつの構成を試したか、データは量子か、サロゲートを検査したか、ショットはいくらか、差はラベルノイズより大きいか。
  * 学習の時間は、表現が誘導する内積、周波数の台、比較の方法論に使ってください。デバイスの時間は、ボトルネックがデバイスであるときだけに使ってください。
  * 量子データの象限を、量子機械学習一般ではなく具体的に注視してください。

[← 第4章: 変分量子回路によるML](<chapter-4.html>) [シリーズトップ →](<index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 本章の比較および監査の演習は、1つの合成60点データセット上で測定、あるいはその周辺で構成したものです。演習4の要旨は説明のための架空の例であり、実在の論文・手法・研究グループを指すものではありません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
