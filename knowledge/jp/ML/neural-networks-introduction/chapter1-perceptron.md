---
title: 第1章：パーセプトロンの基礎
chapter_title: 第1章：パーセプトロンの基礎
subtitle: ニューラルネットワークの原点 - 最も単純な学習モデル
reading_time: 20-25分
difficulty: 入門
code_examples: 9
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ パーセプトロンの構造と動作原理を理解する
  * ✅ 重み（weight）とバイアス（bias）の役割を説明できる
  * ✅ 論理ゲート（AND、OR、NAND）をPythonで実装できる
  * ✅ 線形分離可能性の概念を理解する
  * ✅ XOR問題から多層化の必要性を学ぶ

* * *

## 1.1 パーセプトロンとは

### 歴史的背景

**パーセプトロン（Perceptron）** は、1957年にフランク・ローゼンブラット（Frank Rosenblatt）によって考案されました。これは人間の脳の神経細胞（ニューロン）を模倣した最初の機械学習アルゴリズムです。

> 「パーセプトロンは、複数の信号を入力として受け取り、1つの信号を出力します。入力信号に重みを掛けて合計し、閾値を超えたら発火（出力1）する仕組みです。」

### パーセプトロンの構造
    
    
    ```mermaid
    graph LR
        x1[入力 x1] -->|重み w1| sum[Σ 総和]
        x2[入力 x2] -->|重み w2| sum
        b[バイアス b] --> sum
        sum --> activation[活性化関数]
        activation --> y[出力 y]
    
        style x1 fill:#e3f2fd
        style x2 fill:#e3f2fd
        style sum fill:#fff3e0
        style activation fill:#f3e5f5
        style y fill:#e8f5e9
    ```

**数式表現** ：

$$ y = \begin{cases} 1 & \text{if } w_1x_1 + w_2x_2 + b > 0 \\\ 0 & \text{otherwise} \end{cases} $$

または、ステップ関数を使って：

$$ y = h(w_1x_1 + w_2x_2 + b) $$

ここで、$h(x)$はヘビサイド関数（Heaviside function）：

$$ h(x) = \begin{cases} 1 & \text{if } x > 0 \\\ 0 & \text{otherwise} \end{cases} $$

### 構成要素の説明

要素 | 記号 | 意味  
---|---|---  
**入力** | $x_1, x_2, \ldots, x_n$ | パーセプトロンへの入力信号  
**重み** | $w_1, w_2, \ldots, w_n$ | 各入力の重要度（調整可能なパラメータ）  
**バイアス** | $b$ | 発火のしやすさ（閾値の調整）  
**出力** | $y$ | 0 または 1（二値分類）  
  
* * *

## 1.2 論理ゲートの実装

### ANDゲート

**真理値表** ：

x1 | x2 | y  
---|---|---  
0 | 0 | 0  
0 | 1 | 0  
1 | 0 | 0  
1 | 1 | 1  
  
**Python実装** ：
    
    
    import numpy as np
    
    def AND(x1, x2):
        """
        ANDゲートの実装
        重み: w1=0.5, w2=0.5
        バイアス: b=-0.7
        """
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.7
    
        # 総和を計算
        tmp = np.sum(w * x) + b
    
        # 活性化関数（ステップ関数）
        if tmp > 0:
            return 1
        else:
            return 0
    
    # テスト
    print("=== ANDゲート ===")
    print(f"AND(0, 0) = {AND(0, 0)}")  # 0
    print(f"AND(0, 1) = {AND(0, 1)}")  # 0
    print(f"AND(1, 0) = {AND(1, 0)}")  # 0
    print(f"AND(1, 1) = {AND(1, 1)}")  # 1
    

**出力** ：
    
    
    === ANDゲート ===
    AND(0, 0) = 0
    AND(0, 1) = 0
    AND(1, 0) = 0
    AND(1, 1) = 1
    

### ORゲート

**真理値表** ：

x1 | x2 | y  
---|---|---  
0 | 0 | 0  
0 | 1 | 1  
1 | 0 | 1  
1 | 1 | 1  
      
    
    def OR(x1, x2):
        """
        ORゲートの実装
        重み: w1=0.5, w2=0.5
        バイアス: b=-0.2
        """
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.2  # ANDより発火しやすい
    
        tmp = np.sum(w * x) + b
    
        if tmp > 0:
            return 1
        else:
            return 0
    
    # テスト
    print("\n=== ORゲート ===")
    print(f"OR(0, 0) = {OR(0, 0)}")  # 0
    print(f"OR(0, 1) = {OR(0, 1)}")  # 1
    print(f"OR(1, 0) = {OR(1, 0)}")  # 1
    print(f"OR(1, 1) = {OR(1, 1)}")  # 1
    

### NANDゲート

**NAND（NOT AND）** は、ANDの出力を反転させたものです。

x1 | x2 | y  
---|---|---  
0 | 0 | 1  
0 | 1 | 1  
1 | 0 | 1  
1 | 1 | 0  
      
    
    def NAND(x1, x2):
        """
        NANDゲートの実装
        重み: w1=-0.5, w2=-0.5（負の重み）
        バイアス: b=0.7
        """
        x = np.array([x1, x2])
        w = np.array([-0.5, -0.5])  # 負の重み
        b = 0.7
    
        tmp = np.sum(w * x) + b
    
        if tmp > 0:
            return 1
        else:
            return 0
    
    # テスト
    print("\n=== NANDゲート ===")
    print(f"NAND(0, 0) = {NAND(0, 0)}")  # 1
    print(f"NAND(0, 1) = {NAND(0, 1)}")  # 1
    print(f"NAND(1, 0) = {NAND(1, 0)}")  # 1
    print(f"NAND(1, 1) = {NAND(1, 1)}")  # 0
    

### 汎用パーセプトロンクラス
    
    
    class Perceptron:
        """汎用パーセプトロンクラス"""
    
        def __init__(self, weights, bias):
            """
            Args:
                weights: 重みのnumpy配列
                bias: バイアス値
            """
            self.w = np.array(weights)
            self.b = bias
    
        def forward(self, x):
            """
            順伝播（forward propagation）
    
            Args:
                x: 入力値の配列
    
            Returns:
                0 または 1
            """
            tmp = np.sum(self.w * x) + self.b
            return 1 if tmp > 0 else 0
    
        def __call__(self, *inputs):
            """呼び出し可能にする"""
            x = np.array(inputs)
            return self.forward(x)
    
    # パーセプトロンで論理ゲートを定義
    and_gate = Perceptron(weights=[0.5, 0.5], bias=-0.7)
    or_gate = Perceptron(weights=[0.5, 0.5], bias=-0.2)
    nand_gate = Perceptron(weights=[-0.5, -0.5], bias=0.7)
    
    # テスト
    print("\n=== 汎用パーセプトロン ===")
    print(f"AND(1, 1) = {and_gate(1, 1)}")    # 1
    print(f"OR(0, 1) = {or_gate(0, 1)}")      # 1
    print(f"NAND(1, 1) = {nand_gate(1, 1)}")  # 0
    

* * *

## 1.3 重みとバイアスの役割

### 重み（Weight）の意味

重みは**入力の重要度** を表します。

  * **大きい重み** : その入力が重要
  * **小さい重み** : その入力は重要でない
  * **負の重み** : その入力が出力を抑制

    
    
    import matplotlib.pyplot as plt
    
    # 重みを変化させたときの出力
    def visualize_weight_effect():
        """重みの効果を可視化"""
        weights = np.linspace(-2, 2, 100)
        x1, x2 = 1, 1
        b = -0.7
    
        outputs = []
        for w in weights:
            tmp = w * x1 + w * x2 + b
            y = 1 if tmp > 0 else 0
            outputs.append(y)
    
        plt.figure(figsize=(10, 4))
        plt.plot(weights, outputs, linewidth=2)
        plt.xlabel('Weight (w1 = w2 = w)', fontsize=12)
        plt.ylabel('Output', fontsize=12)
        plt.title('重みの変化とパーセプトロン出力 (x1=1, x2=1, b=-0.7)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=0.35, color='g', linestyle='--', alpha=0.5, label='閾値')
        plt.legend()
        plt.show()
    
    visualize_weight_effect()
    

### バイアス（Bias）の意味

バイアスは**発火のしやすさ** を調整します。

  * **大きいバイアス** : 発火しやすい（出力1になりやすい）
  * **小さいバイアス** : 発火しにくい（出力0になりやすい）

    
    
    def compare_bias():
        """バイアスの違いを比較"""
        x1, x2 = 0.5, 0.5
        w1, w2 = 0.5, 0.5
    
        biases = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
        print("=== バイアスの効果 ===")
        print(f"入力: x1={x1}, x2={x2}")
        print(f"重み: w1={w1}, w2={w2}")
        print()
    
        for b in biases:
            tmp = w1*x1 + w2*x2 + b
            y = 1 if tmp > 0 else 0
            print(f"バイアス b={b:5.1f} → 総和={tmp:5.2f} → 出力={y}")
    
    compare_bias()
    

**出力** ：
    
    
    === バイアスの効果 ===
    入力: x1=0.5, x2=0.5
    重み: w1=0.5, w2=0.5
    
    バイアス b= -1.0 → 総和=-0.50 → 出力=0
    バイアス b= -0.5 → 総和= 0.00 → 出力=0
    バイアス b=  0.0 → 総和= 0.50 → 出力=1
    バイアス b=  0.5 → 総和= 1.00 → 出力=1
    バイアス b=  1.0 → 総和= 1.50 → 出力=1
    

* * *

## 1.4 線形分離可能性

### 概念の説明

**線形分離可能（Linearly Separable）** とは、データを**1本の直線（2次元）または平面（高次元）で分離できる** ことを意味します。

パーセプトロンが学習できるのは、**線形分離可能な問題のみ** です。
    
    
    ```mermaid
    graph LR
        A[線形分離可能] --> B[ANDゲート]
        A --> C[ORゲート]
        A --> D[NANDゲート]
        E[線形分離不可能] --> F[XORゲート]
    
        style A fill:#e8f5e9
        style E fill:#ffebee
    ```

### ANDゲートの決定境界
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot_decision_boundary_AND():
        """ANDゲートの決定境界を可視化"""
        # データ点
        x1 = np.array([0, 0, 1, 1])
        x2 = np.array([0, 1, 0, 1])
        y = np.array([0, 0, 0, 1])  # ANDの出力
    
        # プロット
        plt.figure(figsize=(8, 6))
    
        # クラス0（出力0）
        plt.scatter(x1[y==0], x2[y==0], s=200, c='blue', marker='o',
                    label='出力 = 0', edgecolors='k', linewidths=2)
    
        # クラス1（出力1）
        plt.scatter(x1[y==1], x2[y==1], s=200, c='red', marker='s',
                    label='出力 = 1', edgecolors='k', linewidths=2)
    
        # 決定境界: w1*x1 + w2*x2 + b = 0
        # 0.5*x1 + 0.5*x2 - 0.7 = 0
        # x2 = -x1 + 1.4
        x_line = np.linspace(-0.5, 1.5, 100)
        y_line = -x_line + 1.4
        plt.plot(x_line, y_line, 'g--', linewidth=2, label='決定境界')
    
        # 領域の塗りつぶし
        plt.fill_between(x_line, y_line, 2, alpha=0.2, color='red', label='出力=1の領域')
        plt.fill_between(x_line, -1, y_line, alpha=0.2, color='blue', label='出力=0の領域')
    
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.xlabel('x1', fontsize=14)
        plt.ylabel('x2', fontsize=14)
        plt.title('ANDゲートの決定境界', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='upper right')
        plt.show()
    
    plot_decision_boundary_AND()
    

### ORゲートの決定境界
    
    
    def plot_decision_boundary_OR():
        """ORゲートの決定境界を可視化"""
        # データ点
        x1 = np.array([0, 0, 1, 1])
        x2 = np.array([0, 1, 0, 1])
        y = np.array([0, 1, 1, 1])  # ORの出力
    
        plt.figure(figsize=(8, 6))
    
        # クラス0
        plt.scatter(x1[y==0], x2[y==0], s=200, c='blue', marker='o',
                    label='出力 = 0', edgecolors='k', linewidths=2)
    
        # クラス1
        plt.scatter(x1[y==1], x2[y==1], s=200, c='red', marker='s',
                    label='出力 = 1', edgecolors='k', linewidths=2)
    
        # 決定境界: 0.5*x1 + 0.5*x2 - 0.2 = 0
        # x2 = -x1 + 0.4
        x_line = np.linspace(-0.5, 1.5, 100)
        y_line = -x_line + 0.4
        plt.plot(x_line, y_line, 'g--', linewidth=2, label='決定境界')
    
        plt.fill_between(x_line, y_line, 2, alpha=0.2, color='red')
        plt.fill_between(x_line, -1, y_line, alpha=0.2, color='blue')
    
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.xlabel('x1', fontsize=14)
        plt.ylabel('x2', fontsize=14)
        plt.title('ORゲートの決定境界', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='upper right')
        plt.show()
    
    plot_decision_boundary_OR()
    

* * *

## 1.5 XOR問題 - パーセプトロンの限界

### XORゲートとは

**XOR（Exclusive OR、排他的論理和）** は、「どちらか一方だけが1のときに1を出力」する論理演算です。

x1 | x2 | y  
---|---|---  
0 | 0 | 0  
0 | 1 | 1  
1 | 0 | 1  
1 | 1 | 0  
  
### 単層パーセプトロンでは実現できない
    
    
    def plot_XOR_problem():
        """XOR問題の可視化 - 線形分離不可能"""
        x1 = np.array([0, 0, 1, 1])
        x2 = np.array([0, 1, 0, 1])
        y = np.array([0, 1, 1, 0])  # XORの出力
    
        plt.figure(figsize=(8, 6))
    
        # クラス0
        plt.scatter(x1[y==0], x2[y==0], s=200, c='blue', marker='o',
                    label='出力 = 0', edgecolors='k', linewidths=2)
    
        # クラス1
        plt.scatter(x1[y==1], x2[y==1], s=200, c='red', marker='s',
                    label='出力 = 1', edgecolors='k', linewidths=2)
    
        # 線形分離不可能を示す
        plt.text(0.5, 1.3, '1本の直線では\n分離できない！',
                 fontsize=14, ha='center',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.xlabel('x1', fontsize=14)
        plt.ylabel('x2', fontsize=14)
        plt.title('XOR問題 - 線形分離不可能', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='upper right')
        plt.show()
    
    plot_XOR_problem()
    

### 多層パーセプトロンによる解決

XOR問題は、**複数のパーセプトロンを組み合わせる** ことで解決できます。
    
    
    ```mermaid
    graph LR
        x1[x1] --> nand[NAND]
        x2[x2] --> nand
        x1 --> or[OR]
        x2 --> or
        nand --> and[AND]
        or --> and
        and --> y[出力 y]
    
        style x1 fill:#e3f2fd
        style x2 fill:#e3f2fd
        style nand fill:#fff3e0
        style or fill:#fff3e0
        style and fill:#f3e5f5
        style y fill:#e8f5e9
    ```
    
    
    def XOR(x1, x2):
        """
        XORゲートの実装
        NAND、OR、ANDを組み合わせる
        """
        s1 = NAND(x1, x2)
        s2 = OR(x1, x2)
        y = AND(s1, s2)
        return y
    
    # テスト
    print("\n=== XORゲート（多層パーセプトロン）===")
    print(f"XOR(0, 0) = {XOR(0, 0)}")  # 0
    print(f"XOR(0, 1) = {XOR(0, 1)}")  # 1
    print(f"XOR(1, 0) = {XOR(1, 0)}")  # 1
    print(f"XOR(1, 1) = {XOR(1, 1)}")  # 0
    

**出力** ：
    
    
    === XORゲート（多層パーセプトロン）===
    XOR(0, 0) = 0
    XOR(0, 1) = 1
    XOR(1, 0) = 1
    XOR(1, 1) = 0
    

### 多層化による表現力の向上

> **重要な洞察** : パーセプトロンを**多層化** することで、線形分離不可能な問題も解けるようになります。これが**ニューラルネットワーク** の本質です。

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **パーセプトロンの構造**

     * 入力、重み、バイアス、活性化関数、出力
     * 数式: $y = h(w_1x_1 + w_2x_2 + b)$
  2. **論理ゲートの実装**

     * AND、OR、NANDは単層パーセプトロンで実現可能
     * 重みとバイアスを適切に設定することで実装
  3. **重みとバイアスの役割**

     * 重み: 入力の重要度
     * バイアス: 発火のしやすさ
  4. **線形分離可能性**

     * パーセプトロンは線形分離可能な問題のみ解ける
     * 決定境界は直線（2D）または超平面（高次元）
  5. **XOR問題と多層化**

     * XORは線形分離不可能
     * 多層パーセプトロンで解決可能
     * これがディープラーニングへの道

### 重要なポイント

概念 | 説明  
---|---  
**パーセプトロン** | 最も単純なニューラルネットワーク  
**重み** | 調整可能なパラメータ、学習の対象  
**バイアス** | 閾値の調整、発火しやすさ  
**活性化関数** | ステップ関数（ヘビサイド関数）  
**線形分離可能性** | 単層パーセプトロンの限界  
**多層化** | 表現力の向上、非線形問題を解決  
  
### 次の章へ

第2章では、**多層パーセプトロン（MLP）と誤差逆伝播法** を学びます：

  * 多層ネットワークの構造
  * 誤差逆伝播法（Backpropagation）
  * 勾配降下法による学習
  * NumPyによる完全実装

* * *

## 演習問題

### 問題1（難易度：easy）

以下の文章の正誤を判定してください。

  1. パーセプトロンは重みとバイアスを持つ
  2. ANDゲートは単層パーセプトロンで実装できる
  3. XORゲートは単層パーセプトロンで実装できる
  4. バイアスが大きいほど発火しにくい

解答例

**解答** ：

  1. **正** \- パーセプトロンの基本構造
  2. **正** \- 線形分離可能なため実装可能
  3. **誤** \- XORは線形分離不可能、多層化が必要
  4. **誤** \- バイアスが大きいほど発火**しやすい**

### 問題2（難易度：medium）

NORゲート（NOT OR）を実装してください。真理値表は以下の通りです：

x1 | x2 | y  
---|---|---  
0 | 0 | 1  
0 | 1 | 0  
1 | 0 | 0  
1 | 1 | 0  
ヒント

  * ORの出力を反転させる
  * 負の重みを使用する
  * 適切なバイアスを設定する

解答例
    
    
    def NOR(x1, x2):
        """
        NORゲートの実装
        重み: w1=-0.5, w2=-0.5
        バイアス: b=0.2
        """
        x = np.array([x1, x2])
        w = np.array([-0.5, -0.5])
        b = 0.2
    
        tmp = np.sum(w * x) + b
    
        if tmp > 0:
            return 1
        else:
            return 0
    
    # テスト
    print("=== NORゲート ===")
    for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
        y = NOR(x1, x2)
        print(f"NOR({x1}, {x2}) = {y}")
    

**出力** ：
    
    
    === NORゲート ===
    NOR(0, 0) = 1
    NOR(0, 1) = 0
    NOR(1, 0) = 0
    NOR(1, 1) = 0
    

### 問題3（難易度：medium）

3入力ANDゲートを実装してください。出力は、3つの入力がすべて1のときのみ1になります。

解答例
    
    
    def AND3(x1, x2, x3):
        """
        3入力ANDゲートの実装
        重み: w1=0.5, w2=0.5, w3=0.5
        バイアス: b=-1.2
        """
        x = np.array([x1, x2, x3])
        w = np.array([0.5, 0.5, 0.5])
        b = -1.2  # 3つの入力の合計が1.5になるときのみ発火
    
        tmp = np.sum(w * x) + b
    
        if tmp > 0:
            return 1
        else:
            return 0
    
    # テスト
    print("=== 3入力ANDゲート ===")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            for x3 in [0, 1]:
                y = AND3(x1, x2, x3)
                print(f"AND3({x1}, {x2}, {x3}) = {y}")
    

**出力** ：
    
    
    === 3入力ANDゲート ===
    AND3(0, 0, 0) = 0
    AND3(0, 0, 1) = 0
    AND3(0, 1, 0) = 0
    AND3(0, 1, 1) = 0
    AND3(1, 0, 0) = 0
    AND3(1, 0, 1) = 0
    AND3(1, 1, 0) = 0
    AND3(1, 1, 1) = 1
    

### 問題4（難易度：hard）

XNOR（XORの否定）ゲートを多層パーセプトロンで実装してください。真理値表：

x1 | x2 | y  
---|---|---  
0 | 0 | 1  
0 | 1 | 0  
1 | 0 | 0  
1 | 1 | 1  
ヒント

  * XORゲートの出力を反転させる
  * OR、NAND、NANDの組み合わせも可能

解答例
    
    
    def XNOR_v1(x1, x2):
        """
        XNORゲート（方法1）: XORの出力を反転
        """
        xor_out = XOR(x1, x2)
        # 反転するにはNOTゲート（NANDで実現可能）
        return 1 if xor_out == 0 else 0
    
    def XNOR_v2(x1, x2):
        """
        XNORゲート（方法2）: OR、NAND、NANDの組み合わせ
        """
        s1 = OR(x1, x2)
        s2 = NAND(x1, x2)
        y = NAND(s1, s2)
        return y
    
    # テスト
    print("=== XNORゲート ===")
    print("方法1（XOR + NOT）:")
    for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
        y = XNOR_v1(x1, x2)
        print(f"XNOR({x1}, {x2}) = {y}")
    
    print("\n方法2（OR + NAND + NAND）:")
    for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
        y = XNOR_v2(x1, x2)
        print(f"XNOR({x1}, {x2}) = {y}")
    

### 問題5（難易度：hard）

パーセプトロンを使って簡単な分類問題を解いてください。以下のデータを正しく分類する重みとバイアスを見つけてください：

  * クラス0: (0, 0), (0, 1)
  * クラス1: (1, 0), (1, 1)

解答例
    
    
    def custom_classifier(x1, x2):
        """
        カスタム分類器
        x1が重要な特徴
        """
        w1 = 1.0  # x1を重視
        w2 = 0.0  # x2は無視
        b = -0.5
    
        tmp = w1*x1 + w2*x2 + b
        return 1 if tmp > 0 else 0
    
    # テスト
    print("=== カスタム分類器 ===")
    data = [
        ((0, 0), 0),
        ((0, 1), 0),
        ((1, 0), 1),
        ((1, 1), 1)
    ]
    
    correct = 0
    for (x1, x2), expected in data:
        pred = custom_classifier(x1, x2)
        result = "✓" if pred == expected else "✗"
        print(f"入力({x1}, {x2}) → 予測={pred}, 正解={expected} {result}")
        if pred == expected:
            correct += 1
    
    print(f"\n精度: {correct}/{len(data)} = {100*correct/len(data):.1f}%")
    

**解説** ：

  * この問題では、x1の値だけで分類可能
  * x1=0 → クラス0、x1=1 → クラス1
  * 従って、w1を大きく、w2を小さく（または0に）設定

* * *

## 参考文献

  1. Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain." _Psychological Review_ , 65(6), 386-408.
  2. Minsky, M., & Papert, S. (1969). _Perceptrons: An Introduction to Computational Geometry_. MIT Press.
  3. 斎藤康毅 (2016). 『ゼロから作るDeep Learning』オライリージャパン.
