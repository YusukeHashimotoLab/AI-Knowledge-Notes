---
title: 第1章：PIの基礎概念とプロセス産業におけるデータ活用
chapter_title: 第1章：PIの基礎概念とプロセス産業におけるデータ活用
subtitle: プロセス産業のデジタル変革の基礎
---

# 第1章：PIの基礎概念とプロセス産業におけるデータ活用

プロセス・インフォマティクス（PI）の基本概念を理解し、プロセス産業における特徴とデータの種類を学びます。データ駆動型アプローチによる実際の改善事例を通じて、PIの価値を実感しましょう。

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ プロセス・インフォマティクス（PI）の定義と目的を説明できる
  * ✅ プロセス産業の特徴と従来の材料開発との違いを理解する
  * ✅ プロセスデータの主要な種類（センサー、操作、品質データ）を分類できる
  * ✅ データ駆動型プロセス改善の具体的な事例を説明できる
  * ✅ Pythonを使った基本的なプロセスデータ可視化ができる

* * *

## 1.1 プロセス・インフォマティクス（PI）とは

### PIの定義

**プロセス・インフォマティクス（Process Informatics; PI）** は、化学プラント、製薬、食品、半導体などのプロセス産業において、**データ駆動型アプローチ** を活用して、プロセスの理解、最適化、制御を行う学問分野です。

具体的には、以下の活動を含みます：

  * **プロセスデータの収集と分析** : センサーデータ、操作条件、品質データの収集と解析
  * **プロセスモデリング** : 機械学習やソフトセンサーによる品質予測モデルの構築
  * **プロセス最適化** : 収率向上、エネルギー削減、品質改善のための条件最適化
  * **プロセス制御** : リアルタイムデータに基づく自動制御システムの構築
  * **異常検知** : データ分析による異常状態の早期発見

### Materials Informatics（MI）との違い

PIとよく混同される概念に、Materials Informatics（MI）があります。両者の違いを明確に理解しましょう。

項目 | Materials Informatics (MI) | Process Informatics (PI)  
---|---|---  
**対象** | 材料そのもの（組成、構造、特性） | 製造プロセス（運転条件、制御、品質）  
**目的** | 新材料の発見・設計 | プロセスの最適化・制御  
**データの種類** | 物性値、結晶構造、組成データ | 時系列センサーデータ、操作条件  
**時間軸** | 静的（材料の固有特性） | 動的（時々刻々と変化）  
**主要手法** | 記述子、材料データベース、スクリーニング | 時系列分析、ソフトセンサー、プロセス制御  
**典型的な課題** | バンドギャップ2.5 eVの材料を探索 | 製品純度98%以上を維持しながらエネルギー消費を10%削減  
  
**重要なポイント** : MIは「何を作るか」に焦点を当て、PIは「どう作るか」に焦点を当てます。両者は補完関係にあり、MIで発見した新材料をPIで効率的に製造する、というように連携します。

* * *

## 1.2 プロセス産業の特徴

PIを理解するには、まずプロセス産業の特徴を知る必要があります。

### プロセスの分類

プロセス産業のプロセスは、大きく2つに分類されます：

#### 1\. 連続プロセス (Continuous Process)

  * **特徴** : 24時間365日、原料を連続的に投入し、製品を連続的に取り出す
  * **例** : 石油精製、化学プラント（エチレン製造など）、製紙
  * **利点** : 高い生産性、安定した品質
  * **課題** : 停止・再起動のコストが高い、柔軟性が低い

#### 2\. バッチプロセス (Batch Process)

  * **特徴** : 原料を投入 → 反応 → 製品取り出しを繰り返す
  * **例** : 医薬品製造、食品加工、ファインケミカル
  * **利点** : 柔軟な製品切り替え、少量多品種生産に適す
  * **課題** : バッチ間のばらつき、生産性が連続プロセスより低い

### プロセス産業の主要分野

産業分野 | 主なプロセス | PIの典型的応用  
---|---|---  
**化学** | 蒸留、反応、分離 | 収率最適化、エネルギー削減  
**石油化学** | 精製、クラッキング | 製品品質予測、装置異常検知  
**製薬** | 合成、晶析、乾燥 | バッチ品質管理、プロセス開発加速  
**食品** | 発酵、殺菌、混合 | 品質一貫性向上、賞味期限予測  
**半導体** | CVD、エッチング、洗浄 | 歩留まり向上、リアルタイム制御  
  
### プロセスの複雑性

プロセス産業のプロセスは、以下の特徴により非常に複雑です：

  1. **多変数性**
     * 数十〜数百の変数が相互に影響
     * 例: 蒸留塔では温度（複数段）、圧力、流量、組成などが連動
  2. **非線形性**
     * 入力と出力の関係が線形でない
     * 例: 反応温度を10°C上げると収率が5%増えるが、さらに10°C上げると副反応が起こり収率が減少
  3. **時間遅れ（タイムラグ）**
     * 操作変更の効果が現れるまで時間がかかる
     * 例: 蒸留塔のリボイラー温度を変更しても、塔頂製品への影響は数分〜数十分後
  4. **外乱の存在**
     * 原料組成の変動、環境温度の変化など
     * これらをリアルタイムで補正する必要

**結論** : この複雑性ゆえに、**データ駆動型アプローチ（PI）** が有効なのです。従来の経験則や単純なモデルでは対応しきれない複雑な関係性を、機械学習やデータ分析で捉えることができます。

* * *

## 1.3 プロセスデータの種類

PIで扱うデータは、主に3つのカテゴリに分類されます。

### 1\. センサーデータ（測定データ）

プロセスの状態をリアルタイムで測定するデータです。

**主要なセンサータイプ** :

センサータイプ | 測定対象 | 典型的な測定頻度 | 用途例  
---|---|---|---  
**温度センサー** | プロセス温度 | 1秒〜1分 | 反応温度監視、加熱炉制御  
**圧力センサー** | プロセス圧力 | 1秒〜1分 | 蒸留塔圧力、反応器圧力  
**流量計** | 流体流量 | 1秒〜1分 | 原料供給量、製品取り出し量  
**液面計** | タンク液面 | 10秒〜1分 | 反応器液面、貯蔵タンク  
**濃度計** | 成分濃度 | 1分〜1時間 | 製品純度、反応進行度  
**pH計** | pH値 | 1秒〜1分 | 化学反応制御  
  
**特徴** :

  * 高頻度（秒〜分単位）で大量のデータが生成される
  * 時系列データとして蓄積
  * 欠損値、外れ値、ノイズが含まれることが多い

### 2\. 操作条件データ（設定値・制御データ）

プロセスを制御するために人間またはDCS（Distributed Control System）が設定する値です。

**主要な操作変数** :

  * **設定温度** : 反応器温度設定値、加熱炉設定温度
  * **設定圧力** : 蒸留塔圧力設定値
  * **流量設定値** : 原料供給速度、冷却水流量
  * **バルブ開度** : 制御弁の開度（0-100%）
  * **攪拌速度** : 反応器の攪拌機回転数

**特徴** :

  * センサーデータと連動（設定値と実測値の比較）
  * PIの目的の1つは、最適な操作条件を見つけること

### 3\. 品質データ（製品特性・分析データ）

製品の品質を評価するデータです。多くの場合、オフライン分析により得られます。

**主要な品質指標** :

品質指標 | 測定方法 | 測定頻度 | 重要性  
---|---|---|---  
**製品純度** | ガスクロマトグラフィー（GC） | 1時間〜1日 | 製品規格の主要項目  
**収率** | 物質収支計算 | バッチごと | 経済性の指標  
**粘度** | 粘度計 | 数時間〜1日 | 製品品質  
**色** | 分光光度計 | 数時間〜1日 | 製品外観  
**不純物含有量** | HPLC、GC-MS | 1日〜1週間 | 品質規格、安全性  
  
**特徴** :

  * 測定頻度が低い（時間〜日単位）
  * 測定に時間とコストがかかる
  * **PIの重要な課題** : センサーデータからリアルタイムに品質を予測する「ソフトセンサー」の構築

### 4\. イベントデータ（補助的データ）

プロセスで発生する各種イベントの記録です。

  * **アラーム** : 異常状態の警告（温度異常、圧力異常など）
  * **運転日誌** : オペレーターによる手動記録
  * **設備保全記録** : メンテナンス履歴
  * **バッチ記録** : バッチごとの開始・終了時刻、使用原料情報

* * *

## 1.4 データ駆動型プロセス改善の事例

理論だけでなく、実際にPIがどのように活用されているのか、具体的な事例を見てみましょう。

### ケーススタディ1: 化学プラントの収率向上（5%改善）

**背景** :

ある化学プラントでは、原料AとBから製品Cを製造していました。理論的な収率は95%ですが、実際の収率は平均85%にとどまっていました。従来は、ベテランオペレーターの経験に基づき、反応温度と圧力を調整していましたが、なぜ収率が変動するのか明確ではありませんでした。

**PIアプローチ** :

  1. **データ収集（1ヶ月）**
     * 過去2年分の運転データ（温度、圧力、流量、原料組成）を収集
     * 同期間の製品品質データ（収率、純度）を収集
     * データポイント数: 約100万点
  2. **探索的データ分析（EDA）**
     * 変数間の相関分析
     * 発見: 反応温度だけでなく、**原料Aの供給速度と反応器攪拌速度の比** が収率に強く影響
     * これは従来知られていなかった関係
  3. **機械学習モデル構築**
     * Random Forestで収率予測モデルを構築
     * 予測精度: R² = 0.82
     * 特徴量重要度分析により、重要な操作変数を特定
  4. **最適化と実装**
     * モデルに基づき、最適な操作条件を探索
     * 最適条件: 反応温度175°C、原料A流量2.5 m³/h、攪拌速度300 rpm
     * 実プラントで試験運転を実施

**結果** :

  * **収率向上** : 85% → 90%（**+5%** ）
  * **経済効果** : 年間約5億円の利益増加（製品価値換算）
  * **副次効果** : エネルギー消費も3%削減

### ケーススタディ2: エネルギー消費削減（15%削減）

**背景** :

石油化学プラントの蒸留塔では、製品を精製するために大量の蒸気を使用していました。エネルギーコストは年間10億円以上で、経営上の大きな課題でした。

**PIアプローチ** :

  1. **エネルギー消費分析**
     * 蒸留塔のリボイラー熱量データと製品品質データを収集
     * 発見: 必要以上に高い還流比で運転されている時間帯が多い
     * 製品純度99.5%が目標だが、実際は99.8%程度で過剰品質
  2. **ソフトセンサー構築**
     * 目的: 塔頂製品純度をリアルタイムで予測
     * 従来: 1日1回のGC分析でしか純度を測定できない
     * PLSモデルで温度プロファイルから純度を予測
     * 予測精度: R² = 0.88
  3. **最適化制御**
     * ソフトセンサーの予測値を用いて、還流比を動的に調整
     * 製品純度99.5-99.6%を維持しながら、エネルギー使用量を最小化

**結果** :

  * **エネルギー削減** : **15%削減**
  * **経済効果** : 年間約1.5億円のコスト削減
  * **CO₂削減** : 年間約5,000トンのCO₂排出削減
  * **投資回収期間** : 約1年（ソフトセンサー開発・実装コストを含む）

### ROI（投資対効果）分析

PIプロジェクトの典型的なROIを見てみましょう。

項目 | ケーススタディ1（収率向上） | ケーススタディ2（エネルギー削減）  
---|---|---  
**初期投資** | データ基盤整備: 2,000万円  
モデル開発: 1,000万円 | ソフトセンサー開発: 1,500万円  
制御システム改修: 3,000万円  
**年間効果** | 5億円/年（利益増） | 1.5億円/年（コスト削減）  
**ROI** | 初年度 **1,567%** | 初年度 **233%**  
**投資回収期間** | **約2ヶ月** | **約4ヶ月**  
  
**結論** : PIプロジェクトは、適切に実施すれば**極めて高いROI** を達成できます。

* * *

## 1.5 Pythonによるプロセスデータ可視化入門

ここから、実際にPythonを使ってプロセスデータを可視化してみましょう。以下の5つのコード例を通じて、PIの基礎スキルを習得します。

### 環境準備

まず、必要なライブラリをインストールします。
    
    
    # 必要なライブラリのインストール
    pip install pandas matplotlib seaborn numpy plotly
    

### コード例1: 時系列センサーデータの可視化

化学反応器の温度データを時系列プロットします。
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # サンプルデータ生成: 反応器温度の1日分のデータ（1分間隔）
    np.random.seed(42)
    time_points = pd.date_range('2025-01-01 00:00', periods=1440, freq='1min')
    temperature = 175 + np.random.normal(0, 2, 1440) + \
                  5 * np.sin(np.linspace(0, 4*np.pi, 1440))  # 温度変動をシミュレート
    
    # DataFrameに格納
    df = pd.DataFrame({
        'timestamp': time_points,
        'temperature': temperature
    })
    
    # 可視化
    plt.figure(figsize=(14, 5))
    plt.plot(df['timestamp'], df['temperature'], linewidth=0.8, color='#11998e')
    plt.axhline(y=175, color='red', linestyle='--', label='Target: 175°C')
    plt.fill_between(df['timestamp'], 173, 177, alpha=0.2, color='green', label='Acceptable range')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('Reactor Temperature - 24 Hour Trend', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 統計量の計算
    print(f"平均温度: {df['temperature'].mean():.2f}°C")
    print(f"標準偏差: {df['temperature'].std():.2f}°C")
    print(f"最高温度: {df['temperature'].max():.2f}°C")
    print(f"最低温度: {df['temperature'].min():.2f}°C")
    

**出力例** :
    
    
    平均温度: 174.98°C
    標準偏差: 3.45°C
    最高温度: 183.12°C
    最低温度: 166.54°C
    

**解説** : この例では、反応器温度の時系列データをプロットしています。目標温度（175°C）と許容範囲（173-177°C）を表示することで、プロセスの安定性を視覚的に確認できます。

### コード例2: 複数センサーの同時プロット

蒸留塔の温度、圧力、流量を同時に表示します。
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # サンプルデータ生成
    np.random.seed(42)
    time_points = pd.date_range('2025-01-01 00:00', periods=1440, freq='1min')
    
    df = pd.DataFrame({
        'timestamp': time_points,
        'temperature': 85 + np.random.normal(0, 1.5, 1440),
        'pressure': 1.2 + np.random.normal(0, 0.05, 1440),
        'flow_rate': 50 + np.random.normal(0, 3, 1440)
    })
    
    # 3つのサブプロットで表示
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 温度
    axes[0].plot(df['timestamp'], df['temperature'], color='#11998e', linewidth=0.8)
    axes[0].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0].set_title('Distillation Column - Multi-Sensor Data', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 圧力
    axes[1].plot(df['timestamp'], df['pressure'], color='#f59e0b', linewidth=0.8)
    axes[1].set_ylabel('Pressure (MPa)', fontsize=11)
    axes[1].grid(alpha=0.3)
    
    # 流量
    axes[2].plot(df['timestamp'], df['flow_rate'], color='#7b2cbf', linewidth=0.8)
    axes[2].set_ylabel('Flow Rate (m³/h)', fontsize=11)
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 変数間の相関
    print("変数間の相関係数:")
    print(df[['temperature', 'pressure', 'flow_rate']].corr())
    

**解説** : 複数のセンサーデータを同じ時間軸で表示することで、変数間の関係や異常パターンを視覚的に把握できます。相関係数により、変数間の定量的な関係も確認できます。

### コード例3: 相関マトリックスのヒートマップ

多変数間の相関関係を一目で把握します。
    
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # サンプルデータ生成: 蒸留塔の8変数
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'Feed_Temp': np.random.normal(60, 5, n),
        'Reflux_Ratio': np.random.uniform(1.5, 3.5, n),
        'Reboiler_Duty': np.random.normal(1500, 200, n),
        'Top_Temp': np.random.normal(85, 3, n),
        'Bottom_Temp': np.random.normal(155, 5, n),
        'Pressure': np.random.normal(1.2, 0.1, n),
        'Purity': np.random.uniform(95, 99.5, n),
        'Yield': np.random.uniform(85, 95, n)
    })
    
    # 相関関係に基づくデータ調整（リアルな相関を作成）
    df['Top_Temp'] = df['Top_Temp'] + 0.3 * df['Reflux_Ratio']
    df['Purity'] = df['Purity'] + 0.5 * df['Reflux_Ratio'] - 0.2 * df['Top_Temp']
    df['Yield'] = df['Yield'] + 0.3 * df['Reboiler_Duty'] / 100
    
    # 相関マトリックスの計算
    corr = df.corr()
    
    # ヒートマップの作成
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Distillation Column Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 強い相関を持つペアを表示
    print("\n強い相関を持つ変数ペア (|r| > 0.5):")
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.5:
                print(f"{corr.columns[i]} vs {corr.columns[j]}: {corr.iloc[i, j]:.3f}")
    

**解説** : 相関マトリックスのヒートマップにより、どの変数同士が強く関連しているかを視覚的に把握できます。これはモデリングの前処理や特徴量選択に有用です。

### コード例4: 散布図マトリックス（ペアプロット）

変数間の関係を散布図で詳細に観察します。
    
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # サンプルデータ生成
    np.random.seed(42)
    n = 300
    
    df = pd.DataFrame({
        'Temperature': np.random.normal(175, 5, n),
        'Pressure': np.random.normal(1.5, 0.2, n),
        'Flow_Rate': np.random.normal(50, 5, n),
        'Yield': np.random.uniform(80, 95, n)
    })
    
    # 関係性の追加（現実的な相関）
    df['Yield'] = df['Yield'] + 0.5 * (df['Temperature'] - 175) + 2 * (df['Pressure'] - 1.5)
    
    # ペアプロットの作成
    sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.6, 'color': '#11998e'},
                 diag_kws={'color': '#11998e'})
    plt.suptitle('Pairplot - Process Variables vs Yield', y=1.01, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    

**解説** : ペアプロットにより、全変数ペアの散布図と各変数の分布を一度に確認できます。非線形な関係や外れ値の検出に有効です。

### コード例5: インタラクティブな可視化（Plotly）

Webブラウザでズームやホバー表示が可能なインタラクティブなグラフを作成します。
    
    
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # サンプルデータ生成
    np.random.seed(42)
    time_points = pd.date_range('2025-01-01', periods=1440, freq='1min')
    
    df = pd.DataFrame({
        'timestamp': time_points,
        'temperature': 175 + np.random.normal(0, 2, 1440),
        'pressure': 1.5 + np.random.normal(0, 0.1, 1440),
        'yield': 90 + np.random.normal(0, 3, 1440)
    })
    
    # サブプロット作成
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Reactor Temperature', 'Reactor Pressure', 'Yield'),
        vertical_spacing=0.12
    )
    
    # 温度
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['temperature'],
                   mode='lines', name='Temperature',
                   line=dict(color='#11998e', width=1.5)),
        row=1, col=1
    )
    
    # 圧力
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['pressure'],
                   mode='lines', name='Pressure',
                   line=dict(color='#f59e0b', width=1.5)),
        row=2, col=1
    )
    
    # 収率
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['yield'],
                   mode='lines', name='Yield',
                   line=dict(color='#7b2cbf', width=1.5)),
        row=3, col=1
    )
    
    # レイアウト設定
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Temp (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure (MPa)", row=2, col=1)
    fig.update_yaxes(title_text="Yield (%)", row=3, col=1)
    
    fig.update_layout(
        title_text="Interactive Process Data Visualization",
        height=900,
        showlegend=False
    )
    
    # グラフを表示（Jupyter Notebookの場合）
    fig.show()
    
    # HTMLファイルとして保存
    fig.write_html("process_data_interactive.html")
    print("インタラクティブなグラフを 'process_data_interactive.html' に保存しました。")
    

**解説** : Plotlyを使うと、ズーム、パン、ホバー表示などのインタラクティブ機能を持つグラフを作成できます。大量のデータを探索する際に非常に便利です。

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **プロセス・インフォマティクス（PI）の定義**
     * データ駆動型アプローチによるプロセスの理解、最適化、制御
     * MIが「何を作るか」、PIが「どう作るか」に焦点
  2. **プロセス産業の特徴**
     * 連続プロセス vs バッチプロセス
     * 多変数性、非線形性、時間遅れの複雑性
  3. **プロセスデータの3つの主要カテゴリ**
     * センサーデータ: 温度、圧力、流量（高頻度）
     * 操作条件データ: 設定値、制御パラメータ
     * 品質データ: 純度、収率（低頻度・オフライン）
  4. **データ駆動型改善の実例**
     * 収率5%向上 → 年間5億円の利益増
     * エネルギー15%削減 → 年間1.5億円のコスト削減
     * 高いROI（数百〜数千%）と短い投資回収期間（2-4ヶ月）
  5. **Pythonによるデータ可視化の基礎**
     * 時系列プロット、複数変数の同時表示
     * 相関マトリックス、ペアプロット
     * インタラクティブな可視化（Plotly）

### 重要なポイント

  * PIは**実用的で即効性が高い** 技術（数ヶ月でROI達成）
  * プロセスの複雑性ゆえに、データ駆動型アプローチが有効
  * データの可視化は、PIの第一歩であり最も重要なステップ
  * ソフトセンサー（リアルタイム品質予測）がPIの核心技術の1つ

### 次の章へ

第2章では、**プロセスデータの前処理と可視化** を詳しく学びます：

  * 時系列データの扱い方（リサンプリング、ローリング統計）
  * 欠損値処理・外れ値検出の実践手法
  * データスケーリングと正規化
  * 高度な可視化テクニック
  * プロセスデータ特有の課題への対処法
