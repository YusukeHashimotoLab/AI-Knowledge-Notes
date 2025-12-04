---
title: 第2章 プロセス監視と品質管理
chapter_title: 第2章 プロセス監視と品質管理
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/food-process-ai/chapter-2.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Food Process Ai](<../../PI/food-process-ai/index.html>)›Chapter 2

## 2.1 リアルタイムプロセス監視

食品製造プロセスのリアルタイム監視は、品質の一貫性と食品安全を確保するために不可欠です。 温度、圧力、流量などの物理量に加え、近赤外分光（NIR）や画像解析による成分・品質データの オンライン取得が可能になっています。AI技術を用いることで、これらの多変量データから 異常の早期検知や品質予測が実現できます。 

### 主要な監視項目

監視項目 | 測定方法 | 管理目的 | AI活用  
---|---|---|---  
温度 | 熱電対、PT100センサー | 殺菌効果、品質維持 | 異常検知、F値予測  
圧力 | 圧力センサー | 装置安全、プロセス制御 | 装置劣化予測  
流量 | 流量計（電磁式、超音波式） | 配合比制御、収量管理 | 流量異常検知  
pH | pHセンサー | 発酵管理、品質制御 | 発酵終点予測  
糖度 | NIR分光、屈折計 | 品質管理、配合制御 | 品質予測  
色 | カラーセンサー、画像解析 | 焼成度、品質評価 | 外観品質判定  
粘度 | 粘度計（回転式、振動式） | 食感制御、配合管理 | 食感予測  
  
📊 コード例1: 多変量プロセス監視システムのシミュレーション
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import seaborn as sns
    
    # リアルタイムプロセスデータのシミュレーション
    np.random.seed(42)
    n_samples = 500
    
    # 正常運転データ（400サンプル）
    normal_data = pd.DataFrame({
        '温度_C': np.random.normal(85, 2, 400),
        '圧力_kPa': np.random.normal(150, 5, 400),
        '流量_L/min': np.random.normal(50, 3, 400),
        'pH': np.random.normal(4.0, 0.2, 400),
        '糖度_Brix': np.random.normal(12, 0.8, 400),
        '粘度_cP': np.random.normal(100, 10, 400),
    })
    normal_data['状態'] = '正常'
    
    # 異常データ（100サンプル）- 複数のパターン
    anomaly1 = pd.DataFrame({  # 温度異常
        '温度_C': np.random.normal(78, 2, 30),
        '圧力_kPa': np.random.normal(150, 5, 30),
        '流量_L/min': np.random.normal(50, 3, 30),
        'pH': np.random.normal(4.0, 0.2, 30),
        '糖度_Brix': np.random.normal(12, 0.8, 30),
        '粘度_cP': np.random.normal(100, 10, 30),
    })
    anomaly1['状態'] = '温度低下'
    
    anomaly2 = pd.DataFrame({  # 圧力異常
        '温度_C': np.random.normal(85, 2, 30),
        '圧力_kPa': np.random.normal(170, 5, 30),
        '流量_L/min': np.random.normal(50, 3, 30),
        'pH': np.random.normal(4.0, 0.2, 30),
        '糖度_Brix': np.random.normal(12, 0.8, 30),
        '粘度_cP': np.random.normal(100, 10, 30),
    })
    anomaly2['状態'] = '圧力上昇'
    
    anomaly3 = pd.DataFrame({  # 多変量異常（品質劣化）
        '温度_C': np.random.normal(83, 2, 40),
        '圧力_kPa': np.random.normal(145, 5, 40),
        '流量_L/min': np.random.normal(55, 3, 40),
        'pH': np.random.normal(4.3, 0.2, 40),
        '糖度_Brix': np.random.normal(13, 0.8, 40),
        '粘度_cP': np.random.normal(110, 10, 40),
    })
    anomaly3['状態'] = '品質劣化'
    
    # データ結合
    data = pd.concat([normal_data, anomaly1, anomaly2, anomaly3], ignore_index=True)
    
    # 特徴量の標準化
    features = ['温度_C', '圧力_kPa', '流量_L/min', 'pH', '糖度_Brix', '粘度_cP']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    
    # PCAによる次元削減（6次元 → 2次元）
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    
    # 可視化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. PCAスコアプロット
    ax1 = fig.add_subplot(gs[0, :])
    colors = {'正常': '#11998e', '温度低下': '#ff6b6b', '圧力上昇': '#ffa500', '品質劣化': '#9b59b6'}
    for state in data['状態'].unique():
        mask = data['状態'] == state
        ax1.scatter(data_pca[mask, 0], data_pca[mask, 1], 
                    label=state, alpha=0.6, s=50, color=colors[state])
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% 説明)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% 説明)', fontsize=12)
    ax1.set_title('主成分分析による異常検知（PCAスコアプロット）', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 寄与率プロット
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(range(len(pca.explained_variance_ratio_)), 
            pca.explained_variance_ratio_, color='#11998e', alpha=0.7)
    ax2.set_xlabel('主成分', fontsize=11)
    ax2.set_ylabel('寄与率', fontsize=11)
    ax2.set_title('主成分の寄与率', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 因子負荷量
    ax3 = fig.add_subplot(gs[1, 1])
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, feature in enumerate(features):
        ax3.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  head_width=0.05, head_length=0.05, fc='#11998e', ec='#11998e')
        ax3.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature,
                 fontsize=10, ha='center', va='center')
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('PC1', fontsize=11)
    ax3.set_ylabel('PC2', fontsize=11)
    ax3.set_title('因子負荷量（変数の寄与）', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4-6. 主要変数の時系列プロット
    time = np.arange(len(data))
    axes = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    plot_vars = ['温度_C', '圧力_kPa']
    for ax, var in zip(axes, plot_vars):
        ax.plot(time[:400], data[var][:400], color='#11998e', alpha=0.7, linewidth=1, label='正常')
        ax.plot(time[400:], data[var][400:], color='#ff6b6b', alpha=0.7, linewidth=1, label='異常')
        ax.set_xlabel('サンプル番号', fontsize=11)
        ax.set_ylabel(var, fontsize=11)
        ax.set_title(f'{var}の時系列変化', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.savefig('multivariate_monitoring.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 統計レポート
    print("=== プロセス監視統計レポート ===")
    print(f"\n総サンプル数: {len(data)}")
    print(f"正常データ: {len(normal_data)} ({len(normal_data)/len(data)*100:.1f}%)")
    print(f"異常データ: {len(data)-len(normal_data)} ({(len(data)-len(normal_data))/len(data)*100:.1f}%)")
    
    print("\n=== 主成分分析結果 ===")
    print(f"累積寄与率（PC1+PC2）: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    print("\n因子負荷量:")
    loadings_df = pd.DataFrame(loadings[:, :2], index=features, columns=['PC1', 'PC2'])
    print(loadings_df.round(3))
    
    print("\n=== 状態別統計 ===")
    for state in data['状態'].unique():
        state_data = data[data['状態'] == state]
        print(f"\n{state}:")
        print(state_data[features].describe().loc[['mean', 'std']].round(2))

## 2.2 官能評価とAI

食品の品質評価において、官能評価（風味、食感、色、香り）は極めて重要です。 しかし、官能評価は主観的で、評価者間のばらつきが大きいという課題があります。 AI技術、特に機械学習とディープラーニングを用いることで、客観的な品質予測や 官能評価データの定量化が可能になります。 

### 🍽️ 官能評価の5つの感覚

  * **味覚** : 甘味、酸味、塩味、苦味、うま味（5基本味）
  * **嗅覚** : 香り成分の複雑な組み合わせ（数千種類）
  * **触覚** : 食感（硬さ、粘性、滑らかさ、クリスピーさ）
  * **視覚** : 色、形、光沢、透明度
  * **聴覚** : 咀嚼音、パリパリ感

📊 コード例2: 官能評価データの多変量解析と品質予測
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    import seaborn as sns
    
    # 官能評価データセットの生成
    np.random.seed(42)
    n_samples = 300
    
    # 理化学分析データ
    sensory_data = pd.DataFrame({
        '糖度_Brix': np.random.uniform(10, 15, n_samples),
        '酸度_%': np.random.uniform(0.3, 0.8, n_samples),
        '水分含量_%': np.random.uniform(80, 90, n_samples),
        '硬度_N': np.random.uniform(5, 15, n_samples),
        '色_L値': np.random.uniform(60, 80, n_samples),
        '香気成分_ppm': np.random.uniform(50, 150, n_samples),
    })
    
    # 官能評価スコア（総合評価）: 複雑な非線形関係をシミュレート
    sensory_data['甘味スコア'] = (
        8 * sensory_data['糖度_Brix'] / 15 +
        np.random.normal(0, 0.5, n_samples)
    ).clip(1, 10)
    
    sensory_data['酸味スコア'] = (
        8 * sensory_data['酸度_%'] / 0.8 +
        np.random.normal(0, 0.5, n_samples)
    ).clip(1, 10)
    
    sensory_data['食感スコア'] = (
        5 + 3 * np.sin(sensory_data['硬度_N'] / 5) +
        np.random.normal(0, 0.5, n_samples)
    ).clip(1, 10)
    
    sensory_data['香りスコア'] = (
        4 + 4 * np.log(sensory_data['香気成分_ppm'] / 50) +
        np.random.normal(0, 0.5, n_samples)
    ).clip(1, 10)
    
    # 総合評価（各スコアの加重平均）
    sensory_data['総合評価'] = (
        0.3 * sensory_data['甘味スコア'] +
        0.2 * sensory_data['酸味スコア'] +
        0.3 * sensory_data['食感スコア'] +
        0.2 * sensory_data['香りスコア'] +
        np.random.normal(0, 0.3, n_samples)
    ).clip(1, 10)
    
    # 特徴量と目的変数
    X = sensory_data[['糖度_Brix', '酸度_%', '水分含量_%', '硬度_N', '色_L値', '香気成分_ppm']]
    y = sensory_data['総合評価']
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # モデル構築
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 予測と評価
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # 可視化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. 予測vs実測値
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, y_pred, alpha=0.6, s=60, color='#11998e', edgecolors='white', linewidth=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='理想ライン')
    ax1.set_xlabel('実測値（官能評価）', fontsize=12)
    ax1.set_ylabel('予測値（AIモデル）', fontsize=12)
    ax1.set_title(f'官能評価予測モデル\n(R²={r2:.3f}, RMSE={rmse:.3f})', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 残差プロット
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=60, color='#38ef7d', edgecolors='white', linewidth=0.5)
    ax2.axhline(0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('予測値', fontsize=12)
    ax2.set_ylabel('残差（実測値 - 予測値）', fontsize=12)
    ax2.set_title('残差プロット', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 特徴量重要度
    ax3 = fig.add_subplot(gs[0, 2])
    feature_importance = pd.DataFrame({
        '特徴量': X.columns,
        '重要度': model.feature_importances_
    }).sort_values('重要度', ascending=True)
    
    ax3.barh(feature_importance['特徴量'], feature_importance['重要度'], color='#11998e')
    ax3.set_xlabel('重要度', fontsize=12)
    ax3.set_title('特徴量重要度', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. 相関行列ヒートマップ
    ax4 = fig.add_subplot(gs[1, :])
    corr_data = sensory_data[['糖度_Brix', '酸度_%', '硬度_N', '香気成分_ppm', 
                               '甘味スコア', '酸味スコア', '食感スコア', '香りスコア', '総合評価']]
    correlation_matrix = corr_data.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={'label': '相関係数'}, ax=ax4)
    ax4.set_title('理化学分析値と官能評価スコアの相関', fontsize=13, fontweight='bold')
    
    plt.savefig('sensory_evaluation_ai.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # レポート出力
    print("=== 官能評価予測モデル性能 ===")
    print(f"決定係数 R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"交差検証 R² (平均±標準偏差): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    print("\n=== 特徴量重要度ランキング ===")
    feature_ranking = pd.DataFrame({
        '特徴量': X.columns,
        '重要度': model.feature_importances_
    }).sort_values('重要度', ascending=False)
    print(feature_ranking.to_string(index=False))
    
    print("\n=== 官能評価スコア統計 ===")
    print(sensory_data[['甘味スコア', '酸味スコア', '食感スコア', '香りスコア', '総合評価']].describe())

## 2.3 画像解析による外観品質評価

食品の外観品質（色、形状、表面状態）は、消費者の購買意欲に直結する重要な要素です。 従来は人の目視検査に依存していましたが、ディープラーニング（特にCNN: Convolutional Neural Network）を 用いた画像解析により、客観的かつ高速な品質判定が可能になっています。 

📊 コード例3: 画像特徴量抽出と品質分類（シミュレーション）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # 画像特徴量データのシミュレーション（実際にはCNNで抽出）
    np.random.seed(42)
    n_samples_per_class = 200
    
    # 3つの品質クラス: A品（優良）、B品（標準）、C品（不良）
    def generate_class_data(mean_color, std_color, mean_shape, std_shape, mean_texture, std_texture, label):
        return pd.DataFrame({
            '平均色_R': np.random.normal(mean_color[0], std_color, n_samples_per_class),
            '平均色_G': np.random.normal(mean_color[1], std_color, n_samples_per_class),
            '平均色_B': np.random.normal(mean_color[2], std_color, n_samples_per_class),
            '形状_円形度': np.random.normal(mean_shape, std_shape, n_samples_per_class),
            '表面粗さ': np.random.normal(mean_texture, std_texture, n_samples_per_class),
            'サイズ_mm2': np.random.normal(100, 10, n_samples_per_class),
            '品質クラス': label
        })
    
    # A品: 明るい色、高い円形度、滑らかな表面
    class_A = generate_class_data(
        mean_color=[180, 150, 120], std_color=10,
        mean_shape=0.85, std_shape=0.05,
        mean_texture=5, std_texture=2,
        label='A品（優良）'
    )
    
    # B品: 中間的な特徴
    class_B = generate_class_data(
        mean_color=[160, 130, 100], std_color=15,
        mean_shape=0.75, std_shape=0.08,
        mean_texture=10, std_texture=3,
        label='B品（標準）'
    )
    
    # C品: 暗い色、低い円形度、粗い表面
    class_C = generate_class_data(
        mean_color=[140, 110, 80], std_color=20,
        mean_shape=0.60, std_shape=0.10,
        mean_texture=18, std_texture=5,
        label='C品（不良）'
    )
    
    # データ結合
    image_data = pd.concat([class_A, class_B, class_C], ignore_index=True)
    
    # 特徴量と目的変数
    X = image_data.drop('品質クラス', axis=1)
    y = image_data['品質クラス']
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 分類モデル構築
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 予測と評価
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred, labels=['A品（優良）', 'B品（標準）', 'C品（不良）'])
    
    # 可視化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. 混同行列
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['A品', 'B品', 'C品'],
                yticklabels=['A品', 'B品', 'C品'],
                ax=ax1, cbar_kws={'label': 'サンプル数'})
    ax1.set_xlabel('予測クラス', fontsize=12)
    ax1.set_ylabel('実際のクラス', fontsize=12)
    ax1.set_title(f'混同行列\n(精度={accuracy:.3f})', fontsize=13, fontweight='bold')
    
    # 2. 特徴量重要度
    ax2 = fig.add_subplot(gs[0, 1])
    feature_importance = pd.DataFrame({
        '特徴量': X.columns,
        '重要度': clf.feature_importances_
    }).sort_values('重要度', ascending=True)
    
    ax2.barh(feature_importance['特徴量'], feature_importance['重要度'], color='#11998e')
    ax2.set_xlabel('重要度', fontsize=12)
    ax2.set_title('画像特徴量の重要度', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. クラス別の色分布
    ax3 = fig.add_subplot(gs[0, 2])
    colors_map = {'A品（優良）': '#4caf50', 'B品（標準）': '#ff9800', 'C品（不良）': '#f44336'}
    for label in ['A品（優良）', 'B品（標準）', 'C品（不良）']:
        class_data = image_data[image_data['品質クラス'] == label]
        ax3.scatter(class_data['平均色_R'], class_data['平均色_G'], 
                    label=label, alpha=0.6, s=40, color=colors_map[label])
    ax3.set_xlabel('平均色 R値', fontsize=12)
    ax3.set_ylabel('平均色 G値', fontsize=12)
    ax3.set_title('色空間での品質クラス分布', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 形状と表面粗さの関係
    ax4 = fig.add_subplot(gs[1, 0])
    for label in ['A品（優良）', 'B品（標準）', 'C品（不良）']:
        class_data = image_data[image_data['品質クラス'] == label]
        ax4.scatter(class_data['形状_円形度'], class_data['表面粗さ'], 
                    label=label, alpha=0.6, s=40, color=colors_map[label])
    ax4.set_xlabel('形状円形度', fontsize=12)
    ax4.set_ylabel('表面粗さ', fontsize=12)
    ax4.set_title('形状と表面粗さの関係', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. クラス別特徴量分布（バイオリンプロット）
    ax5 = fig.add_subplot(gs[1, 1:])
    plot_data = image_data[['形状_円形度', '表面粗さ', '品質クラス']].melt(
        id_vars='品質クラス', var_name='特徴量', value_name='値')
    
    sns.violinplot(data=plot_data, x='特徴量', y='値', hue='品質クラス',
                   palette={'A品（優良）': '#4caf50', 'B品（標準）': '#ff9800', 'C品（不良）': '#f44336'},
                   split=False, ax=ax5)
    ax5.set_title('品質クラス別の特徴量分布', fontsize=13, fontweight='bold')
    ax5.legend(title='品質クラス', fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('image_quality_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # レポート出力
    print("=== 画像品質分類モデル性能 ===")
    print(f"精度（Accuracy）: {accuracy:.4f}")
    print(f"交差検証精度 (平均±標準偏差): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    print("\n=== 詳細分類レポート ===")
    print(classification_report(y_test, y_pred))
    
    print("\n=== クラス別統計 ===")
    for label in ['A品（優良）', 'B品（標準）', 'C品（不良）']:
        class_data = image_data[image_data['品質クラス'] == label]
        print(f"\n{label}:")
        print(class_data[['平均色_R', '形状_円形度', '表面粗さ']].describe().loc[['mean', 'std']].round(2))

## 2.4 統計的品質管理（SQC）

統計的品質管理（Statistical Quality Control: SQC）は、プロセスの変動を統計的手法で監視・管理する手法です。 管理図（Control Chart）を用いて、プロセスが管理状態にあるかを判定し、異常の早期検知を行います。 

📊 コード例4: X-R管理図とプロセス能力指数
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # 製造ロットデータの生成（サンプルサイズ n=5 のサブグループ）
    np.random.seed(42)
    n_subgroups = 30
    subgroup_size = 5
    target = 100  # 目標値
    spec_lower = 95  # 規格下限
    spec_upper = 105  # 規格上限
    
    # 正常プロセス（20サブグループ）+ 異常プロセス（10サブグループ、平均シフト）
    normal_data = np.random.normal(target, 2, (20, subgroup_size))
    abnormal_data = np.random.normal(target + 3, 2, (10, subgroup_size))
    data = np.vstack([normal_data, abnormal_data])
    
    # X-bar（平均値）とR（範囲）の計算
    xbar = data.mean(axis=1)
    R = data.max(axis=1) - data.min(axis=1)
    
    # 管理限界線の計算
    xbar_grand = xbar.mean()
    R_bar = R.mean()
    
    # 管理図係数（n=5の場合）
    A2 = 0.577  # X-bar管理図係数
    D3 = 0      # R管理図下限係数
    D4 = 2.114  # R管理図上限係数
    
    # X-bar管理図の管理限界
    UCL_xbar = xbar_grand + A2 * R_bar
    LCL_xbar = xbar_grand - A2 * R_bar
    
    # R管理図の管理限界
    UCL_R = D4 * R_bar
    LCL_R = D3 * R_bar
    
    # プロセス能力指数の計算
    sigma_hat = R_bar / 1.693  # d2=2.326 for n=5, sigma = R_bar / d2
    Cp = (spec_upper - spec_lower) / (6 * sigma_hat)
    Cpk = min((spec_upper - xbar_grand), (xbar_grand - spec_lower)) / (3 * sigma_hat)
    
    # 可視化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. X-bar管理図
    ax1.plot(range(1, n_subgroups+1), xbar, marker='o', color='#11998e', linewidth=2, markersize=6)
    ax1.axhline(xbar_grand, color='green', linestyle='-', linewidth=2, label='中心線（CL）')
    ax1.axhline(UCL_xbar, color='red', linestyle='--', linewidth=2, label='上方管理限界（UCL）')
    ax1.axhline(LCL_xbar, color='red', linestyle='--', linewidth=2, label='下方管理限界（LCL）')
    ax1.axhline(spec_upper, color='orange', linestyle=':', linewidth=2, label='規格上限（USL）')
    ax1.axhline(spec_lower, color='orange', linestyle=':', linewidth=2, label='規格下限（LSL）')
    
    # 異常点の検出とハイライト
    out_of_control = (xbar > UCL_xbar) | (xbar < LCL_xbar)
    if out_of_control.any():
        ax1.scatter(np.where(out_of_control)[0] + 1, xbar[out_of_control], 
                    color='red', s=150, marker='x', linewidths=3, zorder=5, label='異常点')
    
    ax1.set_xlabel('サブグループ番号', fontsize=12)
    ax1.set_ylabel('サブグループ平均値 (X̄)', fontsize=12)
    ax1.set_title('X̄管理図（平均値管理図）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. R管理図
    ax2.plot(range(1, n_subgroups+1), R, marker='s', color='#38ef7d', linewidth=2, markersize=6)
    ax2.axhline(R_bar, color='green', linestyle='-', linewidth=2, label='中心線（CL）')
    ax2.axhline(UCL_R, color='red', linestyle='--', linewidth=2, label='上方管理限界（UCL）')
    if LCL_R > 0:
        ax2.axhline(LCL_R, color='red', linestyle='--', linewidth=2, label='下方管理限界（LCL）')
    
    # 異常点の検出
    out_of_control_R = (R > UCL_R) | (R < LCL_R)
    if out_of_control_R.any():
        ax2.scatter(np.where(out_of_control_R)[0] + 1, R[out_of_control_R], 
                    color='red', s=150, marker='x', linewidths=3, zorder=5, label='異常点')
    
    ax2.set_xlabel('サブグループ番号', fontsize=12)
    ax2.set_ylabel('範囲 (R)', fontsize=12)
    ax2.set_title('R管理図（範囲管理図）', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. ヒストグラムとプロセス能力
    all_data = data.flatten()
    ax3.hist(all_data, bins=30, density=True, alpha=0.7, color='#11998e', edgecolor='white')
    ax3.axvline(target, color='green', linestyle='-', linewidth=2, label='目標値')
    ax3.axvline(spec_upper, color='red', linestyle='--', linewidth=2, label='規格上限（USL）')
    ax3.axvline(spec_lower, color='red', linestyle='--', linewidth=2, label='規格下限（LSL）')
    
    # 正規分布フィット
    from scipy.stats import norm
    mu, sigma = all_data.mean(), all_data.std()
    x = np.linspace(all_data.min(), all_data.max(), 100)
    ax3.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='正規分布フィット')
    
    ax3.set_xlabel('測定値', fontsize=12)
    ax3.set_ylabel('密度', fontsize=12)
    ax3.set_title(f'ヒストグラムとプロセス能力\nCp={Cp:.3f}, Cpk={Cpk:.3f}', 
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 規格外率の推定
    ax4.axis('off')
    report_text = f"""
    === 統計的品質管理（SQC）レポート ===
    
    【管理図分析結果】
    • サブグループ数: {n_subgroups}
    • サブグループサイズ: {subgroup_size}
    • 総サンプル数: {n_subgroups * subgroup_size}
    
    【X̄管理図（平均値管理図）】
    • 中心線（CL）: {xbar_grand:.2f}
    • 上方管理限界（UCL）: {UCL_xbar:.2f}
    • 下方管理限界（LCL）: {LCL_xbar:.2f}
    • 管理外れ点数: {out_of_control.sum()} / {n_subgroups} サブグループ
    
    【R管理図（範囲管理図）】
    • 中心線（R̄）: {R_bar:.2f}
    • 上方管理限界（UCL）: {UCL_R:.2f}
    • 管理外れ点数: {out_of_control_R.sum()} / {n_subgroups} サブグループ
    
    【プロセス能力指数】
    • Cp（工程能力指数）: {Cp:.3f}
    • Cpk（偏心を考慮した能力指数）: {Cpk:.3f}
    • 推定標準偏差（σ̂）: {sigma_hat:.3f}
    
    【判定】
    • プロセス能力: {"適正" if Cp >= 1.33 else "改善必要" if Cp >= 1.0 else "不適正"}
    • 偏心状態: {"良好" if Cpk >= 1.33 else "要注意" if Cpk >= 1.0 else "不良"}
    
    【規格外率推定】
    • 規格上限超過率: {(all_data > spec_upper).sum() / len(all_data) * 100:.2f}%
    • 規格下限未満率: {(all_data < spec_lower).sum() / len(all_data) * 100:.2f}%
    • 総規格外率: {((all_data > spec_upper) | (all_data < spec_lower)).sum() / len(all_data) * 100:.2f}%
    """
    
    ax4.text(0.1, 0.9, report_text, fontsize=11, verticalalignment='top', 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('sqc_control_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(report_text)

### ⚠️ 品質管理実装時の注意点

  * **サブグループの選定** : 短時間内の連続サンプルをサブグループ化（合理的サブグループ）
  * **管理限界の再計算** : プロセス改善後は管理限界を再設定
  * **異常パターンの判定** : 点の管理外れ以外に、連・トレンド・周期性も検知
  * **Cp/Cpk の目標値** : 一般に Cp ≥ 1.33, Cpk ≥ 1.33 を目標
  * **規格と管理限界の違い** : 規格は顧客要求、管理限界はプロセス能力に基づく

## まとめ

本章では、食品プロセスの監視と品質管理のAI技術を学びました：

  * 多変量プロセス監視とPCAによる異常検知
  * 官能評価データの定量化と機械学習による品質予測
  * 画像解析による外観品質の自動分類
  * 統計的品質管理（X-R管理図、プロセス能力指数）

次章では、プロセス最適化とベイズ最適化の実践的手法を学びます。

[← 第1章: 食品プロセスとAIの基礎](<chapter-1.html>) [第3章: プロセス最適化 →](<chapter-3.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
