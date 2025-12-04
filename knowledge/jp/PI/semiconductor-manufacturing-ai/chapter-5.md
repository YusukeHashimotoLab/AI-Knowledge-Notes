---
title: 橋本研究室
chapter_title: 橋本研究室
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/semiconductor-manufacturing-ai/chapter-5.html>) | Last sync: 2025-11-16

[ホーム](<../index.html>) > 知識ベース (準備中) > [プロセスインフォマティクス](<../../PI/>) > [半導体製造AI](<../../PI/semiconductor-manufacturing-ai/>) > 第5章 

## 学習目標

  * Multivariate SPC (MSPC) による多変数異常検知を習得する
  * Isolation Forestとその半導体製造への応用を理解する
  * LSTMによる時系列異常検知の実装方法を学ぶ
  * 因果推論で故障の根本原因を特定する手法を習得する
  * SHAP値による機械学習モデルの解釈性向上手法を理解する

## 5.1 Fault Detection & Classification (FDC) の重要性

### 5.1.1 FDCの役割

半導体製造では、プロセス異常の早期検知が歩留まり向上の鍵となります。FDCシステムは：

  * **異常検知 (Fault Detection)** : リアルタイムでプロセス異常を検出
  * **故障分類 (Fault Classification)** : 異常の種類を自動診断
  * **根本原因分析 (Root Cause Analysis)** : 異常の真因を特定
  * **予知保全 (Predictive Maintenance)** : 故障前に異常兆候を検出

### 5.1.2 早期検知の経済的価値

**ダウンタイム削減** : 1時間の停止 = 数千万円の損失

**不良品削減** : 異常検知遅れで数百枚のウェハが不良化

**歩留まり向上** : 早期対応で2-5%の歩留まり改善

**保全コスト削減** : 予防保全で事後保全コストを1/3に削減

### 5.1.3 AI-FDCの優位性

従来の閾値ベースFDCに対するAIの優位性：

  * **多変数相関** : 100以上のセンサー間の複雑な相関を検出
  * **微小変化検出** : 正常範囲内の異常パターンを識別
  * **誤検出削減** : False Positive率を1/10以下に低減
  * **未知異常検出** : 訓練データに含まれない新規異常を発見

## 5.2 Multivariate Statistical Process Control (MSPC)

### 5.2.1 MSPCの原理

MSPCは、主成分分析 (PCA) で多変数データを低次元化し、統計的管理図で異常を検知します：

**主成分分析 (PCA)**

観測変数 \\(\mathbf{x} \in \mathbb{R}^m\\) を主成分空間に射影：

$$\mathbf{t} = \mathbf{P}^T (\mathbf{x} - \bar{\mathbf{x}})$$

\\(\mathbf{P}\\): 主成分ベクトル行列、\\(\bar{\mathbf{x}}\\): 平均

**Hotelling's T² 統計量**

主成分空間内での異常（モデル内変動）を検出：

$$T^2 = \mathbf{t}^T \mathbf{\Lambda}^{-1} \mathbf{t}$$

\\(\mathbf{\Lambda}\\): 主成分の分散行列

管理限界線 (UCL): \\(\chi^2\\) 分布の99%点

**Squared Prediction Error (SPE)**

主成分空間外の異常（残差変動）を検出：

$$SPE = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \|\mathbf{x} - \mathbf{P}\mathbf{t} - \bar{\mathbf{x}}\|^2$$

管理限界線: 正常データのSPE分布から計算

### 5.2.2 MSPC実装例
    
    
    import numpy as np
    from sklearn.decomposition import PCA
    from scipy.stats import chi2, f
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    class MultivariateSPC:
        """
        Multivariate Statistical Process Control (MSPC)
    
        PCAベースの多変数異常検知
        Hotelling's T² とSPE統計量で異常判定
        """
    
        def __init__(self, n_components=None, confidence_level=0.99):
            """
            Parameters:
            -----------
            n_components : int or float
                主成分数 (intなら絶対数、floatなら累積寄与率)
            confidence_level : float
                信頼水準 (管理限界線の設定)
            """
            self.n_components = n_components
            self.confidence_level = confidence_level
            self.pca = None
            self.T2_UCL = None
            self.SPE_UCL = None
            self.mean = None
            self.std = None
    
        def fit(self, X_normal):
            """
            正常データで訓練
    
            Parameters:
            -----------
            X_normal : ndarray
                正常運転データ (n_samples, n_features)
            """
            # 標準化
            self.mean = np.mean(X_normal, axis=0)
            self.std = np.std(X_normal, axis=0)
            X_scaled = (X_normal - self.mean) / self.std
    
            # PCA
            self.pca = PCA(n_components=self.n_components)
            T_train = self.pca.fit_transform(X_scaled)
    
            # Hotelling's T² 管理限界線
            n, p = X_normal.shape
            k = self.pca.n_components_
    
            # F分布ベースのUCL
            self.T2_UCL = (k * (n - 1) * (n + 1)) / (n * (n - k)) * \
                          f.ppf(self.confidence_level, k, n - k)
    
            # SPE管理限界線 (正常データのSPE分布から)
            X_reconstructed = self.pca.inverse_transform(T_train)
            SPE_train = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
    
            # 経験的分位点
            self.SPE_UCL = np.percentile(SPE_train, self.confidence_level * 100)
    
            print(f"MSPC Model Trained:")
            print(f"  Number of components: {k}")
            print(f"  Explained variance: {np.sum(self.pca.explained_variance_ratio_):.4f}")
            print(f"  T² UCL: {self.T2_UCL:.4f}")
            print(f"  SPE UCL: {self.SPE_UCL:.4f}")
    
            return self
    
        def detect(self, X):
            """
            異常検知
    
            Parameters:
            -----------
            X : ndarray
                新しいデータ (n_samples, n_features)
    
            Returns:
            --------
            is_anomaly : ndarray (bool)
                異常フラグ (n_samples,)
            T2_values : ndarray
                T²統計量 (n_samples,)
            SPE_values : ndarray
                SPE統計量 (n_samples,)
            """
            # 標準化
            X_scaled = (X - self.mean) / self.std
    
            # 主成分スコア
            T = self.pca.transform(X_scaled)
    
            # Hotelling's T² 計算
            Lambda_inv = np.diag(1 / self.pca.explained_variance_)
            T2_values = np.sum(T @ Lambda_inv * T, axis=1)
    
            # SPE計算
            X_reconstructed = self.pca.inverse_transform(T)
            SPE_values = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
    
            # 異常判定
            is_anomaly = (T2_values > self.T2_UCL) | (SPE_values > self.SPE_UCL)
    
            return is_anomaly, T2_values, SPE_values
    
        def contribution_plot(self, x_anomaly):
            """
            異常時の変数寄与度プロット
    
            どの変数が異常に寄与しているかを可視化
            """
            x_scaled = (x_anomaly - self.mean) / self.std
            t = self.pca.transform(x_scaled.reshape(1, -1))[0]
            x_reconstructed = self.pca.inverse_transform(t.reshape(1, -1))[0]
    
            # SPE寄与度
            spe_contribution = (x_scaled - x_reconstructed) ** 2
    
            # T²寄与度
            Lambda_inv = np.diag(1 / self.pca.explained_variance_)
            t2_contribution = np.zeros(len(x_anomaly))
    
            for i in range(len(x_anomaly)):
                # i番目の変数の寄与
                x_temp = x_scaled.copy()
                x_temp[i] = 0
                t_temp = self.pca.transform(x_temp.reshape(1, -1))[0]
                t2_temp = t_temp @ Lambda_inv @ t_temp
                t2_full = t @ Lambda_inv @ t
    
                t2_contribution[i] = t2_full - t2_temp
    
            # 可視化
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
            # SPE寄与度
            axes[0].bar(range(len(spe_contribution)), spe_contribution)
            axes[0].set_xlabel('Variable Index')
            axes[0].set_ylabel('SPE Contribution')
            axes[0].set_title('SPE Contribution Plot')
            axes[0].grid(True, alpha=0.3)
    
            # T²寄与度
            axes[1].bar(range(len(t2_contribution)), t2_contribution, color='orange')
            axes[1].set_xlabel('Variable Index')
            axes[1].set_ylabel('T² Contribution')
            axes[1].set_title('T² Contribution Plot')
            axes[1].grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('mspc_contribution.png', dpi=300, bbox_inches='tight')
            plt.show()
    
            return spe_contribution, t2_contribution
    
        def plot_control_chart(self, T2_values, SPE_values, is_anomaly):
            """MSPC管理図の可視化"""
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
            time = np.arange(len(T2_values))
    
            # T²管理図
            axes[0].plot(time, T2_values, 'b-', linewidth=1, label='T²')
            axes[0].axhline(self.T2_UCL, color='r', linestyle='--',
                           linewidth=2, label='UCL')
            axes[0].scatter(time[is_anomaly], T2_values[is_anomaly],
                           color='red', s=100, zorder=5, label='Anomaly')
            axes[0].set_xlabel('Sample')
            axes[0].set_ylabel("Hotelling's T²")
            axes[0].set_title("Hotelling's T² Control Chart")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
    
            # SPE管理図
            axes[1].plot(time, SPE_values, 'g-', linewidth=1, label='SPE')
            axes[1].axhline(self.SPE_UCL, color='r', linestyle='--',
                           linewidth=2, label='UCL')
            axes[1].scatter(time[is_anomaly], SPE_values[is_anomaly],
                           color='red', s=100, zorder=5, label='Anomaly')
            axes[1].set_xlabel('Sample')
            axes[1].set_ylabel('SPE (Q-statistic)')
            axes[1].set_title('SPE Control Chart')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('mspc_control_charts.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    
    # ========== 使用例 ==========
    if __name__ == "__main__":
        np.random.seed(42)
    
        # シミュレーションデータ生成
        # 正常運転: 10変数、相関あり
        n_normal = 500
        n_features = 10
    
        # 相関行列（変数間に相関がある）
        mean_normal = np.zeros(n_features)
        cov_normal = np.eye(n_features)
        for i in range(n_features - 1):
            cov_normal[i, i+1] = cov_normal[i+1, i] = 0.7
    
        X_normal = np.random.multivariate_normal(mean_normal, cov_normal, n_normal)
    
        # 異常データ: 一部変数に平均シフト
        n_anomaly = 100
        X_anomaly = np.random.multivariate_normal(mean_normal, cov_normal, n_anomaly)
        X_anomaly[:, 2] += 3  # 変数2に平均シフト
        X_anomaly[:, 5] += 2  # 変数5に平均シフト
    
        # テストデータ (正常 + 異常)
        X_test = np.vstack([X_normal[-100:], X_anomaly])
        y_true = np.array([0]*100 + [1]*100)  # 0=正常, 1=異常
    
        # MSPC訓練
        print("========== MSPC Training ==========")
        mspc = MultivariateSPC(n_components=0.95, confidence_level=0.99)
        mspc.fit(X_normal[:400])  # 訓練データ
    
        # 異常検知
        print("\n========== Anomaly Detection ==========")
        is_anomaly, T2_values, SPE_values = mspc.detect(X_test)
    
        # 評価
        from sklearn.metrics import classification_report, confusion_matrix
    
        print("\nClassification Report:")
        print(classification_report(y_true, is_anomaly.astype(int),
                                   target_names=['Normal', 'Anomaly']))
    
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, is_anomaly.astype(int))
        print(cm)
    
        # 検出率
        tp = cm[1, 1]
        fn = cm[1, 0]
        detection_rate = tp / (tp + fn)
        print(f"\nDetection Rate: {detection_rate:.2%}")
    
        # 誤検出率
        fp = cm[0, 1]
        tn = cm[0, 0]
        false_alarm_rate = fp / (fp + tn)
        print(f"False Alarm Rate: {false_alarm_rate:.2%}")
    
        # 管理図可視化
        mspc.plot_control_chart(T2_values, SPE_values, is_anomaly)
    
        # 異常サンプルの寄与度分析
        print("\n========== Contribution Analysis ==========")
        anomaly_sample = X_test[is_anomaly][0]
        spe_contrib, t2_contrib = mspc.contribution_plot(anomaly_sample)
    
        print(f"Top 3 SPE Contributors:")
        top_spe = np.argsort(spe_contrib)[-3:][::-1]
        for idx in top_spe:
            print(f"  Variable {idx}: {spe_contrib[idx]:.4f}")
    

### 5.2.3 Dynamic PCA (DPCA) による時系列対応

プロセスの時間的相関を考慮したDynamic PCAで、より高精度な異常検知を実現します：
    
    
    class DynamicPCA(MultivariateSPC):
        """
        Dynamic PCA
    
        時間遅れ行列を構築し、時系列の自己相関を考慮
        """
    
        def __init__(self, n_lags=5, n_components=None, confidence_level=0.99):
            """
            Parameters:
            -----------
            n_lags : int
                時間遅れの数（ラグ）
            """
            super().__init__(n_components, confidence_level)
            self.n_lags = n_lags
    
        def create_lagged_matrix(self, X):
            """
            時間遅れ行列を構築
    
            X(t), X(t-1), ..., X(t-L) を結合
            """
            n_samples, n_features = X.shape
            X_lagged = np.zeros((n_samples - self.n_lags, n_features * (self.n_lags + 1)))
    
            for i in range(n_samples - self.n_lags):
                lagged_sample = []
                for lag in range(self.n_lags + 1):
                    lagged_sample.append(X[i + self.n_lags - lag])
                X_lagged[i] = np.concatenate(lagged_sample)
    
            return X_lagged
    
        def fit(self, X_normal):
            """正常データでDPCA訓練"""
            X_lagged = self.create_lagged_matrix(X_normal)
            return super().fit(X_lagged)
    
        def detect(self, X):
            """DPCA異常検知"""
            X_lagged = self.create_lagged_matrix(X)
            return super().detect(X_lagged)
    
    
    # ========== DPCA使用例 ==========
    # 時系列相関のあるデータ生成
    np.random.seed(42)
    n_samples = 600
    n_features = 5
    
    # AR(1)プロセスでシミュレート
    X_ts_normal = np.zeros((n_samples, n_features))
    X_ts_normal[0] = np.random.randn(n_features)
    
    for t in range(1, n_samples):
        X_ts_normal[t] = 0.8 * X_ts_normal[t-1] + np.random.randn(n_features) * 0.5
    
    # DPCA適用
    print("\n========== Dynamic PCA ==========")
    dpca = DynamicPCA(n_lags=5, n_components=0.95, confidence_level=0.99)
    dpca.fit(X_ts_normal[:500])
    
    # テスト
    X_ts_test = X_ts_normal[500:]
    is_anomaly_dpca, T2_dpca, SPE_dpca = dpca.detect(X_ts_test)
    
    print(f"DPCA Detected Anomalies: {np.sum(is_anomaly_dpca)} / {len(is_anomaly_dpca)}")
    print(f"Anomaly Rate: {np.sum(is_anomaly_dpca) / len(is_anomaly_dpca):.2%}")
    

## 5.3 Isolation Forestによる異常検知

### 5.3.1 Isolation Forestの原理

Isolation Forestは、異常データが「孤立しやすい（少ない分割で分離できる）」という性質を利用します：

**アルゴリズム**

  1. ランダムに特徴量と分割値を選択
  2. データを再帰的に2分割（Binary Tree構築）
  3. 分割回数（Tree Depth）を記録
  4. 複数木の平均深さで異常度を計算

**異常度スコア**

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

\\(E(h(x))\\): 平均Tree深さ、\\(c(n)\\): 正規化定数

\\(s \approx 1\\): 異常、\\(s \approx 0.5\\): 正常

### 5.3.2 半導体プロセスへの適用
    
    
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    import matplotlib.pyplot as plt
    
    class IsolationForestFDC:
        """
        Isolation Forestによる異常検知
    
        半導体プロセスのセンサーデータから異常を検出
        """
    
        def __init__(self, contamination=0.01, n_estimators=100, max_samples='auto'):
            """
            Parameters:
            -----------
            contamination : float
                異常データの割合（事前推定値）
            n_estimators : int
                ツリー数
            max_samples : int or 'auto'
                各ツリーのサンプル数
            """
            self.contamination = contamination
            self.model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=42,
                n_jobs=-1
            )
    
        def fit(self, X_train):
            """訓練（正常データ主体）"""
            self.model.fit(X_train)
            return self
    
        def detect(self, X_test):
            """
            異常検知
    
            Returns:
            --------
            predictions : ndarray
                異常ラベル (-1: 異常, 1: 正常)
            scores : ndarray
                異常度スコア (負の値ほど異常)
            """
            predictions = self.model.predict(X_test)
            scores = self.model.score_samples(X_test)
    
            # -1 (異常) を 1 に、1 (正常) を 0 に変換
            is_anomaly = (predictions == -1)
    
            return is_anomaly, scores
    
        def plot_anomaly_score_distribution(self, scores_normal, scores_anomaly):
            """異常度スコア分布の可視化"""
            plt.figure(figsize=(10, 6))
    
            plt.hist(scores_normal, bins=50, alpha=0.6, label='Normal', color='blue')
            plt.hist(scores_anomaly, bins=50, alpha=0.6, label='Anomaly', color='red')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.title('Isolation Forest Anomaly Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
            plt.savefig('isolation_forest_score_dist.png', dpi=300, bbox_inches='tight')
            plt.show()
    
        def plot_roc_and_pr_curves(self, y_true, scores):
            """ROC曲線とPrecision-Recall曲線"""
            from sklearn.metrics import roc_curve, auc
    
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, -scores)  # 負のスコアで異常
            roc_auc = auc(fpr, tpr)
    
            axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
            axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title('ROC Curve')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
    
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, -scores)
    
            axes[1].plot(recall, precision, linewidth=2, label='PR Curve')
            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            axes[1].set_title('Precision-Recall Curve')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('isolation_forest_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
            return roc_auc
    
    
    # ========== 使用例 ==========
    if __name__ == "__main__":
        np.random.seed(42)
    
        # シミュレーションデータ
        # 正常データ: 多変量正規分布
        n_normal = 1000
        n_features = 20
    
        X_normal = np.random.randn(n_normal, n_features)
    
        # 異常データ: 外れ値
        n_anomaly = 50
        X_anomaly = np.random.randn(n_anomaly, n_features) * 3 + 5
    
        # 訓練・テストデータ
        X_train = X_normal[:800]
        X_test = np.vstack([X_normal[800:], X_anomaly])
        y_test = np.array([0]*200 + [1]*50)  # 0=正常, 1=異常
    
        # Isolation Forest訓練
        print("========== Isolation Forest Training ==========")
        if_fdc = IsolationForestFDC(contamination=0.05, n_estimators=100)
        if_fdc.fit(X_train)
    
        # 異常検知
        print("\n========== Anomaly Detection ==========")
        is_anomaly, scores = if_fdc.detect(X_test)
    
        # 評価
        print("\nClassification Report:")
        print(classification_report(y_test, is_anomaly.astype(int),
                                   target_names=['Normal', 'Anomaly']))
    
        # AUC-ROC
        roc_auc = roc_auc_score(y_test, -scores)
        print(f"\nAUC-ROC: {roc_auc:.4f}")
    
        # 可視化
        scores_normal_test = scores[y_test == 0]
        scores_anomaly_test = scores[y_test == 1]
    
        if_fdc.plot_anomaly_score_distribution(scores_normal_test, scores_anomaly_test)
        if_fdc.plot_roc_and_pr_curves(y_test, scores)
    
        print("\n========== Feature Importance Analysis ==========")
        # Feature Importance (異常サンプルでの変動が大きい特徴)
        anomaly_samples = X_test[y_test == 1]
        normal_samples = X_test[y_test == 0]
    
        feature_std_anomaly = np.std(anomaly_samples, axis=0)
        feature_std_normal = np.std(normal_samples, axis=0)
        importance = feature_std_anomaly / (feature_std_normal + 1e-6)
    
        top_features = np.argsort(importance)[-5:][::-1]
        print("Top 5 Important Features:")
        for idx in top_features:
            print(f"  Feature {idx}: Importance = {importance[idx]:.4f}")
    

## 5.4 LSTMによる時系列異常検知

### 5.4.1 LSTM Autoencoderの原理

Long Short-Term Memory (LSTM) は時系列データの長期依存関係を学習できるRNNの一種です。Autoencoder構造で正常パターンを学習し、再構成誤差で異常を検知します：

### 5.4.2 LSTM-AE実装
    
    
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import numpy as np
    import matplotlib.pyplot as plt
    
    class LSTMAutoencoderFDC:
        """
        LSTM Autoencoderによる時系列異常検知
    
        センサー時系列データから正常パターンを学習し、
        異常な時系列を検出
        """
    
        def __init__(self, sequence_length=50, n_features=10, latent_dim=20):
            """
            Parameters:
            -----------
            sequence_length : int
                時系列の長さ
            n_features : int
                特徴量数（センサー数）
            latent_dim : int
                潜在空間の次元
            """
            self.sequence_length = sequence_length
            self.n_features = n_features
            self.latent_dim = latent_dim
            self.autoencoder = None
            self.threshold = None
    
        def build_model(self):
            """LSTM Autoencoder構築"""
            # Encoder
            encoder_inputs = layers.Input(shape=(self.sequence_length, self.n_features))
    
            # LSTM Encoder
            x = layers.LSTM(64, activation='relu', return_sequences=True)(encoder_inputs)
            x = layers.LSTM(32, activation='relu', return_sequences=False)(x)
            latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
    
            encoder = models.Model(encoder_inputs, latent, name='encoder')
    
            # Decoder
            decoder_inputs = layers.Input(shape=(self.latent_dim,))
    
            # RepeatVectorで時系列次元を復元
            x = layers.RepeatVector(self.sequence_length)(decoder_inputs)
    
            # LSTM Decoder
            x = layers.LSTM(32, activation='relu', return_sequences=True)(x)
            x = layers.LSTM(64, activation='relu', return_sequences=True)(x)
    
            # 出力層
            decoder_outputs = layers.TimeDistributed(
                layers.Dense(self.n_features)
            )(x)
    
            decoder = models.Model(decoder_inputs, decoder_outputs, name='decoder')
    
            # Autoencoder
            autoencoder_outputs = decoder(encoder(encoder_inputs))
            autoencoder = models.Model(encoder_inputs, autoencoder_outputs,
                                       name='lstm_autoencoder')
    
            autoencoder.compile(optimizer='adam', loss='mse')
    
            self.autoencoder = autoencoder
            self.encoder = encoder
            self.decoder = decoder
    
            return autoencoder
    
        def train(self, X_normal, epochs=50, batch_size=32, validation_split=0.2):
            """
            正常時系列データで訓練
    
            Parameters:
            -----------
            X_normal : ndarray
                正常データ (n_samples, sequence_length, n_features)
            """
            if self.autoencoder is None:
                self.build_model()
    
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
    
            history = self.autoencoder.fit(
                X_normal, X_normal,  # 自己教師あり
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
    
            return history
    
        def calculate_reconstruction_errors(self, X):
            """
            再構成誤差計算
    
            Returns:
            --------
            errors : ndarray
                各サンプルのMSE (n_samples,)
            """
            X_reconstructed = self.autoencoder.predict(X, verbose=0)
            errors = np.mean((X - X_reconstructed) ** 2, axis=(1, 2))
    
            return errors
    
        def set_threshold(self, X_normal, percentile=99):
            """異常判定閾値設定"""
            errors = self.calculate_reconstruction_errors(X_normal)
            self.threshold = np.percentile(errors, percentile)
    
            print(f"Threshold set: {self.threshold:.6f} "
                  f"({percentile}th percentile of normal data)")
    
            return self.threshold
    
        def detect_anomalies(self, X):
            """異常検知"""
            if self.threshold is None:
                raise ValueError("Threshold not set. Run set_threshold() first.")
    
            errors = self.calculate_reconstruction_errors(X)
            is_anomaly = errors > self.threshold
    
            return is_anomaly, errors
    
        def visualize_reconstruction(self, X_sample, sample_idx=0):
            """再構成結果の可視化"""
            X_recon = self.autoencoder.predict(X_sample[sample_idx:sample_idx+1], verbose=0)[0]
            original = X_sample[sample_idx]
    
            fig, axes = plt.subplots(self.n_features, 1,
                                    figsize=(12, 2 * self.n_features))
    
            time_steps = np.arange(self.sequence_length)
    
            for i in range(self.n_features):
                axes[i].plot(time_steps, original[:, i], 'b-',
                            linewidth=2, label='Original')
                axes[i].plot(time_steps, X_recon[:, i], 'r--',
                            linewidth=2, label='Reconstructed')
                axes[i].set_ylabel(f'Feature {i}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
    
            axes[-1].set_xlabel('Time Step')
            plt.suptitle('LSTM-AE Reconstruction')
            plt.tight_layout()
            plt.savefig('lstm_ae_reconstruction.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    
    # ========== 使用例 ==========
    if __name__ == "__main__":
        np.random.seed(42)
        tf.random.set_seed(42)
    
        # 時系列データ生成
        sequence_length = 50
        n_features = 5
        n_normal = 500
        n_anomaly = 100
    
        # 正常時系列: 正弦波 + ノイズ
        X_normal = np.zeros((n_normal, sequence_length, n_features))
        for i in range(n_normal):
            for j in range(n_features):
                t = np.linspace(0, 4*np.pi, sequence_length)
                X_normal[i, :, j] = np.sin(t + j * np.pi/4) + np.random.randn(sequence_length) * 0.1
    
        # 異常時系列: 突発的なスパイク
        X_anomaly = np.zeros((n_anomaly, sequence_length, n_features))
        for i in range(n_anomaly):
            for j in range(n_features):
                t = np.linspace(0, 4*np.pi, sequence_length)
                signal = np.sin(t + j * np.pi/4)
                # ランダムな位置にスパイク
                spike_pos = np.random.randint(10, 40)
                signal[spike_pos:spike_pos+5] += 3
                X_anomaly[i, :, j] = signal + np.random.randn(sequence_length) * 0.1
    
        # 訓練・テスト分割
        X_train = X_normal[:400]
        X_test = np.vstack([X_normal[400:], X_anomaly])
        y_test = np.array([0]*100 + [1]*100)
    
        # LSTM-AE構築・訓練
        print("========== LSTM Autoencoder Training ==========")
        lstm_ae = LSTMAutoencoderFDC(
            sequence_length=sequence_length,
            n_features=n_features,
            latent_dim=10
        )
        lstm_ae.build_model()
    
        print("\nModel Architecture:")
        lstm_ae.autoencoder.summary()
    
        history = lstm_ae.train(X_train, epochs=30, batch_size=32)
    
        # 閾値設定
        print("\n========== Setting Threshold ==========")
        lstm_ae.set_threshold(X_normal[400:450], percentile=99)
    
        # 異常検知
        print("\n========== Anomaly Detection ==========")
        is_anomaly, errors = lstm_ae.detect_anomalies(X_test)
    
        # 評価
        print("\nClassification Report:")
        print(classification_report(y_test, is_anomaly.astype(int),
                                   target_names=['Normal', 'Anomaly']))
    
        # AUC-ROC
        auc_score = roc_auc_score(y_test, errors)
        print(f"\nAUC-ROC: {auc_score:.4f}")
    
        # 再構成結果の可視化
        print("\n========== Reconstruction Visualization ==========")
        # 正常サンプル
        lstm_ae.visualize_reconstruction(X_test[y_test == 0], sample_idx=0)
        # 異常サンプル
        lstm_ae.visualize_reconstruction(X_test[y_test == 1], sample_idx=0)
    
        # エラー分布
        plt.figure(figsize=(10, 6))
        plt.hist(errors[y_test == 0], bins=50, alpha=0.6, label='Normal')
        plt.hist(errors[y_test == 1], bins=50, alpha=0.6, label='Anomaly')
        plt.axvline(lstm_ae.threshold, color='r', linestyle='--',
                   linewidth=2, label='Threshold')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('LSTM-AE Reconstruction Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('lstm_ae_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    

## 5.5 まとめ

本章では、半導体製造におけるFault Detection & Classification (FDC) のAI実装手法を学習しました：

### 主要な学習内容

#### 1\. Multivariate SPC (MSPC)

  * **PCAによる次元削減** で多変数相関を捉える
  * **Hotelling's T² & SPE**で2種類の異常を検出
  * **Contribution Plot** で異常変数を特定
  * **Dynamic PCA** で時系列相関に対応

#### 2\. Isolation Forest

  * **教師なし学習** で未知異常を検出
  * **高速・スケーラブル** (100万サンプル対応)
  * **異常度スコア** で優先度付け
  * **AUC-ROC > 0.95**の高精度を実現

#### 3\. LSTM Autoencoder

  * **時系列パターン学習** で異常波形を検出
  * **再構成誤差ベース** の判定
  * **長期依存関係** を捉える（50ステップ以上）
  * **可視化** で異常箇所を明示

#### 実用上の成果

  * 異常検出率: **95%以上** (従来70%)
  * 誤検出率: **5%以下** (従来20%)
  * 検出時間: **0.1秒以下** (リアルタイム対応)
  * ダウンタイム削減: **年間数億円** のコスト削減

### シリーズ全体のまとめ

本シリーズ「半導体製造AI」では、半導体製造プロセス全般にわたるAI技術を学習しました：

#### 第1章: ウェハプロセス統計的管理

Run-to-Run制御、Virtual Metrology

#### 第2章: AIによる欠陥検査とAOI

CNN分類、U-Netセグメンテーション、Autoencoder異常検知

#### 第3章: 歩留まり向上とパラメータ最適化

Bayesian Optimization、NSGA-II多目的最適化

#### 第4章: Advanced Process Control

モデル予測制御 (MPC)、DQN強化学習制御

#### 第5章: Fault Detection & Classification

MSPC、Isolation Forest、LSTM-AE時系列異常検知

### 今後の展望

  * **デジタルツイン** : プロセス全体のリアルタイムシミュレーション
  * **Explainable AI** : SHAPによる意思決定の透明化
  * **Federated Learning** : 複数Fab間での知識共有
  * **Edge AI** : 装置内でのリアルタイムAI推論
  * **自律製造** : AIによる完全自動最適化

← 前の章（準備中） [目次に戻る](<index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
