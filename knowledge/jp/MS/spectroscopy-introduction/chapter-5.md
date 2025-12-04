---
title: "第5章: Python実践：分光データ解析ワークフロー"
chapter_title: "第5章: Python実践：分光データ解析ワークフロー"
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/spectroscopy-introduction/chapter-5.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Spectroscopy](<../../MS/spectroscopy-introduction/index.html>)›Chapter 5

# 第5章: Python実践：分光データ解析ワークフロー

**この章で学ぶこと:** 本章では、第1章から第4章で学んだ分光分析法（IR、Raman、UV-Vis、XPS）を統合し、実践的なPythonデータ解析ワークフローを構築します。汎用スペクトルデータローダー、自動ピーク検出アルゴリズム、機械学習による分類・回帰、バッチ処理システム、インタラクティブ可視化まで、研究現場で即座に活用できる実用的なツールキットを開発します。全てのコードは再利用可能なモジュール形式で提供され、読者自身の研究データに容易に適用できます。

## 5.1 統合分光データローダーの設計

### 5.1.1 汎用データ読み込みクラス

分光装置から出力されるデータ形式は多様です（CSV、TXT、装置固有バイナリ形式等）。汎用的なデータローダークラスを設計することで、異なる形式のデータを統一的に扱えます。
    
    
    ```mermaid
    flowchart TD
            A[分光データファイル] --> B{ファイル形式判定}
            B -->|CSV| C[CSVパーサー]
            B -->|TXT| D[テキストパーサー]
            B -->|Binary| E[バイナリパーサー]
            C --> F[データ正規化]
            D --> F
            E --> F
            F --> G[SpectralDataオブジェクト]
            G --> H[解析パイプライン]
    
            style A fill:#e3f2fd
            style B fill:#fff3e0
            style F fill:#e8f5e9
            style G fill:#fce4ec
            style H fill:#ffe0b2
        
    コード例1: 汎用スペクトルデータローダークラス
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from typing import Union, Tuple, Optional
    
    class SpectralData:
        """
        汎用スペクトルデータクラス
    
        Attributes:
        -----------
        x : array
            横軸データ（波長、波数、結合エネルギー等）
        y : array
            縦軸データ（強度、吸光度等）
        x_label : str
            横軸ラベル
        y_label : str
            縦軸ラベル
        metadata : dict
            メタデータ（測定日時、装置、条件等）
        """
    
        def __init__(self, x: np.ndarray, y: np.ndarray,
                     x_label: str = "X", y_label: str = "Intensity",
                     metadata: Optional[dict] = None):
            """
            スペクトルデータを初期化
    
            Parameters:
            -----------
            x : array
                横軸データ
            y : array
                縦軸データ
            x_label : str
                横軸ラベル
            y_label : str
                縦軸ラベル
            metadata : dict, optional
                メタデータ
            """
            self.x = np.asarray(x)
            self.y = np.asarray(y)
            self.x_label = x_label
            self.y_label = y_label
            self.metadata = metadata or {}
    
            # データサイズ検証
            if len(self.x) != len(self.y):
                raise ValueError(f"x and y must have the same length: {len(x)} != {len(y)}")
    
        def __repr__(self):
            return (f"SpectralData(n_points={len(self.x)}, "
                    f"x_range=[{self.x.min():.2f}, {self.x.max():.2f}], "
                    f"y_range=[{self.y.min():.2f}, {self.y.max():.2f}])")
    
        def copy(self):
            """データのコピーを作成"""
            return SpectralData(self.x.copy(), self.y.copy(),
                               self.x_label, self.y_label,
                               self.metadata.copy())
    
        def trim(self, x_min: float, x_max: float):
            """
            指定範囲でデータを切り出し
    
            Parameters:
            -----------
            x_min, x_max : float
                切り出し範囲
            """
            mask = (self.x >= x_min) & (self.x <= x_max)
            self.x = self.x[mask]
            self.y = self.y[mask]
    
        def normalize(self, method: str = 'max'):
            """
            スペクトルを正規化
    
            Parameters:
            -----------
            method : str
                'max': 最大値で正規化
                'minmax': 0-1範囲に正規化
                'area': 面積で正規化
            """
            if method == 'max':
                self.y = self.y / self.y.max()
            elif method == 'minmax':
                self.y = (self.y - self.y.min()) / (self.y.max() - self.y.min())
            elif method == 'area':
                area = np.trapz(self.y, self.x)
                self.y = self.y / area
            else:
                raise ValueError(f"Unknown normalization method: {method}")
    
    
    class SpectralDataLoader:
        """
        汎用スペクトルデータローダー
        """
    
        @staticmethod
        def load_csv(filepath: Union[str, Path], x_col: int = 0, y_col: int = 1,
                     delimiter: str = ',', skiprows: int = 0,
                     x_label: str = "X", y_label: str = "Intensity") -> SpectralData:
            """
            CSVファイルからスペクトルデータを読み込み
    
            Parameters:
            -----------
            filepath : str or Path
                ファイルパス
            x_col : int
                横軸データの列番号
            y_col : int
                縦軸データの列番号
            delimiter : str
                区切り文字
            skiprows : int
                スキップする行数
            x_label, y_label : str
                軸ラベル
    
            Returns:
            --------
            data : SpectralData
                スペクトルデータオブジェクト
            """
            df = pd.read_csv(filepath, delimiter=delimiter, skiprows=skiprows, header=None)
            x = df.iloc[:, x_col].values
            y = df.iloc[:, y_col].values
    
            metadata = {
                'filename': Path(filepath).name,
                'source': 'CSV',
                'columns': f"x={x_col}, y={y_col}"
            }
    
            return SpectralData(x, y, x_label, y_label, metadata)
    
        @staticmethod
        def load_txt(filepath: Union[str, Path], x_col: int = 0, y_col: int = 1,
                     skiprows: int = 0, x_label: str = "X", y_label: str = "Intensity") -> SpectralData:
            """
            テキストファイルからスペクトルデータを読み込み（空白区切り）
    
            Parameters:
            -----------
            filepath : str or Path
                ファイルパス
            x_col, y_col : int
                横軸・縦軸データの列番号
            skiprows : int
                スキップする行数
            x_label, y_label : str
                軸ラベル
    
            Returns:
            --------
            data : SpectralData
                スペクトルデータオブジェクト
            """
            data_array = np.loadtxt(filepath, skiprows=skiprows)
            x = data_array[:, x_col]
            y = data_array[:, y_col]
    
            metadata = {
                'filename': Path(filepath).name,
                'source': 'TXT',
                'columns': f"x={x_col}, y={y_col}"
            }
    
            return SpectralData(x, y, x_label, y_label, metadata)
    
        @staticmethod
        def auto_load(filepath: Union[str, Path], **kwargs) -> SpectralData:
            """
            ファイル拡張子から自動的に適切なローダーを選択
    
            Parameters:
            -----------
            filepath : str or Path
                ファイルパス
            **kwargs
                各ローダーへの追加引数
    
            Returns:
            --------
            data : SpectralData
                スペクトルデータオブジェクト
            """
            filepath = Path(filepath)
            ext = filepath.suffix.lower()
    
            if ext == '.csv':
                return SpectralDataLoader.load_csv(filepath, **kwargs)
            elif ext in ['.txt', '.dat']:
                return SpectralDataLoader.load_txt(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
    
    
    # 使用例
    if __name__ == "__main__":
        # シミュレーションデータの生成と保存
        x_sim = np.linspace(400, 700, 300)
        y_sim = 0.8 * np.exp(-((x_sim - 550)**2) / (2 * 40**2)) + np.random.normal(0, 0.02, len(x_sim))
    
        # CSVファイルとして保存
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            for xi, yi in zip(x_sim, y_sim):
                f.write(f"{xi},{yi}\n")
            temp_path = f.name
    
        # データ読み込み
        loader = SpectralDataLoader()
        spectrum = loader.auto_load(temp_path, x_label="Wavelength (nm)", y_label="Absorbance")
    
        print(spectrum)
        print(f"Metadata: {spectrum.metadata}")
    
        # データ操作
        spectrum_copy = spectrum.copy()
        spectrum_copy.trim(500, 600)
        spectrum_copy.normalize(method='max')
    
        print(f"Trimmed and normalized: {spectrum_copy}")
    
        # クリーンアップ
        import os
        os.unlink(temp_path)
    
    5.2 自動ピーク検出と特徴抽出
    5.2.1 ピーク検出アルゴリズムの統合
    スペクトルからピークを自動検出することは、多数のサンプルを効率的に解析する上で重要です。scipy.signal.find_peaksを拡張し、分光分析に特化したピーク検出クラスを実装します。
    コード例2: 高度な自動ピーク検出システム
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks, peak_widths, peak_prominences
    from scipy.optimize import curve_fit
    from dataclasses import dataclass
    from typing import List, Tuple
    
    @dataclass
    class Peak:
        """
        ピーク情報を格納するデータクラス
    
        Attributes:
        -----------
        position : float
            ピーク位置（横軸座標）
        height : float
            ピーク高さ
        width : float
            半値全幅（FWHM）
        area : float
            ピーク面積
        prominence : float
            ピークの卓越度
        """
        position: float
        height: float
        width: float
        area: float
        prominence: float
    
        def __repr__(self):
            return (f"Peak(pos={self.position:.2f}, height={self.height:.2f}, "
                    f"FWHM={self.width:.2f}, area={self.area:.1f})")
    
    
    class PeakDetector:
        """
        スペクトルデータからピークを自動検出するクラス
        """
    
        def __init__(self, spectral_data: SpectralData):
            """
            Parameters:
            -----------
            spectral_data : SpectralData
                スペクトルデータオブジェクト
            """
            self.data = spectral_data
            self.peaks: List[Peak] = []
    
        def detect_peaks(self, height: float = None, prominence: float = None,
                        distance: int = None, width: Tuple[float, float] = None) -> List[Peak]:
            """
            ピークを検出
    
            Parameters:
            -----------
            height : float, optional
                最小ピーク高さ
            prominence : float, optional
                最小卓越度
            distance : int, optional
                ピーク間の最小距離（データポイント数）
            width : tuple, optional
                ピーク幅の範囲 (min_width, max_width)
    
            Returns:
            --------
            peaks : list of Peak
                検出されたピークのリスト
            """
            # ピーク検出
            peak_indices, properties = find_peaks(self.data.y,
                                                  height=height,
                                                  prominence=prominence,
                                                  distance=distance,
                                                  width=width)
    
            # ピーク幅の計算（FWHM）
            widths, width_heights, left_ips, right_ips = peak_widths(
                self.data.y, peak_indices, rel_height=0.5
            )
    
            # ピーク卓越度
            prominences, _, _ = peak_prominences(self.data.y, peak_indices)
    
            # Peakオブジェクトのリスト作成
            self.peaks = []
            for i, idx in enumerate(peak_indices):
                # ピーク位置（横軸座標）
                position = self.data.x[idx]
    
                # ピーク高さ
                height_val = self.data.y[idx]
    
                # FWHM（データポイント単位から横軸単位に変換）
                dx = np.mean(np.diff(self.data.x))
                fwhm = widths[i] * dx
    
                # ピーク面積の近似（ガウス近似: 面積 ≈ 高さ × FWHM × sqrt(π/(4ln2))）
                area = height_val * fwhm * np.sqrt(np.pi / (4 * np.log(2)))
    
                peak = Peak(
                    position=position,
                    height=height_val,
                    width=fwhm,
                    area=area,
                    prominence=prominences[i]
                )
                self.peaks.append(peak)
    
            return self.peaks
    
        def fit_peaks_gaussian(self, peak_region_width: float = 5.0) -> List[dict]:
            """
            各ピークをガウス関数でフィッティング
    
            Parameters:
            -----------
            peak_region_width : float
                各ピークのフィッティング領域幅（両側）
    
            Returns:
            --------
            fit_results : list of dict
                各ピークのフィッティング結果
            """
            fit_results = []
    
            for peak in self.peaks:
                # フィッティング領域の抽出
                mask = (self.data.x >= peak.position - peak_region_width) & \
                       (self.data.x <= peak.position + peak_region_width)
                x_fit = self.data.x[mask]
                y_fit = self.data.y[mask]
    
                if len(x_fit) < 5:
                    continue
    
                # ガウス関数フィッティング
                def gaussian(x, A, mu, sigma):
                    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))
    
                try:
                    # 初期パラメータ
                    p0 = [peak.height, peak.position, peak.width / (2 * np.sqrt(2 * np.log(2)))]
                    popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=5000)
    
                    A_fit, mu_fit, sigma_fit = popt
                    fwhm_fit = 2 * np.sqrt(2 * np.log(2)) * sigma_fit
                    area_fit = A_fit * sigma_fit * np.sqrt(2 * np.pi)
    
                    fit_results.append({
                        'peak_position': mu_fit,
                        'amplitude': A_fit,
                        'fwhm': fwhm_fit,
                        'area': area_fit,
                        'fit_quality': 'success'
                    })
                except RuntimeError:
                    fit_results.append({
                        'peak_position': peak.position,
                        'fit_quality': 'failed'
                    })
    
            return fit_results
    
        def plot_detected_peaks(self, show_labels: bool = True):
            """
            検出されたピークを可視化
    
            Parameters:
            -----------
            show_labels : bool
                ピーク位置のラベルを表示するか
            """
            fig, ax = plt.subplots(figsize=(12, 6))
    
            # スペクトルをプロット
            ax.plot(self.data.x, self.data.y, 'b-', linewidth=1.5, label='Spectrum', alpha=0.7)
    
            # ピーク位置をマーク
            if self.peaks:
                peak_positions = [p.position for p in self.peaks]
                peak_heights = [p.height for p in self.peaks]
    
                ax.plot(peak_positions, peak_heights, 'ro', markersize=10,
                       label=f'Detected Peaks (n={len(self.peaks)})', zorder=5)
    
                # ピークラベル
                if show_labels:
                    for peak in self.peaks:
                        ax.annotate(f'{peak.position:.1f}',
                                   xy=(peak.position, peak.height),
                                   xytext=(0, 10), textcoords='offset points',
                                   ha='center', fontsize=9, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
            ax.set_xlabel(self.data.x_label, fontsize=12)
            ax.set_ylabel(self.data.y_label, fontsize=12)
            ax.set_title('自動ピーク検出結果', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    
    # 使用例
    if __name__ == "__main__":
        # シミュレーションスペクトル（3つのピーク）
        x = np.linspace(400, 700, 600)
        y = (0.8 * np.exp(-((x - 450)**2) / (2 * 30**2)) +
             0.6 * np.exp(-((x - 550)**2) / (2 * 40**2)) +
             0.9 * np.exp(-((x - 620)**2) / (2  25**2)) +
             np.random.normal(0, 0.02, len(x)))
    
        spectrum = SpectralData(x, y, x_label="Wavelength (nm)", y_label="Absorbance")
    
        # ピーク検出
        detector = PeakDetector(spectrum)
        peaks = detector.detect_peaks(height=0.2, prominence=0.15, distance=50)
    
        print(f"検出されたピーク数: {len(peaks)}")
        for i, peak in enumerate(peaks, 1):
            print(f"  Peak {i}: {peak}")
    
        # ガウスフィッティング
        fit_results = detector.fit_peaks_gaussian(peak_region_width=50)
        print("\nガウスフィッティング結果:")
        for i, result in enumerate(fit_results, 1):
            if result['fit_quality'] == 'success':
                print(f"  Peak {i}: pos={result['peak_position']:.2f}, "
                      f"FWHM={result['fwhm']:.2f}, area={result['area']:.1f}")
    
        # プロット
        detector.plot_detected_peaks(show_labels=True)
    
    5.3 機械学習による分光スペクトル分類
    5.3.1 スペクトル特徴量の抽出
    機械学習モデルにスペクトルを入力するには、適切な特徴量を抽出する必要があります。
    
    スペクトル特徴量の種類
    
    ピーク位置・高さ・幅: 化学構造に直接関連
    スペクトル形状記述子: 統計的特徴（平均、分散、歪度、尖度）
    主成分分析（PCA）: 次元削減と情報圧縮
    畳み込みニューラルネットワーク（CNN）: 自動特徴抽出
    
    
    コード例3: スペクトル分類パイプライン（ランダムフォレスト）
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.decomposition import PCA
    import seaborn as sns
    
    class SpectralClassifier:
        """
        スペクトルデータを機械学習で分類するクラス
        """
    
        def __init__(self, n_estimators: int = 100, use_pca: bool = False, n_components: int = 10):
            """
            Parameters:
            -----------
            n_estimators : int
                ランダムフォレストの木の数
            use_pca : bool
                PCA前処理を使用するか
            n_components : int
                PCA成分数
            """
            self.classifier = RandomForestClassifier(n_estimators=n_estimators,
                                                     max_depth=10,
                                                     random_state=42)
            self.use_pca = use_pca
            self.pca = PCA(n_components=n_components) if use_pca else None
            self.class_names = None
    
        def extract_features(self, spectra_list: List[SpectralData]) -> np.ndarray:
            """
            スペクトルから特徴量を抽出
    
            Parameters:
            -----------
            spectra_list : list of SpectralData
                スペクトルデータのリスト
    
            Returns:
            --------
            features : array, shape (n_samples, n_features)
                特徴量行列
            """
            features = []
    
            for spectrum in spectra_list:
                # 特徴量1: スペクトル全体（リサンプリング）
                # 全スペクトルを固定長（例: 100点）にリサンプル
                x_uniform = np.linspace(spectrum.x.min(), spectrum.x.max(), 100)
                y_resampled = np.interp(x_uniform, spectrum.x, spectrum.y)
    
                # 特徴量2: 統計的特徴
                mean_val = np.mean(spectrum.y)
                std_val = np.std(spectrum.y)
                max_val = np.max(spectrum.y)
                min_val = np.min(spectrum.y)
    
                # 結合
                feature_vector = np.concatenate([
                    y_resampled,
                    [mean_val, std_val, max_val, min_val]
                ])
                features.append(feature_vector)
    
            return np.array(features)
    
        def train(self, X: np.ndarray, y: np.ndarray, class_names: List[str]):
            """
            分類器を訓練
    
            Parameters:
            -----------
            X : array, shape (n_samples, n_features)
                特徴量行列
            y : array, shape (n_samples,)
                クラスラベル
            class_names : list of str
                クラス名のリスト
            """
            self.class_names = class_names
    
            # PCA前処理
            if self.use_pca:
                X = self.pca.fit_transform(X)
                print(f"PCA: {X.shape[1]} components explain "
                      f"{self.pca.explained_variance_ratio_.sum():.2%} of variance")
    
            # 訓練
            self.classifier.fit(X, y)
    
            # クロスバリデーション
            cv_scores = cross_val_score(self.classifier, X, y, cv=5)
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
        def predict(self, X: np.ndarray) -> np.ndarray:
            """
            クラスを予測
    
            Parameters:
            -----------
            X : array, shape (n_samples, n_features)
                特徴量行列
    
            Returns:
            --------
            y_pred : array, shape (n_samples,)
                予測クラスラベル
            """
            if self.use_pca:
                X = self.pca.transform(X)
            return self.classifier.predict(X)
    
        def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
            """
            テストデータで評価
    
            Parameters:
            -----------
            X_test : array
                テスト特徴量
            y_test : array
                テストラベル
            """
            y_pred = self.predict(X_test)
    
            # 分類レポート
            print("\n分類レポート:")
            print(classification_report(y_test, y_pred, target_names=self.class_names))
    
            # 混同行列
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.xlabel('予測ラベル', fontsize=12)
            plt.ylabel('真のラベル', fontsize=12)
            plt.title('混同行列', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
    
    # 使用例：3種類の材料の分光スペクトル分類
    if __name__ == "__main__":
        np.random.seed(42)
    
        # シミュレーションデータ生成（3クラス × 各30サンプル）
        def generate_spectrum_class(peak_positions, peak_widths, n_samples=30):
            spectra = []
            x = np.linspace(400, 700, 300)
            for _ in range(n_samples):
                y = np.zeros_like(x)
                for pos, width in zip(peak_positions, peak_widths):
                    # ピーク位置にランダムノイズ
                    pos_noise = pos + np.random.normal(0, 5)
                    amplitude = np.random.uniform(0.7, 1.0)
                    y += amplitude * np.exp(-((x - pos_noise)**2) / (2 * width**2))
                y += np.random.normal(0, 0.03, len(x))
                spectra.append(SpectralData(x, y, "Wavelength (nm)", "Intensity"))
            return spectra
    
        # クラス1: 単一ピーク（550 nm）
        class1_spectra = generate_spectrum_class([550], [40], n_samples=30)
    
        # クラス2: 2つのピーク（450 nm, 620 nm）
        class2_spectra = generate_spectrum_class([450, 620], [30, 35], n_samples=30)
    
        # クラス3: 3つのピーク（480 nm, 550 nm, 650 nm）
        class3_spectra = generate_spectrum_class([480, 550, 650], [25, 30, 30], n_samples=30)
    
        # データ統合
        all_spectra = class1_spectra + class2_spectra + class3_spectra
        labels = np.array([0]*30 + [1]*30 + [2]*30)
        class_names = ['Material A (1 peak)', 'Material B (2 peaks)', 'Material C (3 peaks)']
    
        # 特徴量抽出
        classifier = SpectralClassifier(n_estimators=100, use_pca=False)
        X = classifier.extract_features(all_spectra)
    
        # トレーニング・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3,
                                                             random_state=42, stratify=labels)
    
        # 訓練
        classifier.train(X_train, y_train, class_names)
    
        # 評価
        classifier.evaluate(X_test, y_test)
    
        print(f"\nTotal samples: {len(all_spectra)}")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    5.4 バッチ処理とデータ管理
    5.4.1 複数ファイルの一括処理
    研究現場では、数十〜数百のスペクトルファイルを処理することが一般的です。効率的なバッチ処理システムが必要です。
    コード例4: バッチ処理パイプライン
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from typing import List, Dict, Callable
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    import json
    
    class BatchProcessor:
        """
        複数スペクトルファイルのバッチ処理クラス
        """
    
        def __init__(self, data_dir: Union[str, Path]):
            """
            Parameters:
            -----------
            data_dir : str or Path
                データディレクトリ
            """
            self.data_dir = Path(data_dir)
            self.results = []
    
        def find_files(self, pattern: str = "*.csv") -> List[Path]:
            """
            指定パターンでファイルを検索
    
            Parameters:
            -----------
            pattern : str
                ファイルパターン（例: "*.csv", "sample_*.txt"）
    
            Returns:
            --------
            files : list of Path
                見つかったファイルのリスト
            """
            files = list(self.data_dir.glob(pattern))
            print(f"Found {len(files)} files matching '{pattern}'")
            return files
    
        def process_file(self, filepath: Path,
                        processing_func: Callable[[SpectralData], Dict]) -> Dict:
            """
            単一ファイルを処理
    
            Parameters:
            -----------
            filepath : Path
                ファイルパス
            processing_func : callable
                処理関数（SpectralDataを受け取り、結果dictを返す）
    
            Returns:
            --------
            result : dict
                処理結果
            """
            try:
                # データ読み込み
                loader = SpectralDataLoader()
                spectrum = loader.auto_load(filepath)
    
                # 処理実行
                result = processing_func(spectrum)
    
                # ファイル情報を追加
                result['filename'] = filepath.name
                result['status'] = 'success'
    
                return result
    
            except Exception as e:
                return {
                    'filename': filepath.name,
                    'status': 'failed',
                    'error': str(e)
                }
    
        def batch_process(self, files: List[Path],
                         processing_func: Callable[[SpectralData], Dict],
                         n_workers: int = 4) -> pd.DataFrame:
            """
            複数ファイルを並列処理
    
            Parameters:
            -----------
            files : list of Path
                処理するファイルのリスト
            processing_func : callable
                処理関数
            n_workers : int
                並列ワーカー数
    
            Returns:
            --------
            results_df : DataFrame
                処理結果のDataFrame
            """
            results = []
    
            # 並列処理
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # タスク投入
                futures = {executor.submit(self.process_file, f, processing_func): f
                          for f in files}
    
                # 進捗表示
                for future in tqdm(as_completed(futures), total=len(files),
                                  desc="Processing files"):
                    result = future.result()
                    results.append(result)
    
            # DataFrameに変換
            results_df = pd.DataFrame(results)
            self.results = results_df
    
            # 成功/失敗のサマリー
            success_count = (results_df['status'] == 'success').sum()
            print(f"\nProcessing complete: {success_count}/{len(files)} successful")
    
            return results_df
    
        def save_results(self, output_path: Union[str, Path], format: str = 'csv'):
            """
            結果を保存
    
            Parameters:
            -----------
            output_path : str or Path
                出力ファイルパス
            format : str
                'csv', 'json', 'excel' のいずれか
            """
            if self.results is None or len(self.results) == 0:
                print("No results to save")
                return
    
            output_path = Path(output_path)
    
            if format == 'csv':
                self.results.to_csv(output_path, index=False)
            elif format == 'json':
                self.results.to_json(output_path, orient='records', indent=2)
            elif format == 'excel':
                self.results.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unknown format: {format}")
    
            print(f"Results saved to {output_path}")
    
    
    # 使用例：バッチ処理の実装
    def example_processing_function(spectrum: SpectralData) -> Dict:
        """
        例：ピーク検出と統計情報の抽出
    
        Parameters:
        -----------
        spectrum : SpectralData
            スペクトルデータ
    
        Returns:
        --------
        result : dict
            解析結果
        """
        # ピーク検出
        detector = PeakDetector(spectrum)
        peaks = detector.detect_peaks(height=0.1, prominence=0.05, distance=20)
    
        # 統計情報
        mean_intensity = np.mean(spectrum.y)
        max_intensity = np.max(spectrum.y)
        std_intensity = np.std(spectrum.y)
    
        # ピーク情報
        peak_positions = [p.position for p in peaks]
        peak_heights = [p.height for p in peaks]
    
        return {
            'n_peaks': len(peaks),
            'peak_positions': peak_positions,
            'peak_heights': peak_heights,
            'mean_intensity': mean_intensity,
            'max_intensity': max_intensity,
            'std_intensity': std_intensity,
            'x_range_min': spectrum.x.min(),
            'x_range_max': spectrum.x.max()
        }
    
    
    if __name__ == "__main__":
        # シミュレーション：複数ファイルの生成と処理
        import tempfile
        import shutil
    
        # 一時ディレクトリ作成
        temp_dir = Path(tempfile.mkdtemp())
        print(f"Temporary directory: {temp_dir}")
    
        # シミュレーションデータ生成（10ファイル）
        for i in range(10):
            x = np.linspace(400, 700, 300)
            # ランダムなピーク
            n_peaks = np.random.randint(1, 4)
            y = np.zeros_like(x)
            for _ in range(n_peaks):
                peak_pos = np.random.uniform(450, 650)
                peak_width = np.random.uniform(20, 40)
                y += np.random.uniform(0.5, 1.0) * np.exp(-((x - peak_pos)**2) / (2 * peak_width**2))
            y += np.random.normal(0, 0.02, len(x))
    
            # CSV保存
            filepath = temp_dir / f"sample_{i:03d}.csv"
            with open(filepath, 'w') as f:
                for xi, yi in zip(x, y):
                    f.write(f"{xi},{yi}\n")
    
        # バッチ処理
        processor = BatchProcessor(temp_dir)
        files = processor.find_files("*.csv")
        results_df = processor.batch_process(files, example_processing_function, n_workers=2)
    
        print("\n処理結果の一部:")
        print(results_df.head())
    
        # 結果保存
        output_path = temp_dir / "batch_results.csv"
        processor.save_results(output_path, format='csv')
    
        # クリーンアップ
        shutil.rmtree(temp_dir)
        print(f"\nTemporary directory removed")
    
    5.5 インタラクティブ可視化とダッシュボード
    5.5.1 Plotlyによる動的プロット
    静的なMatplotlibに加え、インタラクティブな可視化ライブラリPlotlyを使用することで、ズーム、パン、ホバー情報表示などの機能を持つ動的なプロットを作成できます。
    コード例5: インタラクティブスペクトルビューア
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from typing import List
    
    class InteractiveSpectralViewer:
        """
        インタラクティブなスペクトル可視化クラス（Plotly使用）
        """
    
        def __init__(self):
            self.fig = None
    
        def plot_single_spectrum(self, spectrum: SpectralData, title: str = "Spectrum"):
            """
            単一スペクトルをプロット
    
            Parameters:
            -----------
            spectrum : SpectralData
                スペクトルデータ
            title : str
                プロットタイトル
            """
            fig = go.Figure()
    
            fig.add_trace(go.Scatter(
                x=spectrum.x,
                y=spectrum.y,
                mode='lines',
                name='Spectrum',
                line=dict(color='blue', width=2),
                hovertemplate='%{x:.2f}%{y:.3f}'
            ))
    
            fig.update_layout(
                title=title,
                xaxis_title=spectrum.x_label,
                yaxis_title=spectrum.y_label,
                hovermode='x unified',
                template='plotly_white',
                width=1000,
                height=600
            )
    
            self.fig = fig
            return fig
    
        def plot_multiple_spectra(self, spectra_list: List[SpectralData],
                                 labels: List[str], title: str = "Multiple Spectra"):
            """
            複数スペクトルを重ね合わせプロット
    
            Parameters:
            -----------
            spectra_list : list of SpectralData
                スペクトルのリスト
            labels : list of str
                各スペクトルのラベル
            title : str
                プロットタイトル
            """
            fig = go.Figure()
    
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
            for i, (spectrum, label) in enumerate(zip(spectra_list, labels)):
                color = colors[i % len(colors)]
    
                fig.add_trace(go.Scatter(
                    x=spectrum.x,
                    y=spectrum.y,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2),
                    hovertemplate=f'{label}%{{x:.2f}}%{{y:.3f}}'
                ))
    
            fig.update_layout(
                title=title,
                xaxis_title=spectra_list[0].x_label,
                yaxis_title=spectra_list[0].y_label,
                hovermode='x unified',
                template='plotly_white',
                width=1000,
                height=600,
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
            )
    
            self.fig = fig
            return fig
    
        def plot_with_peaks(self, spectrum: SpectralData, peaks: List[Peak],
                           title: str = "Spectrum with Detected Peaks"):
            """
            ピーク検出結果を含むスペクトルをプロット
    
            Parameters:
            -----------
            spectrum : SpectralData
                スペクトルデータ
            peaks : list of Peak
                検出されたピーク
            title : str
                プロットタイトル
            """
            fig = go.Figure()
    
            # スペクトル
            fig.add_trace(go.Scatter(
                x=spectrum.x,
                y=spectrum.y,
                mode='lines',
                name='Spectrum',
                line=dict(color='blue', width=2),
                hovertemplate='%{x:.2f}%{y:.3f}'
            ))
    
            # ピーク
            if peaks:
                peak_x = [p.position for p in peaks]
                peak_y = [p.height for p in peaks]
                peak_info = [f"Position: {p.position:.2f}Height: {p.height:.3f}FWHM: {p.width:.2f}"
                            for p in peaks]
    
                fig.add_trace(go.Scatter(
                    x=peak_x,
                    y=peak_y,
                    mode='markers',
                    name=f'Peaks (n={len(peaks)})',
                    marker=dict(color='red', size=12, symbol='diamond'),
                    text=peak_info,
                    hovertemplate='%{text}'
                ))
    
            fig.update_layout(
                title=title,
                xaxis_title=spectrum.x_label,
                yaxis_title=spectrum.y_label,
                hovermode='closest',
                template='plotly_white',
                width=1000,
                height=600
            )
    
            self.fig = fig
            return fig
    
        def plot_comparison_grid(self, spectra_dict: Dict[str, SpectralData],
                                title: str = "Spectral Comparison"):
            """
            複数スペクトルをグリッド表示
    
            Parameters:
            -----------
            spectra_dict : dict
                {label: SpectralData} の辞書
            title : str
                プロットタイトル
            """
            n_spectra = len(spectra_dict)
            rows = (n_spectra + 1) // 2
            cols = 2
    
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=list(spectra_dict.keys()),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
    
            for i, (label, spectrum) in enumerate(spectra_dict.items(), 1):
                row = (i - 1) // cols + 1
                col = (i - 1) % cols + 1
    
                fig.add_trace(
                    go.Scatter(
                        x=spectrum.x,
                        y=spectrum.y,
                        mode='lines',
                        name=label,
                        line=dict(width=2),
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
                fig.update_xaxes(title_text=spectrum.x_label, row=row, col=col)
                fig.update_yaxes(title_text=spectrum.y_label, row=row, col=col)
    
            fig.update_layout(
                title_text=title,
                template='plotly_white',
                width=1200,
                height=300 * rows
            )
    
            self.fig = fig
            return fig
    
        def show(self):
            """プロットを表示"""
            if self.fig:
                self.fig.show()
    
    
    # 使用例
    if __name__ == "__main__":
        # シミュレーションデータ生成
        x = np.linspace(400, 700, 500)
    
        # スペクトル1
        y1 = 0.8 * np.exp(-((x - 500)**2) / (2 * 40**2)) + np.random.normal(0, 0.02, len(x))
        spectrum1 = SpectralData(x, y1, "Wavelength (nm)", "Absorbance")
    
        # スペクトル2
        y2 = 0.6 * np.exp(-((x - 550)**2) / (2 * 35**2)) + np.random.normal(0, 0.02, len(x))
        spectrum2 = SpectralData(x, y2, "Wavelength (nm)", "Absorbance")
    
        # スペクトル3
        y3 = 0.9 * np.exp(-((x - 600)**2) / (2 * 30**2)) + np.random.normal(0, 0.02, len(x))
        spectrum3 = SpectralData(x, y3, "Wavelength (nm)", "Absorbance")
    
        # インタラクティブビューア
        viewer = InteractiveSpectralViewer()
    
        # 単一スペクトル
        print("1. 単一スペクトルのプロット")
        fig1 = viewer.plot_single_spectrum(spectrum1, title="Sample Spectrum")
        # fig1.show()  # ブラウザで表示
    
        # 複数スペクトル重ね合わせ
        print("2. 複数スペクトルの重ね合わせ")
        fig2 = viewer.plot_multiple_spectra(
            [spectrum1, spectrum2, spectrum3],
            ['Sample A (500 nm)', 'Sample B (550 nm)', 'Sample C (600 nm)'],
            title="Multiple Spectra Overlay"
        )
        # fig2.show()
    
        # ピーク検出結果付き
        print("3. ピーク検出結果の表示")
        detector = PeakDetector(spectrum1)
        peaks = detector.detect_peaks(height=0.2, prominence=0.1)
        fig3 = viewer.plot_with_peaks(spectrum1, peaks, title="Spectrum with Peaks")
        # fig3.show()
    
        # グリッド表示
        print("4. グリッド表示")
        spectra_dict = {
            'Sample A': spectrum1,
            'Sample B': spectrum2,
            'Sample C': spectrum3
        }
        fig4 = viewer.plot_comparison_grid(spectra_dict, title="Spectral Comparison Grid")
        # fig4.show()
    
        print("\nインタラクティブプロットの使用方法:")
        print("  - マウスホバーでデータポイントの値を表示")
        print("  - ドラッグでズーム、ダブルクリックでリセット")
        print("  - 凡例クリックで系列の表示/非表示")
    
    5.6 統合解析ワークフローの実装
    5.6.1 エンドツーエンドパイプライン
    これまでのコンポーネントを統合し、データ読み込みから結果出力までの完全なワークフローを構築します。
    
        flowchart LR
            A[生データファイル群] --> B[バッチ読み込み]
            B --> C[前処理ノイズ除去・正規化]
            C --> D[ピーク検出]
            D --> E[特徴量抽出]
            E --> F{解析タイプ}
            F -->|分類| G[機械学習分類器]
            F -->|定量| H[ピーク面積定量]
            F -->|比較| I[統計的比較]
            G --> J[結果可視化]
            H --> J
            I --> J
            J --> K[レポート出力CSV/JSON/HTML]
    
            style A fill:#e3f2fd
            style C fill:#fff3e0
            style F fill:#fce4ec
            style J fill:#e8f5e9
            style K fill:#ffe0b2
        
    コード例6: 統合スペクトル解析パイプライン
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from typing import Dict, List
    import json
    
    class SpectralAnalysisPipeline:
        """
        統合スペクトル解析パイプライン
        """
    
        def __init__(self, config: Dict):
            """
            Parameters:
            -----------
            config : dict
                パイプライン設定
                {
                    'data_dir': Path,
                    'output_dir': Path,
                    'file_pattern': str,
                    'preprocessing': {...},
                    'peak_detection': {...},
                    'analysis_type': 'classification' or 'quantification'
                }
            """
            self.config = config
            self.data_dir = Path(config['data_dir'])
            self.output_dir = Path(config['output_dir'])
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
            self.spectra = []
            self.results = None
    
        def run_pipeline(self):
            """
            パイプライン全体を実行
            """
            print("="*60)
            print("Spectral Analysis Pipeline")
            print("="*60)
    
            # Step 1: データ読み込み
            print("\n[Step 1/5] Loading data...")
            self._load_data()
    
            # Step 2: 前処理
            print("\n[Step 2/5] Preprocessing...")
            self._preprocess()
    
            # Step 3: ピーク検出
            print("\n[Step 3/5] Peak detection...")
            self._detect_peaks()
    
            # Step 4: 解析
            print("\n[Step 4/5] Analysis...")
            self._analyze()
    
            # Step 5: 結果出力
            print("\n[Step 5/5] Exporting results...")
            self._export_results()
    
            print("\n" + "="*60)
            print("Pipeline completed successfully!")
            print("="*60)
    
        def _load_data(self):
            """データ読み込み"""
            processor = BatchProcessor(self.data_dir)
            files = processor.find_files(self.config.get('file_pattern', '*.csv'))
    
            loader = SpectralDataLoader()
            for file in files:
                try:
                    spectrum = loader.auto_load(file)
                    spectrum.metadata['source_file'] = file.name
                    self.spectra.append(spectrum)
                except Exception as e:
                    print(f"  Warning: Failed to load {file.name}: {e}")
    
            print(f"  Loaded {len(self.spectra)} spectra")
    
        def _preprocess(self):
            """前処理"""
            preproc_config = self.config.get('preprocessing', {})
    
            for spectrum in self.spectra:
                # ノイズ除去
                if preproc_config.get('smooth', False):
                    from scipy.signal import savgol_filter
                    spectrum.y = savgol_filter(spectrum.y, window_length=11, polyorder=3)
    
                # トリミング
                if 'trim_range' in preproc_config:
                    x_min, x_max = preproc_config['trim_range']
                    spectrum.trim(x_min, x_max)
    
                # 正規化
                if 'normalize' in preproc_config:
                    spectrum.normalize(method=preproc_config['normalize'])
    
            print(f"  Preprocessed {len(self.spectra)} spectra")
    
        def _detect_peaks(self):
            """ピーク検出"""
            peak_config = self.config.get('peak_detection', {})
    
            for spectrum in self.spectra:
                detector = PeakDetector(spectrum)
                peaks = detector.detect_peaks(
                    height=peak_config.get('height'),
                    prominence=peak_config.get('prominence'),
                    distance=peak_config.get('distance')
                )
                spectrum.metadata['peaks'] = peaks
    
            total_peaks = sum(len(s.metadata['peaks']) for s in self.spectra)
            print(f"  Detected {total_peaks} peaks across {len(self.spectra)} spectra")
    
        def _analyze(self):
            """解析"""
            analysis_type = self.config.get('analysis_type', 'quantification')
    
            if analysis_type == 'quantification':
                self._analyze_quantification()
            elif analysis_type == 'classification':
                self._analyze_classification()
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
    
        def _analyze_quantification(self):
            """定量解析"""
            results = []
    
            for spectrum in self.spectra:
                peaks = spectrum.metadata.get('peaks', [])
    
                result = {
                    'filename': spectrum.metadata.get('source_file', 'unknown'),
                    'n_peaks': len(peaks),
                    'peak_positions': [p.position for p in peaks],
                    'peak_areas': [p.area for p in peaks],
                    'total_area': sum(p.area for p in peaks)
                }
                results.append(result)
    
            self.results = pd.DataFrame(results)
            print(f"  Quantification completed for {len(results)} samples")
    
        def _analyze_classification(self):
            """分類解析"""
            # 特徴量抽出
            classifier = SpectralClassifier(n_estimators=100, use_pca=False)
            X = classifier.extract_features(self.spectra)
    
            # ここでは簡易的にクラスタリング（教師なし分類）
            from sklearn.cluster import KMeans
            n_clusters = self.config.get('n_clusters', 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
    
            results = []
            for i, spectrum in enumerate(self.spectra):
                result = {
                    'filename': spectrum.metadata.get('source_file', 'unknown'),
                    'cluster': int(labels[i]),
                    'n_peaks': len(spectrum.metadata.get('peaks', []))
                }
                results.append(result)
    
            self.results = pd.DataFrame(results)
            print(f"  Classification completed: {n_clusters} clusters identified")
    
        def _export_results(self):
            """結果出力"""
            if self.results is None:
                print("  No results to export")
                return
    
            # CSV出力
            csv_path = self.output_dir / 'results.csv'
            self.results.to_csv(csv_path, index=False)
            print(f"  Saved CSV: {csv_path}")
    
            # JSON出力
            json_path = self.output_dir / 'results.json'
            self.results.to_json(json_path, orient='records', indent=2)
            print(f"  Saved JSON: {json_path}")
    
            # サマリーレポート
            summary = {
                'total_samples': len(self.spectra),
                'analysis_type': self.config.get('analysis_type'),
                'output_files': [str(csv_path), str(json_path)]
            }
    
            summary_path = self.output_dir / 'summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved summary: {summary_path}")
    
    
    # 使用例
    if __name__ == "__main__":
        import tempfile
        import shutil
    
        # 一時ディレクトリ作成
        temp_dir = Path(tempfile.mkdtemp())
        data_dir = temp_dir / 'data'
        output_dir = temp_dir / 'output'
        data_dir.mkdir()
    
        # シミュレーションデータ生成
        for i in range(15):
            x = np.linspace(400, 700, 300)
            y = np.zeros_like(x)
            n_peaks = np.random.randint(1, 4)
            for _ in range(n_peaks):
                pos = np.random.uniform(450, 650)
                width = np.random.uniform(20, 40)
                y += np.random.uniform(0.5, 1.0) * np.exp(-((x - pos)**2) / (2 * width**2))
            y += np.random.normal(0, 0.02, len(x))
    
            filepath = data_dir / f"sample_{i:03d}.csv"
            with open(filepath, 'w') as f:
                for xi, yi in zip(x, y):
                    f.write(f"{xi},{yi}\n")
    
        # パイプライン設定
        config = {
            'data_dir': data_dir,
            'output_dir': output_dir,
            'file_pattern': '*.csv',
            'preprocessing': {
                'smooth': True,
                'trim_range': (420, 680),
                'normalize': 'max'
            },
            'peak_detection': {
                'height': 0.2,
                'prominence': 0.1,
                'distance': 20
            },
            'analysis_type': 'quantification',  # or 'classification'
            'n_clusters': 3
        }
    
        # パイプライン実行
        pipeline = SpectralAnalysisPipeline(config)
        pipeline.run_pipeline()
    
        # 結果確認
        print("\n結果の一部:")
        print(pipeline.results.head())
    
        # クリーンアップ
        shutil.rmtree(temp_dir)
    
    5.7 実践演習：実データ解析プロジェクト
    5.7.1 プロジェクトテンプレート
    本章で学んだツールを使用して、実際の研究データを解析するためのプロジェクトテンプレートを提供します。
    コード例7: プロジェクトテンプレート生成ツール
    import os
    from pathlib import Path
    from typing import Optional
    
    class SpectralProjectTemplate:
        """
        スペクトル解析プロジェクトのテンプレートを生成
        """
    
        @staticmethod
        def create_project(project_name: str, base_dir: Optional[Path] = None):
            """
            プロジェクトディレクトリ構造を作成
    
            Parameters:
            -----------
            project_name : str
                プロジェクト名
            base_dir : Path, optional
                ベースディレクトリ（指定しない場合はカレントディレクトリ）
            """
            if base_dir is None:
                base_dir = Path.cwd()
    
            project_dir = base_dir / project_name
            project_dir.mkdir(parents=True, exist_ok=True)
    
            # ディレクトリ構造
            dirs = [
                'data/raw',
                'data/processed',
                'results',
                'figures',
                'notebooks',
                'scripts',
                'config'
            ]
    
            for dir_path in dirs:
                (project_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
            # READMEファイル
            readme_content = f"""# {project_name}
    
    ## Project Structure
    
    - `data/raw/`: 生データファイル（オリジナル、編集禁止）
    - `data/processed/`: 前処理済みデータ
    - `results/`: 解析結果（CSV, JSON等）
    - `figures/`: グラフ・図表
    - `notebooks/`: Jupyter Notebook
    - `scripts/`: Python解析スクリプト
    - `config/`: 設定ファイル
    
    ## Workflow
    
    1. 生データを `data/raw/` に配置
    2. `scripts/preprocess.py` で前処理実行
    3. `scripts/analyze.py` で解析実行
    4. `results/` に結果出力、`figures/` に図表保存
    
    ## Requirements
    
    ```bash
    pip install numpy scipy matplotlib pandas scikit-learn plotly
    ```
    
    ## Quick Start
    
    ```python
    python scripts/analyze.py --config config/config.json
    ```
    """
            with open(project_dir / 'README.md', 'w') as f:
                f.write(readme_content)
    
            # 設定ファイルテンプレート
            config_content = """{
      "data_dir": "data/raw",
      "output_dir": "results",
      "file_pattern": "*.csv",
      "preprocessing": {
        "smooth": true,
        "trim_range": [400, 700],
        "normalize": "max"
      },
      "peak_detection": {
        "height": 0.2,
        "prominence": 0.1,
        "distance": 20
      },
      "analysis_type": "quantification"
    }
    """
            with open(project_dir / 'config/config.json', 'w') as f:
                f.write(config_content)
    
            # 解析スクリプトテンプレート
            analyze_script = """#!/usr/bin/env python
    # -*- coding: utf-8 -*-
    \"\"\"
    スペクトル解析メインスクリプト
    \"\"\"
    
    import argparse
    import json
    from pathlib import Path
    
    # 以下に本章のSpectralAnalysisPipelineクラスをインポート
    # from spectral_tools import SpectralAnalysisPipeline
    
    def main():
        parser = argparse.ArgumentParser(description='Spectral Analysis Pipeline')
        parser.add_argument('--config', type=str, required=True,
                           help='Path to config JSON file')
        args = parser.parse_args()
    
        # 設定読み込み
        with open(args.config, 'r') as f:
            config = json.load(f)
    
        # パイプライン実行
        # pipeline = SpectralAnalysisPipeline(config)
        # pipeline.run_pipeline()
    
        print("Analysis completed!")
    
    if __name__ == "__main__":
        main()
    """
            with open(project_dir / 'scripts/analyze.py', 'w') as f:
                f.write(analyze_script)
    
            # .gitignoreファイル
            gitignore_content = """# Data files
    data/raw/*
    !data/raw/.gitkeep
    *.csv
    *.txt
    *.dat
    
    # Python
    __pycache__/
    *.py[cod]
    *.so
    .ipynb_checkpoints/
    
    # Results
    results/*
    !results/.gitkeep
    figures/*
    !figures/.gitkeep
    """
            with open(project_dir / '.gitignore', 'w') as f:
                f.write(gitignore_content)
    
            # .gitkeepファイル（空ディレクトリをGit管理下に置くため）
            for dir_path in ['data/raw', 'results', 'figures']:
                (project_dir / dir_path / '.gitkeep').touch()
    
            print(f"✅ Project '{project_name}' created successfully!")
            print(f"📁 Location: {project_dir}")
            print(f"\n次のステップ:")
            print(f"  1. cd {project_dir}")
            print(f"  2. データファイルを data/raw/ に配置")
            print(f"  3. config/config.json を編集")
            print(f"  4. python scripts/analyze.py --config config/config.json")
    
    
    # 使用例
    if __name__ == "__main__":
        # プロジェクト作成
        SpectralProjectTemplate.create_project("my_spectroscopy_project")
    
    5.8 演習問題
    
    基礎問題（Easy）
    問題1: SpectralDataクラスの使用
    波長400-700 nm、100点のデータを持つSpectralDataオブジェクトを作成し、500-600 nmの範囲にトリミングした後、最大値で正規化せよ。
    
    解答を見る
    
    解答:
    import numpy as np
    
    # データ生成
    x = np.linspace(400, 700, 100)
    y = np.exp(-((x - 550)**2) / (2 * 50**2))
    
    # SpectralDataオブジェクト作成
    spectrum = SpectralData(x, y, "Wavelength (nm)", "Intensity")
    
    # トリミング
    spectrum.trim(500, 600)
    
    # 正規化
    spectrum.normalize(method='max')
    
    print(f"トリミング後のデータ点数: {len(spectrum.x)}")
    print(f"正規化後の最大値: {spectrum.y.max()}")  # 1.0
    
    
    
    問題2: ピーク検出の基本
    シミュレーションデータから、高さ0.3以上、卓越度0.2以上のピークを検出し、各ピークの位置と高さを出力せよ。
    
    解答を見る
    
    解答:
    # シミュレーションデータ
    x = np.linspace(400, 700, 500)
    y = (0.8 * np.exp(-((x - 500)**2) / (2 * 30**2)) +
         0.5 * np.exp(-((x - 600)**2) / (2 * 40**2)))
    
    spectrum = SpectralData(x, y, "Wavelength (nm)", "Intensity")
    
    # ピーク検出
    detector = PeakDetector(spectrum)
    peaks = detector.detect_peaks(height=0.3, prominence=0.2)
    
    print(f"検出されたピーク数: {len(peaks)}")
    for i, peak in enumerate(peaks, 1):
        print(f"  Peak {i}: position={peak.position:.1f} nm, height={peak.height:.3f}")
    
    
    
    問題3: データローダーの使用
    CSVファイル（2列: 波長、強度）からスペクトルデータを読み込み、データ点数と波長範囲を出力せよ。
    
    解答を見る
    
    解答:
    # データローダー
    loader = SpectralDataLoader()
    spectrum = loader.load_csv('sample.csv', x_col=0, y_col=1,
                              x_label="Wavelength (nm)", y_label="Intensity")
    
    print(f"データ点数: {len(spectrum.x)}")
    print(f"波長範囲: {spectrum.x.min():.1f} - {spectrum.x.max():.1f} nm")
    print(f"強度範囲: {spectrum.y.min():.3f} - {spectrum.y.max():.3f}")
    
    
    
    
    
    中級問題（Medium）
    問題4: カスタム前処理パイプライン
    以下の前処理を順次適用する関数を作成せよ：(1) Savitzky-Golay平滑化、(2) ベースライン補正（線形）、(3) 最大値正規化。
    
    解答を見る
    
    解答:
    from scipy.signal import savgol_filter
    from scipy.stats import linregress
    
    def custom_preprocessing_pipeline(spectrum: SpectralData) -> SpectralData:
        """
        カスタム前処理パイプライン
    
        Parameters:
        -----------
        spectrum : SpectralData
            入力スペクトル
    
        Returns:
        --------
        processed_spectrum : SpectralData
            前処理済みスペクトル
        """
        # コピーを作成
        processed = spectrum.copy()
    
        # (1) Savitzky-Golay平滑化
        processed.y = savgol_filter(processed.y, window_length=11, polyorder=3)
    
        # (2) ベースライン補正（線形フィット）
        # 両端10%のデータでベースラインを推定
        n = len(processed.x)
        baseline_indices = list(range(int(n * 0.1))) + list(range(int(n * 0.9), n))
        x_base = processed.x[baseline_indices]
        y_base = processed.y[baseline_indices]
    
        slope, intercept, _, _, _ = linregress(x_base, y_base)
        baseline = slope * processed.x + intercept
        processed.y = processed.y - baseline
    
        # (3) 最大値正規化
        processed.normalize(method='max')
    
        return processed
    
    # 使用例
    x = np.linspace(400, 700, 300)
    y = 0.8 * np.exp(-((x - 550)**2) / (2 * 40**2)) + 0.02 * x + np.random.normal(0, 0.02, len(x))
    spectrum = SpectralData(x, y, "Wavelength (nm)", "Intensity")
    
    processed_spectrum = custom_preprocessing_pipeline(spectrum)
    
    print("前処理完了:")
    print(f"  元のy範囲: [{spectrum.y.min():.3f}, {spectrum.y.max():.3f}]")
    print(f"  処理後のy範囲: [{processed_spectrum.y.min():.3f}, {processed_spectrum.y.max():.3f}]")
    
    
    
    問題5: バッチ処理による統計解析
    複数のスペクトルファイルを読み込み、各ファイルのピーク数、最大ピーク高さ、平均強度を計算し、DataFrameとして出力せよ。
    
    解答を見る
    
    解答:
    def batch_statistical_analysis(spectrum: SpectralData) -> Dict:
        """
        スペクトルの統計解析
    
        Parameters:
        -----------
        spectrum : SpectralData
            スペクトルデータ
    
        Returns:
        --------
        stats : dict
            統計情報
        """
        # ピーク検出
        detector = PeakDetector(spectrum)
        peaks = detector.detect_peaks(height=0.1, prominence=0.05)
    
        # 統計計算
        max_peak_height = max([p.height for p in peaks]) if peaks else 0.0
        mean_intensity = np.mean(spectrum.y)
    
        return {
            'n_peaks': len(peaks),
            'max_peak_height': max_peak_height,
            'mean_intensity': mean_intensity,
            'intensity_std': np.std(spectrum.y)
        }
    
    # バッチ処理
    processor = BatchProcessor('data_directory')
    files = processor.find_files("*.csv")
    results_df = processor.batch_process(files, batch_statistical_analysis, n_workers=2)
    
    print("統計解析結果:")
    print(results_df.describe())
    
    
    
    問題6: 時系列スペクトル解析
    時間的に連続測定された複数スペクトルから、特定ピーク（例: 550 nm）の高さの時間変化をプロットせよ。
    
    解答を見る
    
    解答:
    import matplotlib.pyplot as plt
    
    def analyze_time_series_spectra(spectra_list: List[SpectralData],
                                    target_wavelength: float = 550.0) -> np.ndarray:
        """
        時系列スペクトルから特定波長の強度変化を抽出
    
        Parameters:
        -----------
        spectra_list : list of SpectralData
            時系列スペクトルのリスト
        target_wavelength : float
            追跡する波長 (nm)
    
        Returns:
        --------
        intensities : array
            各時刻における強度
        """
        intensities = []
    
        for spectrum in spectra_list:
            # 最も近い波長のインデックスを取得
            idx = np.argmin(np.abs(spectrum.x - target_wavelength))
            intensities.append(spectrum.y[idx])
    
        return np.array(intensities)
    
    # 使用例（シミュレーション）
    time_points = np.arange(0, 60, 2)  # 0-60分、2分間隔
    spectra_series = []
    
    for t in time_points:
        x = np.linspace(400, 700, 300)
        # 時間とともにピーク高さが減衰
        decay = np.exp(-t / 30)
        y = 0.8 * decay * np.exp(-((x - 550)**2) / (2 * 40**2))
        spectra_series.append(SpectralData(x, y, "Wavelength (nm)", "Intensity"))
    
    # 550 nmのピーク追跡
    intensities = analyze_time_series_spectra(spectra_series, target_wavelength=550)
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, intensities, 'o-', linewidth=2, markersize=8)
    plt.xlabel('時間 (分)', fontsize=12)
    plt.ylabel('強度 @ 550 nm', fontsize=12)
    plt.title('時系列スペクトル解析：ピーク強度の時間変化', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"初期強度: {intensities[0]:.3f}")
    print(f"最終強度: {intensities[-1]:.3f}")
    print(f"減衰率: {(1 - intensities[-1]/intensities[0]) * 100:.1f}%")
    
    
    
    
    
    上級問題（Hard）
    問題7: 深層学習によるスペクトル分類（CNN）
    1D畳み込みニューラルネットワーク（CNN）を構築し、スペクトルデータを3クラスに分類するモデルを訓練せよ。TensorFlow/Kerasを使用すること。
    
    解答を見る
    
    解答:
    # TensorFlow/Kerasが必要: pip install tensorflow
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    def build_1d_cnn(input_length: int, n_classes: int):
        """
        1D CNNモデルの構築
    
        Parameters:
        -----------
        input_length : int
            入力スペクトルの長さ
        n_classes : int
            クラス数
    
        Returns:
        --------
        model : keras.Model
            CNNモデル
        """
        model = keras.Sequential([
            # 入力層
            layers.Input(shape=(input_length, 1)),
    
            # 畳み込み層1
            layers.Conv1D(filters=32, kernel_size=7, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
    
            # 畳み込み層2
            layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
    
            # 畳み込み層3
            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            layers.GlobalAveragePooling1D(),
    
            # 全結合層
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
    
            # 出力層
            layers.Dense(n_classes, activation='softmax')
        ])
    
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
        return model
    
    # データ準備（シミュレーション）
    X_train = []  # shape: (n_samples, seq_length, 1)
    y_train = []
    
    for class_label in range(3):
        for _ in range(100):
            x = np.linspace(0, 1, 100)
            # クラスごとに異なるスペクトル形状
            if class_label == 0:
                y = np.exp(-((x - 0.5)**2) / 0.1)
            elif class_label == 1:
                y = np.exp(-((x - 0.3)**2) / 0.05) + np.exp(-((x - 0.7)**2) / 0.05)
            else:
                y = np.sin(x * 10) * 0.5 + 0.5
    
            y += np.random.normal(0, 0.05, len(x))
            X_train.append(y)
            y_train.append(class_label)
    
    X_train = np.array(X_train).reshape(-1, 100, 1)
    y_train = np.array(y_train)
    
    # モデル訓練
    model = build_1d_cnn(input_length=100, n_classes=3)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                       validation_split=0.2, verbose=0)
    
    print(f"最終訓練精度: {history.history['accuracy'][-1]:.3f}")
    print(f"最終検証精度: {history.history['val_accuracy'][-1]:.3f}")
    
    # 学習曲線プロット
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='訓練精度')
    plt.plot(history.history['val_accuracy'], label='検証精度')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CNN学習曲線')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    答え: 高精度（90%以上）でスペクトル分類を達成
    
    
    問題8: リアルタイムスペクトル監視システム
    新しいスペクトルファイルがディレクトリに追加されたら自動的に解析を実行し、異常検知（通常パターンからの逸脱）を行うシステムを実装せよ。
    
    解答を見る
    
    解答:
    import time
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    
    class SpectralMonitor(FileSystemEventHandler):
        """
        スペクトルファイル監視・異常検知システム
        """
    
        def __init__(self, reference_spectra: List[SpectralData], threshold: float = 0.3):
            """
            Parameters:
            -----------
            reference_spectra : list of SpectralData
                正常データの参照スペクトル
            threshold : float
                異常検知閾値（距離）
            """
            self.reference_spectra = reference_spectra
            self.threshold = threshold
    
            # 参照スペクトルの平均を計算
            X_ref = np.array([s.y for s in reference_spectra])
            self.reference_mean = np.mean(X_ref, axis=0)
            self.reference_std = np.std(X_ref, axis=0)
    
        def on_created(self, event):
            """ファイル作成時のコールバック"""
            if event.is_directory:
                return
    
            if event.src_path.endswith('.csv'):
                print(f"\n新しいファイル検出: {event.src_path}")
                time.sleep(0.5)  # ファイル書き込み完了待ち
                self.analyze_new_spectrum(event.src_path)
    
        def analyze_new_spectrum(self, filepath: str):
            """新しいスペクトルを解析"""
            try:
                # データ読み込み
                loader = SpectralDataLoader()
                spectrum = loader.auto_load(filepath)
    
                # 異常検知（ユークリッド距離）
                distance = np.linalg.norm(spectrum.y - self.reference_mean)
    
                if distance > self.threshold:
                    print(f"⚠️  異常検出！ 距離: {distance:.3f} (閾値: {self.threshold:.3f})")
                    print(f"    → アラート送信: {Path(filepath).name}")
                else:
                    print(f"✅ 正常範囲内。距離: {distance:.3f}")
    
            except Exception as e:
                print(f"❌ エラー: {e}")
    
    # 使用例（実際の使用時にはコメント解除）
    """
    # 参照データ準備
    reference_spectra = [...]  # 正常スペクトルのリスト
    
    # 監視システム起動
    monitor = SpectralMonitor(reference_spectra, threshold=0.3)
    observer = Observer()
    observer.schedule(monitor, path='./watch_directory', recursive=False)
    observer.start()
    
    print("監視開始...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    """
    
    print("リアルタイム監視システムのテンプレート実装完了")
    
    答え: watchdogライブラリを使用したリアルタイム監視システム
    
    
    問題9: スペクトルデータベースと検索システム
    大量のスペクトルデータをSQLiteデータベースに保存し、類似スペクトル検索機能を実装せよ。
    
    解答を見る
    
    解答:
    import sqlite3
    import pickle
    
    class SpectralDatabase:
        """
        スペクトルデータベース管理クラス
        """
    
        def __init__(self, db_path: str = 'spectra.db'):
            """
            Parameters:
            -----------
            db_path : str
                データベースファイルパス
            """
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path)
            self._create_table()
    
        def _create_table(self):
            """テーブル作成"""
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS spectra (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    x_data BLOB,
                    y_data BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
    
        def insert_spectrum(self, spectrum: SpectralData):
            """スペクトルを挿入"""
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO spectra (filename, x_data, y_data, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                spectrum.metadata.get('source_file', 'unknown'),
                pickle.dumps(spectrum.x),
                pickle.dumps(spectrum.y),
                json.dumps(spectrum.metadata)
            ))
            self.conn.commit()
            return cursor.lastrowid
    
        def search_similar_spectra(self, query_spectrum: SpectralData, top_k: int = 5):
            """
            類似スペクトルを検索
    
            Parameters:
            -----------
            query_spectrum : SpectralData
                クエリスペクトル
            top_k : int
                上位k件を返す
    
            Returns:
            --------
            results : list of tuples
                (id, filename, similarity_score)
            """
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, filename, y_data FROM spectra")
    
            results = []
            for row in cursor.fetchall():
                id, filename, y_blob = row
                y_data = pickle.loads(y_blob)
    
                # コサイン類似度
                similarity = np.dot(query_spectrum.y, y_data) / \
                            (np.linalg.norm(query_spectrum.y) * np.linalg.norm(y_data))
    
                results.append((id, filename, similarity))
    
            # 類似度でソート
            results.sort(key=lambda x: x[2], reverse=True)
    
            return results[:top_k]
    
        def close(self):
            """データベース接続を閉じる"""
            self.conn.close()
    
    
    # 使用例
    db = SpectralDatabase('test_spectra.db')
    
    # スペクトル挿入
    for i in range(10):
        x = np.linspace(400, 700, 100)
        y = np.exp(-((x - (500 + i*10))**2) / (2 * 30**2))
        spectrum = SpectralData(x, y, "Wavelength", "Intensity")
        spectrum.metadata['source_file'] = f"sample_{i}.csv"
        db.insert_spectrum(spectrum)
    
    # 類似検索
    query_x = np.linspace(400, 700, 100)
    query_y = np.exp(-((query_x - 550)**2) / (2 * 30**2))
    query_spectrum = SpectralData(query_x, query_y, "Wavelength", "Intensity")
    
    similar_spectra = db.search_similar_spectra(query_spectrum, top_k=3)
    
    print("類似スペクトル検索結果:")
    for id, filename, score in similar_spectra:
        print(f"  ID: {id}, ファイル: {filename}, 類似度: {score:.3f}")
    
    db.close()
    
    答え: SQLiteを使用したスペクトルデータベースと類似検索システム
    
    
    
    
    学習目標の確認
    以下の項目について、自己評価してください：
    レベル1: 基本理解
    
    SpectralDataクラスの基本的な使い方を理解している
    データローダーでCSV/TXTファイルを読み込める
    自動ピーク検出の基本パラメータを理解している
    バッチ処理の概念を理解している
    
    レベル2: 実践スキル
    
    カスタム前処理パイプラインを構築できる
    複数ファイルのバッチ処理を実装できる
    機械学習による分光スペクトル分類ができる
    Plotlyでインタラクティブな可視化ができる
    統合解析パイプラインを構築できる
    
    レベル3: 応用力
    
    深層学習（CNN）によるスペクトル分類ができる
    リアルタイム監視・異常検知システムを実装できる
    スペクトルデータベースと類似検索システムを構築できる
    実際の研究プロジェクトに適用できる
    
    
    
    参考文献
    
    McKinney, W. (2017). Python for Data Analysis (2nd ed.). O'Reilly Media, pp. 89-95 (DataFrame operations), pp. 125-145 (data cleaning), pp. 263-270 (time series), pp. 310-325 (aggregation). - Pandas、NumPyを用いたデータ処理の実践的解説
    VanderPlas, J. (2023). Python Data Science Handbook (2nd ed.). O'Reilly Media, pp. 200-225 (NumPy), pp. 280-310 (pandas), pp. 330-365 (matplotlib), pp. 400-435 (scikit-learn), pp. 470-500 (dimensionality reduction). - 機械学習、scikit-learn、データ可視化の包括的ガイド
    Eilers, P. H. C., Boelens, H. F. M. (2005). Baseline correction with asymmetric least squares smoothing. Analytical Chemistry, 77(21), 6729-6736. DOI: 10.1021/ac051370e - 非対称最小二乗法によるベースライン補正アルゴリズム（コード例で実装）
    Geladi, P., Kowalski, B. R. (1986). Partial least-squares regression: a tutorial. Analytica Chimica Acta, 185, 1-17. DOI: 10.1016/0003-2670(86)80028-9 - ケモメトリクス、PLS回帰によるスペクトル解析の基礎
    SciPy 1.11 documentation. scipy.signal.find_peaks, scipy.signal.savgol_filter, scipy.signal.peak_widths. https://docs.scipy.org/doc/scipy/reference/signal.html - 信号処理アルゴリズム、ピーク検出、フィルタリング
    scikit-learn 1.3 documentation. RandomForestClassifier, Pipeline, StandardScaler. https://scikit-learn.org/stable/modules/ensemble.html - ランダムフォレスト、機械学習パイプライン構築
    Plotly 5.x documentation. plotly.express, plotly.graph_objects. https://plotly.com/python/ - インタラクティブ可視化ライブラリの公式ドキュメント
    Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O'Reilly Media, pp. 190-220 (ensemble methods), pp. 450-480 (CNNs), pp. 510-535 (time series with deep learning). - 深層学習、CNNによる時系列・スペクトルデータ解析
    pandas 2.0 documentation. Chunking large datasets, Dask integration. https://pandas.pydata.org/docs/user_guide/scale.html - 大規模データのバッチ処理、最適化手法
    
    
    
    
    免責事項
    
    本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
    本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
    外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
    本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
    本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
    本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
    ```
