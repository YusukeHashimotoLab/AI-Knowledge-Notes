---
title: 第5章：Pythonとシミュレーターの連携
chapter_title: 第5章：Pythonとシミュレーターの連携
subtitle: 自動化、最適化、機械学習の統合ワークフロー
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ DWSIM等のシミュレーターをPythonから操作できる
  * ✅ 自動化されたパラメータスイープとデータ抽出を実装できる
  * ✅ pandasでシミュレーション結果を効率的に解析できる
  * ✅ scipyによる最適化とシミュレーターの統合ができる
  * ✅ 機械学習モデルでプロセス予測を高速化できる
  * ✅ 完全な自動化ワークフロー（simulation → analysis → optimization）を構築できる

* * *

## 5.1 DWSIM Pythonインターフェース基礎

### DWSIMとは

**DWSIM** は、オープンソースのプロセスシミュレーターです（Aspen HYSYSやAspen Plusの代替）。Pythonから操作することで、自動化と高度な解析が可能になります。

> **注意** ：DWSIMはWindows環境で最も安定して動作します。Linuxでは`mono`、macOSではWine経由で実行可能ですが、本章ではWindows前提でコード例を示します。 
    
    
    """
    Example 1: DWSIM Python Interface Basics
    DWSIM Pythonインターフェースの基礎
    """
    import os
    import sys
    import clr  # pythonnet (pip install pythonnet)
    
    # DWSIMのパスを追加（環境に応じて変更）
    DWSIM_PATH = r"C:\Program Files\DWSIM\DWSIM.exe"
    sys.path.append(os.path.dirname(DWSIM_PATH))
    
    # DWSIM .NET アセンブリをロード
    clr.AddReference("DWSIM.Automation")
    clr.AddReference("DWSIM.Interfaces")
    clr.AddReference("DWSIM.Thermodynamics")
    
    from DWSIM.Automation import Automation3
    from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType
    
    class DWSIMInterface:
        """DWSIM Python インターフェース"""
    
        def __init__(self):
            """DWSIMオートメーションオブジェクトを初期化"""
            self.interf = Automation3()
            self.flowsheet = None
    
        def create_flowsheet(self, name="PythonFlowsheet"):
            """新しいフローシートを作成"""
            self.flowsheet = self.interf.CreateFlowsheet()
            self.flowsheet.Options.SelectedPropertyPackage = "Peng-Robinson"
            print(f"Flowsheet created: {name}")
            return self.flowsheet
    
        def add_compound(self, compound_name):
            """化合物を追加"""
            self.flowsheet.AddComponent(compound_name)
            print(f"Added compound: {compound_name}")
    
        def add_material_stream(self, name, temperature=298.15,
                               pressure=101325, mass_flow=1000,
                               composition=None):
            """
            物質ストリームを追加
    
            Args:
                name: ストリーム名
                temperature: 温度 [K]
                pressure: 圧力 [Pa]
                mass_flow: 質量流量 [kg/h]
                composition: 組成 {compound: mole_fraction}
            """
            stream = self.flowsheet.AddObject(ObjectType.MaterialStream, name)
    
            # 条件設定
            stream.SetTemperature(temperature)
            stream.SetPressure(pressure)
            stream.SetMassFlow(mass_flow)
    
            # 組成設定
            if composition:
                comp_list = []
                for compound, frac in composition.items():
                    comp_list.append(frac)
                stream.SetComposition(comp_list)
    
            print(f"Added material stream: {name}")
            return stream
    
        def calculate(self):
            """フローシートを計算"""
            self.flowsheet.Solve()
            print("Flowsheet calculated successfully")
    
        def get_stream_property(self, stream_name, property_name):
            """ストリームのプロパティを取得"""
            stream = self.flowsheet.GetFlowsheetObject(stream_name)
    
            if property_name == "Temperature":
                return stream.GetTemperature()
            elif property_name == "Pressure":
                return stream.GetPressure()
            elif property_name == "MassFlow":
                return stream.GetMassFlow()
            elif property_name == "Composition":
                return stream.GetComposition()
            else:
                raise ValueError(f"Unknown property: {property_name}")
    
        def save_flowsheet(self, filepath):
            """フローシートを保存"""
            self.flowsheet.SaveToFile(filepath)
            print(f"Flowsheet saved to: {filepath}")
    
        def load_flowsheet(self, filepath):
            """フローシートをロード"""
            self.flowsheet = self.interf.LoadFlowsheet(filepath)
            print(f"Flowsheet loaded from: {filepath}")
            return self.flowsheet
    
    # 使用例
    if __name__ == "__main__":
        # DWSIMインターフェース初期化
        dwsim = DWSIMInterface()
    
        # 新規フローシート作成
        fs = dwsim.create_flowsheet("SimpleFlowsheet")
    
        # 化合物追加
        dwsim.add_compound("Water")
        dwsim.add_compound("Ethanol")
    
        # 物質ストリーム追加
        stream1 = dwsim.add_material_stream(
            "Feed",
            temperature=298.15,
            pressure=101325,
            mass_flow=1000,
            composition={"Water": 0.5, "Ethanol": 0.5}
        )
    
        # 計算実行
        dwsim.calculate()
    
        # 結果取得
        temp = dwsim.get_stream_property("Feed", "Temperature")
        print(f"Feed temperature: {temp:.2f} K")
    
        # 保存
        dwsim.save_flowsheet("simple_flowsheet.dwxmz")
    

* * *

## 5.2 自動フローシート作成

### Pythonスクリプトでプロセスを構築

手作業でGUIを操作する代わりに、Pythonスクリプトで完全なフローシートを自動生成します。
    
    
    """
    Example 2: Automated Flowsheet Creation
    自動フローシート作成（蒸留塔のセットアップ）
    """
    from dwsim_interface import DWSIMInterface
    
    class DistillationColumnBuilder:
        """蒸留塔を自動構築"""
    
        def __init__(self, dwsim_interface):
            self.dwsim = dwsim_interface
            self.fs = None
    
        def build_simple_distillation(self):
            """簡単な蒸留塔フローシートを構築"""
    
            # 1. フローシート作成
            self.fs = self.dwsim.create_flowsheet("Distillation")
    
            # 2. 化合物追加
            compounds = ["Benzene", "Toluene", "Ethylbenzene"]
            for comp in compounds:
                self.dwsim.add_compound(comp)
    
            # 3. フィードストリーム
            feed = self.dwsim.add_material_stream(
                "Feed",
                temperature=350,  # K
                pressure=101325,  # Pa
                mass_flow=10000,  # kg/h
                composition={
                    "Benzene": 0.33,
                    "Toluene": 0.34,
                    "Ethylbenzene": 0.33
                }
            )
    
            # 4. 蒸留塔追加
            column = self.fs.AddObject(ObjectType.DistillationColumn,
                                       "Column-01")
    
            # 蒸留塔の仕様設定
            column.NumberOfStages = 20
            column.CondenserType = 0  # Total condenser
            column.ReboilerType = 0   # Kettle reboiler
            column.FeedStage = 10
    
            # 運転条件
            column.RefluxRatio = 2.0
            column.BottomsFlowRate = 6500  # kg/h
    
            # 5. ストリーム接続
            # Feed → Column
            self.fs.ConnectObjects(feed.Name, column.Name, 0, 0)
    
            # Column → Distillate
            distillate = self.dwsim.add_material_stream("Distillate")
            self.fs.ConnectObjects(column.Name, distillate.Name, 0, 0)
    
            # Column → Bottoms
            bottoms = self.dwsim.add_material_stream("Bottoms")
            self.fs.ConnectObjects(column.Name, bottoms.Name, 1, 0)
    
            print("Distillation flowsheet built successfully")
    
            return self.fs
    
        def run_and_extract_results(self):
            """計算実行と結果抽出"""
            # 計算実行
            self.dwsim.calculate()
    
            # 結果抽出
            results = {
                'distillate': {
                    'T': self.dwsim.get_stream_property("Distillate", "Temperature"),
                    'P': self.dwsim.get_stream_property("Distillate", "Pressure"),
                    'F': self.dwsim.get_stream_property("Distillate", "MassFlow"),
                    'comp': self.dwsim.get_stream_property("Distillate", "Composition")
                },
                'bottoms': {
                    'T': self.dwsim.get_stream_property("Bottoms", "Temperature"),
                    'P': self.dwsim.get_stream_property("Bottoms", "Pressure"),
                    'F': self.dwsim.get_stream_property("Bottoms", "MassFlow"),
                    'comp': self.dwsim.get_stream_property("Bottoms", "Composition")
                }
            }
    
            return results
    
    # 使用例
    dwsim = DWSIMInterface()
    builder = DistillationColumnBuilder(dwsim)
    
    # フローシート構築
    flowsheet = builder.build_simple_distillation()
    
    # 計算と結果取得
    results = builder.run_and_extract_results()
    
    print("\n=== Simulation Results ===")
    print(f"Distillate flow: {results['distillate']['F']:.2f} kg/h")
    print(f"Bottoms flow: {results['bottoms']['F']:.2f} kg/h")
    

* * *

## 5.3 パラメータスイープとデータ抽出

### 大量のケース計算を自動化

設計パラメータの範囲をスキャンし、最適な運転条件を探索します。
    
    
    """
    Example 3: Parameter Sweep and Data Extraction
    パラメータスイープとデータ抽出
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    class ParameterSweep:
        """パラメータスイープ自動化"""
    
        def __init__(self, dwsim_interface):
            self.dwsim = dwsim_interface
            self.results = []
    
        def sweep_reflux_ratio(self, reflux_range):
            """
            還流比をスイープして性能評価
    
            Args:
                reflux_range: 還流比の範囲（配列）
    
            Returns:
                DataFrame: 結果データ
            """
            for reflux in reflux_range:
                print(f"Calculating reflux ratio = {reflux:.2f}...")
    
                # 還流比を設定
                column = self.dwsim.flowsheet.GetFlowsheetObject("Column-01")
                column.RefluxRatio = reflux
    
                try:
                    # 計算実行
                    self.dwsim.calculate()
    
                    # 結果取得
                    dist_flow = self.dwsim.get_stream_property("Distillate", "MassFlow")
                    dist_comp = self.dwsim.get_stream_property("Distillate", "Composition")
                    btms_comp = self.dwsim.get_stream_property("Bottoms", "Composition")
    
                    # Benzene purity in distillate
                    benzene_purity = dist_comp[0]  # Assuming Benzene is first
    
                    # Reboiler duty (simplified)
                    reboiler_duty = column.ReboilerDuty / 1000  # kW
    
                    self.results.append({
                        'reflux_ratio': reflux,
                        'distillate_flow': dist_flow,
                        'benzene_purity': benzene_purity,
                        'reboiler_duty': reboiler_duty
                    })
    
                except Exception as e:
                    print(f"  ⚠ Calculation failed: {e}")
                    self.results.append({
                        'reflux_ratio': reflux,
                        'distillate_flow': np.nan,
                        'benzene_purity': np.nan,
                        'reboiler_duty': np.nan
                    })
    
            return pd.DataFrame(self.results)
    
        def plot_results(self, df):
            """結果をプロット"""
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
            # Distillate flow
            axes[0].plot(df['reflux_ratio'], df['distillate_flow'], 'o-')
            axes[0].set_xlabel('Reflux Ratio')
            axes[0].set_ylabel('Distillate Flow [kg/h]')
            axes[0].grid(True, alpha=0.3)
    
            # Benzene purity
            axes[1].plot(df['reflux_ratio'], df['benzene_purity'] * 100, 'o-')
            axes[1].set_xlabel('Reflux Ratio')
            axes[1].set_ylabel('Benzene Purity [%]')
            axes[1].axhline(y=95, color='r', linestyle='--', label='Target 95%')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
    
            # Reboiler duty
            axes[2].plot(df['reflux_ratio'], df['reboiler_duty'], 'o-', color='orangered')
            axes[2].set_xlabel('Reflux Ratio')
            axes[2].set_ylabel('Reboiler Duty [kW]')
            axes[2].grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
    # 使用例
    sweep = ParameterSweep(dwsim)
    
    # 還流比を1.5から5.0まで変化
    reflux_range = np.linspace(1.5, 5.0, 10)
    
    # スイープ実行
    df_results = sweep.sweep_reflux_ratio(reflux_range)
    
    # 結果表示
    print(df_results)
    
    # プロット
    sweep.plot_results(df_results)
    
    # CSVで保存
    df_results.to_csv("reflux_sweep_results.csv", index=False)
    print("Results saved to reflux_sweep_results.csv")
    

* * *

## 5.4 pandasによるデータ解析

### シミュレーション結果の統計解析

pandasを活用して、大量のシミュレーション結果を効率的に解析します。
    
    
    """
    Example 4: Data Analysis with Pandas
    pandasによるシミュレーション結果の解析
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    class SimulationDataAnalyzer:
        """シミュレーションデータ解析ツール"""
    
        def __init__(self, csv_file=None):
            """
            Args:
                csv_file: 結果CSVファイルのパス
            """
            if csv_file:
                self.df = pd.read_csv(csv_file)
            else:
                self.df = None
    
        def load_multiple_sweeps(self, file_pattern):
            """
            複数のスイープ結果を統合
    
            Args:
                file_pattern: ファイルパターン（例: "sweep_*.csv"）
            """
            import glob
            files = glob.glob(file_pattern)
    
            dfs = []
            for file in files:
                df = pd.read_csv(file)
                dfs.append(df)
    
            self.df = pd.concat(dfs, ignore_index=True)
            print(f"Loaded {len(files)} files, {len(self.df)} total rows")
    
        def calculate_economics(self, product_price=1000,
                               energy_cost=0.05):
            """
            経済性指標を計算
    
            Args:
                product_price: 製品価格 [$/ton]
                energy_cost: エネルギーコスト [$/kWh]
            """
            # Revenue
            self.df['revenue'] = (self.df['distillate_flow'] / 1000) * \
                                product_price * \
                                self.df['benzene_purity']
    
            # Energy cost
            self.df['energy_cost'] = self.df['reboiler_duty'] * energy_cost
    
            # Profit
            self.df['profit'] = self.df['revenue'] - self.df['energy_cost']
    
            return self.df
    
        def find_optimal_condition(self, objective='profit',
                                  constraint_col='benzene_purity',
                                  constraint_val=0.95):
            """
            最適条件を探索
    
            Args:
                objective: 最大化する指標
                constraint_col: 制約カラム
                constraint_val: 制約値
            """
            # 制約を満たすデータのみ
            df_valid = self.df[self.df[constraint_col] >= constraint_val]
    
            if len(df_valid) == 0:
                print("No solutions satisfy the constraint")
                return None
    
            # 最適解
            optimal_idx = df_valid[objective].idxmax()
            optimal = df_valid.loc[optimal_idx]
    
            print("\n=== Optimal Condition ===")
            print(optimal)
    
            return optimal
    
        def sensitivity_analysis(self):
            """感度分析（相関行列）"""
            # 数値列のみ選択
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
    
            # 相関行列
            corr_matrix = self.df[numeric_cols].corr()
    
            # ヒートマップ
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                       center=0, vmin=-1, vmax=1)
            plt.title('Correlation Matrix - Sensitivity Analysis')
            plt.tight_layout()
            plt.show()
    
            return corr_matrix
    
        def plot_pareto_front(self, obj1='benzene_purity', obj2='profit'):
            """
            パレートフロント（トレードオフ曲線）
    
            Args:
                obj1: 目的関数1
                obj2: 目的関数2
            """
            plt.figure(figsize=(8, 6))
            plt.scatter(self.df[obj1] * 100, self.df[obj2],
                       c=self.df['reflux_ratio'], cmap='viridis',
                       s=80, alpha=0.7)
            plt.colorbar(label='Reflux Ratio')
            plt.xlabel(f'{obj1} [%]')
            plt.ylabel(f'{obj2} [$/h]')
            plt.title('Pareto Front: Purity vs Profit')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    # 使用例
    analyzer = SimulationDataAnalyzer("reflux_sweep_results.csv")
    
    # 経済性計算
    df_econ = analyzer.calculate_economics(
        product_price=1000,
        energy_cost=0.05
    )
    
    # 最適条件探索
    optimal = analyzer.find_optimal_condition(
        objective='profit',
        constraint_col='benzene_purity',
        constraint_val=0.95
    )
    
    # 感度分析
    corr = analyzer.sensitivity_analysis()
    
    # パレートフロント
    analyzer.plot_pareto_front('benzene_purity', 'profit')
    

* * *

## 5.5 scipy最適化との統合

### シミュレーターと数値最適化の連携

scipyの最適化アルゴリズムとDWSIMを組み合わせ、最適設計を自動探索します。
    
    
    """
    Example 5: Integration with scipy Optimization
    scipy最適化とシミュレーターの統合
    """
    from scipy.optimize import minimize, differential_evolution
    import numpy as np
    
    class SimulatorOptimizer:
        """シミュレーターベース最適化"""
    
        def __init__(self, dwsim_interface):
            self.dwsim = dwsim_interface
            self.eval_count = 0
    
        def objective_function(self, x):
            """
            目的関数：利益最大化
    
            Args:
                x: 決定変数 [reflux_ratio, feed_stage_ratio]
    
            Returns:
                -profit (最小化問題)
            """
            reflux_ratio, feed_stage_ratio = x
            self.eval_count += 1
    
            try:
                # DWSIMパラメータ設定
                column = self.dwsim.flowsheet.GetFlowsheetObject("Column-01")
                column.RefluxRatio = reflux_ratio
    
                # Feed stage (1 = top, 20 = bottom)
                feed_stage = int(20 * feed_stage_ratio)
                column.FeedStage = max(1, min(feed_stage, 19))
    
                # シミュレーション実行
                self.dwsim.calculate()
    
                # 結果取得
                dist_flow = self.dwsim.get_stream_property("Distillate", "MassFlow")
                dist_comp = self.dwsim.get_stream_property("Distillate", "Composition")
                reboiler_duty = column.ReboilerDuty / 1000  # kW
    
                # 経済性計算
                revenue = (dist_flow / 1000) * 1000 * dist_comp[0]  # Benzene purity
                energy_cost = reboiler_duty * 0.05
    
                profit = revenue - energy_cost
    
                # ペナルティ（純度制約）
                if dist_comp[0] < 0.95:
                    penalty = 1000 * (0.95 - dist_comp[0])
                    profit -= penalty
    
                print(f"Eval {self.eval_count}: x={x}, profit=${profit:.2f}/h")
    
                return -profit  # 最大化→最小化
    
            except Exception as e:
                print(f"  ⚠ Simulation failed: {e}")
                return 1e6  # ペナルティ
    
        def optimize_local(self, x0):
            """局所最適化"""
            bounds = [
                (1.5, 5.0),   # reflux_ratio
                (0.3, 0.7)    # feed_stage_ratio
            ]
    
            result = minimize(
                self.objective_function,
                x0,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 50, 'disp': True}
            )
    
            return result
    
        def optimize_global(self):
            """大域最適化"""
            bounds = [
                (1.5, 5.0),   # reflux_ratio
                (0.3, 0.7)    # feed_stage_ratio
            ]
    
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=20,
                popsize=10,
                disp=True,
                workers=1  # シミュレーターは並列化不可
            )
    
            return result
    
    # 使用例
    optimizer = SimulatorOptimizer(dwsim)
    
    # 初期推定値
    x0 = np.array([2.5, 0.5])
    
    # 局所最適化
    print("=== Local Optimization ===")
    result_local = optimizer.optimize_local(x0)
    
    print(f"\nOptimal reflux ratio: {result_local.x[0]:.4f}")
    print(f"Optimal feed stage: {int(20 * result_local.x[1])}")
    print(f"Maximum profit: ${-result_local.fun:.2f}/h")
    
    # （大域最適化は計算時間がかかるため、必要に応じて実行）
    # result_global = optimizer.optimize_global()
    

* * *

## 5.6 機械学習によるプロセス予測

### シミュレーション代替モデル（Surrogate Model）

時間のかかるシミュレーションの代わりに、機械学習モデルで高速予測します。
    
    
    """
    Example 6: Machine Learning for Process Prediction
    機械学習によるプロセス予測（サロゲートモデル）
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error
    import matplotlib.pyplot as plt
    
    class SurrogateModel:
        """シミュレーション代替モデル"""
    
        def __init__(self):
            self.model = None
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
    
        def train(self, df, features, target):
            """
            モデル訓練
    
            Args:
                df: シミュレーション結果データ
                features: 入力特徴量のカラム名リスト
                target: 出力目標のカラム名
            """
            X = df[features].values
            y = df[target].values.reshape(-1, 1)
    
            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
    
            # 標準化
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_test_scaled = self.scaler_X.transform(X_test)
            y_train_scaled = self.scaler_y.fit_transform(y_train)
    
            # モデル訓練（Random Forest）
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train_scaled.ravel())
    
            # 性能評価
            y_pred_scaled = self.model.predict(X_test_scaled)
            y_pred = self.scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            )
    
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
    
            print(f"\n=== Model Performance ===")
            print(f"R² score: {r2:.4f}")
            print(f"MAE: {mae:.4f}")
    
            # Parity plot
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    'r--', lw=2, label='Perfect prediction')
            plt.xlabel(f'Actual {target}')
            plt.ylabel(f'Predicted {target}')
            plt.title(f'Parity Plot (R²={r2:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
            return r2, mae
    
        def predict(self, X_new):
            """新しい条件で予測"""
            X_scaled = self.scaler_X.transform(X_new)
            y_pred_scaled = self.model.predict(X_scaled)
            y_pred = self.scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            )
            return y_pred.flatten()
    
        def feature_importance(self, feature_names):
            """特徴量の重要度"""
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
    
            plt.figure(figsize=(8, 5))
            plt.bar(range(len(importances)),
                   importances[indices],
                   color='steelblue')
            plt.xticks(range(len(importances)),
                      [feature_names[i] for i in indices],
                      rotation=45, ha='right')
            plt.ylabel('Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
    
    # 使用例
    # シミュレーション結果から訓練データ作成
    df_train = pd.DataFrame({
        'reflux_ratio': np.random.uniform(1.5, 5.0, 200),
        'feed_stage': np.random.randint(5, 16, 200),
        'feed_temp': np.random.uniform(340, 360, 200),
        'benzene_purity': np.random.uniform(0.85, 0.99, 200),
        'reboiler_duty': np.random.uniform(800, 2000, 200)
    })
    
    # サロゲートモデル訓練
    surrogate = SurrogateModel()
    
    features = ['reflux_ratio', 'feed_stage', 'feed_temp']
    target = 'benzene_purity'
    
    r2, mae = surrogate.train(df_train, features, target)
    
    # 特徴量重要度
    surrogate.feature_importance(features)
    
    # 新しい条件で予測（シミュレーションの1/1000の時間）
    X_new = np.array([
        [3.0, 10, 350],
        [2.5, 12, 345],
        [4.0, 8, 355]
    ])
    
    predictions = surrogate.predict(X_new)
    print("\n=== Predictions ===")
    for i, pred in enumerate(predictions):
        print(f"Condition {i+1}: Predicted purity = {pred:.4f}")
    

* * *

## 5.7 自動レポート生成

### 結果を自動文書化

シミュレーション結果を自動的にPDFレポートやHTMLダッシュボードに変換します。
    
    
    """
    Example 7: Automated Report Generation
    自動レポート生成（HTML/PDF）
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from jinja2 import Template
    import base64
    from io import BytesIO
    
    class ReportGenerator:
        """シミュレーション結果の自動レポート生成"""
    
        def __init__(self, title="Process Simulation Report"):
            self.title = title
            self.sections = []
    
        def add_table(self, df, caption):
            """データテーブルを追加"""
            html_table = df.to_html(classes='table', index=False)
            self.sections.append({
                'type': 'table',
                'caption': caption,
                'content': html_table
            })
    
        def add_plot(self, fig, caption):
            """プロットを追加"""
            # 図をBase64エンコード
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
            self.sections.append({
                'type': 'plot',
                'caption': caption,
                'content': img_base64
            })
    
        def add_text(self, text):
            """テキストセクションを追加"""
            self.sections.append({
                'type': 'text',
                'content': text
            })
    
        def generate_html(self, output_file="report.html"):
            """HTMLレポート生成"""
    
            template_str = """
    
    
    
    
    
        
    
    # {{ title }}
    
    
        
    
    Generated: {{ timestamp }}
    
    
    
        {% for section in sections %}
            {% if section.type == 'table' %}
                
    
    ## {{ section.caption }}
    
    
                {{ section.content|safe }}
    
            {% elif section.type == 'plot' %}
                
    
    
                    
    
    ## {{ section.caption }}
    
    
                    ![](data:image/png;base64,{{ section.content }})
                
    
    
    
            {% elif section.type == 'text' %}
                
    
    {{ section.content }}
    
    
    
            {% endif %}
        {% endfor %}
    
    
    
            """
    
            template = Template(template_str)
    
            from datetime import datetime
            html_content = template.render(
                title=self.title,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                sections=self.sections
            )
    
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
    
            print(f"Report generated: {output_file}")
    
    # 使用例
    report = ReportGenerator("Distillation Column Optimization Report")
    
    # テキスト追加
    report.add_text(
        "This report summarizes the results of the distillation column "
        "optimization study. The objective was to maximize benzene purity "
        "while minimizing energy consumption."
    )
    
    # テーブル追加
    df_summary = pd.DataFrame({
        'Parameter': ['Reflux Ratio', 'Feed Stage', 'Benzene Purity', 'Reboiler Duty'],
        'Optimal Value': [3.2, 10, '96.5%', '1250 kW']
    })
    report.add_table(df_summary, "Optimal Operating Conditions")
    
    # プロット追加
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(1.5, 5.0, 50)
    y = 95 + 3 * np.exp(-0.5 * (x - 3)**2)  # Example curve
    ax.plot(x, y, 'o-', linewidth=2)
    ax.set_xlabel('Reflux Ratio')
    ax.set_ylabel('Benzene Purity [%]')
    ax.set_title('Purity vs Reflux Ratio')
    ax.grid(True, alpha=0.3)
    
    report.add_plot(fig, "Performance Curve")
    
    # HTMLレポート生成
    report.generate_html("optimization_report.html")
    print("Open optimization_report.html in your browser")
    

* * *

## 5.8 完全ワークフロー：シミュレーション→解析→最適化

### エンドツーエンドの自動化パイプライン

全てのステップを統合した自動化ワークフローを構築します。
    
    
    """
    Example 8: Complete Workflow - Simulation to Optimization
    完全ワークフロー（シミュレーション→解析→最適化→レポート）
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    class AutomatedWorkflow:
        """プロセス設計の完全自動化ワークフロー"""
    
        def __init__(self, dwsim_interface):
            self.dwsim = dwsim_interface
            self.results = []
            self.optimal_conditions = None
    
        def step1_initial_design(self):
            """Step 1: 初期設計（フローシート構築）"""
            print("\n=== Step 1: Initial Design ===")
    
            # フローシート構築（Example 2のコード）
            builder = DistillationColumnBuilder(self.dwsim)
            self.flowsheet = builder.build_simple_distillation()
    
            print("✓ Flowsheet created")
    
        def step2_parameter_sweep(self):
            """Step 2: パラメータスイープ"""
            print("\n=== Step 2: Parameter Sweep ===")
    
            sweep = ParameterSweep(self.dwsim)
            reflux_range = np.linspace(1.5, 5.0, 20)
    
            self.df_sweep = sweep.sweep_reflux_ratio(reflux_range)
    
            print(f"✓ Completed {len(self.df_sweep)} simulations")
    
        def step3_data_analysis(self):
            """Step 3: データ解析"""
            print("\n=== Step 3: Data Analysis ===")
    
            analyzer = SimulationDataAnalyzer()
            analyzer.df = self.df_sweep
    
            # 経済性計算
            self.df_economics = analyzer.calculate_economics(
                product_price=1000,
                energy_cost=0.05
            )
    
            # 最適条件探索
            self.optimal_conditions = analyzer.find_optimal_condition(
                objective='profit',
                constraint_col='benzene_purity',
                constraint_val=0.95
            )
    
            print("✓ Analysis completed")
    
        def step4_surrogate_model(self):
            """Step 4: サロゲートモデル訓練"""
            print("\n=== Step 4: Surrogate Model Training ===")
    
            surrogate = SurrogateModel()
            features = ['reflux_ratio']
            target = 'profit'
    
            r2, mae = surrogate.train(self.df_economics, features, target)
            self.surrogate = surrogate
    
            print(f"✓ Model trained (R²={r2:.3f})")
    
        def step5_optimization(self):
            """Step 5: 最適化"""
            print("\n=== Step 5: Optimization ===")
    
            # サロゲートモデルベース高速最適化
            def fast_objective(x):
                return -self.surrogate.predict(x.reshape(1, -1))[0]
    
            from scipy.optimize import minimize
            result = minimize(
                fast_objective,
                x0=[3.0],
                bounds=[(1.5, 5.0)],
                method='L-BFGS-B'
            )
    
            self.optimized_reflux = result.x[0]
            print(f"✓ Optimal reflux ratio: {self.optimized_reflux:.4f}")
    
        def step6_verification(self):
            """Step 6: 最適解の検証"""
            print("\n=== Step 6: Verification ===")
    
            # 最適条件で詳細シミュレーション実行
            column = self.dwsim.flowsheet.GetFlowsheetObject("Column-01")
            column.RefluxRatio = self.optimized_reflux
    
            self.dwsim.calculate()
    
            # 結果取得
            dist_comp = self.dwsim.get_stream_property("Distillate", "Composition")
            reboiler_duty = column.ReboilerDuty / 1000
    
            print(f"✓ Verified benzene purity: {dist_comp[0]*100:.2f}%")
            print(f"✓ Reboiler duty: {reboiler_duty:.1f} kW")
    
        def step7_generate_report(self):
            """Step 7: レポート生成"""
            print("\n=== Step 7: Report Generation ===")
    
            report = ReportGenerator("Automated Process Design Report")
    
            # 結果サマリー
            report.add_text(
                f"Automated workflow completed on {datetime.now().strftime('%Y-%m-%d')}. "
                f"Optimal reflux ratio determined: {self.optimized_reflux:.4f}"
            )
    
            # 最適条件テーブル
            report.add_table(
                pd.DataFrame([self.optimal_conditions]),
                "Optimal Operating Conditions"
            )
    
            # プロット
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(self.df_economics['reflux_ratio'],
                   self.df_economics['profit'], 'o-')
            ax.axvline(x=self.optimized_reflux, color='r',
                      linestyle='--', label='Optimized')
            ax.set_xlabel('Reflux Ratio')
            ax.set_ylabel('Profit [$/h]')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
            report.add_plot(fig, "Profit vs Reflux Ratio")
    
            report.generate_html("automated_workflow_report.html")
    
            print("✓ Report generated")
    
        def run_full_workflow(self):
            """完全ワークフローの実行"""
            print("="*50)
            print("AUTOMATED PROCESS DESIGN WORKFLOW")
            print("="*50)
    
            start_time = datetime.now()
    
            try:
                self.step1_initial_design()
                self.step2_parameter_sweep()
                self.step3_data_analysis()
                self.step4_surrogate_model()
                self.step5_optimization()
                self.step6_verification()
                self.step7_generate_report()
    
                elapsed = (datetime.now() - start_time).total_seconds()
    
                print("\n" + "="*50)
                print("✓ WORKFLOW COMPLETED SUCCESSFULLY")
                print(f"Total time: {elapsed:.1f} seconds")
                print("="*50)
    
            except Exception as e:
                print(f"\n⚠ Workflow failed: {e}")
                raise
    
    # 使用例（実際にはDWSIMインターフェースが必要）
    # dwsim = DWSIMInterface()
    # workflow = AutomatedWorkflow(dwsim)
    # workflow.run_full_workflow()
    

* * *

## 学習目標の確認

この章を完了すると、以下ができるようになります：

### 基本理解

  * ✅ DWSIMのPythonインターフェースの仕組みを理解する
  * ✅ シミュレーターとデータ解析ツールの統合方法を説明できる
  * ✅ サロゲートモデルの役割と利点を理解する

### 実践スキル

  * ✅ Pythonスクリプトでフローシートを自動構築できる
  * ✅ パラメータスイープを自動化し、結果をpandasで解析できる
  * ✅ scipyの最適化アルゴリズムとシミュレーターを統合できる
  * ✅ 機械学習モデルでプロセス性能を予測できる
  * ✅ 結果を自動的にHTMLレポートにまとめられる

### 応用力

  * ✅ エンドツーエンドの自動化ワークフローを構築できる
  * ✅ 大規模な設計空間探索を効率的に実行できる
  * ✅ 商用シミュレーターとオープンソースツールを組み合わせた実践的システムを開発できる

* * *

## まとめ

この章では、Pythonとプロセスシミュレーターの連携による高度な自動化を習得しました：

  * **DWSIM連携** ：Pythonから完全なフローシート操作
  * **パラメータスイープ** ：大量のケース計算自動化
  * **データ解析** ：pandasによる効率的な結果解析と可視化
  * **最適化統合** ：scipyアルゴリズムとシミュレーターの連携
  * **機械学習** ：サロゲートモデルによる高速予測
  * **レポート自動生成** ：HTML/PDF形式での結果文書化
  * **完全ワークフロー** ：simulation → analysis → optimization の統合

**実務への応用** ：

  * プロセス設計の大幅な効率化（手作業の1/10の時間）
  * 最適運転条件の自動探索
  * AIとシミュレーションの融合による予測精度向上
  * 再現可能で文書化された設計プロセス

これでプロセスシミュレーション入門シリーズは完結です。実際のプロジェクトに応用し、さらなる発展を目指してください！

* * *

### 免責事項

  * 本コンテンツは教育目的で作成されており、実プラント設計には専門家の監修が必要です
  * DWSIMのPythonインターフェースはバージョンにより仕様が異なる場合があります
  * 機械学習モデルは訓練データの範囲内でのみ信頼性が保証されます
  * 最適化結果は必ず詳細シミュレーションで検証してください
  * 商用利用の際は各ソフトウェアのライセンスを確認してください
