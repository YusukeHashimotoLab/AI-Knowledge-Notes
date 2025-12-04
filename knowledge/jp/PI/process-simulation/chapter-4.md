---
title: 第4章：プロセスフローシートシミュレーション
chapter_title: 第4章：プロセスフローシートシミュレーション
subtitle: 複数ユニットの統合、リサイクルストリームの収束、最適化
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ フローシートのトポロジーをグラフ理論で表現できる
  * ✅ 逐次計算法とリサイクルストリームの収束計算を実装できる
  * ✅ 複数ユニットを統合した完全なプロセスシミュレーションを構築できる
  * ✅ 感度分析とパラメータ最適化を実行できる
  * ✅ ヒートインテグレーションの基礎を理解する

* * *

## 4.1 フローシートのトポロジー表現

### グラフ理論によるフローシート表現

化学プロセスのフローシートは有向グラフ（Directed Graph）として表現できます。各ユニットはノード、物質・エネルギーストリームはエッジに対応します。
    
    
    """
    Example 1: Flowsheet Topology Representation
    フローシートトポロジーの表現（有向グラフ）
    """
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from typing import Dict, List, Tuple
    
    class FlowsheetTopology:
        """フローシートのトポロジーを管理するクラス"""
    
        def __init__(self):
            self.graph = nx.DiGraph()
            self.streams = {}  # ストリーム情報
    
        def add_unit(self, unit_name: str, unit_type: str):
            """ユニットを追加"""
            self.graph.add_node(unit_name, type=unit_type)
    
        def add_stream(self, from_unit: str, to_unit: str,
                       stream_name: str, stream_data: Dict = None):
            """ストリームを追加"""
            self.graph.add_edge(from_unit, to_unit, name=stream_name)
            if stream_data:
                self.streams[stream_name] = stream_data
    
        def get_calculation_order(self) -> List[str]:
            """トポロジカルソートで計算順序を決定"""
            try:
                # リサイクルがない場合は単純なトポロジカルソート
                return list(nx.topological_sort(self.graph))
            except nx.NetworkXError:
                # サイクル（リサイクル）がある場合
                return self._handle_recycle()
    
        def _handle_recycle(self) -> List[str]:
            """リサイクルストリームを含む計算順序の決定"""
            # 最も影響の小さいエッジを切断してトポロジカルソート
            cycles = list(nx.simple_cycles(self.graph))
            print(f"Recycle loops detected: {cycles}")
    
            # リサイクルストリームを特定（簡易版：最初のサイクルのみ）
            if cycles:
                recycle_edge = (cycles[0][-1], cycles[0][0])
                temp_graph = self.graph.copy()
                temp_graph.remove_edge(*recycle_edge)
                order = list(nx.topological_sort(temp_graph))
                print(f"Calculation order (with recycle): {order}")
                return order
            return []
    
        def visualize(self):
            """フローシートを可視化"""
            plt.figure(figsize=(10, 6))
            pos = nx.spring_layout(self.graph, seed=42)
    
            # ノードの色分け（ユニットタイプ別）
            node_colors = []
            for node in self.graph.nodes():
                unit_type = self.graph.nodes[node].get('type', 'unknown')
                colors = {'reactor': '#ff6b6b', 'separator': '#4ecdc4',
                         'heater': '#ffe66d', 'cooler': '#95e1d3'}
                node_colors.append(colors.get(unit_type, '#gray'))
    
            nx.draw(self.graph, pos, with_labels=True,
                   node_color=node_colors, node_size=2000,
                   font_size=10, font_weight='bold',
                   arrows=True, arrowsize=20, edge_color='#666')
    
            # エッジラベル（ストリーム名）
            edge_labels = nx.get_edge_attributes(self.graph, 'name')
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels)
    
            plt.title("Process Flowsheet Topology")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    # 使用例
    flowsheet = FlowsheetTopology()
    
    # ユニットを追加
    flowsheet.add_unit("FEED", "feed")
    flowsheet.add_unit("R-101", "reactor")
    flowsheet.add_unit("T-101", "separator")
    flowsheet.add_unit("PRODUCT", "product")
    
    # ストリームを追加
    flowsheet.add_stream("FEED", "R-101", "S-01")
    flowsheet.add_stream("R-101", "T-101", "S-02")
    flowsheet.add_stream("T-101", "PRODUCT", "S-03")
    flowsheet.add_stream("T-101", "R-101", "S-04")  # Recycle
    
    # 計算順序を取得
    calc_order = flowsheet.get_calculation_order()
    print(f"Calculation order: {calc_order}")
    
    # 可視化
    flowsheet.visualize()
    

### 計算順序の決定

トポロジカルソート（Topological Sort）により、上流から下流への計算順序が自動的に決定されます。リサイクルストリームがある場合は反復計算が必要です。

* * *

## 4.2 リサイクルストリームの収束計算

### 逐次代入法（Successive Substitution）

リサイクルストリームを含むフローシートでは、仮定値から開始して収束するまで反復計算を行います。
    
    
    """
    Example 2: Recycle Stream Convergence
    リサイクルストリームの収束計算（逐次代入法）
    """
    import numpy as np
    from scipy.optimize import fsolve
    
    class RecycleConvergence:
        """リサイクルストリームの収束計算"""
    
        def __init__(self, max_iter=100, tol=1e-6):
            self.max_iter = max_iter
            self.tol = tol
            self.history = []
    
        def successive_substitution(self, flowsheet_func,
                                    initial_guess, damping=0.5):
            """
            逐次代入法による収束計算
    
            Args:
                flowsheet_func: フローシート計算関数（入力→出力）
                initial_guess: リサイクルストリームの初期推定値
                damping: 減衰係数（0-1、収束安定性向上）
    
            Returns:
                converged_value: 収束値
                iterations: 反復回数
            """
            x_old = np.array(initial_guess)
    
            for iteration in range(self.max_iter):
                # フローシート計算
                x_new = flowsheet_func(x_old)
    
                # 減衰（収束性向上）
                x_damped = damping * x_new + (1 - damping) * x_old
    
                # 収束判定
                error = np.linalg.norm(x_damped - x_old)
                self.history.append({'iter': iteration, 'error': error,
                                   'value': x_damped.copy()})
    
                print(f"Iter {iteration}: error = {error:.2e}")
    
                if error < self.tol:
                    print(f"Converged in {iteration} iterations")
                    return x_damped, iteration
    
                x_old = x_damped
    
            print("WARNING: Did not converge")
            return x_old, self.max_iter
    
        def wegstein_acceleration(self, flowsheet_func, initial_guess):
            """
            Wegstein加速法（収束性向上）
    
            逐次代入法より高速に収束する場合が多い
            """
            x0 = np.array(initial_guess)
            x1 = flowsheet_func(x0)
    
            for iteration in range(self.max_iter):
                x2 = flowsheet_func(x1)
    
                # Wegstein加速パラメータ
                q = (x2 - x1) / (x1 - x0 + 1e-10)
                q_safe = np.clip(q, -5, 0)  # 安定化
    
                # 加速
                x_new = (q_safe * x1 - x0) / (q_safe - 1)
    
                # 収束判定
                error = np.linalg.norm(x_new - x1)
                self.history.append({'iter': iteration, 'error': error})
    
                if error < self.tol:
                    print(f"Wegstein converged in {iteration} iterations")
                    return x_new, iteration
    
                x0, x1 = x1, x_new
    
            return x1, self.max_iter
    
    # 使用例：簡単なリサイクルループ
    def simple_recycle_flowsheet(recycle_flow):
        """
        簡単なリサイクルフローシート
        recycle_flow: リサイクル流量 [kmol/h]
        returns: 新しいリサイクル流量
        """
        feed_flow = 100.0  # kmol/h
        conversion = 0.8
    
        total_feed = feed_flow + recycle_flow
        product = total_feed * conversion
        recycle_new = total_feed * (1 - conversion)
    
        return np.array([recycle_new])
    
    # 収束計算
    solver = RecycleConvergence(tol=1e-6)
    initial = np.array([10.0])  # 初期推定値
    
    # 逐次代入法
    recycle_conv, iters = solver.successive_substitution(
        simple_recycle_flowsheet, initial, damping=0.7
    )
    print(f"Converged recycle flow: {recycle_conv[0]:.2f} kmol/h")
    

* * *

## 4.3 完全フローシートシミュレーション

### 蒸留塔＋反応器＋分離器の統合

実用的なフローシートでは、複数のユニットが相互作用します。ここでは蒸留塔で精製した原料を反応器で反応させ、生成物を分離するプロセスを構築します。
    
    
    """
    Example 3: Complete Flowsheet Simulation
    完全フローシートシミュレーション（蒸留+反応+分離+リサイクル）
    """
    import numpy as np
    from dataclasses import dataclass
    from typing import Dict
    
    @dataclass
    class Stream:
        """ストリームのデータ構造"""
        flow: float  # kmol/h
        composition: Dict[str, float]  # mol fraction
        temperature: float  # K
        pressure: float  # bar
    
        def __repr__(self):
            return f"Stream(F={self.flow:.1f}, T={self.temperature:.1f}K)"
    
    class CompleteFlowsheet:
        """完全なプロセスフローシート"""
    
        def __init__(self):
            self.streams = {}
            self.convergence_history = []
    
        def reactor(self, feed: Stream, conversion=0.75) -> Stream:
            """反応器: A → B"""
            product = Stream(
                flow=feed.flow,
                composition={
                    'A': feed.composition['A'] * (1 - conversion),
                    'B': feed.composition['A'] * conversion + feed.composition.get('B', 0)
                },
                temperature=feed.temperature + 50,  # 発熱反応
                pressure=feed.pressure
            )
            return product
    
        def distillation(self, feed: Stream, recovery_A=0.95) -> Tuple[Stream, Stream]:
            """蒸留塔: AとBを分離"""
            # 塔頂（A rich）
            distillate_flow = feed.flow * feed.composition['A'] * recovery_A
            distillate = Stream(
                flow=distillate_flow,
                composition={'A': 0.98, 'B': 0.02},
                temperature=350,
                pressure=1.5
            )
    
            # 塔底（B rich）
            bottoms_flow = feed.flow - distillate_flow
            bottoms = Stream(
                flow=bottoms_flow,
                composition={'A': 0.05, 'B': 0.95},
                temperature=400,
                pressure=1.5
            )
    
            return distillate, bottoms
    
        def mixer(self, streams: List[Stream]) -> Stream:
            """ミキサー：複数ストリームの混合"""
            total_flow = sum(s.flow for s in streams)
    
            # 物質収支
            comp_A = sum(s.flow * s.composition['A'] for s in streams) / total_flow
            comp_B = sum(s.flow * s.composition['B'] for s in streams) / total_flow
    
            mixed = Stream(
                flow=total_flow,
                composition={'A': comp_A, 'B': comp_B},
                temperature=sum(s.flow * s.temperature for s in streams) / total_flow,
                pressure=min(s.pressure for s in streams)
            )
            return mixed
    
        def simulate(self, feed: Stream, recycle_ratio=0.5,
                    max_iter=50, tol=1e-4):
            """
            完全フローシートシミュレーション
    
            フロー: Feed → Mixer → Reactor → Distillation
                                 ↑              ↓
                                 ← Recycle ←─── Distillate
            """
            # 初期リサイクル推定
            recycle = Stream(
                flow=feed.flow * recycle_ratio,
                composition={'A': 0.98, 'B': 0.02},
                temperature=350,
                pressure=1.5
            )
    
            for iteration in range(max_iter):
                # 1. ミキサー
                mixed = self.mixer([feed, recycle])
    
                # 2. 反応器
                reactor_out = self.reactor(mixed, conversion=0.75)
    
                # 3. 蒸留塔
                distillate, bottoms = self.distillation(reactor_out, recovery_A=0.95)
    
                # 4. リサイクル更新
                recycle_new = distillate
    
                # 収束判定
                error = abs(recycle_new.flow - recycle.flow)
                self.convergence_history.append(error)
    
                print(f"Iter {iteration}: Recycle flow error = {error:.2e} kmol/h")
    
                if error < tol:
                    print(f"✓ Converged in {iteration} iterations")
    
                    # 結果保存
                    self.streams = {
                        'feed': feed,
                        'mixed': mixed,
                        'reactor_out': reactor_out,
                        'distillate': distillate,
                        'bottoms': bottoms,
                        'recycle': recycle_new
                    }
    
                    return self.streams
    
                recycle = recycle_new
    
            print("⚠ Did not converge")
            return self.streams
    
        def print_results(self):
            """結果の表示"""
            print("\n=== Flowsheet Simulation Results ===")
            for name, stream in self.streams.items():
                print(f"{name:15s}: {stream}")
                print(f"  Composition: A={stream.composition['A']:.3f}, "
                      f"B={stream.composition['B']:.3f}")
    
    # 使用例
    flowsheet = CompleteFlowsheet()
    
    # フィード条件
    feed = Stream(
        flow=100.0,
        composition={'A': 1.0, 'B': 0.0},
        temperature=300,
        pressure=2.0
    )
    
    # シミュレーション実行
    results = flowsheet.simulate(feed, recycle_ratio=0.3)
    flowsheet.print_results()
    
    # 収束履歴をプロット
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.semilogy(flowsheet.convergence_history, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Recycle Flow Error [kmol/h]')
    plt.title('Convergence History')
    plt.grid(True, alpha=0.3)
    plt.show()
    

* * *

## 4.4 感度分析（Sensitivity Analysis）

### パラメータ変化の影響評価

操作条件の変化が製品収率やエネルギー消費に与える影響を定量化します。
    
    
    """
    Example 4: Sensitivity Analysis
    感度分析（パラメータ変化の影響評価）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    class SensitivityAnalyzer:
        """感度分析ツール"""
    
        def __init__(self, flowsheet):
            self.flowsheet = flowsheet
    
        def analyze_parameter(self, param_name, param_range,
                             output_metrics):
            """
            単一パラメータの感度分析
    
            Args:
                param_name: パラメータ名
                param_range: パラメータの範囲（array）
                output_metrics: 出力指標のリスト
            """
            results = {metric: [] for metric in output_metrics}
    
            for param_value in param_range:
                # パラメータ設定してシミュレーション
                # （実装は省略：flowsheet内のパラメータを変更）
    
                # 例：反応器転化率の影響
                conversion = param_value
                # ここでflowsheet.simulateを実行
    
                # 出力指標を計算
                yield_B = 0.75 * conversion  # 簡略化
                energy = 1000 + 500 * conversion  # kW
    
                results['yield'].append(yield_B)
                results['energy'].append(energy)
    
            return results
    
        def plot_sensitivity(self, param_name, param_range, results):
            """感度分析結果のプロット"""
            fig, axes = plt.subplots(1, len(results), figsize=(12, 4))
    
            if len(results) == 1:
                axes = [axes]
    
            for ax, (metric, values) in zip(axes, results.items()):
                ax.plot(param_range, values, 'o-', linewidth=2)
                ax.set_xlabel(param_name)
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{metric} vs {param_name}')
    
            plt.tight_layout()
            plt.show()
    
        def tornado_plot(self, params_impact):
            """
            トルネード図（複数パラメータの影響度比較）
    
            Args:
                params_impact: {param_name: (min_value, max_value), ...}
            """
            params = list(params_impact.keys())
            impacts = [abs(max_val - min_val)
                      for min_val, max_val in params_impact.values()]
    
            # 影響度順にソート
            sorted_idx = np.argsort(impacts)
            params_sorted = [params[i] for i in sorted_idx]
            impacts_sorted = [impacts[i] for i in sorted_idx]
    
            plt.figure(figsize=(8, 6))
            plt.barh(params_sorted, impacts_sorted, color='steelblue')
            plt.xlabel('Impact on Product Yield')
            plt.title('Tornado Diagram - Parameter Sensitivity')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.show()
    
    # 使用例
    analyzer = SensitivityAnalyzer(flowsheet)
    
    # 反応器転化率の感度分析
    conversion_range = np.linspace(0.5, 0.95, 10)
    results = {
        'yield': 100 * conversion_range * 0.75,
        'energy': 1000 + 500 * conversion_range
    }
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(conversion_range, results['yield'], 'o-', linewidth=2)
    ax1.set_xlabel('Reactor Conversion')
    ax1.set_ylabel('Product Yield [kmol/h]')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(conversion_range, results['energy'], 'o-',
            linewidth=2, color='orangered')
    ax2.set_xlabel('Reactor Conversion')
    ax2.set_ylabel('Energy Consumption [kW]')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 4.5 最適化（Optimization）

### 経済性と環境負荷のバランス

収益最大化、コスト最小化、環境負荷低減などの目的関数を最適化します。
    
    
    """
    Example 5: Flowsheet Optimization
    フローシート最適化（収益最大化）
    """
    from scipy.optimize import minimize, differential_evolution
    import numpy as np
    
    class FlowsheetOptimizer:
        """フローシート最適化"""
    
        def __init__(self, flowsheet):
            self.flowsheet = flowsheet
    
        def objective_profit(self, x):
            """
            目的関数：利益最大化 = 収益 - コスト
    
            Args:
                x: 決定変数 [conversion, recycle_ratio]
    
            Returns:
                -profit (最小化問題に変換)
            """
            conversion, recycle_ratio = x
    
            # フローシートシミュレーション（簡略化）
            feed_flow = 100.0
            product_flow = feed_flow * conversion * 0.9
    
            # 収益
            product_price = 1000  # $/kmol
            revenue = product_flow * product_price
    
            # コスト
            raw_material_cost = feed_flow * 500  # $/kmol
            energy_cost = (1000 + 500 * conversion) * 0.1  # $/kW
            recycle_cost = recycle_ratio * 1000
    
            total_cost = raw_material_cost + energy_cost + recycle_cost
    
            profit = revenue - total_cost
    
            return -profit  # 最大化 → 最小化
    
        def constraints(self, x):
            """制約条件"""
            conversion, recycle_ratio = x
    
            # 制約：転化率は0.5-0.95、リサイクル比は0-0.8
            constraints = [
                conversion - 0.5,      # conversion >= 0.5
                0.95 - conversion,     # conversion <= 0.95
                recycle_ratio,         # recycle_ratio >= 0
                0.8 - recycle_ratio    # recycle_ratio <= 0.8
            ]
    
            return constraints
    
        def optimize_local(self, x0):
            """局所最適化（勾配法）"""
            bounds = [(0.5, 0.95), (0, 0.8)]
    
            result = minimize(
                self.objective_profit,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'disp': True}
            )
    
            return result
    
        def optimize_global(self):
            """大域最適化（Differential Evolution）"""
            bounds = [(0.5, 0.95), (0, 0.8)]
    
            result = differential_evolution(
                self.objective_profit,
                bounds,
                maxiter=100,
                popsize=15,
                disp=True
            )
    
            return result
    
    # 使用例
    optimizer = FlowsheetOptimizer(flowsheet)
    
    # 初期推定値
    x0 = np.array([0.75, 0.3])
    
    # 局所最適化
    print("=== Local Optimization ===")
    result_local = optimizer.optimize_local(x0)
    print(f"Optimal conversion: {result_local.x[0]:.4f}")
    print(f"Optimal recycle ratio: {result_local.x[1]:.4f}")
    print(f"Maximum profit: ${-result_local.fun:.2f}/h")
    
    # 大域最適化
    print("\n=== Global Optimization ===")
    result_global = optimizer.optimize_global()
    print(f"Optimal conversion: {result_global.x[0]:.4f}")
    print(f"Optimal recycle ratio: {result_global.x[1]:.4f}")
    print(f"Maximum profit: ${-result_global.fun:.2f}/h")
    

* * *

## 4.6 ヒートインテグレーション基礎

### Pinch解析の導入

プロセス内の熱交換を最適化し、エネルギー消費を削減します。
    
    
    """
    Example 6: Heat Integration Basics
    ヒートインテグレーション基礎（ピンチ解析）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class HeatStream:
        """熱ストリーム"""
        name: str
        type: str  # 'hot' or 'cold'
        T_supply: float  # K
        T_target: float  # K
        heat_capacity_flow: float  # kW/K
    
        @property
        def heat_load(self):
            """熱負荷 [kW]"""
            return abs(self.heat_capacity_flow * (self.T_target - self.T_supply))
    
    class PinchAnalyzer:
        """Pinch解析ツール"""
    
        def __init__(self, min_approach_temp=10):
            """
            Args:
                min_approach_temp: 最小接近温度差 [K]
            """
            self.dT_min = min_approach_temp
            self.hot_streams = []
            self.cold_streams = []
    
        def add_stream(self, stream: HeatStream):
            """ストリームを追加"""
            if stream.type == 'hot':
                self.hot_streams.append(stream)
            else:
                self.cold_streams.append(stream)
    
        def calculate_composite_curves(self):
            """複合曲線を計算"""
            # 温度間隔の作成
            temps = []
            for stream in self.hot_streams + self.cold_streams:
                temps.extend([stream.T_supply, stream.T_target])
            temps = sorted(set(temps))
    
            # Hot composite curve
            hot_curve = []
            Q_hot = 0
            for T in sorted(temps, reverse=True):
                for stream in self.hot_streams:
                    if stream.T_target <= T <= stream.T_supply:
                        dT = T - max(T - 1, stream.T_target)
                        Q_hot += stream.heat_capacity_flow * abs(dT)
                hot_curve.append((T, Q_hot))
    
            # Cold composite curve (shifted by dT_min)
            cold_curve = []
            Q_cold = 0
            for T in sorted(temps):
                for stream in self.cold_streams:
                    if stream.T_supply <= T <= stream.T_target:
                        dT = min(T + 1, stream.T_target) - T
                        Q_cold += stream.heat_capacity_flow * abs(dT)
                cold_curve.append((T + self.dT_min, Q_cold))
    
            return hot_curve, cold_curve
    
        def find_pinch_point(self, hot_curve, cold_curve):
            """ピンチポイントを特定"""
            # 最小距離を探索（簡略化）
            min_distance = float('inf')
            pinch_temp = None
    
            for T_hot, Q_hot in hot_curve:
                for T_cold, Q_cold in cold_curve:
                    if abs(T_hot - T_cold) < min_distance:
                        min_distance = abs(T_hot - T_cold)
                        pinch_temp = T_hot
    
            return pinch_temp, min_distance
    
        def calculate_utilities(self):
            """必要なユーティリティ（加熱・冷却）を計算"""
            total_hot_load = sum(s.heat_load for s in self.hot_streams)
            total_cold_load = sum(s.heat_load for s in self.cold_streams)
    
            if total_hot_load > total_cold_load:
                heating_utility = 0
                cooling_utility = total_hot_load - total_cold_load
            else:
                heating_utility = total_cold_load - total_hot_load
                cooling_utility = 0
    
            return heating_utility, cooling_utility
    
        def plot_composite_curves(self):
            """複合曲線をプロット"""
            hot_curve, cold_curve = self.calculate_composite_curves()
    
            plt.figure(figsize=(10, 6))
    
            # Hot composite curve
            T_hot, Q_hot = zip(*hot_curve)
            plt.plot(Q_hot, T_hot, 'r-', linewidth=2, label='Hot Composite')
    
            # Cold composite curve
            T_cold, Q_cold = zip(*cold_curve)
            plt.plot(Q_cold, T_cold, 'b-', linewidth=2, label='Cold Composite')
    
            plt.xlabel('Heat Load [kW]')
            plt.ylabel('Temperature [K]')
            plt.title('Composite Curves (Pinch Analysis)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    # 使用例
    analyzer = PinchAnalyzer(min_approach_temp=10)
    
    # 熱ストリームを追加
    analyzer.add_stream(HeatStream('H1', 'hot', 450, 350, 20))
    analyzer.add_stream(HeatStream('H2', 'hot', 400, 320, 15))
    analyzer.add_stream(HeatStream('C1', 'cold', 300, 420, 18))
    analyzer.add_stream(HeatStream('C2', 'cold', 330, 390, 12))
    
    # ユーティリティ計算
    heating, cooling = analyzer.calculate_utilities()
    print(f"Heating utility required: {heating:.1f} kW")
    print(f"Cooling utility required: {cooling:.1f} kW")
    
    # 複合曲線をプロット
    analyzer.plot_composite_curves()
    

* * *

## 4.7 ケーススタディ：化学プラント全体のシミュレーション

### 統合プロセス設計

実際の化学プラント設計では、反応、分離、熱交換、リサイクルを全て統合します。
    
    
    """
    Example 7: Complete Chemical Plant Simulation
    化学プラント全体のシミュレーション（統合ケーススタディ）
    """
    import numpy as np
    from dataclasses import dataclass, field
    from typing import Dict, List
    
    @dataclass
    class PlantPerformance:
        """プラント性能指標"""
        production_rate: float  # kmol/h
        yield_overall: float  # %
        energy_consumption: float  # kW
        raw_material_usage: float  # kmol/h
        profit_per_hour: float  # $/h
    
        def display(self):
            print("\n=== Plant Performance ===")
            print(f"Production rate:    {self.production_rate:.2f} kmol/h")
            print(f"Overall yield:      {self.yield_overall:.2f} %")
            print(f"Energy consumption: {self.energy_consumption:.1f} kW")
            print(f"Raw material usage: {self.raw_material_usage:.2f} kmol/h")
            print(f"Profit:            ${self.profit_per_hour:.2f}/h")
    
    class ChemicalPlant:
        """化学プラント全体のシミュレーター"""
    
        def __init__(self, config: Dict):
            self.config = config
            self.streams = {}
            self.unit_operations = {}
    
        def simulate_full_plant(self, feed_rate=100.0,
                               optimization_mode=False):
            """
            プラント全体のシミュレーション
    
            Process Flow:
            Feed → Preheater → Reactor → Flash → Distillation → Product
                      ↑           ↓                    ↓
                  Heat Recovery   ←──── Recycle ───────┘
            """
            # 1. フィード予熱器
            feed_temp_in = 300  # K
            feed_temp_out = 400  # K
            preheat_duty = feed_rate * 2.5 * (feed_temp_out - feed_temp_in)
    
            # 2. 反応器
            conversion = self.config.get('conversion', 0.80)
            reaction_temp = 450  # K
            reaction_heat = feed_rate * conversion * 50  # kJ/kmol
    
            product_from_reactor = feed_rate * conversion
            unreacted = feed_rate * (1 - conversion)
    
            # 3. フラッシュ分離器
            vapor_frac = 0.3
            flash_vapor = (product_from_reactor + unreacted) * vapor_frac
            flash_liquid = (product_from_reactor + unreacted) * (1 - vapor_frac)
    
            # 4. 蒸留塔
            distillate_purity = 0.98
            distillate_recovery = 0.95
    
            distillate = flash_liquid * distillate_recovery * distillate_purity
            bottoms = flash_liquid - distillate
    
            # 5. リサイクル
            recycle_ratio = self.config.get('recycle_ratio', 0.5)
            recycle_flow = distillate * recycle_ratio
    
            # 6. エネルギー統合
            heat_recovery = 0.7 * reaction_heat  # 70%回収
            heating_utility = preheat_duty - heat_recovery
            cooling_utility = reaction_heat - heat_recovery
    
            total_energy = heating_utility + cooling_utility
    
            # 性能指標計算
            performance = PlantPerformance(
                production_rate=distillate * (1 - recycle_ratio),
                yield_overall=100 * distillate * (1 - recycle_ratio) / feed_rate,
                energy_consumption=total_energy,
                raw_material_usage=feed_rate,
                profit_per_hour=self._calculate_profit(
                    distillate * (1 - recycle_ratio),
                    feed_rate,
                    total_energy
                )
            )
    
            return performance
    
        def _calculate_profit(self, product, feed, energy):
            """利益計算"""
            revenue = product * 1000  # $/kmol
            feed_cost = feed * 500
            energy_cost = energy * 0.1  # $/kW
            return revenue - feed_cost - energy_cost
    
        def optimize_operation(self):
            """運転最適化"""
            from scipy.optimize import minimize
    
            def objective(x):
                conversion, recycle_ratio = x
                self.config['conversion'] = conversion
                self.config['recycle_ratio'] = recycle_ratio
    
                perf = self.simulate_full_plant()
                return -perf.profit_per_hour  # 最大化→最小化
    
            result = minimize(
                objective,
                x0=[0.75, 0.4],
                bounds=[(0.6, 0.95), (0.2, 0.7)],
                method='SLSQP'
            )
    
            return result
    
    # 使用例
    config = {
        'conversion': 0.80,
        'recycle_ratio': 0.5,
        'min_approach_temp': 10
    }
    
    plant = ChemicalPlant(config)
    
    # 通常運転
    print("=== Normal Operation ===")
    perf_normal = plant.simulate_full_plant(feed_rate=100.0)
    perf_normal.display()
    
    # 最適化運転
    print("\n=== Optimized Operation ===")
    result = plant.optimize_operation()
    print(f"Optimal conversion: {result.x[0]:.4f}")
    print(f"Optimal recycle ratio: {result.x[1]:.4f}")
    
    perf_optimized = plant.simulate_full_plant(feed_rate=100.0)
    perf_optimized.display()
    
    # 改善率
    improvement = (perf_optimized.profit_per_hour - perf_normal.profit_per_hour) / perf_normal.profit_per_hour * 100
    print(f"\nProfit improvement: {improvement:.2f}%")
    

* * *

## 4.8 動的シミュレーション入門

### 時間変化を考慮したシミュレーション

起動・停止、外乱応答などの動的挙動を解析します。
    
    
    """
    Example 8: Dynamic Simulation Introduction
    動的シミュレーション入門（時間変化の考慮）
    """
    import numpy as np
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    
    class DynamicCSTR:
        """動的連続槽型反応器（CSTR）"""
    
        def __init__(self, volume=10.0, k=0.5, rho=1000, Cp=4.18):
            """
            Args:
                volume: 反応器容積 [m³]
                k: 反応速度定数 [1/min]
                rho: 密度 [kg/m³]
                Cp: 比熱 [kJ/kg/K]
            """
            self.V = volume
            self.k = k
            self.rho = rho
            self.Cp = Cp
    
        def model(self, y, t, F_in, C_in, T_in, Q_heat):
            """
            動的モデル（微分方程式）
    
            状態変数: y = [C, T]
            C: 濃度 [kmol/m³]
            T: 温度 [K]
            """
            C, T = y
    
            # 物質収支
            r = self.k * C  # 反応速度 [kmol/m³/min]
            dC_dt = (F_in / self.V) * (C_in - C) - r
    
            # エネルギー収支
            dH_r = -50  # 反応熱 [kJ/kmol]
            Q_reaction = r * dH_r * self.V
    
            dT_dt = (F_in / self.V) * (T_in - T) + \
                    (Q_heat + Q_reaction) / (self.rho * self.V * self.Cp)
    
            return [dC_dt, dT_dt]
    
        def simulate(self, t_span, y0, F_in, C_in, T_in, Q_heat):
            """
            動的シミュレーション
    
            Args:
                t_span: 時間範囲 [min]
                y0: 初期状態 [C0, T0]
                F_in: 入口流量 [m³/min]（時間の関数も可）
                C_in: 入口濃度 [kmol/m³]
                T_in: 入口温度 [K]
                Q_heat: 加熱量 [kW]
            """
            solution = odeint(self.model, y0, t_span,
                             args=(F_in, C_in, T_in, Q_heat))
    
            return solution
    
    # 使用例：起動シミュレーション
    reactor = DynamicCSTR(volume=10.0, k=0.5)
    
    # 時間範囲
    t = np.linspace(0, 60, 300)  # 0-60分
    
    # 初期条件（空の反応器）
    y0 = [0.0, 300.0]  # [C0=0, T0=300K]
    
    # 運転条件
    F_in = 1.0  # m³/min
    C_in = 2.0  # kmol/m³
    T_in = 320.0  # K
    Q_heat = 50.0  # kW
    
    # シミュレーション実行
    solution = reactor.simulate(t, y0, F_in, C_in, T_in, Q_heat)
    
    # 結果プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 濃度の時間変化
    ax1.plot(t, solution[:, 0], 'b-', linewidth=2)
    ax1.set_xlabel('Time [min]')
    ax1.set_ylabel('Concentration [kmol/m³]')
    ax1.set_title('CSTR Startup: Concentration Profile')
    ax1.grid(True, alpha=0.3)
    
    # 温度の時間変化
    ax2.plot(t, solution[:, 1], 'r-', linewidth=2)
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Temperature [K]')
    ax2.set_title('CSTR Startup: Temperature Profile')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 定常状態到達時間
    C_steady = solution[-1, 0]
    idx_90 = np.where(solution[:, 0] >= 0.9 * C_steady)[0][0]
    print(f"Time to reach 90% of steady state: {t[idx_90]:.2f} min")
    

* * *

## 学習目標の確認

この章を完了すると、以下ができるようになります：

### 基本理解

  * ✅ フローシートをグラフ理論で表現し、計算順序を決定できる
  * ✅ リサイクルストリームの収束計算の原理を理解する
  * ✅ ヒートインテグレーションの基礎（Pinch解析）を説明できる

### 実践スキル

  * ✅ NetworkXを使ってフローシートトポロジーを構築できる
  * ✅ 逐次代入法とWegstein法でリサイクル計算を実装できる
  * ✅ 複数ユニットを統合した完全フローシートを構築できる
  * ✅ 感度分析とパラメータ最適化を実行できる

### 応用力

  * ✅ 実プラントの経済性を評価し、最適運転条件を決定できる
  * ✅ エネルギー統合でユーティリティコストを削減できる
  * ✅ 動的シミュレーションで起動・外乱応答を解析できる

* * *

## まとめ

この章では、プロセスフローシートシミュレーションの実践的手法を習得しました：

  * **トポロジー表現** ：グラフ理論で計算順序を自動決定
  * **リサイクル収束** ：逐次代入法とWegstein加速法
  * **統合シミュレーション** ：反応・分離・熱交換の全体最適化
  * **感度分析** ：パラメータ変化の影響を定量評価
  * **経済性最適化** ：利益最大化の運転条件決定
  * **ヒートインテグレーション** ：Pinch解析でエネルギー削減
  * **動的挙動** ：時間変化を考慮した起動・外乱シミュレーション

**次章の予告** ：第5章では、PythonとDWSIM等の商用シミュレーターを連携させ、より高度なプロセス設計を実現します。

* * *

### 免責事項

  * 本コンテンツは教育目的で作成されており、実プラント設計には専門家の監修が必要です
  * コード例は概念理解のための簡略化モデルであり、実システムには追加検証が必要です
  * 数値例は説明用であり、実際のプロセスパラメータとは異なる場合があります
  * 安全性・環境規制は実際のプロジェクトで必ず専門家に確認してください
