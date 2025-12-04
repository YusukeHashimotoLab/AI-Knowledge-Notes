---
title: 第2章：統計的プロセス管理（SPC）
chapter_title: 第2章：統計的プロセス管理（SPC）
subtitle: 管理図の作成とプロセス能力評価の実践
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 統計的プロセス管理（SPC）の基礎理論を理解する
  * ✅ シューハート管理図（X̄-R, I-MR）を作成し解釈できる
  * ✅ プロセス能力指数（Cp, Cpk）を計算し評価できる
  * ✅ 高度なSPC手法（CUSUM, EWMA, Hotelling's T²）を実装できる
  * ✅ 管理図の異常判定ルールを適用できる

* * *

## 2.1 統計的プロセス管理（SPC）の基礎

### SPCとは

**統計的プロセス管理（SPC: Statistical Process Control）** は、統計的手法を用いてプロセスの変動を監視し、異常を早期に検出してプロセスを管理状態に保つための手法です。

**SPCの主な目的:**

  * **プロセスの安定性確認** : 統計的管理状態にあることを確認
  * **異常の早期検出** : 特殊原因による変動を迅速に発見
  * **プロセス能力評価** : 規格を満たす能力があるかを定量評価
  * **継続的改善** : データに基づく改善活動の推進
  * **予防保全** : 問題が発生する前に対処

### 変動の2つのタイプ

変動のタイプ | 別名 | 特徴 | 原因例 | 対応  
---|---|---|---|---  
**偶然原因（Common Cause）** | システム内変動 | 予測可能、常時存在 | 測定誤差、原料のばらつき、環境変動 | システム改善  
**特殊原因（Special Cause）** | システム外変動 | 予測不能、突発的 | 設備故障、操作ミス、原料異常 | 即座に対処  
  
### 管理図の基本構造
    
    
    ```mermaid
    graph TD
        A[プロセスデータ] --> B[サンプリング]
        B --> C[統計量計算]
        C --> D[管理図プロット]
        D --> E{管理限界内？}
        E -->|Yes| F[プロセス安定]
        E -->|No| G[異常検知]
        G --> H[原因調査と対策]
        H --> A
    
        style F fill:#c8e6c9
        style G fill:#ffcdd2
        style D fill:#b3e5fc
    ```

**管理図の基本要素:**

  * **中心線（CL: Center Line）** : プロセスの平均値
  * **上方管理限界（UCL: Upper Control Limit）** : CL + 3σ
  * **下方管理限界（LCL: Lower Control Limit）** : CL - 3σ
  * **データポイント** : 時系列にプロットされた統計量

* * *

## 2.2 コード例：管理図とプロセス能力分析

#### コード例1: X̄-R管理図（平均値と範囲の管理図）の実装

**目的** : サブグループデータからX̄-R管理図を作成し、プロセスの安定性を評価する。
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 日本語フォント設定
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    np.random.seed(42)
    
    # シミュレーションデータ生成
    # 25サブグループ、各サブグループ5個のサンプル
    n_subgroups = 25
    subgroup_size = 5
    
    # プロセスデータ（製品の重量、目標値: 100g）
    data = []
    for i in range(n_subgroups):
        # 正常なプロセス（最初の20サブグループ）
        if i < 20:
            subgroup = np.random.normal(100, 2, subgroup_size)
        # プロセス平均が2gシフト（最後の5サブグループ）
        else:
            subgroup = np.random.normal(102, 2, subgroup_size)
        data.append(subgroup)
    
    data = np.array(data)
    
    # 統計量の計算
    xbar = np.mean(data, axis=1)  # サブグループ平均
    R = np.max(data, axis=1) - np.min(data, axis=1)  # サブグループ範囲
    
    # 管理図定数（n=5の場合）
    # これらの定数は統計的に導出された値
    A2 = 0.577  # X̄管理図用
    D3 = 0.0    # R管理図下限用
    D4 = 2.115  # R管理図上限用
    
    # X̄管理図の管理限界
    xbar_center = np.mean(xbar)
    R_bar = np.mean(R)
    xbar_UCL = xbar_center + A2 * R_bar
    xbar_LCL = xbar_center - A2 * R_bar
    
    # R管理図の管理限界
    R_center = R_bar
    R_UCL = D4 * R_bar
    R_LCL = D3 * R_bar
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # X̄管理図
    axes[0].plot(range(1, n_subgroups + 1), xbar, 'o-', color='#11998e',
                 markersize=6, linewidth=1.5, label='サブグループ平均')
    axes[0].axhline(y=xbar_center, color='blue', linestyle='-', linewidth=2, label=f'中心線 (CL={xbar_center:.2f})')
    axes[0].axhline(y=xbar_UCL, color='red', linestyle='--', linewidth=2, label=f'UCL={xbar_UCL:.2f}')
    axes[0].axhline(y=xbar_LCL, color='red', linestyle='--', linewidth=2, label=f'LCL={xbar_LCL:.2f}')
    
    # 管理限界外のポイントを強調
    out_of_control = (xbar > xbar_UCL) | (xbar < xbar_LCL)
    if out_of_control.any():
        axes[0].scatter(np.where(out_of_control)[0] + 1, xbar[out_of_control],
                       color='red', s=100, marker='o', zorder=5, label='管理限界外')
    
    axes[0].set_xlabel('サブグループ番号', fontsize=12)
    axes[0].set_ylabel('平均値 X̄ (g)', fontsize=12)
    axes[0].set_title('X̄管理図（平均値管理図）', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=0.3)
    
    # R管理図
    axes[1].plot(range(1, n_subgroups + 1), R, 'o-', color='#f59e0b',
                 markersize=6, linewidth=1.5, label='サブグループ範囲')
    axes[1].axhline(y=R_center, color='blue', linestyle='-', linewidth=2, label=f'中心線 (CL={R_center:.2f})')
    axes[1].axhline(y=R_UCL, color='red', linestyle='--', linewidth=2, label=f'UCL={R_UCL:.2f}')
    if R_LCL > 0:
        axes[1].axhline(y=R_LCL, color='red', linestyle='--', linewidth=2, label=f'LCL={R_LCL:.2f}')
    
    # 管理限界外のポイントを強調
    out_of_control_R = (R > R_UCL) | (R < R_LCL)
    if out_of_control_R.any():
        axes[1].scatter(np.where(out_of_control_R)[0] + 1, R[out_of_control_R],
                       color='red', s=100, marker='o', zorder=5, label='管理限界外')
    
    axes[1].set_xlabel('サブグループ番号', fontsize=12)
    axes[1].set_ylabel('範囲 R (g)', fontsize=12)
    axes[1].set_title('R管理図（範囲管理図）', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 統計サマリー
    print("=== X̄-R管理図 統計サマリー ===")
    print(f"\n【X̄管理図】")
    print(f"  中心線（CL）: {xbar_center:.3f} g")
    print(f"  上方管理限界（UCL）: {xbar_UCL:.3f} g")
    print(f"  下方管理限界（LCL）: {xbar_LCL:.3f} g")
    print(f"  管理限界外のポイント数: {out_of_control.sum()}/{n_subgroups}")
    
    print(f"\n【R管理図】")
    print(f"  中心線（CL）: {R_center:.3f} g")
    print(f"  上方管理限界（UCL）: {R_UCL:.3f} g")
    print(f"  下方管理限界（LCL）: {R_LCL:.3f} g")
    print(f"  管理限界外のポイント数: {out_of_control_R.sum()}/{n_subgroups}")
    
    # 判定
    if out_of_control.any():
        print(f"\n⚠️ 警告: サブグループ {np.where(out_of_control)[0] + 1} でプロセス平均の異常を検出")
    if out_of_control_R.any():
        print(f"⚠️ 警告: サブグループ {np.where(out_of_control_R)[0] + 1} でプロセス変動の異常を検出")
    if not (out_of_control.any() or out_of_control_R.any()):
        print("\n✅ プロセスは統計的管理状態にあります")
    

**期待される出力** :
    
    
    === X̄-R管理図 統計サマリー ===
    
    【X̄管理図】
      中心線（CL）: 100.405 g
      上方管理限界（UCL）: 103.088 g
      下方管理限界（LCL）: 97.722 g
      管理限界外のポイント数: 3/25
    
    【R管理図】
      中心線（CL）: 4.651 g
      上方管理限界（UCL）: 9.836 g
      下方管理限界（LCL）: 0.000 g
      管理限界外のポイント数: 0/25
    
    ⚠️ 警告: サブグループ [22 23 25] でプロセス平均の異常を検出
    

**解説** : X̄-R管理図は、サブグループデータのプロセス平均（X̄管理図）とプロセス変動（R管理図）を同時に監視します。この例では、サブグループ21以降でプロセス平均が2gシフトしており、X̄管理図がこれを検出しています。X̄管理図とR管理図は常にペアで使用し、両方が管理状態にある場合のみ「プロセスが安定」と判断します。

#### コード例2: I-MR管理図（個別値と移動範囲の管理図）

**目的** : 個別測定値からI-MR管理図を作成する（サブグループ化できない場合）。
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # 個別測定値データ生成（1日1回の測定、30日間）
    n_observations = 30
    individual_values = np.random.normal(50, 3, n_observations)
    
    # 15日目以降にプロセス変動が増加
    individual_values[15:] += np.random.normal(0, 5, n_observations - 15)
    
    # 移動範囲（Moving Range）の計算
    # mR = |X_i - X_{i-1}|
    moving_range = np.abs(np.diff(individual_values))
    
    # 統計量の計算
    I_bar = np.mean(individual_values)
    mR_bar = np.mean(moving_range)
    
    # 管理図定数（移動範囲 n=2）
    d2 = 1.128  # 移動範囲の期待値の定数
    E2 = 2.66   # I管理図用
    D3 = 0.0    # mR管理図下限用
    D4 = 3.267  # mR管理図上限用
    
    # I管理図の管理限界
    I_UCL = I_bar + E2 * mR_bar
    I_LCL = I_bar - E2 * mR_bar
    
    # mR管理図の管理限界
    mR_center = mR_bar
    mR_UCL = D4 * mR_bar
    mR_LCL = D3 * mR_bar
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # I管理図（個別値管理図）
    axes[0].plot(range(1, n_observations + 1), individual_values, 'o-',
                 color='#11998e', markersize=6, linewidth=1.5, label='個別測定値')
    axes[0].axhline(y=I_bar, color='blue', linestyle='-', linewidth=2,
                    label=f'中心線 (CL={I_bar:.2f})')
    axes[0].axhline(y=I_UCL, color='red', linestyle='--', linewidth=2,
                    label=f'UCL={I_UCL:.2f}')
    axes[0].axhline(y=I_LCL, color='red', linestyle='--', linewidth=2,
                    label=f'LCL={I_LCL:.2f}')
    
    # 管理限界外のポイントを検出
    out_of_control_I = (individual_values > I_UCL) | (individual_values < I_LCL)
    if out_of_control_I.any():
        axes[0].scatter(np.where(out_of_control_I)[0] + 1,
                       individual_values[out_of_control_I],
                       color='red', s=100, marker='o', zorder=5, label='管理限界外')
    
    axes[0].set_xlabel('観測番号', fontsize=12)
    axes[0].set_ylabel('測定値 (単位)', fontsize=12)
    axes[0].set_title('I管理図（個別値管理図）', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=0.3)
    
    # mR管理図（移動範囲管理図）
    axes[1].plot(range(2, n_observations + 1), moving_range, 'o-',
                 color='#f59e0b', markersize=6, linewidth=1.5, label='移動範囲')
    axes[1].axhline(y=mR_center, color='blue', linestyle='-', linewidth=2,
                    label=f'中心線 (CL={mR_center:.2f})')
    axes[1].axhline(y=mR_UCL, color='red', linestyle='--', linewidth=2,
                    label=f'UCL={mR_UCL:.2f}')
    if mR_LCL > 0:
        axes[1].axhline(y=mR_LCL, color='red', linestyle='--', linewidth=2,
                        label=f'LCL={mR_LCL:.2f}')
    
    # 管理限界外のポイントを検出
    out_of_control_mR = (moving_range > mR_UCL) | (moving_range < mR_LCL)
    if out_of_control_mR.any():
        axes[1].scatter(np.where(out_of_control_mR)[0] + 2,
                       moving_range[out_of_control_mR],
                       color='red', s=100, marker='o', zorder=5, label='管理限界外')
    
    axes[1].set_xlabel('観測番号', fontsize=12)
    axes[1].set_ylabel('移動範囲 mR (単位)', fontsize=12)
    axes[1].set_title('mR管理図（移動範囲管理図）', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== I-MR管理図 統計サマリー ===")
    print(f"\n【I管理図】")
    print(f"  中心線（CL）: {I_bar:.3f}")
    print(f"  上方管理限界（UCL）: {I_UCL:.3f}")
    print(f"  下方管理限界（LCL）: {I_LCL:.3f}")
    print(f"  管理限界外のポイント数: {out_of_control_I.sum()}/{n_observations}")
    
    print(f"\n【mR管理図】")
    print(f"  中心線（CL）: {mR_center:.3f}")
    print(f"  上方管理限界（UCL）: {mR_UCL:.3f}")
    print(f"  下方管理限界（LCL）: {mR_LCL:.3f}")
    print(f"  管理限界外のポイント数: {out_of_control_mR.sum()}/{len(moving_range)}")
    
    if out_of_control_mR.any():
        print(f"\n⚠️ 警告: プロセス変動の増加を検出（観測 {np.where(out_of_control_mR)[0] + 2}）")
    

**解説** : I-MR管理図は、サブグループ化できない個別測定値（例：1日1回の検査、バッチプロセス、高価な破壊試験）に使用します。移動範囲（連続する2つの測定値の差の絶対値）を使ってプロセス変動を推定します。この例では、15日目以降でプロセス変動が増加しており、mR管理図がこれを検出しています。

#### コード例3: プロセス能力指数（Cp, Cpk）の計算と解釈

**目的** : プロセス能力指数を計算し、プロセスが規格を満たす能力を評価する。
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    np.random.seed(42)
    
    # プロセスデータ生成（製品寸法、単位: mm）
    n_samples = 200
    process_data = np.random.normal(10.0, 0.15, n_samples)
    
    # 規格限界
    USL = 10.5  # 上側規格限界（Upper Specification Limit）
    LSL = 9.5   # 下側規格限界（Lower Specification Limit）
    target = 10.0  # 目標値
    
    # プロセス統計量の計算
    process_mean = np.mean(process_data)
    process_std = np.std(process_data, ddof=1)  # 不偏標準偏差
    
    # プロセス能力指数の計算
    # Cp: プロセス能力指数（Process Capability）
    Cp = (USL - LSL) / (6 * process_std)
    
    # Cpk: プロセス能力指数（プロセス平均のずれを考慮）
    Cpk_upper = (USL - process_mean) / (3 * process_std)
    Cpk_lower = (process_mean - LSL) / (3 * process_std)
    Cpk = min(Cpk_upper, Cpk_lower)
    
    # Cpm: 目標値からのずれを考慮したプロセス能力指数
    Cpm = (USL - LSL) / (6 * np.sqrt(process_std**2 + (process_mean - target)**2))
    
    # 規格外率の計算（理論値）
    # プロセスが正規分布に従うと仮定
    ppm_below_LSL = stats.norm.cdf(LSL, loc=process_mean, scale=process_std) * 1e6
    ppm_above_USL = (1 - stats.norm.cdf(USL, loc=process_mean, scale=process_std)) * 1e6
    total_ppm = ppm_below_LSL + ppm_above_USL
    
    # 実際の規格外数（サンプルデータ）
    actual_out_of_spec = np.sum((process_data < LSL) | (process_data > USL))
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ヒストグラムと規格限界
    axes[0].hist(process_data, bins=30, density=True, alpha=0.6, color='#11998e',
                 edgecolor='black', label='プロセスデータ')
    
    # 正規分布の理論曲線
    x = np.linspace(process_data.min(), process_data.max(), 100)
    axes[0].plot(x, stats.norm.pdf(x, loc=process_mean, scale=process_std),
                 'r-', linewidth=2, label='正規分布（理論）')
    
    # 規格限界と目標値
    axes[0].axvline(x=LSL, color='red', linestyle='--', linewidth=2, label=f'LSL = {LSL}')
    axes[0].axvline(x=USL, color='red', linestyle='--', linewidth=2, label=f'USL = {USL}')
    axes[0].axvline(x=target, color='green', linestyle='--', linewidth=2, label=f'目標値 = {target}')
    axes[0].axvline(x=process_mean, color='blue', linestyle='-', linewidth=2,
                    label=f'プロセス平均 = {process_mean:.3f}')
    
    axes[0].set_xlabel('測定値 (mm)', fontsize=12)
    axes[0].set_ylabel('確率密度', fontsize=12)
    axes[0].set_title('プロセス分布と規格限界', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    
    # プロセス能力指数の可視化
    capability_indices = {
        'Cp': Cp,
        'Cpk': Cpk,
        'Cpm': Cpm
    }
    
    colors = ['#11998e', '#f59e0b', '#7b2cbf']
    bars = axes[1].bar(capability_indices.keys(), capability_indices.values(),
                       color=colors, alpha=0.7, edgecolor='black')
    
    # 能力評価基準線
    axes[1].axhline(y=1.33, color='orange', linestyle='--', linewidth=2,
                    label='良好基準 (1.33)')
    axes[1].axhline(y=1.67, color='green', linestyle='--', linewidth=2,
                    label='優秀基準 (1.67)')
    
    # 各バーに数値を表示
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    axes[1].set_ylabel('能力指数', fontsize=12)
    axes[1].set_title('プロセス能力指数の比較', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].set_ylim(0, max(capability_indices.values()) * 1.2)
    
    plt.tight_layout()
    plt.show()
    
    # 結果サマリー
    print("=== プロセス能力分析 ===")
    print(f"\n【プロセス統計量】")
    print(f"  プロセス平均: {process_mean:.4f} mm")
    print(f"  プロセス標準偏差: {process_std:.4f} mm")
    print(f"  サンプル数: {n_samples}")
    
    print(f"\n【規格情報】")
    print(f"  上側規格限界（USL）: {USL} mm")
    print(f"  下側規格限界（LSL）: {LSL} mm")
    print(f"  目標値: {target} mm")
    print(f"  規格幅: {USL - LSL} mm")
    
    print(f"\n【プロセス能力指数】")
    print(f"  Cp  = {Cp:.3f}  （規格幅とプロセス幅の比）")
    print(f"  Cpk = {Cpk:.3f}  （プロセス平均のずれを考慮）")
    print(f"  Cpm = {Cpm:.3f}  （目標値からのずれを考慮）")
    
    print(f"\n【規格外率（理論値）】")
    print(f"  下側規格外: {ppm_below_LSL:.2f} ppm")
    print(f"  上側規格外: {ppm_above_USL:.2f} ppm")
    print(f"  合計規格外: {total_ppm:.2f} ppm ({total_ppm/1e4:.4f}%)")
    
    print(f"\n【実際の規格外数】")
    print(f"  規格外サンプル数: {actual_out_of_spec}/{n_samples} ({actual_out_of_spec/n_samples*100:.2f}%)")
    
    # 能力判定
    print(f"\n【プロセス能力判定】")
    if Cpk >= 1.67:
        print("  ✅ 優秀: プロセスは十分な能力があります（Cpk ≥ 1.67）")
    elif Cpk >= 1.33:
        print("  ✅ 良好: プロセスは適切な能力があります（Cpk ≥ 1.33）")
    elif Cpk >= 1.0:
        print("  ⚠️  最低限: プロセス能力は最低限です（Cpk ≥ 1.0）、改善を推奨")
    else:
        print("  ❌ 不十分: プロセス能力が不足しています（Cpk < 1.0）、早急な改善が必要")
    

**解説** : プロセス能力指数は、プロセスが規格を満たす能力を定量評価します。**Cp** はプロセス変動のみを考慮し、**Cpk** はプロセス平均のずれも考慮します。一般的に、Cpk ≥ 1.33が「良好」、Cpk ≥ 1.67が「優秀」とされます。Cpkが1.0未満の場合、規格外品が多く発生する可能性が高いため、プロセス改善が必要です。

#### コード例4: CUSUM管理図（累積和管理図）の実装

**目的** : CUSUM管理図で小さなプロセス平均のシフトを検出する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # プロセスデータ生成
    n_samples = 100
    target_mean = 50.0
    process_std = 2.0
    
    # データ生成（40サンプル目からプロセス平均が0.5σシフト）
    data = np.zeros(n_samples)
    data[:40] = np.random.normal(target_mean, process_std, 40)
    data[40:] = np.random.normal(target_mean + 0.5 * process_std, process_std, 60)
    
    # CUSUM パラメータ設定
    # K: 参照値（検出したいシフトの半分）
    # H: 決定区間（アラームを出す閾値）
    shift_to_detect = 0.5 * process_std  # 0.5σのシフトを検出
    K = shift_to_detect / 2
    H = 5 * process_std  # 一般的には4σ~5σ
    
    # CUSUM計算
    # 上側CUSUM (プロセス平均の増加を検出)
    Cp = np.zeros(n_samples)
    # 下側CUSUM (プロセス平均の減少を検出)
    Cn = np.zeros(n_samples)
    
    for i in range(1, n_samples):
        Cp[i] = max(0, Cp[i-1] + (data[i] - target_mean) - K)
        Cn[i] = max(0, Cn[i-1] - (data[i] - target_mean) - K)
    
    # アラーム検出
    alarm_high = Cp > H
    alarm_low = Cn > H
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 元データのプロット
    axes[0].plot(range(n_samples), data, 'o-', color='#11998e',
                 markersize=4, linewidth=1, alpha=0.7, label='測定値')
    axes[0].axhline(y=target_mean, color='blue', linestyle='-', linewidth=2,
                    label=f'目標値 = {target_mean}')
    axes[0].axvline(x=40, color='red', linestyle='--', alpha=0.5,
                    label='プロセス変化点')
    axes[0].set_ylabel('測定値', fontsize=11)
    axes[0].set_title('プロセスデータ（40サンプル目から0.5σシフト）',
                      fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 上側CUSUM
    axes[1].plot(range(n_samples), Cp, 'o-', color='#f59e0b',
                 markersize=4, linewidth=1.5, label='上側CUSUM (C+)')
    axes[1].axhline(y=H, color='red', linestyle='--', linewidth=2,
                    label=f'決定区間 H = {H:.1f}')
    axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=1)
    if alarm_high.any():
        first_alarm = np.where(alarm_high)[0][0]
        axes[1].scatter(np.where(alarm_high)[0], Cp[alarm_high],
                       color='red', s=80, marker='o', zorder=5, label='アラーム')
        axes[1].axvline(x=first_alarm, color='red', linestyle=':', alpha=0.7)
    axes[1].set_ylabel('CUSUM C+', fontsize=11)
    axes[1].set_title('上側CUSUM管理図（プロセス平均の増加検出）',
                      fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 下側CUSUM
    axes[2].plot(range(n_samples), Cn, 'o-', color='#7b2cbf',
                 markersize=4, linewidth=1.5, label='下側CUSUM (C-)')
    axes[2].axhline(y=H, color='red', linestyle='--', linewidth=2,
                    label=f'決定区間 H = {H:.1f}')
    axes[2].axhline(y=0, color='gray', linestyle='-', linewidth=1)
    if alarm_low.any():
        first_alarm_low = np.where(alarm_low)[0][0]
        axes[2].scatter(np.where(alarm_low)[0], Cn[alarm_low],
                       color='red', s=80, marker='o', zorder=5, label='アラーム')
        axes[2].axvline(x=first_alarm_low, color='red', linestyle=':', alpha=0.7)
    axes[2].set_xlabel('サンプル番号', fontsize=11)
    axes[2].set_ylabel('CUSUM C-', fontsize=11)
    axes[2].set_title('下側CUSUM管理図（プロセス平均の減少検出）',
                      fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== CUSUM管理図 分析結果 ===")
    print(f"\nパラメータ設定:")
    print(f"  目標値（μ₀）: {target_mean}")
    print(f"  標準偏差（σ）: {process_std}")
    print(f"  参照値（K）: {K:.2f} (検出シフトの半分)")
    print(f"  決定区間（H）: {H:.2f}")
    
    print(f"\nプロセス統計:")
    print(f"  前半（1-40）平均: {np.mean(data[:40]):.2f}")
    print(f"  後半（41-100）平均: {np.mean(data[40:]):.2f}")
    print(f"  実際のシフト: {np.mean(data[40:]) - np.mean(data[:40]):.2f}")
    
    if alarm_high.any():
        first_alarm = np.where(alarm_high)[0][0]
        detection_delay = first_alarm - 40
        print(f"\n✅ 上側アラーム検出:")
        print(f"  最初のアラーム: サンプル {first_alarm + 1}")
        print(f"  検出遅れ: {detection_delay} サンプル")
    else:
        print("\n❌ 上側アラーム: 検出なし")
    
    if alarm_low.any():
        first_alarm_low = np.where(alarm_low)[0][0]
        print(f"\n✅ 下側アラーム検出: サンプル {first_alarm_low + 1}")
    else:
        print("\n❌ 下側アラーム: 検出なし")
    

**解説** : CUSUM（Cumulative Sum）管理図は、シューハート管理図では検出が難しい小さなプロセス平均のシフト（0.5σ〜2σ）を効率的に検出します。累積和を計算することで、持続的な小さな変化を増幅して検出します。この例では、40サンプル目からの0.5σシフトを、従来のX̄管理図より早く検出できることを示しています。

#### コード例5: EWMA管理図（指数加重移動平均管理図）の実装

**目的** : EWMA管理図で時系列の重み付けにより異常を検出する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # プロセスデータ生成
    n_samples = 100
    target_mean = 100.0
    process_std = 3.0
    
    # データ生成（50サンプル目からプロセス平均が1σシフト）
    data = np.zeros(n_samples)
    data[:50] = np.random.normal(target_mean, process_std, 50)
    data[50:] = np.random.normal(target_mean + 1.0 * process_std, process_std, 50)
    
    # EWMAパラメータ
    # λ (lambda): 重み付け定数 (0 < λ ≤ 1)
    # λが小さい: 過去のデータに重きを置く（小さな変化に敏感）
    # λが大きい: 現在のデータに重きを置く（大きな変化に敏感）
    lambda_param = 0.2
    L = 3  # 管理限界の係数（通常2.7～3）
    
    # EWMA計算
    z = np.zeros(n_samples)
    z[0] = target_mean  # 初期値は目標値
    
    for i in range(1, n_samples):
        z[i] = lambda_param * data[i] + (1 - lambda_param) * z[i-1]
    
    # 管理限界の計算
    # 各時点での管理限界（時間とともに収束）
    ucl = np.zeros(n_samples)
    lcl = np.zeros(n_samples)
    
    for i in range(n_samples):
        # 管理限界の標準偏差
        std_z = process_std * np.sqrt(lambda_param / (2 - lambda_param) *
                                       (1 - (1 - lambda_param)**(2 * (i + 1))))
        ucl[i] = target_mean + L * std_z
        lcl[i] = target_mean - L * std_z
    
    # アラーム検出
    alarm = (z > ucl) | (z < lcl)
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 元データ
    axes[0].plot(range(n_samples), data, 'o', color='lightgray',
                 markersize=4, alpha=0.5, label='個別測定値')
    axes[0].axhline(y=target_mean, color='blue', linestyle='-', linewidth=2,
                    label=f'目標値 = {target_mean}')
    axes[0].axvline(x=50, color='red', linestyle='--', alpha=0.5,
                    label='プロセス変化点')
    axes[0].set_ylabel('測定値', fontsize=11)
    axes[0].set_title('プロセスデータ（50サンプル目から1σシフト）',
                      fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # EWMA管理図
    axes[1].plot(range(n_samples), z, 'o-', color='#11998e',
                 markersize=5, linewidth=1.5, label=f'EWMA (λ={lambda_param})')
    axes[1].plot(range(n_samples), ucl, 'r--', linewidth=2, label=f'UCL (L={L})')
    axes[1].plot(range(n_samples), lcl, 'r--', linewidth=2, label=f'LCL (L={L})')
    axes[1].axhline(y=target_mean, color='blue', linestyle='-', linewidth=2,
                    label=f'中心線 = {target_mean}')
    
    # アラームポイントを強調
    if alarm.any():
        first_alarm = np.where(alarm)[0][0]
        axes[1].scatter(np.where(alarm)[0], z[alarm],
                       color='red', s=100, marker='o', zorder=5, label='アラーム')
        axes[1].axvline(x=first_alarm, color='red', linestyle=':', alpha=0.7)
    
    axes[1].set_xlabel('サンプル番号', fontsize=11)
    axes[1].set_ylabel('EWMA統計量', fontsize=11)
    axes[1].set_title('EWMA管理図', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== EWMA管理図 分析結果 ===")
    print(f"\nパラメータ設定:")
    print(f"  目標値（μ₀）: {target_mean}")
    print(f"  標準偏差（σ）: {process_std}")
    print(f"  重み付け定数（λ）: {lambda_param}")
    print(f"  管理限界係数（L）: {L}")
    
    print(f"\nプロセス統計:")
    print(f"  前半（1-50）平均: {np.mean(data[:50]):.2f}")
    print(f"  後半（51-100）平均: {np.mean(data[50:]):.2f}")
    print(f"  実際のシフト: {np.mean(data[50:]) - target_mean:.2f}")
    
    print(f"\nEWMA統計:")
    print(f"  前半EWMA平均: {np.mean(z[:50]):.2f}")
    print(f"  後半EWMA平均: {np.mean(z[50:]):.2f}")
    
    if alarm.any():
        first_alarm = np.where(alarm)[0][0]
        detection_delay = first_alarm - 50
        print(f"\n✅ アラーム検出:")
        print(f"  最初のアラーム: サンプル {first_alarm + 1}")
        print(f"  検出遅れ: {detection_delay} サンプル")
        print(f"  アラーム総数: {alarm.sum()}")
    else:
        print("\n❌ アラーム: 検出なし")
    
    # λの影響を比較
    print(f"\n【λパラメータの影響】")
    print(f"  λ = 0.1 : 過去20サンプルを重視（小さなシフトに敏感）")
    print(f"  λ = 0.2 : 過去10サンプルを重視（バランス型）← 今回")
    print(f"  λ = 0.5 : 過去4サンプルを重視（大きなシフトに敏感）")
    print(f"  λ = 1.0 : 現在のサンプルのみ（シューハート管理図と同等）")
    

**解説** : EWMA（Exponentially Weighted Moving Average）管理図は、過去のデータに指数的に減衰する重みを付けて平均を計算します。λパラメータにより、小さなシフトへの感度を調整できます。λが小さいほど小さなシフトの検出に優れ、λが大きいほど大きなシフトの検出に優れます。一般的にλ=0.2が推奨されます。

#### コード例6: Hotelling's T²管理図（多変量管理図）

**目的** : 複数の変数を同時に監視する多変量管理図を実装する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    np.random.seed(42)
    
    # 2変数の多変量プロセスデータ生成
    n_samples = 100
    
    # 相関のある2変数（温度と圧力）
    mean_vector = np.array([100, 50])  # 温度100°C、圧力50kPa
    # 共分散行列（正の相関）
    cov_matrix = np.array([[9, 4],
                           [4, 4]])
    
    # 正常データ（最初の70サンプル）
    data_normal = np.random.multivariate_normal(mean_vector, cov_matrix, 70)
    
    # 異常データ（最後の30サンプル：温度が上昇）
    mean_abnormal = np.array([105, 51])
    data_abnormal = np.random.multivariate_normal(mean_abnormal, cov_matrix, 30)
    
    # データの結合
    data = np.vstack([data_normal, data_abnormal])
    
    # Hotelling's T²統計量の計算
    # T² = (x - μ)ᵀ Σ⁻¹ (x - μ)
    # ここで、μは平均ベクトル、Σは共分散行列
    
    # 正常データから平均と共分散を推定（最初の70サンプル）
    mean_est = np.mean(data_normal, axis=0)
    cov_est = np.cov(data_normal, rowvar=False)
    cov_inv = np.linalg.inv(cov_est)
    
    # 各サンプルのT²統計量を計算
    T2_values = np.zeros(n_samples)
    for i in range(n_samples):
        diff = data[i] - mean_est
        T2_values[i] = diff @ cov_inv @ diff
    
    # 管理限界の計算
    # フェーズI（過去データの評価）の管理限界
    p = 2  # 変数の数
    m = 70  # 正常データのサンプル数
    alpha = 0.01  # 有意水準
    
    # ベータ分布に基づく管理限界
    UCL = (p * (m + 1) * (m - 1) / (m * (m - p))) * \
          stats.f.ppf(1 - alpha, p, m - p)
    
    # アラーム検出
    alarm = T2_values > UCL
    
    # 可視化
    fig = plt.figure(figsize=(16, 10))
    
    # 2変数の散布図
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(data_normal[:, 0], data_normal[:, 1], c='#11998e', s=40,
               alpha=0.6, label='正常データ (1-70)')
    ax1.scatter(data_abnormal[:, 0], data_abnormal[:, 1], c='orange', s=40,
               alpha=0.6, label='異常データ (71-100)')
    
    # 管理限界楕円を描画
    from matplotlib.patches import Ellipse
    
    # T²=UCLの楕円
    eigenvalues, eigenvectors = np.linalg.eig(cov_est)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    width, height = 2 * np.sqrt(UCL * eigenvalues)
    
    ellipse = Ellipse(mean_est, width, height, angle=np.degrees(angle),
                     facecolor='none', edgecolor='red', linewidth=2,
                     linestyle='--', label='管理限界 (T²=UCL)')
    ax1.add_patch(ellipse)
    
    ax1.scatter(mean_est[0], mean_est[1], c='blue', s=100, marker='x',
               linewidths=3, label='プロセス中心')
    ax1.set_xlabel('温度 (°C)', fontsize=11)
    ax1.set_ylabel('圧力 (kPa)', fontsize=11)
    ax1.set_title('2変数散布図と管理限界楕円', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # T²管理図
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(range(1, n_samples + 1), T2_values, 'o-', color='#11998e',
            markersize=5, linewidth=1.5, label='T²統計量')
    ax2.axhline(y=UCL, color='red', linestyle='--', linewidth=2,
               label=f'UCL = {UCL:.2f}')
    ax2.axvline(x=70, color='gray', linestyle='--', alpha=0.5,
               label='プロセス変化点')
    
    # アラームポイントを強調
    if alarm.any():
        ax2.scatter(np.where(alarm)[0] + 1, T2_values[alarm],
                   color='red', s=100, marker='o', zorder=5, label='アラーム')
    
    ax2.set_xlabel('サンプル番号', fontsize=11)
    ax2.set_ylabel('T²統計量', fontsize=11)
    ax2.set_title('Hotelling\'s T²管理図', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 温度の時系列
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(range(1, n_samples + 1), data[:, 0], 'o-', color='#f59e0b',
            markersize=4, linewidth=1, label='温度')
    ax3.axhline(y=mean_est[0], color='blue', linestyle='-', linewidth=2,
               label=f'平均 = {mean_est[0]:.1f}°C')
    ax3.axvline(x=70, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('サンプル番号', fontsize=11)
    ax3.set_ylabel('温度 (°C)', fontsize=11)
    ax3.set_title('温度の時系列', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 圧力の時系列
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(range(1, n_samples + 1), data[:, 1], 'o-', color='#7b2cbf',
            markersize=4, linewidth=1, label='圧力')
    ax4.axhline(y=mean_est[1], color='blue', linestyle='-', linewidth=2,
               label=f'平均 = {mean_est[1]:.1f}kPa')
    ax4.axvline(x=70, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('サンプル番号', fontsize=11)
    ax4.set_ylabel('圧力 (kPa)', fontsize=11)
    ax4.set_title('圧力の時系列', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Hotelling's T²管理図 分析結果 ===")
    print(f"\nプロセス設定:")
    print(f"  変数数 (p): {p}")
    print(f"  サンプル数: {n_samples}")
    print(f"  正常データ数: {m}")
    
    print(f"\nプロセス統計（正常データから推定）:")
    print(f"  平均ベクトル:")
    print(f"    温度: {mean_est[0]:.2f} °C")
    print(f"    圧力: {mean_est[1]:.2f} kPa")
    print(f"  共分散行列:")
    print(f"{cov_est}")
    print(f"  相関係数: {cov_est[0,1] / (np.sqrt(cov_est[0,0]) * np.sqrt(cov_est[1,1])):.3f}")
    
    print(f"\n管理限界:")
    print(f"  UCL (α={alpha}): {UCL:.2f}")
    
    print(f"\nT²統計量:")
    print(f"  正常データ平均T²: {np.mean(T2_values[:70]):.2f}")
    print(f"  異常データ平均T²: {np.mean(T2_values[70:]):.2f}")
    
    if alarm.any():
        first_alarm = np.where(alarm)[0][0]
        detection_delay = first_alarm - 70 if first_alarm >= 70 else 0
        print(f"\n✅ アラーム検出:")
        print(f"  最初のアラーム: サンプル {first_alarm + 1}")
        if first_alarm >= 70:
            print(f"  検出遅れ: {detection_delay} サンプル")
        print(f"  アラーム総数: {alarm.sum()}/{n_samples}")
    else:
        print("\n❌ アラーム: 検出なし")
    

**解説** : Hotelling's T²管理図は、複数の変数を同時に監視する多変量管理図です。変数間の相関関係を考慮しながら、多次元空間での異常を検出します。単変量管理図では見逃す可能性のある、複数変数の組み合わせによる異常を検出できます。プロセス産業では、温度、圧力、流量などの相関のある変数を同時に監視する場合に有用です。

#### コード例7: Western Electric ルール（異常パターン検出）の実装

**目的** : 管理図上の異常パターンを検出する8つのルールを実装する。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # プロセスデータ生成（様々な異常パターンを含む）
    n_samples = 100
    target_mean = 100
    sigma = 3
    
    data = np.random.normal(target_mean, sigma, n_samples)
    
    # 意図的な異常パターンを挿入
    # パターン1: 管理限界外（サンプル15）
    data[15] = target_mean + 3.5 * sigma
    
    # パターン2: 連続9点が中心線の同じ側（サンプル30-38）
    data[30:39] = np.random.normal(target_mean + 0.8 * sigma, sigma * 0.5, 9)
    
    # パターン3: 連続6点が増加傾向（サンプル50-55）
    data[50:56] = target_mean + np.linspace(-sigma, 2*sigma, 6)
    
    # パターン4: 連続14点が交互に上下（サンプル70-83）
    for i in range(70, 84):
        if i % 2 == 0:
            data[i] = target_mean + sigma * 0.8
        else:
            data[i] = target_mean - sigma * 0.8
    
    # 管理限界の計算
    UCL = target_mean + 3 * sigma
    LCL = target_mean - 3 * sigma
    UCL_2sigma = target_mean + 2 * sigma
    LCL_2sigma = target_mean - 2 * sigma
    UCL_1sigma = target_mean + 1 * sigma
    LCL_1sigma = target_mean - 1 * sigma
    
    # Western Electric ルールの実装
    def apply_western_electric_rules(data, mean, sigma):
        """
        Western Electric ルールを適用して異常パターンを検出
    
        8つのルール:
        1. 管理限界外の点（3σ超）
        2. 連続9点が中心線の同じ側
        3. 連続6点が増加または減少傾向
        4. 連続14点が交互に上下
        5. 連続3点のうち2点が2σ超（中心線の同じ側）
        6. 連続5点のうち4点が1σ超（中心線の同じ側）
        7. 連続15点が1σ以内（中心線の両側）
        8. 連続8点が1σ超（中心線の両側）
        """
        n = len(data)
        violations = {f'Rule{i}': [] for i in range(1, 9)}
    
        UCL = mean + 3 * sigma
        LCL = mean - 3 * sigma
    
        for i in range(n):
            # Rule 1: 管理限界外
            if data[i] > UCL or data[i] < LCL:
                violations['Rule1'].append(i)
    
            # Rule 2: 連続9点が中心線の同じ側
            if i >= 8:
                if all(data[i-j] > mean for j in range(9)) or \
                   all(data[i-j] < mean for j in range(9)):
                    violations['Rule2'].append(i)
    
            # Rule 3: 連続6点が増加または減少傾向
            if i >= 5:
                if all(data[i-j] < data[i-j-1] for j in range(5)) or \
                   all(data[i-j] > data[i-j-1] for j in range(5)):
                    violations['Rule3'].append(i)
    
            # Rule 4: 連続14点が交互に上下
            if i >= 13:
                alternating = True
                for j in range(13):
                    if (data[i-j] - data[i-j-1]) * (data[i-j-1] - data[i-j-2]) >= 0:
                        alternating = False
                        break
                if alternating:
                    violations['Rule4'].append(i)
    
            # Rule 5: 連続3点のうち2点が2σ超（同じ側）
            if i >= 2:
                above_2sigma = sum(1 for j in range(3) if data[i-j] > mean + 2*sigma)
                below_2sigma = sum(1 for j in range(3) if data[i-j] < mean - 2*sigma)
                if above_2sigma >= 2 or below_2sigma >= 2:
                    violations['Rule5'].append(i)
    
            # Rule 6: 連続5点のうち4点が1σ超（同じ側）
            if i >= 4:
                above_1sigma = sum(1 for j in range(5) if data[i-j] > mean + 1*sigma)
                below_1sigma = sum(1 for j in range(5) if data[i-j] < mean - 1*sigma)
                if above_1sigma >= 4 or below_1sigma >= 4:
                    violations['Rule6'].append(i)
    
            # Rule 7: 連続15点が1σ以内
            if i >= 14:
                if all(abs(data[i-j] - mean) < sigma for j in range(15)):
                    violations['Rule7'].append(i)
    
            # Rule 8: 連続8点が1σ超（両側）
            if i >= 7:
                if all(abs(data[i-j] - mean) > sigma for j in range(8)):
                    violations['Rule8'].append(i)
    
        return violations
    
    # ルールの適用
    violations = apply_western_electric_rules(data, target_mean, sigma)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # データプロット
    ax.plot(range(n_samples), data, 'o-', color='#11998e',
            markersize=4, linewidth=1, alpha=0.6, label='測定値')
    
    # 管理限界
    ax.axhline(y=target_mean, color='blue', linestyle='-', linewidth=2,
               label=f'中心線 = {target_mean}')
    ax.axhline(y=UCL, color='red', linestyle='--', linewidth=2, label='±3σ (UCL/LCL)')
    ax.axhline(y=LCL, color='red', linestyle='--', linewidth=2)
    ax.axhline(y=UCL_2sigma, color='orange', linestyle='--', linewidth=1,
               alpha=0.5, label='±2σ')
    ax.axhline(y=LCL_2sigma, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=UCL_1sigma, color='green', linestyle='--', linewidth=1,
               alpha=0.5, label='±1σ')
    ax.axhline(y=LCL_1sigma, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    # 異常ポイントを色分けして強調
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
    for idx, (rule, indices) in enumerate(violations.items()):
        if indices:
            ax.scatter(indices, data[indices], color=colors[idx], s=80,
                      marker='o', zorder=5, label=f'{rule} 検出')
    
    ax.set_xlabel('サンプル番号', fontsize=12)
    ax.set_ylabel('測定値', fontsize=12)
    ax.set_title('Western Electric ルールによる異常パターン検出',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 結果サマリー
    print("=== Western Electric ルール 検出結果 ===\n")
    total_violations = 0
    for rule, indices in violations.items():
        if indices:
            print(f"{rule}: {len(indices)}個の違反を検出")
            print(f"  位置: サンプル {[i+1 for i in indices]}")
            total_violations += len(indices)
        else:
            print(f"{rule}: 違反なし")
    
    print(f"\n合計違反数: {total_violations}")
    print(f"違反率: {total_violations/n_samples*100:.1f}%")
    
    # ルールの説明
    print("\n【Western Electric ルール説明】")
    print("Rule 1: 管理限界外の点（3σ超）→ 極端な異常")
    print("Rule 2: 連続9点が中心線の同じ側 → プロセス平均のシフト")
    print("Rule 3: 連続6点が増加/減少傾向 → トレンド")
    print("Rule 4: 連続14点が交互に上下 → システマティックな変動")
    print("Rule 5: 連続3点のうち2点が2σ超 → 中程度の異常")
    print("Rule 6: 連続5点のうち4点が1σ超 → 小さなシフト")
    print("Rule 7: 連続15点が1σ以内 → 変動不足（データ改ざんの可能性）")
    print("Rule 8: 連続8点が1σ超 → 変動過多または2つの分布の混在")
    

**解説** : Western Electric ルール（または Nelson ルール）は、管理限界内であっても異常なパターンを検出する8つの判定ルールです。単に管理限界を超えるかどうかだけでなく、連続性、トレンド、周期性などのパターンを検出することで、プロセス異常の早期発見が可能になります。これらのルールは、実際のプロセス監視システムで広く使用されています。

#### コード例8: SPC統合システム（アラーム管理システム）

**目的** : 複数のSPC手法を統合し、アラーム生成と優先順位付けを行う実践的なシステムを構築する。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    
    class SPCMonitoringSystem:
        """統合SPCモニタリングシステム"""
    
        def __init__(self, target_mean, target_std, spec_limits=None):
            """
            Parameters:
            -----------
            target_mean : float
                目標プロセス平均
            target_std : float
                目標プロセス標準偏差
            spec_limits : tuple
                規格限界 (LSL, USL)
            """
            self.target_mean = target_mean
            self.target_std = target_std
            self.spec_limits = spec_limits
    
            # 管理限界
            self.UCL = target_mean + 3 * target_std
            self.LCL = target_mean - 3 * target_std
    
            # アラームログ
            self.alarm_log = []
    
        def check_shewhart(self, value, sample_id):
            """シューハート管理図による検査"""
            if value > self.UCL:
                self.alarm_log.append({
                    'sample_id': sample_id,
                    'timestamp': datetime.now(),
                    'method': 'Shewhart',
                    'severity': 'HIGH',
                    'message': f'値 {value:.2f} が UCL {self.UCL:.2f} を超過'
                })
                return False
            elif value < self.LCL:
                self.alarm_log.append({
                    'sample_id': sample_id,
                    'timestamp': datetime.now(),
                    'method': 'Shewhart',
                    'severity': 'HIGH',
                    'message': f'値 {value:.2f} が LCL {self.LCL:.2f} を下回る'
                })
                return False
            return True
    
        def check_specification(self, value, sample_id):
            """規格限界チェック"""
            if self.spec_limits is None:
                return True
    
            LSL, USL = self.spec_limits
            if value > USL:
                self.alarm_log.append({
                    'sample_id': sample_id,
                    'timestamp': datetime.now(),
                    'method': 'Specification',
                    'severity': 'CRITICAL',
                    'message': f'規格外: 値 {value:.2f} が USL {USL} を超過'
                })
                return False
            elif value < LSL:
                self.alarm_log.append({
                    'sample_id': sample_id,
                    'timestamp': datetime.now(),
                    'method': 'Specification',
                    'severity': 'CRITICAL',
                    'message': f'規格外: 値 {value:.2f} が LSL {LSL} を下回る'
                })
                return False
            return True
    
        def check_trend(self, recent_data, sample_id, window=6):
            """トレンド検出（連続6点の増加/減少）"""
            if len(recent_data) < window:
                return True
    
            last_window = recent_data[-window:]
            increasing = all(last_window[i] < last_window[i+1]
                            for i in range(len(last_window)-1))
            decreasing = all(last_window[i] > last_window[i+1]
                            for i in range(len(last_window)-1))
    
            if increasing or decreasing:
                trend_type = '増加' if increasing else '減少'
                self.alarm_log.append({
                    'sample_id': sample_id,
                    'timestamp': datetime.now(),
                    'method': 'Trend',
                    'severity': 'MEDIUM',
                    'message': f'連続{window}点の{trend_type}トレンドを検出'
                })
                return False
            return True
    
        def monitor_process(self, data):
            """プロセス全体の監視"""
            results = []
            recent_data = []
    
            for i, value in enumerate(data):
                sample_id = i + 1
                recent_data.append(value)
    
                # 各チェックを実行
                shewhart_ok = self.check_shewhart(value, sample_id)
                spec_ok = self.check_specification(value, sample_id)
                trend_ok = self.check_trend(recent_data, sample_id)
    
                status = 'OK' if (shewhart_ok and spec_ok and trend_ok) else 'ALARM'
                results.append(status)
    
            return results
    
        def get_alarm_summary(self):
            """アラームサマリーを取得"""
            if not self.alarm_log:
                return "アラームなし"
    
            df_alarms = pd.DataFrame(self.alarm_log)
            summary = df_alarms.groupby(['severity', 'method']).size().reset_index(name='count')
            return summary
    
        def generate_report(self):
            """監視レポートを生成"""
            if not self.alarm_log:
                print("✅ プロセスは正常に運転されています。アラームはありません。")
                return
    
            print("=" * 70)
            print("SPC監視システム - アラームレポート")
            print("=" * 70)
    
            # 重要度別カウント
            df_alarms = pd.DataFrame(self.alarm_log)
            severity_counts = df_alarms['severity'].value_counts()
    
            print(f"\n【アラーム統計】")
            print(f"総アラーム数: {len(self.alarm_log)}")
            for severity, count in severity_counts.items():
                print(f"  {severity}: {count}件")
    
            # 手法別カウント
            print(f"\n【検出手法別】")
            method_counts = df_alarms['method'].value_counts()
            for method, count in method_counts.items():
                print(f"  {method}: {count}件")
    
            # 直近のアラーム（最大5件）
            print(f"\n【最新アラーム（最大5件）】")
            for alarm in self.alarm_log[-5:]:
                print(f"\n  サンプル {alarm['sample_id']} | "
                      f"{alarm['severity']} | {alarm['method']}")
                print(f"  → {alarm['message']}")
    
            print("\n" + "=" * 70)
    
    # システムのデモンストレーション
    print("=== SPC統合監視システム デモンストレーション ===\n")
    
    # プロセス設定
    target_mean = 50.0
    target_std = 2.0
    LSL, USL = 44, 56  # 規格限界（±3σ）
    
    # システム初期化
    spc_system = SPCMonitoringSystem(target_mean, target_std,
                                     spec_limits=(LSL, USL))
    
    # プロセスデータ生成（100サンプル）
    n_samples = 100
    data = np.random.normal(target_mean, target_std, n_samples)
    
    # 意図的な異常を挿入
    data[30] = 58  # 規格外（UCL超過）
    data[60:66] = target_mean + np.linspace(0, 4, 6)  # トレンド
    data[80] = 42  # 規格外（LCL未満）
    
    # プロセス監視実行
    print("プロセス監視を開始...\n")
    statuses = spc_system.monitor_process(data)
    
    # レポート生成
    spc_system.generate_report()
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # プロセストレンド
    alarm_samples = [i for i, s in enumerate(statuses) if s == 'ALARM']
    ok_samples = [i for i, s in enumerate(statuses) if s == 'OK']
    
    axes[0].plot(ok_samples, data[ok_samples], 'o', color='#11998e',
                markersize=5, label='正常', alpha=0.6)
    axes[0].plot(alarm_samples, data[alarm_samples], 'o', color='red',
                markersize=8, label='アラーム', zorder=5)
    
    # 管理限界と規格限界
    axes[0].axhline(y=target_mean, color='blue', linestyle='-', linewidth=2,
                   label='目標値')
    axes[0].axhline(y=spc_system.UCL, color='orange', linestyle='--', linewidth=2,
                   label='管理限界 (UCL/LCL)')
    axes[0].axhline(y=spc_system.LCL, color='orange', linestyle='--', linewidth=2)
    axes[0].axhline(y=USL, color='red', linestyle='--', linewidth=2,
                   label='規格限界 (USL/LSL)')
    axes[0].axhline(y=LSL, color='red', linestyle='--', linewidth=2)
    
    axes[0].set_xlabel('サンプル番号', fontsize=12)
    axes[0].set_ylabel('測定値', fontsize=12)
    axes[0].set_title('SPC統合監視 - プロセストレンド', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=0.3)
    
    # アラーム分布
    if spc_system.alarm_log:
        df_alarms = pd.DataFrame(spc_system.alarm_log)
    
        # 重要度別のアラーム数
        severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        severity_counts = df_alarms['severity'].value_counts()
        severity_counts = severity_counts.reindex(severity_order, fill_value=0)
    
        colors_severity = {'CRITICAL': 'red', 'HIGH': 'orange',
                          'MEDIUM': 'yellow', 'LOW': 'green'}
        bar_colors = [colors_severity[s] for s in severity_counts.index]
    
        bars = axes[1].bar(severity_counts.index, severity_counts.values,
                          color=bar_colors, alpha=0.7, edgecolor='black')
    
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
    
        axes[1].set_ylabel('アラーム件数', fontsize=12)
        axes[1].set_title('アラーム重要度別分布', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # プロセス能力評価
    process_mean = np.mean(data)
    process_std = np.std(data, ddof=1)
    Cp = (USL - LSL) / (6 * process_std)
    Cpk = min((USL - process_mean) / (3 * process_std),
              (process_mean - LSL) / (3 * process_std))
    
    print(f"\n【プロセス能力評価】")
    print(f"  Cp  = {Cp:.3f}")
    print(f"  Cpk = {Cpk:.3f}")
    if Cpk >= 1.33:
        print("  ✅ プロセス能力: 良好")
    elif Cpk >= 1.0:
        print("  ⚠️  プロセス能力: 最低限（改善推奨）")
    else:
        print("  ❌ プロセス能力: 不十分（早急な改善が必要）")
    

**解説** : この統合SPCシステムは、複数の検出手法（シューハート管理図、規格限界チェック、トレンド検出）を組み合わせ、アラームを重要度別に分類・管理します。実際のプロセス監視システムでは、このような統合的なアプローチにより、異常の早期検出と適切な対応が可能になります。アラームの優先順位付け（CRITICAL > HIGH > MEDIUM > LOW）により、オペレーターは重要な異常に集中できます。

* * *

## 2.3 本章のまとめ

### 学んだこと

  1. **SPCの基礎理論**
     * 偶然原因と特殊原因の違い
     * 管理図の基本構造（CL, UCL, LCL）
     * 3シグマ管理限界の統計的根拠
  2. **シューハート管理図**
     * X̄-R管理図：サブグループデータの管理
     * I-MR管理図：個別測定値の管理
     * 管理図の作成手順と解釈
  3. **プロセス能力評価**
     * Cp：プロセス変動の評価
     * Cpk：プロセス平均のずれを考慮した評価
     * Cpm：目標値からのずれを考慮した評価
     * 能力指数の判定基準（1.33, 1.67）
  4. **高度なSPC手法**
     * CUSUM：小さなシフトの早期検出
     * EWMA：時系列重み付けによる監視
     * Hotelling's T²：多変量プロセスの監視
  5. **異常パターン検出**
     * Western Electric ルール（8つの判定ルール）
     * トレンド、周期性、変動の異常検出
     * 統合的なアラーム管理システム

### 重要なポイント

  * **管理図の選択** : データの特性（サブグループ化の可否、変数の数）に応じて適切な管理図を選択
  * **プロセス能力** : CpとCpkの違いを理解し、プロセス平均のずれも考慮した評価が重要
  * **感度と誤報** : CUSUM/EWMAは小さなシフトに敏感だが、パラメータ設定により誤報率も変化
  * **多変量監視** : 相関のある変数は、Hotelling's T²で同時監視することで検出力向上
  * **統合的アプローチ** : 複数の手法を組み合わせ、重要度別にアラームを管理することが実践的

### 次の章へ

第3章では、**異常検知とプロセス監視** を学びます：

  * ルールベース異常検知（閾値ベース）
  * 統計的異常検知（Z-score, 修正Z-score）
  * 機械学習による異常検知（Isolation Forest, One-Class SVM）
  * 時系列異常検知（LSTM Autoencoder）
  * アラーム管理システムの設計と誤報削減
