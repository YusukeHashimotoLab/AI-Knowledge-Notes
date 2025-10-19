# 第2章: ロボティクス実験の基礎

**学習時間: 25-30分**

---

## 導入

実験自動化の心臓部は、ロボットアーム、液体ハンドリングシステム、センサーネットワークです。本章では、これらの基礎技術をPythonプログラミングを通じて実践的に学びます。

OpenTrons OT-2液体ハンドリングロボットを中心に、実際に動くコード例を通じて、試薬の分注、プレートの移動、センサーデータの取得など、自動化実験の基本操作を習得します。

---

## 学習目標

本章を学習することで、以下を習得できます：

1. **ロボットアーム制御**: 逆運動学、経路計画の基礎とPython実装
2. **液体ハンドリング**: OpenTrons OT-2での精密ピペッティング
3. **固体ハンドリング**: 粉末計量、錠剤成形の自動化手法
4. **センサー統合**: カメラ、分光計、XRDとのインターフェース
5. **安全設計**: エラーハンドリング、緊急停止、異常検知
6. **ラボウェア標準化**: マイクロプレート、バイアル、キュベットの統一規格

---

## 2.1 ロボットアーム制御の基礎

### 2.1.1 順運動学と逆運動学

ロボットアームの制御には2つの運動学問題があります。

**順運動学（Forward Kinematics）**:
関節角度 $\theta_1, \theta_2, ..., \theta_n$ からエンドエフェクタの位置・姿勢 $(x, y, z, roll, pitch, yaw)$ を計算

**逆運動学（Inverse Kinematics, IK）**:
目標位置・姿勢からそれを実現する関節角度を計算（実験では通常こちらを使用）

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimpleRobotArm:
    """
    2リンク2D平面ロボットアームのシミュレーション
    材料科学実験での試薬ボトルへのアクセス、サンプル移動などを想定
    """

    def __init__(self, link1_length=0.3, link2_length=0.25):
        """
        Args:
            link1_length: 第1リンクの長さ（メートル）
            link2_length: 第2リンクの長さ（メートル）
        """
        self.L1 = link1_length
        self.L2 = link2_length

    def forward_kinematics(self, theta1, theta2):
        """
        順運動学: 関節角度からエンドエフェクタ位置を計算

        Args:
            theta1: 第1関節の角度（度）
            theta2: 第2関節の角度（度）

        Returns:
            (x, y): エンドエフェクタの位置
        """
        # 度をラジアンに変換
        th1 = np.radians(theta1)
        th2 = np.radians(theta2)

        # 第1リンクの先端位置
        x1 = self.L1 * np.cos(th1)
        y1 = self.L1 * np.sin(th1)

        # エンドエフェクタの位置
        x = x1 + self.L2 * np.cos(th1 + th2)
        y = y1 + self.L2 * np.sin(th1 + th2)

        return x, y

    def inverse_kinematics(self, target_x, target_y):
        """
        逆運動学: 目標位置から関節角度を計算

        Args:
            target_x: 目標X座標
            target_y: 目標Y座標

        Returns:
            (theta1, theta2): 関節角度（度）または None（到達不能）
        """
        # 到達可能性チェック
        distance = np.sqrt(target_x**2 + target_y**2)
        if distance > (self.L1 + self.L2) or distance < abs(self.L1 - self.L2):
            print(f"警告: 目標位置 ({target_x:.2f}, {target_y:.2f}) は到達不能")
            return None

        # 余弦定理による第2関節角度の計算
        cos_theta2 = (target_x**2 + target_y**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        # 数値誤差対策
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

        # Elbow-upの解を選択（実験で一般的）
        theta2_rad = np.arccos(cos_theta2)

        # 第1関節角度の計算
        k1 = self.L1 + self.L2 * np.cos(theta2_rad)
        k2 = self.L2 * np.sin(theta2_rad)
        theta1_rad = np.arctan2(target_y, target_x) - np.arctan2(k2, k1)

        # ラジアンから度に変換
        theta1 = np.degrees(theta1_rad)
        theta2 = np.degrees(theta2_rad)

        return theta1, theta2

    def plot_arm(self, theta1, theta2, target_point=None):
        """アームの現在位置を可視化"""
        th1 = np.radians(theta1)
        th2 = np.radians(theta2)

        # 各関節の位置
        x0, y0 = 0, 0  # ベース
        x1 = self.L1 * np.cos(th1)
        y1 = self.L1 * np.sin(th1)
        x2 = x1 + self.L2 * np.cos(th1 + th2)
        y2 = y1 + self.L2 * np.sin(th1 + th2)

        plt.figure(figsize=(8, 8))
        plt.plot([x0, x1, x2], [y0, y1, y2], 'o-', linewidth=3, markersize=10, label='ロボットアーム')
        plt.plot(x0, y0, 'ro', markersize=15, label='ベース')
        plt.plot(x2, y2, 'go', markersize=12, label='エンドエフェクタ')

        if target_point:
            plt.plot(target_point[0], target_point[1], 'r*', markersize=20, label='目標位置')

        # 到達範囲の円
        theta = np.linspace(0, 2*np.pi, 100)
        r_max = self.L1 + self.L2
        r_min = abs(self.L1 - self.L2)
        plt.plot(r_max * np.cos(theta), r_max * np.sin(theta), 'k--', alpha=0.3, label='最大到達範囲')
        plt.plot(r_min * np.cos(theta), r_min * np.sin(theta), 'k--', alpha=0.3)

        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.xlabel('X 位置 (m)', fontsize=12)
        plt.ylabel('Y 位置 (m)', fontsize=12)
        plt.title('2リンクロボットアーム', fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()


# 使用例
robot = SimpleRobotArm(link1_length=0.3, link2_length=0.25)

# 逆運動学の使用例: 試薬ボトル (0.4, 0.2) にアクセス
target_x, target_y = 0.4, 0.2
angles = robot.inverse_kinematics(target_x, target_y)

if angles:
    theta1, theta2 = angles
    print(f"目標位置: ({target_x}, {target_y})")
    print(f"必要な関節角度: θ1 = {theta1:.2f}°, θ2 = {theta2:.2f}°")

    # 検証: 順運動学で位置を確認
    x_check, y_check = robot.forward_kinematics(theta1, theta2)
    error = np.sqrt((x_check - target_x)**2 + (y_check - target_y)**2)
    print(f"検証: 実際の位置 ({x_check:.4f}, {y_check:.4f}), 誤差 {error:.6f}m")

    # 可視化
    robot.plot_arm(theta1, theta2, target_point=(target_x, target_y))
    plt.savefig('robot_arm_ik.png', dpi=300, bbox_inches='tight')
    plt.show()
```

**コード解説**:
1. **順運動学**: 三角関数で各リンクの位置を計算
2. **逆運動学**: 余弦定理とアークタンジェントで関節角度を逆算
3. **到達可能性**: 目標距離が $|L_1 - L_2| \leq d \leq L_1 + L_2$ の範囲にあるかチェック
4. **Elbow-up/down**: 同じ目標位置に2つの解がある場合、実験では通常elbow-upを選択

---

### 2.1.2 軌道計画（Path Planning）

試薬ボトルから反応容器への移動など、スムーズで安全な軌道を計画します。

```python
def linear_trajectory(start_pos, end_pos, num_points=50):
    """
    2点間の直線軌道を生成

    Args:
        start_pos: 開始位置 (x, y)
        end_pos: 終了位置 (x, y)
        num_points: 軌道上の点数

    Returns:
        軌道上の点のリスト [(x1, y1), (x2, y2), ...]
    """
    x_traj = np.linspace(start_pos[0], end_pos[0], num_points)
    y_traj = np.linspace(start_pos[1], end_pos[1], num_points)

    trajectory = list(zip(x_traj, y_traj))
    return trajectory

def execute_trajectory(robot, trajectory, plot=True):
    """
    軌道を実行（シミュレーション）

    Args:
        robot: RobotArmインスタンス
        trajectory: 目標位置のリスト
        plot: 軌道を可視化するか
    """
    joint_angles = []
    successful_points = []

    for i, (x, y) in enumerate(trajectory):
        angles = robot.inverse_kinematics(x, y)
        if angles:
            joint_angles.append(angles)
            successful_points.append((x, y))
        else:
            print(f"警告: 点 {i} ({x:.3f}, {y:.3f}) は到達不能")

    if plot and successful_points:
        plt.figure(figsize=(10, 8))

        # 到達範囲
        theta = np.linspace(0, 2*np.pi, 100)
        r_max = robot.L1 + robot.L2
        plt.plot(r_max * np.cos(theta), r_max * np.sin(theta), 'k--', alpha=0.2, label='最大到達範囲')

        # 軌道
        traj_x, traj_y = zip(*successful_points)
        plt.plot(traj_x, traj_y, 'b-', linewidth=2, alpha=0.6, label='計画軌道')
        plt.plot(traj_x[0], traj_y[0], 'go', markersize=15, label='開始位置')
        plt.plot(traj_x[-1], traj_y[-1], 'ro', markersize=15, label='終了位置')

        # いくつかの中間姿勢を表示
        for i in range(0, len(joint_angles), len(joint_angles)//5):
            theta1, theta2 = joint_angles[i]
            th1 = np.radians(theta1)
            th2 = np.radians(theta2)

            x1 = robot.L1 * np.cos(th1)
            y1 = robot.L1 * np.sin(th1)
            x2 = x1 + robot.L2 * np.cos(th1 + th2)
            y2 = y1 + robot.L2 * np.sin(th1 + th2)

            plt.plot([0, x1, x2], [0, y1, y2], 'gray', alpha=0.3, linewidth=1)

        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.xlabel('X 位置 (m)', fontsize=12)
        plt.ylabel('Y 位置 (m)', fontsize=12)
        plt.title('ロボットアームの軌道計画', fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.savefig('trajectory_planning.png', dpi=300, bbox_inches='tight')
        plt.show()

    return joint_angles


# 使用例: 試薬ボトル (0.35, 0.15) から反応容器 (0.25, 0.35) への移動
robot = SimpleRobotArm()
start = (0.35, 0.15)
end = (0.25, 0.35)

trajectory = linear_trajectory(start, end, num_points=30)
print(f"軌道生成: {len(trajectory)}点")

joint_angles = execute_trajectory(robot, trajectory, plot=True)
print(f"実行成功: {len(joint_angles)}/{len(trajectory)}点")
```

---

## 2.2 液体ハンドリング: OpenTrons OT-2

### 2.2.1 OpenTrons OT-2の概要

OpenTrons OT-2は、研究室で広く使われるオープンソースの液体ハンドリングロボットです。

**主な仕様**:
- **精度**: ±1 µL（1-20 µL）、±2%（20-300 µL）
- **容量**: 1-1000 µL（ピペット交換で対応）
- **デッキサイズ**: 11スロット（マイクロプレート、チューブラック、試薬ボトル）
- **価格**: 約$10,000（学術割引あり）
- **プログラミング**: Python API（直感的で学習容易）

### 2.2.2 基本的なピペッティング

```python
from opentrons import protocol_api

# OT-2プロトコル: 96ウェルプレートへの試薬分注
metadata = {
    'protocolName': '基本的なピペッティング',
    'author': 'Materials Lab',
    'description': '試薬を96ウェルプレートに分注',
    'apiLevel': '2.13'
}

def run(protocol: protocol_api.ProtocolContext):
    """
    基本的なピペッティングプロトコル

    Args:
        protocol: OpenTrons ProtocolContext
    """
    # デッキレイアウトの設定
    # スロット1: 96ウェルプレート（反応用）
    plate = protocol.load_labware('corning_96_wellplate_360ul_flat', location='1')

    # スロット2: チューブラック（試薬ボトル）
    tuberack = protocol.load_labware('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', location='2')

    # スロット3: ピペットチップラック
    tiprack = protocol.load_labware('opentrons_96_tiprack_300ul', location='3')

    # ピペットの装着（P300 Single-Channel）
    pipette = protocol.load_instrument('p300_single_gen2', mount='left', tip_racks=[tiprack])

    # ピペッティング操作
    # 試薬A（チューブA1）を各ウェルに50 µL分注
    reagent_a = tuberack.wells_by_name()['A1']

    for well in plate.wells():
        pipette.pick_up_tip()  # 新しいチップを取る
        pipette.aspirate(50, reagent_a)  # 試薬を吸引（50 µL）
        pipette.dispense(50, well)  # ウェルに分注
        pipette.blow_out(well.top())  # 余剰液を排出
        pipette.drop_tip()  # チップを廃棄

    protocol.comment("プロトコル完了: 96ウェルすべてに試薬Aを分注しました")


# プロトコルのシミュレーション（実機なしで動作確認）
# ターミナルで実行: opentrons_simulate basic_pipetting.py
```

**コード解説**:
1. **Labware読み込み**: プレート、チューブラック、チップラックをデッキに配置
2. **Pipette装着**: シングルチャンネル300 µLピペット
3. **ピペッティングループ**: 各ウェルに試薬を分注
4. **クロスコンタミネーション防止**: 毎回新しいチップを使用

---

### 2.2.3 マルチチャンネルピペッティング

96ウェルプレート全体への高速分注には、8チャンネルピペットを使用します。

```python
def run(protocol: protocol_api.ProtocolContext):
    """
    マルチチャンネルピペッティング（8列同時分注）
    """
    # デッキレイアウト
    source_plate = protocol.load_labware('nest_12_reservoir_15ml', location='1')  # 試薬リザーバー
    dest_plate = protocol.load_labware('corning_96_wellplate_360ul_flat', location='2')
    tiprack = protocol.load_labware('opentrons_96_tiprack_300ul', location='3')

    # 8チャンネルピペット
    p300_multi = protocol.load_instrument('p300_multi_gen2', mount='left', tip_racks=[tiprack])

    # 試薬リザーバーから96ウェルプレートへ
    # 8チャンネルピペットは縦1列（8ウェル）を同時処理
    p300_multi.pick_up_tip()

    # 12列を順次処理（96ウェル = 12列 × 8行）
    for col in dest_plate.columns():
        p300_multi.aspirate(100, source_plate['A1'])  # リザーバーから吸引
        p300_multi.dispense(100, col[0])  # 列の最初のウェル（A1, A2, ..., A12）
        p300_multi.blow_out()

    p300_multi.drop_tip()
    protocol.comment("96ウェル全てに100 µL分注完了")


# 効率比較
print("96ウェルプレートへの分注時間:")
print("  シングルチャンネル: 96ウェル × 20秒 = 32分")
print("  マルチチャンネル: 12列 × 20秒 = 4分")
print("  効率化: 8倍高速")
```

---

### 2.2.4 段階希釈（Serial Dilution）

触媒スクリーニングなどで重要な段階希釈を自動化します。

```python
def run(protocol: protocol_api.ProtocolContext):
    """
    段階希釈プロトコル（10倍希釈系列）
    濃度: 10^0, 10^-1, 10^-2, ..., 10^-7 M
    """
    plate = protocol.load_labware('corning_96_wellplate_360ul_flat', location='1')
    tiprack = protocol.load_labware('opentrons_96_tiprack_300ul', location='2')
    reservoir = protocol.load_labware('nest_12_reservoir_15ml', location='3')

    pipette = protocol.load_instrument('p300_single_gen2', mount='left', tip_racks=[tiprack])

    # 希釈系列を作成（A列: 10^0 → 10^-7）
    # ウェルA1: 原液、A2-A8: 希釈液

    # 1. 各ウェルに溶媒を分注（A2-A8）
    solvent = reservoir['A1']
    for well in plate.columns()[0][1:8]:  # A2からA8
        pipette.pick_up_tip()
        pipette.transfer(180, solvent, well, new_tip='never')
        pipette.drop_tip()

    # 2. 原液を A1 に分注
    stock_solution = reservoir['A2']
    pipette.transfer(200, stock_solution, plate['A1'], new_tip='once')

    # 3. 段階希釈の実行
    # A1 → A2 → A3 → ... → A8
    pipette.pick_up_tip()
    for i in range(7):
        source_well = plate.columns()[0][i]  # A1, A2, ..., A7
        dest_well = plate.columns()[0][i+1]  # A2, A3, ..., A8

        # 20 µLを次のウェルに移す
        pipette.aspirate(20, source_well)
        pipette.dispense(20, dest_well)
        pipette.mix(3, 100, dest_well)  # 3回混合（100 µL）

    pipette.drop_tip()
    protocol.comment("段階希釈完了: 10^0 → 10^-7 M")


# 段階希釈の濃度計算
import pandas as pd

dilution_factor = 10  # 10倍希釈
num_dilutions = 8
initial_concentration = 1.0  # M

concentrations = [initial_concentration / (dilution_factor ** i) for i in range(num_dilutions)]
wells = [f'A{i+1}' for i in range(num_dilutions)]

df_dilution = pd.DataFrame({
    'ウェル': wells,
    '濃度 (M)': concentrations,
    '対数濃度': [f'10^{int(np.log10(c))}' if c >= 1e-10 else '0' for c in concentrations]
})

print("段階希釈系列:")
print(df_dilution.to_string(index=False))
```

---

## 2.3 固体ハンドリング

### 2.3.1 粉末計量の自動化

固体試薬の自動計量には、精密天秤とロボットアームの連携が必要です。

```python
class PowderDispenserSimulator:
    """
    粉末分注システムのシミュレーター
    実際の装置: Mettler Toledo Balance + Robot Arm
    """

    def __init__(self, accuracy=0.001):
        """
        Args:
            accuracy: 計量精度（グラム）
        """
        self.accuracy = accuracy
        self.dispensed_amounts = []

    def dispense_powder(self, target_mass, powder_name='試薬A'):
        """
        粉末を目標質量まで分注

        Args:
            target_mass: 目標質量（グラム）
            powder_name: 粉末の名前

        Returns:
            actual_mass: 実際に分注された質量
        """
        # シミュレーション: 正規分布でばらつきを模擬
        actual_mass = np.random.normal(target_mass, self.accuracy)
        self.dispensed_amounts.append(actual_mass)

        error = actual_mass - target_mass
        print(f"{powder_name} 分注: 目標 {target_mass:.3f}g, 実測 {actual_mass:.3f}g (誤差: {error:+.4f}g)")

        return actual_mass

    def multi_component_dispensing(self, composition_dict):
        """
        多成分粉末の自動配合

        Args:
            composition_dict: {成分名: 質量(g)}

        Returns:
            actual_composition: 実際の組成
        """
        print("多成分配合開始:")
        actual_composition = {}

        for component, target_mass in composition_dict.items():
            actual_mass = self.dispense_powder(target_mass, component)
            actual_composition[component] = actual_mass

        total_mass = sum(actual_composition.values())
        print(f"\n合計質量: {total_mass:.3f}g")

        # 組成比（重量%）
        print("\n実際の組成比（wt%）:")
        for component, mass in actual_composition.items():
            percentage = (mass / total_mass) * 100
            print(f"  {component}: {percentage:.2f}%")

        return actual_composition


# 使用例: 三元系触媒 (NiO, CoO, MnO2) の配合
dispenser = PowderDispenserSimulator(accuracy=0.002)

# 目標組成: Ni:Co:Mn = 60:20:20 (wt%)
total_mass = 1.0  # 合計1.0g
composition = {
    'NiO': 0.6,
    'CoO': 0.2,
    'MnO2': 0.2
}

actual_comp = dispenser.multi_component_dispensing(composition)

# 精度評価
print("\n精度評価:")
for component, target_mass in composition.items():
    actual_mass = actual_comp[component]
    error_percent = abs((actual_mass - target_mass) / target_mass) * 100
    print(f"  {component}: 誤差 {error_percent:.2f}%")
```

---

### 2.3.2 固体試料の移送

```python
def solid_sample_transfer_protocol():
    """
    固体試料の自動移送プロトコル（疑似コード）
    実際の実装はロボットアームのAPIに依存
    """
    protocol_steps = [
        "1. ロボットアームがサンプルホルダーを把持",
        "2. XRD測定位置に移動",
        "3. サンプルをXRDステージに配置",
        "4. XRD測定開始（外部トリガー）",
        "5. 測定完了を待機",
        "6. サンプルを回収",
        "7. 次のサンプル位置に移動",
        "8. ステップ1-7を繰り返し"
    ]

    for step in protocol_steps:
        print(step)

    # 疑似コード: 実際のロボット制御
    """
    # Universal Robots UR5eの例
    import urx

    robot = urx.Robot("192.168.1.100")  # ロボットのIPアドレス

    # サンプル位置（XYZ座標、ミリメートル）
    sample_position = [300, 200, 100, 0, 0, 0]  # X, Y, Z, RX, RY, RZ
    xrd_position = [500, 200, 150, 0, 0, 0]

    # 移動
    robot.movel(sample_position, acc=0.1, vel=0.1)  # 線形移動
    # グリッパーでサンプル把持
    robot.set_digital_out(0, True)  # デジタル出力でグリッパー制御

    robot.movel(xrd_position, acc=0.1, vel=0.1)
    # サンプルを配置
    robot.set_digital_out(0, False)

    robot.close()
    """

solid_sample_transfer_protocol()
```

---

## 2.4 センサー統合

### 2.4.1 分光計との連携

UV-Vis分光計でリアルタイムに反応をモニタリングします。

```python
import time

class SpectrometerSimulator:
    """
    UV-Vis分光計シミュレーター
    実際の装置: Ocean Optics USB4000, Agilent Cary 60
    """

    def __init__(self, wavelength_range=(200, 800), resolution=1):
        """
        Args:
            wavelength_range: 波長範囲（nm）
            resolution: 波長分解能（nm）
        """
        self.wavelengths = np.arange(wavelength_range[0], wavelength_range[1], resolution)

    def measure_absorbance(self, sample_id, concentration=0.1):
        """
        吸光度測定（シミュレーション）

        Args:
            sample_id: サンプルID
            concentration: 濃度（M）

        Returns:
            wavelengths, absorbance: 波長と吸光度の配列
        """
        # Beer-Lambert則: A = ε * c * l
        # ピーク波長: 450 nm（仮想的な化合物）
        peak_wavelength = 450
        peak_absorbance = concentration * 10  # εcl = 10 (仮定)

        # ガウス型吸収スペクトル
        absorbance = peak_absorbance * np.exp(-((self.wavelengths - peak_wavelength) / 50)**2)

        # ノイズ追加
        noise = np.random.normal(0, 0.01, len(self.wavelengths))
        absorbance += noise

        print(f"サンプル {sample_id} 測定完了: ピーク波長 {peak_wavelength}nm, 吸光度 {peak_absorbance:.3f}")

        return self.wavelengths, absorbance

    def kinetic_measurement(self, duration=60, interval=5):
        """
        反応速度論測定（時間変化を追跡）

        Args:
            duration: 測定時間（秒）
            interval: 測定間隔（秒）

        Returns:
            times, absorbance_at_450nm: 時間と450nmでの吸光度
        """
        times = np.arange(0, duration, interval)
        absorbance_450 = []

        print("反応速度論測定開始...")
        for t in times:
            # 一次反応のシミュレーション: [A] = [A]0 * exp(-kt)
            k = 0.02  # s^-1
            concentration = 0.1 * np.exp(-k * t)

            _, abs_spectrum = self.measure_absorbance(f't={t}s', concentration)
            # 450nmでの吸光度を抽出
            idx_450 = np.argmin(np.abs(self.wavelengths - 450))
            absorbance_450.append(abs_spectrum[idx_450])

            time.sleep(0.1)  # 実際の測定では実時間で待機

        return times, np.array(absorbance_450)


# 使用例
spectrometer = SpectrometerSimulator()

# スペクトル測定
wavelengths, absorbance = spectrometer.measure_absorbance('Sample_001', concentration=0.15)

plt.figure(figsize=(10, 6))
plt.plot(wavelengths, absorbance, linewidth=2)
plt.xlabel('波長 (nm)', fontsize=12)
plt.ylabel('吸光度', fontsize=12)
plt.title('UV-Vis吸収スペクトル', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('uv_vis_spectrum.png', dpi=300, bbox_inches='tight')
plt.show()

# 反応速度論測定
times, abs_450 = spectrometer.kinetic_measurement(duration=100, interval=10)

plt.figure(figsize=(10, 6))
plt.plot(times, abs_450, 'o-', linewidth=2, markersize=8)
plt.xlabel('時間 (s)', fontsize=12)
plt.ylabel('吸光度 (450 nm)', fontsize=12)
plt.title('反応速度論測定', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('kinetic_measurement.png', dpi=300, bbox_inches='tight')
plt.show()

# 一次反応速度定数のフィッティング
from scipy.optimize import curve_fit

def first_order_kinetics(t, A0, k):
    return A0 * np.exp(-k * t)

params, covariance = curve_fit(first_order_kinetics, times, abs_450, p0=[1.5, 0.02])
A0_fit, k_fit = params

print(f"\n一次反応フィッティング:")
print(f"  初期吸光度 A0 = {A0_fit:.3f}")
print(f"  速度定数 k = {k_fit:.4f} s^-1")
print(f"  半減期 t1/2 = {np.log(2)/k_fit:.1f} s")
```

---

### 2.4.2 カメラによる画像解析

結晶成長、沈殿形成、色変化などをカメラで自動記録・解析します。

```python
from PIL import Image, ImageDraw, ImageFont
import cv2

class LabCameraSimulator:
    """
    実験用カメラシミュレーター
    実際の装置: Basler ace, FLIR Blackfly
    """

    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution

    def capture_wellplate(self, plate_id='Plate_001'):
        """
        96ウェルプレートの画像取得（シミュレーション）

        Returns:
            image: PIL Image
        """
        # シミュレーション: 96ウェルプレートのグリッド画像を生成
        img = Image.new('RGB', self.resolution, color='white')
        draw = ImageDraw.Draw(img)

        # 8行 × 12列のウェル
        well_diameter = 50
        spacing = 70
        offset_x, offset_y = 200, 100

        for row in range(8):
            for col in range(12):
                center_x = offset_x + col * spacing
                center_y = offset_y + row * spacing

                # ウェルの色（濃度に応じて変化をシミュレート）
                intensity = int(255 * (1 - (row + col) / 20))  # 徐々に濃くなる
                color = (intensity, intensity, 255)

                # ウェルを描画
                draw.ellipse([center_x - well_diameter//2, center_y - well_diameter//2,
                              center_x + well_diameter//2, center_y + well_diameter//2],
                             fill=color, outline='black')

        print(f"プレート {plate_id} の画像取得完了")
        return img

    def analyze_well_color(self, image, well_position):
        """
        特定ウェルの色を解析

        Args:
            image: PIL Image
            well_position: (row, col) ウェル位置

        Returns:
            rgb_mean: RGB平均値
        """
        row, col = well_position
        well_diameter = 50
        spacing = 70
        offset_x, offset_y = 200, 100

        center_x = offset_x + col * spacing
        center_y = offset_y + row * spacing

        # ウェル領域を切り出し
        crop_box = (center_x - well_diameter//2, center_y - well_diameter//2,
                    center_x + well_diameter//2, center_y + well_diameter//2)
        well_img = image.crop(crop_box)

        # RGB平均値を計算
        well_array = np.array(well_img)
        rgb_mean = well_array.mean(axis=(0, 1))

        return rgb_mean


# 使用例
camera = LabCameraSimulator()
plate_image = camera.capture_wellplate('Plate_001')
plate_image.save('wellplate_image.png')

# ウェルA1の色解析
rgb = camera.analyze_well_color(plate_image, well_position=(0, 0))
print(f"ウェルA1のRGB値: R={rgb[0]:.1f}, G={rgb[1]:.1f}, B={rgb[2]:.1f}")

# 全ウェルの色解析
print("\n全ウェルの青色成分（B値）マップ:")
blue_values = np.zeros((8, 12))
for row in range(8):
    for col in range(12):
        rgb = camera.analyze_well_color(plate_image, (row, col))
        blue_values[row, col] = rgb[2]

plt.figure(figsize=(12, 6))
plt.imshow(blue_values, cmap='Blues', interpolation='nearest')
plt.colorbar(label='青色成分 (B値)')
plt.xlabel('列', fontsize=12)
plt.ylabel('行', fontsize=12)
plt.title('96ウェルプレートの色分布', fontsize=14, fontweight='bold')
plt.xticks(range(12), [f'{i+1}' for i in range(12)])
plt.yticks(range(8), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
plt.tight_layout()
plt.savefig('wellplate_color_map.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 2.5 安全性とエラーハンドリング

### 2.5.1 緊急停止機能

```python
class SafetyController:
    """
    実験自動化システムの安全制御
    """

    def __init__(self):
        self.emergency_stop = False
        self.error_log = []

    def check_safety(self, temperature, pressure, liquid_level):
        """
        安全パラメータのチェック

        Args:
            temperature: 温度（℃）
            pressure: 圧力（bar）
            liquid_level: 液面レベル（%）

        Returns:
            is_safe: 安全かどうか
        """
        is_safe = True
        warnings = []

        # 温度チェック
        if temperature > 100:
            warnings.append(f"警告: 温度が高すぎます ({temperature}℃)")
            is_safe = False

        # 圧力チェック
        if pressure > 5:
            warnings.append(f"警告: 圧力が高すぎます ({pressure} bar)")
            is_safe = False

        # 液面レベルチェック
        if liquid_level < 10:
            warnings.append(f"警告: 液面が低すぎます ({liquid_level}%)")
            is_safe = False

        if not is_safe:
            for warning in warnings:
                print(warning)
                self.error_log.append({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'message': warning})

        return is_safe

    def emergency_stop_sequence(self):
        """緊急停止シーケンス"""
        print("\n🚨 緊急停止を実行します 🚨")
        self.emergency_stop = True

        # 停止アクション
        actions = [
            "1. すべてのロボット動作を停止",
            "2. ヒーターを停止",
            "3. 冷却水を循環",
            "4. バルブを安全位置に移動",
            "5. オペレーターに通知",
            "6. ログを記録"
        ]

        for action in actions:
            print(action)
            time.sleep(0.5)

        print("\n緊急停止完了。システムは安全状態です。")

    def print_error_log(self):
        """エラーログを表示"""
        print("\n=== エラーログ ===")
        for error in self.error_log:
            print(f"[{error['time']}] {error['message']}")


# 使用例
safety = SafetyController()

# 正常な状態
print("=== 正常運転 ===")
is_safe = safety.check_safety(temperature=80, pressure=2.0, liquid_level=50)
print(f"安全状態: {is_safe}\n")

# 異常な状態
print("=== 異常検知 ===")
is_safe = safety.check_safety(temperature=120, pressure=6.5, liquid_level=5)
print(f"安全状態: {is_safe}")

if not is_safe:
    safety.emergency_stop_sequence()

safety.print_error_log()
```

---

### 2.5.2 エラーリカバリー

```python
def robust_pipetting_with_retry(pipette, source, dest, volume, max_retries=3):
    """
    エラーリカバリー付きピペッティング

    Args:
        pipette: OT-2 pipette
        source: 吸引元
        dest: 分注先
        volume: 容量（µL）
        max_retries: 最大リトライ回数

    Returns:
        success: 成功したかどうか
    """
    for attempt in range(max_retries):
        try:
            pipette.pick_up_tip()
            pipette.aspirate(volume, source)
            pipette.dispense(volume, dest)
            pipette.drop_tip()

            print(f"ピペッティング成功: {volume} µL ({attempt+1}回目)")
            return True

        except Exception as e:
            print(f"エラー発生 ({attempt+1}回目): {e}")

            # リカバリーアクション
            if pipette.has_tip:
                pipette.drop_tip()

            if attempt < max_retries - 1:
                print("リトライします...")
                time.sleep(1)
            else:
                print("最大リトライ回数に達しました。操作を中止します。")
                return False

    return False


# 疑似コードでの使用例
print("エラーリカバリー付きピペッティングのデモ:\n")

class MockPipette:
    def __init__(self, fail_probability=0.3):
        self.has_tip = False
        self.fail_probability = fail_probability

    def pick_up_tip(self):
        if np.random.random() < self.fail_probability:
            raise Exception("チップ取得失敗")
        self.has_tip = True

    def aspirate(self, volume, source):
        if np.random.random() < self.fail_probability:
            raise Exception("吸引失敗")

    def dispense(self, volume, dest):
        if np.random.random() < self.fail_probability:
            raise Exception("分注失敗")

    def drop_tip(self):
        self.has_tip = False

mock_pipette = MockPipette(fail_probability=0.4)  # 40%の確率で失敗
success = robust_pipetting_with_retry(mock_pipette, 'A1', 'B1', 100, max_retries=5)
print(f"\n最終結果: {'成功' if success else '失敗'}")
```

---

## 2.6 ラボウェアの標準化

### 2.6.1 SBS（Society for Biomolecular Screening）規格

96ウェルプレート、384ウェルプレートなど、ラボウェアの寸法は国際規格で標準化されています。

```python
class SBS_Labware:
    """
    SBS規格のラボウェア仕様
    """

    @staticmethod
    def plate_96_well():
        """96ウェルプレートの仕様"""
        specs = {
            '名称': '96ウェルプレート',
            'フットプリント': '127.76 mm × 85.48 mm (SBS規格)',
            'ウェル配置': '8行 × 12列',
            'ウェル間隔': '9.0 mm (中心間)',
            'ウェル容量': '通常 300-360 µL',
            'ウェル形状': 'フラット、U字型、V字型',
            '用途': '一般的なスクリーニング、アッセイ'
        }
        return specs

    @staticmethod
    def plate_384_well():
        """384ウェルプレートの仕様"""
        specs = {
            '名称': '384ウェルプレート',
            'フットプリント': '127.76 mm × 85.48 mm (96ウェルと同じ)',
            'ウェル配置': '16行 × 24列',
            'ウェル間隔': '4.5 mm (中心間、96ウェルの半分)',
            'ウェル容量': '通常 50-100 µL',
            '用途': '高密度スクリーニング、創薬'
        }
        return specs

    @staticmethod
    def plot_wellplate_layout(n_wells=96):
        """ウェルプレートのレイアウトを可視化"""
        if n_wells == 96:
            rows, cols = 8, 12
            well_spacing = 9.0  # mm
        elif n_wells == 384:
            rows, cols = 16, 24
            well_spacing = 4.5  # mm
        else:
            raise ValueError("96または384ウェルのみサポート")

        fig, ax = plt.subplots(figsize=(12, 6))

        # ウェルを描画
        for row in range(rows):
            for col in range(cols):
                x = col * well_spacing
                y = row * well_spacing
                circle = plt.Circle((x, y), radius=well_spacing*0.4, color='lightblue', ec='black')
                ax.add_patch(circle)

                # ウェル名を表示（96ウェルのみ、見やすさのため）
                if n_wells == 96:
                    well_name = f"{chr(65+row)}{col+1}"
                    ax.text(x, y, well_name, ha='center', va='center', fontsize=8)

        ax.set_xlim(-well_spacing, cols * well_spacing)
        ax.set_ylim(-well_spacing, rows * well_spacing)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Y軸を反転（A行が上）
        ax.set_xlabel('X 方向 (mm)', fontsize=12)
        ax.set_ylabel('Y 方向 (mm)', fontsize=12)
        ax.set_title(f'{n_wells}ウェルプレート レイアウト (SBS規格)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{n_wells}_wellplate_layout.png', dpi=300, bbox_inches='tight')
        plt.show()


# 仕様の表示
labware = SBS_Labware()

print("=== 96ウェルプレート ===")
for key, value in labware.plate_96_well().items():
    print(f"{key}: {value}")

print("\n=== 384ウェルプレート ===")
for key, value in labware.plate_384_well().items():
    print(f"{key}: {value}")

# レイアウトの可視化
labware.plot_wellplate_layout(n_wells=96)
```

---

## 2.7 演習問題

### 演習1: ロボットアームの軌道最適化（難易度: Medium）

2つの試薬ボトル（位置A: (0.35, 0.15)、位置B: (0.25, 0.35)）と反応容器（位置C: (0.4, 0.3)）がある場合、最短時間で A → B → C の順に訪問する軌道を計画してください。

<details>
<summary>ヒント</summary>

直線軌道の移動時間は距離に比例します。A→B→C と A→C→B の2つのルートを比較し、総移動距離が短い方を選択してください。

</details>

<details>
<summary>解答例</summary>

```python
def calculate_path_length(points):
    """経路の総距離を計算"""
    total_distance = 0
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        distance = np.sqrt(dx**2 + dy**2)
        total_distance += distance
    return total_distance

# 試薬ボトルと反応容器の位置
A = (0.35, 0.15)  # 試薬A
B = (0.25, 0.35)  # 試薬B
C = (0.4, 0.3)    # 反応容器

# ルート1: A → B → C
route1 = [A, B, C]
distance1 = calculate_path_length(route1)

# ルート2: A → C → B
route2 = [A, C, B]
distance2 = calculate_path_length(route2)

print("ルート比較:")
print(f"  A → B → C: {distance1:.3f} m")
print(f"  A → C → B: {distance2:.3f} m")

if distance1 < distance2:
    print(f"\n最適ルート: A → B → C（{distance1:.3f} m）")
    optimal_route = route1
else:
    print(f"\n最適ルート: A → C → B（{distance2:.3f} m）")
    optimal_route = route2

# 可視化
robot = SimpleRobotArm()
fig, ax = plt.subplots(figsize=(10, 8))

# 到達範囲
theta = np.linspace(0, 2*np.pi, 100)
r_max = robot.L1 + robot.L2
ax.plot(r_max * np.cos(theta), r_max * np.sin(theta), 'k--', alpha=0.2, label='最大到達範囲')

# 位置をプロット
for point, label in zip([A, B, C], ['試薬A', '試薬B', '反応容器']):
    ax.plot(point[0], point[1], 'o', markersize=15, label=label)

# 最適ルートを描画
route_x, route_y = zip(*optimal_route)
ax.plot(route_x, route_y, 'r-', linewidth=2, alpha=0.6, label='最適経路')

ax.axis('equal')
ax.grid(alpha=0.3)
ax.set_xlabel('X 位置 (m)', fontsize=12)
ax.set_ylabel('Y 位置 (m)', fontsize=12)
ax.set_title('経路最適化', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('path_optimization.png', dpi=300, bbox_inches='tight')
plt.show()
```

</details>

---

### 演習2: 96ウェルプレートへの複数試薬分注（難易度: Medium）

96ウェルプレートに3種類の試薬（A、B、C）を以下のパターンで分注するOpenTronsプロトコルを作成してください。

- 試薬A: 列1-4（ウェルA1-H4）に50 µL
- 試薬B: 列5-8（ウェルA5-H8）に50 µL
- 試薬C: 列9-12（ウェルA9-H12）に50 µL

<details>
<summary>ヒント</summary>

8チャンネルピペットを使用すれば、各列を1回のピペッティングで処理できます。`plate.columns()[0:4]`で列1-4を選択できます。

</details>

<details>
<summary>解答例</summary>

```python
from opentrons import protocol_api

metadata = {
    'protocolName': '96ウェル 3試薬分注',
    'author': 'Materials Lab',
    'description': '3種類の試薬を列ごとに分注',
    'apiLevel': '2.13'
}

def run(protocol: protocol_api.ProtocolContext):
    # デッキレイアウト
    plate = protocol.load_labware('corning_96_wellplate_360ul_flat', location='1')
    reservoir = protocol.load_labware('nest_12_reservoir_15ml', location='2')
    tiprack = protocol.load_labware('opentrons_96_tiprack_300ul', location='3')

    # 8チャンネルピペット
    pipette = protocol.load_instrument('p300_multi_gen2', mount='left', tip_racks=[tiprack])

    # 試薬の位置
    reagent_a = reservoir['A1']
    reagent_b = reservoir['A2']
    reagent_c = reservoir['A3']

    # 試薬Aを列1-4に分注
    pipette.pick_up_tip()
    for col in plate.columns()[0:4]:  # 列1-4
        pipette.aspirate(50, reagent_a)
        pipette.dispense(50, col[0])  # 列の先頭ウェル（A1, A2, A3, A4）
        pipette.blow_out()
    pipette.drop_tip()

    protocol.comment("試薬A 分注完了")

    # 試薬Bを列5-8に分注
    pipette.pick_up_tip()
    for col in plate.columns()[4:8]:  # 列5-8
        pipette.aspirate(50, reagent_b)
        pipette.dispense(50, col[0])
        pipette.blow_out()
    pipette.drop_tip()

    protocol.comment("試薬B 分注完了")

    # 試薬Cを列9-12に分注
    pipette.pick_up_tip()
    for col in plate.columns()[8:12]:  # 列9-12
        pipette.aspirate(50, reagent_c)
        pipette.dispense(50, col[0])
        pipette.blow_out()
    pipette.drop_tip()

    protocol.comment("試薬C 分注完了")
    protocol.comment("プロトコル完了: 96ウェル全てに試薬を分注しました")

# シミュレーション実行
print("プロトコル作成完了")
print("実行コマンド: opentrons_simulate three_reagent_protocol.py")
```

</details>

---

### 演習3: センサーデータの異常検知（難易度: Hard）

UV-Vis分光計で反応をモニタリング中、異常な吸光度変化（急激な増加または減少）を検知し、緊急停止するシステムを実装してください。

**条件**:
- 測定間隔: 10秒
- 異常判定: 前回測定値から20%以上変化した場合
- 異常が2回連続した場合に緊急停止

<details>
<summary>ヒント</summary>

各測定で前回値と比較し、変化率を計算します。異常カウンターを導入し、2回連続で異常が検知されたら緊急停止を実行します。

</details>

<details>
<summary>解答例</summary>

```python
class ReactionMonitor:
    """
    反応モニタリングと異常検知
    """

    def __init__(self, anomaly_threshold=0.2, consecutive_anomalies=2):
        """
        Args:
            anomaly_threshold: 異常判定の閾値（変化率）
            consecutive_anomalies: 緊急停止までの連続異常回数
        """
        self.threshold = anomaly_threshold
        self.consecutive_threshold = consecutive_anomalies
        self.anomaly_count = 0
        self.previous_value = None
        self.measurements = []

    def check_anomaly(self, current_value):
        """
        異常検知

        Args:
            current_value: 現在の測定値

        Returns:
            is_anomaly: 異常かどうか
        """
        if self.previous_value is None:
            self.previous_value = current_value
            return False

        # 変化率を計算
        change_rate = abs((current_value - self.previous_value) / self.previous_value)

        is_anomaly = change_rate > self.threshold

        if is_anomaly:
            self.anomaly_count += 1
            print(f"⚠️ 異常検知: 変化率 {change_rate*100:.1f}% (閾値: {self.threshold*100:.1f}%)")
            print(f"   前回値: {self.previous_value:.3f}, 現在値: {current_value:.3f}")
            print(f"   連続異常回数: {self.anomaly_count}/{self.consecutive_threshold}")
        else:
            self.anomaly_count = 0  # 正常なら異常カウンターをリセット

        self.previous_value = current_value
        return is_anomaly

    def should_emergency_stop(self):
        """緊急停止が必要か判定"""
        return self.anomaly_count >= self.consecutive_threshold


# シミュレーション
print("=== 反応モニタリング開始 ===\n")
monitor = ReactionMonitor(anomaly_threshold=0.2, consecutive_anomalies=2)
safety = SafetyController()

# 測定データ（シミュレーション）
# 正常 → 正常 → 異常1 → 異常2（緊急停止）
simulated_absorbance = [1.0, 1.05, 1.08, 1.40, 1.75]

for i, absorbance in enumerate(simulated_absorbance):
    print(f"--- 測定 {i+1} ---")
    print(f"吸光度: {absorbance:.3f}")

    monitor.measurements.append(absorbance)
    is_anomaly = monitor.check_anomaly(absorbance)

    if monitor.should_emergency_stop():
        print("\n🚨 連続異常検知！ 緊急停止を実行します。")
        safety.emergency_stop_sequence()
        break

    print()

# 測定データのプロット
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(monitor.measurements) + 1), monitor.measurements, 'o-', linewidth=2, markersize=10)
plt.axhline(y=monitor.measurements[0] * (1 + monitor.threshold), color='red', linestyle='--', label='上限閾値')
plt.axhline(y=monitor.measurements[0] * (1 - monitor.threshold), color='red', linestyle='--', label='下限閾値')
plt.xlabel('測定回数', fontsize=12)
plt.ylabel('吸光度', fontsize=12)
plt.title('反応モニタリングと異常検知', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight')
plt.show()
```

</details>

---

## 本章のまとめ

本章では、ロボティクス実験の基礎技術を実践的に学びました。

### キーポイント

1. **ロボットアーム制御**:
   - 順運動学: 関節角度→位置
   - 逆運動学: 目標位置→関節角度（実験で使用）
   - 軌道計画: スムーズで安全な経路生成

2. **液体ハンドリング（OpenTrons OT-2）**:
   - 基本ピペッティング: 精度±1 µL
   - マルチチャンネル: 8倍高速化
   - 段階希釈: 10^-7 Mまでの自動希釈系列

3. **固体ハンドリング**:
   - 粉末計量: ±0.001 gの精度
   - 多成分配合: 自動化による再現性向上

4. **センサー統合**:
   - UV-Vis分光計: リアルタイム反応モニタリング
   - カメラ: 画像解析による定量評価

5. **安全設計**:
   - 緊急停止: 異常検知時の自動停止
   - エラーリカバリー: リトライ機能
   - ログ記録: トレーサビリティ確保

6. **ラボウェア標準化**:
   - SBS規格: 96/384ウェルプレートの国際標準
   - 互換性: 異なるメーカー間でも使用可能

### 次章予告

第3章では、ベイズ最適化とロボット実験を統合したクローズドループ最適化を学びます。実験→測定→解析→予測→次実験の自動サイクルを実装し、材料探索を劇的に加速します。

---

## 参考文献

1. OpenTrons. "OT-2 Robot Documentation." https://docs.opentrons.com/
2. Lynch, K. M., & Park, F. C. (2017). *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press.
3. Granda, J. M. et al. (2018). "Controlling an organic synthesis robot with machine learning to search for new reactivity." *Nature*, 559, 377-381.
4. SBS (Society for Laboratory Automation and Screening). "ANSI/SLAS Microplate Standards." https://www.slas.org/education/ansi-slas-microplate-standards/

---

**次の章へ**: [第3章: クローズドループ最適化](chapter-3.html)

[目次に戻る](index.html)
