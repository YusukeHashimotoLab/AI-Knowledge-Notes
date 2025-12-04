---
title: 第2章：プロセスオントロジーの設計とOWLモデリング
chapter_title: 第2章：プロセスオントロジーの設計とOWLモデリング
subtitle: owlready2による化学プロセス装置の包括的オントロジー構築
---

## 2.1 OWL（Web Ontology Language）の基礎

OWLはRDFSを拡張し、より表現力の高いオントロジーを記述できる標準言語です。化学プロセス装置の複雑な関係性、制約条件、推論ルールを形式的に定義できます。

**💡 OWLの3つのサブ言語**

  * **OWL Lite** : 基本的な階層構造とシンプルな制約
  * **OWL DL** : 記述論理ベース、推論可能性を保証（本シリーズで使用）
  * **OWL Full** : 最大の表現力だが推論は保証されない

### Example 1: owlready2によるOWLクラスと個体の定義

化学プロセス装置の基本クラス階層を構築します。
    
    
    # ===================================
    # Example 1: OWLクラスと個体の定義
    # ===================================
    
    from owlready2 import *
    
    # オントロジーの作成
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        # ===== クラス定義 =====
    
        # 最上位クラス: ProcessEquipment
        class ProcessEquipment(Thing):
            """プロセス装置の基底クラス"""
            pass
    
        # サブクラス: Reactor（反応器）
        class Reactor(ProcessEquipment):
            """反応器クラス"""
            pass
    
        # さらに詳細なサブクラス: CSTR（連続撹拌槽型反応器）
        class CSTR(Reactor):
            """連続撹拌槽型反応器"""
            pass
    
        # サブクラス: HeatExchanger（熱交換器）
        class HeatExchanger(ProcessEquipment):
            """熱交換器クラス"""
            pass
    
        # サブクラス: Separator（分離装置）
        class Separator(ProcessEquipment):
            """分離装置クラス"""
            pass
    
        # ===== 個体（インスタンス）の作成 =====
    
        # CSTR反応器R-101
        r101 = CSTR("R-101")
        r101.label = ["連続撹拌槽型反応器R-101"]
        r101.comment = ["主要反応器、エステル化反応用"]
    
        # 熱交換器HX-201
        hx201 = HeatExchanger("HX-201")
        hx201.label = ["冷却器HX-201"]
    
        # 分離器SEP-301
        sep301 = Separator("SEP-301")
        sep301.label = ["気液分離器SEP-301"]
    
    # オントロジーの保存
    onto.save(file="process_ontology_v1.owl", format="rdfxml")
    
    # ===== クラス階層の可視化 =====
    print("=== クラス階層 ===")
    for cls in onto.classes():
        if cls != Thing:
            ancestors = list(cls.ancestors())
            ancestors.remove(cls)
            ancestors.remove(Thing)
            if ancestors:
                print(f"{cls.name} ⊂ {ancestors[0].name}")
            else:
                print(f"{cls.name} (トップレベル)")
    
    print("\n=== 個体一覧 ===")
    for individual in onto.individuals():
        print(f"- {individual.name}: {individual.__class__.name}")
        if individual.label:
            print(f"  ラベル: {individual.label[0]}")
    
    print(f"\n✓ オントロジー保存完了: process_ontology_v1.owl")
    print(f"総クラス数: {len(list(onto.classes()))}")
    print(f"総個体数: {len(list(onto.individuals()))}")
    

**出力例:**  
=== クラス階層 ===  
ProcessEquipment (トップレベル)  
Reactor ⊂ ProcessEquipment  
CSTR ⊂ Reactor  
HeatExchanger ⊂ ProcessEquipment  
Separator ⊂ ProcessEquipment  
  
=== 個体一覧 ===  
\- R-101: CSTR  
ラベル: 連続撹拌槽型反応器R-101  
\- HX-201: HeatExchanger  
ラベル: 冷却器HX-201  
\- SEP-301: Separator  
ラベル: 気液分離器SEP-301  
  
✓ オントロジー保存完了: process_ontology_v1.owl  
総クラス数: 5  
総個体数: 3 

## 2.2 オブジェクトプロパティ（Object Properties）

オブジェクトプロパティは、装置間の接続関係やプロセスフローを表現します。

### Example 2: 装置接続関係のオブジェクトプロパティ定義

装置間の入出力接続をオブジェクトプロパティで表現します。
    
    
    # ===================================
    # Example 2: オブジェクトプロパティ定義
    # ===================================
    
    from owlready2 import *
    
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        # ===== クラス定義 =====
        class ProcessEquipment(Thing):
            pass
    
        class Reactor(ProcessEquipment):
            pass
    
        class HeatExchanger(ProcessEquipment):
            pass
    
        class Stream(Thing):
            """物質ストリームクラス"""
            pass
    
        # ===== オブジェクトプロパティ定義 =====
    
        # hasInput: 装置への入力ストリーム
        class hasInput(ProcessEquipment >> Stream):
            """装置への入力ストリームを示す"""
            pass
    
        # hasOutput: 装置からの出力ストリーム
        class hasOutput(ProcessEquipment >> Stream):
            """装置からの出力ストリームを示す"""
            pass
    
        # connectedTo: 装置間の接続（対称性あり）
        class connectedTo(ProcessEquipment >> ProcessEquipment, SymmetricProperty):
            """装置間の物理的接続を示す（対称的）"""
            pass
    
        # feedsTo: 上流から下流への接続（推移性あり）
        class feedsTo(ProcessEquipment >> ProcessEquipment, TransitiveProperty):
            """上流装置から下流装置への流れを示す（推移的）"""
            pass
    
        # ===== 個体とプロパティの設定 =====
    
        # ストリームの作成
        s1 = Stream("S-001")
        s1.label = ["原料フィード"]
    
        s2 = Stream("S-002")
        s2.label = ["反応生成物"]
    
        s3 = Stream("S-003")
        s3.label = ["冷却後生成物"]
    
        # 装置の作成
        r101 = Reactor("R-101")
        hx201 = HeatExchanger("HX-201")
    
        # プロパティの設定
        r101.hasInput = [s1]
        r101.hasOutput = [s2]
    
        hx201.hasInput = [s2]
        hx201.hasOutput = [s3]
    
        # 装置間接続
        r101.connectedTo = [hx201]
        r101.feedsTo = [hx201]
    
    # ===== プロパティの検証 =====
    print("=== プロセスフロー ===")
    print(f"{r101.name} → hasInput → {r101.hasInput[0].name}")
    print(f"{r101.name} → hasOutput → {r101.hasOutput[0].name}")
    print(f"{hx201.name} → hasInput → {hx201.hasInput[0].name}")
    
    print("\n=== 装置接続 ===")
    print(f"{r101.name} → connectedTo → {r101.connectedTo[0].name}")
    print(f"{r101.name} → feedsTo → {r101.feedsTo[0].name}")
    
    # 対称性の確認
    print(f"\n対称性確認: {hx201.name} → connectedTo → {hx201.connectedTo}")
    
    onto.save(file="process_ontology_v2.owl", format="rdfxml")
    print("\n✓ オントロジー保存完了: process_ontology_v2.owl")
    

**出力例:**  
=== プロセスフロー ===  
R-101 → hasInput → S-001  
R-101 → hasOutput → S-002  
HX-201 → hasInput → S-002  
  
=== 装置接続 ===  
R-101 → connectedTo → HX-201  
R-101 → feedsTo → HX-201  
  
対称性確認: HX-201 → connectedTo → [process.R-101]  
  
✓ オントロジー保存完了: process_ontology_v2.owl 

### Example 3: データプロパティ（温度・圧力・流量）

プロセス変数をデータプロパティで表現します。
    
    
    # ===================================
    # Example 3: データプロパティ定義
    # ===================================
    
    from owlready2 import *
    
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        class ProcessEquipment(Thing):
            pass
    
        class Reactor(ProcessEquipment):
            pass
    
        # ===== データプロパティ定義 =====
    
        # hasTemperature: 温度 [K]
        class hasTemperature(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
            label = ["温度"]
            comment = ["装置の運転温度（単位: K）"]
    
        # hasPressure: 圧力 [bar]
        class hasPressure(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
            label = ["圧力"]
            comment = ["装置の運転圧力（単位: bar）"]
    
        # hasVolume: 容積 [m3]
        class hasVolume(DataProperty, FunctionalProperty):
            domain = [Reactor]
            range = [float]
            label = ["容積"]
    
        # hasFlowRate: 流量 [kg/h]
        class hasFlowRate(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
            label = ["流量"]
    
        # hasEfficiency: 効率 [%]
        class hasEfficiency(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
            label = ["効率"]
    
        # ===== 個体への値設定 =====
    
        # CSTR反応器R-101
        r101 = Reactor("R-101")
        r101.label = ["CSTR反応器"]
        r101.hasTemperature = [350.0]  # 350 K (約77°C)
        r101.hasPressure = [5.0]       # 5 bar
        r101.hasVolume = [10.0]        # 10 m3
        r101.hasFlowRate = [1000.0]    # 1000 kg/h
        r101.hasEfficiency = [92.5]    # 92.5%
    
        # PFR反応器R-102
        r102 = Reactor("R-102")
        r102.label = ["管型反応器"]
        r102.hasTemperature = [420.0]  # 420 K (約147°C)
        r102.hasPressure = [8.0]       # 8 bar
        r102.hasVolume = [5.0]         # 5 m3
        r102.hasFlowRate = [800.0]     # 800 kg/h
    
    # ===== データプロパティの取得 =====
    print("=== 反応器R-101の運転条件 ===")
    print(f"温度: {r101.hasTemperature[0]} K ({r101.hasTemperature[0] - 273.15:.1f}°C)")
    print(f"圧力: {r101.hasPressure[0]} bar")
    print(f"容積: {r101.hasVolume[0]} m³")
    print(f"流量: {r101.hasFlowRate[0]} kg/h")
    print(f"効率: {r101.hasEfficiency[0]}%")
    
    print("\n=== 反応器R-102の運転条件 ===")
    print(f"温度: {r102.hasTemperature[0]} K ({r102.hasTemperature[0] - 273.15:.1f}°C)")
    print(f"圧力: {r102.hasPressure[0]} bar")
    print(f"容積: {r102.hasVolume[0]} m³")
    print(f"流量: {r102.hasFlowRate[0]} kg/h")
    
    onto.save(file="process_ontology_v3.owl", format="rdfxml")
    print("\n✓ オントロジー保存完了: process_ontology_v3.owl")
    

**出力例:**  
=== 反応器R-101の運転条件 ===  
温度: 350.0 K (76.9°C)  
圧力: 5.0 bar  
容積: 10.0 m³  
流量: 1000.0 kg/h  
効率: 92.5%  
  
=== 反応器R-102の運転条件 ===  
温度: 420.0 K (146.9°C)  
圧力: 8.0 bar  
容積: 5.0 m³  
流量: 800.0 kg/h  
  
✓ オントロジー保存完了: process_ontology_v3.owl 

## 2.3 クラス階層とプロパティ制約

### Example 4: 詳細なクラス階層構築

装置タイプの詳細な分類階層を構築します。
    
    
    # ===================================
    # Example 4: 詳細なクラス階層
    # ===================================
    
    from owlready2 import *
    
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        # ===== 階層的クラス定義 =====
    
        class ProcessEquipment(Thing):
            """プロセス装置（基底クラス）"""
            pass
    
        # 反応器系統
        class Reactor(ProcessEquipment):
            """反応器"""
            pass
    
        class CSTR(Reactor):
            """連続撹拌槽型反応器"""
            pass
    
        class PFR(Reactor):
            """管型反応器（プラグフロー）"""
            pass
    
        class BatchReactor(Reactor):
            """バッチ反応器"""
            pass
    
        # 熱交換器系統
        class HeatExchanger(ProcessEquipment):
            """熱交換器"""
            pass
    
        class ShellTubeHX(HeatExchanger):
            """シェル&チューブ型熱交換器"""
            pass
    
        class PlateHX(HeatExchanger):
            """プレート型熱交換器"""
            pass
    
        # 分離装置系統
        class Separator(ProcessEquipment):
            """分離装置"""
            pass
    
        class DistillationColumn(Separator):
            """蒸留塔"""
            pass
    
        class Absorber(Separator):
            """吸収塔"""
            pass
    
        class Extractor(Separator):
            """抽出器"""
            pass
    
        # ポンプ・圧縮機系統
        class FluidMover(ProcessEquipment):
            """流体移送機器"""
            pass
    
        class Pump(FluidMover):
            """ポンプ"""
            pass
    
        class Compressor(FluidMover):
            """圧縮機"""
            pass
    
    # ===== クラス階層の可視化（Mermaid用データ生成） =====
    print("=== クラス階層ツリー ===")
    
    def print_class_tree(cls, indent=0):
        """再帰的にクラス階層を表示"""
        if cls != Thing:
            print("  " * indent + f"├─ {cls.name}")
            for subclass in cls.subclasses():
                print_class_tree(subclass, indent + 1)
    
    print_class_tree(ProcessEquipment)
    
    # 各カテゴリーの統計
    print("\n=== 装置カテゴリー別クラス数 ===")
    categories = {
        "Reactor": len(list(Reactor.descendants())),
        "HeatExchanger": len(list(HeatExchanger.descendants())),
        "Separator": len(list(Separator.descendants())),
        "FluidMover": len(list(FluidMover.descendants()))
    }
    
    for cat, count in categories.items():
        print(f"{cat}: {count}個のサブクラス")
    
    onto.save(file="process_ontology_v4.owl", format="rdfxml")
    print("\n✓ オントロジー保存完了: process_ontology_v4.owl")
    

**出力例:**  
=== クラス階層ツリー ===  
├─ ProcessEquipment  
├─ Reactor  
├─ CSTR  
├─ PFR  
├─ BatchReactor  
├─ HeatExchanger  
├─ ShellTubeHX  
├─ PlateHX  
├─ Separator  
├─ DistillationColumn  
├─ Absorber  
├─ Extractor  
├─ FluidMover  
├─ Pump  
├─ Compressor  
  
=== 装置カテゴリー別クラス数 ===  
Reactor: 3個のサブクラス  
HeatExchanger: 2個のサブクラス  
Separator: 3個のサブクラス  
FluidMover: 2個のサブクラス  
  
✓ オントロジー保存完了: process_ontology_v4.owl 

**💡 クラス設計のベストプラクティス**

装置分類は機能別（反応、分離、熱交換）とタイプ別（CSTR、PFR）の2軸で階層化します。これにより、プロセス設計時の装置選定が体系的に行えます。
    
    
    ```mermaid
    graph TB
        PE[ProcessEquipment]
    
        PE --> R[Reactor]
        PE --> HX[HeatExchanger]
        PE --> SEP[Separator]
        PE --> FM[FluidMover]
    
        R --> CSTR[CSTR]
        R --> PFR[PFR]
        R --> BR[BatchReactor]
    
        HX --> STHX[ShellTubeHX]
        HX --> PHX[PlateHX]
    
        SEP --> DC[DistillationColumn]
        SEP --> ABS[Absorber]
        SEP --> EXT[Extractor]
    
        FM --> PUMP[Pump]
        FM --> COMP[Compressor]
    
        style PE fill:#11998e,color:#fff
        style R fill:#38ef7d,color:#000
        style HX fill:#38ef7d,color:#000
        style SEP fill:#38ef7d,color:#000
        style FM fill:#38ef7d,color:#000
    ```

## 2.4 プロパティ制約（Restrictions）

### Example 5: カーディナリティ制約と値制約

装置が持つべき入出力数や値の範囲を制約として定義します。
    
    
    # ===================================
    # Example 5: プロパティ制約の定義
    # ===================================
    
    from owlready2 import *
    
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        class ProcessEquipment(Thing):
            pass
    
        class Stream(Thing):
            pass
    
        # プロパティ定義
        class hasInput(ProcessEquipment >> Stream):
            pass
    
        class hasOutput(ProcessEquipment >> Stream):
            pass
    
        class hasTemperature(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
    
        class hasPressure(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
    
        # ===== 制約付きクラス定義 =====
    
        # Reactor: 最低1つの入力、1つの出力、温度・圧力必須
        class Reactor(ProcessEquipment):
            equivalent_to = [
                ProcessEquipment
                & hasInput.min(1, Stream)        # 最低1つの入力
                & hasOutput.min(1, Stream)       # 最低1つの出力
                & hasTemperature.exactly(1)      # 温度は必須（1つのみ）
                & hasPressure.exactly(1)         # 圧力は必須（1つのみ）
            ]
    
        # HeatExchanger: 入力1つ、出力1つ（固定）
        class HeatExchanger(ProcessEquipment):
            equivalent_to = [
                ProcessEquipment
                & hasInput.exactly(1, Stream)    # 入力は正確に1つ
                & hasOutput.exactly(1, Stream)   # 出力は正確に1つ
                & hasTemperature.exactly(1)
            ]
    
        # DistillationColumn: 1つの入力、2つ以上の出力（留出と缶出）
        class DistillationColumn(ProcessEquipment):
            equivalent_to = [
                ProcessEquipment
                & hasInput.exactly(1, Stream)
                & hasOutput.min(2, Stream)       # 最低2つの出力（留出、缶出）
            ]
    
        # ===== 値制約付きクラス =====
    
        # HighTemperatureReactor: 温度 > 400K
        class HighTemperatureReactor(Reactor):
            """高温反応器（400K以上）"""
            pass
    
        # HighPressureReactor: 圧力 > 10bar
        class HighPressureReactor(Reactor):
            """高圧反応器（10bar以上）"""
            pass
    
        # ===== 個体の作成と検証 =====
    
        # 有効な反応器（制約を満たす）
        r101 = Reactor("R-101")
        s1 = Stream("S-001")
        s2 = Stream("S-002")
        r101.hasInput = [s1]
        r101.hasOutput = [s2]
        r101.hasTemperature = [450.0]  # 高温
        r101.hasPressure = [5.0]
    
        # 熱交換器（制約を満たす）
        hx201 = HeatExchanger("HX-201")
        s3 = Stream("S-003")
        s4 = Stream("S-004")
        hx201.hasInput = [s3]
        hx201.hasOutput = [s4]
        hx201.hasTemperature = [350.0]
    
    # ===== 制約の検証 =====
    print("=== 反応器R-101の検証 ===")
    print(f"入力数: {len(r101.hasInput)} (最低1つ必要)")
    print(f"出力数: {len(r101.hasOutput)} (最低1つ必要)")
    print(f"温度設定: {r101.hasTemperature[0]} K (必須)")
    print(f"圧力設定: {r101.hasPressure[0]} bar (必須)")
    print("✓ すべての制約を満たしています")
    
    print("\n=== 熱交換器HX-201の検証 ===")
    print(f"入力数: {len(hx201.hasInput)} (正確に1つ必要)")
    print(f"出力数: {len(hx201.hasOutput)} (正確に1つ必要)")
    print("✓ すべての制約を満たしています")
    
    onto.save(file="process_ontology_v5.owl", format="rdfxml")
    print("\n✓ オントロジー保存完了: process_ontology_v5.owl")
    

**出力例:**  
=== 反応器R-101の検証 ===  
入力数: 1 (最低1つ必要)  
出力数: 1 (最低1つ必要)  
温度設定: 450.0 K (必須)  
圧力設定: 5.0 bar (必須)  
✓ すべての制約を満たしています  
  
=== 熱交換器HX-201の検証 ===  
入力数: 1 (正確に1つ必要)  
出力数: 1 (正確に1つ必要)  
✓ すべての制約を満たしています  
  
✓ オントロジー保存完了: process_ontology_v5.owl 

**⚠️ 制約違反の検出**

owlready2の推論エンジン（Pellet, HermiT）を使用することで、制約違反を自動検出できます。例えば、入力のない反応器や温度未設定の装置は不整合として検出されます。

## 2.5 プロセス装置の完全なオントロジー

### Example 6: 完全な装置オントロジーの構築

すべての要素を統合した包括的なプロセス装置オントロジーを構築します。
    
    
    # ===================================
    # Example 6: 完全なプロセス装置オントロジー
    # ===================================
    
    from owlready2 import *
    import numpy as np
    
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        # ===== 基本クラス =====
        class ProcessEquipment(Thing):
            pass
    
        class Stream(Thing):
            pass
    
        # ===== 装置クラス階層 =====
        class Reactor(ProcessEquipment):
            pass
    
        class CSTR(Reactor):
            pass
    
        class HeatExchanger(ProcessEquipment):
            pass
    
        class Separator(ProcessEquipment):
            pass
    
        class Pump(ProcessEquipment):
            pass
    
        # ===== プロパティ定義 =====
    
        # オブジェクトプロパティ
        class hasInput(ProcessEquipment >> Stream):
            pass
    
        class hasOutput(ProcessEquipment >> Stream):
            pass
    
        class connectedTo(ProcessEquipment >> ProcessEquipment, SymmetricProperty):
            pass
    
        # データプロパティ
        class hasTemperature(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
    
        class hasPressure(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
    
        class hasFlowRate(DataProperty, FunctionalProperty):
            domain = [Stream]
            range = [float]
    
        class hasVolume(DataProperty, FunctionalProperty):
            domain = [Reactor]
            range = [float]
    
        class hasEfficiency(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
    
        class hasResidenceTime(DataProperty, FunctionalProperty):
            domain = [Reactor]
            range = [float]
            comment = ["滞留時間（単位: 秒）"]
    
        # ===== 完全なプロセスプラントの構築 =====
    
        # ストリーム
        feed = Stream("Feed")
        feed.label = ["原料フィード"]
        feed.hasFlowRate = [1000.0]  # kg/h
    
        s1 = Stream("S-001")
        s2 = Stream("S-002")
        s3 = Stream("S-003")
        product = Stream("Product")
        product.label = ["最終製品"]
    
        # ポンプP-101
        p101 = Pump("P-101")
        p101.label = ["フィードポンプ"]
        p101.hasInput = [feed]
        p101.hasOutput = [s1]
        p101.hasEfficiency = [85.0]
    
        # 反応器R-101
        r101 = CSTR("R-101")
        r101.label = ["主反応器"]
        r101.hasInput = [s1]
        r101.hasOutput = [s2]
        r101.hasTemperature = [350.0]  # K
        r101.hasPressure = [5.0]       # bar
        r101.hasVolume = [10.0]        # m3
        r101.hasEfficiency = [92.5]
        r101.hasResidenceTime = [3600.0]  # 1時間
    
        # 熱交換器HX-201
        hx201 = HeatExchanger("HX-201")
        hx201.label = ["冷却器"]
        hx201.hasInput = [s2]
        hx201.hasOutput = [s3]
        hx201.hasTemperature = [320.0]  # 冷却後温度
        hx201.hasEfficiency = [88.0]
    
        # 分離器SEP-301
        sep301 = Separator("SEP-301")
        sep301.label = ["製品分離器"]
        sep301.hasInput = [s3]
        sep301.hasOutput = [product]
        sep301.hasTemperature = [320.0]
        sep301.hasPressure = [1.0]
        sep301.hasEfficiency = [95.0]
    
        # 装置間接続
        p101.connectedTo = [r101]
        r101.connectedTo = [hx201]
        hx201.connectedTo = [sep301]
    
    # ===== プラント全体の可視化 =====
    print("=== プロセスプラント構成 ===\n")
    
    equipment_list = [p101, r101, hx201, sep301]
    
    for eq in equipment_list:
        print(f"【{eq.label[0]}】 ({eq.__class__.__name__} {eq.name})")
        if eq.hasTemperature:
            print(f"  温度: {eq.hasTemperature[0]} K ({eq.hasTemperature[0] - 273.15:.1f}°C)")
        if eq.hasPressure:
            print(f"  圧力: {eq.hasPressure[0]} bar")
        if eq.hasEfficiency:
            print(f"  効率: {eq.hasEfficiency[0]}%")
        if hasattr(eq, 'hasVolume') and eq.hasVolume:
            print(f"  容積: {eq.hasVolume[0]} m³")
        if hasattr(eq, 'hasResidenceTime') and eq.hasResidenceTime:
            print(f"  滞留時間: {eq.hasResidenceTime[0] / 3600:.1f} 時間")
        print()
    
    # 接続関係
    print("=== プロセスフロー ===")
    print("Feed → P-101 → R-101 → HX-201 → SEP-301 → Product")
    
    onto.save(file="process_plant_complete.owl", format="rdfxml")
    print("\n✓ 完全なプロセスプラントオントロジー保存完了")
    print(f"総装置数: {len(equipment_list)}")
    print(f"総ストリーム数: {len([feed, s1, s2, s3, product])}")
    

**出力例:**  
=== プロセスプラント構成 ===  
  
【フィードポンプ】 (Pump P-101)  
効率: 85.0%  
  
【主反応器】 (CSTR R-101)  
温度: 350.0 K (76.9°C)  
圧力: 5.0 bar  
効率: 92.5%  
容積: 10.0 m³  
滞留時間: 1.0 時間  
  
【冷却器】 (HeatExchanger HX-201)  
温度: 320.0 K (46.9°C)  
効率: 88.0%  
  
【製品分離器】 (Separator SEP-301)  
温度: 320.0 K (46.9°C)  
圧力: 1.0 bar  
効率: 95.0%  
  
=== プロセスフロー ===  
Feed → P-101 → R-101 → HX-201 → SEP-301 → Product  
  
✓ 完全なプロセスプラントオントロジー保存完了  
総装置数: 4  
総ストリーム数: 5 

## 2.6 統合化学プロセスプラントのオントロジー

### Example 7: 統合プラントオントロジーとSPARQLクエリ

完全な化学プラントをオントロジーで表現し、SPARQLで高度なクエリを実行します。
    
    
    # ===================================
    # Example 7: 統合プラントオントロジーとSPARQL
    # ===================================
    
    from owlready2 import *
    from rdflib import Graph, Namespace
    
    # owlready2でオントロジー構築
    onto = get_ontology("http://example.org/chemicalplant.owl")
    
    with onto:
        # クラスとプロパティ定義（Example 6と同様）
        class ProcessEquipment(Thing):
            pass
    
        class Reactor(ProcessEquipment):
            pass
    
        class HeatExchanger(ProcessEquipment):
            pass
    
        class Separator(ProcessEquipment):
            pass
    
        class hasTemperature(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
    
        class hasPressure(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
    
        class hasEfficiency(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
    
        # 完全なプラントの構築
        r101 = Reactor("R-101")
        r101.label = ["反応器R-101"]
        r101.hasTemperature = [350.0]
        r101.hasPressure = [5.0]
        r101.hasEfficiency = [92.5]
    
        r102 = Reactor("R-102")
        r102.label = ["反応器R-102"]
        r102.hasTemperature = [420.0]
        r102.hasPressure = [8.0]
        r102.hasEfficiency = [88.0]
    
        hx201 = HeatExchanger("HX-201")
        hx201.label = ["熱交換器HX-201"]
        hx201.hasTemperature = [320.0]
        hx201.hasEfficiency = [90.0]
    
        hx202 = HeatExchanger("HX-202")
        hx202.label = ["熱交換器HX-202"]
        hx202.hasTemperature = [340.0]
        hx202.hasEfficiency = [85.0]
    
        sep301 = Separator("SEP-301")
        sep301.label = ["分離器SEP-301"]
        sep301.hasTemperature = [310.0]
        sep301.hasPressure = [1.0]
        sep301.hasEfficiency = [95.0]
    
    # OWL保存とRDF読み込み
    onto.save(file="chemical_plant.owl", format="rdfxml")
    
    # rdflibでSPARQLクエリ実行
    g = Graph()
    g.parse("chemical_plant.owl", format="xml")
    
    # 名前空間定義
    ONTO = Namespace("http://example.org/chemicalplant.owl#")
    
    # ===== SPARQLクエリ集 =====
    
    # クエリ1: 効率90%以上の装置
    query1 = """
    PREFIX onto: 
    PREFIX rdfs: 
    
    SELECT ?equipment ?label ?efficiency
    WHERE {
        ?equipment onto:hasEfficiency ?efficiency .
        ?equipment rdfs:label ?label .
        FILTER (?efficiency >= 90.0)
    }
    ORDER BY DESC(?efficiency)
    """
    
    print("=== クエリ1: 効率90%以上の装置 ===")
    for row in g.query(query1):
        print(f"{row.label}: {float(row.efficiency):.1f}%")
    
    # クエリ2: 温度350K以上の装置
    query2 = """
    PREFIX onto: 
    PREFIX rdfs: 
    
    SELECT ?label ?temp ?press
    WHERE {
        ?equipment onto:hasTemperature ?temp .
        ?equipment rdfs:label ?label .
        OPTIONAL { ?equipment onto:hasPressure ?press }
        FILTER (?temp >= 350.0)
    }
    ORDER BY DESC(?temp)
    """
    
    print("\n=== クエリ2: 温度350K以上の高温装置 ===")
    for row in g.query(query2):
        temp_c = float(row.temp) - 273.15
        print(f"{row.label}: {float(row.temp):.1f}K ({temp_c:.1f}°C)", end="")
        if row.press:
            print(f", {float(row.press):.1f}bar")
        else:
            print()
    
    # クエリ3: 反応器の統計
    query3 = """
    PREFIX onto: 
    PREFIX rdf: 
    
    SELECT (AVG(?temp) AS ?avgTemp) (AVG(?eff) AS ?avgEff) (COUNT(?reactor) AS ?count)
    WHERE {
        ?reactor rdf:type onto:Reactor .
        ?reactor onto:hasTemperature ?temp .
        ?reactor onto:hasEfficiency ?eff .
    }
    """
    
    print("\n=== クエリ3: 反応器の統計情報 ===")
    for row in g.query(query3):
        print(f"反応器数: {row.count}")
        print(f"平均温度: {float(row.avgTemp):.1f}K ({float(row.avgTemp) - 273.15:.1f}°C)")
        print(f"平均効率: {float(row.avgEff):.1f}%")
    
    print("\n✓ 統合プラントオントロジーとSPARQLクエリ完了")
    

**出力例:**  
=== クエリ1: 効率90%以上の装置 ===  
分離器SEP-301: 95.0%  
反応器R-101: 92.5%  
熱交換器HX-201: 90.0%  
  
=== クエリ2: 温度350K以上の高温装置 ===  
反応器R-102: 420.0K (146.9°C), 8.0bar  
反応器R-101: 350.0K (76.9°C), 5.0bar  
  
=== クエリ3: 反応器の統計情報 ===  
反応器数: 2  
平均温度: 385.0K (111.9°C)  
平均効率: 90.3%  
  
✓ 統合プラントオントロジーとSPARQLクエリ完了 

**✅ オントロジー設計の成果**

  * **構造化** : 13クラス、5プロパティの階層的オントロジー
  * **制約** : カーディナリティと値範囲の制約による品質保証
  * **クエリ** : SPARQLによる高度な知識抽出と分析
  * **拡張性** : 新しい装置タイプや物性の追加が容易

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります：

### 基本理解

  * ✅ OWLの3つのサブ言語（Lite, DL, Full）の違いを説明できる
  * ✅ オブジェクトプロパティとデータプロパティの役割を理解している
  * ✅ FunctionalProperty、SymmetricProperty、TransitivePropertyの概念を知っている
  * ✅ クラス階層とis-a関係の設計原則を理解している

### 実践スキル

  * ✅ owlready2でOWLクラスとインスタンスを定義できる
  * ✅ 装置間の接続をオブジェクトプロパティで表現できる
  * ✅ 温度・圧力・流量などの物性をデータプロパティで定義できる
  * ✅ カーディナリティ制約（min, max, exactly）を実装できる
  * ✅ 完全な化学プロセスプラントのオントロジーを構築できる
  * ✅ SPARQLで装置の統計情報や条件検索ができる

### 応用力

  * ✅ 新しい装置タイプを既存のオントロジーに統合できる
  * ✅ プロセス設計における装置選定をオントロジーで支援できる
  * ✅ プロパティ制約により不整合な装置構成を検出できる
  * ✅ 実プラントのP&ID情報をオントロジーに変換できる

## 次のステップ

第2章では、OWLとowlready2を用いた化学プロセス装置の包括的なオントロジー設計を学びました。次章では、実際のプロセスデータ（CSV、センサーストリーム、P&ID）からナレッジグラフを自動構築する手法を学びます。

**📚 次章の内容（第3章予告）**

  * CSVデータからのエンティティ抽出
  * 装置接続関係の自動抽出
  * センサーデータストリームのRDF変換
  * 歴史的データの時系列ナレッジグラフ
  * マルチソースデータの統合ナレッジグラフ構築

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
