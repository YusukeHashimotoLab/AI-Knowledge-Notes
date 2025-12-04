---
title: 第1章：オントロジーとセマンティックWebの基礎
chapter_title: 第1章：オントロジーとセマンティックWebの基礎
subtitle: RDF、RDFS、SPARQLによるプロセス知識の構造化
---

## 1.1 RDF（Resource Description Framework）の基礎

セマンティックWebの基盤技術であるRDFは、情報を「主語（Subject）」「述語（Predicate）」「目的語（Object）」の3つ組（トリプル）で表現します。この構造により、化学プロセスの複雑な知識を機械可読な形式で記述できます。

**💡 RDFトリプルの構造**

  * **主語（Subject）** : 記述対象のリソース（例: 反応器R-101）
  * **述語（Predicate）** : リソース間の関係（例: hasTemperature）
  * **目的語（Object）** : 値またはリソース（例: 350°C）

### Example 1: rdflibによる基本的なRDFグラフの構築

化学反応器の基本情報をRDFトリプルで表現します。
    
    
    # ===================================
    # Example 1: RDFグラフの基本構築
    # ===================================
    
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, XSD
    
    # 名前空間の定義
    PROC = Namespace("http://example.org/process/")
    UNIT = Namespace("http://example.org/unit/")
    
    # グラフの作成
    g = Graph()
    g.bind("proc", PROC)
    g.bind("unit", UNIT)
    
    # トリプルの追加（反応器R-101の情報）
    reactor = PROC["R-101"]
    
    # 基本属性
    g.add((reactor, RDF.type, PROC.Reactor))
    g.add((reactor, RDFS.label, Literal("連続撹拌槽型反応器", lang="ja")))
    g.add((reactor, PROC.hasTemperature, Literal(350, datatype=XSD.double)))
    g.add((reactor, PROC.hasPressure, Literal(5.0, datatype=XSD.double)))
    g.add((reactor, PROC.hasVolume, Literal(10.0, datatype=XSD.double)))
    g.add((reactor, PROC.unit, UNIT.degC))
    
    # 反応器への入力物質
    g.add((reactor, PROC.hasInput, PROC["Stream-01"]))
    g.add((PROC["Stream-01"], RDFS.label, Literal("原料フィード")))
    g.add((PROC["Stream-01"], PROC.flowRate, Literal(100.0, datatype=XSD.double)))
    
    # 反応器からの出力物質
    g.add((reactor, PROC.hasOutput, PROC["Stream-02"]))
    g.add((PROC["Stream-02"], RDFS.label, Literal("反応生成物")))
    
    # Turtle形式でシリアライズ（人が読みやすい）
    print("=== Turtle形式 ===")
    print(g.serialize(format="turtle"))
    
    # トリプル数の確認
    print(f"\n総トリプル数: {len(g)}")
    
    # 特定の述語でクエリ
    print("\n=== 温度情報の取得 ===")
    for s, p, o in g.triples((None, PROC.hasTemperature, None)):
        print(f"{s} の温度: {o}°C")
    

**出力例:**  
=== Turtle形式 ===  
@prefix proc: <http://example.org/process/> .  
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .  
  
proc:R-101 a proc:Reactor ;  
rdfs:label "連続撹拌槽型反応器"@ja ;  
proc:hasTemperature 350.0 ;  
proc:hasPressure 5.0 .  
  
総トリプル数: 11  
http://example.org/process/R-101 の温度: 350.0°C 

### Example 2: RDF/XMLとTurtle記法の変換

異なるシリアライゼーション形式間の変換を実装します。
    
    
    # ===================================
    # Example 2: シリアライゼーション形式の変換
    # ===================================
    
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS
    
    # 蒸留塔のRDFグラフ
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # 蒸留塔D-201の情報
    column = PROC["D-201"]
    g.add((column, RDF.type, PROC.DistillationColumn))
    g.add((column, RDFS.label, Literal("蒸留塔D-201", lang="ja")))
    g.add((column, PROC.numberOfTrays, Literal(30)))
    g.add((column, PROC.refluxRatio, Literal(2.5)))
    g.add((column, PROC.feedTray, Literal(15)))
    
    # RDF/XML形式
    print("=== RDF/XML形式 ===")
    rdfxml = g.serialize(format="xml")
    print(rdfxml)
    
    # Turtle形式
    print("\n=== Turtle形式 ===")
    turtle = g.serialize(format="turtle")
    print(turtle)
    
    # N-Triples形式（最もシンプル）
    print("\n=== N-Triples形式 ===")
    ntriples = g.serialize(format="nt")
    print(ntriples)
    
    # JSON-LD形式（Web APIで便利）
    print("\n=== JSON-LD形式 ===")
    jsonld = g.serialize(format="json-ld", indent=2)
    print(jsonld)
    
    # ファイルへの保存
    g.serialize(destination="distillation_column.ttl", format="turtle")
    print("\n✓ Turtle形式でファイル保存完了: distillation_column.ttl")
    
    # ファイルからの読み込み
    g_loaded = Graph()
    g_loaded.parse("distillation_column.ttl", format="turtle")
    print(f"✓ 読み込み完了: {len(g_loaded)} トリプル")
    

**出力例:**  
=== Turtle形式 ===  
proc:D-201 a proc:DistillationColumn ;  
rdfs:label "蒸留塔D-201"@ja ;  
proc:numberOfTrays 30 ;  
proc:refluxRatio 2.5 ;  
proc:feedTray 15 .  
  
✓ Turtle形式でファイル保存完了  
✓ 読み込み完了: 5 トリプル 

## 1.2 RDFS（RDF Schema）によるクラス階層

RDFSはRDFを拡張し、クラスやプロパティの階層構造を定義できます。化学プロセス装置の分類体系を構築する上で重要な概念です。

### Example 3: RDFS階層構造の定義

化学装置のクラス階層とプロパティを定義します。
    
    
    # ===================================
    # Example 3: RDFS階層構造の定義
    # ===================================
    
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS
    
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # ===== クラス階層の定義 =====
    
    # 最上位クラス: ProcessEquipment
    g.add((PROC.ProcessEquipment, RDF.type, RDFS.Class))
    g.add((PROC.ProcessEquipment, RDFS.label, Literal("プロセス装置")))
    
    # サブクラス定義
    # Reactor（反応器）
    g.add((PROC.Reactor, RDF.type, RDFS.Class))
    g.add((PROC.Reactor, RDFS.subClassOf, PROC.ProcessEquipment))
    g.add((PROC.Reactor, RDFS.label, Literal("反応器")))
    
    # HeatExchanger（熱交換器）
    g.add((PROC.HeatExchanger, RDF.type, RDFS.Class))
    g.add((PROC.HeatExchanger, RDFS.subClassOf, PROC.ProcessEquipment))
    g.add((PROC.HeatExchanger, RDFS.label, Literal("熱交換器")))
    
    # Separator（分離装置）
    g.add((PROC.Separator, RDF.type, RDFS.Class))
    g.add((PROC.Separator, RDFS.subClassOf, PROC.ProcessEquipment))
    g.add((PROC.Separator, RDFS.label, Literal("分離装置")))
    
    # さらなるサブクラス: DistillationColumn（蒸留塔）
    g.add((PROC.DistillationColumn, RDF.type, RDFS.Class))
    g.add((PROC.DistillationColumn, RDFS.subClassOf, PROC.Separator))
    g.add((PROC.DistillationColumn, RDFS.label, Literal("蒸留塔")))
    
    # ===== プロパティ定義 =====
    
    # hasInput（入力）
    g.add((PROC.hasInput, RDF.type, RDF.Property))
    g.add((PROC.hasInput, RDFS.domain, PROC.ProcessEquipment))
    g.add((PROC.hasInput, RDFS.range, PROC.Stream))
    g.add((PROC.hasInput, RDFS.label, Literal("入力")))
    
    # hasOutput（出力）
    g.add((PROC.hasOutput, RDF.type, RDF.Property))
    g.add((PROC.hasOutput, RDFS.domain, PROC.ProcessEquipment))
    g.add((PROC.hasOutput, RDFS.range, PROC.Stream))
    g.add((PROC.hasOutput, RDFS.label, Literal("出力")))
    
    # hasTemperature（温度）
    g.add((PROC.hasTemperature, RDF.type, RDF.Property))
    g.add((PROC.hasTemperature, RDFS.domain, PROC.ProcessEquipment))
    g.add((PROC.hasTemperature, RDFS.label, Literal("温度")))
    
    # ===== インスタンスの作成 =====
    reactor = PROC["R-101"]
    g.add((reactor, RDF.type, PROC.Reactor))
    g.add((reactor, RDFS.label, Literal("CSTR反応器")))
    
    # クラス階層の可視化
    print("=== クラス階層 ===")
    for subclass in g.subjects(RDFS.subClassOf, None):
        for superclass in g.objects(subclass, RDFS.subClassOf):
            sub_label = g.value(subclass, RDFS.label)
            super_label = g.value(superclass, RDFS.label)
            print(f"{sub_label} → {super_label}")
    
    # プロパティの一覧
    print("\n=== プロパティ一覧 ===")
    for prop in g.subjects(RDF.type, RDF.Property):
        label = g.value(prop, RDFS.label)
        domain = g.value(prop, RDFS.domain)
        range_val = g.value(prop, RDFS.range)
        print(f"- {label}: {domain} → {range_val}")
    
    print(f"\n総トリプル数: {len(g)}")
    print(g.serialize(format="turtle"))
    

**出力例:**  
=== クラス階層 ===  
反応器 → プロセス装置  
熱交換器 → プロセス装置  
分離装置 → プロセス装置  
蒸留塔 → 分離装置  
  
=== プロパティ一覧 ===  
\- 入力: ProcessEquipment → Stream  
\- 出力: ProcessEquipment → Stream  
\- 温度: ProcessEquipment → (未定義)  
  
総トリプル数: 28 

## 1.3 SPARQLクエリの基礎

SPARQLはRDFグラフに対するクエリ言語です。SQLに似た構文で、複雑なパターンマッチングと知識抽出が可能です。

### Example 4: SPARQL SELECTクエリ

プロセス装置に関する情報を抽出します。
    
    
    # ===================================
    # Example 4: SPARQL SELECTクエリ
    # ===================================
    
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD
    
    # サンプルデータ作成
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # 複数の反応器データ
    reactors = [
        ("R-101", "CSTR反応器", 350, 5.0, 100),
        ("R-102", "PFR反応器", 400, 8.0, 150),
        ("R-103", "バッチ反応器", 320, 3.0, 80),
    ]
    
    for id, label, temp, press, flow in reactors:
        reactor = PROC[id]
        g.add((reactor, RDF.type, PROC.Reactor))
        g.add((reactor, RDFS.label, Literal(label, lang="ja")))
        g.add((reactor, PROC.hasTemperature, Literal(temp, datatype=XSD.double)))
        g.add((reactor, PROC.hasPressure, Literal(press, datatype=XSD.double)))
        g.add((reactor, PROC.flowRate, Literal(flow, datatype=XSD.double)))
    
    # ===== SPARQL クエリ実行 =====
    
    # クエリ1: すべての反応器の基本情報
    query1 = """
    PREFIX proc: 
    PREFIX rdfs: 
    
    SELECT ?reactor ?label ?temp ?press
    WHERE {
        ?reactor a proc:Reactor .
        ?reactor rdfs:label ?label .
        ?reactor proc:hasTemperature ?temp .
        ?reactor proc:hasPressure ?press .
    }
    ORDER BY DESC(?temp)
    """
    
    print("=== クエリ1: 全反応器（温度降順） ===")
    results1 = g.query(query1)
    for row in results1:
        print(f"{row.label}: {row.temp}°C, {row.press}bar")
    
    # クエリ2: 条件付き検索（温度 > 340°C かつ 圧力 > 4bar）
    query2 = """
    PREFIX proc: 
    PREFIX rdfs: 
    
    SELECT ?label ?temp ?press
    WHERE {
        ?reactor a proc:Reactor .
        ?reactor rdfs:label ?label .
        ?reactor proc:hasTemperature ?temp .
        ?reactor proc:hasPressure ?press .
        FILTER (?temp > 340 && ?press > 4.0)
    }
    """
    
    print("\n=== クエリ2: 高温高圧反応器 ===")
    results2 = g.query(query2)
    for row in results2:
        print(f"{row.label}: {row.temp}°C, {row.press}bar")
    
    # クエリ3: 集約（平均温度、最大圧力）
    query3 = """
    PREFIX proc: 
    
    SELECT (AVG(?temp) AS ?avgTemp) (MAX(?press) AS ?maxPress) (COUNT(?reactor) AS ?count)
    WHERE {
        ?reactor a proc:Reactor .
        ?reactor proc:hasTemperature ?temp .
        ?reactor proc:hasPressure ?press .
    }
    """
    
    print("\n=== クエリ3: 統計情報 ===")
    results3 = g.query(query3)
    for row in results3:
        print(f"反応器数: {row.count}")
        print(f"平均温度: {float(row.avgTemp):.1f}°C")
        print(f"最大圧力: {float(row.maxPress)}bar")
    

**出力例:**  
=== クエリ1: 全反応器（温度降順） ===  
PFR反応器: 400.0°C, 8.0bar  
CSTR反応器: 350.0°C, 5.0bar  
バッチ反応器: 320.0°C, 3.0bar  
  
=== クエリ2: 高温高圧反応器 ===  
PFR反応器: 400.0°C, 8.0bar  
CSTR反応器: 350.0°C, 5.0bar  
  
=== クエリ3: 統計情報 ===  
反応器数: 3  
平均温度: 356.7°C  
最大圧力: 8.0bar 

## 1.4 化学プロセスの知識表現

### Example 5: 装置接続関係のグラフ構築

プロセスフロー図をRDFグラフで表現します。
    
    
    # ===================================
    # Example 5: プロセスフロー図のRDF表現
    # ===================================
    
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS
    
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # ===== プロセスフロー: Feed → Reactor → HeatExchanger → Separator =====
    
    # 1. 原料タンク（Feed Tank）
    feed_tank = PROC["TK-001"]
    g.add((feed_tank, RDF.type, PROC.StorageTank))
    g.add((feed_tank, RDFS.label, Literal("原料タンク")))
    g.add((feed_tank, PROC.capacity, Literal(50000)))  # リットル
    
    # 2. 反応器（Reactor）
    reactor = PROC["R-101"]
    g.add((reactor, RDF.type, PROC.Reactor))
    g.add((reactor, RDFS.label, Literal("主反応器")))
    
    # 3. 熱交換器（Heat Exchanger）
    hx = PROC["HX-201"]
    g.add((hx, RDF.type, PROC.HeatExchanger))
    g.add((hx, RDFS.label, Literal("冷却器")))
    
    # 4. 分離器（Separator）
    separator = PROC["SEP-301"]
    g.add((separator, RDF.type, PROC.Separator))
    g.add((separator, RDFS.label, Literal("気液分離器")))
    
    # ===== 物質ストリーム =====
    s1 = PROC["S-001"]  # Feed → Reactor
    s2 = PROC["S-002"]  # Reactor → HX
    s3 = PROC["S-003"]  # HX → Separator
    
    for stream in [s1, s2, s3]:
        g.add((stream, RDF.type, PROC.Stream))
    
    # ===== 装置間接続 =====
    # Feed Tank → Reactor
    g.add((feed_tank, PROC.hasOutput, s1))
    g.add((reactor, PROC.hasInput, s1))
    
    # Reactor → Heat Exchanger
    g.add((reactor, PROC.hasOutput, s2))
    g.add((hx, PROC.hasInput, s2))
    
    # Heat Exchanger → Separator
    g.add((hx, PROC.hasOutput, s3))
    g.add((separator, PROC.hasInput, s3))
    
    # ===== プロセスフローの可視化 =====
    print("=== プロセスフロー図 ===\n")
    
    # 装置リスト
    print("装置一覧:")
    for eq in g.subjects(RDF.type, None):
        if eq != PROC.Stream:
            eq_type = g.value(eq, RDF.type)
            eq_label = g.value(eq, RDFS.label)
            if eq_type and eq_type != RDFS.Class:
                print(f"  - {eq_label} ({eq_type.split('/')[-1]})")
    
    # 接続関係
    print("\n接続関係:")
    for s in g.subjects(PROC.hasOutput, None):
        source_label = g.value(s, RDFS.label)
        for stream in g.objects(s, PROC.hasOutput):
            for target in g.subjects(PROC.hasInput, stream):
                target_label = g.value(target, RDFS.label)
                print(f"  {source_label} → {target_label}")
    
    # SPARQLで経路探索
    query = """
    PREFIX proc: 
    PREFIX rdfs: 
    
    SELECT ?source_label ?target_label
    WHERE {
        ?source proc:hasOutput ?stream .
        ?target proc:hasInput ?stream .
        ?source rdfs:label ?source_label .
        ?target rdfs:label ?target_label .
    }
    """
    
    print("\n=== SPARQLクエリ結果（接続関係） ===")
    for row in g.query(query):
        print(f"{row.source_label} ⟶ {row.target_label}")
    

**出力例:**  
=== プロセスフロー図 ===  
  
装置一覧:  
\- 原料タンク (StorageTank)  
\- 主反応器 (Reactor)  
\- 冷却器 (HeatExchanger)  
\- 気液分離器 (Separator)  
  
接続関係:  
原料タンク → 主反応器  
主反応器 → 冷却器  
冷却器 → 気液分離器 

**💡 実務への示唆**

このRDFグラフ構造により、P&ID（配管計装図）の情報をデジタル化し、機械可読な形式で管理できます。装置の追加や変更も柔軟に対応可能です。

## 1.5 物質と物性の表現

### Example 6: 化学物質と物性のRDFモデル

化学物質の物性データをRDFで構造化します。
    
    
    # ===================================
    # Example 6: 化学物質と物性のRDFモデル
    # ===================================
    
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD
    
    g = Graph()
    CHEM = Namespace("http://example.org/chemistry/")
    PROP = Namespace("http://example.org/property/")
    g.bind("chem", CHEM)
    g.bind("prop", PROP)
    
    # ===== 化学物質の定義 =====
    
    # エタノール
    ethanol = CHEM["Ethanol"]
    g.add((ethanol, RDF.type, CHEM.Chemical))
    g.add((ethanol, RDFS.label, Literal("エタノール", lang="ja")))
    g.add((ethanol, CHEM.formula, Literal("C2H5OH")))
    g.add((ethanol, CHEM.cas, Literal("64-17-5")))
    g.add((ethanol, CHEM.smiles, Literal("CCO")))
    
    # 物性データ
    g.add((ethanol, PROP.molecularWeight, Literal(46.07, datatype=XSD.double)))
    g.add((ethanol, PROP.boilingPoint, Literal(78.37, datatype=XSD.double)))
    g.add((ethanol, PROP.meltingPoint, Literal(-114.1, datatype=XSD.double)))
    g.add((ethanol, PROP.density, Literal(0.789, datatype=XSD.double)))
    
    # 水
    water = CHEM["Water"]
    g.add((water, RDF.type, CHEM.Chemical))
    g.add((water, RDFS.label, Literal("水", lang="ja")))
    g.add((water, CHEM.formula, Literal("H2O")))
    g.add((water, CHEM.cas, Literal("7732-18-5")))
    g.add((water, PROP.molecularWeight, Literal(18.015, datatype=XSD.double)))
    g.add((water, PROP.boilingPoint, Literal(100.0, datatype=XSD.double)))
    g.add((water, PROP.meltingPoint, Literal(0.0, datatype=XSD.double)))
    g.add((water, PROP.density, Literal(1.0, datatype=XSD.double)))
    
    # ===== 混合物の表現 =====
    mixture = CHEM["EthanolWaterMixture"]
    g.add((mixture, RDF.type, CHEM.Mixture))
    g.add((mixture, RDFS.label, Literal("エタノール水溶液")))
    g.add((mixture, CHEM.contains, ethanol))
    g.add((mixture, CHEM.contains, water))
    g.add((mixture, CHEM.composition, Literal("50% vol/vol")))
    
    # ===== SPARQLクエリ: 沸点が80°C以下の物質 =====
    query = """
    PREFIX chem: 
    PREFIX prop: 
    PREFIX rdfs: 
    
    SELECT ?name ?formula ?bp
    WHERE {
        ?chemical a chem:Chemical .
        ?chemical rdfs:label ?name .
        ?chemical chem:formula ?formula .
        ?chemical prop:boilingPoint ?bp .
        FILTER (?bp <= 80)
    }
    ORDER BY ?bp
    """
    
    print("=== 沸点80°C以下の物質 ===")
    for row in g.query(query):
        print(f"{row.name} ({row.formula}): 沸点 {row.bp}°C")
    
    # 分子量の比較
    query2 = """
    PREFIX chem: 
    PREFIX prop: 
    PREFIX rdfs: 
    
    SELECT ?name ?mw
    WHERE {
        ?chemical a chem:Chemical .
        ?chemical rdfs:label ?name .
        ?chemical prop:molecularWeight ?mw .
    }
    ORDER BY DESC(?mw)
    """
    
    print("\n=== 分子量順 ===")
    for row in g.query(query2):
        print(f"{row.name}: {float(row.mw):.2f} g/mol")
    
    # 混合物の構成成分
    print("\n=== 混合物の構成 ===")
    for component in g.objects(mixture, CHEM.contains):
        label = g.value(component, RDFS.label)
        print(f"- {label}")
    

**出力例:**  
=== 沸点80°C以下の物質 ===  
エタノール (C2H5OH): 沸点 78.37°C  
  
=== 分子量順 ===  
エタノール: 46.07 g/mol  
水: 18.02 g/mol  
  
=== 混合物の構成 ===  
\- エタノール  
\- 水 

## 1.6 名前空間とURI管理

### Example 7: 複数名前空間の統合管理

異なるオントロジーを統合する際の名前空間管理を実装します。
    
    
    # ===================================
    # Example 7: 名前空間の統合管理
    # ===================================
    
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, OWL, SKOS, DCTERMS
    
    # グラフの作成と名前空間バインディング
    g = Graph()
    
    # 標準名前空間
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    
    # カスタム名前空間
    PROC = Namespace("http://example.org/process/")
    CHEM = Namespace("http://example.org/chemistry/")
    SENSOR = Namespace("http://example.org/sensor/")
    UNIT = Namespace("http://example.org/unit/")
    
    g.bind("proc", PROC)
    g.bind("chem", CHEM)
    g.bind("sensor", SENSOR)
    g.bind("unit", UNIT)
    
    # ===== オントロジーメタデータ =====
    ontology_uri = PROC["ontology"]
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, DCTERMS.title, Literal("プロセスオントロジー", lang="ja")))
    g.add((ontology_uri, DCTERMS.creator, Literal("Hashimoto Lab")))
    g.add((ontology_uri, DCTERMS.created, Literal("2025-10-26")))
    g.add((ontology_uri, OWL.versionInfo, Literal("1.0")))
    
    # ===== 複数名前空間を使ったデータ =====
    
    # 温度センサー
    temp_sensor = SENSOR["TE-101"]
    g.add((temp_sensor, RDF.type, SENSOR.TemperatureSensor))
    g.add((temp_sensor, RDFS.label, Literal("温度センサーTE-101")))
    g.add((temp_sensor, SENSOR.measuredProperty, PROC.Temperature))
    g.add((temp_sensor, SENSOR.unit, UNIT.degC))
    g.add((temp_sensor, SENSOR.installedAt, PROC["R-101"]))
    
    # 反応器R-101
    reactor = PROC["R-101"]
    g.add((reactor, RDF.type, PROC.Reactor))
    g.add((reactor, RDFS.label, Literal("主反応器")))
    g.add((reactor, PROC.processes, CHEM["EsterificationReaction"]))
    
    # 化学反応
    reaction = CHEM["EsterificationReaction"]
    g.add((reaction, RDF.type, CHEM.ChemicalReaction))
    g.add((reaction, RDFS.label, Literal("エステル化反応")))
    g.add((reaction, SKOS.definition, Literal("アルコールとカルボン酸からエステルを生成する反応")))
    
    # ===== 名前空間の検証 =====
    print("=== バインド済み名前空間 ===")
    for prefix, namespace in g.namespaces():
        print(f"{prefix}: {namespace}")
    
    # URIの構築確認
    print("\n=== URI構築例 ===")
    print(f"反応器URI: {reactor}")
    print(f"センサーURI: {temp_sensor}")
    print(f"反応URI: {reaction}")
    
    # 名前空間別トリプル数
    print("\n=== 名前空間別トリプル数 ===")
    namespace_counts = {}
    for s, p, o in g:
        # 主語の名前空間をカウント
        ns = str(s).rsplit('/', 1)[0] + '/'
        namespace_counts[ns] = namespace_counts.get(ns, 0) + 1
    
    for ns, count in sorted(namespace_counts.items(), key=lambda x: x[1], reverse=True):
        # 名前空間からプレフィックスを逆引き
        prefix = None
        for p, n in g.namespaces():
            if str(n) == ns:
                prefix = p
                break
        print(f"{prefix or 'unknown'}: {count} トリプル")
    
    # Turtle形式で出力（名前空間が整理される）
    print("\n=== Turtle形式（抜粋） ===")
    print(g.serialize(format="turtle")[:800])
    

**出力例:**  
=== バインド済み名前空間 ===  
rdf: http://www.w3.org/1999/02/22-rdf-syntax-ns#  
rdfs: http://www.w3.org/2000/01/rdf-schema#  
proc: http://example.org/process/  
chem: http://example.org/chemistry/  
sensor: http://example.org/sensor/  
unit: http://example.org/unit/  
  
=== URI構築例 ===  
反応器URI: http://example.org/process/R-101  
センサーURI: http://example.org/sensor/TE-101  
反応URI: http://example.org/chemistry/EsterificationReaction  
  
=== 名前空間別トリプル数 ===  
sensor: 5 トリプル  
proc: 4 トリプル  
chem: 2 トリプル 

**✅ ベストプラクティス**

  * **一貫した名前空間URI** : 組織ドメインを含む永続的なURIを使用
  * **標準オントロジー活用** : Dublin Core、SKOSなど既存の標準を積極的に使用
  * **バージョン管理** : owl:versionInfoでオントロジーのバージョンを記録

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります：

### 基本理解

  * ✅ RDFトリプルの構造（Subject-Predicate-Object）を説明できる
  * ✅ RDFS階層構造の概念を理解している
  * ✅ SPARQLクエリの基本構文を知っている
  * ✅ 名前空間とURIの役割を理解している

### 実践スキル

  * ✅ rdflibでRDFグラフを作成・操作できる
  * ✅ Turtle/RDF-XML形式でRDFをシリアライズできる
  * ✅ SPARQLでフィルタリング・集約クエリを書ける
  * ✅ 化学プロセスのフロー図をRDFで表現できる
  * ✅ 複数の名前空間を統合管理できる

### 応用力

  * ✅ P&ID情報をRDFグラフに変換できる
  * ✅ 化学物質の物性データベースをRDFで構築できる
  * ✅ プロセス装置の階層的分類を設計できる

## 次のステップ

第1章では、RDF/RDFSによるセマンティックWeb技術の基礎とSPARQLクエリを学びました。次章では、OWL（Web Ontology Language）を用いた高度なプロセスオントロジー設計と推論可能な知識モデリングを学びます。

**📚 次章の内容（第2章予告）**

  * OWLクラスとプロパティの定義
  * カーディナリティ制約と値制約
  * プロセス装置の完全なオントロジー設計
  * owlready2による実装

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
