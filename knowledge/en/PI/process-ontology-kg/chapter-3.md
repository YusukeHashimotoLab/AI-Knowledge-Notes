---
title: "Chapter 3: Knowledge Graph Construction from Process Data"
chapter_title: "Chapter 3: Knowledge Graph Construction from Process Data"
subtitle: Automatic RDF Conversion and Triple Generation from CSV, Sensor, and P&ID Data
---

This chapter covers Knowledge Graph Construction from Process Data. You will learn entity extraction process from CSV data.

## 3.1 Entity Extraction from CSV Data

In real chemical plants, equipment information and operational data are managed in CSV format. We will learn methods to automatically construct knowledge graphs from this data.

**💡 Three Steps of Knowledge Graph Construction**

  1. **Entity Extraction** : Identify equipment and streams from data
  2. **Relationship Extraction** : Specify connections and causal relationships between equipment
  3. **Triple Generation** : Convert to RDF format (Subject-Predicate-Object)

### Example 1: Equipment Entity Extraction from CSV Data

Automatically generate RDF triples from equipment master data.
    
    
    # ===================================
    # Example 1: Entity Extraction from CSV
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, XSD
    import io
    
    # CSV data (equipment master)
    csv_data = """EquipmentID,Type,Name,Temperature_K,Pressure_bar,Volume_m3,Efficiency_pct
    R-101,CSTR,Main Reactor,350.0,5.0,10.0,92.5
    R-102,PFR,Tubular Reactor,420.0,8.0,5.0,88.0
    HX-201,HeatExchanger,Cooler HX-201,320.0,,,90.0
    HX-202,HeatExchanger,Heater HX-202,450.0,,,85.0
    SEP-301,Separator,Distillation Column,340.0,1.5,,95.0
    P-401,Pump,Feed Pump,300.0,10.0,,85.0"""
    
    # Load into DataFrame
    df = pd.read_csv(io.StringIO(csv_data))
    
    print("=== Original CSV Data ===")
    print(df.head(3))
    
    # Create RDF graph
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # ===== Convert CSV to RDF triples =====
    
    def csv_to_rdf(row):
        """Convert CSV row to RDF triples"""
        equipment_uri = PROC[row['EquipmentID']]
    
        # Basic triples
        g.add((equipment_uri, RDF.type, PROC[row['Type']]))
        g.add((equipment_uri, RDFS.label, Literal(row['Name'], lang='en')))
    
        # Temperature (required)
        g.add((equipment_uri, PROC.hasTemperature,
               Literal(row['Temperature_K'], datatype=XSD.double)))
    
        # Pressure (optional)
        if pd.notna(row['Pressure_bar']):
            g.add((equipment_uri, PROC.hasPressure,
                   Literal(row['Pressure_bar'], datatype=XSD.double)))
    
        # Volume (optional)
        if pd.notna(row['Volume_m3']):
            g.add((equipment_uri, PROC.hasVolume,
                   Literal(row['Volume_m3'], datatype=XSD.double)))
    
        # Efficiency (required)
        g.add((equipment_uri, PROC.hasEfficiency,
               Literal(row['Efficiency_pct'], datatype=XSD.double)))
    
        return len(g)  # Current triple count
    
    # Convert all rows
    initial_count = len(g)
    for idx, row in df.iterrows():
        csv_to_rdf(row)
    
    print(f"\n=== RDF Conversion Results ===")
    print(f"Rows processed: {len(df)}")
    print(f"Triples generated: {len(g) - initial_count}")
    
    # Equipment type statistics
    print("\n=== Equipment Type Statistics ===")
    type_counts = df['Type'].value_counts()
    for eq_type, count in type_counts.items():
        print(f"{eq_type}: {count} unit(s)")
    
    # Output in Turtle format (excerpt)
    print("\n=== Turtle Format (Excerpt) ===")
    print(g.serialize(format="turtle")[:600])
    
    # Save to file
    g.serialize(destination="equipment_from_csv.ttl", format="turtle")
    print("\n✓ RDF file saved: equipment_from_csv.ttl")
    

**Output Example:**  
=== Original CSV Data ===  
EquipmentID Type Temperature_K Pressure_bar  
0 R-101 CSTR 350.0 5.0  
1 R-102 PFR 420.0 8.0  
2 HX-201 HeatExchanger 320.0 NaN  
  
=== RDF Conversion Results ===  
Rows processed: 6  
Triples generated: 28  
  
=== Equipment Type Statistics ===  
HeatExchanger: 2 unit(s)  
CSTR: 1 unit(s)  
PFR: 1 unit(s)  
Separator: 1 unit(s)  
Pump: 1 unit(s)  
  
✓ RDF file saved: equipment_from_csv.ttl 

## 3.2 Equipment Connection Relationship Extraction

### Example 2: Relationship Extraction from Flow Data

Automatically extract connection relationships between equipment from material flow data.
    
    
    # ===================================
    # Example 2: Relationship Extraction from Flow Data
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD
    import io
    
    # Flow connection data
    flow_data = """StreamID,SourceEquipment,TargetEquipment,FlowRate_kgh,Composition
    S-001,Feed,R-101,1000.0,Raw Material Mixture
    S-002,R-101,HX-201,980.0,Reaction Product
    S-003,HX-201,SEP-301,975.0,Cooled Product
    S-004,SEP-301,Product,920.0,Product
    S-005,SEP-301,R-101,55.0,Recycle"""
    
    df_flow = pd.read_csv(io.StringIO(flow_data))
    
    print("=== Flow Data ===")
    print(df_flow)
    
    # Create RDF graph
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # ===== Generate triples from flow data =====
    
    for idx, row in df_flow.iterrows():
        # Stream triples
        stream = PROC[row['StreamID']]
        g.add((stream, RDF.type, PROC.Stream))
        g.add((stream, RDFS.label, Literal(row['Composition'], lang='en')))
        g.add((stream, PROC.hasFlowRate,
               Literal(row['FlowRate_kgh'], datatype=XSD.double)))
    
        # Source equipment
        source = PROC[row['SourceEquipment']]
        g.add((source, PROC.hasOutput, stream))
    
        # Target equipment
        target = PROC[row['TargetEquipment']]
        g.add((target, PROC.hasInput, stream))
    
        # Direct connection between equipment
        g.add((source, PROC.connectedTo, target))
    
    print(f"\n=== Triple Generation Results ===")
    print(f"Total streams: {len(df_flow)}")
    print(f"Total triples: {len(g)}")
    
    # ===== Visualize connection relationships =====
    print("\n=== Process Flow Connections ===")
    
    # Get connections via SPARQL query
    query = """
    PREFIX proc: 
    PREFIX rdfs: 
    
    SELECT ?source ?target ?stream ?flowrate ?composition
    WHERE {
        ?source proc:hasOutput ?stream .
        ?target proc:hasInput ?stream .
        ?stream proc:hasFlowRate ?flowrate .
        ?stream rdfs:label ?composition .
    }
    ORDER BY ?source
    """
    
    for row in g.query(query):
        source = str(row.source).split('/')[-1]
        target = str(row.target).split('/')[-1]
        print(f"{source} → {target}: {float(row.flowrate):.0f} kg/h ({row.composition})")
    
    # Recycle loop detection
    print("\n=== Recycle Stream Detection ===")
    recycled = df_flow[df_flow['Composition'].str.contains('Recycle', na=False)]
    for idx, row in recycled.iterrows():
        print(f"✓ {row['SourceEquipment']} → {row['TargetEquipment']} (Recycle)")
    
    g.serialize(destination="process_flow.ttl", format="turtle")
    print("\n✓ Flow graph saved: process_flow.ttl")
    

**Output Example:**  
=== Flow Data ===  
StreamID SourceEquipment TargetEquipment FlowRate_kgh  
0 S-001 Feed R-101 1000.0  
1 S-002 R-101 HX-201 980.0  
2 S-003 HX-201 SEP-301 975.0  
  
=== Triple Generation Results ===  
Total streams: 5  
Total triples: 23  
  
=== Process Flow Connections ===  
Feed → R-101: 1000 kg/h (Raw Material Mixture)  
R-101 → HX-201: 980 kg/h (Reaction Product)  
HX-201 → SEP-301: 975 kg/h (Cooled Product)  
SEP-301 → Product: 920 kg/h (Product)  
SEP-301 → R-101: 55 kg/h (Recycle)  
  
=== Recycle Stream Detection ===  
✓ SEP-301 → R-101 (Recycle)  
  
✓ Flow graph saved: process_flow.ttl 
    
    
    ```mermaid
    graph LR
        Feed[Feed] -->|S-0011000 kg/h| R101[R-101Reactor]
        R101 -->|S-002980 kg/h| HX201[HX-201Cooler]
        HX201 -->|S-003975 kg/h| SEP301[SEP-301Separator]
        SEP301 -->|S-004920 kg/h| Product[Product]
        SEP301 -.->|S-00555 kg/hRecycle| R101
    
        style Feed fill:#e3f2fd
        style Product fill:#e8f5e9
        style R101 fill:#fff3e0
        style HX201 fill:#f3e5f5
        style SEP301 fill:#fce4ec
    ```

### Example 3: Automatic Triple Generation from pandas

Implement a general-purpose conversion function from DataFrame to RDF.
    
    
    # ===================================
    # Example 3: Generic DataFrame to RDF Conversion
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, XSD
    import io
    
    def dataframe_to_rdf(df, namespace_uri, entity_column, type_name=None):
        """Convert pandas DataFrame to RDF graph
    
        Args:
            df: pandas DataFrame
            namespace_uri: Namespace URI
            entity_column: Column name to be used as entity ID
            type_name: Entity class name (defaults to 'Entity' if None)
    
        Returns:
            rdflib.Graph: RDF graph
        """
        g = Graph()
        NS = Namespace(namespace_uri)
        g.bind("data", NS)
    
        type_name = type_name or "Entity"
    
        for idx, row in df.iterrows():
            # Entity URI
            entity_id = str(row[entity_column])
            entity_uri = NS[entity_id]
    
            # Type triple
            g.add((entity_uri, RDF.type, NS[type_name]))
    
            # Add each column as a property
            for col in df.columns:
                if col == entity_column:
                    continue  # Skip ID column
    
                value = row[col]
                if pd.isna(value):
                    continue  # Skip missing values
    
                # Property URI
                prop_uri = NS[col]
    
                # Determine data type and generate appropriate literal
                if isinstance(value, (int, float)):
                    g.add((entity_uri, prop_uri,
                           Literal(value, datatype=XSD.double)))
                elif isinstance(value, bool):
                    g.add((entity_uri, prop_uri,
                           Literal(value, datatype=XSD.boolean)))
                else:
                    g.add((entity_uri, prop_uri, Literal(str(value))))
    
        return g
    
    # ===== Test data =====
    
    sensor_data = """SensorID,Location,Type,Value,Unit,Timestamp
    TE-101,R-101,Temperature,77.5,degC,2025-10-26 10:00:00
    PE-101,R-101,Pressure,5.2,bar,2025-10-26 10:00:00
    FE-201,HX-201,FlowRate,980.0,kg/h,2025-10-26 10:00:00
    TE-201,HX-201,Temperature,45.3,degC,2025-10-26 10:00:00"""
    
    df_sensor = pd.read_csv(io.StringIO(sensor_data))
    
    print("=== Sensor Data ===")
    print(df_sensor)
    
    # RDF conversion
    g_sensor = dataframe_to_rdf(
        df_sensor,
        namespace_uri="http://example.org/sensor/",
        entity_column="SensorID",
        type_name="Sensor"
    )
    
    print(f"\n=== RDF Conversion Results ===")
    print(f"Number of sensors: {len(df_sensor)}")
    print(f"Total triples: {len(g_sensor)}")
    
    # Verify data with SPARQL query
    query = """
    PREFIX data: 
    PREFIX rdf: 
    
    SELECT ?sensor ?location ?type ?value ?unit
    WHERE {
        ?sensor rdf:type data:Sensor .
        ?sensor data:Location ?location .
        ?sensor data:Type ?type .
        ?sensor data:Value ?value .
        ?sensor data:Unit ?unit .
    }
    """
    
    print("\n=== Sensor Information List ===")
    for row in g_sensor.query(query):
        sensor = str(row.sensor).split('/')[-1]
        print(f"{sensor} @ {row.location}: {row.type} = {float(row.value):.1f} {row.unit}")
    
    # Turtle output
    print("\n=== Turtle Format (Excerpt) ===")
    turtle_output = g_sensor.serialize(format="turtle")
    print(turtle_output[:400])
    
    g_sensor.serialize(destination="sensor_data.ttl", format="turtle")
    print("\n✓ Sensor data RDF saved: sensor_data.ttl")
    

**Output Example:**  
=== Sensor Data ===  
SensorID Location Type Value Unit  
0 TE-101 R-101 Temperature 77.5 degC  
1 PE-101 R-101 Pressure 5.2 bar  
2 FE-201 HX-201 FlowRate 980.0 kg/h  
  
=== RDF Conversion Results ===  
Number of sensors: 4  
Total triples: 25  
  
=== Sensor Information List ===  
TE-101 @ R-101: Temperature = 77.5 degC  
PE-101 @ R-101: Pressure = 5.2 bar  
FE-201 @ HX-201: FlowRate = 980.0 kg/h  
TE-201 @ HX-201: Temperature = 45.3 degC  
  
✓ Sensor data RDF saved: sensor_data.ttl 

## 3.3 Knowledge Extraction from P&ID Text

### Example 4: Parsing P&ID Descriptions and Knowledge Extraction

Parse equipment connections from P&ID (Piping and Instrumentation Diagram) text information.
    
    
    # ===================================
    # Example 4: P&ID Description Parsing and Knowledge Extraction
    # ===================================
    
    import re
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS
    
    # P&ID description text (simple DSL format)
    pid_text = """
    # Esterification Plant P&ID
    
    [EQUIPMENT]
    R-101: Type=CSTR, Name=Main Reactor, Temp=350K, Press=5bar, Vol=10m3
    HX-201: Type=HeatExchanger, Name=Cooler, Temp=320K
    HX-202: Type=HeatExchanger, Name=Heater, Temp=450K
    SEP-301: Type=Separator, Name=Distillation Column, Temp=340K, Press=1.5bar
    P-401: Type=Pump, Name=Feed Pump
    
    [CONNECTIONS]
    Feed -> P-401 (S-001, 1000kg/h)
    P-401 -> R-101 (S-002, 1000kg/h)
    R-101 -> HX-201 (S-003, 980kg/h)
    HX-201 -> SEP-301 (S-004, 975kg/h)
    SEP-301 -> Product (S-005, 920kg/h)
    SEP-301 -> HX-202 (S-006, 55kg/h, recycle)
    HX-202 -> R-101 (S-007, 55kg/h)
    """
    
    # ===== Parser functions =====
    
    def parse_equipment_line(line):
        """Parse equipment definition line"""
        match = re.match(r'(\S+):\s*(.+)', line)
        if not match:
            return None
    
        eq_id = match.group(1)
        params_str = match.group(2)
    
        # Extract parameters
        params = {}
        for param in params_str.split(','):
            param = param.strip()
            if '=' in param:
                key, value = param.split('=', 1)
                params[key.strip()] = value.strip()
    
        return eq_id, params
    
    def parse_connection_line(line):
        """Parse connection line"""
        # Example: "Feed -> P-401 (S-001, 1000kg/h)"
        match = re.match(r'(\S+)\s*->\s*(\S+)\s*\(([^)]+)\)', line)
        if not match:
            return None
    
        source = match.group(1)
        target = match.group(2)
        stream_info = match.group(3)
    
        # Extract stream information
        stream_parts = [p.strip() for p in stream_info.split(',')]
        stream_id = stream_parts[0]
        flowrate = stream_parts[1] if len(stream_parts) > 1 else None
        is_recycle = 'recycle' in stream_info.lower()
    
        return source, target, stream_id, flowrate, is_recycle
    
    # ===== Parse P&ID text =====
    
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    lines = pid_text.strip().split('\n')
    section = None
    
    equipment_count = 0
    connection_count = 0
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
    
        if line.startswith('[EQUIPMENT]'):
            section = 'equipment'
            continue
        elif line.startswith('[CONNECTIONS]'):
            section = 'connections'
            continue
    
        if section == 'equipment':
            parsed = parse_equipment_line(line)
            if parsed:
                eq_id, params = parsed
                eq_uri = PROC[eq_id]
    
                # Generate RDF triples
                if 'Type' in params:
                    g.add((eq_uri, RDF.type, PROC[params['Type']]))
                if 'Name' in params:
                    g.add((eq_uri, RDFS.label, Literal(params['Name'], lang='en')))
    
                equipment_count += 1
    
        elif section == 'connections':
            parsed = parse_connection_line(line)
            if parsed:
                source, target, stream_id, flowrate, is_recycle = parsed
    
                # Stream triples
                stream_uri = PROC[stream_id]
                g.add((stream_uri, RDF.type, PROC.Stream))
    
                if flowrate:
                    # Extract numeric value from "1000kg/h"
                    flow_value = re.search(r'(\d+)', flowrate)
                    if flow_value:
                        g.add((stream_uri, PROC.hasFlowRate,
                               Literal(float(flow_value.group(1)))))
    
                # Connection triples
                source_uri = PROC[source]
                target_uri = PROC[target]
                g.add((source_uri, PROC.hasOutput, stream_uri))
                g.add((target_uri, PROC.hasInput, stream_uri))
    
                if is_recycle:
                    g.add((stream_uri, PROC.isRecycle, Literal(True)))
    
                connection_count += 1
    
    print("=== P&ID Parsing Results ===")
    print(f"Equipment count: {equipment_count}")
    print(f"Connection count: {connection_count}")
    print(f"Total triples: {len(g)}")
    
    # Equipment list
    print("\n=== Equipment List ===")
    query_eq = """
    PREFIX proc: 
    PREFIX rdfs: 
    
    SELECT ?eq ?label
    WHERE {
        ?eq rdfs:label ?label .
    }
    """
    for row in g.query(query_eq):
        eq_id = str(row.eq).split('/')[-1]
        print(f"- {eq_id}: {row.label}")
    
    # Recycle streams
    print("\n=== Recycle Streams ===")
    query_recycle = """
    PREFIX proc: 
    
    SELECT ?stream
    WHERE {
        ?stream proc:isRecycle true .
    }
    """
    recycle_streams = list(g.query(query_recycle))
    print(f"Recycle count: {len(recycle_streams)}")
    for row in recycle_streams:
        print(f"✓ {str(row.stream).split('/')[-1]}")
    
    g.serialize(destination="pid_knowledge.ttl", format="turtle")
    print("\n✓ P&ID knowledge graph saved: pid_knowledge.ttl")
    

**Output Example:**  
=== P&ID Parsing Results ===  
Equipment count: 5  
Connection count: 7  
Total triples: 38  
  
=== Equipment List ===  
\- R-101: Main Reactor  
\- HX-201: Cooler  
\- HX-202: Heater  
\- SEP-301: Distillation Column  
\- P-401: Feed Pump  
  
=== Recycle Streams ===  
Recycle count: 1  
✓ S-006  
  
✓ P&ID knowledge graph saved: pid_knowledge.ttl 

**💡 Extending P &ID Data Sources**

In practice, P&IDs are managed by CAD software (AutoCAD Plant 3D, Intergraph SmartPlant, etc.). These tools have XML/JSON export functionality, allowing knowledge extraction using similar methods.

## 3.4 RDF Conversion of Sensor Stream Data

### Example 5: RDF Conversion of Real-Time Sensor Data

Represent time-series sensor data in RDF while preserving temporal information.
    
    
    # ===================================
    # Example 5: RDF Conversion of Sensor Streams
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD
    from datetime import datetime, timedelta
    import numpy as np
    
    # Generate time-series sensor data
    base_time = datetime(2025, 10, 26, 10, 0, 0)
    time_points = [base_time + timedelta(minutes=5*i) for i in range(12)]
    
    # Simulation data (1 hour, 5-minute intervals)
    sensor_stream = pd.DataFrame({
        'Timestamp': time_points,
        'TE-101_degC': 77.5 + np.random.normal(0, 0.5, 12),  # Temperature
        'PE-101_bar': 5.0 + np.random.normal(0, 0.1, 12),    # Pressure
        'FE-101_kgh': 1000 + np.random.normal(0, 10, 12),    # Flow rate
    })
    
    print("=== Sensor Stream Data (Excerpt) ===")
    print(sensor_stream.head(3))
    
    # Create RDF graph
    g = Graph()
    SENSOR = Namespace("http://example.org/sensor/")
    TIME = Namespace("http://www.w3.org/2006/time#")
    g.bind("sensor", SENSOR)
    g.bind("time", TIME)
    
    # ===== RDF conversion of time-series data =====
    
    for idx, row in sensor_stream.iterrows():
        # Timestamp
        timestamp = row['Timestamp']
        instant_uri = SENSOR[f"Instant_{idx}"]
    
        g.add((instant_uri, RDF.type, TIME.Instant))
        g.add((instant_uri, TIME.inXSDDateTime,
               Literal(timestamp.isoformat(), datatype=XSD.dateTime)))
    
        # Each sensor value
        for col in ['TE-101_degC', 'PE-101_bar', 'FE-101_kgh']:
            sensor_id, unit = col.rsplit('_', 1)
            measurement_uri = SENSOR[f"{sensor_id}_M{idx}"]
    
            # Measurement triples
            g.add((measurement_uri, RDF.type, SENSOR.Measurement))
            g.add((measurement_uri, SENSOR.sensor, SENSOR[sensor_id]))
            g.add((measurement_uri, SENSOR.hasTimestamp, instant_uri))
            g.add((measurement_uri, SENSOR.hasValue,
                   Literal(row[col], datatype=XSD.double)))
            g.add((measurement_uri, SENSOR.hasUnit, Literal(unit)))
    
    print(f"\n=== RDF Conversion Results ===")
    print(f"Time points: {len(sensor_stream)}")
    print(f"Total triples: {len(g)}")
    
    # ===== SPARQL query: Temperature statistics =====
    
    query_stats = """
    PREFIX sensor: 
    PREFIX xsd: 
    
    SELECT (AVG(?value) AS ?avgTemp) (MIN(?value) AS ?minTemp) (MAX(?value) AS ?maxTemp)
    WHERE {
        ?measurement sensor:sensor sensor:TE-101 .
        ?measurement sensor:hasValue ?value .
    }
    """
    
    print("\n=== Temperature Sensor TE-101 Statistics (1 hour) ===")
    for row in g.query(query_stats):
        print(f"Average temperature: {float(row.avgTemp):.2f}°C")
        print(f"Minimum temperature: {float(row.minTemp):.2f}°C")
        print(f"Maximum temperature: {float(row.maxTemp):.2f}°C")
    
    # ===== Anomaly detection (threshold-based) =====
    
    query_anomaly = """
    PREFIX sensor: 
    PREFIX time: 
    
    SELECT ?timestamp ?value
    WHERE {
        ?measurement sensor:sensor sensor:TE-101 .
        ?measurement sensor:hasValue ?value .
        ?measurement sensor:hasTimestamp ?instant .
        ?instant time:inXSDDateTime ?timestamp .
        FILTER (?value > 78.5 || ?value < 76.5)
    }
    ORDER BY ?timestamp
    """
    
    print("\n=== Temperature Anomaly Detection (Threshold: 76.5-78.5°C) ===")
    anomalies = list(g.query(query_anomaly))
    print(f"Anomalous data count: {len(anomalies)}")
    for row in anomalies[:3]:  # First 3 entries
        print(f"{row.timestamp}: {float(row.value):.2f}°C")
    
    g.serialize(destination="sensor_stream.ttl", format="turtle")
    print("\n✓ Sensor stream RDF saved: sensor_stream.ttl")
    

**Output Example:**  
=== Sensor Stream Data (Excerpt) ===  
Timestamp TE-101_degC PE-101_bar FE-101_kgh  
0 2025-10-26 10:00:00 77.45 5.02 998.3  
1 2025-10-26 10:05:00 77.58 4.98 1005.1  
2 2025-10-26 10:10:00 77.32 5.03 995.7  
  
=== RDF Conversion Results ===  
Time points: 12  
Total triples: 182  
  
=== Temperature Sensor TE-101 Statistics (1 hour) ===  
Average temperature: 77.48°C  
Minimum temperature: 76.85°C  
Maximum temperature: 78.12°C  
  
=== Temperature Anomaly Detection (Threshold: 76.5-78.5°C) ===  
Anomalous data count: 2  
2025-10-26T10:20:00: 78.67°C  
2025-10-26T10:45:00: 76.32°C  
  
✓ Sensor stream RDF saved: sensor_stream.ttl 

## 3.5 Integration of Historical Data

### Example 6: Integration with Historical Operating Data

Integrate past operational performance data as time-series properties.
    
    
    # ===================================
    # Example 6: Historical Data Integration
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD
    from datetime import datetime, timedelta
    import numpy as np
    
    # Historical operating data (1 month of daily data)
    dates = pd.date_range(start='2025-09-26', end='2025-10-25', freq='D')
    
    historical_data = pd.DataFrame({
        'Date': dates,
        'R101_Conversion': np.random.uniform(0.88, 0.95, len(dates)),  # Conversion
        'R101_Yield': np.random.uniform(0.85, 0.92, len(dates)),       # Yield
        'R101_Temp_avg': np.random.uniform(348, 352, len(dates)),      # Average temperature
        'R101_Uptime_pct': np.random.uniform(95, 100, len(dates)),     # Uptime
    })
    
    print("=== Historical Operating Data (Last 7 days) ===")
    print(historical_data.tail(7)[['Date', 'R101_Conversion', 'R101_Yield']])
    
    # Create RDF graph
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    PERF = Namespace("http://example.org/performance/")
    g.bind("proc", PROC)
    g.bind("perf", PERF)
    
    # Define reactor R-101
    r101 = PROC["R-101"]
    g.add((r101, RDF.type, PROC.Reactor))
    g.add((r101, RDFS.label, Literal("Main Reactor R-101", lang='en')))
    
    # ===== Convert historical data to RDF =====
    
    for idx, row in historical_data.iterrows():
        date = row['Date']
        date_str = date.strftime('%Y-%m-%d')
    
        # Daily performance record
        record_uri = PERF[f"R101_Daily_{date_str}"]
    
        g.add((record_uri, RDF.type, PERF.DailyPerformance))
        g.add((record_uri, PERF.equipment, r101))
        g.add((record_uri, PERF.date,
               Literal(date_str, datatype=XSD.date)))
    
        # Performance metrics
        g.add((record_uri, PERF.conversion,
               Literal(row['R101_Conversion'], datatype=XSD.double)))
        g.add((record_uri, PERF.yieldValue,
               Literal(row['R101_Yield'], datatype=XSD.double)))
        g.add((record_uri, PERF.avgTemperature,
               Literal(row['R101_Temp_avg'], datatype=XSD.double)))
        g.add((record_uri, PERF.uptime,
               Literal(row['R101_Uptime_pct'], datatype=XSD.double)))
    
    print(f"\n=== RDF Conversion Results ===")
    print(f"Daily records: {len(historical_data)}")
    print(f"Total triples: {len(g)}")
    
    # ===== Statistical analysis query =====
    
    query_monthly_stats = """
    PREFIX perf: 
    
    SELECT
        (AVG(?conv) AS ?avgConversion)
        (AVG(?yield) AS ?avgYield)
        (AVG(?temp) AS ?avgTemp)
        (AVG(?uptime) AS ?avgUptime)
        (MIN(?conv) AS ?minConversion)
        (MAX(?conv) AS ?maxConversion)
    WHERE {
        ?record a perf:DailyPerformance .
        ?record perf:conversion ?conv .
        ?record perf:yieldValue ?yield .
        ?record perf:avgTemperature ?temp .
        ?record perf:uptime ?uptime .
    }
    """
    
    print("\n=== Monthly Performance Statistics (R-101) ===")
    for row in g.query(query_monthly_stats):
        print(f"Average conversion: {float(row.avgConversion) * 100:.2f}%")
        print(f"Average yield: {float(row.avgYield) * 100:.2f}%")
        print(f"Average temperature: {float(row.avgTemp):.1f}K ({float(row.avgTemp) - 273.15:.1f}°C)")
        print(f"Average uptime: {float(row.avgUptime):.2f}%")
        print(f"Conversion range: {float(row.minConversion) * 100:.2f}% - {float(row.maxConversion) * 100:.2f}%")
    
    # ===== Low performance day detection =====
    
    query_low_performance = """
    PREFIX perf: 
    
    SELECT ?date ?conv ?yield
    WHERE {
        ?record a perf:DailyPerformance .
        ?record perf:date ?date .
        ?record perf:conversion ?conv .
        ?record perf:yieldValue ?yield .
        FILTER (?conv < 0.90 || ?yield < 0.87)
    }
    ORDER BY ?date
    """
    
    print("\n=== Performance Degradation Days (Conversion<90% or Yield<87%) ===")
    low_perf_days = list(g.query(query_low_performance))
    print(f"Applicable days: {len(low_perf_days)}")
    for row in low_perf_days[:3]:  # First 3 entries
        print(f"{row.date}: Conversion {float(row.conv) * 100:.1f}%, Yield {float(row.yield) * 100:.1f}%")
    
    g.serialize(destination="historical_performance.ttl", format="turtle")
    print("\n✓ Historical performance data RDF saved: historical_performance.ttl")
    

**Output Example:**  
=== Historical Operating Data (Last 7 days) ===  
Date R101_Conversion R101_Yield  
23 2025-10-19 0.9245 0.8932  
24 2025-10-20 0.9012 0.8765  
25 2025-10-21 0.9356 0.9087  
  
=== RDF Conversion Results ===  
Daily records: 30  
Total triples: 152  
  
=== Monthly Performance Statistics (R-101) ===  
Average conversion: 91.48%  
Average yield: 88.72%  
Average temperature: 350.2K (77.0°C)  
Average uptime: 97.45%  
Conversion range: 88.23% - 94.87%  
  
=== Performance Degradation Days (Conversion<90% or Yield<87%) ===  
Applicable days: 4  
2025-09-28: Conversion 89.5%, Yield 86.3%  
2025-10-05: Conversion 88.8%, Yield 85.9%  
2025-10-12: Conversion 89.2%, Yield 86.7%  
  
✓ Historical performance data RDF saved: historical_performance.ttl 

**⚠️ Handling Large-Scale Time-Series Data**

For multi-year second-level data (hundreds of millions of triples), use of triple stores (Apache Jena Fuseki, Virtuoso) is recommended. rdflib is practical for medium-scale data (up to ~1 million triples).

## 3.6 Integrated Knowledge Graph from Multi-Source Data

### Example 7: Building a Complete Integrated Knowledge Graph

Build a comprehensive knowledge graph that integrates all data sources.
    
    
    # ===================================
    # Example 7: Multi-Source Integrated Knowledge Graph
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD, OWL
    from datetime import datetime
    import io
    
    # ===== Define multiple data sources =====
    
    # 1. Equipment master data
    equipment_csv = """EquipmentID,Type,Name,InstallDate,Manufacturer
    R-101,CSTR,Main Reactor,2020-03-15,Mitsubishi Chemical
    HX-201,HeatExchanger,Cooler,2020-04-01,Kobe Steel
    SEP-301,Separator,Distillation Column,2020-05-10,Sumitomo Heavy"""
    
    # 2. Current operating conditions
    operating_csv = """EquipmentID,Temperature_K,Pressure_bar,FlowRate_kgh,Efficiency_pct
    R-101,350.5,5.1,1005.0,92.8
    HX-201,320.2,5.0,980.0,89.5
    SEP-301,340.0,1.5,975.0,95.2"""
    
    # 3. Connection information
    connection_csv = """StreamID,Source,Target,FlowRate_kgh
    S-001,Feed,R-101,1000.0
    S-002,R-101,HX-201,980.0
    S-003,HX-201,SEP-301,975.0
    S-004,SEP-301,Product,920.0"""
    
    # ===== Build integrated RDF graph =====
    
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    MAINT = Namespace("http://example.org/maintenance/")
    g.bind("proc", PROC)
    g.bind("maint", MAINT)
    
    # Load DataFrames
    df_equipment = pd.read_csv(io.StringIO(equipment_csv))
    df_operating = pd.read_csv(io.StringIO(operating_csv))
    df_connection = pd.read_csv(io.StringIO(connection_csv))
    
    print("=== Data Source Integration ===")
    print(f"Equipment master: {len(df_equipment)} items")
    print(f"Operating data: {len(df_operating)} items")
    print(f"Connection data: {len(df_connection)} items")
    
    # ===== 1. Integrate equipment master data =====
    
    for idx, row in df_equipment.iterrows():
        eq_uri = PROC[row['EquipmentID']]
    
        g.add((eq_uri, RDF.type, PROC[row['Type']]))
        g.add((eq_uri, RDFS.label, Literal(row['Name'], lang='en')))
        g.add((eq_uri, MAINT.installDate,
               Literal(row['InstallDate'], datatype=XSD.date)))
        g.add((eq_uri, MAINT.manufacturer, Literal(row['Manufacturer'])))
    
    # ===== 2. Integrate operating condition data =====
    
    current_time = datetime.now()
    
    for idx, row in df_operating.iterrows():
        eq_uri = PROC[row['EquipmentID']]
    
        # Current operating state
        state_uri = PROC[f"{row['EquipmentID']}_State_{current_time.strftime('%Y%m%d')}"]
        g.add((state_uri, RDF.type, PROC.OperatingState))
        g.add((state_uri, PROC.equipment, eq_uri))
        g.add((state_uri, PROC.timestamp,
               Literal(current_time.isoformat(), datatype=XSD.dateTime)))
    
        # Operating parameters
        g.add((state_uri, PROC.temperature,
               Literal(row['Temperature_K'], datatype=XSD.double)))
        g.add((state_uri, PROC.pressure,
               Literal(row['Pressure_bar'], datatype=XSD.double)))
        g.add((state_uri, PROC.flowRate,
               Literal(row['FlowRate_kgh'], datatype=XSD.double)))
        g.add((state_uri, PROC.efficiency,
               Literal(row['Efficiency_pct'], datatype=XSD.double)))
    
    # ===== 3. Integrate connection information =====
    
    for idx, row in df_connection.iterrows():
        stream_uri = PROC[row['StreamID']]
        source_uri = PROC[row['Source']]
        target_uri = PROC[row['Target']]
    
        g.add((stream_uri, RDF.type, PROC.Stream))
        g.add((stream_uri, PROC.flowRate,
               Literal(row['FlowRate_kgh'], datatype=XSD.double)))
    
        g.add((source_uri, PROC.hasOutput, stream_uri))
        g.add((target_uri, PROC.hasInput, stream_uri))
        g.add((source_uri, PROC.connectedTo, target_uri))
    
    print(f"\n=== Integrated Knowledge Graph ===")
    print(f"Total triples: {len(g)}")
    
    # ===== Query integrated data =====
    
    # Query 1: Complete equipment information (master + operating state)
    query_complete = """
    PREFIX proc: 
    PREFIX maint: 
    PREFIX rdfs: 
    
    SELECT ?id ?name ?manufacturer ?temp ?press ?eff
    WHERE {
        ?equipment rdfs:label ?name .
        ?equipment maint:manufacturer ?manufacturer .
    
        ?state proc:equipment ?equipment .
        ?state proc:temperature ?temp .
        ?state proc:pressure ?press .
        ?state proc:efficiency ?eff .
    
        BIND(STRAFTER(STR(?equipment), "#") AS ?id)
    }
    """
    
    print("\n=== Complete Equipment Information (Master + Operating State) ===")
    for row in g.query(query_complete):
        print(f"{row.name} ({row.manufacturer})")
        print(f"  Temperature: {float(row.temp):.1f}K, Pressure: {float(row.press):.1f}bar, Efficiency: {float(row.eff):.1f}%")
    
    # Query 2: Process flow (connection + flow rate)
    query_flow = """
    PREFIX proc: 
    
    SELECT ?source ?target ?flowrate
    WHERE {
        ?source proc:hasOutput ?stream .
        ?target proc:hasInput ?stream .
        ?stream proc:flowRate ?flowrate .
    }
    """
    
    print("\n=== Process Flow (Connection + Flow Rate) ===")
    for row in g.query(query_flow):
        source = str(row.source).split('/')[-1]
        target = str(row.target).split('/')[-1]
        print(f"{source} → {target}: {float(row.flowrate):.0f} kg/h")
    
    # ===== Derive new knowledge through reasoning =====
    
    # Inference rule: Equipment with efficiency >= 90% is "HighPerformance" class
    print("\n=== Inference Results (Efficiency-Based Classification) ===")
    for s, p, o in g.triples((None, PROC.efficiency, None)):
        if float(o) >= 90.0:
            equipment = g.value(s, PROC.equipment)
            g.add((equipment, RDF.type, PROC.HighPerformanceEquipment))
            eq_name = g.value(equipment, RDFS.label)
            print(f"✓ {eq_name}: HighPerformance ({float(o):.1f}%)")
    
    print(f"\nTotal triples (after inference): {len(g)}")
    
    # Save
    g.serialize(destination="integrated_knowledge_graph.ttl", format="turtle")
    print("\n✓ Integrated knowledge graph saved: integrated_knowledge_graph.ttl")
    
    # Also save in OWL format (can be opened with Protégé)
    g.serialize(destination="integrated_knowledge_graph.owl", format="xml")
    print("✓ OWL format saved: integrated_knowledge_graph.owl")
    

**Output Example:**  
=== Data Source Integration ===  
Equipment master: 3 items  
Operating data: 3 items  
Connection data: 4 items  
  
=== Integrated Knowledge Graph ===  
Total triples: 48  
  
=== Complete Equipment Information (Master + Operating State) ===  
Main Reactor (Mitsubishi Chemical)  
Temperature: 350.5K, Pressure: 5.1bar, Efficiency: 92.8%  
Cooler (Kobe Steel)  
Temperature: 320.2K, Pressure: 5.0bar, Efficiency: 89.5%  
Distillation Column (Sumitomo Heavy)  
Temperature: 340.0K, Pressure: 1.5bar, Efficiency: 95.2%  
  
=== Process Flow (Connection + Flow Rate) ===  
Feed → R-101: 1000 kg/h  
R-101 → HX-201: 980 kg/h  
HX-201 → SEP-301: 975 kg/h  
SEP-301 → Product: 920 kg/h  
  
=== Inference Results (Efficiency-Based Classification) ===  
✓ Main Reactor: HighPerformance (92.8%)  
✓ Distillation Column: HighPerformance (95.2%)  
  
Total triples (after inference): 50  
  
✓ Integrated knowledge graph saved: integrated_knowledge_graph.ttl  
✓ OWL format saved: integrated_knowledge_graph.owl 

**✅ Achievements of Integrated Knowledge Graph**

  * **Multi-Source Integration** : Unified equipment master, operating data, and connection information into a single graph
  * **Temporal Information Retention** : Manage current operating states and historical data in time series
  * **Knowledge Expansion Through Reasoning** : Automatically classify new knowledge (high-performance equipment) based on rules
  * **Standard Format Output** : Compatible with other tools (Protégé, GraphDB) via Turtle/OWL formats

## Learning Objectives Verification

After completing this chapter, you will be able to explain and implement the following:

### Basic Understanding

  * ✅ Explain the entity extraction process from CSV data
  * ✅ Understand equipment connection relationship extraction patterns
  * ✅ Know methods for RDF representation of time-series data
  * ✅ Understand challenges and solutions for multi-source data integration

### Practical Skills

  * ✅ Automatically convert pandas DataFrames to RDF graphs
  * ✅ Generate triples for equipment connections from flow data
  * ✅ Parse P&ID text and extract knowledge
  * ✅ Convert sensor stream data to time-series RDF
  * ✅ Integrate historical operating data for performance analysis
  * ✅ Integrate multiple data sources into a single knowledge graph
  * ✅ Perform advanced queries on integrated data with SPARQL

### Applied Capabilities

  * ✅ Plan strategies for RDF conversion of various data sources from real plants
  * ✅ Discover anomalies and performance degradation via SPARQL queries
  * ✅ Automatically derive new knowledge through rule-based reasoning
  * ✅ Select triple stores for large-scale data
  * ✅ Visualize and edit knowledge graphs using external tools like Protégé

## Next Steps

In Chapter 3, we learned comprehensive methods for automatically building knowledge graphs from actual process data. In the next chapter, we will learn advanced SPARQL reasoning, integration with machine learning, and industrial application cases.

**📚 Next Chapter Preview (Chapter 4)**

  * Knowledge reasoning with SPARQL inference engines
  * Integration of knowledge graphs and machine learning
  * Application of Graph Neural Networks (GNN)
  * Process anomaly detection and rule-based diagnosis
  * Industry implementation cases and best practices

## References

  1. Montgomery, D. C. (2019). _Design and Analysis of Experiments_ (9th ed.). Wiley.
  2. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). _Statistics for Experimenters: Design, Innovation, and Discovery_ (2nd ed.). Wiley.
  3. Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). _Process Dynamics and Control_ (4th ed.). Wiley.
  4. McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." _Technometrics_ , 42(1), 55-61.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
