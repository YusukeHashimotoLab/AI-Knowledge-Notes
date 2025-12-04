---
title: "Chapter 1: Fundamentals of Ontology and Semantic Web"
chapter_title: "Chapter 1: Fundamentals of Ontology and Semantic Web"
subtitle: Structuring Process Knowledge with RDF, RDFS, and SPARQL
---

This chapter covers the fundamentals of Fundamentals of Ontology and Semantic Web, which forms the foundation of this area. You will learn concept of RDFS hierarchical structure, Know the basic syntax of SPARQL queries, and role of namespaces.

## 1.1 Fundamentals of RDF (Resource Description Framework)

RDF, the foundational technology of the Semantic Web, expresses information as triples consisting of "Subject," "Predicate," and "Object." This structure enables the description of complex knowledge about chemical processes in a machine-readable format.

**💡 Structure of RDF Triples**

  * **Subject** : The resource being described (e.g., Reactor R-101)
  * **Predicate** : The relationship between resources (e.g., hasTemperature)
  * **Object** : A value or resource (e.g., 350°C)

### Example 1: Building Basic RDF Graphs with rdflib

Express basic information about a chemical reactor using RDF triples.
    
    
    # ===================================
    # Example 1: Basic RDF Graph Construction
    # ===================================
    
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, XSD
    
    # Define namespaces
    PROC = Namespace("http://example.org/process/")
    UNIT = Namespace("http://example.org/unit/")
    
    # Create graph
    g = Graph()
    g.bind("proc", PROC)
    g.bind("unit", UNIT)
    
    # Add triples (Reactor R-101 information)
    reactor = PROC["R-101"]
    
    # Basic attributes
    g.add((reactor, RDF.type, PROC.Reactor))
    g.add((reactor, RDFS.label, Literal("Continuous Stirred Tank Reactor", lang="en")))
    g.add((reactor, PROC.hasTemperature, Literal(350, datatype=XSD.double)))
    g.add((reactor, PROC.hasPressure, Literal(5.0, datatype=XSD.double)))
    g.add((reactor, PROC.hasVolume, Literal(10.0, datatype=XSD.double)))
    g.add((reactor, PROC.unit, UNIT.degC))
    
    # Input streams to reactor
    g.add((reactor, PROC.hasInput, PROC["Stream-01"]))
    g.add((PROC["Stream-01"], RDFS.label, Literal("Raw Material Feed")))
    g.add((PROC["Stream-01"], PROC.flowRate, Literal(100.0, datatype=XSD.double)))
    
    # Output streams from reactor
    g.add((reactor, PROC.hasOutput, PROC["Stream-02"]))
    g.add((PROC["Stream-02"], RDFS.label, Literal("Reaction Product")))
    
    # Serialize in Turtle format (human-readable)
    print("=== Turtle Format ===")
    print(g.serialize(format="turtle"))
    
    # Check number of triples
    print(f"\nTotal triples: {len(g)}")
    
    # Query for specific predicate
    print("\n=== Retrieving Temperature Information ===")
    for s, p, o in g.triples((None, PROC.hasTemperature, None)):
        print(f"{s} temperature: {o}°C")
    

**Output example:**  
=== Turtle Format ===  
@prefix proc: <http://example.org/process/> .  
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .  
  
proc:R-101 a proc:Reactor ;  
rdfs:label "Continuous Stirred Tank Reactor"@en ;  
proc:hasTemperature 350.0 ;  
proc:hasPressure 5.0 .  
  
Total triples: 11  
http://example.org/process/R-101 temperature: 350.0°C 

### Example 2: Converting Between RDF/XML and Turtle Notation

Implement conversion between different serialization formats.
    
    
    # ===================================
    # Example 2: Serialization Format Conversion
    # ===================================
    
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS
    
    # RDF graph for distillation column
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # Distillation column D-201 information
    column = PROC["D-201"]
    g.add((column, RDF.type, PROC.DistillationColumn))
    g.add((column, RDFS.label, Literal("Distillation Column D-201", lang="en")))
    g.add((column, PROC.numberOfTrays, Literal(30)))
    g.add((column, PROC.refluxRatio, Literal(2.5)))
    g.add((column, PROC.feedTray, Literal(15)))
    
    # RDF/XML format
    print("=== RDF/XML Format ===")
    rdfxml = g.serialize(format="xml")
    print(rdfxml)
    
    # Turtle format
    print("\n=== Turtle Format ===")
    turtle = g.serialize(format="turtle")
    print(turtle)
    
    # N-Triples format (simplest)
    print("\n=== N-Triples Format ===")
    ntriples = g.serialize(format="nt")
    print(ntriples)
    
    # JSON-LD format (convenient for Web APIs)
    print("\n=== JSON-LD Format ===")
    jsonld = g.serialize(format="json-ld", indent=2)
    print(jsonld)
    
    # Save to file
    g.serialize(destination="distillation_column.ttl", format="turtle")
    print("\n✓ File saved in Turtle format: distillation_column.ttl")
    
    # Load from file
    g_loaded = Graph()
    g_loaded.parse("distillation_column.ttl", format="turtle")
    print(f"✓ Loading complete: {len(g_loaded)} triples")
    

**Output example:**  
=== Turtle Format ===  
proc:D-201 a proc:DistillationColumn ;  
rdfs:label "Distillation Column D-201"@en ;  
proc:numberOfTrays 30 ;  
proc:refluxRatio 2.5 ;  
proc:feedTray 15 .  
  
✓ File saved in Turtle format  
✓ Loading complete: 5 triples 

## 1.2 Class Hierarchy with RDFS (RDF Schema)

RDFS extends RDF and enables the definition of hierarchical structures for classes and properties. This is an important concept for building classification systems for chemical process equipment.

### Example 3: Defining RDFS Hierarchical Structure

Define class hierarchy and properties for chemical equipment.
    
    
    # ===================================
    # Example 3: Defining RDFS Hierarchical Structure
    # ===================================
    
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS
    
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # ===== Define Class Hierarchy =====
    
    # Top-level class: ProcessEquipment
    g.add((PROC.ProcessEquipment, RDF.type, RDFS.Class))
    g.add((PROC.ProcessEquipment, RDFS.label, Literal("Process Equipment")))
    
    # Subclass definitions
    # Reactor
    g.add((PROC.Reactor, RDF.type, RDFS.Class))
    g.add((PROC.Reactor, RDFS.subClassOf, PROC.ProcessEquipment))
    g.add((PROC.Reactor, RDFS.label, Literal("Reactor")))
    
    # HeatExchanger
    g.add((PROC.HeatExchanger, RDF.type, RDFS.Class))
    g.add((PROC.HeatExchanger, RDFS.subClassOf, PROC.ProcessEquipment))
    g.add((PROC.HeatExchanger, RDFS.label, Literal("Heat Exchanger")))
    
    # Separator
    g.add((PROC.Separator, RDF.type, RDFS.Class))
    g.add((PROC.Separator, RDFS.subClassOf, PROC.ProcessEquipment))
    g.add((PROC.Separator, RDFS.label, Literal("Separator")))
    
    # Further subclass: DistillationColumn
    g.add((PROC.DistillationColumn, RDF.type, RDFS.Class))
    g.add((PROC.DistillationColumn, RDFS.subClassOf, PROC.Separator))
    g.add((PROC.DistillationColumn, RDFS.label, Literal("Distillation Column")))
    
    # ===== Property Definitions =====
    
    # hasInput
    g.add((PROC.hasInput, RDF.type, RDF.Property))
    g.add((PROC.hasInput, RDFS.domain, PROC.ProcessEquipment))
    g.add((PROC.hasInput, RDFS.range, PROC.Stream))
    g.add((PROC.hasInput, RDFS.label, Literal("Input")))
    
    # hasOutput
    g.add((PROC.hasOutput, RDF.type, RDF.Property))
    g.add((PROC.hasOutput, RDFS.domain, PROC.ProcessEquipment))
    g.add((PROC.hasOutput, RDFS.range, PROC.Stream))
    g.add((PROC.hasOutput, RDFS.label, Literal("Output")))
    
    # hasTemperature
    g.add((PROC.hasTemperature, RDF.type, RDF.Property))
    g.add((PROC.hasTemperature, RDFS.domain, PROC.ProcessEquipment))
    g.add((PROC.hasTemperature, RDFS.label, Literal("Temperature")))
    
    # ===== Create Instances =====
    reactor = PROC["R-101"]
    g.add((reactor, RDF.type, PROC.Reactor))
    g.add((reactor, RDFS.label, Literal("CSTR Reactor")))
    
    # Visualize class hierarchy
    print("=== Class Hierarchy ===")
    for subclass in g.subjects(RDFS.subClassOf, None):
        for superclass in g.objects(subclass, RDFS.subClassOf):
            sub_label = g.value(subclass, RDFS.label)
            super_label = g.value(superclass, RDFS.label)
            print(f"{sub_label} → {super_label}")
    
    # List of properties
    print("\n=== Property List ===")
    for prop in g.subjects(RDF.type, RDF.Property):
        label = g.value(prop, RDFS.label)
        domain = g.value(prop, RDFS.domain)
        range_val = g.value(prop, RDFS.range)
        print(f"- {label}: {domain} → {range_val}")
    
    print(f"\nTotal triples: {len(g)}")
    print(g.serialize(format="turtle"))
    

**Output example:**  
=== Class Hierarchy ===  
Reactor → Process Equipment  
Heat Exchanger → Process Equipment  
Separator → Process Equipment  
Distillation Column → Separator  
  
=== Property List ===  
\- Input: ProcessEquipment → Stream  
\- Output: ProcessEquipment → Stream  
\- Temperature: ProcessEquipment → (undefined)  
  
Total triples: 28 

## 1.3 SPARQL Query Fundamentals

SPARQL is a query language for RDF graphs. With SQL-like syntax, it enables complex pattern matching and knowledge extraction.

### Example 4: SPARQL SELECT Queries

Extract information about process equipment.
    
    
    # ===================================
    # Example 4: SPARQL SELECT Queries
    # ===================================
    
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD
    
    # Create sample data
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # Multiple reactor data
    reactors = [
        ("R-101", "CSTR Reactor", 350, 5.0, 100),
        ("R-102", "PFR Reactor", 400, 8.0, 150),
        ("R-103", "Batch Reactor", 320, 3.0, 80),
    ]
    
    for id, label, temp, press, flow in reactors:
        reactor = PROC[id]
        g.add((reactor, RDF.type, PROC.Reactor))
        g.add((reactor, RDFS.label, Literal(label, lang="en")))
        g.add((reactor, PROC.hasTemperature, Literal(temp, datatype=XSD.double)))
        g.add((reactor, PROC.hasPressure, Literal(press, datatype=XSD.double)))
        g.add((reactor, PROC.flowRate, Literal(flow, datatype=XSD.double)))
    
    # ===== Execute SPARQL Queries =====
    
    # Query 1: Basic information for all reactors
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
    
    print("=== Query 1: All Reactors (Descending Temperature) ===")
    results1 = g.query(query1)
    for row in results1:
        print(f"{row.label}: {row.temp}°C, {row.press}bar")
    
    # Query 2: Conditional search (temperature > 340°C and pressure > 4bar)
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
    
    print("\n=== Query 2: High Temperature High Pressure Reactors ===")
    results2 = g.query(query2)
    for row in results2:
        print(f"{row.label}: {row.temp}°C, {row.press}bar")
    
    # Query 3: Aggregation (average temperature, maximum pressure)
    query3 = """
    PREFIX proc: 
    
    SELECT (AVG(?temp) AS ?avgTemp) (MAX(?press) AS ?maxPress) (COUNT(?reactor) AS ?count)
    WHERE {
        ?reactor a proc:Reactor .
        ?reactor proc:hasTemperature ?temp .
        ?reactor proc:hasPressure ?press .
    }
    """
    
    print("\n=== Query 3: Statistical Information ===")
    results3 = g.query(query3)
    for row in results3:
        print(f"Reactor count: {row.count}")
        print(f"Average temperature: {float(row.avgTemp):.1f}°C")
        print(f"Maximum pressure: {float(row.maxPress)}bar")
    

**Output example:**  
=== Query 1: All Reactors (Descending Temperature) ===  
PFR Reactor: 400.0°C, 8.0bar  
CSTR Reactor: 350.0°C, 5.0bar  
Batch Reactor: 320.0°C, 3.0bar  
  
=== Query 2: High Temperature High Pressure Reactors ===  
PFR Reactor: 400.0°C, 8.0bar  
CSTR Reactor: 350.0°C, 5.0bar  
  
=== Query 3: Statistical Information ===  
Reactor count: 3  
Average temperature: 356.7°C  
Maximum pressure: 8.0bar 

## 1.4 Knowledge Representation of Chemical Processes

### Example 5: Building Equipment Connection Graphs

Represent process flow diagrams as RDF graphs.
    
    
    # ===================================
    # Example 5: RDF Representation of Process Flow Diagrams
    # ===================================
    
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS
    
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # ===== Process Flow: Feed → Reactor → HeatExchanger → Separator =====
    
    # 1. Feed Tank
    feed_tank = PROC["TK-001"]
    g.add((feed_tank, RDF.type, PROC.StorageTank))
    g.add((feed_tank, RDFS.label, Literal("Feed Tank")))
    g.add((feed_tank, PROC.capacity, Literal(50000)))  # liters
    
    # 2. Reactor
    reactor = PROC["R-101"]
    g.add((reactor, RDF.type, PROC.Reactor))
    g.add((reactor, RDFS.label, Literal("Main Reactor")))
    
    # 3. Heat Exchanger
    hx = PROC["HX-201"]
    g.add((hx, RDF.type, PROC.HeatExchanger))
    g.add((hx, RDFS.label, Literal("Cooler")))
    
    # 4. Separator
    separator = PROC["SEP-301"]
    g.add((separator, RDF.type, PROC.Separator))
    g.add((separator, RDFS.label, Literal("Vapor-Liquid Separator")))
    
    # ===== Material Streams =====
    s1 = PROC["S-001"]  # Feed → Reactor
    s2 = PROC["S-002"]  # Reactor → HX
    s3 = PROC["S-003"]  # HX → Separator
    
    for stream in [s1, s2, s3]:
        g.add((stream, RDF.type, PROC.Stream))
    
    # ===== Equipment Connections =====
    # Feed Tank → Reactor
    g.add((feed_tank, PROC.hasOutput, s1))
    g.add((reactor, PROC.hasInput, s1))
    
    # Reactor → Heat Exchanger
    g.add((reactor, PROC.hasOutput, s2))
    g.add((hx, PROC.hasInput, s2))
    
    # Heat Exchanger → Separator
    g.add((hx, PROC.hasOutput, s3))
    g.add((separator, PROC.hasInput, s3))
    
    # ===== Visualize Process Flow =====
    print("=== Process Flow Diagram ===\n")
    
    # Equipment list
    print("Equipment list:")
    for eq in g.subjects(RDF.type, None):
        if eq != PROC.Stream:
            eq_type = g.value(eq, RDF.type)
            eq_label = g.value(eq, RDFS.label)
            if eq_type and eq_type != RDFS.Class:
                print(f"  - {eq_label} ({eq_type.split('/')[-1]})")
    
    # Connection relationships
    print("\nConnection relationships:")
    for s in g.subjects(PROC.hasOutput, None):
        source_label = g.value(s, RDFS.label)
        for stream in g.objects(s, PROC.hasOutput):
            for target in g.subjects(PROC.hasInput, stream):
                target_label = g.value(target, RDFS.label)
                print(f"  {source_label} → {target_label}")
    
    # Path exploration with SPARQL
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
    
    print("\n=== SPARQL Query Results (Connection Relationships) ===")
    for row in g.query(query):
        print(f"{row.source_label} ⟶ {row.target_label}")
    

**Output example:**  
=== Process Flow Diagram ===  
  
Equipment list:  
\- Feed Tank (StorageTank)  
\- Main Reactor (Reactor)  
\- Cooler (HeatExchanger)  
\- Vapor-Liquid Separator (Separator)  
  
Connection relationships:  
Feed Tank → Main Reactor  
Main Reactor → Cooler  
Cooler → Vapor-Liquid Separator 

**💡 Practical Implications**

This RDF graph structure enables digitization of P&ID (Piping and Instrumentation Diagram) information and management in machine-readable format. Adding and modifying equipment can be handled flexibly.

## 1.5 Representation of Substances and Properties

### Example 6: RDF Model for Chemical Substances and Properties

Structure chemical substance property data using RDF.
    
    
    # ===================================
    # Example 6: RDF Model for Chemical Substances and Properties
    # ===================================
    
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD
    
    g = Graph()
    CHEM = Namespace("http://example.org/chemistry/")
    PROP = Namespace("http://example.org/property/")
    g.bind("chem", CHEM)
    g.bind("prop", PROP)
    
    # ===== Define Chemical Substances =====
    
    # Ethanol
    ethanol = CHEM["Ethanol"]
    g.add((ethanol, RDF.type, CHEM.Chemical))
    g.add((ethanol, RDFS.label, Literal("Ethanol", lang="en")))
    g.add((ethanol, CHEM.formula, Literal("C2H5OH")))
    g.add((ethanol, CHEM.cas, Literal("64-17-5")))
    g.add((ethanol, CHEM.smiles, Literal("CCO")))
    
    # Property data
    g.add((ethanol, PROP.molecularWeight, Literal(46.07, datatype=XSD.double)))
    g.add((ethanol, PROP.boilingPoint, Literal(78.37, datatype=XSD.double)))
    g.add((ethanol, PROP.meltingPoint, Literal(-114.1, datatype=XSD.double)))
    g.add((ethanol, PROP.density, Literal(0.789, datatype=XSD.double)))
    
    # Water
    water = CHEM["Water"]
    g.add((water, RDF.type, CHEM.Chemical))
    g.add((water, RDFS.label, Literal("Water", lang="en")))
    g.add((water, CHEM.formula, Literal("H2O")))
    g.add((water, CHEM.cas, Literal("7732-18-5")))
    g.add((water, PROP.molecularWeight, Literal(18.015, datatype=XSD.double)))
    g.add((water, PROP.boilingPoint, Literal(100.0, datatype=XSD.double)))
    g.add((water, PROP.meltingPoint, Literal(0.0, datatype=XSD.double)))
    g.add((water, PROP.density, Literal(1.0, datatype=XSD.double)))
    
    # ===== Representation of Mixtures =====
    mixture = CHEM["EthanolWaterMixture"]
    g.add((mixture, RDF.type, CHEM.Mixture))
    g.add((mixture, RDFS.label, Literal("Ethanol-Water Solution")))
    g.add((mixture, CHEM.contains, ethanol))
    g.add((mixture, CHEM.contains, water))
    g.add((mixture, CHEM.composition, Literal("50% vol/vol")))
    
    # ===== SPARQL Query: Substances with boiling point ≤ 80°C =====
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
    
    print("=== Substances with Boiling Point ≤ 80°C ===")
    for row in g.query(query):
        print(f"{row.name} ({row.formula}): Boiling point {row.bp}°C")
    
    # Molecular weight comparison
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
    
    print("\n=== By Molecular Weight ===")
    for row in g.query(query2):
        print(f"{row.name}: {float(row.mw):.2f} g/mol")
    
    # Mixture components
    print("\n=== Mixture Composition ===")
    for component in g.objects(mixture, CHEM.contains):
        label = g.value(component, RDFS.label)
        print(f"- {label}")
    

**Output example:**  
=== Substances with Boiling Point ≤ 80°C ===  
Ethanol (C2H5OH): Boiling point 78.37°C  
  
=== By Molecular Weight ===  
Ethanol: 46.07 g/mol  
Water: 18.02 g/mol  
  
=== Mixture Composition ===  
\- Ethanol  
\- Water 

## 1.6 Namespace and URI Management

### Example 7: Integrated Management of Multiple Namespaces

Implement namespace management when integrating different ontologies.
    
    
    # ===================================
    # Example 7: Integrated Namespace Management
    # ===================================
    
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, OWL, SKOS, DCTERMS
    
    # Create graph and namespace bindings
    g = Graph()
    
    # Standard namespaces
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    
    # Custom namespaces
    PROC = Namespace("http://example.org/process/")
    CHEM = Namespace("http://example.org/chemistry/")
    SENSOR = Namespace("http://example.org/sensor/")
    UNIT = Namespace("http://example.org/unit/")
    
    g.bind("proc", PROC)
    g.bind("chem", CHEM)
    g.bind("sensor", SENSOR)
    g.bind("unit", UNIT)
    
    # ===== Ontology Metadata =====
    ontology_uri = PROC["ontology"]
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, DCTERMS.title, Literal("Process Ontology", lang="en")))
    g.add((ontology_uri, DCTERMS.creator, Literal("Hashimoto Lab")))
    g.add((ontology_uri, DCTERMS.created, Literal("2025-10-26")))
    g.add((ontology_uri, OWL.versionInfo, Literal("1.0")))
    
    # ===== Data Using Multiple Namespaces =====
    
    # Temperature sensor
    temp_sensor = SENSOR["TE-101"]
    g.add((temp_sensor, RDF.type, SENSOR.TemperatureSensor))
    g.add((temp_sensor, RDFS.label, Literal("Temperature Sensor TE-101")))
    g.add((temp_sensor, SENSOR.measuredProperty, PROC.Temperature))
    g.add((temp_sensor, SENSOR.unit, UNIT.degC))
    g.add((temp_sensor, SENSOR.installedAt, PROC["R-101"]))
    
    # Reactor R-101
    reactor = PROC["R-101"]
    g.add((reactor, RDF.type, PROC.Reactor))
    g.add((reactor, RDFS.label, Literal("Main Reactor")))
    g.add((reactor, PROC.processes, CHEM["EsterificationReaction"]))
    
    # Chemical reaction
    reaction = CHEM["EsterificationReaction"]
    g.add((reaction, RDF.type, CHEM.ChemicalReaction))
    g.add((reaction, RDFS.label, Literal("Esterification Reaction")))
    g.add((reaction, SKOS.definition, Literal("A reaction that produces esters from alcohols and carboxylic acids")))
    
    # ===== Verify Namespaces =====
    print("=== Bound Namespaces ===")
    for prefix, namespace in g.namespaces():
        print(f"{prefix}: {namespace}")
    
    # Verify URI construction
    print("\n=== URI Construction Examples ===")
    print(f"Reactor URI: {reactor}")
    print(f"Sensor URI: {temp_sensor}")
    print(f"Reaction URI: {reaction}")
    
    # Triple count by namespace
    print("\n=== Triple Count by Namespace ===")
    namespace_counts = {}
    for s, p, o in g:
        # Count namespace of subject
        ns = str(s).rsplit('/', 1)[0] + '/'
        namespace_counts[ns] = namespace_counts.get(ns, 0) + 1
    
    for ns, count in sorted(namespace_counts.items(), key=lambda x: x[1], reverse=True):
        # Reverse lookup prefix from namespace
        prefix = None
        for p, n in g.namespaces():
            if str(n) == ns:
                prefix = p
                break
        print(f"{prefix or 'unknown'}: {count} triples")
    
    # Output in Turtle format (namespaces are organized)
    print("\n=== Turtle Format (excerpt) ===")
    print(g.serialize(format="turtle")[:800])
    

**Output example:**  
=== Bound Namespaces ===  
rdf: http://www.w3.org/1999/02/22-rdf-syntax-ns#  
rdfs: http://www.w3.org/2000/01/rdf-schema#  
proc: http://example.org/process/  
chem: http://example.org/chemistry/  
sensor: http://example.org/sensor/  
unit: http://example.org/unit/  
  
=== URI Construction Examples ===  
Reactor URI: http://example.org/process/R-101  
Sensor URI: http://example.org/sensor/TE-101  
Reaction URI: http://example.org/chemistry/EsterificationReaction  
  
=== Triple Count by Namespace ===  
sensor: 5 triples  
proc: 4 triples  
chem: 2 triples 

**✅ Best Practices**

  * **Consistent Namespace URIs** : Use persistent URIs that include organizational domain
  * **Leverage Standard Ontologies** : Actively use existing standards like Dublin Core and SKOS
  * **Version Management** : Record ontology versions with owl:versionInfo

## Learning Objectives Review

Upon completing this chapter, you will be able to explain and implement the following:

### Basic Understanding

  * ✅ Explain the structure of RDF triples (Subject-Predicate-Object)
  * ✅ Understand the concept of RDFS hierarchical structure
  * ✅ Know the basic syntax of SPARQL queries
  * ✅ Understand the role of namespaces and URIs

### Practical Skills

  * ✅ Create and manipulate RDF graphs with rdflib
  * ✅ Serialize RDF in Turtle/RDF-XML formats
  * ✅ Write SPARQL filtering and aggregation queries
  * ✅ Represent chemical process flow diagrams using RDF
  * ✅ Manage multiple namespaces in an integrated manner

### Application Ability

  * ✅ Convert P&ID information to RDF graphs
  * ✅ Build chemical substance property databases using RDF
  * ✅ Design hierarchical classification of process equipment

## Next Steps

In Chapter 1, we learned the fundamentals of Semantic Web technology with RDF/RDFS and SPARQL queries. In the next chapter, we will learn advanced process ontology design using OWL (Web Ontology Language) and knowledge modeling that enables reasoning.

**📚 Next Chapter Preview (Chapter 2)**

  * Defining OWL classes and properties
  * Cardinality constraints and value restrictions
  * Complete ontology design for process equipment
  * Implementation using owlready2

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
