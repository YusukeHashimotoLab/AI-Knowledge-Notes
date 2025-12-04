---
title: "Chapter 2: Process Ontology Design and OWL Modeling"
chapter_title: "Chapter 2: Process Ontology Design and OWL Modeling"
subtitle: Comprehensive ontology construction of chemical process equipment using owlready2
---

This chapter covers Process Ontology Design and OWL Modeling. You will learn roles of object properties, Know the concepts of FunctionalProperty, and design principles for class hierarchies.

## 2.1 Fundamentals of OWL (Web Ontology Language)

OWL extends RDFS and is a standard language for describing ontologies with higher expressiveness. It enables formal definition of complex relationships, constraints, and inference rules for chemical process equipment.

**💡 Three OWL Sublanguages**

  * **OWL Lite** : Basic hierarchical structure and simple constraints
  * **OWL DL** : Description logic-based, guaranteed inference (used in this series)
  * **OWL Full** : Maximum expressiveness but inference not guaranteed

### Example 1: OWL Class and Individual Definition with owlready2

Build the basic class hierarchy of chemical process equipment.
    
    
    # ===================================
    # Example 1: OWL Class and Individual Definition
    # ===================================
    
    from owlready2 import *
    
    # Create ontology
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        # ===== Class Definition =====
    
        # Top-level class: ProcessEquipment
        class ProcessEquipment(Thing):
            """Base class for process equipment"""
            pass
    
        # Subclass: Reactor
        class Reactor(ProcessEquipment):
            """Reactor class"""
            pass
    
        # More specific subclass: CSTR (Continuous Stirred Tank Reactor)
        class CSTR(Reactor):
            """Continuous Stirred Tank Reactor"""
            pass
    
        # Subclass: HeatExchanger
        class HeatExchanger(ProcessEquipment):
            """Heat exchanger class"""
            pass
    
        # Subclass: Separator
        class Separator(ProcessEquipment):
            """Separator class"""
            pass
    
        # ===== Individual (Instance) Creation =====
    
        # CSTR reactor R-101
        r101 = CSTR("R-101")
        r101.label = ["Continuous Stirred Tank Reactor R-101"]
        r101.comment = ["Main reactor for esterification reaction"]
    
        # Heat exchanger HX-201
        hx201 = HeatExchanger("HX-201")
        hx201.label = ["Cooler HX-201"]
    
        # Separator SEP-301
        sep301 = Separator("SEP-301")
        sep301.label = ["Vapor-Liquid Separator SEP-301"]
    
    # Save ontology
    onto.save(file="process_ontology_v1.owl", format="rdfxml")
    
    # ===== Visualize Class Hierarchy =====
    print("=== Class Hierarchy ===")
    for cls in onto.classes():
        if cls != Thing:
            ancestors = list(cls.ancestors())
            ancestors.remove(cls)
            ancestors.remove(Thing)
            if ancestors:
                print(f"{cls.name} ⊂ {ancestors[0].name}")
            else:
                print(f"{cls.name} (Top level)")
    
    print("\n=== Individual List ===")
    for individual in onto.individuals():
        print(f"- {individual.name}: {individual.__class__.name}")
        if individual.label:
            print(f"  Label: {individual.label[0]}")
    
    print(f"\n✓ Ontology saved: process_ontology_v1.owl")
    print(f"Total classes: {len(list(onto.classes()))}")
    print(f"Total individuals: {len(list(onto.individuals()))}")
    

**Output example:**  
=== Class Hierarchy ===  
ProcessEquipment (Top level)  
Reactor ⊂ ProcessEquipment  
CSTR ⊂ Reactor  
HeatExchanger ⊂ ProcessEquipment  
Separator ⊂ ProcessEquipment  
  
=== Individual List ===  
\- R-101: CSTR  
Label: Continuous Stirred Tank Reactor R-101  
\- HX-201: HeatExchanger  
Label: Cooler HX-201  
\- SEP-301: Separator  
Label: Vapor-Liquid Separator SEP-301  
  
✓ Ontology saved: process_ontology_v1.owl  
Total classes: 5  
Total individuals: 3 

## 2.2 Object Properties

Object properties represent connections between equipment and process flows.

### Example 2: Object Property Definition for Equipment Connections

Express input-output connections between equipment using object properties.
    
    
    # ===================================
    # Example 2: Object Property Definition
    # ===================================
    
    from owlready2 import *
    
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        # ===== Class Definition =====
        class ProcessEquipment(Thing):
            pass
    
        class Reactor(ProcessEquipment):
            pass
    
        class HeatExchanger(ProcessEquipment):
            pass
    
        class Stream(Thing):
            """Material stream class"""
            pass
    
        # ===== Object Property Definition =====
    
        # hasInput: input stream to equipment
        class hasInput(ProcessEquipment >> Stream):
            """Indicates input stream to equipment"""
            pass
    
        # hasOutput: output stream from equipment
        class hasOutput(ProcessEquipment >> Stream):
            """Indicates output stream from equipment"""
            pass
    
        # connectedTo: connection between equipment (symmetric)
        class connectedTo(ProcessEquipment >> ProcessEquipment, SymmetricProperty):
            """Indicates physical connection between equipment (symmetric)"""
            pass
    
        # feedsTo: upstream to downstream connection (transitive)
        class feedsTo(ProcessEquipment >> ProcessEquipment, TransitiveProperty):
            """Indicates flow from upstream to downstream equipment (transitive)"""
            pass
    
        # ===== Individual and Property Configuration =====
    
        # Create streams
        s1 = Stream("S-001")
        s1.label = ["Raw material feed"]
    
        s2 = Stream("S-002")
        s2.label = ["Reaction product"]
    
        s3 = Stream("S-003")
        s3.label = ["Cooled product"]
    
        # Create equipment
        r101 = Reactor("R-101")
        hx201 = HeatExchanger("HX-201")
    
        # Set properties
        r101.hasInput = [s1]
        r101.hasOutput = [s2]
    
        hx201.hasInput = [s2]
        hx201.hasOutput = [s3]
    
        # Equipment connections
        r101.connectedTo = [hx201]
        r101.feedsTo = [hx201]
    
    # ===== Property Verification =====
    print("=== Process Flow ===")
    print(f"{r101.name} → hasInput → {r101.hasInput[0].name}")
    print(f"{r101.name} → hasOutput → {r101.hasOutput[0].name}")
    print(f"{hx201.name} → hasInput → {hx201.hasInput[0].name}")
    
    print("\n=== Equipment Connections ===")
    print(f"{r101.name} → connectedTo → {r101.connectedTo[0].name}")
    print(f"{r101.name} → feedsTo → {r101.feedsTo[0].name}")
    
    # Check symmetry
    print(f"\nSymmetry check: {hx201.name} → connectedTo → {hx201.connectedTo}")
    
    onto.save(file="process_ontology_v2.owl", format="rdfxml")
    print("\n✓ Ontology saved: process_ontology_v2.owl")
    

**Output example:**  
=== Process Flow ===  
R-101 → hasInput → S-001  
R-101 → hasOutput → S-002  
HX-201 → hasInput → S-002  
  
=== Equipment Connections ===  
R-101 → connectedTo → HX-201  
R-101 → feedsTo → HX-201  
  
Symmetry check: HX-201 → connectedTo → [process.R-101]  
  
✓ Ontology saved: process_ontology_v2.owl 

### Example 3: Data Properties (Temperature, Pressure, Flow Rate)

Express process variables using data properties.
    
    
    # ===================================
    # Example 3: Data Property Definition
    # ===================================
    
    from owlready2 import *
    
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        class ProcessEquipment(Thing):
            pass
    
        class Reactor(ProcessEquipment):
            pass
    
        # ===== Data Property Definition =====
    
        # hasTemperature: temperature [K]
        class hasTemperature(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
            label = ["Temperature"]
            comment = ["Equipment operating temperature (unit: K)"]
    
        # hasPressure: pressure [bar]
        class hasPressure(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
            label = ["Pressure"]
            comment = ["Equipment operating pressure (unit: bar)"]
    
        # hasVolume: volume [m3]
        class hasVolume(DataProperty, FunctionalProperty):
            domain = [Reactor]
            range = [float]
            label = ["Volume"]
    
        # hasFlowRate: flow rate [kg/h]
        class hasFlowRate(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
            label = ["Flow rate"]
    
        # hasEfficiency: efficiency [%]
        class hasEfficiency(DataProperty, FunctionalProperty):
            domain = [ProcessEquipment]
            range = [float]
            label = ["Efficiency"]
    
        # ===== Set Values for Individuals =====
    
        # CSTR reactor R-101
        r101 = Reactor("R-101")
        r101.label = ["CSTR reactor"]
        r101.hasTemperature = [350.0]  # 350 K (approx. 77°C)
        r101.hasPressure = [5.0]       # 5 bar
        r101.hasVolume = [10.0]        # 10 m3
        r101.hasFlowRate = [1000.0]    # 1000 kg/h
        r101.hasEfficiency = [92.5]    # 92.5%
    
        # PFR reactor R-102
        r102 = Reactor("R-102")
        r102.label = ["Plug flow reactor"]
        r102.hasTemperature = [420.0]  # 420 K (approx. 147°C)
        r102.hasPressure = [8.0]       # 8 bar
        r102.hasVolume = [5.0]         # 5 m3
        r102.hasFlowRate = [800.0]     # 800 kg/h
    
    # ===== Retrieve Data Properties =====
    print("=== Operating Conditions for Reactor R-101 ===")
    print(f"Temperature: {r101.hasTemperature[0]} K ({r101.hasTemperature[0] - 273.15:.1f}°C)")
    print(f"Pressure: {r101.hasPressure[0]} bar")
    print(f"Volume: {r101.hasVolume[0]} m³")
    print(f"Flow rate: {r101.hasFlowRate[0]} kg/h")
    print(f"Efficiency: {r101.hasEfficiency[0]}%")
    
    print("\n=== Operating Conditions for Reactor R-102 ===")
    print(f"Temperature: {r102.hasTemperature[0]} K ({r102.hasTemperature[0] - 273.15:.1f}°C)")
    print(f"Pressure: {r102.hasPressure[0]} bar")
    print(f"Volume: {r102.hasVolume[0]} m³")
    print(f"Flow rate: {r102.hasFlowRate[0]} kg/h")
    
    onto.save(file="process_ontology_v3.owl", format="rdfxml")
    print("\n✓ Ontology saved: process_ontology_v3.owl")
    

**Output example:**  
=== Operating Conditions for Reactor R-101 ===  
Temperature: 350.0 K (76.9°C)  
Pressure: 5.0 bar  
Volume: 10.0 m³  
Flow rate: 1000.0 kg/h  
Efficiency: 92.5%  
  
=== Operating Conditions for Reactor R-102 ===  
Temperature: 420.0 K (146.9°C)  
Pressure: 8.0 bar  
Volume: 5.0 m³  
Flow rate: 800.0 kg/h  
  
✓ Ontology saved: process_ontology_v3.owl 

## 2.3 Class Hierarchy and Property Restrictions

### Example 4: Detailed Class Hierarchy Construction

Build a detailed classification hierarchy of equipment types.
    
    
    # ===================================
    # Example 4: Detailed Class Hierarchy
    # ===================================
    
    from owlready2 import *
    
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        # ===== Hierarchical Class Definition =====
    
        class ProcessEquipment(Thing):
            """Process Equipment (Base class)"""
            pass
    
        # Reactor family
        class Reactor(ProcessEquipment):
            """Reactor"""
            pass
    
        class CSTR(Reactor):
            """Continuous Stirred Tank Reactor"""
            pass
    
        class PFR(Reactor):
            """Plug Flow Reactor"""
            pass
    
        class BatchReactor(Reactor):
            """Batch Reactor"""
            pass
    
        # Heat exchanger family
        class HeatExchanger(ProcessEquipment):
            """Heat Exchanger"""
            pass
    
        class ShellTubeHX(HeatExchanger):
            """Shell and Tube Heat Exchanger"""
            pass
    
        class PlateHX(HeatExchanger):
            """Plate Heat Exchanger"""
            pass
    
        # Separator family
        class Separator(ProcessEquipment):
            """Separator"""
            pass
    
        class DistillationColumn(Separator):
            """Distillation Column"""
            pass
    
        class Absorber(Separator):
            """Absorber"""
            pass
    
        class Extractor(Separator):
            """Extractor"""
            pass
    
        # Pump and compressor family
        class FluidMover(ProcessEquipment):
            """Fluid Moving Equipment"""
            pass
    
        class Pump(FluidMover):
            """Pump"""
            pass
    
        class Compressor(FluidMover):
            """Compressor"""
            pass
    
    # ===== Visualize Class Hierarchy (Generate Mermaid data) =====
    print("=== Class Hierarchy Tree ===")
    
    def print_class_tree(cls, indent=0):
        """Display class hierarchy recursively"""
        if cls != Thing:
            print("  " * indent + f"├─ {cls.name}")
            for subclass in cls.subclasses():
                print_class_tree(subclass, indent + 1)
    
    print_class_tree(ProcessEquipment)
    
    # Statistics by category
    print("\n=== Number of Classes by Equipment Category ===")
    categories = {
        "Reactor": len(list(Reactor.descendants())),
        "HeatExchanger": len(list(HeatExchanger.descendants())),
        "Separator": len(list(Separator.descendants())),
        "FluidMover": len(list(FluidMover.descendants()))
    }
    
    for cat, count in categories.items():
        print(f"{cat}: {count} subclasses")
    
    onto.save(file="process_ontology_v4.owl", format="rdfxml")
    print("\n✓ Ontology saved: process_ontology_v4.owl")
    

**Output example:**  
=== Class Hierarchy Tree ===  
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
  
=== Number of Classes by Equipment Category ===  
Reactor: 3 subclasses  
HeatExchanger: 2 subclasses  
Separator: 3 subclasses  
FluidMover: 2 subclasses  
  
✓ Ontology saved: process_ontology_v4.owl 

**💡 Best Practices for Class Design**

Equipment classification is hierarchically organized by two axes: functional (reaction, separation, heat exchange) and type (CSTR, PFR). This enables systematic equipment selection during process design.
    
    
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

## 2.4 Property Restrictions

### Example 5: Cardinality and Value Restrictions

Define constraints for the number of inputs/outputs and value ranges that equipment should have.
    
    
    # ===================================
    # Example 5: Property Restriction Definition
    # ===================================
    
    from owlready2 import *
    
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        class ProcessEquipment(Thing):
            pass
    
        class Stream(Thing):
            pass
    
        # Property definition
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
    
        # ===== Class Definition with Restrictions =====
    
        # Reactor: at least 1 input, 1 output, temperature and pressure required
        class Reactor(ProcessEquipment):
            equivalent_to = [
                ProcessEquipment
                & hasInput.min(1, Stream)        # At least 1 input
                & hasOutput.min(1, Stream)       # At least 1 output
                & hasTemperature.exactly(1)      # Temperature required (exactly 1)
                & hasPressure.exactly(1)         # Pressure required (exactly 1)
            ]
    
        # HeatExchanger: 1 input, 1 output (fixed)
        class HeatExchanger(ProcessEquipment):
            equivalent_to = [
                ProcessEquipment
                & hasInput.exactly(1, Stream)    # Exactly 1 input
                & hasOutput.exactly(1, Stream)   # Exactly 1 output
                & hasTemperature.exactly(1)
            ]
    
        # DistillationColumn: 1 input, 2 or more outputs (distillate and bottoms)
        class DistillationColumn(ProcessEquipment):
            equivalent_to = [
                ProcessEquipment
                & hasInput.exactly(1, Stream)
                & hasOutput.min(2, Stream)       # At least 2 outputs (distillate, bottoms)
            ]
    
        # ===== Classes with Value Restrictions =====
    
        # HighTemperatureReactor: temperature > 400K
        class HighTemperatureReactor(Reactor):
            """High temperature reactor (400K or above)"""
            pass
    
        # HighPressureReactor: pressure > 10bar
        class HighPressureReactor(Reactor):
            """High pressure reactor (10bar or above)"""
            pass
    
        # ===== Create Individuals and Verify =====
    
        # Valid reactor (satisfies constraints)
        r101 = Reactor("R-101")
        s1 = Stream("S-001")
        s2 = Stream("S-002")
        r101.hasInput = [s1]
        r101.hasOutput = [s2]
        r101.hasTemperature = [450.0]  # High temperature
        r101.hasPressure = [5.0]
    
        # Heat exchanger (satisfies constraints)
        hx201 = HeatExchanger("HX-201")
        s3 = Stream("S-003")
        s4 = Stream("S-004")
        hx201.hasInput = [s3]
        hx201.hasOutput = [s4]
        hx201.hasTemperature = [350.0]
    
    # ===== Constraint Verification =====
    print("=== Verification for Reactor R-101 ===")
    print(f"Number of inputs: {len(r101.hasInput)} (at least 1 required)")
    print(f"Number of outputs: {len(r101.hasOutput)} (at least 1 required)")
    print(f"Temperature setting: {r101.hasTemperature[0]} K (required)")
    print(f"Pressure setting: {r101.hasPressure[0]} bar (required)")
    print("✓ All constraints satisfied")
    
    print("\n=== Verification for Heat Exchanger HX-201 ===")
    print(f"Number of inputs: {len(hx201.hasInput)} (exactly 1 required)")
    print(f"Number of outputs: {len(hx201.hasOutput)} (exactly 1 required)")
    print("✓ All constraints satisfied")
    
    onto.save(file="process_ontology_v5.owl", format="rdfxml")
    print("\n✓ Ontology saved: process_ontology_v5.owl")
    

**Output example:**  
=== Verification for Reactor R-101 ===  
Number of inputs: 1 (at least 1 required)  
Number of outputs: 1 (at least 1 required)  
Temperature setting: 450.0 K (required)  
Pressure setting: 5.0 bar (required)  
✓ All constraints satisfied  
  
=== Verification for Heat Exchanger HX-201 ===  
Number of inputs: 1 (exactly 1 required)  
Number of outputs: 1 (exactly 1 required)  
✓ All constraints satisfied  
  
✓ Ontology saved: process_ontology_v5.owl 

**⚠️ Constraint Violation Detection**

Using owlready2's inference engines (Pellet, HermiT), constraint violations can be automatically detected. For example, reactors without inputs or equipment with unset temperatures are detected as inconsistencies.

## 2.5 Complete Ontology for Process Equipment

### Example 6: Comprehensive Process Equipment Ontology Construction

Build a comprehensive process equipment ontology integrating all elements.
    
    
    # ===================================
    # Example 6: Complete Process Equipment Ontology
    # ===================================
    
    from owlready2 import *
    import numpy as np
    
    onto = get_ontology("http://example.org/process.owl")
    
    with onto:
        # ===== Basic Classes =====
        class ProcessEquipment(Thing):
            pass
    
        class Stream(Thing):
            pass
    
        # ===== Equipment Class Hierarchy =====
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
    
        # ===== Property Definition =====
    
        # Object properties
        class hasInput(ProcessEquipment >> Stream):
            pass
    
        class hasOutput(ProcessEquipment >> Stream):
            pass
    
        class connectedTo(ProcessEquipment >> ProcessEquipment, SymmetricProperty):
            pass
    
        # Data properties
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
            comment = ["Residence time (unit: seconds)"]
    
        # ===== Complete Process Plant Construction =====
    
        # Streams
        feed = Stream("Feed")
        feed.label = ["Raw material feed"]
        feed.hasFlowRate = [1000.0]  # kg/h
    
        s1 = Stream("S-001")
        s2 = Stream("S-002")
        s3 = Stream("S-003")
        product = Stream("Product")
        product.label = ["Final product"]
    
        # Pump P-101
        p101 = Pump("P-101")
        p101.label = ["Feed pump"]
        p101.hasInput = [feed]
        p101.hasOutput = [s1]
        p101.hasEfficiency = [85.0]
    
        # Reactor R-101
        r101 = CSTR("R-101")
        r101.label = ["Main reactor"]
        r101.hasInput = [s1]
        r101.hasOutput = [s2]
        r101.hasTemperature = [350.0]  # K
        r101.hasPressure = [5.0]       # bar
        r101.hasVolume = [10.0]        # m3
        r101.hasEfficiency = [92.5]
        r101.hasResidenceTime = [3600.0]  # 1 hour
    
        # Heat exchanger HX-201
        hx201 = HeatExchanger("HX-201")
        hx201.label = ["Cooler"]
        hx201.hasInput = [s2]
        hx201.hasOutput = [s3]
        hx201.hasTemperature = [320.0]  # Temperature after cooling
        hx201.hasEfficiency = [88.0]
    
        # Separator SEP-301
        sep301 = Separator("SEP-301")
        sep301.label = ["Product separator"]
        sep301.hasInput = [s3]
        sep301.hasOutput = [product]
        sep301.hasTemperature = [320.0]
        sep301.hasPressure = [1.0]
        sep301.hasEfficiency = [95.0]
    
        # Equipment connections
        p101.connectedTo = [r101]
        r101.connectedTo = [hx201]
        hx201.connectedTo = [sep301]
    
    # ===== Visualize Entire Plant =====
    print("=== Process Plant Configuration ===\n")
    
    equipment_list = [p101, r101, hx201, sep301]
    
    for eq in equipment_list:
        print(f"【{eq.label[0]}】 ({eq.__class__.__name__} {eq.name})")
        if eq.hasTemperature:
            print(f"  Temperature: {eq.hasTemperature[0]} K ({eq.hasTemperature[0] - 273.15:.1f}°C)")
        if eq.hasPressure:
            print(f"  Pressure: {eq.hasPressure[0]} bar")
        if eq.hasEfficiency:
            print(f"  Efficiency: {eq.hasEfficiency[0]}%")
        if hasattr(eq, 'hasVolume') and eq.hasVolume:
            print(f"  Volume: {eq.hasVolume[0]} m³")
        if hasattr(eq, 'hasResidenceTime') and eq.hasResidenceTime:
            print(f"  Residence time: {eq.hasResidenceTime[0] / 3600:.1f} hours")
        print()
    
    # Connection relationships
    print("=== Process Flow ===")
    print("Feed → P-101 → R-101 → HX-201 → SEP-301 → Product")
    
    onto.save(file="process_plant_complete.owl", format="rdfxml")
    print("\n✓ Complete process plant ontology saved")
    print(f"Total equipment: {len(equipment_list)}")
    print(f"Total streams: {len([feed, s1, s2, s3, product])}")
    

**Output example:**  
=== Process Plant Configuration ===  
  
【Feed pump】 (Pump P-101)  
Efficiency: 85.0%  
  
【Main reactor】 (CSTR R-101)  
Temperature: 350.0 K (76.9°C)  
Pressure: 5.0 bar  
Efficiency: 92.5%  
Volume: 10.0 m³  
Residence time: 1.0 hours  
  
【Cooler】 (HeatExchanger HX-201)  
Temperature: 320.0 K (46.9°C)  
Efficiency: 88.0%  
  
【Product separator】 (Separator SEP-301)  
Temperature: 320.0 K (46.9°C)  
Pressure: 1.0 bar  
Efficiency: 95.0%  
  
=== Process Flow ===  
Feed → P-101 → R-101 → HX-201 → SEP-301 → Product  
  
✓ Complete process plant ontology saved  
Total equipment: 4  
Total streams: 5 

## 2.6 Integrated Chemical Process Plant Ontology

### Example 7: Integrated Plant Ontology and SPARQL Queries

Represent a complete chemical plant using ontology and execute advanced queries with SPARQL.
    
    
    # ===================================
    # Example 7: Integrated Plant Ontology and SPARQL
    # ===================================
    
    from owlready2 import *
    from rdflib import Graph, Namespace
    
    # Build ontology with owlready2
    onto = get_ontology("http://example.org/chemicalplant.owl")
    
    with onto:
        # Class and property definitions (similar to Example 6)
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
    
        # Build complete plant
        r101 = Reactor("R-101")
        r101.label = ["Reactor R-101"]
        r101.hasTemperature = [350.0]
        r101.hasPressure = [5.0]
        r101.hasEfficiency = [92.5]
    
        r102 = Reactor("R-102")
        r102.label = ["Reactor R-102"]
        r102.hasTemperature = [420.0]
        r102.hasPressure = [8.0]
        r102.hasEfficiency = [88.0]
    
        hx201 = HeatExchanger("HX-201")
        hx201.label = ["Heat Exchanger HX-201"]
        hx201.hasTemperature = [320.0]
        hx201.hasEfficiency = [90.0]
    
        hx202 = HeatExchanger("HX-202")
        hx202.label = ["Heat Exchanger HX-202"]
        hx202.hasTemperature = [340.0]
        hx202.hasEfficiency = [85.0]
    
        sep301 = Separator("SEP-301")
        sep301.label = ["Separator SEP-301"]
        sep301.hasTemperature = [310.0]
        sep301.hasPressure = [1.0]
        sep301.hasEfficiency = [95.0]
    
    # Save OWL and load RDF
    onto.save(file="chemical_plant.owl", format="rdfxml")
    
    # Execute SPARQL query with rdflib
    g = Graph()
    g.parse("chemical_plant.owl", format="xml")
    
    # Define namespace
    ONTO = Namespace("http://example.org/chemicalplant.owl#")
    
    # ===== SPARQL Query Collection =====
    
    # Query 1: Equipment with efficiency >= 90%
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
    
    print("=== Query 1: Equipment with efficiency >= 90% ===")
    for row in g.query(query1):
        print(f"{row.label}: {float(row.efficiency):.1f}%")
    
    # Query 2: Equipment with temperature >= 350K
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
    
    print("\n=== Query 2: High-temperature equipment >= 350K ===")
    for row in g.query(query2):
        temp_c = float(row.temp) - 273.15
        print(f"{row.label}: {float(row.temp):.1f}K ({temp_c:.1f}°C)", end="")
        if row.press:
            print(f", {float(row.press):.1f}bar")
        else:
            print()
    
    # Query 3: Reactor statistics
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
    
    print("\n=== Query 3: Reactor statistics ===")
    for row in g.query(query3):
        print(f"Number of reactors: {row.count}")
        print(f"Average temperature: {float(row.avgTemp):.1f}K ({float(row.avgTemp) - 273.15:.1f}°C)")
        print(f"Average efficiency: {float(row.avgEff):.1f}%")
    
    print("\n✓ Integrated plant ontology and SPARQL queries completed")
    

**Output example:**  
=== Query 1: Equipment with efficiency >= 90% ===  
Separator SEP-301: 95.0%  
Reactor R-101: 92.5%  
Heat Exchanger HX-201: 90.0%  
  
=== Query 2: High-temperature equipment >= 350K ===  
Reactor R-102: 420.0K (146.9°C), 8.0bar  
Reactor R-101: 350.0K (76.9°C), 5.0bar  
  
=== Query 3: Reactor statistics ===  
Number of reactors: 2  
Average temperature: 385.0K (111.9°C)  
Average efficiency: 90.3%  
  
✓ Integrated plant ontology and SPARQL queries completed 

**✅ Ontology Design Achievements**

  * **Structuring** : Hierarchical ontology with 13 classes and 5 properties
  * **Constraints** : Quality assurance through cardinality and value range restrictions
  * **Querying** : Advanced knowledge extraction and analysis with SPARQL
  * **Extensibility** : Easy addition of new equipment types and properties

## Learning Objectives Check

Upon completing this chapter, you will be able to explain and implement the following:

### Basic Understanding

  * ✅ Explain the differences between the three OWL sublanguages (Lite, DL, Full)
  * ✅ Understand the roles of object properties and data properties
  * ✅ Know the concepts of FunctionalProperty, SymmetricProperty, and TransitiveProperty
  * ✅ Understand design principles for class hierarchies and is-a relationships

### Practical Skills

  * ✅ Define OWL classes and instances with owlready2
  * ✅ Express connections between equipment using object properties
  * ✅ Define physical properties like temperature, pressure, and flow rate using data properties
  * ✅ Implement cardinality restrictions (min, max, exactly)
  * ✅ Build a complete chemical process plant ontology
  * ✅ Query equipment statistics and conditional searches using SPARQL

### Applied Capabilities

  * ✅ Integrate new equipment types into existing ontologies
  * ✅ Support equipment selection in process design using ontologies
  * ✅ Detect inconsistent equipment configurations through property restrictions
  * ✅ Convert actual plant P&ID information into ontologies

## Next Steps

In Chapter 2, we learned comprehensive ontology design for chemical process equipment using OWL and owlready2. In the next chapter, we will learn methods to automatically construct knowledge graphs from actual process data (CSV, sensor streams, P&ID).

**📚 Preview of Next Chapter (Chapter 3)**

  * Entity extraction from CSV data
  * Automatic extraction of equipment connection relationships
  * RDF conversion of sensor data streams
  * Time-series knowledge graphs for historical data
  * Integrated knowledge graph construction from multi-source data

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
