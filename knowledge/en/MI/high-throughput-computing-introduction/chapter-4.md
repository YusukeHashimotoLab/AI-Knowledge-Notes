---
title: "Chapter 4: Data Management and Post-Processing (FireWorks, AiiDA)"
chapter_title: "Chapter 4: Data Management and Post-Processing (FireWorks, AiiDA)"
subtitle: 
reading_time: 20-25 minutes
difficulty: Advanced
code_examples: 6
exercises: 0
version: 1.0
created_at: 2025-10-17
---

# Chapter 4: Data Management and Post-Processing (FireWorks, AiiDA)

This chapter demonstrates specific procedures for scaling with SLURM and cloud platforms. You will also learn the essentials of cost estimation and optimization.

**💡 Note:** Rough estimation using unit cost × time × number of instances → set upper limits. Be cautious as excessive parallelization increases failure rates.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Build complex workflows with FireWorks
  * ✅ Master standard workflows in Atomate
  * ✅ Record data provenance with AiiDA
  * ✅ Store calculation results in structured databases
  * ✅ Publish results to NOMAD

* * *

## 4.1 Workflow Management with FireWorks

### FireWorks Architecture
    
    
    ```mermaid
    flowchart TD
        A["Firework"] -->|Chain| B["Workflow"]
        B --> C["LaunchPadMongoDB"]
        C -->|Task Assignment| D["Rocket Launcher"]
        D -->|Execute| E["Compute Node"]
        E -->|Results| C
    
        style C fill:#4ecdc4
        style A fill:#ffe66d
        style B fill:#ff6b6b
    ```

**Key Components** : \- **Firework** : Single task (one DFT calculation) \- **Workflow** : Chain of multiple Fireworks \- **LaunchPad** : Database (MongoDB) \- **Rocket** : Task execution engine

### Installation and Configuration
    
    
    # Install FireWorks
    pip install fireworks
    
    # Install MongoDB (macOS)
    brew install mongodb-community
    
    # Start MongoDB
    brew services start mongodb-community
    
    # Initialize FireWorks
    lpad init
    # → Generates my_launchpad.yaml
    

**my_launchpad.yaml** :
    
    
    host: localhost
    port: 27017
    name: fireworks
    username: null
    password: null
    

### Creating Basic Fireworks
    
    
    from fireworks import Firework, LaunchPad, ScriptTask
    from fireworks.core.rocket_launcher import rapidfire
    
    # Connect to LaunchPad (database)
    launchpad = LaunchPad(host='localhost', port=27017, name='fireworks')
    
    # Define task (run VASP)
    vasp_task = ScriptTask.from_str(
        'mpirun -np 48 vasp_std',
        use_shell=True
    )
    
    # Create Firework
    fw = Firework(
        vasp_task,
        name='VASP relaxation',
        spec={'_category': 'VASP'}
    )
    
    # Add to LaunchPad
    launchpad.add_wf(fw)
    
    # Execute
    rapidfire(launchpad)
    

### Standard Workflows in Atomate

**Atomate** is a library of standardized workflows used by Materials Project.
    
    
    from atomate.vasp.workflows.base.core import get_wf
    from pymatgen.core import Structure
    
    # Load structure
    structure = Structure.from_file("LiCoO2.cif")
    
    # Get standard workflow
    # optimize_structure_and_properties:
    #   1. Structure optimization
    #   2. Static calculation
    #   3. Band structure
    #   4. DOS calculation
    wf = get_wf(
        structure,
        "optimize_structure_and_properties.yaml"
    )
    
    # Add to LaunchPad
    launchpad.add_wf(wf)
    
    # Execute
    rapidfire(launchpad, nlaunches='infinite')
    

### Creating Custom Workflows
    
    
    from fireworks import Firework, Workflow
    from fireworks.core.firework import FiretaskBase
    from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
    from atomate.vasp.firetasks.run_calc import RunVaspCustodian
    from atomate.vasp.firetasks.parse_outputs import VaspToDb
    from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
    
    class CustomWorkflow:
        """Custom workflow generation"""
    
        @staticmethod
        def bandgap_workflow(structure):
            """
            Band gap calculation workflow
            1. Structure optimization → 2. Static calculation → 3. Band gap extraction
            """
            # Firework 1: Structure optimization
            fw1 = Firework(
                [
                    WriteVaspFromIOSet(structure=structure, vasp_input_set=MPRelaxSet),
                    RunVaspCustodian(vasp_cmd="vasp_std"),
                    VaspToDb(db_file="db.json", task_label="relax")
                ],
                name="Structural relaxation",
                spec={"_category": "relax"}
            )
    
            # Firework 2: Static calculation
            fw2 = Firework(
                [
                    WriteVaspFromIOSet(structure=structure, vasp_input_set=MPStaticSet),
                    RunVaspCustodian(vasp_cmd="vasp_std"),
                    VaspToDb(db_file="db.json", task_label="static")
                ],
                name="Static calculation",
                spec={"_category": "static"}
            )
    
            # Firework 3: Band gap extraction
            fw3 = Firework(
                [ExtractBandgapTask()],
                name="Extract bandgap"
            )
    
            # Build workflow (fw1 → fw2 → fw3)
            wf = Workflow(
                [fw1, fw2, fw3],
                links_dict={fw1: [fw2], fw2: [fw3]},
                name=f"Bandgap workflow: {structure.composition.reduced_formula}"
            )
    
            return wf
    
    
    class ExtractBandgapTask(FiretaskBase):
        """Band gap extraction task"""
    
        def run_task(self, fw_spec):
            from pymatgen.io.vasp.outputs import Vasprun
    
            vasprun = Vasprun("vasprun.xml")
            bandgap = vasprun.get_band_structure().get_band_gap()
    
            print(f"Band gap: {bandgap['energy']:.3f} eV")
    
            # Save result to database
            return {"bandgap": bandgap['energy']}
    
    # Usage example
    structure = Structure.from_file("LiCoO2.cif")
    wf = CustomWorkflow.bandgap_workflow(structure)
    
    launchpad.add_wf(wf)
    rapidfire(launchpad)
    

### Error Handling and Restart
    
    
    from custodian.custodian import Custodian
    from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler
    from custodian.vasp.jobs import VaspJob
    
    # Configure error handlers
    handlers = [
        VaspErrorHandler(),           # General VASP errors
        UnconvergedErrorHandler(),    # Convergence errors
    ]
    
    # Define VASP job
    jobs = [
        VaspJob(
            vasp_cmd=["mpirun", "-np", "48", "vasp_std"],
            output_file="vasp.out",
            auto_npar=False
        )
    ]
    
    # Run Custodian (automatic error handling)
    c = Custodian(
        handlers=handlers,
        jobs=jobs,
        max_errors=5
    )
    
    c.run()
    

* * *

## 4.2 Provenance Management with AiiDA

### The Importance of Data Provenance

**Provenance** is the complete record of how calculation results were obtained.

**Information to track** : \- Input data (structures, parameters) \- Software used (VASP 6.3.0, etc.) \- Computing environment (nodes, cores, date/time) \- Intermediate results \- Final results

### Installing AiiDA
    
    
    # Install AiiDA
    pip install aiida-core aiida-vasp
    
    # Initialize database
    verdi quicksetup
    
    # Check status
    verdi status
    

### Running Calculations with AiiDA
    
    
    from aiida import orm, engine
    from aiida.plugins import CalculationFactory, DataFactory
    
    # Get VASP calculation class
    VaspCalculation = CalculationFactory('vasp.vasp')
    
    # Data types
    StructureData = DataFactory('structure')
    KpointsData = DataFactory('array.kpoints')
    
    # Create structure data
    structure = StructureData()
    # (Set structure)
    
    # Set k-points
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([8, 8, 8])
    
    # Calculation parameters
    parameters = orm.Dict(dict={
        'ENCUT': 520,
        'EDIFF': 1e-5,
        'ISMEAR': 0,
        'SIGMA': 0.05,
    })
    
    # Build calculation
    builder = VaspCalculation.get_builder()
    builder.structure = structure
    builder.kpoints = kpoints
    builder.parameters = parameters
    builder.code = orm.Code.get_from_string('vasp@localhost')
    
    # Submit
    calc_node = engine.submit(builder)
    print(f"Calculation node: {calc_node.pk}")
    

### Data Queries
    
    
    from aiida.orm import QueryBuilder
    
    # Search all VASP calculations
    qb = QueryBuilder()
    qb.append(VaspCalculation, filters={'attributes.exit_status': 0})
    
    for calc in qb.all():
        print(f"PK: {calc[0].pk}, Formula: {calc[0].inputs.structure.get_formula()}")
    
    # Search for materials with band gap 1.5-2.5 eV
    qb = QueryBuilder()
    qb.append(VaspCalculation, tag='calc')
    qb.append(orm.Dict, with_incoming='calc', filters={
        'attributes.band_gap': {'and': [{'>=': 1.5}, {'<=': 2.5}]}
    })
    
    results = qb.all()
    print(f"Matching materials: {len(results)}")
    

* * *

## 4.3 Structuring Calculation Data

### JSON Schema Design
    
    
    import json
    from datetime import datetime
    
    class MaterialsDataSchema:
        """Schema for materials calculation data"""
    
        @staticmethod
        def create_entry(material_id, formula, structure, calculation_results):
            """
            Create database entry
    
            Returns:
            --------
            entry : dict
                Structured data
            """
            entry = {
                # Identification
                "material_id": material_id,
                "formula": formula,
                "created_at": datetime.now().isoformat(),
    
                # Structure information
                "structure": {
                    "lattice": structure.lattice.matrix.tolist(),
                    "species": [str(site.specie) for site in structure],
                    "coords": [site.frac_coords.tolist() for site in structure],
                    "space_group": structure.get_space_group_info()[1]
                },
    
                # Calculation results
                "properties": {
                    "energy": calculation_results.get("energy"),
                    "band_gap": calculation_results.get("band_gap"),
                    "formation_energy": calculation_results.get("formation_energy"),
                },
    
                # Calculation metadata
                "calculation_metadata": {
                    "software": "VASP 6.3.0",
                    "functional": "PBE",
                    "encut": 520,
                    "kpoints": calculation_results.get("kpoints"),
                    "converged": calculation_results.get("converged"),
                    "calculation_time": calculation_results.get("calculation_time"),
                },
    
                # Provenance
                "provenance": {
                    "input_structure_source": "Materials Project",
                    "workflow": "Atomate optimize_structure",
                    "hostname": calculation_results.get("hostname"),
                    "date": calculation_results.get("date"),
                }
            }
    
            return entry
    
    # Usage example
    entry = MaterialsDataSchema.create_entry(
        material_id="custom-0001",
        formula="LiCoO2",
        structure=structure,
        calculation_results={
            "energy": -45.67,
            "band_gap": 2.3,
            "converged": True,
            "kpoints": [12, 12, 8],
            "calculation_time": "2.5 hours",
            "hostname": "hpc.university.edu",
            "date": "2025-10-17",
        }
    )
    
    # Save as JSON
    with open("data/custom-0001.json", 'w') as f:
        json.dump(entry, f, indent=2)
    

### Data Management with MongoDB
    
    
    from pymongo import MongoClient
    import json
    
    class MaterialsDatabase:
        """Materials database using MongoDB"""
    
        def __init__(self, host='localhost', port=27017, db_name='materials'):
            self.client = MongoClient(host, port)
            self.db = self.client[db_name]
            self.collection = self.db['calculations']
    
            # Create indices (speed up searches)
            self.collection.create_index("material_id", unique=True)
            self.collection.create_index("formula")
            self.collection.create_index("properties.band_gap")
    
        def insert(self, entry):
            """Insert data"""
            result = self.collection.insert_one(entry)
            return result.inserted_id
    
        def find_by_formula(self, formula):
            """Search by chemical formula"""
            return list(self.collection.find({"formula": formula}))
    
        def find_by_bandgap_range(self, min_gap, max_gap):
            """Search by band gap range"""
            query = {
                "properties.band_gap": {
                    "$gte": min_gap,
                    "$lte": max_gap
                }
            }
            return list(self.collection.find(query))
    
        def get_statistics(self):
            """Get statistics"""
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total": {"$sum": 1},
                        "avg_bandgap": {"$avg": "$properties.band_gap"},
                        "avg_energy": {"$avg": "$properties.energy"}
                    }
                }
            ]
    
            result = list(self.collection.aggregate(pipeline))
            return result[0] if result else {}
    
    # Usage example
    db = MaterialsDatabase()
    
    # Insert data
    db.insert(entry)
    
    # Search
    licoo2_results = db.find_by_formula("LiCoO2")
    print(f"LiCoO2 calculations: {len(licoo2_results)}")
    
    # Band gap search
    semiconductors = db.find_by_bandgap_range(0.5, 3.0)
    print(f"Semiconductors (0.5-3.0 eV): {len(semiconductors)}")
    
    # Statistics
    stats = db.get_statistics()
    print(f"Average band gap: {stats['avg_bandgap']:.2f} eV")
    

* * *

## 4.4 Automating Post-Processing

### Automatic DOS/Band Structure Plotting
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    from pymatgen.io.vasp.outputs import Vasprun
    from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
    import matplotlib.pyplot as plt
    
    def auto_plot_band_dos(directory):
        """
        Automatically plot band structure and DOS
    
        Parameters:
        -----------
        directory : str
            Directory containing vasprun.xml
        """
        vasprun = Vasprun(f"{directory}/vasprun.xml")
    
        # Band structure
        bs = vasprun.get_band_structure()
        bs_plotter = BSPlotter(bs)
    
        fig = bs_plotter.get_plot(ylim=[-5, 5])
        fig.savefig(f"{directory}/band_structure.png", dpi=300)
        print(f"Band structure saved: {directory}/band_structure.png")
    
        # DOS
        dos = vasprun.complete_dos
        dos_plotter = DosPlotter()
        dos_plotter.add_dos("Total", dos)
    
        fig = dos_plotter.get_plot(xlim=[-5, 5])
        fig.savefig(f"{directory}/dos.png", dpi=300)
        print(f"DOS saved: {directory}/dos.png")
    
        # Band gap
        bandgap = bs.get_band_gap()
        print(f"Band gap: {bandgap['energy']:.3f} eV ({bandgap['transition']})")
    
    # Usage example
    auto_plot_band_dos("calculations/LiCoO2")
    

### Automatic Report Generation
    
    
    from jinja2 import Template
    from datetime import datetime
    
    def generate_report(material_data, output_file="report.html"):
        """
        Automatically generate calculation result report
    
        Parameters:
        -----------
        material_data : dict
            Material data
        output_file : str
            Output HTML file
        """
        template_str = """
    
    
    
    
    
        Calculation Report: {{ material_data.formula }}
        
    
    
        
    
    # {{ material_data.formula }}
    
    
        
    
    Generated: {{ generation_time }}
    
    
    
        
    
    ## Structure Information
    
    
        
            Property| Value  
    ---|---  
    
            Space Group| {{ material_data.structure.space_group }}  
    
            Lattice Parameter a| {{ "%.3f"|format(material_data.structure.lattice[0][0]) }} Å  
    
        
    
        
    
    ## Calculation Results
    
    
        
            Property| Value  
    ---|---  
    
            Energy| {{ "%.3f"|format(material_data.properties.energy) }} eV/atom  
    
            Band Gap| {{ "%.3f"|format(material_data.properties.band_gap) }} eV  
    
        
    
        
    
    ## Band Structure
    
    
        ![Band Structure](band_structure.png)
    
        
    
    ## Density of States
    
    
        ![DOS](dos.png)
    
    
        """
    
        template = Template(template_str)
    
        html = template.render(
            material_data=material_data,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
        with open(output_file, 'w') as f:
            f.write(html)
    
        print(f"Report generated: {output_file}")
    
    # Usage example
    generate_report(entry, output_file="LiCoO2_report.html")
    

* * *

## 4.5 Data Sharing and Archiving

### Uploading to NOMAD Repository
    
    
    # Requirements:
    # - Python 3.9+
    # - requests>=2.31.0
    
    import requests
    import json
    
    def upload_to_nomad(data_files, metadata):
        """
        Upload data to NOMAD Repository
    
        Parameters:
        -----------
        data_files : list
            List of files to upload
        metadata : dict
            Metadata
        """
        nomad_url = "https://nomad-lab.eu/prod/rae/api/v1/uploads"
    
        # Prepare metadata
        upload_metadata = {
            "upload_name": metadata.get("name", "Untitled"),
            "references": metadata.get("references", []),
            "coauthors": metadata.get("coauthors", []),
        }
    
        # Upload files
        files = []
        for file_path in data_files:
            files.append(('file', open(file_path, 'rb')))
    
        response = requests.post(
            nomad_url,
            files=files,
            data={'metadata': json.dumps(upload_metadata)},
            headers={'Authorization': f'Bearer {NOMAD_API_TOKEN}'}
        )
    
        if response.status_code == 200:
            upload_id = response.json()['upload_id']
            print(f"Upload successful: {upload_id}")
            print(f"URL: https://nomad-lab.eu/prod/rae/gui/uploads/{upload_id}")
            return upload_id
        else:
            print(f"Upload failed: {response.status_code}")
            return None
    
    # Usage example
    files = [
        "calculations/LiCoO2/vasprun.xml",
        "calculations/LiCoO2/OUTCAR",
        "calculations/LiCoO2/CONTCAR",
    ]
    
    metadata = {
        "name": "LiCoO2 battery material calculations",
        "references": ["https://doi.org/10.xxxx/xxxxx"],
        "coauthors": ["Dr. Yusuke Hashimoto"]
    }
    
    upload_to_nomad(files, metadata)
    

* * *

## 4.6 Exercises

### Exercise 1 (Difficulty: medium)

**Problem** : Create a 2-step workflow with FireWorks: structural optimization → static calculation.

Solution Example
    
    
    from fireworks import Firework, Workflow
    from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
    from atomate.vasp.firetasks.run_calc import RunVaspCustodian
    from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
    from pymatgen.core import Structure
    
    structure = Structure.from_file("POSCAR")
    
    # Firework 1: Structural optimization
    fw1 = Firework(
        [
            WriteVaspFromIOSet(structure=structure, vasp_input_set=MPRelaxSet),
            RunVaspCustodian(vasp_cmd="vasp_std")
        ],
        name="Relax"
    )
    
    # Firework 2: Static calculation
    fw2 = Firework(
        [
            WriteVaspFromIOSet(structure=structure, vasp_input_set=MPStaticSet),
            RunVaspCustodian(vasp_cmd="vasp_std")
        ],
        name="Static"
    )
    
    # Workflow
    wf = Workflow([fw1, fw2], links_dict={fw1: [fw2]})
    
    launchpad.add_wf(wf)
    

### Exercise 2 (Difficulty: hard)

**Problem** : From 1000 materials stored in MongoDB, extract the following: 1\. Semiconductors with band gap 1.0-2.0 eV 2\. Negative (stable) formation energy 3\. Save results to CSV

Solution Example
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Problem: From 1000 materials stored in MongoDB, extract the 
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    
    db = MaterialsDatabase()
    
    # Query
    query = {
        "properties.band_gap": {"$gte": 1.0, "$lte": 2.0},
        "properties.formation_energy": {"$lt": 0}
    }
    
    results = list(db.collection.find(query))
    
    # Create DataFrame
    data = []
    for r in results:
        data.append({
            "formula": r["formula"],
            "band_gap": r["properties"]["band_gap"],
            "formation_energy": r["properties"]["formation_energy"],
            "space_group": r["structure"]["space_group"]
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv("stable_semiconductors.csv", index=False)
    print(f"Matching materials: {len(df)}")
    print(df.head())
    

* * *

## 4.7 Summary

**Key Points** :

  1. **FireWorks** : Standard workflow management for Materials Project
  2. **Atomate** : Standardized calculation workflows
  3. **AiiDA** : Data provenance tracking
  4. **MongoDB** : Large-scale data management
  5. **NOMAD** : Data publication and FAIR principles

**Next Step** : Chapter 5 covers cloud HPC utilization.

**[Chapter 5: Cloud HPC Utilization and Optimization →](<chapter-5.html>)**

* * *

**License** : CC BY 4.0 **Date Created** : 2025-10-17 **Author** : Dr. Yusuke Hashimoto, Tohoku University
