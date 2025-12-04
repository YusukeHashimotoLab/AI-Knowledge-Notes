---
title: "Chapter 1: The Need for High-Throughput Computing and Workflow Design"
chapter_title: "Chapter 1: The Need for High-Throughput Computing and Workflow Design"
subtitle: 
reading_time: 20-30 minutes
difficulty: Intermediate
code_examples: 0
exercises: 0
version: 1.0
---

# Chapter 1: The Need for High-Throughput Computing and Workflow Design

From one-off execution to "systems that run calculations." Quickly grasp the value and scope of HTC.

**💡 Note:** The shift from "people running tasks" to "systems running tasks." Even small automation yields cumulative benefits.

## Learning Objectives

By reading this chapter, you will master:

  * ✅ Quantitatively understand the vastness of materials exploration space
  * ✅ Explain the four elements of High-Throughput Computing (automation, parallelization, standardization, data management)
  * ✅ Analyze success factors of Materials Project, AFLOW, and others
  * ✅ Understand and apply principles of effective workflow design
  * ✅ Quantitatively evaluate cost reduction effects

* * *

## 1.1 Challenges in Materials Discovery: Why High-Throughput Computing is Necessary

### The Vastness of the Exploration Space

The greatest challenge in materials science is that **the space to be explored is enormously vast**.

**Example: Ternary alloy exploration**

Consider Li-Ni-Co oxide battery materials (LiₓNiᵧCoᵧO₂): \- Li composition: 0.0-1.0 (10 levels) \- Ni composition: 0.0-1.0 (10 levels) \- Co composition: 0.0-1.0 (10 levels)

Simple calculation gives **10³ = 1,000 combinations**.

**Example: Quinary high-entropy alloys**

For CoCrFeNiMn systems with varying composition ratios: \- Each element: 0-100% (11 levels at 10% intervals) \- Constraint that total composition = 100%

The combinations reach **tens of thousands**.

**Typical scale of materials exploration**
    
    
    ```mermaid
    flowchart LR
        A[Single material] -->|Substitution| B[10-100 candidates]
        B -->|Composite materials| C[1,000-10,000 candidates]
        C -->|High-dimensional search| D[100,000-1,000,000 candidates]
        D -->|Exhaustive search| E[10^12-10^60 candidates]
    
        style E fill:#ff6b6b
    ```

In actual materials exploration, there exist **10¹² (1 trillion) to 10⁶⁰ combinations**.

### Limitations of Traditional Methods

**Cost per material (traditional experiment-driven approach)**

Item | Time | Cost (Estimate)  
---|---|---  
Literature survey | 1-2 weeks | $10,000  
Sample synthesis | 1-4 weeks | $30,000-100,000  
Characterization | 2-8 weeks | $50,000-200,000  
Data analysis | 1-2 weeks | $10,000-30,000  
**Total** | **2-4 months** | **$100,000-340,000**  
  
**Annual exploration capacity**

  * Experienced researcher (1 person): **10-50 materials/year**
  * Research group (5-10 people): **50-200 materials/year**

**Problems**

  1. **Time** : New material development takes 15-20 years
  2. **Cost** : Hundreds of thousands of dollars per material
  3. **Scalability** : Adding personnel only improves linearly
  4. **Reproducibility** : Complete recording of experimental conditions is difficult

### Materials Genome Initiative (MGI) Proposal

In 2011, the Obama administration announced the **Materials Genome Initiative (MGI)** :

**Goals** : \- **Halve** new material development time (20 years → 10 years) \- **Closed loop** of computational science and experiments \- **Public resource** databases and computational infrastructure

**Budget** : $100M in first year, $250M over 10 years

**Outcomes** (2011-2021): \- Materials Project: DFT calculations for **140,000 materials** \- AFLOW: Automated analysis of **3,500,000 crystal structures** \- Development period: Actually achieved **30-50% reduction**

* * *

## 1.2 Definition of High-Throughput Computing and Its Four Elements

### Definition

> **High-Throughput Computing (HTC)** is a methodology that efficiently executes large volumes of computational tasks through automation and parallelization, under standardized workflows and data management.

### Four Elements
    
    
    ```mermaid
    flowchart TD
        A[High-Throughput Computing] --> B[1. Automation]
        A --> C[2. Parallelization]
        A --> D[3. Standardization]
        A --> E[4. Data Management]
    
        B --> B1[Automated input generation]
        B --> B2[Automated job submission]
        B --> B3[Automated error handling]
    
        C --> C1[Simultaneous multi-material calculation]
        C --> C2[MPI node parallelization]
        C --> C3[Task and data parallelism]
    
        D --> D1[Unified computational conditions]
        D --> D2[Unified file formats]
        D --> D3[Quality control standards]
    
        E --> E1[Database design]
        E --> E2[Metadata recording]
        E --> E3[Search and visualization]
    
        style A fill:#4ecdc4
        style B fill:#ffe66d
        style C fill:#ff6b6b
        style D fill:#95e1d3
        style E fill:#a8e6cf
    ```

#### Element 1: Automation

Systems that execute and manage calculations without human intervention.

**Example: Structure optimization automation**

Manual case: 1\. Prepare initial structure → Create input file → Submit job 2\. Check completion → Review results → Judge convergence 3\. If not converged, change settings → Resubmit 4\. If converged, move to next material

**30 minutes to 2 hours per material** (50-200 hours for 100 materials)

After automation:
    
    
    for structure in structures:
        optimize_structure(structure)  # Fully automatic
        if converged:
            calculate_properties(structure)
    # 100 materials completed overnight (zero human effort)
    

#### Element 2: Parallelization

Improving throughput by executing multiple calculations simultaneously.

**Three parallelization levels**

  1. **Task parallelism** : Simultaneous calculation of different materials \- 1000 materials parallel execution on 100 nodes → 10x speedup

  2. **Data parallelism** : k-point parallel calculation for same material \- 2-4x speedup with VASP KPAR settings

  3. **MPI parallelism** : Distributing single calculation across multiple cores \- 10-20x speedup on 48 cores (50-70% scaling efficiency)

**Parallel efficiency examples**

Parallelization method | Nodes | Speedup ratio | Efficiency  
---|---|---|---  
Task parallelism only | 100 | 100x | 100%  
MPI parallelism only | 4 | 3.2x | 80%  
Hybrid | 100x4 | 320x | 80%  
  
#### Element 3: Standardization

Unifying computational conditions, data formats, and quality standards.

**Materials Project standard settings**
    
    
    # VASP settings example (Materials Project)
    {
        "ENCUT": 520,  # Energy cutoff (eV)
        "EDIFF": 1e-5,  # Energy convergence criterion
        "K-point density": 1000,  # k-point density (Å⁻³)
        "ISMEAR": -5,  # Tetrahedron method
    }
    

**Benefits** : \- **Fair comparison** between different materials \- Ensuring calculation **reproducibility** \- **Ease** of error detection

#### Element 4: Data Management

Structuring and storing computational results for searchability.

**Database schema example**
    
    
    {
      "material_id": "mp-1234",
      "formula": "LiCoO2",
      "structure": {...},
      "energy": -45.67,  // eV/atom
      "band_gap": 2.3,   // eV
      "calculation_metadata": {
        "vasp_version": "6.3.0",
        "encut": 520,
        "kpoints": [12, 12, 8],
        "calculation_date": "2025-10-17"
      }
    }
    

**Search example** :
    
    
    # Search for oxides with band gap 1.5-2.5 eV
    results = db.find({
        "band_gap": {"$gte": 1.5, "$lte": 2.5},
        "elements": {"$all": ["O"]}
    })
    

* * *

## 1.3 Success Stories: Materials Project, AFLOW, OQMD

### Materials Project (USA)

**Scale** (as of 2025): \- Number of materials: **140,000+** \- Calculation tasks: **5,000,000+** \- DFT computation time: Over **500 million CPU hours**

**Technology stack** : \- Calculation code: VASP \- Workflow: FireWorks + Atomate \- Database: MongoDB \- API: pymatgen + RESTful API

**Achievements** : \- Li-ion battery materials: **67% development time reduction** \- Thermoelectric materials: **90% prediction accuracy** for ZT values \- Perovskite solar cells: **Screening of 50,000 candidates**

**Impact** : \- Citations: **20,000+ times** (Google Scholar) \- Industrial use: Tesla, Panasonic, Samsung, etc. \- Users: **100,000+** (API registrations)

### AFLOW (Duke University)

**Scale** : \- Crystal structures: **3,500,000+** \- Prototypes: **1,000,000+** \- Calculated properties: Band gaps, elastic constants, thermodynamic stability

**Features** : \- **Crystal symmetry analysis** : Automatic space group identification \- **Prototype database** : Generation from known structures \- **AFLOW-ML** : Machine learning integration

**Applications** : \- High-entropy alloys: **Phase stability prediction** \- Superconducting materials: **Tc prediction**

### OQMD (Northwestern University)

**Scale** : \- Number of materials: **815,000+** \- DFT calculations: Quantum ESPRESSO

**Features** : \- **Thermodynamic data** : Formation energy, phase equilibria \- **Chemical potential diagrams** : Stability visualization

### JARVIS (NIST)

**Scale** : \- Number of materials: **40,000+** \- Diverse properties: Optical, elastic, magnetic, topological

**Features** : \- **Machine learning models** : Pre-trained model provision \- **2D materials** : Large database of monolayer materials

### Comparison table

Project | Materials | Calculation code | Features  
---|---|---|---  
Materials Project | 140k+ | VASP | Comprehensive, industrial use  
AFLOW | 3.5M+ | VASP | Crystal structure focused  
OQMD | 815k+ | QE | Thermodynamic data  
JARVIS | 40k+ | VASP | Diverse properties  
  
* * *

## 1.4 Principles of Workflow Design

Effective High-Throughput Computing requires appropriate workflow design.

### Principle 1: Modularity

Divide each task into independent modules for reusability.

**Good example** :
    
    
    # Modularized workflow
    structure = generate_structure(formula)
    relaxed = relax_structure(structure)
    energy = static_calculation(relaxed)
    band_gap = calculate_band_structure(relaxed)
    dos = calculate_dos(relaxed)
    

**Bad example** :
    
    
    # Monolithic script
    # Difficult to reuse parts
    run_everything(formula)  # Black box
    

### Principle 2: Error Handling

Calculation failures are inevitable, requiring appropriate error handling.

**Error classification**
    
    
    ```mermaid
    flowchart TD
        A[Calculation errors] --> B[Retryable]
        A --> C[Requires setting changes]
        A --> D[Fatal errors]
    
        B --> B1[Temporary I/O errors]
        B --> B2[Job timeout]
    
        C --> C1[Convergence issues]
        C --> C2[Memory shortage]
    
        D --> D1[Structure abnormality]
        D --> D2[Software bug]
    
        B1 -->|Auto retry| E[Success]
        C1 -->|Relax settings| E
        D1 -->|Skip| F[Next material]
    ```

**Implementation example** :
    
    
    def robust_calculation(structure, max_retries=3):
        for attempt in range(max_retries):
            try:
                result = run_vasp(structure)
                if result.converged:
                    return result
                else:
                    # Convergence issue → Change settings
                    structure = adjust_parameters(structure)
            except MemoryError:
                # Memory shortage → Reduce cores
                reduce_cores()
            except TimeoutError:
                # Timeout → Extend time limit
                extend_time_limit()
    
        # Finally failed
        log_failure(structure)
        return None
    

### Principle 3: Reproducibility

Enable other researchers to obtain the same results.

**Essential recording items** :

  1. **Calculation conditions** : All parameters
  2. **Software version** : VASP 6.3.0, etc.
  3. **Pseudopotentials** : PBE, PAW, etc.
  4. **Computational environment** : OS, compiler, libraries

**Implementation example** :
    
    
    # Provenance recording
    metadata = {
        "software": "VASP 6.3.0",
        "potcar": "PBE_54",
        "encut": 520,
        "kpoints": [12, 12, 8],
        "convergence": {
            "energy": 1e-5,
            "force": 0.01
        },
        "compute_environment": {
            "hostname": "hpc.university.edu",
            "nodes": 4,
            "cores_per_node": 48,
            "date": "2025-10-17T10:30:00Z"
        }
    }
    

### Principle 4: Scalability

Design that can scale from 10 materials → 1,000 materials → 100,000 materials.

**Scalability checklist** :

  * [ ] Can the database handle large volumes of data (MongoDB, PostgreSQL)
  * [ ] Can the file system withstand hundreds of thousands of files
  * [ ] Is network bandwidth sufficient
  * [ ] Are job scheduler limits (maximum jobs) acceptable
  * [ ] Is data analysis parallelized

**Scalability testing** :
    
    
    # Small-scale test: 10 materials
    test_workflow(n_materials=10)  # 1 hour
    
    # Medium-scale test: 100 materials
    test_workflow(n_materials=100)  # 10 hours
    
    # Large-scale test: 1,000 materials
    test_workflow(n_materials=1000)  # 100 hours
    
    # Check scaling efficiency
    scaling_efficiency = (time_10 * 100) / time_1000
    # Ideal is 1.0 (linear scaling)
    

* * *

## 1.5 Quantitative Analysis of Costs and Benefits

### Traditional Method vs High-Throughput Computing

**Scenario** : Screening 1,000 materials

#### Traditional method (experiment-driven)

Item | Unit price | Quantity | Total  
---|---|---|---  
Researcher personnel costs | $80,000/year | 5 people × 2 years | $800,000  
Sample synthesis | $50,000 | 1,000 materials | $50,000,000  
Characterization | $30,000 | 1,000 materials | $30,000,000  
**Total cost** |  |  | **$80,800,000**  
**Duration** |  |  | **2 years**  
  
#### High-Throughput Computing

Item | Unit price | Quantity | Total  
---|---|---|---  
Researcher personnel costs | $80,000/year | 2 people × 6 months | $80,000  
Computing resources | $0.5/CPU hour | 1M CPU hours | $500,000  
Storage | $0.02/GB/month | 10TB × 6 months | $10,000  
Software licenses | $300,000/year | 1 year | $300,000  
**Total cost** |  |  | **$890,000**  
**Duration** |  |  | **6 months**  
  
**Reduction effects** : \- Cost reduction: **89%** ($80.8M → $0.89M) \- Time reduction: **75%** (2 years → 6 months)

### ROI (Return on Investment) Calculation

**Initial investment** : \- Environment setup: $50,000 \- Personnel training: $30,000 \- **Total** : $80,000

**Annual savings** (for 1,000 materials/year): \- Experimental cost reduction: $80M/year \- Personnel cost reduction: $240,000/year \- **Total** : $80.24M/year

**ROI** :
    
    
    ROI = (Annual savings - Operating costs) / Initial investment
        = ($80.24M - $0.89M) / $0.08M
        = 991x
    

**Payback period** : About 1 day

### Non-monetary Benefits

  1. **Innovation acceleration** : Trial-and-error cycle 2 years → 6 months
  2. **Competitive advantage** : Market entry 6-12 months ahead of competitors
  3. **Data assets** : Accumulated database becomes valuable property
  4. **Human resource development** : Acquisition of computational materials science skills

* * *

## 1.6 Workflow Design Examples

### Example 1: Band Gap Screening

**Goal** : Discover oxides with band gaps of 1.5-2.5 eV

**Workflow** :
    
    
    ```mermaid
    flowchart TD
        A[Structure generation] --> B[Structure optimization]
        B --> C[Static calculation]
        C --> D[Band structure calculation]
        D --> E[Band gap extraction]
        E --> F{1.5-2.5 eV?}
        F -->|Yes| G[Candidate material list]
        F -->|No| H[Exclude]
    
        B -->|Convergence failure| I[Error handling]
        I -->|Change settings| B
        I -->|Fatal| H
    ```

**Python pseudocode** :
    
    
    candidate_materials = []
    
    for formula in oxide_formulas:
        # Step 1: Structure generation
        structure = generate_structure(formula)
    
        # Step 2: Structure optimization
        try:
            relaxed = relax_structure(structure)
        except ConvergenceError:
            relaxed = relax_structure(structure, strict=False)
    
        # Step 3: Static calculation
        energy, forces = static_calculation(relaxed)
    
        # Step 4: Band structure
        band_gap = calculate_band_gap(relaxed)
    
        # Step 5: Filtering
        if 1.5 <= band_gap <= 2.5:
            candidate_materials.append({
                "formula": formula,
                "band_gap": band_gap,
                "energy": energy
            })
    

### Example 2: Thermodynamic Stability Screening

**Goal** : Discover materials with negative (stable) formation energy

**Workflow** :
    
    
    stable_materials = []
    
    for composition in compositions:
        # Formation energy calculation
        E_compound = calculate_energy(composition)
        E_elements = sum([calculate_energy(el) for el in composition.elements])
    
        E_formation = E_compound - E_elements
    
        if E_formation < 0:
            # Check decomposition energy too
            E_decomp = calculate_decomposition_energy(composition)
    
            if E_decomp > 0:  # Does not decompose
                stable_materials.append({
                    "composition": composition,
                    "E_formation": E_formation,
                    "E_decomp": E_decomp
                })
    

* * *

## 1.7 Exercises

### Exercise 1 (Difficulty: easy)

**Problem** : Consider band gap calculations for 100 materials. Estimate the time required for traditional methods (manual) and High-Throughput Computing.

**Conditions** : \- Calculation time per material: 8 hours (48-core parallelism) \- Manual work: 20 min input preparation, 5 min job submission, 10 min result check \- High-Throughput: Input preparation is automatic, 100 materials can be submitted simultaneously

Hint Manual case: \- Calculation time: 8 hours × 100 = 800 hours (serial) \- Manual work: 35 min × 100 = 3,500 min = 58.3 hours High-Throughput: \- Calculation time: 8 hours (parallel) \- Manual work: 1 hour (workflow setup only)  Solution **Traditional method**: \- Total time = Calculation time (serial) + Manual work \- = 800 hours + 58.3 hours \- = **858.3 hours ≈ 36 days** **High-Throughput**: \- Total time = Calculation time (parallel) + Initial setup \- = 8 hours + 1 hour \- = **9 hours** **Efficiency improvement**: 858.3 / 9 ≈ **95x** 

### Exercise 2 (Difficulty: medium)

**Problem** : Estimate the CPU time required to calculate Materials Project's 140,000 materials.

**Conditions** : \- Average per material: 8 hours structure optimization + 2 hours static calculation + 2 hours band structure = 12 hours \- Cores used: Average 48 cores/material

Hint Total CPU time = Number of materials × CPU time per material CPU time per material = Calculation time × Number of cores  Solution **Calculation**: 
    
    
    Total CPU time = 140,000 materials × 12 hours × 48 cores
             = 140,000 × 576 CPU hours
             = 80,640,000 CPU hours
             ≈ 80.64 million CPU hours
    

**Real-time conversion** (using 1,000 nodes, each with 48 cores): 
    
    
    Real time = 80.64M CPU hours / (1,000 nodes × 48 cores)
          = 1,680 hours
          ≈ 70 days
    

Materials Project has actually accumulated over 10+ years, averaging about 14,000 materials per year. 

### Exercise 3 (Difficulty: hard)

**Problem** : Propose a High-Throughput Computing system design.

**Scenario** : \- Goal: Screen 10,000 materials (within 6 months) \- Budget: $2,000,000 \- Calculation content: Structure optimization + static calculation (12 hours per material, 48 cores)

**Items to propose** : 1\. Required computational resources (number of nodes, cores) 2\. Workflow design (tool selection) 3\. Data management strategy 4\. Cost estimate

Hint 1\. Calculate total CPU time 2\. Determine required parallelism to complete in 6 months 3\. Compare cloud vs on-premise 4\. Consider tools like FireWorks  Solution **1. Required resources** Total CPU time: 
    
    
    10,000 materials × 12 hours × 48 cores = 5,760,000 CPU hours
    

To complete in 6 months (180 days, 24 hours operation): 
    
    
    Required cores = 5,760,000 / (180 days × 24 hours)
              = 1,333 cores
              ≈ 28 nodes (48 cores/node)
    

**2. Workflow design** \- **Tool**: FireWorks + Atomate \- Reason: Materials Project track record, VASP integration \- **Job Scheduler**: SLURM \- **Database**: MongoDB **3. Data management** \- Calculation data: 100MB per material → 1TB total \- Database: Metadata 10GB \- Backup: 2TB (redundancy) **4. Cost estimate** **Option A: Cloud (AWS)** | Item | Unit price | Quantity | Total | |----|------|-----|------| | EC2 (c5.12xlarge, 48 cores) | $2.04/hour | 28 nodes × 4,320 hours | $247,000 | | Storage (EBS) | $0.10/GB/month | 2TB × 6 months | $1,200 | | Data transfer | $0.09/GB | 500GB | $45 | | **Total** | | | **$248,245** | **Option B: On-premise HPC usage** | Item | Unit price | Quantity | Total | |----|------|-----|------| | HPC usage fee | $0.1/CPU hour | 5.76M CPU hours | $576,000 | | Personnel costs | $80,000/year | 1 person × 0.5 year | $40,000 | | Software | $300,000/year | 0.5 year | $150,000 | | **Total** | | | **$766,000** | **Recommendation**: Option B (On-premise HPC) \- Within budget ($2,000,000) \- Utilize university HPC clusters \- Invest surplus budget in experimental validation 

* * *

## 1.8 Summary

In this chapter, we learned about the necessity of High-Throughput Computing and principles of workflow design.

**Key points** :

  1. **Vastness of exploration space** : 10¹²-10⁶⁰ combinations
  2. **Four elements** : Automation, parallelization, standardization, data management
  3. **Success stories** : Materials Project (140k materials), AFLOW (3.5M structures)
  4. **Design principles** : Modularity, error handling, reproducibility, scalability
  5. **Cost reduction** : 89% cost reduction, 75% time reduction

**Next steps** :

In Chapter 2, we will practice **DFT calculation automation** using ASE and pymatgen. We will learn automatic generation of VASP and Quantum ESPRESSO input files, error detection and restart, and automated result analysis.

**[Chapter 2: DFT Calculation Automation →](<chapter-2.html>)**

* * *

* * *

## Data License and Citation

### Datasets Used

License information for databases mentioned in this chapter:

Database | License | Citation requirements | Access  
---|---|---|---  
Materials Project | CC BY 4.0 | Paper citation required | https://materialsproject.org  
AFLOW | Open data | Paper citation recommended | http://aflowlib.org  
OQMD | Open data | Paper citation recommended | http://oqmd.org  
JARVIS | NIST Public Data | Paper citation recommended | https://jarvis.nist.gov  
  
**Notes on data usage** : \- Always cite original papers when writing papers \- Check terms of use for each database for commercial use \- Original license applies to data redistribution

### Citation Methods

**When using Materials Project** :
    
    
    Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., ... & Persson, K. A. (2013).
    Commentary: The Materials Project: A materials genome approach to accelerating materials innovation.
    APL materials, 1(1), 011002.
    

**To cite this chapter** :
    
    
    Hashimoto, Y. (2025). "The Need for High-Throughput Computing and Workflow Design"
    High-Throughput Computing Introduction Series Chapter 1. Materials Informatics Dojo Project.
    

* * *

## Practical Pitfalls

Common problems and solutions when implementing High-Throughput Computing:

### Pitfall 1: Over-allocation of computational resources

**Problem** : Allocating maximum resources (48 cores, 24 hours) to all materials

**Symptoms** : \- Small structures (few atoms) still use 48 cores \- Actual calculation takes 1 hour but 24 hours reserved \- Resource waste, increased wait times

**Solution** :
    
    
    def estimate_resources(structure):
        """
        Estimate appropriate resources based on structure size
        """
        n_atoms = len(structure)
    
        if n_atoms < 10:
            return {'cores': 12, 'time': '4:00:00'}
        elif n_atoms < 50:
            return {'cores': 24, 'time': '12:00:00'}
        else:
            return {'cores': 48, 'time': '24:00:00'}
    
    # Use in SLURM script generation
    resources = estimate_resources(structure)
    

**Lesson** : Dynamically adjust resources according to structure size and k-point count

### Pitfall 2: Neglecting error logs

**Problem** : Not noticing that 20 out of 100 materials have failed

**Symptoms** : \- Only checking completion status, not verifying errors \- Treating unconverged calculations as "complete" \- Panic when noticed during paper writing

**Solution** :
    
    
    def comprehensive_check(directory):
        """
        Multi-faceted calculation health check
        """
        checks = {
            'file_exists': os.path.exists(f"{directory}/OUTCAR"),
            'converged': False,
            'energy_reasonable': False,
            'forces_converged': False
        }
    
        if checks['file_exists']:
            with open(f"{directory}/OUTCAR", 'r') as f:
                content = f.read()
    
                # Convergence check
                checks['converged'] = 'reached required accuracy' in content
    
                # Energy check (detect abnormal values)
                energy = extract_energy(content)
                checks['energy_reasonable'] = -100 < energy < 0  # eV/atom
    
                # Force convergence check
                max_force = extract_max_force(content)
                checks['forces_converged'] = max_force < 0.05  # eV/Å
    
        # Only successful if all True
        return all(checks.values()), checks
    
    # Usage example
    success, details = comprehensive_check('calculations/LiCoO2')
    if not success:
        print(f"Error details: {details}")
    

**Lesson** : Enforce quality control with automated checking scripts

### Pitfall 3: File system limits

**Problem** : 10,000 materials × 50 files each = 500,000 files make file system slow

**Symptoms** : \- `ls` command takes several minutes \- File deletion takes hours \- Backup failures

**Solution** :
    
    
    # Bad example
    calculations/
      ├── material_0001/
      ├── material_0002/
      ...
      └── material_10000/  # 10,000 directories at same level
    
    # Good example (hierarchical)
    calculations/
      ├── 00/
      │   ├── 00/material_0000/
      │   ├── 01/material_0001/
      │   ...
      │   └── 99/material_0099/
      ├── 01/
      │   ├── 00/material_0100/
      ...
    
    
    
    def get_hierarchical_path(material_id):
        """
        Generate hierarchical path
        """
        # material_id = 1234 → calculations/12/34/material_1234
        id_str = f"{material_id:06d}"
        level1 = id_str[:2]
        level2 = id_str[2:4]
    
        path = f"calculations/{level1}/{level2}/material_{id_str}"
        os.makedirs(path, exist_ok=True)
    
        return path
    

**Lesson** : Use hierarchical directory structures for large-scale calculations

### Pitfall 4: Network file system overload

**Problem** : All nodes writing to NFS simultaneously, causing I/O bottleneck

**Symptoms** : \- Calculations complete but waiting to write results \- Frequent file system errors \- Parallel efficiency below 20%

**Solution** :
    
    
    # Utilize local disk (fast)
    #!/bin/bash
    #SBATCH ...
    
    # Use local scratch directory
    LOCAL_SCRATCH=/scratch/job_${SLURM_JOB_ID}
    mkdir -p $LOCAL_SCRATCH
    
    # Calculate locally
    cd $LOCAL_SCRATCH
    cp $SLURM_SUBMIT_DIR/INCAR .
    cp $SLURM_SUBMIT_DIR/POSCAR .
    cp $SLURM_SUBMIT_DIR/KPOINTS .
    
    mpirun -np 48 vasp_std
    
    # Copy only results to NFS after completion
    cp OUTCAR CONTCAR vasprun.xml $SLURM_SUBMIT_DIR/
    

**Lesson** : Calculate on local storage, copy only results to shared storage

### Pitfall 5: Missing dependency records

**Problem** : Cannot reproduce environment when trying 6 months later

**Symptoms** : \- "It worked back then..." \- Library versions unknown \- Cannot recall calculation settings

**Solution** :
    
    
    import json
    from datetime import datetime
    import subprocess
    
    def record_environment():
        """
        Complete environment recording
        """
        env_record = {
            'timestamp': datetime.now().isoformat(),
            'python_version': subprocess.check_output(['python', '--version']).decode(),
            'packages': subprocess.check_output(['pip', 'freeze']).decode().split('\n'),
            'hostname': subprocess.check_output(['hostname']).decode().strip(),
            'slurm_version': subprocess.check_output(['sinfo', '--version']).decode(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        }
    
        with open('environment_snapshot.json', 'w') as f:
            json.dump(env_record, f, indent=2)
    
        return env_record
    
    # Record at calculation start
    record_environment()
    

**Lesson** : Record environment snapshots for all calculations

* * *

## Quality Checklist

Items to check before calculation start and after completion:

### Pre-calculation Checklist

**Project Planning** \- [ ] Estimated total CPU time (number of materials × time per material) \- [ ] Checked budget (if cloud) \- [ ] Calculated data storage capacity (assuming 100MB per material) \- [ ] Set completion date

**Workflow Design** \- [ ] Implemented error handling \- [ ] Added automatic restart functionality \- [ ] Prepared progress monitoring script \- [ ] Adopted hierarchical directory structure

**Calculation Settings** \- [ ] Explicitly stated convergence criteria (EDIFF, EDIFFG, etc.) \- [ ] Unified k-point density \- [ ] Determined energy cutoff \- [ ] Managed calculation settings with Git

**Reproducibility** \- [ ] Recorded software version \- [ ] Prepared environment setup script \- [ ] Created README.md \- [ ] Version controlled input file generation scripts

### Post-calculation Checklist

**Quality Control** \- [ ] Verified all calculations converged \- [ ] Confirmed energies in reasonable range (-100 ~ 0 eV/atom) \- [ ] Detected and investigated outliers \- [ ] Checked error logs for failed calculations

**Data Management** \- [ ] Saved results to database \- [ ] Backed up raw data (minimum 2 locations) \- [ ] Recorded metadata (date/time, settings) \- [ ] Made data searchable (JSON, MongoDB, etc.)

**Documentation** \- [ ] Documented calculation conditions \- [ ] Recorded failure causes and countermeasures \- [ ] Described reproduction procedures in README \- [ ] Created result summary

**Sharing and Publishing** \- [ ] Prepared data for publication on NOMAD, etc. \- [ ] Created figures and tables for papers \- [ ] Published code on GitHub (if possible) \- [ ] Obtained DOI (when publishing dataset)

* * *

## Code Reproducibility Specifications

Environment required to reproduce code examples in this chapter:

### Software Versions
    
    
    # Python environment
    Python 3.10+
    numpy==1.24.0
    scipy==1.10.0
    matplotlib==3.7.0
    pandas==2.0.0
    
    # DFT calculation codes (either)
    VASP 6.3.0 or higher (commercial license)
    Quantum ESPRESSO 7.0 or higher (open source)
    
    # Job scheduler
    SLURM 22.05 or higher
    or PBS Pro 2021+
    

### Verified Environments

**On-premise HPC** : \- TSUBAME3.0 (Tokyo Institute of Technology) \- Fugaku (RIKEN) \- University clusters (general SLURM systems)

**Cloud HPC** : \- AWS Parallel Cluster 3.6.0 \- Google Cloud HPC Toolkit 1.25.0

### Installation Script
    
    
    # conda environment setup
    conda create -n htc-env python=3.10
    conda activate htc-env
    
    # Essential packages
    pip install numpy scipy matplotlib pandas
    pip install ase pymatgen
    
    # Optional (workflow management)
    pip install fireworks atomate
    

### Troubleshooting

**Issue 1** : `ImportError: No module named 'ase'`
    
    
    # Solution
    pip install ase
    # or
    conda install -c conda-forge ase
    

**Issue 2** : VASP not found
    
    
    # Solution: Add VASP executable to PATH
    export PATH=/path/to/vasp/bin:$PATH
    # or add to .bashrc
    echo 'export PATH=/path/to/vasp/bin:$PATH' >> ~/.bashrc
    

**Issue 3** : SLURM commands unavailable
    
    
    # Solution: Login to HPC system
    ssh username@hpc.university.edu
    # SLURM commands cannot be used on local PC
    

* * *

## References

### Essential References (Cited in this chapter)

  1. **Materials Project** Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., ... & Persson, K. A. (2013). "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." _APL Materials_ , 1(1), 011002. DOI: [10.1063/1.4812323](<https://doi.org/10.1063/1.4812323>)

  2. **AFLOW** Curtarolo, S., Setyawan, W., Hart, G. L., Jahnatek, M., Chepulskii, R. V., Taylor, R. H., ... & Levy, O. (2012). "AFLOW: An automatic framework for high-throughput materials discovery." _Computational Materials Science_ , 58, 218-226. DOI: [10.1016/j.commatsci.2012.02.005](<https://doi.org/10.1016/j.commatsci.2012.02.005>)

  3. **OQMD** Saal, J. E., Kirklin, S., Aykol, M., Meredig, B., & Wolverton, C. (2013). "Materials design and discovery with high-throughput density functional theory: the open quantum materials database (OQMD)." _JOM_ , 65(11), 1501-1509. DOI: [10.1007/s11837-013-0755-4](<https://doi.org/10.1007/s11837-013-0755-4>)

  4. **JARVIS** Choudhary, K., Garrity, K. F., Reid, A. C., DeCost, B., Biacchi, A. J., Hight Walker, A. R., ... & Tavazza, F. (2020). "The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design." _npj Computational Materials_ , 6(1), 173. DOI: [10.1038/s41524-020-00440-1](<https://doi.org/10.1038/s41524-020-00440-1>)

  5. **Materials Genome Initiative** Materials Genome Initiative for Global Competitiveness (2011). Office of Science and Technology Policy, USA. URL: https://www.mgi.gov/

### Recommended References (Advanced Learning)

  6. **Theory of High-Throughput Computing** Hautier, G., Jain, A., & Ong, S. P. (2012). "From the computer to the laboratory: materials discovery and design using first-principles calculations." _Journal of Materials Science_ , 47(21), 7317-7340.

  7. **Workflow Design Practice** Mathew, K., Montoya, J. H., Faghaninia, A., Dwarakanath, S., Aykol, M., Tang, H., ... & Persson, K. A. (2017). "Atomate: A high-level interface to generate, execute, and analyze computational materials science workflows." _Computational Materials Science_ , 139, 140-152.

  8. **Parallel Computing Optimization** Gropp, W., Lusk, E., & Skjellum, A. (2014). _Using MPI: portable parallel programming with the message-passing interface._ MIT press.

### Online Resources

  * **Materials Project Documentation** : https://docs.materialsproject.org/
  * **AFLOW Tutorial** : http://aflowlib.org/tutorial/
  * **OQMD API** : http://oqmd.org/documentation/
  * **SLURM Documentation** : https://slurm.schedmd.com/documentation.html

* * *

## Next Steps

In Chapter 2, we will practice **DFT calculation automation** using ASE and pymatgen.

**[Chapter 2: DFT Calculation Automation →](<chapter-2.html>)**

* * *

**License** : CC BY 4.0 **Creation date** : 2025-10-17 **Last updated** : 2025-10-19 **Author** : Dr. Yusuke Hashimoto, Tohoku University **Version** : 1.1
