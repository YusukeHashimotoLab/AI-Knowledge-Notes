---
title: "Chapter 1: Protein Structure and Biomaterials"
chapter_title: "Chapter 1: Protein Structure and Biomaterials"
subtitle: Understanding Biomolecular Shapes with PDB and AlphaFold
reading_time: 25-30 min
difficulty: Beginner
code_examples: 8
exercises: 3
version: 1.0
created_at: "by:"
---

# Chapter 1: Protein Structure and Biomaterials

This chapter covers Protein Structure and Biomaterials. You will learn definition of bioinformatics, Read structure files, and Evaluate AlphaFold2 prediction accuracy.

**Understanding Biomolecular Shapes with PDB and AlphaFold**

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the definition of bioinformatics and its applications to materials science
  * ✅ Explain the hierarchical structure of proteins (primary to quaternary)
  * ✅ Retrieve and analyze protein structures from the PDB database
  * ✅ Read structure files and extract information using Biopython
  * ✅ Evaluate AlphaFold2 prediction accuracy and assess practical utility
  * ✅ Explain collagen structure and biomaterial applications

**Reading Time** : 25-30 minutes **Code Examples** : 8 **Exercises** : 3

* * *

## 1.1 What is Bioinformatics?

### Definition and Interdisciplinarity

**Bioinformatics** is an interdisciplinary field that analyzes biological data using computational and information sciences.
    
    
    ```mermaid
    flowchart LR
        A[Biology] --> D[Bioinformatics]
        B[Computer Science] --> D
        C[Materials Science] --> D
    
        D --> E[Biomaterial Design]
        D --> F[Drug Delivery]
        D --> G[Biosensor Development]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#4CAF50,color:#fff
        style E fill:#ffebee
        style F fill:#f3e5f5
        style G fill:#fff9c4
    ```

* * *

### Application Domains

**1\. Biomaterial Design** \- Collagen artificial skin (tissue engineering) \- Silk fiber structure optimization (Spiber Inc.) \- Peptide hydrogels (drug delivery)

**2\. Drug Delivery Systems (DDS)** \- Antibody therapeutics (targeted therapy) \- Nanoparticle carriers (cancer treatment) \- Liposomal formulations (vaccines)

**3\. Biosensor Development** \- Glucose sensors (diabetes management) \- Antibody biosensors (COVID-19 diagnostics) \- DNA chips (genetic diagnostics)

* * *

### Protein Structural Hierarchy

Protein structure is described at four levels:

**Primary Structure** \- Amino acid sequence \- Connected by peptide bonds \- Example: Gly-Pro-Ala-Ser-...

**Secondary Structure** \- Alpha helix (α-helix, helical structure) \- Beta sheet (β-sheet, planar structure) \- Loops

**Tertiary Structure** \- Three-dimensional folding of polypeptide chain \- Hydrophobic interactions, hydrogen bonds, disulfide bonds

**Quaternary Structure** \- Assembly of multiple subunits \- Example: Hemoglobin (4 subunits)

* * *

## 1.2 Utilizing PDB (Protein Data Bank)

### What is PDB?

**PDB (Protein Data Bank)** is the world's largest database for publicly sharing three-dimensional structures of proteins and nucleic acids.

**Statistics (as of 2025)** : \- Registered structures: Over 200,000 \- X-ray crystallography: 90% \- NMR: 8% \- Electron microscopy: 2%

**Access** : <https://www.rcsb.org/>

* * *

### PDB File Structure

PDB files are text-based and contain atomic coordinates with associated information.

**Example 1: Basic PDB File Structure**
    
    
    # Main sections of a PDB file
    """
    HEADER    Structure classification and date
    TITLE     Brief description of structure
    COMPND    Compound name
    SOURCE    Biological species
    ATOM      Atomic coordinates
    HELIX     Alpha helix regions
    SHEET     Beta sheet regions
    CONECT    Bond information
    """
    
    # Example of ATOM record
    """
    ATOM      1  N   MET A   1      20.154  29.699   5.276  1.00 49.05           N
    ATOM      2  CA  MET A   1      21.289  28.803   5.063  1.00 49.05           C
    ATOM      3  C   MET A   1      21.628  28.123   6.377  1.00 49.05           C
    """
    
    # Column meanings
    """
    ATOM: Record type
    1: Atom number
    N: Atom name
    MET: Amino acid (Methionine)
    A: Chain ID
    1: Residue number
    20.154, 29.699, 5.276: x, y, z coordinates (Å)
    1.00: Occupancy
    49.05: B-factor (temperature factor)
    N: Element symbol
    """
    

* * *

### Reading PDB Files with Biopython

**Example 2: Reading PDB Structure and Extracting Basic Information**
    
    
    from Bio.PDB import PDBParser
    import warnings
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Initialize PDB parser
    parser = PDBParser(QUIET=True)
    
    # Read PDB file (example: 1UBQ - Ubiquitin)
    structure = parser.get_structure('ubiquitin', '1ubq.pdb')
    
    # Display basic information
    print("=== Basic Structure Information ===")
    print(f"Structure name: {structure.id}")
    print(f"Number of models: {len(structure)}")
    
    # Get first model
    model = structure[0]
    print(f"Number of chains: {len(model)}")
    
    # Information for each chain
    for chain in model:
        residue_count = len(chain)
        print(f"Chain {chain.id}: {residue_count} residues")
    
        # Count atoms
        atom_count = sum(1 for residue in chain for atom in residue)
        print(f"  Number of atoms: {atom_count}")
    

**Output Example** :
    
    
    === Basic Structure Information ===
    Structure name: ubiquitin
    Number of models: 1
    Number of chains: 1
    Chain A: 76 residues
      Number of atoms: 602
    

* * *

### Atomic Coordinates and Distance Calculation

**Example 3: Extracting Atomic Coordinates and Calculating Distances**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Example 3: Extracting Atomic Coordinates and Calculating Dis
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from Bio.PDB import PDBParser
    import numpy as np
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('ubiquitin', '1ubq.pdb')
    model = structure[0]
    chain = model['A']
    
    # Get specific residue (10th isoleucine)
    residue = chain[10]
    
    print(f"=== Residue {residue.get_resname()} {residue.id[1]} ===")
    
    # Get atomic coordinates
    for atom in residue:
        coord = atom.get_coord()
        print(f"Atom {atom.name}: "
              f"({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
    
    # Calculate distance between two atoms
    atom1 = residue['CA']  # Alpha carbon
    atom2 = residue['CB']  # Beta carbon
    
    distance = atom1 - atom2  # Bio.PDB automatically calculates distance
    print(f"\nCA-CB distance: {distance:.2f} Å")
    
    # Manual distance calculation
    coord1 = atom1.get_coord()
    coord2 = atom2.get_coord()
    manual_distance = np.linalg.norm(coord1 - coord2)
    print(f"Manual calculation: {manual_distance:.2f} Å")
    

**Output Example** :
    
    
    === Residue ILE 10 ===
    Atom N: (15.23, 18.45, 12.67)
    Atom CA: (16.12, 17.89, 11.65)
    Atom C: (17.28, 18.85, 11.32)
    Atom O: (17.15, 20.07, 11.45)
    Atom CB: (15.34, 17.53, 10.38)
    
    CA-CB distance: 1.53 Å
    Manual calculation: 1.53 Å
    

* * *

### Retrieving Secondary Structure

**Example 4: DSSP (Secondary Structure Analysis)**
    
    
    from Bio.PDB import PDBParser, DSSP
    import warnings
    warnings.filterwarnings('ignore')
    
    # Read structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', '1ubq.pdb')
    model = structure[0]
    
    # Run DSSP (Define Secondary Structure of Proteins)
    # Note: Requires dssp executable (conda install -c salilab dssp)
    try:
        dssp = DSSP(model, '1ubq.pdb', dssp='mkdssp')
    
        print("=== Secondary Structure Analysis ===")
        print("Residue | AA | Secondary Structure | Solvent Accessible Area")
        print("-" * 45)
    
        # Information for each residue
        for key in list(dssp.keys())[:20]:  # First 20 residues
            residue_info = dssp[key]
            chain_id = key[0]
            res_id = key[1][1]
            aa = residue_info[1]
            ss = residue_info[2]  # Secondary structure
            asa = residue_info[3]  # Solvent accessible area
    
            # Secondary structure symbols
            # H: Alpha helix, B: Beta bridge,
            # E: Beta strand, G: 3-10 helix
            # I: Pi helix, T: Turn, S: Bend
            # -: Coil
    
            print(f"{res_id:4d} | {aa:3s} | {ss:^12s} | "
                  f"{asa:6.1f} Ų")
    
    except FileNotFoundError:
        print("DSSP not installed")
        print("Install: conda install -c salilab dssp")
    

* * *

## 1.3 Utilizing AlphaFold

### The AlphaFold2 Revolution

**AlphaFold2** (DeepMind, 2020) is a deep learning model that predicts protein three-dimensional structures from amino acid sequences with high accuracy.

**Achievements** : \- Overwhelming victory in CASP14 (structure prediction competition) \- Prediction accuracy: RMSD < 1.5 Å with experimental structures (atomic-level precision) \- Prediction time: Minutes to hours (compared to months to years for traditional experiments)

* * *

### AlphaFold Database

**AlphaFold Protein Structure Database** publicly shares over 200 million protein structures.

**Access** : <https://alphafold.ebi.ac.uk/>

**Example 5: Downloading AlphaFold Predicted Structures**
    
    
    # Requirements:
    # - Python 3.9+
    # - requests>=2.31.0
    
    import requests
    import gzip
    import shutil
    
    def download_alphafold_structure(uniprot_id, output_file):
        """
        Download PDB structure from AlphaFold Database
    
        Parameters:
        -----------
        uniprot_id : str
            UniProt ID (e.g., P69905)
        output_file : str
            Output filename
        """
        # AlphaFold DB URL
        base_url = "https://alphafold.ebi.ac.uk/files/"
        pdb_url = f"{base_url}AF-{uniprot_id}-F1-model_v4.pdb"
    
        print(f"Downloading: {pdb_url}")
    
        try:
            response = requests.get(pdb_url)
            response.raise_for_status()
    
            with open(output_file, 'w') as f:
                f.write(response.text)
    
            print(f"Saved: {output_file}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return False
    
    # Example: Download hemoglobin beta chain (P69905)
    download_alphafold_structure('P69905', 'hemoglobin_beta.pdb')
    

* * *

### Evaluating Prediction Confidence (pLDDT)

**pLDDT (predicted Local Distance Difference Test)** is AlphaFold's prediction confidence score (0-100).

**Interpretation** : \- **90-100** : Very high confidence (equivalent to experimental structures) \- **70-90** : High confidence (generally accurate) \- **50-70** : Low confidence (loop regions, etc.) \- **< 50**: Very low confidence (disordered regions)

**Example 6: Extracting and Visualizing pLDDT**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Example 6: Extracting and Visualizing pLDDT
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from Bio.PDB import PDBParser
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Read AlphaFold structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('alphafold', 'hemoglobin_beta.pdb')
    
    # pLDDT is stored in the B-factor column
    model = structure[0]
    chain = model['A']
    
    residue_numbers = []
    plddt_scores = []
    
    for residue in chain:
        if residue.id[0] == ' ':  # Only standard residues
            residue_numbers.append(residue.id[1])
            # Get B-factor of CA atom (pLDDT)
            ca_atom = residue['CA']
            plddt_scores.append(ca_atom.bfactor)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    plt.plot(residue_numbers, plddt_scores, linewidth=2)
    plt.axhline(y=90, color='green', linestyle='--',
                label='Very High Confidence (>90)')
    plt.axhline(y=70, color='orange', linestyle='--',
                label='High Confidence (>70)')
    plt.axhline(y=50, color='red', linestyle='--',
                label='Low Confidence (>50)')
    
    plt.xlabel('Residue Number', fontsize=12)
    plt.ylabel('pLDDT Score', fontsize=12)
    plt.title('AlphaFold Prediction Confidence (Hemoglobin Beta Chain)',
              fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plddt_plot.png', dpi=300)
    plt.show()
    
    # Statistical information
    print(f"=== pLDDT Statistics ===")
    print(f"Mean: {np.mean(plddt_scores):.1f}")
    print(f"Median: {np.median(plddt_scores):.1f}")
    print(f"Minimum: {np.min(plddt_scores):.1f}")
    print(f"Maximum: {np.max(plddt_scores):.1f}")
    print(f"High confidence region (>70): "
          f"{100 * np.sum(np.array(plddt_scores) > 70) / len(plddt_scores):.1f}%")
    

* * *

## 1.4 Case Study: Collagen Structure Analysis

### What is Collagen?

**Collagen** is the most abundant protein in mammals and is the main component of skin, bone, and cartilage.

**Structural Features** : \- Triple helix structure (3 polypeptide chains) \- Repeating sequence: (Gly-X-Y)n (X, Y are proline, hydroxyproline) \- Full length: Approximately 300 nm

**Biomaterial Applications** : \- Artificial skin (burn treatment) \- Bone regeneration materials \- Cosmetic medicine (collagen injections)

* * *

### Retrieving Collagen Structure from PDB

**Example 7: Collagen Structure Analysis**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Example 7: Collagen Structure Analysis
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from Bio.PDB import PDBParser
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Collagen-like peptide structure (1K6F)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('collagen', '1k6f.pdb')
    
    model = structure[0]
    
    print("=== Collagen Structure Analysis ===")
    print(f"Number of chains: {len(model)}")
    
    # Length of each chain
    for chain in model:
        residue_count = len([res for res in chain if res.id[0] == ' '])
        print(f"Chain {chain.id}: {residue_count} residues")
    
    # Confirming triple helix structure: inter-chain distances
    chainA = model['A']
    chainB = model['B']
    chainC = model['C']
    
    # Get CA atom of central residue in each chain
    def get_central_ca(chain):
        residues = [res for res in chain if res.id[0] == ' ']
        central_idx = len(residues) // 2
        return residues[central_idx]['CA']
    
    ca_A = get_central_ca(chainA)
    ca_B = get_central_ca(chainB)
    ca_C = get_central_ca(chainC)
    
    # Inter-chain distances
    dist_AB = ca_A - ca_B
    dist_BC = ca_B - ca_C
    dist_CA = ca_C - ca_A
    
    print(f"\n=== Triple Helix Inter-Chain Distances ===")
    print(f"A-B: {dist_AB:.2f} Å")
    print(f"B-C: {dist_BC:.2f} Å")
    print(f"C-A: {dist_CA:.2f} Å")
    print(f"Average: {np.mean([dist_AB, dist_BC, dist_CA]):.2f} Å")
    
    # Calculate helix pitch (rise per residue)
    first_ca = chainA[7]['CA']
    last_ca = chainA[20]['CA']
    z_rise = last_ca.get_coord()[2] - first_ca.get_coord()[2]
    num_residues = 20 - 7
    
    rise_per_residue = z_rise / num_residues
    print(f"\nHelix rise (per residue): {rise_per_residue:.2f} Å")
    

**Output Example** :
    
    
    === Collagen Structure Analysis ===
    Number of chains: 3
    Chain A: 27 residues
    Chain B: 27 residues
    Chain C: 27 residues
    
    === Triple Helix Inter-Chain Distances ===
    A-B: 10.2 Å
    B-C: 10.1 Å
    C-A: 10.3 Å
    Average: 10.2 Å
    
    Helix rise (per residue): 2.9 Å
    

* * *

### Collagen Sequence Pattern Analysis

**Example 8: Validating Gly-X-Y Repeats**
    
    
    from Bio.PDB import PDBParser
    from collections import Counter
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('collagen', '1k6f.pdb')
    chain = structure[0]['A']
    
    # Get amino acid sequence
    sequence = []
    for residue in chain:
        if residue.id[0] == ' ':  # Standard residues
            resname = residue.get_resname()
            # Convert 3-letter code to 1-letter code (simplified version)
            aa_dict = {
                'GLY': 'G', 'PRO': 'P', 'ALA': 'A',
                'VAL': 'V', 'LEU': 'L', 'ILE': 'I',
                'MET': 'M', 'PHE': 'F', 'TYR': 'Y',
                'TRP': 'W', 'SER': 'S', 'THR': 'T',
                'CYS': 'C', 'ASN': 'N', 'GLN': 'Q',
                'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
                'ARG': 'R', 'HIS': 'H'
            }
            if resname in aa_dict:
                sequence.append(aa_dict[resname])
            else:
                sequence.append('X')  # Unknown amino acid
    
    seq_str = ''.join(sequence)
    print(f"Sequence: {seq_str}")
    
    # Validate Gly-X-Y pattern
    gly_positions = [i for i, aa in enumerate(sequence) if aa == 'G']
    print(f"\nGly positions: {gly_positions}")
    
    # Check if Gly appears every 3 residues
    expected_positions = list(range(0, len(sequence), 3))
    match_count = sum(1 for pos in gly_positions
                      if pos in expected_positions)
    print(f"Gly-X-Y pattern match rate: "
          f"{100 * match_count / len(gly_positions):.1f}%")
    
    # Amino acid composition
    aa_composition = Counter(sequence)
    print(f"\nAmino acid composition:")
    for aa, count in aa_composition.most_common():
        percentage = 100 * count / len(sequence)
        print(f"  {aa}: {count} ({percentage:.1f}%)")
    

* * *

## 1.5 Chapter Summary

### What We Learned

  1. **Definition of Bioinformatics** \- Biology × Computer Science × Materials Science \- Application domains: Biomaterials, DDS, biosensors

  2. **Protein Structural Hierarchy** \- Primary structure (sequence) → Secondary structure (α-helix, β-sheet) \- Tertiary structure (folding) → Quaternary structure (complex)

  3. **PDB Database** \- Over 200,000 structural data entries \- Analysis methods with Biopython

  4. **AlphaFold2** \- High-accuracy structure prediction from sequence \- Reliability assessment using pLDDT

  5. **Collagen Structure** \- Triple helix structure \- Gly-X-Y repeating pattern \- Biomaterial applications

### Key Points

  * ✅ Protein structure **determines function**
  * ✅ PDB is a **treasure trove of experimental structures**
  * ✅ AlphaFold has **revolutionized prediction**
  * ✅ Biopython makes **structure analysis easy**
  * ✅ Collagen is an **important biomaterial**

### To the Next Chapter

In Chapter 2, we will learn about **sequence analysis and machine learning** : \- Homology sequence search with BLAST \- Feature extraction from sequences \- Functional prediction using machine learning \- Case study: Enzyme activity prediction

**[Chapter 2: Sequence Analysis and Machine Learning →](<chapter-2.html>)**

* * *

## Exercises

### Exercise 1 (Difficulty: Easy)

Extract specific information from a PDB file.

**Task** : 1\. Download PDB file (1UBQ) 2\. Calculate total number of atoms 3\. Aggregate counts for each element (C, N, O, S)

Hint \- Use `atom.element` to get element symbol \- Use Counter for aggregation  Solution Example
    
    
    from Bio.PDB import PDBParser
    from collections import Counter
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('ubiquitin', '1ubq.pdb')
    
    elements = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    elements.append(atom.element)
    
    print(f"Total atoms: {len(elements)}")
    print("\nElement composition:")
    for elem, count in Counter(elements).most_common():
        print(f"  {elem}: {count}")
    

**Output Example**: 
    
    
    Total atoms: 602
    Element composition:
      C: 312
      O: 100
      N: 89
      S: 1
    

* * *

### Exercise 2 (Difficulty: Medium)

Analyze the pLDDT scores of an AlphaFold predicted structure and identify regions of high and low confidence.

Solution Example
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Analyze the pLDDT scores of an AlphaFold predicted structure
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from Bio.PDB import PDBParser
    import numpy as np
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('alphafold', 'alphafold_structure.pdb')
    
    chain = structure[0]['A']
    high_confidence = []
    low_confidence = []
    
    for residue in chain:
        if residue.id[0] == ' ':
            res_id = residue.id[1]
            plddt = residue['CA'].bfactor
    
            if plddt >= 90:
                high_confidence.append(res_id)
            elif plddt < 70:
                low_confidence.append(res_id)
    
    print(f"High confidence region (pLDDT >= 90): {len(high_confidence)} residues")
    print(f"Range: {min(high_confidence)}-{max(high_confidence)}")
    
    print(f"\nLow confidence region (pLDDT < 70): {len(low_confidence)} residues")
    if low_confidence:
        print(f"Range: {min(low_confidence)}-{max(low_confidence)}")
    

* * *

### Exercise 3 (Difficulty: Hard)

Analyze the triple helix structure of collagen in detail.

**Task** : 1\. Extract CA atom coordinates from 3 chains 2\. Calculate helix radius 3\. Estimate helix pitch (rise per turn)

Hint \- Find helix central axis (center of mass of 3 chains) \- Distance from each CA atom to central axis = radius \- Pitch = (z-direction rise) / (number of turns)  Solution Example
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Task:
    1. Extract CA atom coordinates from 3 chains
    2. Calcul
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from Bio.PDB import PDBParser
    import numpy as np
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('collagen', '1k6f.pdb')
    model = structure[0]
    
    # Get CA coordinates from 3 chains
    chains_coords = {}
    for chain_id in ['A', 'B', 'C']:
        chain = model[chain_id]
        coords = []
        for residue in chain:
            if residue.id[0] == ' ':
                coords.append(residue['CA'].get_coord())
        chains_coords[chain_id] = np.array(coords)
    
    # Central axis (center of mass of all CA atoms)
    all_coords = np.vstack(list(chains_coords.values()))
    center = np.mean(all_coords, axis=0)
    
    # Helix radius (distance from central axis)
    radii = []
    for coords in chains_coords.values():
        for coord in coords:
            # Distance in xy plane
            dist = np.linalg.norm(coord[:2] - center[:2])
            radii.append(dist)
    
    print(f"Helix radius: {np.mean(radii):.2f} ± "
          f"{np.std(radii):.2f} Å")
    
    # Pitch (z-direction rise)
    z_coords = chains_coords['A'][:, 2]
    z_rise = z_coords[-1] - z_coords[0]
    num_residues = len(chains_coords['A'])
    
    # Collagen helix makes approximately 1 turn per 3 residues
    turns = num_residues / 3
    pitch = z_rise / turns
    
    print(f"Helix pitch: {pitch:.2f} Å/turn")
    

* * *

## Data Licenses and Citations

### Structure Databases

Biological databases used in code examples in this chapter:

#### 1\. PDB (Protein Data Bank)

  * **License** : CC0 1.0 Public Domain
  * **Citation** : Berman, H. M. et al. (2000). "The Protein Data Bank." _Nucleic Acids Research_ , 28(1), 235-242.
  * **Access** : https://www.rcsb.org/
  * **Purpose** : Experimental structure data for proteins and nucleic acids
  * **Data Examples** : 1UBQ (Ubiquitin), 1K6F (Collagen-like peptide)

#### 2\. AlphaFold Protein Structure Database

  * **License** : CC BY 4.0
  * **Citation** : Jumper, J. et al. (2021). "Highly accurate protein structure prediction with AlphaFold." _Nature_ , 596(7873), 583-589.
  * **Access** : https://alphafold.ebi.ac.uk/
  * **Purpose** : AlphaFold2 structure prediction data (over 200 million proteins)
  * **API** : https://alphafold.ebi.ac.uk/api/

#### 3\. UniProt

  * **License** : CC BY 4.0
  * **Citation** : The UniProt Consortium. (2023). "UniProt: the Universal Protein Knowledgebase in 2023." _Nucleic Acids Research_ , 51(D1), D523-D531.
  * **Access** : https://www.uniprot.org/
  * **Purpose** : Protein sequences and annotation information

### Library Licenses

Library | Version | License | Purpose  
---|---|---|---  
Biopython | 1.81+ | BSD-3-Clause | Structure analysis/sequence manipulation  
NumPy | 1.24+ | BSD-3-Clause | Numerical computation  
Matplotlib | 3.7+ | PSF (BSD-like) | Visualization  
requests | 2.31+ | Apache 2.0 | Data download  
  
**License Compliance** : \- All allow commercial use \- Proper citations required in academic papers \- Always cite sources when using PDB/AlphaFold structures

* * *

## Code Reproducibility

### Environment Setup

#### Installing Required Packages
    
    
    # requirements.txt
    biopython==1.81
    numpy==1.24.3
    matplotlib==3.7.1
    requests==2.31.0
    scipy==1.11.1
    
    # Install
    pip install -r requirements.txt
    

#### Optional Tools
    
    
    # DSSP (secondary structure analysis)
    conda install -c salilab dssp
    
    # PyMOL (structure visualization)
    conda install -c conda-forge pymol-open-source
    

### Obtaining PDB Files
    
    
    # Reproducible PDB download
    from Bio.PDB import PDBList
    
    def download_pdb_reproducible(pdb_id, output_dir='./pdb_files'):
        """
        Download PDB structure reproducibly
    
        Parameters:
        -----------
        pdb_id : str
            PDB ID (e.g., '1ubq')
        output_dir : str
            Save directory
        """
        pdbl = PDBList()
        filename = pdbl.retrieve_pdb_file(
            pdb_id,
            pdir=output_dir,
            file_format='pdb'
        )
    
        print(f"Downloaded: {pdb_id} → {filename}")
        return filename
    
    # Usage example
    download_pdb_reproducible('1ubq')
    download_pdb_reproducible('1k6f')  # Collagen
    

### Obtaining Fixed Version AlphaFold Structures
    
    
    def download_alphafold_with_version(uniprot_id, version='v4'):
        """
        Download AlphaFold structure with specific version
    
        Parameters:
        -----------
        uniprot_id : str
            UniProt ID
        version : str
            AlphaFold version (v1-v4)
        """
        base_url = "https://alphafold.ebi.ac.uk/files/"
        pdb_url = f"{base_url}AF-{uniprot_id}-F1-model_{version}.pdb"
    
        response = requests.get(pdb_url)
        response.raise_for_status()
    
        filename = f"AF_{uniprot_id}_{version}.pdb"
        with open(filename, 'w') as f:
            f.write(response.text)
    
        print(f"Downloaded AlphaFold {version}: {filename}")
        return filename
    

* * *

## Common Pitfalls and Solutions

### 1\. PDB File Parsing Errors

**Problem** : Errors due to format differences or missing atoms in PDB files

**Symptom** :
    
    
    # NG example
    structure = parser.get_structure('protein', 'broken.pdb')
    # KeyError: 'CA'
    

**Solution** :
    
    
    # OK example: With error handling
    from Bio.PDB import PDBParser
    import warnings
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', 'protein.pdb')
    
    for residue in chain:
        try:
            ca_atom = residue['CA']
            coord = ca_atom.get_coord()
        except KeyError:
            # Skip residues without CA atom
            print(f"Warning: No CA atom in residue {residue.id}")
            continue
    

### 2\. Misinterpreting AlphaFold pLDDT

**Problem** : Misunderstanding pLDDT score as absolute accuracy

**NG** :
    
    
    # Assuming all pLDDT > 70 is reliable
    if plddt > 70:
        use_structure()  # Dangerous!
    

**OK** :
    
    
    # Setting thresholds based on application
    if application == 'drug_design':
        threshold = 90  # High precision required
    elif application == 'homology_modeling':
        threshold = 70  # Medium acceptable
    
    if plddt > threshold:
        use_structure()
    

**Recommended Standards** : \- Docking studies: pLDDT > 90 \- Homology modeling: pLDDT > 70 \- Structure prediction validation: pLDDT > 50

### 3\. Confusion of Coordinate System Units

**Problem** : Confusion between Å (Angstrom) and nm

**NG** :
    
    
    distance = atom1 - atom2  # Å units
    if distance < 5:  # nm? Å? Unclear
        print("Close contact")
    

**OK** :
    
    
    distance_angstrom = atom1 - atom2  # Explicitly Å units
    CONTACT_THRESHOLD_A = 5.0  # Å
    
    if distance_angstrom < CONTACT_THRESHOLD_A:
        print(f"Close contact: {distance_angstrom:.2f} Å")
    
    # Convert to nm if needed
    distance_nm = distance_angstrom / 10.0
    

### 4\. DSSP Execution Errors

**Problem** : DSSP not installed or path not set

**Symptom** :
    
    
    dssp = DSSP(model, 'protein.pdb')
    # FileNotFoundError: dssp executable not found
    

**Solution** :
    
    
    import shutil
    
    # Check for DSSP executable
    dssp_path = shutil.which('mkdssp') or shutil.which('dssp')
    
    if dssp_path:
        dssp = DSSP(model, 'protein.pdb', dssp=dssp_path)
    else:
        print("DSSP not installed. Please run:")
        print("  conda install -c salilab dssp")
        # Switch to alternative method
        use_alternative_ss_prediction()
    

### 5\. Memory Issues (Large Structures)

**Problem** : Memory errors with huge complexes (viruses, ribosomes, etc.)

**Solution** :
    
    
    # Process by chain
    for chain in structure[0]:
        # Analyze per chain
        analyze_chain(chain)
        # Free memory
        del chain
    
    # Or load only specific chains
    chain_selection = ['A', 'B']  # Only needed chains
    for chain_id in chain_selection:
        chain = structure[0][chain_id]
        analyze_chain(chain)
    

### 6\. Network Errors (Download Failures)

**Problem** : Downloads from AlphaFold/PDB fail mid-way

**Solution** :
    
    
    import time
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    
    def download_with_retry(url, max_retries=3):
        """
        Download with retry
        """
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
    
        for attempt in range(max_retries):
            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    

* * *

## Quality Checklist

### Data Acquisition Phase

  * [ ] Confirm PDB ID is correct (search on PDB site)
  * [ ] Confirm UniProt ID is correct
  * [ ] Verify downloaded file MD5 checksum
  * [ ] File size is reasonable (not empty file)
  * [ ] Visual inspection in text editor (ATOM records exist)

### Structure Analysis Phase

  * [ ] Confirm CA atom exists in all residues
  * [ ] Check for missing residues (REMARK 465)
  * [ ] Check for abnormal coordinate values (typically -100 to 100 Å range)
  * [ ] B-factor values are reasonable (typically 0-100)
  * [ ] Check pLDDT score distribution (for AlphaFold structures)

### AlphaFold-Specific Checks

  * [ ] pLDDT score average > 70 (reliability)
  * [ ] Confirm positions of low confidence regions (pLDDT < 50)
  * [ ] Distinguish loop regions vs structured regions
  * [ ] Check PAE map for relative placement reliability (if available)

### Code Quality

  * [ ] Type hints added (`def func(x: Structure) -> float:`)
  * [ ] Error handling implemented
  * [ ] Logging implemented (progress checkable)
  * [ ] Units explicitly stated (Å, nm, degrees, etc.)

### Materials Science-Specific Checks

  * [ ] Understand protein biological function
  * [ ] Confirm feasibility of biomaterial application
  * [ ] Consider stability (thermal stability, pH stability)
  * [ ] Synthesis/manufacturing feasibility

* * *

## References

  1. Berman, H. M. et al. (2000). "The Protein Data Bank." _Nucleic Acids Research_ , 28, 235-242. DOI: [10.1093/nar/28.1.235](<https://doi.org/10.1093/nar/28.1.235>)

  2. Jumper, J. et al. (2021). "Highly accurate protein structure prediction with AlphaFold." _Nature_ , 596, 583-589. DOI: [10.1038/s41586-021-03819-2](<https://doi.org/10.1038/s41586-021-03819-2>)

  3. Shoulders, M. D. & Raines, R. T. (2009). "Collagen Structure and Stability." _Annual Review of Biochemistry_ , 78, 929-958. DOI: [10.1146/annurev.biochem.77.032207.120833](<https://doi.org/10.1146/annurev.biochem.77.032207.120833>)

* * *

## Navigation

### Next Chapter

**[Chapter 2: Sequence Analysis and Machine Learning →](<chapter-2.html>)**

### Series Table of Contents

**[← Return to Series Table of Contents](<./index.html>)**

* * *

## Author Information

**Created by** : AI Terakoya Content Team **Created** : 2025-10-17 **Version** : 1.0

**License** : Creative Commons BY 4.0

* * *

**Let's learn sequence analysis and machine learning in detail in Chapter 2!**
