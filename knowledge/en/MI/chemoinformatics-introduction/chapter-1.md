---
title: Molecular Representation and RDKit Fundamentals
chapter_title: Molecular Representation and RDKit Fundamentals
subtitle: 
reading_time: 25-30 min
difficulty: Beginner
code_examples: 10
exercises: 4
version: 1.0
created_at: 2025-10-17
---

# Chapter 1: Molecular Representation and RDKit Fundamentals

This chapter covers the fundamentals of Molecular Representation and RDKit Fundamentals, which what you will learn in this chapter. You will learn essential concepts and techniques.

## What You Will Learn in This Chapter

In this chapter, we will learn the fundamentals of chemoinformatics, focusing on computational representation of molecules and basic molecular manipulation using RDKit.

### Learning Objectives

  * ✅ Explain the definition and application areas of chemoinformatics
  * ✅ Understand major molecular representation methods including SMILES, InChI, and molecular graphs
  * ✅ Read, visualize, and edit molecules using RDKit
  * ✅ Perform substructure searches using SMARTS
  * ✅ Retrieve and process molecular information from pharmaceutical databases

* * *

## 1.1 What is Chemoinformatics?

### Definition

**Chemoinformatics** is an interdisciplinary field that combines chemistry and data science, encompassing techniques for managing, analyzing, and predicting chemical information using computational methods.
    
    
    ```mermaid
    flowchart TD
        A[Chemoinformatics] --> B[Chemistry]
        A --> C[Computer Science]
        A --> D[Data Science]
        A --> E[Machine Learning]
    
        B --> F[Organic Chemistry]
        B --> G[Medicinal Chemistry]
        B --> H[Materials Chemistry]
    
        C --> I[Molecular Representation]
        C --> J[Databases]
        C --> K[Algorithms]
    
        style A fill:#4CAF50,color:#fff
    ```

### Application Areas

Chemoinformatics is widely used in the following areas:

  1. **Drug Discovery** \- Design and optimization of new drug candidates \- Virtual screening to narrow down candidate compounds \- ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) prediction \- Side effect prediction and drug-drug interaction analysis

  2. **Organic Materials Development** \- Design of organic semiconductor materials \- Prediction of luminescence properties for OLED materials \- Design of conductive polymers \- Materials exploration for dye-sensitized solar cells

  3. **Catalyst Design** \- Activity prediction for organocatalysts \- Ligand design and optimization \- Optimization of reaction conditions \- Selectivity prediction

  4. **Polymer Design** \- Polymer property prediction \- Monomer composition optimization \- Thermal property prediction \- Mechanical property prediction

### Differences from Materials Informatics (MI)

Aspect | Chemoinformatics (CI) | Materials Informatics (MI)  
---|---|---  
**Target** | Molecules (organic compounds, pharmaceuticals) | Materials in general (inorganic materials, alloys, ceramics)  
**Representation** | SMILES, InChI, molecular graphs | Crystal structure, composition, microstructure  
**Descriptors** | Molecular descriptors, fingerprints | Compositional descriptors, structural descriptors  
**Applications** | Drug discovery, organic materials, catalysts | Alloys, ceramics, battery materials  
**Complementarity** | Both fields converge in organic materials development | Organic-inorganic hybrid materials  
  
### Historical Background
    
    
    ```mermaid
    timeline
        title History of Chemoinformatics
        1960s : Early molecular representations (WLN)
        1970s : Development of SMILES notation (David Weininger)
        1980s : Establishment of 3D QSAR methods
        1990s : Proliferation of High-Throughput Screening (HTS)
        2000s : Large-scale chemical databases (PubChem, ChEMBL)
        2010s : Full-scale introduction of machine learning
        2020s : Application of deep learning, GNN, Transformers
    ```

**1970s - Birth of SMILES** : SMILES (Simplified Molecular Input Line Entry System), developed by David Weininger, became the foundation of chemoinformatics as an innovative method for representing molecules as character strings.

**2000s - Data Explosion** : The emergence of large-scale databases such as PubChem (2004) and ChEMBL (2009) enabled data-driven molecular design.

**2020s - AI Revolution** : The introduction of deep learning methods such as Graph Neural Networks (GNN) and Transformers has made possible the prediction of complex structure-activity relationships that were previously impossible.

* * *

## 1.2 Fundamentals of Molecular Representation

To handle molecules computationally, appropriate representation methods are necessary. Here we will learn four major representation methods.

### 1.2.1 SMILES Notation

**SMILES (Simplified Molecular Input Line Entry System)** is a linear notation that represents molecular structure as a character string.

#### Basic Rules

Symbol | Meaning | Example  
---|---|---  
C, N, O, S, P | Atoms | C (carbon), N (nitrogen)  
- | Single bond (usually omitted) | C-C → CC  
= | Double bond | C=C  
# | Triple bond | C#C  
( ) | Branching | CC(C)C  
[ ] | Detailed atom specification | [CH3]  
@ | Stereochemistry | C[C@H]N  
  
#### Specific Examples
    
    
    # SMILES of representative molecules
    
    Molecule       SMILES          Description
    ----------------------------------------------
    Methane        C               Simplest hydrocarbon
    Ethanol        CCO             Linear C-C-O structure
    Acetic acid    CC(=O)O         Carboxylic acid (with branching)
    Benzene        c1ccccc1        Aromatic ring (lowercase = aromatic)
    Toluene        Cc1ccccc1       Benzene with methyl group
    Aspirin        CC(=O)Oc1ccccc1C(=O)O  Analgesic antipyretic
    Caffeine       CN1C=NC2=C1C(=O)N(C(=O)N2C)C  Stimulant
    

#### Code Example 1: Creating Molecules from SMILES
    
    
    # RDKit installation (first time only)
    # conda install -c conda-forge rdkit
    # pip install rdkit  # Also available via pip
    
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    # Create molecule object from SMILES
    smiles = "CCO"  # Ethanol
    mol = Chem.MolFromSmiles(smiles)
    
    # Basic molecular information
    print(f"Molecular formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    print(f"Molecular weight: {Chem.rdMolDescriptors.CalcExactMolWt(mol):.2f}")
    print(f"Number of atoms: {mol.GetNumAtoms()}")
    print(f"Number of bonds: {mol.GetNumBonds()}")
    
    # Draw molecule
    img = Draw.MolToImage(mol, size=(300, 300))
    img.save("ethanol.png")
    

**Output Example:**
    
    
    Molecular formula: C2H6O
    Molecular weight: 46.04
    Number of atoms: 3
    Number of bonds: 2
    

#### Code Example 2: Batch Drawing of Multiple Molecules
    
    
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    # SMILES of major pharmaceuticals
    drugs = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1"
    }
    
    # Create list of molecule objects
    mols = [Chem.MolFromSmiles(smi) for smi in drugs.values()]
    legends = list(drugs.keys())
    
    # Grid display
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=2,
        subImgSize=(300, 300),
        legends=legends
    )
    img.save("common_drugs.png")
    

### 1.2.2 InChI/InChIKey

**InChI (International Chemical Identifier)** is a molecular identifier standardized by IUPAC.

#### InChI vs SMILES

Aspect | SMILES | InChI  
---|---|---  
**Uniqueness** | Multiple SMILES for same molecule | One InChI per molecule  
**Human Readability** | Relatively readable | Difficult to read  
**Standardization** | Non-standard | IUPAC standard  
**Use Cases** | Molecular input, visualization | Database search, identification  
  
#### Code Example 3: Generating InChI and InChIKey
    
    
    from rdkit import Chem
    
    # Create molecule from SMILES
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    mol = Chem.MolFromSmiles(smiles)
    
    # Generate InChI
    inchi = Chem.MolToInchi(mol)
    print(f"InChI: {inchi}")
    
    # InChIKey (fixed-length 27-character hash)
    inchi_key = Chem.MolToInchiKey(mol)
    print(f"InChIKey: {inchi_key}")
    
    # Convert InChI back to SMILES
    mol_from_inchi = Chem.MolFromInchi(inchi)
    smiles_regenerated = Chem.MolToSmiles(mol_from_inchi)
    print(f"Regenerated SMILES: {smiles_regenerated}")
    

**Output Example:**
    
    
    InChI: InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)
    InChIKey: BSYNRYMUTXBXSQ-UHFFFAOYSA-N
    Regenerated SMILES: CC(=O)Oc1ccccc1C(=O)O
    

### 1.2.3 Molecular Graphs

Molecules can be represented as **graph structures** : \- **Nodes (vertices)** : Atoms \- **Edges** : Chemical bonds
    
    
    ```mermaid
    flowchart LR
        C1((C)) --|Single bond| C2((C))
        C2 --|Single bond| O((O))
    
        style C1 fill:#4CAF50,color:#fff
        style C2 fill:#4CAF50,color:#fff
        style O fill:#F44336,color:#fff
    ```

#### Code Example 4: Analyzing Molecular Graphs
    
    
    # Requirements:
    # - Python 3.9+
    # - networkx>=3.1.0
    
    """
    Example: Code Example 4: Analyzing Molecular Graphs
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from rdkit import Chem
    import networkx as nx
    
    # Create molecule from SMILES
    smiles = "CCO"  # Ethanol
    mol = Chem.MolFromSmiles(smiles)
    
    # Convert to NetworkX graph
    G = nx.Graph()
    
    # Add nodes (atoms)
    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            symbol=atom.GetSymbol(),
            degree=atom.GetDegree()
        )
    
    # Add edges (bonds)
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=str(bond.GetBondType())
        )
    
    # Basic graph information
    print(f"Number of nodes (atoms): {G.number_of_nodes()}")
    print(f"Number of edges (bonds): {G.number_of_edges()}")
    print(f"Degree centrality: {nx.degree_centrality(G)}")
    
    # Information for each atom
    for idx, data in G.nodes(data=True):
        print(f"Atom {idx}: {data['symbol']}, degree {data['degree']}")
    

**Output Example:**
    
    
    Number of nodes (atoms): 3
    Number of edges (bonds): 2
    Degree centrality: {0: 0.5, 1: 1.0, 2: 0.5}
    Atom 0: C, degree 1
    Atom 1: C, degree 2
    Atom 2: O, degree 1
    

### 1.2.4 3D Structure and Stereochemistry

The 3D structure of molecules significantly affects biological activity and physical properties.

#### Code Example 5: Generating and Visualizing 3D Structures
    
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import PyMol
    
    # Create molecule from SMILES
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
    mol = Chem.MolFromSmiles(smiles)
    
    # Generate 3D coordinates (ETKDG method)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    
    # Optimize with molecular force field (MMFF94)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Save 3D structure (mol2 format)
    writer = Chem.rdmolfiles.MolToMol2Block(mol)
    with open("ibuprofen_3d.mol2", "w") as f:
        f.write(writer)
    
    print("3D structure generated")
    
    # Get atomic coordinates
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        print(
            f"{atom.GetSymbol()} {i}: "
            f"({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})"
        )
    

**Column: Importance of Stereochemistry**

Enantiomers have the same molecular formula and structural formula but have 3D structures that are mirror images of each other.

**Example: Thalidomide Tragedy** \- (R)-Thalidomide: Sedative effect (effective) \- (S)-Thalidomide: Teratogenicity (harmful)

Accurate representation of stereochemistry has life-or-death importance in drug discovery.

* * *

## 1.3 Basic Operations with RDKit

RDKit (**R** apidly **D** eveloping **Kit**) is the most widely used Python library for chemoinformatics.

### Environment Setup

#### Option 1: Anaconda (Recommended)
    
    
    # Create virtual environment
    conda create -n cheminf python=3.11
    
    # Activate environment
    conda activate cheminf
    
    # Install RDKit
    conda install -c conda-forge rdkit
    
    # Additional libraries
    conda install -c conda-forge mordred pandas matplotlib \
      seaborn scikit-learn
    

#### Option 2: Google Colab (No Installation Required)
    
    
    # Run the following in Google Colab
    !pip install rdkit mordred
    

### 1.3.1 Reading and Visualizing Molecules

#### Code Example 6: Reading from Various Input Formats
    
    
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    # Method 1: From SMILES
    mol1 = Chem.MolFromSmiles("c1ccccc1")
    
    # Method 2: From InChI
    mol2 = Chem.MolFromInchi(
        "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"
    )
    
    # Method 3: From MOL file
    # mol3 = Chem.MolFromMolFile("benzene.mol")
    
    # Method 4: From SMARTS (substructure pattern)
    pattern = Chem.MolFromSmarts("c1ccccc1")
    
    # Configure drawing options
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.ipython_useSVG = True  # High quality SVG format
    
    # Display atom numbers
    for atom in mol1.GetAtoms():
        atom.SetProp("atomLabel", str(atom.GetIdx()))
    
    img = Draw.MolToImage(mol1, size=(400, 400))
    img.save("benzene_labeled.png")
    

### 1.3.2 Substructure Search (SMARTS)

**SMARTS (SMILES ARbitrary Target Specification)** is a pattern matching language for flexibly describing substructures.

#### Major SMARTS Patterns

Pattern | Description | Example  
---|---|---  
`[#6]` | Carbon atom | Any carbon  
`[OH]` | Hydroxyl group | Alcohol, phenol  
`C(=O)O` | Carboxyl group | Carboxylic acid  
`c1ccccc1` | Benzene ring | Aromatic ring  
`[$([NX3;H2])]` | Primary amine | RNH₂  
  
#### Code Example 7: Substructure Search
    
    
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    # List of molecules to search
    molecules = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Benzene": "c1ccccc1"
    }
    
    # Search pattern (carboxylic acid)
    pattern = Chem.MolFromSmarts("C(=O)[OH]")
    
    # Pattern matching for each molecule
    for name, smiles in molecules.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol.HasSubstructMatch(pattern):
            # Get indices of matched atoms
            matches = mol.GetSubstructMatches(pattern)
            print(f"{name}: Contains carboxylic acid (matches: {matches})")
        else:
            print(f"{name}: Does not contain carboxylic acid")
    
    # Highlight matched positions
    mol_highlight = Chem.MolFromSmiles(molecules["Aspirin"])
    matches = mol_highlight.GetSubstructMatches(pattern)
    img = Draw.MolToImage(
        mol_highlight,
        highlightAtoms=[atom for match in matches for atom in match]
    )
    img.save("aspirin_highlighted.png")
    

**Output Example:**
    
    
    Aspirin: Contains carboxylic acid (matches: ((7, 8, 9),))
    Ibuprofen: Contains carboxylic acid (matches: ((12, 13, 14),))
    Paracetamol: Does not contain carboxylic acid
    Benzene: Does not contain carboxylic acid
    

### 1.3.3 Editing and Transforming Molecules

#### Code Example 8: Adding and Removing Functional Groups
    
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    # Start with benzene
    mol = Chem.MolFromSmiles("c1ccccc1")
    
    # Convert to editable molecule object
    mol_edit = Chem.RWMol(mol)
    
    # Add new atom (oxygen)
    new_atom_idx = mol_edit.AddAtom(Chem.Atom(8))  # 8 = oxygen
    
    # Bond existing carbon (index 0) with new oxygen
    mol_edit.AddBond(0, new_atom_idx, Chem.BondType.SINGLE)
    
    # Add hydrogens to complete molecule
    final_mol = mol_edit.GetMol()
    final_mol = Chem.AddHs(final_mol)
    
    # Get SMILES
    smiles_final = Chem.MolToSmiles(Chem.RemoveHs(final_mol))
    print(f"Generated molecule: {smiles_final}")
    # Output: c1ccccc1O (phenol)
    
    # Draw
    img = Draw.MolToImage(Chem.RemoveHs(final_mol))
    img.save("phenol.png")
    

* * *

## 1.4 Case Study: Pharmaceutical Database Search

Retrieve pharmaceutical information from actual databases and analyze molecular structures.

### Types of Databases

Database | Description | Data Count | Access  
---|---|---|---  
**ChEMBL** | Bioactive molecule data | 2.1M+ | [ebi.ac.uk/chembl](<https://www.ebi.ac.uk/chembl/>)  
**PubChem** | Compound database | 110M+ | [pubchem.ncbi.nlm.nih.gov](<https://pubchem.ncbi.nlm.nih.gov/>)  
**DrugBank** | Approved drug data | 14,000+ | [drugbank.com](<https://go.drugbank.com/>)  
**ZINC** | Commercially available compounds | 1B+ | [zinc15.docking.org](<https://zinc15.docking.org/>)  
  
### Code Example 9: Retrieving Data from ChEMBL
    
    
    # Install chembl_webresource_client
    # pip install chembl_webresource_client
    
    from chembl_webresource_client.new_client import new_client
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import pandas as pd
    
    # Search for targets (target proteins)
    target = new_client.target
    target_query = target.search('EGFR')  # Epidermal growth factor receptor
    targets = list(target_query)
    
    # Get first target ID
    chembl_id = targets[0]['target_chembl_id']
    print(f"Target ID: {chembl_id}")
    
    # Get assay data for the target
    activity = new_client.activity
    activities = activity.filter(
        target_chembl_id=chembl_id,
        type="IC50",
        relation="="
    ).filter(assay_type="B")
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(activities)
    
    # If SMILES column exists
    if 'canonical_smiles' in df.columns:
        # Keep only valid SMILES
        df['mol'] = df['canonical_smiles'].apply(
            Chem.MolFromSmiles
        )
        df = df[df['mol'].notna()]
    
        # Calculate molecular descriptors
        df['MW'] = df['mol'].apply(
            Descriptors.MolWt
        )
        df['LogP'] = df['mol'].apply(
            Descriptors.MolLogP
        )
    
        print(f"\nNumber of compounds retrieved: {len(df)}")
        print(df[['canonical_smiles', 'MW', 'LogP']].head())
    

### Code Example 10: SMILES Validation and Sanitization
    
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    def validate_and_sanitize_smiles(smiles_list):
        """
        Validate and sanitize SMILES list
    
        Parameters:
        -----------
        smiles_list : list
            List of SMILES
    
        Returns:
        --------
        valid_smiles : list
            List of valid SMILES
        invalid_smiles : list
            List of invalid SMILES
        """
        valid_smiles = []
        invalid_smiles = []
    
        for smi in smiles_list:
            try:
                # Create molecule from SMILES
                mol = Chem.MolFromSmiles(smi)
    
                if mol is None:
                    invalid_smiles.append(smi)
                    continue
    
                # Sanitization (chemical validity check)
                Chem.SanitizeMol(mol)
    
                # Get canonical SMILES
                canonical = Chem.MolToSmiles(mol)
                valid_smiles.append(canonical)
    
            except Exception as e:
                print(f"Error: {smi} - {e}")
                invalid_smiles.append(smi)
    
        return valid_smiles, invalid_smiles
    
    # Test data
    test_smiles = [
        "CCO",                      # Valid
        "c1ccccc1",                 # Valid
        "CC(=O)Oc1ccccc1C(=O)O",   # Valid (aspirin)
        "C1CC",                     # Invalid (open ring)
        "INVALID",                  # Invalid
        "CC(C)(C)c1ccc(O)cc1"      # Valid (BHT)
    ]
    
    valid, invalid = validate_and_sanitize_smiles(test_smiles)
    
    print(f"Valid: {len(valid)} entries")
    print(f"Invalid: {len(invalid)} entries")
    print("\nValid SMILES:")
    for smi in valid:
        mol = Chem.MolFromSmiles(smi)
        print(f"  {smi} - MW: {Descriptors.MolWt(mol):.2f}")
    

**Output Example:**
    
    
    Valid: 4 entries
    Invalid: 2 entries
    
    Valid SMILES:
      CCO - MW: 46.07
      c1ccccc1 - MW: 78.11
      CC(=O)Oc1ccccc1C(=O)O - MW: 180.16
      CC(C)(C)c1ccc(O)cc1 - MW: 150.22
    

* * *

## Exercises

### Exercise 1: Basic Molecular Manipulation

Create SMILES for the following molecules and visualize them with RDKit: 1\. Propanol (C₃H₇OH) 2\. Toluene (methylbenzene) 3\. Acetaminophen (paracetamol)

Solution
    
    
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    smiles_dict = {
        "Propanol": "CCCO",
        "Toluene": "Cc1ccccc1",
        "Acetaminophen": "CC(=O)Nc1ccc(O)cc1"
    }
    
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_dict.values()]
    legends = list(smiles_dict.keys())
    
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(300, 300),
        legends=legends
    )
    img.save("exercise1.png")
    

### Exercise 2: Substructure Search

Search for molecules containing the following functional groups from the sample dataset: \- Hydroxyl group (-OH) \- Amino group (-NH₂) \- Nitro group (-NO₂)

Solution
    
    
    from rdkit import Chem
    
    # Sample molecules
    molecules = {
        "Ethanol": "CCO",
        "Aniline": "c1ccccc1N",
        "Nitrobenzene": "c1ccc(cc1)[N+](=O)[O-]",
        "Phenol": "c1ccc(cc1)O",
        "Benzoic acid": "c1ccc(cc1)C(=O)O"
    }
    
    # Search patterns
    patterns = {
        "Hydroxyl": "[OH]",
        "Amino": "[NH2]",
        "Nitro": "[N+](=O)[O-]"
    }
    
    # Execute search
    for func_group, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        print(f"\nMolecules containing {func_group} group:")
    
        for name, smiles in molecules.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol.HasSubstructMatch(pattern):
                print(f"  - {name}")
    

**Output Example:** 
    
    
    Molecules containing Hydroxyl group:
      - Ethanol
      - Phenol
      - Benzoic acid
    
    Molecules containing Amino group:
      - Aniline
    
    Molecules containing Nitro group:
      - Nitrobenzene
    

### Exercise 3: Calculating Molecular Descriptors

Use RDKit to calculate the following molecular descriptors: \- Molecular Weight \- logP (lipophilicity) \- TPSA (Topological Polar Surface Area) \- Number of rotatable bonds

Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Use RDKit to calculate the following molecular descriptors:
    
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import pandas as pd
    
    # Sample pharmaceuticals
    drugs = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1"
    }
    
    # Calculate descriptors
    data = []
    for name, smiles in drugs.items():
        mol = Chem.MolFromSmiles(smiles)
        data.append({
            'Name': name,
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol)
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    

**Output Example:** 
    
    
            Name      MW  LogP  TPSA  RotBonds
         Aspirin  180.16  1.19 63.60         3
       Ibuprofen  206.28  3.50 37.30         4
     Paracetamol  151.16  0.46 49.33         1
    

### Exercise 4: Retrieving Data from PubChem

Use PubChem's REST API to retrieve information for a specific compound (e.g., caffeine) and visualize its structure.

Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - requests>=2.31.0
    
    """
    Example: Use PubChem's REST API to retrieve information for a specifi
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import requests
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    # PubChem REST API (search by compound name)
    compound_name = "caffeine"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/" \
          f"compound/name/{compound_name}/property/" \
          f"CanonicalSMILES,MolecularWeight,IUPACName/JSON"
    
    response = requests.get(url)
    data = response.json()
    
    # Retrieve data
    properties = data['PropertyTable']['Properties'][0]
    smiles = properties['CanonicalSMILES']
    mw = properties['MolecularWeight']
    iupac = properties.get('IUPACName', 'N/A')
    
    print(f"Compound name: {compound_name}")
    print(f"IUPAC name: {iupac}")
    print(f"SMILES: {smiles}")
    print(f"Molecular weight: {mw}")
    
    # Draw structure
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(400, 400))
    img.save(f"{compound_name}_structure.png")
    

**Output Example:** 
    
    
    Compound name: caffeine
    IUPAC name: 1,3,7-trimethylpurine-2,6-dione
    SMILES: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
    Molecular weight: 194.19
    

* * *

## Summary

In this chapter, we learned the following:

### Content Covered

  1. **Definition of Chemoinformatics** \- Fusion of chemistry and data science \- Applications in drug discovery, organic materials, catalysis, and polymers

  2. **Molecular Representation Methods** \- SMILES: Linear string representation \- InChI/InChIKey: Standardized identifiers \- Molecular graphs: Nodes and edges \- 3D structure: Importance of stereochemistry

  3. **Basic Operations with RDKit** \- Reading and visualizing molecules \- Substructure search (SMARTS) \- Editing and transforming molecules

  4. **Practice: Pharmaceutical Database Search** \- Retrieving data from ChEMBL/PubChem \- SMILES validation and sanitization

### Next Steps

In Chapter 2, we will learn about calculating molecular descriptors and QSAR/QSPR modeling.

**[Chapter 2: Introduction to QSAR/QSPR - Fundamentals of Property Prediction →](<chapter-2.html>)**

* * *

## References

  1. Weininger, D. (1988). "SMILES, a chemical language and information system." _Journal of Chemical Information and Computer Sciences_ , 28(1), 31-36. DOI: 10.1021/ci00057a005
  2. Heller, S. et al. (2015). "InChI, the IUPAC International Chemical Identifier." _Journal of Cheminformatics_ , 7, 23. DOI: 10.1186/s13321-015-0068-4
  3. Landrum, G. (2024). "RDKit: Open-source cheminformatics." [rdkit.org](<https://www.rdkit.org/>)
  4. Gaulton, A. et al. (2017). "The ChEMBL database in 2017." _Nucleic Acids Research_ , 45(D1), D945-D954. DOI: 10.1093/nar/gkw1074

* * *

**[← Back to Series Home](<./index.html>)** | **[Next Chapter →](<chapter-2.html>)**
