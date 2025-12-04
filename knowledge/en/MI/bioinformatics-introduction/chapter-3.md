---
title: "Chapter 3: Molecular Docking and Interaction Analysis"
chapter_title: "Chapter 3: Molecular Docking and Interaction Analysis"
subtitle: Predicting Protein-Ligand Binding
reading_time: 25-30 min
difficulty: Intermediate
code_examples: 9
exercises: 3
version: 1.0
created_at: 2025-10-17
---

# Chapter 3: Molecular Docking and Interaction Analysis

This chapter covers Molecular Docking and Interaction Analysis. You will learn principles of molecular docking, Execute molecular docking with AutoDock Vina, and Evaluate binding affinity.

**Predicting Protein-Ligand Binding**

## Learning Objectives

  * ✅ Understand the principles of molecular docking and scoring functions
  * ✅ Execute molecular docking with AutoDock Vina
  * ✅ Evaluate binding affinity and compare with experimental data
  * ✅ Visualize binding sites and interactions with PyMOL
  * ✅ Build binding prediction models using machine learning

**Reading Time** : 25-30 min | **Code Examples** : 9 | **Exercises** : 3

* * *

## 3.1 Fundamentals of Molecular Docking

### What is Molecular Docking?

**Molecular docking** is a computational method to predict the binding mode of a ligand (small molecule) to a protein (receptor).
    
    
    ```mermaid
    flowchart LR
        A[Protein Structure] --> C[Docking\nSimulation]
        B[Ligand Structure] --> C
        C --> D[Binding Pose]
        C --> E[Binding Affinity]
        D --> F[Drug Design]
        E --> F
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#4CAF50,color:#fff
        style D fill:#f3e5f5
        style E fill:#ffebee
        style F fill:#fff9c4
    ```

**Applications** : \- Drug discovery (hit compound identification) \- Biosensor design (ligand selection) \- Drug delivery systems design (binding to target proteins)

* * *

### Two Steps of Docking

**1\. Sampling** \- Explore possible binding poses of the ligand \- Methods: Genetic algorithms, Monte Carlo, systematic search

**2\. Scoring** \- Evaluate the binding affinity of each pose \- Scoring functions: Force field-based, empirical, knowledge-based

* * *

### Overview of AutoDock Vina

**AutoDock Vina** is the most widely used molecular docking software.

**Features** : \- Fast (100x faster than classic AutoDock) \- High accuracy (RMSD < 2 Å vs experimental values) \- Open source

**Installation** :
    
    
    # Install with Conda
    conda install -c conda-forge vina
    
    # Or download from official website
    # http://vina.scripps.edu/
    

* * *

### Example 1: Preparing Protein and Ligand
    
    
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from Bio.PDB import PDBParser
    import os
    
    # Protein preparation (PDB file)
    def prepare_protein(pdb_file, output_pdbqt):
        """
        Prepare protein for docking
    
        Parameters:
        -----------
        pdb_file : str
            Input PDB file
        output_pdbqt : str
            Output PDBQT file (AutoDock format)
        """
        # Use Open Babel to convert PDB → PDBQT
        # PDBQT: Extended PDB format with charges and atom types
        cmd = f"obabel {pdb_file} -O {output_pdbqt} -xr"
        os.system(cmd)
    
        print(f"Protein preparation complete: {output_pdbqt}")
    
    # Ligand preparation (SMILES)
    def prepare_ligand_from_smiles(smiles, output_pdbqt):
        """
        Prepare ligand from SMILES
    
        Parameters:
        -----------
        smiles : str
            SMILES notation of ligand
        output_pdbqt : str
            Output PDBQT file
        """
        # Generate molecule object
        mol = Chem.MolFromSmiles(smiles)
    
        # Add hydrogens
        mol = Chem.AddHs(mol)
    
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    
        # Save in PDB format
        temp_pdb = "ligand_temp.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
    
        # Convert PDB → PDBQT
        cmd = f"obabel {temp_pdb} -O {output_pdbqt} -xh"
        os.system(cmd)
    
        # Remove temporary file
        os.remove(temp_pdb)
    
        print(f"Ligand preparation complete: {output_pdbqt}")
    
        # Display molecule information
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        print(f"  Molecular weight: {mw:.1f}")
        print(f"  LogP: {logp:.2f}")
    
    # Usage example
    # Aspirin (analgesic)
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    prepare_ligand_from_smiles(aspirin_smiles, "aspirin.pdbqt")
    

* * *

### Example 2: Running AutoDock Vina
    
    
    import subprocess
    import os
    
    def run_autodock_vina(
        receptor_pdbqt,
        ligand_pdbqt,
        center_x, center_y, center_z,
        size_x=20, size_y=20, size_z=20,
        output_pdbqt="output.pdbqt",
        exhaustiveness=8
    ):
        """
        Run AutoDock Vina
    
        Parameters:
        -----------
        receptor_pdbqt : str
            Protein PDBQT file
        ligand_pdbqt : str
            Ligand PDBQT file
        center_x, center_y, center_z : float
            Center coordinates of search box (Å)
        size_x, size_y, size_z : float
            Size of search box (Å)
        output_pdbqt : str
            Output file
        exhaustiveness : int
            Exhaustiveness of search (default 8, use 16-32 for higher accuracy)
        """
    
        # Vina command
        cmd = [
            "vina",
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--center_x", str(center_x),
            "--center_y", str(center_y),
            "--center_z", str(center_z),
            "--size_x", str(size_x),
            "--size_y", str(size_y),
            "--size_z", str(size_z),
            "--out", output_pdbqt,
            "--exhaustiveness", str(exhaustiveness)
        ]
    
        print("=== Running AutoDock Vina ===")
        print(" ".join(cmd))
    
        # Execute
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
    
        # Display results
        print(result.stdout)
    
        # Extract binding affinities
        affinities = []
        for line in result.stdout.split('\n'):
            if line.strip().startswith('1') or \
               line.strip().startswith('2') or \
               line.strip().startswith('3'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        affinity = float(parts[1])
                        affinities.append(affinity)
                    except ValueError:
                        pass
    
        if affinities:
            print(f"\n=== Binding Affinity ===")
            print(f"Best score: {affinities[0]:.1f} kcal/mol")
            print(f"Top 3: {affinities[:3]}")
    
        return affinities
    
    # Usage example (fictional coordinates)
    # In actual use, specify coordinates of binding site
    """
    affinities = run_autodock_vina(
        receptor_pdbqt="protein.pdbqt",
        ligand_pdbqt="ligand.pdbqt",
        center_x=10.5,
        center_y=20.3,
        center_z=15.8,
        size_x=20,
        size_y=20,
        size_z=20,
        output_pdbqt="docking_result.pdbqt"
    )
    """
    

**Output example** :
    
    
    === Running AutoDock Vina ===
    vina --receptor protein.pdbqt --ligand ligand.pdbqt ...
    
    Performing docking...
    mode |   affinity | dist from best mode
         | (kcal/mol) | rmsd l.b.| rmsd u.b.
    -----+------------+----------+----------
       1      -8.5       0.000       0.000
       2      -8.2       1.823       3.145
       3      -7.9       2.456       4.021
    
    === Binding Affinity ===
    Best score: -8.5 kcal/mol
    Top 3: [-8.5, -8.2, -7.9]
    

* * *

## 3.2 Visualization and Analysis of Interactions

### Visualization with PyMOL

**Example 3: Generating PyMOL Script**
    
    
    def generate_pymol_script(
        protein_pdb,
        ligand_pdbqt,
        output_script="visualize.pml"
    ):
        """
        Generate PyMOL visualization script
    
        Parameters:
        -----------
        protein_pdb : str
            Protein PDB file
        ligand_pdbqt : str
            Docking result (PDBQT)
        output_script : str
            Output PyMOL script
        """
        script = f"""
    # PyMOL visualization script
    
    # Load files
    load {protein_pdb}, protein
    load {ligand_pdbqt}, ligand
    
    # Protein display settings
    hide everything, protein
    show cartoon, protein
    color cyan, protein
    
    # Display binding site (within 5Å of ligand)
    select binding_site, protein within 5 of ligand
    show sticks, binding_site
    color green, binding_site
    
    # Ligand display
    hide everything, ligand
    show sticks, ligand
    color yellow, ligand
    util.cbay ligand
    
    # Display hydrogen bonds
    distance hbonds, ligand, binding_site, mode=2
    color red, hbonds
    
    # View settings
    zoom ligand, 5
    set stick_radius, 0.3
    set sphere_scale, 0.25
    
    # White background
    bg_color white
    
    # Rendering settings
    set ray_shadows, 0
    set antialias, 2
    
    # Save image
    ray 1200, 1200
    png binding_site.png, dpi=300
    
    print("Visualization complete: binding_site.png")
    """
    
        with open(output_script, 'w') as f:
            f.write(script)
    
        print(f"PyMOL script generated: {output_script}")
        print("Run with: pymol visualize.pml")
    
    # Usage example
    generate_pymol_script(
        "protein.pdb",
        "docking_result.pdbqt",
        "visualize.pml"
    )
    

* * *

### Quantitative Analysis of Interactions

**Example 4: Detecting Hydrogen Bonds**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    from Bio.PDB import PDBParser, NeighborSearch
    import numpy as np
    
    def detect_hydrogen_bonds(
        protein_structure,
        ligand_structure,
        distance_cutoff=3.5,  # Å
        angle_cutoff=120  # degrees
    ):
        """
        Detect hydrogen bonds
    
        Parameters:
        -----------
        protein_structure : Bio.PDB.Structure
            Protein structure
        ligand_structure : Bio.PDB.Structure
            Ligand structure
        distance_cutoff : float
            Distance threshold (Å)
        angle_cutoff : float
            Angle threshold (degrees)
    
        Returns:
        --------
        list: List of hydrogen bonds
        """
        # All protein atoms
        protein_atoms = [
            atom for atom in protein_structure.get_atoms()
        ]
    
        # All ligand atoms
        ligand_atoms = [
            atom for atom in ligand_structure.get_atoms()
        ]
    
        # Neighbor search
        ns = NeighborSearch(protein_atoms)
    
        hbonds = []
    
        # For each ligand atom
        for ligand_atom in ligand_atoms:
            # Donor/acceptor determination (simplified)
            if ligand_atom.element not in ['N', 'O']:
                continue
    
            # Nearby protein atoms
            nearby_atoms = ns.search(
                ligand_atom.get_coord(),
                distance_cutoff
            )
    
            for protein_atom in nearby_atoms:
                if protein_atom.element not in ['N', 'O']:
                    continue
    
                # Calculate distance
                distance = ligand_atom - protein_atom
    
                if distance <= distance_cutoff:
                    hbonds.append({
                        'ligand_atom': ligand_atom.get_full_id(),
                        'protein_atom': protein_atom.get_full_id(),
                        'distance': distance
                    })
    
        return hbonds
    
    # Usage example (fictional structure)
    """
    parser = PDBParser(QUIET=True)
    protein = parser.get_structure('protein', 'protein.pdb')
    ligand = parser.get_structure('ligand', 'ligand.pdb')
    
    hbonds = detect_hydrogen_bonds(protein, ligand)
    
    print(f"=== Hydrogen Bonds ===")
    print(f"Number detected: {len(hbonds)}")
    
    for i, hb in enumerate(hbonds[:5], 1):
        print(f"\nHydrogen bond {i}:")
        print(f"  Distance: {hb['distance']:.2f} Å")
        print(f"  Ligand atom: {hb['ligand_atom']}")
        print(f"  Protein atom: {hb['protein_atom']}")
    """
    

* * *

### Analysis of Hydrophobic Interactions

**Example 5: Detecting Hydrophobic Contacts**
    
    
    def detect_hydrophobic_contacts(
        protein_structure,
        ligand_structure,
        distance_cutoff=4.5  # Å
    ):
        """
        Detect hydrophobic interactions
    
        Parameters:
        -----------
        protein_structure : Bio.PDB.Structure
        ligand_structure : Bio.PDB.Structure
        distance_cutoff : float
            Threshold for hydrophobic interaction (Å)
    
        Returns:
        --------
        list: List of hydrophobic contacts
        """
        # Hydrophobic amino acids
        hydrophobic_residues = [
            'ALA', 'VAL', 'ILE', 'LEU', 'MET',
            'PHE', 'TRP', 'PRO'
        ]
    
        # Carbon atoms in hydrophobic residues
        protein_hydrophobic_atoms = []
        for residue in protein_structure.get_residues():
            if residue.get_resname() in hydrophobic_residues:
                for atom in residue:
                    if atom.element == 'C':
                        protein_hydrophobic_atoms.append(atom)
    
        # Carbon atoms in ligand
        ligand_carbon_atoms = [
            atom for atom in ligand_structure.get_atoms()
            if atom.element == 'C'
        ]
    
        # Neighbor search
        ns = NeighborSearch(protein_hydrophobic_atoms)
    
        contacts = []
    
        for ligand_atom in ligand_carbon_atoms:
            nearby_atoms = ns.search(
                ligand_atom.get_coord(),
                distance_cutoff
            )
    
            for protein_atom in nearby_atoms:
                distance = ligand_atom - protein_atom
    
                if distance <= distance_cutoff:
                    contacts.append({
                        'ligand_atom': ligand_atom.get_full_id(),
                        'protein_atom': protein_atom.get_full_id(),
                        'residue': protein_atom.get_parent().get_resname(),
                        'distance': distance
                    })
    
        return contacts
    
    # Usage example
    """
    contacts = detect_hydrophobic_contacts(protein, ligand)
    
    print(f"\n=== Hydrophobic Contacts ===")
    print(f"Number detected: {len(contacts)}")
    
    # Count by residue
    from collections import Counter
    residue_counts = Counter(
        [c['residue'] for c in contacts]
    )
    
    print("\nContacts by residue:")
    for residue, count in residue_counts.most_common(5):
        print(f"  {residue}: {count}")
    """
    

* * *

## 3.3 Machine Learning for Binding Prediction

### Graph Neural Networks (GNN)

**Example 6: Graph Representation of Protein-Ligand Complex**
    
    
    # Requirements:
    # - Python 3.9+
    # - networkx>=3.1.0
    # - numpy>=1.24.0, <2.0.0
    
    import networkx as nx
    import numpy as np
    from rdkit import Chem
    
    def protein_ligand_to_graph(protein_atoms, ligand_mol):
        """
        Convert protein-ligand complex to graph
    
        Parameters:
        -----------
        protein_atoms : list
            List of protein atoms
        ligand_mol : RDKit Mol
            Ligand molecule
    
        Returns:
        --------
        networkx.Graph: Complex graph
        """
        G = nx.Graph()
    
        # Ligand graph (molecular graph)
        for atom in ligand_mol.GetAtoms():
            G.add_node(
                f"L{atom.GetIdx()}",
                atom_type=atom.GetSymbol(),
                atomic_num=atom.GetAtomicNum(),
                hybridization=str(atom.GetHybridization()),
                node_type='ligand'
            )
    
        # Ligand bonds
        for bond in ligand_mol.GetBonds():
            G.add_edge(
                f"L{bond.GetBeginAtomIdx()}",
                f"L{bond.GetEndAtomIdx()}",
                bond_type=str(bond.GetBondType())
            )
    
        # Protein atoms (simplified: representative atoms only)
        for i, atom in enumerate(protein_atoms[:50]):  # First 50 atoms
            G.add_node(
                f"P{i}",
                atom_type=atom.element,
                node_type='protein'
            )
    
        # Protein-ligand interaction edges
        # (Distance-based, 4Å threshold)
        ligand_coords = ligand_mol.GetConformer().GetPositions()
    
        for i, protein_atom in enumerate(protein_atoms[:50]):
            protein_coord = protein_atom.get_coord()
    
            for j, ligand_coord in enumerate(ligand_coords):
                distance = np.linalg.norm(
                    protein_coord - ligand_coord
                )
    
                if distance < 4.0:  # Within 4Å
                    G.add_edge(
                        f"P{i}",
                        f"L{j}",
                        interaction='contact',
                        distance=distance
                    )
    
        return G
    
    # Usage example
    """
    ligand_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    ligand_mol = Chem.MolFromSmiles(ligand_smiles)
    ligand_mol = Chem.AddHs(ligand_mol)
    AllChem.EmbedMolecule(ligand_mol)
    
    # protein_atoms obtained from Bio.PDB.Structure
    # complex_graph = protein_ligand_to_graph(
    #     protein_atoms, ligand_mol
    # )
    
    # Graph statistics
    # print(f"Number of nodes: {complex_graph.number_of_nodes()}")
    # print(f"Number of edges: {complex_graph.number_of_edges()}")
    """
    

* * *

### DeepDocking-Style Prediction Model

**Example 7: Binding Affinity Prediction Model**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    
    def extract_complex_features(
        protein_structure,
        ligand_mol
    ):
        """
        Extract features from protein-ligand complex
    
        Returns:
        --------
        dict: Feature dictionary
        """
        from rdkit.Chem import Descriptors
    
        # Ligand features
        ligand_features = {
            'ligand_mw': Descriptors.MolWt(ligand_mol),
            'ligand_logp': Descriptors.MolLogP(ligand_mol),
            'ligand_hbd': Descriptors.NumHDonors(ligand_mol),
            'ligand_hba': Descriptors.NumHAcceptors(ligand_mol),
            'ligand_rotatable_bonds': Descriptors.NumRotatableBonds(
                ligand_mol
            ),
            'ligand_aromatic_rings': Descriptors.NumAromaticRings(
                ligand_mol
            )
        }
    
        # Protein features (simplified)
        # In reality, use binding site residue composition, etc.
        protein_features = {
            'binding_site_residues': 10,  # Placeholder value
            'binding_site_hydrophobic': 0.4,
            'binding_site_charged': 0.3
        }
    
        return {**ligand_features, **protein_features}
    
    # Generate sample data (in practice, use database)
    np.random.seed(42)
    
    n_samples = 200
    X_features = []
    y_affinities = []
    
    for _ in range(n_samples):
        # Random features (for demo)
        features = {
            'ligand_mw': np.random.uniform(150, 500),
            'ligand_logp': np.random.uniform(-2, 5),
            'ligand_hbd': np.random.randint(0, 6),
            'ligand_hba': np.random.randint(0, 10),
            'ligand_rotatable_bonds': np.random.randint(0, 10),
            'ligand_aromatic_rings': np.random.randint(0, 4),
            'binding_site_residues': np.random.randint(5, 20),
            'binding_site_hydrophobic': np.random.uniform(0.2, 0.6),
            'binding_site_charged': np.random.uniform(0.1, 0.5)
        }
    
        # Binding affinity (kcal/mol)
        # Generated based on realistic relationships
        affinity = (
            -0.1 * features['ligand_mw'] / 100
            - 1.5 * features['ligand_hbd']
            - 1.0 * features['ligand_hba']
            + 2.0 * features['ligand_logp']
            + np.random.randn() * 1.0
        )
    
        X_features.append(list(features.values()))
        y_affinities.append(affinity)
    
    X = np.array(X_features)
    y = np.array(y_affinities)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model training
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"=== Binding Affinity Prediction Model ===")
    print(f"MAE: {mae:.2f} kcal/mol")
    print(f"R²: {r2:.3f}")
    
    # Feature importance
    feature_names = [
        'MW', 'LogP', 'HBD', 'HBA', 'RotBonds',
        'AromRings', 'BSResidues', 'BSHydrophobic',
        'BSCharged'
    ]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature importance:")
    print(importance_df)
    

* * *

## 3.4 Case Study: Antibody-Antigen Interactions

### Antibody Structure Modeling

**Example 8: Modeling Antibody Variable Regions**
    
    
    def model_antibody_structure(
        heavy_chain_seq,
        light_chain_seq,
        output_pdb="antibody_model.pdb"
    ):
        """
        Model antibody structure
    
        Parameters:
        -----------
        heavy_chain_seq : str
            Heavy chain sequence
        light_chain_seq : str
            Light chain sequence
        output_pdb : str
            Output PDB file
    
        Note:
        -----
        In practice, use specialized tools (Modeller, AlphaFold2)
        """
        print("=== Antibody Structure Modeling ===")
        print(f"Heavy chain length: {len(heavy_chain_seq)} aa")
        print(f"Light chain length: {len(light_chain_seq)} aa")
    
        # CDR (Complementarity-Determining Region) estimation
        # Simplified: Assume heavy chain CDR3 is positions 95-102
        cdr_h3_start = 95
        cdr_h3_end = 102
    
        if len(heavy_chain_seq) >= cdr_h3_end:
            cdr_h3_seq = heavy_chain_seq[cdr_h3_start:cdr_h3_end]
            print(f"\nCDR-H3 sequence: {cdr_h3_seq}")
            print(f"Length: {len(cdr_h3_seq)} aa")
    
        # Actual modeling should be performed with AlphaFold2 or Modeller
        # Here, only display information for demo
    
        print(f"\nModel will be saved to: {output_pdb}")
        print("Use AlphaFold2 for actual modeling")
    
    # Usage example
    heavy_chain = "QVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWN" * 3
    light_chain = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWY" * 3
    
    model_antibody_structure(heavy_chain, light_chain)
    

* * *

### Epitope Prediction

**Example 9: Linear Epitope Prediction**
    
    
    def predict_linear_epitopes(
        antigen_sequence,
        window_size=9
    ):
        """
        Predict linear epitopes (simplified version)
    
        Parameters:
        -----------
        antigen_sequence : str
            Amino acid sequence of antigen
        window_size : int
            Window size for epitopes
    
        Returns:
        --------
        list: List of epitope candidates
        """
        # Simplified prediction (in practice, use Bepipred, IEDB tools, etc.)
    
        # Scoring considering hydrophobicity, charge, surface exposure
        hydrophobic_aa = "AVILMFWP"
        charged_aa = "KRHDE"
        polar_aa = "STNQYC"
    
        epitope_candidates = []
    
        for i in range(len(antigen_sequence) - window_size + 1):
            peptide = antigen_sequence[i:i+window_size]
    
            # Calculate score
            hydrophobic_count = sum(
                1 for aa in peptide if aa in hydrophobic_aa
            )
            charged_count = sum(
                1 for aa in peptide if aa in charged_aa
            )
            polar_count = sum(
                1 for aa in peptide if aa in polar_aa
            )
    
            # Simple score: prefer balanced peptides
            score = (
                0.3 * charged_count +
                0.5 * polar_count +
                0.2 * hydrophobic_count
            )
    
            epitope_candidates.append({
                'position': i,
                'peptide': peptide,
                'score': score
            })
    
        # Sort by score
        epitope_candidates.sort(
            key=lambda x: x['score'],
            reverse=True
        )
    
        return epitope_candidates
    
    # Usage example
    antigen_seq = """
    MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHS
    TQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNI
    IRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNK
    SWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGY
    FKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLT
    PGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETK
    CTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASV
    YAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSF
    VIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYN
    YLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPT
    NGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF
    """
    antigen_seq = antigen_seq.replace("\n", "").replace(" ", "")
    
    epitopes = predict_linear_epitopes(antigen_seq, window_size=9)
    
    print(f"=== Epitope Prediction ===")
    print(f"Sequence length: {len(antigen_seq)} aa")
    print(f"\nTop 5 candidates:")
    
    for i, ep in enumerate(epitopes[:5], 1):
        print(f"\n{i}. Position {ep['position']}")
        print(f"   Sequence: {ep['peptide']}")
        print(f"   Score: {ep['score']:.2f}")
    

* * *

## 3.5 Chapter Summary

### What We Learned

  1. **Molecular Docking** \- Docking with AutoDock Vina \- Binding affinity scores

  2. **Interaction Analysis** \- Hydrogen bond detection \- Hydrophobic contacts \- PyMOL visualization

  3. **Machine Learning** \- Graph representation \- Binding affinity prediction

  4. **Antibody-Antigen** \- Antibody structure modeling \- Epitope prediction

### Next Chapter

In Chapter 4, we will learn about **Biosensor and Drug Delivery System (DDS) Materials Design**.

**[Chapter 4: Biosensor and DDS Materials Design →](<chapter-4.html>)**

* * *

## References

  1. Trott, O. & Olson, A. J. (2010). "AutoDock Vina: improving the speed and accuracy of docking with a new scoring function." _Journal of Computational Chemistry_ , 31(2), 455-461.

  2. Burley, S. K. et al. (2021). "RCSB Protein Data Bank." _Nucleic Acids Research_ , 49(D1), D437-D451.

* * *

## Navigation

**[← Chapter 2](<chapter-2.html>)** | **[Chapter 4 →](<chapter-4.html>)** | **[Table of Contents](<./index.html>)**
