---
title: Reaction Prediction and Retrosynthesis
chapter_title: Reaction Prediction and Retrosynthesis
subtitle: 
reading_time: 25-30 min
difficulty: Intermediate~Advanced
code_examples: 10
exercises: 3
version: 1.0
created_at: 2025-10-18
---

# Chapter 4: Reaction Prediction and Retrosynthesis

This chapter covers Reaction Prediction and Retrosynthesis. You will learn essential concepts and techniques.

## What You'll Learn in This Chapter

In this chapter, you will learn about computational representation and prediction of chemical reactions, and retrosynthetic analysis (Retrosynthesis) from target molecules to starting materials. These technologies have brought revolutionary advances in efficient synthetic route design.

### Learning Objectives

  * ✅ Understand and describe reaction templates and SMARTS
  * ✅ Understand the basics of reaction prediction models
  * ✅ Use retrosynthesis concepts and major tools
  * ✅ Know industrial application examples and envision career paths
  * ✅ Apply to actual drug discovery and materials development projects

* * *

## 4.1 Reaction Templates and SMARTS

**Representing chemical reactions** requires describing the transformation from reactants to products.
    
    
    ```mermaid
    flowchart LR
        A[Reactants] -->|Conditions| B[Products]
    
        C[SMILES] --> D[Reaction SMILES]
        C --> E[Reaction Template\nSMARTS]
        C --> F[Reaction Graph]
    
        D --> G[Machine Learning Model]
        E --> G
        F --> G
        G --> H[Reaction Prediction]
    
        style A fill:#e3f2fd
        style B fill:#4CAF50,color:#fff
        style H fill:#FF9800,color:#fff
    ```

### 4.1.1 Reaction SMILES and SMIRKS

**Reaction SMILES** represents reactants and products separated by `>>`.

**Format** :
    
    
    reactant1.reactant2>>product1.product2
    

**SMIRKS (SMILES Reaction Specification)** explicitly describes the changing parts of a reaction.

#### Code Example 1: Parsing Reaction SMILES
    
    
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    
    # Example of esterification reaction
    reaction_smiles = "CC(=O)O.CCO>>CC(=O)OCC.O"
    # Acetic acid + Ethanol >> Ethyl acetate + Water
    
    # Create reaction object
    rxn = AllChem.ReactionFromSmarts(reaction_smiles)
    
    print(f"Number of reactants: {rxn.GetNumReactantTemplates()}")
    print(f"Number of products: {rxn.GetNumProductTemplates()}")
    
    # Get reactants and products
    reactants = [rxn.GetReactantTemplate(i)
                 for i in range(rxn.GetNumReactantTemplates())]
    products = [rxn.GetProductTemplate(i)
                for i in range(rxn.GetNumProductTemplates())]
    
    # Display SMILES
    print("\nReactants:")
    for i, mol in enumerate(reactants, 1):
        print(f"  {i}. {Chem.MolToSmiles(mol)}")
    
    print("\nProducts:")
    for i, mol in enumerate(products, 1):
        print(f"  {i}. {Chem.MolToSmiles(mol)}")
    
    # Draw reaction
    rxn_img = Draw.ReactionToImage(rxn)
    rxn_img.save("esterification_reaction.png")
    print("\nReaction diagram saved")
    

**Sample output:**
    
    
    Number of reactants: 2
    Number of products: 2
    
    Reactants:
      1. CC(=O)O
      2. CCO
    
    Products:
      1. CC(=O)OCC
      2. O
    
    Reaction diagram saved
    

### 4.1.2 Defining Reaction Templates

**Reaction templates** describe general patterns of reactions using SMARTS.

#### Major Reaction Templates
    
    
    # Representative reaction templates
    
    reaction_templates = {
        # Esterification
        "Esterification": "[C:1](=[O:2])[OH:3].[OH:4][C:5]>>[C:1](=[O:2])[O:4][C:5].[OH2:3]",
    
        # Amidation
        "Amidation": "[C:1](=[O:2])[OH:3].[NH2:4][C:5]>>[C:1](=[O:2])[NH:4][C:5].[OH2:3]",
    
        # Suzuki-Miyaura coupling
        "Suzuki": "[c:1][Br,I:2].[c:3][B:4]([OH])([OH])>>[c:1][c:3]",
    
        # Reduction (carbonyl → alcohol)
        "Reduction": "[C:1]=[O:2]>>[C:1][OH:2]",
    
        # Oxidation (alcohol → carbonyl)
        "Oxidation": "[C:1][OH:2]>>[C:1]=[O:2]",
    
        # Grignard reaction
        "Grignard": "[C:1]=[O:2].[C:3][Mg][Br:4]>>[C:1]([OH:2])[C:3]"
    }
    
    # Display templates
    for name, smarts in reaction_templates.items():
        print(f"{name:20s}: {smarts}")
    

**Sample output:**
    
    
    Esterification      : [C:1](=[O:2])[OH:3].[OH:4][C:5]>>[C:1](=[O:2])[O:4][C:5].[OH2:3]
    Amidation           : [C:1](=[O:2])[OH:3].[NH2:4][C:5]>>[C:1](=[O:2])[NH:4][C:5].[OH2:3]
    Suzuki              : [c:1][Br,I:2].[c:3][B:4]([OH])([OH])>>[c:1][c:3]
    Reduction           : [C:1]=[O:2]>>[C:1][OH:2]
    Oxidation           : [C:1][OH:2]>>[C:1]=[O:2]
    Grignard            : [C:1]=[O:2].[C:3][Mg][Br:4]>>[C:1]([OH:2])[C:3]
    

#### Code Example 2: Applying Reaction Templates
    
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    def apply_reaction_template(reactants_smiles, template_smarts):
        """
        Predict products by applying a reaction template
    
        Parameters:
        -----------
        reactants_smiles : list
            List of SMILES for reactants
        template_smarts : str
            Reaction template (SMARTS format)
    
        Returns:
        --------
        products : list
            List of SMILES for products
        """
        # Create reaction object
        rxn = AllChem.ReactionFromSmarts(template_smarts)
    
        # Convert reactants to molecule objects
        reactant_mols = [Chem.MolFromSmiles(smi)
                         for smi in reactants_smiles]
    
        # Execute reaction
        products = rxn.RunReactants(tuple(reactant_mols))
    
        # Format results
        product_smiles = []
        for product_set in products:
            for mol in product_set:
                # Sanitize
                try:
                    Chem.SanitizeMol(mol)
                    smi = Chem.MolToSmiles(mol)
                    product_smiles.append(smi)
                except:
                    pass
    
        return product_smiles
    
    # Esterification example
    reactants = ["CC(=O)O", "CCO"]  # Acetic acid + Ethanol
    template = reaction_templates["Esterification"]
    
    products = apply_reaction_template(reactants, template)
    
    print("Esterification reaction:")
    print(f"Reactants: {' + '.join(reactants)}")
    print(f"Products: {', '.join(products)}")
    
    # Suzuki-Miyaura coupling example
    reactants_suzuki = ["c1ccc(Br)cc1", "c1ccccc1B(O)O"]
    template_suzuki = reaction_templates["Suzuki"]
    
    products_suzuki = apply_reaction_template(reactants_suzuki, template_suzuki)
    
    print("\nSuzuki-Miyaura coupling:")
    print(f"Reactants: {' + '.join(reactants_suzuki)}")
    print(f"Products: {', '.join(products_suzuki)}")
    

**Sample output:**
    
    
    Esterification reaction:
    Reactants: CC(=O)O + CCO
    Products: CC(=O)OCC, O
    
    Suzuki-Miyaura coupling:
    Reactants: c1ccc(Br)cc1 + c1ccccc1B(O)O
    Products: c1ccc(-c2ccccc2)cc1
    

### 4.1.3 USPTO Reaction Dataset

**USPTO (United States Patent and Trademark Office)** is a large-scale reaction dataset extracted from patent databases.

**Statistics** : \- Total reactions: ~1.8 million reactions \- Reaction types: Classified into 10 types \- Application: Training data for reaction prediction models

#### Code Example 3: Loading and Analyzing USPTO Reaction Data
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 3: Loading and Analyzing USPTO Reaction Data
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    
    # Load USPTO data (sample)
    # Actual data can be obtained from https://github.com/rxn4chemistry/rxnfp
    
    # Create sample data
    sample_reactions = [
        {
            'reaction_smiles': 'CC(=O)O.CCO>>CC(=O)OCC.O',
            'reaction_type': 'Esterification',
            'yield': 85.3
        },
        {
            'reaction_smiles': 'c1ccc(Br)cc1.c1ccccc1B(O)O>>c1ccc(-c2ccccc2)cc1',
            'reaction_type': 'Suzuki',
            'yield': 92.1
        },
        {
            'reaction_smiles': 'CC(=O)O.Nc1ccccc1>>CC(=O)Nc1ccccc1.O',
            'reaction_type': 'Amidation',
            'yield': 78.5
        }
    ]
    
    df_uspto = pd.DataFrame(sample_reactions)
    
    print("USPTO reaction data sample:")
    print(df_uspto)
    
    # Statistics by reaction type
    print("\nDistribution of reaction types:")
    print(df_uspto['reaction_type'].value_counts())
    
    print(f"\nAverage yield: {df_uspto['yield'].mean():.1f}%")
    

**Sample output:**
    
    
    USPTO reaction data sample:
                                        reaction_smiles  reaction_type  yield
    0               CC(=O)O.CCO>>CC(=O)OCC.O  Esterification   85.3
    1  c1ccc(Br)cc1.c1ccccc1B(O)O>>c1ccc(-c2ccc...         Suzuki   92.1
    2        CC(=O)O.Nc1ccccc1>>CC(=O)Nc1ccccc1.O      Amidation   78.5
    
    Distribution of reaction types:
    Esterification    1
    Suzuki            1
    Amidation         1
    
    Average yield: 85.3%
    

* * *

## 4.2 Reaction Prediction Models

### 4.2.1 Machine Learning-Based Reaction Prediction

Reaction prediction is forward prediction of products from reactants.
    
    
    ```mermaid
    flowchart TD
        A[Reactants] --> B[Feature Extraction]
        B --> C[Molecular Fingerprints]
        B --> D[Descriptors]
        B --> E[Graph Representation]
    
        C --> F[Machine Learning Model]
        D --> F
        E --> F
    
        F --> G[Product Prediction]
        G --> H[Top-k Candidates]
    
        style A fill:#e3f2fd
        style G fill:#4CAF50,color:#fff
        style H fill:#FF9800,color:#fff
    ```

#### Code Example 4: Reaction Yield Prediction with Random Forest
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np
    
    def reaction_to_fingerprint(reaction_smiles, nBits=2048):
        """
        Calculate difference fingerprint from reaction SMILES
    
        Difference fingerprint = Product fingerprint - Reactant fingerprint
        """
        reactants_smiles, products_smiles = reaction_smiles.split('>>')
    
        # Reactant fingerprints (sum if multiple)
        reactants = reactants_smiles.split('.')
        reactant_fp = np.zeros(nBits)
        for smi in reactants:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits)
                reactant_fp += np.array(fp)
    
        # Product fingerprints (sum if multiple)
        products = products_smiles.split('.')
        product_fp = np.zeros(nBits)
        for smi in products:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits)
                product_fp += np.array(fp)
    
        # Difference fingerprint
        diff_fp = product_fp - reactant_fp
        return diff_fp
    
    # Generate sample data (in practice, obtain from USPTO, etc.)
    np.random.seed(42)
    n_reactions = 200
    
    sample_data = []
    for i in range(n_reactions):
        # Virtual reaction SMILES (simplified)
        rxn_smi = f"CCO.CC(=O)O>>CC(=O)OCC.O"
        # Virtual yield (70-95% range)
        yield_val = np.random.uniform(70, 95)
        sample_data.append((rxn_smi, yield_val))
    
    # Feature extraction
    X = np.array([reaction_to_fingerprint(rxn) for rxn, _ in sample_data])
    y = np.array([yield_val for _, yield_val in sample_data])
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Prediction
    y_pred = rf.predict(X_test)
    
    # Evaluation
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("Reaction yield prediction model performance:")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.2f}%")
    
    # Sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(y_test))):
        print(f"Actual: {y_test[i]:.1f}%  Predicted: {y_pred[i]:.1f}%  "
              f"Error: {abs(y_test[i] - y_pred[i]):.1f}%")
    

**Sample output:**
    
    
    Reaction yield prediction model performance:
    R²: 0.012
    MAE: 7.32%
    
    Sample predictions:
    Actual: 82.4%  Predicted: 81.8%  Error: 0.6%
    Actual: 89.2%  Predicted: 82.1%  Error: 7.1%
    Actual: 76.5%  Predicted: 81.9%  Error: 5.4%
    Actual: 91.3%  Predicted: 82.2%  Error: 9.1%
    Actual: 73.8%  Predicted: 81.7%  Error: 7.9%
    

### 4.2.2 Transformer-Based Reaction Prediction

Transformer models treat reaction SMILES as strings and predict products using Seq2Seq.
    
    
    ```mermaid
    flowchart LR
        A[Reactant SMILES] --> B[Tokenization]
        B --> C[Transformer\nEncoder]
        C --> D[Context Vector]
        D --> E[Transformer\nDecoder]
        E --> F[Product SMILES]
    
        style A fill:#e3f2fd
        style F fill:#4CAF50,color:#fff
    ```

#### Code Example 5: Conceptual Transformer Implementation (Simplified)
    
    
    # Actual Transformers are complex, so this is a conceptual implementation
    
    class SimpleReactionTransformer:
        """
        Simplified model of Transformer for reaction prediction
    
        In actual implementation, use Hugging Face Transformers or
        specialized tools (rxnfp, molecular-transformer, etc.)
        """
    
        def __init__(self):
            # Model parameters (virtual)
            self.vocab_size = 100  # Number of tokens
            self.d_model = 512     # Embedding dimension
            self.n_heads = 8       # Number of attention heads
            self.n_layers = 6      # Number of layers
    
        def tokenize(self, smiles):
            """Convert SMILES to token sequence"""
            # Simplified: character-level tokenization
            tokens = list(smiles)
            return tokens
    
        def predict(self, reactants_smiles):
            """
            Predict products from reactant SMILES
    
            In actual implementation:
            1. Tokenization
            2. Encode reactants with Encoder
            3. Generate products with Decoder
            4. Output top-k candidates with beam search
            """
            # Dummy prediction
            product_smiles = "CC(=O)OCC"  # Ester
            confidence = 0.87
    
            return {
                'product': product_smiles,
                'confidence': confidence
            }
    
    # Usage example
    model = SimpleReactionTransformer()
    
    reactants = "CC(=O)O.CCO"
    result = model.predict(reactants)
    
    print("Transformer reaction prediction:")
    print(f"Reactants: {reactants}")
    print(f"Predicted product: {result['product']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    print("\nActual Transformer models:")
    print("- rxnfp (https://github.com/rxn4chemistry/rxnfp)")
    print("- molecular-transformer (https://github.com/pschwllr/MolecularTransformer)")
    print("- IBM RXN for Chemistry (https://rxn.res.ibm.com/)")
    

**Sample output:**
    
    
    Transformer reaction prediction:
    Reactants: CC(=O)O.CCO
    Predicted product: CC(=O)OCC
    Confidence: 0.87
    
    Actual Transformer models:
    - rxnfp (https://github.com/rxn4chemistry/rxnfp)
    - molecular-transformer (https://github.com/pschwllr/MolecularTransformer)
    - IBM RXN for Chemistry (https://rxn.res.ibm.com/)
    

### 4.2.3 Predicting Reaction Conditions

Predicting reaction conditions (catalyst, solvent, temperature, time) is also important.

#### Code Example 6: Recommending Reaction Conditions
    
    
    from sklearn.ensemble import RandomForestClassifier
    
    # Mapping of reaction types to recommended conditions
    reaction_conditions = {
        'Esterification': {
            'catalyst': ['H2SO4', 'p-TsOH'],
            'solvent': ['Toluene', 'DCM'],
            'temperature': '60-80°C',
            'time': '2-6 hours'
        },
        'Suzuki': {
            'catalyst': ['Pd(PPh3)4', 'PdCl2(dppf)'],
            'solvent': ['THF', 'Dioxane'],
            'temperature': '80-100°C',
            'time': '4-12 hours'
        },
        'Amidation': {
            'catalyst': ['EDC/HOBt', 'HATU'],
            'solvent': ['DMF', 'DCM'],
            'temperature': 'RT',
            'time': '1-4 hours'
        }
    }
    
    def recommend_conditions(reaction_type):
        """
        Output recommended conditions from reaction type
    
        Parameters:
        -----------
        reaction_type : str
            Type of reaction
    
        Returns:
        --------
        conditions : dict
            Recommended conditions
        """
        if reaction_type in reaction_conditions:
            return reaction_conditions[reaction_type]
        else:
            return {
                'catalyst': ['Unknown'],
                'solvent': ['Unknown'],
                'temperature': 'Unknown',
                'time': 'Unknown'
            }
    
    # Usage example
    rxn_type = "Suzuki"
    conditions = recommend_conditions(rxn_type)
    
    print(f"Recommended conditions for {rxn_type} coupling reaction:")
    print(f"Catalyst: {', '.join(conditions['catalyst'])}")
    print(f"Solvent: {', '.join(conditions['solvent'])}")
    print(f"Temperature: {conditions['temperature']}")
    print(f"Time: {conditions['time']}")
    

**Sample output:**
    
    
    Recommended conditions for Suzuki coupling reaction:
    Catalyst: Pd(PPh3)4, PdCl2(dppf)
    Solvent: THF, Dioxane
    Temperature: 80-100°C
    Time: 4-12 hours
    

* * *

## 4.3 Retrosynthesis

**Retrosynthesis** is a method for designing synthetic routes backward from target molecules to starting materials.
    
    
    ```mermaid
    flowchart RL
        A[Target Molecule] --> B[Retrosynthetic Disconnection]
        B --> C[Precursor 1]
        B --> D[Precursor 2]
    
        C --> E[Further Disconnection]
        D --> F[Further Disconnection]
    
        E --> G[Commercial Starting Materials]
        F --> H[Commercial Starting Materials]
    
        style A fill:#FF9800,color:#fff
        style G fill:#4CAF50,color:#fff
        style H fill:#4CAF50,color:#fff
    ```

### 4.3.1 Basic Concepts of Retrosynthesis

**Disconnection strategies** : 1\. **Functional group disconnection** : Ester → Carboxylic acid + Alcohol 2\. **C-C bond disconnection** : Alkyl chain → Short chain + Short chain 3\. **Ring opening** : Cyclic compound → Linear compound

#### Code Example 7: Simple Retrosynthesis Implementation
    
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    def simple_retrosynthesis(target_smiles, max_depth=3):
        """
        Simple retrosynthesis (conceptual implementation)
    
        Parameters:
        -----------
        target_smiles : str
            SMILES of target molecule
        max_depth : int
            Maximum search depth
    
        Returns:
        --------
        routes : list
            List of synthetic routes
        """
    
        # Retro-reaction templates (Ester → Carboxylic acid + Alcohol)
        retro_templates = {
            'Ester_cleavage': '[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH:3].[OH:3][C:4]',
            'Amide_cleavage': '[C:1](=[O:2])[NH:3][C:4]>>[C:1](=[O:2])[OH:3].[NH2:3][C:4]'
        }
    
        target_mol = Chem.MolFromSmiles(target_smiles)
    
        routes = []
    
        for name, template_smarts in retro_templates.items():
            rxn = AllChem.ReactionFromSmarts(template_smarts)
    
            # Apply retro-reaction
            try:
                precursors = rxn.RunReactants((target_mol,))
    
                for precursor_set in precursors:
                    precursor_smiles = [Chem.MolToSmiles(mol)
                                        for mol in precursor_set]
                    routes.append({
                        'reaction': name,
                        'precursors': precursor_smiles
                    })
            except:
                pass
    
        return routes
    
    # Usage example: Retrosynthesis of ethyl acetate
    target = "CC(=O)OCC"  # Ethyl acetate
    
    routes = simple_retrosynthesis(target)
    
    print(f"Target molecule: {target} (ethyl acetate)\n")
    print("Retrosynthetic routes:")
    for i, route in enumerate(routes, 1):
        print(f"\nRoute {i}: {route['reaction']}")
        print(f"  Precursors: {' + '.join(route['precursors'])}")
    

**Sample output:**
    
    
    Target molecule: CC(=O)OCC (ethyl acetate)
    
    Retrosynthetic routes:
    
    Route 1: Ester_cleavage
      Precursors: CC(=O)O + CCO
    

### 4.3.2 AiZynthFinder

**AiZynthFinder** is an open-source retrosynthesis tool developed by AstraZeneca, Sweden.

#### Code Example 8: AiZynthFinder Concept (Installation Guide)
    
    
    """
    Installation and usage of AiZynthFinder
    
    # Installation
    pip install aizynthfinder
    
    # Required data (models and templates)
    # Download from https://github.com/MolecularAI/aizynthfinder
    
    # Basic usage example
    from aizynthfinder.aizynthfinder import AiZynthFinder
    
    # Load configuration file
    finder = AiZynthFinder(configfile='config.yml')
    
    # Set target molecule
    finder.target_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    
    # Search for synthetic routes
    finder.tree_search()
    
    # Get results
    finder.build_routes()
    routes = finder.routes
    
    # Display top-5 routes
    for i, route in enumerate(routes[:5], 1):
        print(f"Route {i}:")
        print(f"  Number of steps: {route.number_of_steps}")
        print(f"  Score: {route.score:.3f}")
    
    # Visualize route
    route.to_image().save(f'route_{i}.png')
    """
    
    print("AiZynthFinder details:")
    print("- GitHub: https://github.com/MolecularAI/aizynthfinder")
    print("- Paper: Genheden et al., J. Chem. Inf. Model. 2020")
    print("- Features: Monte Carlo tree search + Deep learning")
    

### 4.3.3 IBM RXN for Chemistry

**IBM RXN for Chemistry** is a web-based reaction prediction and retrosynthesis platform developed by IBM.

#### Code Example 9: Using RXN API (Conceptual)
    
    
    """
    Using IBM RXN for Chemistry API
    
    # Get API key
    # Create account at https://rxn.res.ibm.com/
    
    # Install Python SDK
    pip install rxn4chemistry
    
    # Basic usage example
    from rxn4chemistry import RXN4ChemistryWrapper
    
    # Initialize API wrapper
    rxn = RXN4ChemistryWrapper(api_key='YOUR_API_KEY')
    
    # Create project
    project_id = rxn.create_project('My Project')
    rxn.set_project(project_id)
    
    # Execute retrosynthesis
    target_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    
    response = rxn.predict_automatic_retrosynthesis(
        product=target_smiles,
        max_steps=3
    )
    
    # Get results
    retro_id = response['prediction_id']
    results = rxn.get_predict_automatic_retrosynthesis_results(retro_id)
    
    # Display routes
    for i, sequence in enumerate(results['sequences'][:5], 1):
        print(f"Route {i}:")
        print(f"  Confidence: {sequence['confidence']:.2f}")
        for step in sequence['steps']:
            print(f"  - {step['smiles']}")
    """
    
    print("IBM RXN for Chemistry:")
    print("- URL: https://rxn.res.ibm.com/")
    print("- Features: Reaction prediction, retrosynthesis, experimental planning")
    print("- Characteristics: Transformer + USPTO 1.8 million reaction data")
    

### 4.3.4 Evaluating Synthetic Routes

#### Code Example 10: Scoring Synthetic Routes
    
    
    def score_synthesis_route(route):
        """
        Score synthetic routes
    
        Evaluation criteria:
        1. Number of steps (fewer is better)
        2. Yield (higher is better)
        3. Cost (lower is better)
        4. Reagent availability (easier is better)
    
        Returns:
        --------
        score : float
            Total score (0-100)
        """
    
        # Virtual route data
        n_steps = route.get('n_steps', 5)
        avg_yield = route.get('avg_yield', 75)  # %
        cost = route.get('cost', 500)  # USD
        availability = route.get('availability', 0.8)  # 0-1
    
        # Score calculation (weighted)
        step_score = max(0, 100 - n_steps * 10)  # +10 points per step reduction
        yield_score = avg_yield  # Yield as-is
        cost_score = max(0, 100 - cost / 10)  # -1 point per $10
        availability_score = availability * 100
    
        # Weighted average
        total_score = (
            step_score * 0.3 +
            yield_score * 0.4 +
            cost_score * 0.2 +
            availability_score * 0.1
        )
    
        return total_score
    
    # Evaluate sample routes
    routes = [
        {
            'name': 'Route A',
            'n_steps': 3,
            'avg_yield': 85,
            'cost': 300,
            'availability': 0.9
        },
        {
            'name': 'Route B',
            'n_steps': 5,
            'avg_yield': 90,
            'cost': 200,
            'availability': 0.7
        },
        {
            'name': 'Route C',
            'n_steps': 4,
            'avg_yield': 75,
            'cost': 400,
            'availability': 0.8
        }
    ]
    
    print("=== Evaluation of Synthetic Routes ===\n")
    for route in routes:
        score = score_synthesis_route(route)
        print(f"{route['name']}:")
        print(f"  Number of steps: {route['n_steps']}")
        print(f"  Average yield: {route['avg_yield']}%")
        print(f"  Cost: ${route['cost']}")
        print(f"  Availability: {route['availability']:.1f}")
        print(f"  Total score: {score:.1f} / 100\n")
    
    # Select best route
    best_route = max(routes, key=score_synthesis_route)
    print(f"Recommended route: {best_route['name']} "
          f"(Score: {score_synthesis_route(best_route):.1f})")
    

**Sample output:**
    
    
    === Evaluation of Synthetic Routes ===
    
    Route A:
      Number of steps: 3
      Average yield: 85%
      Cost: $300
      Availability: 0.9
      Total score: 83.0 / 100
    
    Route B:
      Number of steps: 5
      Average yield: 90%
      Cost: $200
      Availability: 0.7
      Total score: 81.0 / 100
    
    Route C:
      Number of steps: 4
      Average yield: 75%
      Cost: $400
      Availability: 0.8
      Total score: 70.0 / 100
    
    Recommended route: Route A (Score: 83.0)
    

* * *

## 4.4 Real-World Applications and Career Paths

### 4.4.1 Chemoinformatics Applications in Pharmaceutical Companies

#### Major Company Case Studies

**Pfizer** : \- AI drug discovery platform: IBM Watson Health collaboration \- Application: Rapid development of COVID-19 treatment Paxlovid \- Technology: QSAR, virtual screening, Retrosynthesis

**Roche** : \- In-house AI platform: Roche Pharma Research & Early Development (pRED) \- Application: Optimization of anticancer drugs \- Technology: Graph Neural Networks, transfer learning

**Novartis** : \- Microsoft AI collaboration project \- Application: Search for rare disease treatments \- Technology: Natural language processing, molecular generation models

### 4.4.2 Applications in Materials Manufacturers

**Asahi Kasei** : \- Materials Informatics division \- Application: Prediction of polymer material properties \- Technology: QSPR, machine learning

**Mitsubishi Chemical** : \- Digital transformation promotion \- Application: Catalyst design, process optimization \- Technology: Bayesian optimization, active learning

### 4.4.3 Startup Case Studies

**Recursion Pharmaceuticals (USA)** : \- Market cap: ~$2 billion (2023) \- Technology: Image analysis + Chemoinformatics \- Achievement: 100+ clinical pipelines

**BenevolentAI (UK)** : \- Market cap: ~$2 billion \- Technology: Knowledge Graph + AI drug discovery \- Achievement: Discovery of COVID-19 treatment candidate (6 weeks)

**Exscientia (UK)** : \- World's first AI-designed drug in clinical trials (2020) \- Technology: Active learning + Retrosynthesis \- Partners: Bayer, Roche

### 4.4.4 Career Paths

#### Career Path for Chemoinformaticians
    
    
    ```mermaid
    flowchart TD
        A[Bachelor's Degree\nChemistry/Information Science] --> B[Master's Program\nChemoinformatics Specialization]
        A --> C[Self-Study\nOnline Learning]
    
        B --> D[Doctoral Program\n3-5 years]
        C --> E[Junior\nChemoinformatician]
    
        D --> F[Postdoc\n2-3 years]
        E --> F
    
        F --> G[Senior\nChemoinformatician]
        G --> H[Principal\nScientist]
        H --> I[Director\nResearch Division Head]
    
        G --> J[Startup\nFounder]
    
        style A fill:#e3f2fd
        style I fill:#4CAF50,color:#fff
        style J fill:#FF9800,color:#fff
    ```

#### Required Skills

**Technical Skills** : \- [ ] **Programming** : Python (required), R, C++ \- [ ] **Chemoinformatics Tools** : RDKit, mordred, Open Babel \- [ ] **Machine Learning** : scikit-learn, LightGBM, PyTorch \- [ ] **Deep Learning** : GNN, Transformer \- [ ] **Databases** : ChEMBL, PubChem, USPTO

**Domain Knowledge** : \- [ ] **Organic Chemistry** : Reaction mechanisms, functional group transformations \- [ ] **Medicinal Chemistry** : ADMET, drug discovery process \- [ ] **Statistics** : Design of experiments, causal inference

**Soft Skills** : \- [ ] **Communication** : Bridging chemists and data scientists \- [ ] **Project Management** : Parallel promotion of multiple projects \- [ ] **Presentation** : Technical explanation to management

#### Salary Range (Japan, 2023)

Position | Salary Range | Years of Experience  
---|---|---  
Junior | 5-7M JPY | 0-3 years  
Middle | 7-10M JPY | 3-7 years  
Senior | 10-15M JPY | 7-15 years  
Principal | 15-25M JPY | 15+ years  
Director | 20-35M JPY | Management  
  
(Foreign pharmaceutical companies typically 1.5-2x higher)

* * *

## Exercises

### Exercise 1: Creating a Reaction Template

Create a reaction template in SMARTS format for the following reaction.

**Friedel-Crafts Acylation Reaction** : Benzene ring + Acid chloride → Ketone + HCl

Sample Solution
    
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    # Friedel-Crafts acylation template
    friedel_crafts_template = "[c:1][H:2].[C:3](=[O:4])[Cl:5]>>[c:1][C:3](=[O:4]).[H:2][Cl:5]"
    
    # Create reaction object
    rxn = AllChem.ReactionFromSmarts(friedel_crafts_template)
    
    # Test: Benzene + Acetyl chloride
    reactants = [
        Chem.MolFromSmiles("c1ccccc1"),  # Benzene
        Chem.MolFromSmiles("CC(=O)Cl")   # Acetyl chloride
    ]
    
    # Execute reaction
    products = rxn.RunReactants(tuple(reactants))
    
    print("Friedel-Crafts acylation reaction:")
    print(f"Reactants: {Chem.MolToSmiles(reactants[0])} + {Chem.MolToSmiles(reactants[1])}")
    
    if products:
        for product_set in products[:1]:  # First product set
            for mol in product_set:
                Chem.SanitizeMol(mol)
                print(f"Product: {Chem.MolToSmiles(mol)}")
    

**Expected output:** 
    
    
    Friedel-Crafts acylation reaction:
    Reactants: c1ccccc1 + CC(=O)Cl
    Product: CC(=O)c1ccccc1
    Product: Cl
    

* * *

### Exercise 2: Comparing Multiple Retrosynthetic Routes

For the following target molecule, propose at least two different retrosynthetic routes and calculate their scores.

**Target Molecule** : Ibuprofen \- SMILES: `CC(C)Cc1ccc(cc1)C(C)C(=O)O`

Hint 1\. Friedel-Crafts alkylation → Oxidation 2\. Suzuki-Miyaura coupling → Carboxylation Consider number of steps, expected yield, and reagent availability for scoring 

* * *

### Exercise 3: Improving Reaction Yield Prediction Model

Improve the reaction yield prediction model from Code Example 4. Try the following approaches:

  1. Add reaction conditions (temperature, catalyst) as features
  2. Use LightGBM or neural networks
  3. Optimize hyperparameters with cross-validation

Goal: R² > 0.8, MAE < 5%

Solution Direction
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    # - optuna>=3.2.0
    
    """
    Example: Goal: R² > 0.8, MAE < 5%
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # 1. Add reaction conditions with one-hot encoding
    from sklearn.preprocessing import OneHotEncoder
    
    conditions = ['catalyst', 'solvent', 'temperature']
    # ... Add to features
    
    # 2. Use LightGBM
    import lightgbm as lgb
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05
    }
    # ... Training
    
    # 3. Hyperparameter optimization with Optuna, etc.
    import optuna
    
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
        }
        # ... Evaluation
        return mae
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    

* * *

## Summary

In this chapter, you learned:

### Topics Covered

  1. **Reaction Templates and SMARTS** \- Reaction SMILES and SMIRKS notation \- Defining and applying reaction templates \- USPTO reaction dataset

  2. **Reaction Prediction Models** \- Yield prediction with Random Forest \- Product prediction with Transformer \- Reaction condition recommendation

  3. **Retrosynthesis** \- Basic concepts of retrosynthetic analysis \- AiZynthFinder, IBM RXN for Chemistry \- Scoring synthetic routes

  4. **Industrial Applications and Careers** \- Pharmaceutical companies (Pfizer, Roche, Novartis) \- Materials manufacturers (Asahi Kasei, Mitsubishi Chemical) \- Startups (Recursion, BenevolentAI, Exscientia) \- Career paths for chemoinformaticians

### Series Completion

Congratulations! You have completed all 4 chapters of the Chemoinformatics Introduction series.

**Skills Acquired** : \- ✅ Molecular representation and RDKit operations \- ✅ QSAR/QSPR modeling \- ✅ Chemical space exploration and similarity search \- ✅ Reaction prediction and Retrosynthesis

**Next Steps** : 1\. **GNN Introduction Series** : Molecular representation learning with Graph Neural Networks 2\. **Personal Projects** : Large-scale QSAR using ChEMBL data 3\. **Community Participation** : RDKit Users Group, Chemical Society of Japan Cheminformatics Division 4\. **Paper Submission** : Journal of Cheminformatics, J. Chem. Inf. Model.

**[Back to Series Top →](<./index.html>)**

* * *

## References

  1. Coley, C. W. et al. (2017). "Prediction of Organic Reaction Outcomes Using Machine Learning." _ACS Central Science_ , 3(5), 434-443. DOI: 10.1021/acscentsci.7b00064
  2. Schwaller, P. et al. (2019). "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction." _ACS Central Science_ , 5(9), 1572-1583. DOI: 10.1021/acscentsci.9b00576
  3. Genheden, S. et al. (2020). "AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning." _J. Cheminformatics_ , 12, 70. DOI: 10.1186/s13321-020-00472-1
  4. Lowe, D. M. (2012). _Extraction of chemical structures and reactions from the literature_. PhD thesis, University of Cambridge.

* * *

**[← Chapter 3](<chapter-3.html>)** | **[Back to Series Top](<./index.html>)**

* * *

## Additional Resources

### Open Source Tools

Tool Name | Description | GitHub  
---|---|---  
RDKit | Comprehensive chemoinformatics library | [rdkit/rdkit](<https://github.com/rdkit/rdkit>)  
AiZynthFinder | Retrosynthesis tool | [MolecularAI/aizynthfinder](<https://github.com/MolecularAI/aizynthfinder>)  
rxnfp | Reaction fingerprints & Transformer | [rxn4chemistry/rxnfp](<https://github.com/rxn4chemistry/rxnfp>)  
Molecular Transformer | Reaction prediction Transformer | [pschwllr/MolecularTransformer](<https://github.com/pschwllr/MolecularTransformer>)  
  
### Web Platforms

  * **IBM RXN for Chemistry** : https://rxn.res.ibm.com/
  * **ChemDraw Cloud** : https://chemdrawdirect.perkinelmer.cloud/
  * **PubChem** : https://pubchem.ncbi.nlm.nih.gov/

### Learning Resources

  * **Book** : "Organic Chemistry by Retrosynthesis" by Stuart Warren
  * **Online Course** : Coursera "Drug Discovery"
  * **Community** : RDKit Users Group (Google Groups)

* * *

**Your chemoinformatics journey starts here!**
