---
title: Drug Discovery MI Practical Case Studies
chapter_title: Drug Discovery MI Practical Case Studies
subtitle: Learning Practical Methods from 5 Industrial Applications
---

# Chapter 4: Case Studies and Real-World Applications

This chapter focuses on practical applications of Case Studies and Real. You will learn ESMFold(Meta AI): Language model-based.

## Learning Objectives

By completing this chapter, you will be able to:

  1. **Understand Real Examples** : Explain specific strategies and achievements of successful AI drug discovery companies and projects
  2. **Technology Commercialization** : Understand how revolutionary technologies like AlphaFold 2 are integrated into drug discovery pipelines
  3. **Molecular Generative AI** : Explain the principles and applications of VAE, GAN, and Transformer-based molecular generation models
  4. **Best Practices** : Analyze success and failure factors in AI drug discovery projects
  5. **Career Paths** : Understand career-building options and required skillsets in the AI drug discovery field

* * *

## 4.1 Industry Success Stories

AI drug discovery has rapidly become practical in recent years, with many startups and pharmaceutical giants achieving results. This section examines representative companies and their strategies in detail.

### 4.1.1 Exscientia: Pioneer of AI-Driven Drug Discovery

**Company Overview** : \- Founded: 2012 (Oxford, UK) \- Founder: Andrew Hopkins (Professor of Pharmacology) \- Employees: ~400 (2023) \- Funding: Over $500 million total \- IPO: NASDAQ (2021, ticker: EXAI)

**Technical Approach** : Exscientia advocates "AI-Designed Medicine" and integrates AI at each stage of the drug discovery process.
    
    
    Traditional Drug Discovery Process:
    Target ID → Hit Discovery → Lead Optimization → Preclinical → Clinical
    (4-5 years)    (2-3 years)        (2-3 years)         (1-2 years)   (6-10 years)
    
    Exscientia's AI Process:
    Target ID → AI Hit Discovery → AI Lead Opt → Preclinical → Clinical
    (6 months)     (8-12 months)       (8-12 months)   (1-2 years)   (6-10 years)
    
    ⇒ Reduces preclinical stage from ~4.5 years to 2-2.5 years
    

**Key Technologies** :

  1. **Active Learning Platform** : \- Iterative cycle of experimental data and computational predictions \- Achieves optimization with few experiments (1/10 of traditional data volume) \- Integration of Bayesian optimization and multitask learning

  2. **Centaur Chemist** : \- Collaborative platform between human chemists and AI \- AI proposes design candidates, humans validate and modify \- Automatic evaluation of synthesizability and patent status

**Specific Achievements** :

Project | Partner | Disease Area | Milestone  
---|---|---|---  
DSP-1181 | Sumitomo Dainippon Pharma | Obsessive-Compulsive Disorder (OCD) | Clinical trial started 2020 (world's first AI-designed drug)  
EXS-21546 | Bristol Myers Squibb | Cancer immunotherapy | Preclinical completed 2021  
CDK7 inhibitor | Sanofi | Cancer | In development (AI design period: 8 months)  
PKC-θ inhibitor | Internal development | Autoimmune diseases | Clinical trial planning 2023  
  
**Business Model** : \- Partnerships with pharmaceutical majors (Sanofi, Bristol Myers Squibb, Bayer, etc.) \- Milestone payments + royalty contracts \- Internal pipeline development (cancer, autoimmune, neurological diseases) \- Platform technology licensing

**Lessons Learned** : \- **Human-AI Collaboration** : Not full automation, but using AI as an assistive tool \- **Data Efficiency** : Learning from limited data through Active Learning \- **Iterative Validation** : Incorporating experimental validation at each stage to improve accuracy \- **Patent Strategy** : Protecting the AI design process itself as intellectual property

* * *

### 4.1.2 Insilico Medicine: Generative AI and Aging Research

**Company Overview** : \- Founded: 2014 (Hong Kong, now headquartered in New York, USA) \- Founder: Alex Zhavoronkov (Bioinformatician) \- Employees: ~400 \- Funding: ~$400 million total \- Specialty: Fusion of aging research and AI drug discovery

**Technology Platform** :

Insilico developed the "Pharma.AI" platform integrating three AI engines:

  1. **PandaOmics** (Target Discovery): \- Multi-omics data analysis \- Identification of disease-related genes and pathways \- Identification of aging markers

  2. **Chemistry42** (Molecular Generation): \- GAN-based (Generative Adversarial Networks) \- Conditional molecular generation (property specification) \- Integration of synthesizability prediction

  3. **InClinico** (Clinical Trial Prediction): \- Clinical trial success probability prediction \- Patient stratification \- Biomarker selection

**Notable Achievements** :

**INS018_055 (Idiopathic Pulmonary Fibrosis Treatment)** : \- Announced 2021: Achieved clinical trial initiation in 18 months from AI design (world record) \- Drastically shortened traditional drug discovery timeline (4-5 years) \- Selected from 78 molecular candidates generated by Chemistry42 \- Phase I started in China 2022, Phase II planned 2023

**Design Process Details** :
    
    
    Step 1: Target Discovery (PandaOmics)
      - Analysis of pulmonary fibrosis-related public data
      - Selected DDR1 (Discoidin Domain Receptor 1) as target
      - Rationale: Key regulator of fibrotic signaling pathways
    
    Step 2: Molecular Generation (Chemistry42)
      Period: 21 days
      - Generated ~30,000 molecules with GAN
      - ADMET filtering → ~3,000 molecules
      - Synthesizability scoring → ~400 molecules
      - Docking simulation → Selected 78 molecules for synthesis
    
    Step 3: Experimental Validation
      Period: 18 months
      - Synthesized 78 molecules
      - In vitro activity evaluation: ~30 molecules showed DDR1 inhibitory activity
      - ADMET experimental evaluation: 6 molecules favorable
      - In vivo animal experiments: 2 molecules showed efficacy
      - Final candidate INS018_055 selected
    
    Step 4: Preclinical Testing
      Period: 12 months
      - GLP toxicity studies
      - Pharmacokinetic studies
      - Safety evaluation
      → Phase I clinical trial approval (China NMPA) June 2022
    

**Technical Innovations** :

Chemistry42 generative AI architecture:
    
    
    Input: Target protein structure + desired properties (ADMET, synthesizability)
        ↓
    [Conditional GAN (cGAN)]
        ↓ Generation
    Molecular candidates (SMILES format)
        ↓
    [Scoring Module]
     - Binding affinity prediction (docking)
     - ADMET prediction (machine learning models)
     - Synthesizability score (retrosynthesis analysis)
     - Patent avoidance check
        ↓
    Optimized molecule output
    

**Other Pipeline** : \- Cancer therapeutics (multiple targets) \- COVID-19 therapeutics (3CL protease inhibitors) \- Parkinson's disease therapeutics \- Aging-related disease therapeutics

**Business Strategy** : \- **Focus on Internal Pipeline** : Emphasis on internal development over partnerships \- **Integration with Aging Research** : Viewing diseases as aspects of aging \- **Global Expansion** : Parallel development in China, USA, and Europe

**Lessons Learned** : \- **Integrated Platform** : Consistent AI system from target discovery to clinical prediction \- **Generative AI Commercialization** : Pioneer example of applying GAN to actual drug discovery \- **Speed Focus** : Record-breaking speed of 18 months to clinical trials \- **Data-Driven** : Continuous model improvement through experimental data feedback

* * *

### 4.1.3 Recursion Pharmaceuticals: Fusion of High-Throughput Experiments and AI

**Company Overview** : \- Founded: 2013 (Salt Lake City, Utah, USA) \- Founder: Chris Gibson (PhD, former medical student) \- Employees: ~500 \- Funding: ~$700 million total \- IPO: NASDAQ (2021, ticker: RXRX) \- Specialty: Holds world's largest biological dataset

**Technical Approach** :

Recursion's unique strategy is "**Automation of Data Generation** ". While traditional AI drug discovery companies rely on public data, Recursion generates large-scale experimental data in-house.

**Data Generation Platform** :

  1. **Automated Lab** : \- 24/7 operation with robotic systems \- Processes over 2.2 million wells per week \- Generates ~2 million experimental data points annually

  2. **Imaging System** : \- Automatic high-resolution cell image acquisition \- Captures ~1.6 million images per week \- Visualizes cell morphology and function with 8 fluorescence channels

  3. **Data Scale** (as of 2023): \- Total images: ~23 billion pixels (18 petabytes) \- Compounds tested: ~2 million types \- Cell lines: ~100 types \- Gene perturbations: ~3,000 types

**AI Analysis Approach** :

Recursion employs "**Phenomics** ":
    
    
    Phenomics: Comprehensive analysis of cellular phenotypes (appearance/function)
    
    1. Cell Imaging
       Administer compounds to cells
       ↓
       Multi-channel microscopy imaging (nucleus, mitochondria, ER, etc.)
       ↓
       Image data (1024×1024 pixels × 8 channels)
    
    2. Feature Extraction (CNN)
       Image → Convolutional Neural Network
       ↓
       High-dimensional feature vector (~1,000 dimensions)
       Examples: nucleus size, mitochondria count, cell morphology, etc.
    
    3. Mapping in Phenotype Space
       Similar phenotype = similar biological action
       ↓
       Compare known drugs with unknown compounds
       ↓
       "This novel compound causes cellular changes similar to known diabetes drugs"
       → Suggests potential application to diabetes
    

**Specific Achievements** :

Project | Disease Area | Status | Features  
---|---|---|---  
REC-994 | Cerebral Cavernous Malformation (CCM) | Phase II | Rare disease, Bayer partnership  
REC-2282 | Neurofibromatosis Type 2 (NF2) | Phase II | Rare disease  
REC-4881 | Familial Adenomatous Polyposis | Preclinical | Rare disease  
Cancer immunotherapy | Solid tumors | Preclinical | Roche/Genentech partnership  
Fibrosis therapeutics | Multiple organs | Preclinical | Bayer partnership  
  
**Strategic Partnership with Bayer** (2020~): \- Total contract value: Up to $5 billion (including milestones) \- Goal: Discover up to 10 new drug candidates over 10 years \- Areas: Cancer, cardiovascular diseases, rare diseases \- Full access to Recursion's platform provided

**Technical Details: Image-Based Drug Efficacy Prediction**

Actual analysis pipeline:
    
    
    # Conceptual code (simplified Recursion system)
    
    # 1. Image data preprocessing
    def preprocess_image(image_path):
        """8-channel cell image preprocessing"""
        img = load_multichannel_image(image_path)  # (1024, 1024, 8)
    
        # Normalization and standardization
        normalized = normalize_channels(img)
    
        # Data augmentation (rotation, flip)
        augmented = augment(normalized)
    
        return augmented
    
    # 2. CNN feature extraction
    class PhenomicEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            # ResNet50-based encoder (modified for 8-channel input)
            self.encoder = ResNet50(input_channels=8)
            self.fc = nn.Linear(2048, 1024)
    
        def forward(self, x):
            # Image → high-dimensional feature vector
            features = self.encoder(x)  # (batch, 2048)
            embedding = self.fc(features)  # (batch, 1024)
            return embedding
    
    # 3. Phenotypic similarity search
    def find_similar_phenotypes(query_compound, reference_library, top_k=10):
        """
        Search for known drugs with similar phenotypes to query compound
        """
        query_embedding = encoder(query_compound.image)  # (1024,)
    
        # Calculate similarity with all compounds in reference library
        similarities = []
        for ref_compound in reference_library:
            ref_embedding = encoder(ref_compound.image)
            similarity = cosine_similarity(query_embedding, ref_embedding)
            similarities.append((ref_compound, similarity))
    
        # Sort by similarity
        ranked = sorted(similarities, key=lambda x: x[1], reverse=True)
    
        return ranked[:top_k]
    
    # 4. Therapeutic efficacy prediction
    def predict_therapeutic_area(compound):
        """Predict therapeutic area from phenotypic similarity"""
        similar_drugs = find_similar_phenotypes(compound, known_drug_library)
    
        # Aggregate disease areas of similar drugs
        disease_votes = {}
        for drug, similarity in similar_drugs:
            for disease in drug.indications:
                if disease not in disease_votes:
                    disease_votes[disease] = 0
                disease_votes[disease] += similarity
    
        # Most likely disease area
        predicted_disease = max(disease_votes, key=disease_votes.get)
        confidence = disease_votes[predicted_disease] / sum(disease_votes.values())
    
        return predicted_disease, confidence
    
    # Usage example
    new_compound = load_compound("CHEMBL12345")
    disease, conf = predict_therapeutic_area(new_compound)
    print(f"Predicted disease area: {disease}, Confidence: {conf:.2f}")
    # Output example: Predicted disease area: Alzheimer's disease, Confidence: 0.78
    

**Lessons Learned** : \- **Data is Key** : Building in-house data generation infrastructure \- **Image AI** : Cell images are useful information sources beyond text/structural data \- **Rare Disease Strategy** : Building track record in less competitive areas \- **Pharma Major Partnerships** : Balance of internal development and partnerships

* * *

### 4.1.4 BenevolentAI: Knowledge Graphs and Scientific Literature Mining

**Company Overview** : \- Founded: 2013 (London, UK) \- Founder: Ken Mulvany (Entrepreneur, PhD in Pharmacy) \- Employees: ~300 \- Funding: ~$300 million total \- IPO: Euronext Amsterdam (2022, via SPAC) \- Specialty: Utilization of knowledge graphs and natural language processing (NLP)

**Technology Platform** :

BenevolentAI's core is the massive biomedical knowledge graph called "**Benevolent Platform** ".

**Knowledge Graph Structure** :
    
    
    Knowledge Graph: Representing knowledge with entities and relations
    
    Entities (Nodes):
    - Genes: ~20,000 types
    - Proteins: ~100,000 types
    - Compounds: ~2 million types
    - Diseases: ~10,000 types
    - Cell types: ~500 types
    - Tissues: ~200 types
    
    Relations (Edges):
    - "Gene A" → [encodes] → "Protein B"
    - "Compound C" → [inhibits] → "Protein B"
    - "Protein B" → [upregulated_in] → "Disease D"
    - "Disease D" → [affects] → "Tissue E"
    
    ⇒ Total nodes: ~3 million
    ⇒ Total edges: ~100 million
    

**Data Sources** : 1\. Scientific literature (PubMed, arXiv): ~30 million papers 2\. Structured databases (ChEMBL, UniProt, DisGeNET) 3\. Clinical trial data (ClinicalTrials.gov) 4\. Patent databases 5\. Internal experimental data

**NLP Technology** :

BenevolentAI develops proprietary biomedical NLP models:
    
    
    # Conceptual example: Automatic knowledge extraction from papers
    
    class BiomedicalNER(nn.Module):
        """Biomedical Named Entity Recognition (NER) model"""
    
        def __init__(self):
            super().__init__()
            # BioBERT-based (pre-trained on PubMed)
            self.bert = BioBERT.from_pretrained('biobert-v1.1')
            self.classifier = nn.Linear(768, num_entity_types)
    
        def extract_entities(self, text):
            """
            Extract biomedical entities from text
    
            Input: "EGFR mutations are associated with lung cancer resistance to gefitinib."
            Output: [
                ("EGFR", "GENE"),
                ("lung cancer", "DISEASE"),
                ("gefitinib", "DRUG")
            ]
            """
            tokens = self.bert.tokenize(text)
            embeddings = self.bert(tokens)
            entity_labels = self.classifier(embeddings)
    
            entities = []
            for token, label in zip(tokens, entity_labels):
                if label != "O":  # "O" = non-entity
                    entities.append((token, label))
    
            return entities
    
    class RelationExtraction(nn.Module):
        """Relation extraction between entities"""
    
        def extract_relations(self, text, entities):
            """
            Input: "EGFR mutations are associated with lung cancer"
                  entities = [("EGFR", "GENE"), ("lung cancer", "DISEASE")]
    
            Output: [
                ("EGFR", "associated_with", "lung cancer", confidence=0.89)
            ]
            """
            # Generate entity pairs
            for e1, e2 in combinations(entities, 2):
                # Context encoding
                context = self.encode_context(text, e1, e2)
    
                # Relation classification
                relation_prob = self.relation_classifier(context)
    
                if relation_prob.max() > threshold:
                    relation = relation_types[relation_prob.argmax()]
                    yield (e1, relation, e2, relation_prob.max())
    
    # Usage example
    ner_model = BiomedicalNER()
    rel_model = RelationExtraction()
    
    text = "Recent studies show that baricitinib inhibits JAK1/JAK2 and may be effective in treating severe COVID-19."
    
    entities = ner_model.extract_entities(text)
    # [("baricitinib", "DRUG"), ("JAK1", "GENE"), ("JAK2", "GENE"), ("COVID-19", "DISEASE")]
    
    relations = rel_model.extract_relations(text, entities)
    # [
    #   ("baricitinib", "inhibits", "JAK1", 0.92),
    #   ("baricitinib", "inhibits", "JAK2", 0.91),
    #   ("baricitinib", "treats", "COVID-19", 0.78)
    # ]
    
    # Add these to knowledge graph
    knowledge_graph.add_relations(relations)
    

**Graph-Based Reasoning** :

Path exploration on knowledge graphs generates new hypotheses:
    
    
    Example: Novel therapeutic target discovery for Alzheimer's disease
    
    Query: "What existing drugs could treat Alzheimer's disease?"
    
    Graph search:
    Alzheimer's Disease →[involves]→ Amyloid-beta protein
                                              ↓
                                          [cleaved_by]
                                              ↓
                                          BACE1 enzyme
                                              ↑
                                          [inhibited_by]
                                              ↑
                              Baricitinib (rheumatoid arthritis drug)
                                              ↑
                                          [inhibits]
                                              ↑
                                          JAK1/JAK2
                                              ↓
                                     [regulates]
                                              ↓
                                       Inflammation
                                              ↓
                                     [associated_with]
                                              ↓
                                  Alzheimer's Disease
    
    Inference: Baricitinib is a rheumatoid arthritis drug, but
              may also be effective for Alzheimer's disease through anti-inflammatory action
    

**COVID-19 Therapeutic Discovery (2020)** :

Real example of BenevolentAI's knowledge graph and AI identifying baricitinib as a COVID-19 therapeutic candidate:
    
    
    Discovery process (February 2020, paper published):
    
    1. Knowledge graph query
       "What approved drugs could inhibit SARS-CoV-2 viral entry mechanism?"
    
    2. Graph reasoning
       SARS-CoV-2 →[enters_via]→ ACE2 receptor
                                       ↓
                                  [endocytosis]
                                       ↓
                               AP2-associated protein kinase 1 (AAK1)
                                       ↑
                                  [inhibited_by]
                                       ↑
                               Baricitinib, Fedratinib, etc.
    
    3. Additional filtering
       - Lung tissue accessibility (pharmacokinetics)
       - Anti-inflammatory effects (COVID-19 severity is excessive immune response)
       - Existing safety data
    
    4. Prediction result
       Identified Baricitinib as top candidate
    
    5. Experimental validation
       → Eli Lilly conducted clinical trials
       → FDA Emergency Use Authorization (EUA) granted November 2020
       → 13% reduction in mortality for severe COVID-19 patients (vs. placebo)
    
    Discovery to approval: ~9 months (traditional drug discovery takes 10-15 years)
    

**Other Pipeline** : \- **BEN-2293** (Atrophic age-related macular degeneration, Phase IIa): AstraZeneca partnership \- **BEN-8744** (Heart failure): Preclinical \- Cancer immunotherapy candidates (multiple)

**Lessons Learned** : \- **Knowledge Integration** : Integrating different data sources to gain new insights \- **Hypothesis Generation** : AI discovers connections humans might miss \- **Drug Repositioning** : Discovering new applications for existing drugs (shortened development time) \- **Real-World Validation** : Success with COVID-19 demonstrated technology effectiveness

* * *

## 4.2 AlphaFold 2 and the Structure-Based Drug Discovery Revolution

### 4.2.1 The AlphaFold 2 Impact

In November 2020, DeepMind (Google subsidiary) announced AlphaFold 2, which essentially "solved" the 50-year-old protein structure prediction problem and revolutionized drug discovery research.

**Pre-AlphaFold 2 Situation** : \- Protein structure determination methods: \- X-ray crystallography: Months to years, 30-50% success rate \- NMR spectroscopy: Small proteins only, months required \- Cryo-EM: High cost, specialized equipment required \- Known structures: ~170,000 (< 1% of all proteins)

**AlphaFold 2 Achievements** : \- Prediction accuracy: CASP14 competition median GDT_TS 92.4 (equivalent to experimental structures) \- Prediction time: Minutes to hours per protein \- Public database: Predicted and released over 200 million protein structures (2023) \- Nature paper (July 2021): >10,000 citations (in 2 years)

**Overwhelming Victory at CASP14** :
    
    
    CASP (Critical Assessment of Structure Prediction):
    International competition for protein structure prediction accuracy (biennial)
    
    Evaluation metric: GDT_TS (Global Distance Test - Total Score)
    - 0-100 score
    - 90+: Equivalent accuracy to experimental structures
    - Traditional methods (up to CASP13): Median 60-70
    
    AlphaFold 2 (CASP14, 2020):
    - Median GDT_TS: 92.4
    - 2/3 of 87 targets with GDT_TS > 90
    - 2nd place team (traditional methods): Median GDT_TS 75
    
    ⇒ Overwhelming victory over other methods
    

### 4.2.2 AlphaFold 2 Technology

**Architecture Overview** :

AlphaFold 2 integrates multiple deep learning technologies:
    
    
    Input: Amino acid sequence (e.g., MKTAYIAKQR...)
      ↓
    [1. MSA (Multiple Sequence Alignment) generation]
      - Search evolutionarily related sequences (UniProt, etc.)
      - Extract coevolution information
      ↓
    [2. Evoformer (attention mechanism-based network)]
      - Iteratively update MSA representation and residue pair representation
      - 48-layer Transformer blocks
      ↓
    [3. Structure Module]
      - Direct 3D coordinate prediction
      - Invariant Point Attention (rotation/translation-invariant attention)
      ↓
    [4. Refinement]
      - Energy minimization
      - Collision removal
      ↓
    Output: 3D structure (PDB format) + confidence score (pLDDT)
    

**Key Technical Innovations** :

  1. **Evoformer** : \- Simultaneous processing of MSA (sequence alignment) and pair representation \- Learning geometric relationships between residues

  2. **Invariant Point Attention (IPA)** : \- Attention mechanism invariant to 3D space rotation and translation \- Direct learning of geometric constraints

  3. **End-to-End Learning** : \- No dependence on template structures \- Direct 3D coordinate prediction from sequence

  4. **Recycling Mechanism** : \- Feedback prediction results to input (up to 3 times) \- Iteratively improve accuracy

**Training Data** : \- PDB (Protein Data Bank): ~170,000 structures \- Auxiliary data: UniProt (sequence database), BFD (Big Fantastic Database)

### 4.2.3 AlphaFold 2 Applications in Drug Discovery

**1\. Target Protein Structure Prediction**

Enables drug discovery for targets with previously unknown structures:
    
    
    # Structure prediction with AlphaFold 2 (using ColabFold)
    
    from colabfold import batch
    
    # Amino acid sequence (e.g., COVID-19 Spike protein RBD)
    sequence = """
    NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF
    """
    
    # Structure prediction
    batch.run(
        sequence=sequence,
        output_dir="./output",
        num_models=5,  # Predict with 5 models
        use_templates=False,  # No template usage
        use_amber=True  # Energy minimization
    )
    
    # Output: PDB file + confidence score (pLDDT)
    # pLDDT > 90: High confidence (equivalent to experimental structure)
    # pLDDT 70-90: Generally accurate (backbone reliable)
    # pLDDT 50-70: Low confidence (locally useful)
    # pLDDT < 50: Low reliability (possible disordered region)
    

**2\. Integration into Drug Design**

Example of structure-based drug discovery using AlphaFold structures:
    
    
    # Docking simulation using AlphaFold structure
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from openbabel import pybel
    import subprocess
    
    def alphafold_based_docking(target_sequence, ligand_smiles):
        """
        Docking using AlphaFold predicted structure
        """
    
        # Step 1: Predict target structure with AlphaFold2
        print("Step 1: Predicting target structure with AlphaFold2...")
        alphafold_structure = predict_structure_alphafold(target_sequence)
        # Output: "target.pdb" + pLDDT scores
    
        # Step 2: Predict binding pocket
        print("Step 2: Identifying binding pocket...")
        binding_pocket = predict_binding_site(alphafold_structure)
        # Methods:
        # - FPocket (geometric pocket detection)
        # - ConSurf (conservation analysis)
        # - AlphaFold pLDDT (prioritize high-confidence regions)
    
        # Step 3: Protein preparation
        print("Step 3: Preparing protein...")
        prepared_protein = prepare_protein(
            pdb_file="target.pdb",
            add_hydrogens=True,
            optimize_h=True,
            remove_waters=True
        )
    
        # Step 4: Ligand preparation
        print("Step 4: Preparing ligand...")
        mol = Chem.MolFromSmiles(ligand_smiles)
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol_3d)
    
        # Step 5: Docking (AutoDock Vina)
        print("Step 5: Docking...")
        docking_result = run_autodock_vina(
            receptor=prepared_protein,
            ligand=mol_3d,
            center=binding_pocket.center,  # Pocket center coordinates
            box_size=(20, 20, 20),  # Search range (Å)
            exhaustiveness=32  # Search precision
        )
    
        # Step 6: Result analysis
        print("Step 6: Analyzing results...")
        best_pose = docking_result.poses[0]
    
        results = {
            'binding_affinity': best_pose.affinity,  # kcal/mol
            'rmsd_lb': best_pose.rmsd_lb,
            'rmsd_ub': best_pose.rmsd_ub,
            'key_interactions': analyze_interactions(best_pose),
            'alphafold_confidence': get_pocket_confidence(alphafold_structure, binding_pocket)
        }
    
        return results
    
    # Usage example
    target_seq = "MKTAYIAKQRQISFVKSHFSRQ..."  # Novel target protein
    ligand = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
    
    result = alphafold_based_docking(target_seq, ligand)
    print(f"Binding Affinity: {result['binding_affinity']:.2f} kcal/mol")
    print(f"Pocket Confidence: {result['alphafold_confidence']:.1f}%")
    # Output example:
    # Binding Affinity: -7.8 kcal/mol (good binding affinity)
    # Pocket Confidence: 92.3% (high confidence)
    

**3\. Real Example: Malaria Therapeutic Development (2023)**

Research by University of Oxford and DNDi (Drugs for Neglected Diseases initiative):
    
    
    Challenge: Unknown structure of essential malaria parasite enzyme PfCLK3
              → Experimental structure determination difficult (crystallization failed)
    
    Solution: Structure prediction with AlphaFold 2
             → pLDDT 87.3 (high confidence)
             → Active site structure clearly revealed
    
    Drug Discovery Process:
    1. Virtual screening with AlphaFold structure
       - Compound library: 5 million types
       - Docking simulation
       - Selected top 500 compounds
    
    2. Experimental validation
       - In vitro enzyme inhibition assays
       - 50 compounds showed activity (10% hit rate, 2× traditional rate)
    
    3. Lead optimization
       - Derivative synthesis guided by AlphaFold structure
       - Obtained multiple compounds with IC50 < 100 nM
    
    4. Preclinical testing
       - Efficacy confirmed in malaria-infected mice
       → Clinical trial start planned for 2024
    
    Comparison with traditional methods:
    - Structure determination period: Years → Hours (AlphaFold)
    - Hit rate: 5% → 10% (2× improvement)
    - Development period: 5-7 years → 2-3 years (estimated)
    

### 4.2.4 AlphaFold 2 Limitations and Challenges

**Technical Limitations** :

  1. **Difficulty Predicting Dynamic Structures** : \- AlphaFold predicts static structures \- Cannot predict protein movements (conformational changes) \- Solution: Combination with molecular dynamics (MD) simulations

  2. **Ligand-Bound State Prediction** : \- Good at apo (ligand-free) structure prediction \- Inaccurate for holo (post-ligand binding) structural changes \- Solution: AlphaFold-Multimer (complex prediction) + docking

  3. **Low Confidence Regions** : \- Intrinsically Disordered Regions (IDRs) \- Flexible loop regions \- → These regions may be unsuitable as drug targets

**Drug Discovery Application Challenges** :

  1. **Binding Affinity Prediction Accuracy** : \- Docking scores don't necessarily correlate with actual binding affinity \- Solution: Always perform experimental validation, correct with machine learning

  2. **Novel Pocket Discovery** : \- AlphaFold learns known structural patterns \- Weak at predicting completely new folds \- Solution: Combined use with experimental structure analysis

**Future Developments** :

  * **AlphaFold 3** (expected 2024): Improvements in complex prediction, dynamic structures, ligand binding
  * **RoseTTAFold Diffusion** (Baker Lab): Diffusion model-based structure prediction
  * **ESMFold** (Meta AI): Language model-based, 60× faster than AlphaFold

* * *

## 4.3 Molecular Generative AI: Key to Next-Generation Drug Discovery

Traditional drug discovery focused on "exploring and optimizing existing compounds," but generative AI enables "creating entirely new molecules."

### 4.3.1 Molecular Generative AI Overview

**Goal** : Automatic design of novel molecules with desired properties

**Approaches** : 1\. **VAE (Variational Autoencoder)** : Encode molecules into latent space, generate via decoding 2\. **GAN (Generative Adversarial Network)** : Adversarial learning between generator and discriminator 3\. **Transformer/RNN** : Generate SMILES strings as language 4\. **Graph Generation Models** : Directly generate molecular graphs 5\. **Reinforcement Learning** : Maximize reward function (desired properties)

### 4.3.2 VAE-Based Molecular Generation

**Principle** : Map molecules to continuous latent space
    
    
    Encoder: Molecule → Latent vector (low-dimensional representation)
    Decoder: Latent vector → Molecule
    
    Latent space properties:
    - Similar molecules mapped to nearby positions
    - Interpolation in latent space generates intermediate molecules
    - Random sampling generates novel molecules
    

**Implementation Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    class MolecularVAE(nn.Module):
        """Molecular generation VAE (SMILES string-based)"""
    
        def __init__(self, vocab_size, latent_dim=128, max_len=120):
            super().__init__()
            self.latent_dim = latent_dim
            self.max_len = max_len
    
            # Encoder (SMILES → latent vector)
            self.encoder = nn.LSTM(
                input_size=vocab_size,
                hidden_size=256,
                num_layers=2,
                batch_first=True
            )
            self.fc_mu = nn.Linear(256, latent_dim)
            self.fc_logvar = nn.Linear(256, latent_dim)
    
            # Decoder (latent vector → SMILES)
            self.decoder_input = nn.Linear(latent_dim, 256)
            self.decoder = nn.LSTM(
                input_size=vocab_size,
                hidden_size=256,
                num_layers=2,
                batch_first=True
            )
            self.output_layer = nn.Linear(256, vocab_size)
    
        def encode(self, x):
            """SMILES → latent vector"""
            _, (h_n, _) = self.encoder(x)
            h = h_n[-1]  # Last hidden state
    
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
    
            return mu, logvar
    
        def reparameterize(self, mu, logvar):
            """Reparameterization trick"""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z
    
        def decode(self, z):
            """Latent vector → SMILES"""
            h = self.decoder_input(z).unsqueeze(0)
    
            # Autoregressively generate characters
            outputs = []
            input_char = torch.zeros(z.size(0), 1, vocab_size).to(z.device)
    
            for t in range(self.max_len):
                output, (h, _) = self.decoder(input_char, (h, None))
                output = self.output_layer(output)
                outputs.append(output)
    
                # Next input is current output
                input_char = torch.softmax(output, dim=-1)
    
            return torch.cat(outputs, dim=1)
    
        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon_x = self.decode(z)
            return recon_x, mu, logvar
    
        def generate(self, num_samples=10):
            """Randomly generate novel molecules"""
            with torch.no_grad():
                # Sample from normal distribution
                z = torch.randn(num_samples, self.latent_dim)
    
                # Decode
                smiles_logits = self.decode(z)
    
                # Convert to strings
                smiles_list = self.logits_to_smiles(smiles_logits)
    
                return smiles_list
    
        def interpolate(self, smiles1, smiles2, steps=10):
            """Interpolate between two molecules"""
            with torch.no_grad():
                # Encode
                z1, _ = self.encode(self.smiles_to_tensor(smiles1))
                z2, _ = self.encode(self.smiles_to_tensor(smiles2))
    
                # Linear interpolation
                interpolated_mols = []
                for alpha in torch.linspace(0, 1, steps):
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    smiles_interp = self.decode(z_interp)
                    interpolated_mols.append(self.logits_to_smiles(smiles_interp))
    
                return interpolated_mols
    
    # Loss function
    def vae_loss(recon_x, x, mu, logvar):
        """VAE loss = reconstruction error + KL divergence"""
        # Reconstruction error (cross-entropy)
        recon_loss = nn.CrossEntropyLoss()(
            recon_x.view(-1, vocab_size),
            x.view(-1)
        )
    
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        return recon_loss + kl_loss
    
    # Usage example
    model = MolecularVAE(vocab_size=50, latent_dim=128)
    
    # Training
    # ... (omitted)
    
    # Generate novel molecules
    new_molecules = model.generate(num_samples=100)
    print("Generated molecules (SMILES):")
    for i, smiles in enumerate(new_molecules[:5]):
        print(f"{i+1}. {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            print(f"   Valid molecule: Yes, MW={Chem.Descriptors.MolWt(mol):.1f}")
        else:
            print(f"   Valid molecule: No (invalid SMILES)")
    
    # Molecular interpolation
    mol_A = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
    mol_B = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    
    interpolated = model.interpolate(mol_A, mol_B, steps=10)
    print(f"\nInterpolated molecules between ibuprofen and aspirin:")
    for i, smiles in enumerate(interpolated):
        print(f"Step {i}: {smiles}")
    

**Output Example** :
    
    
    Generated molecules (SMILES):
    1. CC1=CC(=O)C=CC1=O
       Valid molecule: Yes, MW=124.1
    2. C1CCC(CC1)N2C=CN=C2
       Valid molecule: Yes, MW=164.2
    3. CC(C)NCC(O)COc1ccccc1
       Valid molecule: Yes, MW=209.3
    4. CCOC(=O)C1=CN(C=C1)C
       Valid molecule: No (invalid SMILES)
    5. O=C1NC(=O)C(=C1)C(=O)O
       Valid molecule: Yes, MW=157.1
    
    Interpolated molecules between ibuprofen and aspirin:
    Step 0: CC(C)Cc1ccc(cc1)C(C)C(O)=O
    Step 1: CC(C)Cc1ccc(cc1)C(=O)C(O)=O
    Step 2: CC(C)Cc1ccc(cc1)C(=O)O
    ...
    

**VAE Advantages and Challenges** : \- ✅ Exploration possible in continuous latent space \- ✅ Stepwise molecular transformation possible through interpolation \- ❌ Low chemical validity of generated molecules (30-50% invalid SMILES) \- ❌ Difficult to control specific properties

### 4.3.3 GAN-Based Molecular Generation

**Principle** : Adversarial learning between Generator and Discriminator
    
    
    Generator: Noise → Fake molecules
    Discriminator: Molecule → Real/fake judgment
    
    Learning process:
    1. Generator creates fake molecules
    2. Discriminator distinguishes real (training data) from fake
    3. Generator learns to fool discriminator
    4. Discriminator learns to detect fakes
    → Through repetition, generator creates increasingly realistic molecules
    

**Implementation Example (MolGAN)** :
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class MolGAN(nn.Module):
        """Molecular generation GAN (graph-based)"""
    
        def __init__(self, latent_dim=128, num_atom_types=9, max_atoms=38):
            super().__init__()
            self.latent_dim = latent_dim
            self.num_atom_types = num_atom_types
            self.max_atoms = max_atoms
    
            # Generator
            self.generator = Generator(latent_dim, num_atom_types, max_atoms)
    
            # Discriminator
            self.discriminator = Discriminator(num_atom_types, max_atoms)
    
            # Reward network (property prediction)
            self.reward_network = PropertyPredictor(num_atom_types, max_atoms)
    
    class Generator(nn.Module):
        """Generate molecular graphs from noise"""
    
        def __init__(self, latent_dim, num_atom_types, max_atoms):
            super().__init__()
    
            # Noise → graph features
            self.fc_layers = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU()
            )
    
            # Graph features → node features (atom types)
            self.node_layer = nn.Linear(512, max_atoms * num_atom_types)
    
            # Graph features → adjacency matrix (bonds)
            self.edge_layer = nn.Linear(512, max_atoms * max_atoms)
    
        def forward(self, z):
            """
            z: (batch, latent_dim) noise vector
    
            Output:
            - nodes: (batch, max_atoms, num_atom_types) atom types (one-hot)
            - edges: (batch, max_atoms, max_atoms) adjacency matrix
            """
            h = self.fc_layers(z)
    
            # Node generation
            nodes_logits = self.node_layer(h)
            nodes_logits = nodes_logits.view(-1, self.max_atoms, self.num_atom_types)
            nodes = torch.softmax(nodes_logits, dim=-1)
    
            # Edge generation
            edges_logits = self.edge_layer(h)
            edges_logits = edges_logits.view(-1, self.max_atoms, self.max_atoms)
            edges = torch.sigmoid(edges_logits)
    
            # Symmetrize (undirected graph)
            edges = (edges + edges.transpose(1, 2)) / 2
    
            return nodes, edges
    
    class Discriminator(nn.Module):
        """Judge whether molecular graph is real or fake"""
    
        def __init__(self, num_atom_types, max_atoms):
            super().__init__()
    
            # Graph Convolutional Layers
            self.gcn1 = GraphConvLayer(num_atom_types, 128)
            self.gcn2 = GraphConvLayer(128, 256)
    
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
    
        def forward(self, nodes, edges):
            """
            nodes: (batch, max_atoms, num_atom_types)
            edges: (batch, max_atoms, max_atoms)
    
            Output: (batch, 1) realness score (0-1)
            """
            # GCN layers
            h = self.gcn1(nodes, edges)
            h = torch.relu(h)
            h = self.gcn2(h, edges)
            h = torch.relu(h)
    
            # Global pooling (whole graph features)
            h_graph = torch.mean(h, dim=1)  # (batch, 256)
    
            # Classification
            score = self.classifier(h_graph)
    
            return score
    
    class PropertyPredictor(nn.Module):
        """Predict molecular properties (for reward calculation)"""
    
        def __init__(self, num_atom_types, max_atoms):
            super().__init__()
    
            self.gcn1 = GraphConvLayer(num_atom_types, 128)
            self.gcn2 = GraphConvLayer(128, 256)
    
            # Property prediction head
            self.property_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)  # e.g., logP value prediction
            )
    
        def forward(self, nodes, edges):
            h = self.gcn1(nodes, edges)
            h = torch.relu(h)
            h = self.gcn2(h, edges)
            h = torch.relu(h)
    
            h_graph = torch.mean(h, dim=1)
            property_value = self.property_head(h_graph)
    
            return property_value
    
    # Loss function
    def gan_loss(real_molecules, generator, discriminator):
        """GAN loss function"""
        batch_size = real_molecules[0].size(0)
    
        # Discriminate real molecules
        real_nodes, real_edges = real_molecules
        real_score = discriminator(real_nodes, real_edges)
    
        # Generate fake molecules
        z = torch.randn(batch_size, generator.latent_dim)
        fake_nodes, fake_edges = generator(z)
        fake_score = discriminator(fake_nodes, fake_edges)
    
        # Discriminator loss (classify real as 1, fake as 0)
        d_loss_real = nn.BCELoss()(real_score, torch.ones_like(real_score))
        d_loss_fake = nn.BCELoss()(fake_score, torch.zeros_like(fake_score))
        d_loss = d_loss_real + d_loss_fake
    
        # Generator loss (fool discriminator)
        g_loss = nn.BCELoss()(fake_score, torch.ones_like(fake_score))
    
        return g_loss, d_loss
    
    # Usage example
    model = MolGAN(latent_dim=128)
    
    # Generate novel molecules
    z = torch.randn(10, 128)  # Generate 10 molecules
    nodes, edges = model.generator(z)
    
    # Convert graph to SMILES (separate implementation needed)
    smiles_list = graph_to_smiles(nodes, edges)
    print("Generated molecules:")
    for smiles in smiles_list:
        print(smiles)
    

**GAN Advantages and Challenges** : \- ✅ Generates valid molecules similar to training data \- ✅ Property control possible with reward network \- ❌ Unstable training (mode collapse problem) \- ❌ Low diversity (tends to generate similar molecules)

### 4.3.4 Transformer-Based Molecular Generation

**Principle** : Treat SMILES strings as natural language and generate with Transformer

**Implementation Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class MolecularTransformer(nn.Module):
        """Transformer-based molecular generation model"""
    
        def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, max_len=150):
            super().__init__()
    
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = PositionalEncoding(d_model, max_len)
    
            # Transformer Decoder (autoregressive generation)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2048,
                dropout=0.1
            )
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
    
            self.output_layer = nn.Linear(d_model, vocab_size)
    
        def forward(self, tgt, memory):
            """
            tgt: (seq_len, batch) target sequence
            memory: (1, batch, d_model) condition (optional)
            """
            # Embedding + Positional Encoding
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoding(tgt_emb)
    
            # Transformer Decoder
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0))
            output = self.transformer(tgt_emb, memory, tgt_mask=tgt_mask)
    
            # Output (probability distribution over vocabulary)
            logits = self.output_layer(output)
    
            return logits
    
        def generate(self, start_token, max_len=100, temperature=1.0):
            """Autoregressively generate molecules"""
            self.eval()
            with torch.no_grad():
                # Initial token
                generated = [start_token]
    
                for _ in range(max_len):
                    # Encode current sequence
                    tgt = torch.LongTensor(generated).unsqueeze(1)
    
                    # Predict next token
                    logits = self.forward(tgt, memory=None)
                    next_token_logits = logits[-1, 0, :] / temperature
    
                    # Sampling
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
    
                    # End condition
                    if next_token == END_TOKEN:
                        break
    
                    generated.append(next_token)
    
                # Convert token sequence to SMILES
                smiles = tokens_to_smiles(generated)
                return smiles
    
    # Conditional generation (specify properties)
    class ConditionalMolecularTransformer(MolecularTransformer):
        """Conditional molecular generation (specify desired properties)"""
    
        def __init__(self, vocab_size, num_properties=5, **kwargs):
            super().__init__(vocab_size, **kwargs)
    
            # Network to embed properties
            self.property_encoder = nn.Sequential(
                nn.Linear(num_properties, 256),
                nn.ReLU(),
                nn.Linear(256, self.d_model)
            )
    
        def generate_with_properties(self, target_properties, max_len=100):
            """
            target_properties: (num_properties,) desired property values
            Example: [logP=2.5, MW=350, TPSA=60, HBD=2, HBA=4]
            """
            # Encode properties
            property_emb = self.property_encoder(target_properties)
            memory = property_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
    
            # Generate
            self.eval()
            with torch.no_grad():
                generated = [START_TOKEN]
    
                for _ in range(max_len):
                    tgt = torch.LongTensor(generated).unsqueeze(1)
                    logits = self.forward(tgt, memory=memory)
    
                    next_token_logits = logits[-1, 0, :]
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
    
                    if next_token == END_TOKEN:
                        break
    
                    generated.append(next_token)
    
                smiles = tokens_to_smiles(generated)
                return smiles
    
    # Usage example
    model = ConditionalMolecularTransformer(vocab_size=50, num_properties=5)
    
    # Specify desired properties
    target_props = torch.tensor([
        2.5,   # logP (lipophilicity)
        350.0, # Molecular weight
        60.0,  # TPSA (topological polar surface area)
        2.0,   # Hydrogen bond donor count
        4.0    # Hydrogen bond acceptor count
    ])
    
    # Conditional generation
    new_molecule = model.generate_with_properties(target_props)
    print(f"Generated molecule: {new_molecule}")
    
    # Verify actual properties
    mol = Chem.MolFromSmiles(new_molecule)
    if mol:
        actual_logP = Descriptors.MolLogP(mol)
        actual_MW = Descriptors.MolWt(mol)
        actual_TPSA = Descriptors.TPSA(mol)
    
        print(f"Actual properties:")
        print(f"  logP: {actual_logP:.2f} (target: 2.5)")
        print(f"  MW: {actual_MW:.1f} (target: 350.0)")
        print(f"  TPSA: {actual_TPSA:.1f} (target: 60.0)")
    

**Transformer Advantages** : \- ✅ Stable learning even with long sequences \- ✅ Easy conditional generation (property specification) \- ✅ High chemical validity (70-90% valid SMILES) \- ✅ Can apply latest large language model technologies

### 4.3.5 Real Example: Novel Antibiotic Discovery (MIT, 2020)

MIT research team's discovery of new antibiotic "Halicin" using deep learning:
    
    
    Challenge: Increasing drug-resistant bacteria
              → Need new antibiotics, but development is difficult
    
    Approach:
    1. Data collection
       - Drug Repurposing Hub (~6,000 compounds)
       - Antibacterial activity data against E. coli
    
    2. Model building
       - Graph Neural Network (GNN)
       - Molecular graph → antibacterial activity prediction
    
    3. Virtual screening
       - Screened ZINC15 database (~170 million compounds)
       - Selected top 5,000 compounds
    
    4. Experimental validation
       - In vitro antibacterial assays
       - Discovered Halicin: existing drug (diabetes drug candidate) but
         antibacterial activity was unknown
    
    5. Halicin properties
       - Effective against wide range of resistant bacteria (Acinetobacter baumannii, Clostridioides difficile, etc.)
       - Different mechanism of action from existing antibiotics (disrupts cell membrane electrochemical gradient)
       - Resistant to resistance development
    
    6. Preclinical testing
       - Efficacy confirmed in mouse infection models
       - Further development ongoing since 2021
    
    Impact:
    - First novel antibiotic discovered by AI
    - Discovering new applications for existing compounds (drug repositioning)
    - Dramatically shortened development time (traditional 10-15 years → 2-3 years possible)
    

* * *

## 4.4 Best Practices and Pitfalls

### 4.4.1 Seven Principles for Success

**1\. Prioritize Data Quality**
    
    
    Good data > Advanced models
    
    Checklist:
    □ Is data source reliable? (papers, public DBs, internal experiments)
    □ Is bias absent? (measurement method bias, publication bias)
    □ Is missing value handling appropriate? (deletion vs. imputation)
    □ Have outliers been checked? (experimental error vs. true outlier)
    □ Is there data leakage? (test data information mixed into training)
    
    Real example: ChEMBL data quality control
    - Duplicate compound removal: InChI key identity confirmation
    - Activity value standardization: IC50, EC50, Ki → unified to pIC50
    - Confidence filtering: Only assay confidence score > 8 used
    - Outlier removal: IQR method detects statistical outliers
    

**2\. Start with Simple Baselines**
    
    
    Development order:
    1. Random Forest (interpretable, easy implementation)
    2. Gradient Boosting (XGBoost, LightGBM)
    3. Neural Networks (only when necessary)
    4. GNN, Transformer (when sufficient data available)
    
    Reasoning:
    - Simple models often achieve 80% performance
    - Complex models difficult to interpret, difficult to debug
    - Higher overfitting risk
    

**3\. Actively Utilize Domain Knowledge**
    
    
    AI + Chemists > AI alone
    
    Utilization examples:
    - Feature design: Select chemically meaningful descriptors
    - Model validation: Validate prediction results with chemical knowledge
    - Failure analysis: Chemically interpret why predictions failed
    - Constraint setting: Practical constraints like synthesizability, patent avoidance
    
    Case study: Exscientia's "Centaur Chemist"
    - AI proposes candidate molecules
    - Human chemists validate chemical validity
    - Feedback returned to AI
    → Accuracy improves through mutual learning
    

**4\. Always Incorporate Experimental Validation**
    
    
    Computational prediction ≠ experimental fact
    
    Active Learning cycle:
    1. AI predicts candidate compounds
    2. Experimentally validate top N (N=10-50)
    3. Add experimental results to data
    4. Retrain model
    5. Return to step 1
    
    Advantages:
    - Data-efficient (optimization with few experiments)
    - Model adapts to reality
    - Prediction accuracy improves iteratively
    
    Real example: Recursion Pharmaceuticals
    - Automated 2.2 million wells/week experiments
    - Immediately reflect data in model
    - Tight integration of experiments and AI
    

**5\. Emphasize Interpretability**
    
    
    Black-box model problems:
    - Unknown prediction basis → chemists don't trust
    - Difficult failure cause analysis
    - Difficult to explain to regulatory authorities
    
    Solutions:
    □ Visualize feature importance with SHAP values
    □ Visualize important substructures with Attention mechanism
    □ Extract simple rules with decision trees
    □ Use chemically interpretable descriptors
    
    Example: Which substructures contribute to activity?
    → Visualize with Attention mechanism
    → Experts confirm pharmacological validity
    

**6\. Avoid Overfitting**
    
    
    Common overfitting signs:
    - Training data accuracy 95%, test data accuracy 60% ← obvious overfitting
    - Overly complex model (parameters >> data points)
    - Large variance in cross-validation results
    
    Countermeasures:
    □ Data augmentation (SMILES Enumeration, conformer sampling)
    □ Regularization (L1/L2, Dropout)
    □ Early Stopping
    □ Cross-validation (5-fold or more)
    □ Final evaluation on external test set
    

**7\. Continuous Model Updates**
    
    
    Models are "living things":
    - Continuously update with new data
    - Drift detection (input distribution changes)
    - Regular performance evaluation
    
    Update strategy:
    - Monthly/quarterly retraining
    - Add new experimental data
    - A/B test comparing old and new models
    - Performance monitoring
    

### 4.4.2 Common Failure Patterns

**Failure 1: Data Leakage**
    
    
    Problem: Test data information mixed into training
    
    Examples:
    - Duplicate compounds (isomers, etc.) scattered in training and test
    - Using future information in time series data
    - Performing preprocessing (standardization) on all data before split
    
    Countermeasures:
    1. Split data first
    2. Only fit preprocessing on training data, transform test data only
    3. Split by compound scaffold (structurally different molecules in test)
    
    Correct implementation example:
    # ❌ Wrong
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Fit on all data
    X_train, X_test = train_test_split(X_scaled)
    
    # ✅ Correct
    X_train, X_test = train_test_split(X)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
    X_test_scaled = scaler.transform(X_test)  # Only transform test data
    

**Failure 2: Inappropriate Evaluation Metrics**
    
    
    Problem: Using metrics unsuitable for task
    
    Example:
    - Using Accuracy on imbalanced data
      → Predicting "all negative" on 99% negative data gives 99% accuracy but meaningless
    
    Countermeasures:
    □ Classification tasks: ROC-AUC, PR-AUC, F1 score
    □ Regression tasks: RMSE, MAE, R²
    □ Imbalanced data: Balanced Accuracy, MCC (Matthews correlation coefficient)
    □ Ranking: Hit Rate @ K, Enrichment Factor
    
    Recommended metrics for drug discovery:
    - Virtual screening: Enrichment Factor @ 1%
      (what % of active compounds in top 1%)
    - QSAR: R² (coefficient of determination), RMSE
    - Classification (active/inactive): ROC-AUC, Balanced Accuracy
    

**Failure 3: Extrapolation Outside Applicability Domain**
    
    
    Problem: Low prediction accuracy for compounds outside training data distribution
    
    Example:
    - Training data: MW 200-500 compounds
    - Prediction target: MW 800 compound
    → Prediction unreliable
    
    Countermeasures:
    □ Define Applicability Domain
    □ Calculate similarity to training data
    □ Implement extrapolation warning system
    
    Implementation example:
    def check_applicability_domain(query_mol, training_mols, threshold=0.3):
        """
        Check if query molecule is within applicability domain of training data
        """
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, 2048)
    
        # Maximum similarity to training data
        max_similarity = 0
        for train_mol in training_mols:
            train_fp = AllChem.GetMorganFingerprintAsBitVect(train_mol, 2, 2048)
            similarity = DataStructs.TanimotoSimilarity(query_fp, train_fp)
            max_similarity = max(max_similarity, similarity)
    
        if max_similarity < threshold:
            print(f"Warning: Query molecule significantly differs from training data")
            print(f"Maximum similarity: {max_similarity:.3f} (threshold: {threshold})")
            print(f"Prediction may be unreliable")
            return False
    
        return True
    

**Failure 4: Ignoring Synthesizability**
    
    
    Problem: Highly active in prediction but actually unsynthesizable molecules
    
    Examples:
    - Theoretically optimal but no synthetic route exists
    - Synthesis requires 100+ steps (impractical)
    - Unstable chemical structure (decomposes immediately)
    
    Countermeasures:
    □ Integrate synthesizability scores (SAScore, SCScore)
    □ Use retrosynthesis analysis tools (RDKit, AiZynthFinder)
    □ Review with chemists
    □ Molecular generation using only known reactions
    
    Implementation example:
    from rdkit.Chem import RDConfig
    import os
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    
    def filter_synthesizable(molecules, sa_threshold=3.0):
        """
        Filter by synthesizability
        SA Score: 1 (easy) ~ 10 (difficult)
        """
        synthesizable = []
    
        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
    
            sa_score = sascorer.calculateScore(mol)
    
            if sa_score <= sa_threshold:
                synthesizable.append({
                    'smiles': smiles,
                    'sa_score': sa_score
                })
            else:
                print(f"Difficult to synthesize: {smiles}, SA={sa_score:.2f}")
    
        return synthesizable
    
    # Usage example
    generated_mols = ["CC(C)Cc1ccc(cc1)C(C)C(O)=O", ...]
    synthesizable_mols = filter_synthesizable(generated_mols, sa_threshold=3.5)
    print(f"Synthesizable molecules: {len(synthesizable_mols)}/{len(generated_mols)}")
    

* * *

## 4.5 Career Paths and Industry Trends

### 4.5.1 Career Options in AI Drug Discovery

AI drug discovery is an interdisciplinary field where people with diverse backgrounds thrive.

**1\. Machine Learning Engineer / Data Scientist**

**Role** : \- AI model development and optimization \- Data pipeline construction \- Model production environment deployment

**Required Skills** : \- Python, PyTorch/TensorFlow \- Machine learning algorithms (deep learning, GNN, Transformer) \- Cloud environments (AWS, GCP, Azure) \- MLOps (model version control, A/B testing)

**Recommended Background** : \- Computer Science \- Statistics \- Mathematics

**Career Path Example** :
    
    
    Junior ML Engineer
        ↓ (2-3 years)
    Senior ML Engineer
        ↓ (3-5 years)
    Lead ML Engineer / ML Architect
        ↓
    VP of AI / Chief Data Scientist
    

**Salary Range (US)** : \- Junior: $100k-150k \- Senior: $150k-250k \- Lead/Principal: $250k-400k \- VP/Chief: $400k-700k+

* * *

**2\. Cheminformatician**

**Role** : \- Chemical data processing and analysis \- Molecular descriptor design \- QSAR model construction \- Virtual screening

**Required Skills** : \- Organic chemistry knowledge \- RDKit, ChEMBL, PubChem \- Statistics and machine learning \- Python, R

**Recommended Background** : \- Chemistry (organic chemistry, pharmaceutical sciences) \- Biochemistry \- Computational chemistry

**Career Path Example** :
    
    
    Cheminformatics Scientist
        ↓
    Senior Cheminformatics Scientist
        ↓
    Principal Scientist / Director of Cheminformatics
    

**Salary Range (US)** : \- Scientist: $80k-120k \- Senior: $120k-180k \- Principal/Director: $180k-300k

* * *

**3\. Computational Chemist**

**Role** : \- Molecular dynamics simulations \- Quantum chemistry calculations \- Docking simulations \- Structure-based drug design

**Required Skills** : \- Quantum chemistry (DFT, semi-empirical methods) \- Molecular dynamics (GROMACS, AMBER, NAMD) \- Docking tools (AutoDock, Glide, GOLD) \- Python, C++, Fortran

**Recommended Background** : \- Theoretical chemistry \- Physical chemistry \- Computational science

**Salary Range (US)** : \- Computational Chemist: $90k-140k \- Senior: $140k-200k \- Principal: $200k-300k

* * *

**4\. Bioinformatician**

**Role** : \- Omics data analysis (genomics, transcriptomics, proteomics) \- Target Identification \- Biomarker discovery \- Systems biology

**Required Skills** : \- Molecular biology knowledge \- Statistical analysis (R, Bioconductor) \- NGS data analysis \- Machine learning

**Recommended Background** : \- Biology \- Biochemistry \- Genetics

**Salary Range (US)** : \- Bioinformatician: $80k-130k \- Senior: $130k-190k \- Principal: $190k-280k

* * *

**5\. Research Scientist**

**Role** : \- Research and development of new AI methods \- Paper writing and conference presentations \- Investigation and implementation of cutting-edge technologies

**Required Skills** : \- Deep expertise (PhD typically required) \- Paper writing ability \- Research track record (peer-reviewed papers) \- Presentation skills

**Recommended Background** : \- PhD (Computer Science, Chemistry, Biology, etc.) \- Postdoc experience

**Career Path Example** :
    
    
    Postdoctoral Researcher
        ↓
    Research Scientist
        ↓
    Senior Research Scientist
        ↓
    Principal Research Scientist / Research Director
        ↓
    VP of Research / Chief Scientific Officer
    

**Salary Range (US)** : \- Research Scientist: $120k-180k \- Senior: $180k-260k \- Principal: $260k-400k \- VP/CSO: $400k-800k+

* * *

### 4.5.2 Skill Development Roadmap

**Level 1: Fundamentals (0-6 months)**
    
    
    □ Python programming
      - Book: 'Python for Data Analysis' (Wes McKinney)
      - Online: Coursera 'Python for Everybody'
    
    □ Machine learning fundamentals
      - Book: 'Deep Learning from Scratch' (Koki Saitoh)
      - Online: Andrew Ng 'Machine Learning' (Coursera)
    
    □ Chemistry basics
      - Book: 'Organic Chemistry' (Vollhardt & Schore)
      - Online: Khan Academy Organic Chemistry
    
    □ Data analysis tools
      - pandas, NumPy, matplotlib
      - Jupyter Notebook
    

**Level 2: Practice (6-18 months)**
    
    
    □ Cheminformatics
      - RDKit tutorial (official documentation)
      - 'Chemoinformatics for Drug Discovery' (book)
    
    □ Deep learning
      - 'Deep Learning' (Ian Goodfellow)
      - PyTorch/TensorFlow tutorials
    
    □ Drug discovery practice
      - Participate in Kaggle competitions (e.g., QSAR tasks)
      - Build QSAR models with ChEMBL data
      - Paper implementations (code published on GitHub)
    
    □ Biology fundamentals
      - 'Molecular Biology of the Cell' (Alberts et al.)
      - Understanding drug discovery processes
    

**Level 3: Specialization (18 months+)**
    
    
    □ Latest technology acquisition
      - Graph Neural Networks (GNN)
      - Transformer for molecules
      - AlphaFold 2 understanding and application
    
    □ Research and development
      - Execute independent projects
      - Submit papers (arXiv, peer-reviewed journals)
      - Publish code on GitHub
    
    □ Domain expertise
      - Pharmacology, toxicology
      - ADMET prediction expertise
      - Structure-based drug design
    
    □ Business skills
      - Project management
      - Cross-functional collaboration
      - Presentation skills
    

### 4.5.3 Industry Trends and Future Outlook

**Rapid Market Growth** :
    
    
    AI drug discovery market size (global):
    - 2020: ~$700 million
    - 2025: ~$4 billion (forecast)
    - 2030: ~$15 billion (forecast)
    
    CAGR (compound annual growth rate): ~40%
    
    Investment:
    - 2021: ~$14 billion invested in AI drug discovery startups
    - 2022: ~$9 billion (market adjustment impact)
    - 2023: Recovery trend
    
    Major investors:
    - Venture capital (Andreessen Horowitz, Flagship Pioneering)
    - Pharmaceutical majors (Pfizer, Roche, AstraZeneca)
    - Tech giants (Google, Microsoft, NVIDIA)
    

**Technology Trends** :

**1\. Generative AI** \- Drug discovery applications of large language models like ChatGPT \- Improved molecular generation accuracy \- Protein design (RFdiffusion, ProteinMPNN)

**2\. Multimodal Learning** \- Integrated learning of structure, sequence, image, and text \- Fusion with knowledge graphs \- Multi-omics data integration

**3\. Lab Automation** \- Integration of robotics and AI (Recursion, Zymergen) \- Automated experimental design \- Closed-loop optimization

**4\. Quantum Computing** \- Acceleration of molecular simulations \- Quantum Machine Learning (QML) \- Still early stage, but revolutionary potential in future

**Industry Challenges** :

**1\. Regulatory Lag** \- FDA/EMA developing AI drug discovery guidelines \- Explainable AI (XAI) requirements \- Validation standardization

**2\. Talent Shortage** \- Few people proficient in both AI and drug discovery \- Need for interdisciplinary education programs \- High salary levels (talent acquisition competition)

**3\. Clinical Trial Validation** \- Few clinical success examples of AI-designed drugs yet \- Need to demonstrate long-term efficacy and safety \- 2025-2030 is critical period

**Japan's Situation** :
    
    
    Strengths:
    - Presence of pharmaceutical majors (Takeda, Astellas, Daiichi Sankyo, etc.)
    - High-quality clinical data
    - Robotics technology
    
    Challenges:
    - AI talent shortage
    - Immature startup ecosystem
    - Conservative drug discovery culture
    
    Major players:
    - Preferred Networks (deep learning drug discovery platform)
    - MOLCURE (AI drug discovery)
    - ExaWizards (AI × Healthcare)
    - University spinoffs (University of Tokyo, Kyoto University, etc.)
    
    Government initiatives:
    - Moonshot R&D (AI drug discovery acceleration)
    - AMED (Japan Agency for Medical Research and Development) support
    - Industry-academia collaboration projects
    

**Future Forecast (2030)** :

  1. **Increased AI-Designed Drug Approvals** \- 10-20 AI-designed drugs expected to be approved by 2030 \- Development period: 10-15 years → shortened to 5-7 years \- Development cost: ~$2.6 billion → reduced to < $1 billion

  2. **Fully Automated Drug Discovery Labs** \- AI generates hypotheses, robots conduct experiments, automatic feedback \- Humans focus on strategic decisions and oversight

  3. **Accelerated Personalized Medicine** \- Drug discovery based on individual genome/omics data \- Personalized treatment becomes realistic with AI

  4. **Democratization of Drug Discovery Platforms** \- Cloud-based AI drug discovery tools \- Accessible to SMEs and academia \- Progress in open-sourcing

* * *

## Summary

This chapter explored real-world applications of AI drug discovery from multiple perspectives:

### What We Learned

  1. **Diversity of Company Strategies** : \- Exscientia: Active Learning and human-AI collaboration \- Insilico Medicine: Generative AI and integrated platform \- Recursion: Large-scale data generation and phenomics \- BenevolentAI: Knowledge graphs and NLP

  2. **Revolutionary Technologies** : \- AlphaFold 2: Structure prediction revolution \- Molecular generative AI: VAE, GAN, Transformer \- Multimodal learning: Multiple data type integration

  3. **Practical Best Practices** : \- Data quality is paramount \- Start with simple models \- Utilize domain knowledge \- Incorporate experimental validation \- Continuous model updates

  4. **Common Pitfalls** : \- Data leakage \- Inappropriate evaluation metrics \- Extrapolation outside applicability domain \- Ignoring synthesizability

  5. **Careers and Industry Trends** : \- Diverse roles (ML Engineer, Cheminformatician, Computational Chemist) \- High salary levels and talent demand \- Rapidly growing market (40% CAGR) \- 2025-2030 is critical period for clinical validation

### Next Steps

AI drug discovery is a rapidly evolving field. Continuous learning and practice are essential:

  1. **Technology Acquisition** : \- Implement hands-on code from Chapter 3 \- Participate in competitions like Kaggle \- Follow latest papers (arXiv, PubMed)

  2. **Community Participation** : \- Contribute to open source projects on GitHub \- Attend conferences (ICML, NeurIPS, ISMB) \- Study groups and hackathons

  3. **Career Building** : \- Internships (AI drug discovery companies) \- Graduate school (interdisciplinary programs) \- Execute and publish personal projects

AI drug discovery is a rewarding field that can contribute to human health. Use the knowledge learned in this series to challenge next-generation drug discovery.

* * *

## Exercises

### Fundamental Level

**Question 1: Understanding Company Strategies**

Explain the key technical approaches of the following companies: 1\. Exscientia 2\. Insilico Medicine 3\. Recursion Pharmaceuticals 4\. BenevolentAI

For each company, include the following points: \- Core technology \- Data strategy \- Notable achievements

**Question 2: AlphaFold 2 Applications**

List three considerations when using AlphaFold 2-predicted structures for drug discovery. Also describe countermeasures for each.

**Question 3: Comparing Molecular Generation Methods**

Create a table comparing the advantages and disadvantages of VAE, GAN, and Transformer-based molecular generation methods.

### Intermediate Level

**Question 4: Detecting Data Leakage**

The following code has data leakage problems. Identify the issues and correct the code.
    
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    
    # Load data
    X, y = load_chembl_data()
    
    # Outlier removal
    mean = X.mean()
    std = X.std()
    X = X[(X > mean - 3*std) & (X < mean + 3*std)]
    
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    print(f"R² = {r2_score(y_test, y_pred):.3f}")
    

**Question 5: Implementing Synthesizability Filter**

Calculate synthesizability scores (SA Score) for the following molecules and rank them in order of ease of synthesis. Also analyze structural features of difficult-to-synthesize molecules.
    
    
    molecules = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O",  # Estradiol
        "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",  # Tryptophan
        "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",  # Complex synthetic molecule
    ]
    

### Advanced Level

**Question 6: Active Learning Pipeline Design**

Design an Active Learning pipeline to discover novel COVID-19 therapeutics with limited experimental budget (only 100 compounds can be synthesized and tested). Include the following elements:

  1. Initial dataset (what data to use)
  2. Feature design
  3. Model selection (why that model)
  4. Acquisition function (how to select next compounds to test)
  5. Number of experimental cycles and compounds per cycle
  6. Success evaluation criteria

Show implementation overview in Python pseudocode.

**Question 7: Knowledge Graph-Based Hypothesis Generation**

Referencing BenevolentAI's approach, design an algorithm to generate new drug discovery hypotheses from a knowledge graph.

Given the following knowledge graph:
    
    
    Nodes:
    - Genes: BRAF, MEK1, ERK1, TP53
    - Proteins: BRAF protein, MEK1 protein, ERK1 protein, p53 protein
    - Diseases: Melanoma, Colorectal cancer
    - Compounds: Vemurafenib, Dabrafenib, Trametinib
    
    Edges:
    - BRAF → [encodes] → BRAF protein
    - BRAF protein → [activates] → MEK1 protein
    - MEK1 protein → [activates] → ERK1 protein
    - BRAF protein → [mutated_in] → Melanoma
    - Vemurafenib → [inhibits] → BRAF protein
    - Dabrafenib → [inhibits] → BRAF protein
    - Trametinib → [inhibits] → MEK1 protein
    

For this knowledge graph: 1\. Design query to propose novel melanoma treatment strategies 2\. Generate hypotheses with path search algorithm 3\. Define criteria to evaluate generated hypothesis validity

Implement in Python (networkx library can be used).

**Question 8: AI Model Interpretability**

After predicting drug activity with a Random Forest model, perform the following interpretability analyses:

  1. Feature importance visualization using SHAP values
  2. Explanation for individual predictions (why was this molecule predicted as highly active?)
  3. Relationship analysis between chemically meaningful substructures (functional groups) and activity

Use ChEMBL data from Chapter 3 (EGFR inhibitors).

* * *

## References

### Papers

  1. **Exscientia** \- Blay, V. et al. (2020). "High-throughput screening: today's biochemical and cell-based approaches." _Drug Discovery Today_ , 25(10), 1807-1821.

  2. **Insilico Medicine** \- Zhavoronkov, A. et al. (2019). "Deep learning enables rapid identification of potent DDR1 kinase inhibitors." _Nature Biotechnology_ , 37(9), 1038-1040.

  3. **Recursion Pharmaceuticals** \- Mabey, B. et al. (2021). "A phenomics approach for antiviral drug discovery." _BMC Biology_ , 19, 156.

  4. **BenevolentAI** \- Richardson, P. et al. (2020). "Baricitinib as potential treatment for 2019-nCoV acute respiratory disease." _The Lancet_ , 395(10223), e30-e31.

  5. **AlphaFold 2** \- Jumper, J. et al. (2021). "Highly accurate protein structure prediction with AlphaFold." _Nature_ , 596(7873), 583-589.

  6. **Molecular Generative AI** \- Gómez-Bombarelli, R. et al. (2018). "Automatic chemical design using a data-driven continuous representation of molecules." _ACS Central Science_ , 4(2), 268-276. \- Segler, M. H., Kogej, T., Tyrchan, C., & Waller, M. P. (2018). "Generating focused molecule libraries for drug discovery with recurrent neural networks." _ACS Central Science_ , 4(1), 120-131. \- Jin, W., Barzilay, R., & Jaakkola, T. (2018). "Junction tree variational autoencoder for molecular graph generation." _ICML 2018_.

  7. **Halicin (MIT antibiotic discovery)** \- Stokes, J. M. et al. (2020). "A deep learning approach to antibiotic discovery." _Cell_ , 180(4), 688-702.

### Books

  1. **AI Drug Discovery General** \- Kimber, T. B., Chen, Y., & Volkamer, A. (2021). _Deep Learning in Chemistry_. Royal Society of Chemistry. \- Schneider, G., & Clark, D. E. (2019). "Automated de novo drug design: Are we nearly there yet?" _Angewandte Chemie International Edition_ , 58(32), 10792-10803.

  2. **Cheminformatics** \- Leach, A. R., & Gillet, V. J. (2007). _An Introduction to Chemoinformatics_. Springer. \- Gasteiger, J. (Ed.). (2003). _Handbook of Chemoinformatics_. Wiley-VCH.

  3. **Machine Learning** \- Goodfellow, I., Bengio, Y., & Courville, A. (2016). _Deep Learning_. MIT Press. \- Bishop, C. M. (2006). _Pattern Recognition and Machine Learning_. Springer.

### Online Resources

  1. **Company Blogs & Technical Information** \- Exscientia Blog: https://www.exscientia.ai/blog \- Insilico Medicine Publications: https://insilico.com/publications \- Recursion Blog: https://www.recursion.com/blog

  2. **Databases & Tools** \- ChEMBL: https://www.ebi.ac.uk/chembl/ \- PubChem: https://pubchem.ncbi.nlm.nih.gov/ \- AlphaFold Protein Structure Database: https://alphafold.ebi.ac.uk/ \- RDKit Documentation: https://www.rdkit.org/docs/

  3. **Educational Resources** \- DeepChem Tutorials: https://deepchem.io/tutorials/ \- TeachOpenCADD: https://github.com/volkamerlab/teachopencadd \- Molecular AI MOOC: https://molecularai.com/

  4. **Community** \- Reddit r/comp_chem: https://www.reddit.com/r/comp_chem/ \- AI in Drug Discovery LinkedIn Group \- ChemML Community: https://github.com/hachmannlab/chemml

* * *

**Next Chapter Preview** : In the next "Catalyst Materials Informatics" series, we will learn how AI technologies are being applied to catalyst design. We will introduce important application examples in the energy and environmental fields, including high-performance catalyst exploration, reaction condition optimization, and elucidation of reaction mechanisms.
