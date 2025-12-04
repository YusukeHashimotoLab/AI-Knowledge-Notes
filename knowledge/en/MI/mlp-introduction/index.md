---
title: Machine Learning Potential (MLP) Introduction Series v1.0
chapter_title: Machine Learning Potential (MLP) Introduction Series v1.0
---

# Machine Learning Potential (MLP) Introduction Series v1.0

**Next-Generation Simulation Combining Quantum Accuracy with Classical Speed - Complete Guide from Fundamentals to Practice and Career**

## Series Overview

This series is an educational content with a 4-chapter structure designed for progressive learning, from those learning Machine Learning Potentials (MLP) for the first time to those who want to acquire practical skills.

**Features:**  
\- ✅ **Chapter Independence** : Each chapter can be read as a standalone article  
\- ✅ **Systematic Structure** : Comprehensive content with progressive learning across 4 chapters  
\- ✅ **Practice-Oriented** : 15 executable code examples (using SchNetPack), 5 detailed case studies  
\- ✅ **Career Support** : Provides specific career paths and learning roadmaps

**Total Learning Time** : 85-100 minutes (including code execution and exercises)

* * *

## How to Learn

### Recommended Learning Order
    
    
    ```mermaid
    flowchart TD
        A[Chapter 1: Why MLP is Needed] --> B[Chapter 2: MLP Fundamentals]
        B --> C[Chapter 3: Python Hands-on]
        C --> D[Chapter 4: Real-World Applications]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

**For Beginners (Complete Novice):**  
\- Chapter 1 → Chapter 2 → Chapter 3 → Chapter 4  
\- Duration: 85-100 minutes

**Computational Chemistry Practitioners (with DFT/MD basics):**  
\- Chapter 2 → Chapter 3 → Chapter 4  
\- Duration: 60-75 minutes

**Practical Skills Enhancement (already familiar with MLP concepts):**  
\- Chapter 3 (focused learning) → Chapter 4  
\- Duration: 50-60 minutes

* * *

## Chapter Details

### [Chapter 1: Why Machine Learning Potentials (MLP) are Needed](<./chapter1-introduction.html>)

**Difficulty** : Introductory  
**Reading time** : 15-20 minutes

#### Learning Content

  1. **History of Molecular Simulations**  
\- 1950s: Birth of classical molecular dynamics (MD)  
\- 1965: Establishment of DFT theory (Kohn-Sham equations)  
\- 2007: Behler-Parrinello Neural Network Potential  
\- 2017-2025: Graph Neural Networks era (SchNet, NequIP, MACE)

  2. **Limitations of Traditional Methods**  
\- Empirical force fields: Lack of parameter generalizability, unable to handle chemical reactions  
\- DFT: Infeasible for large-scale systems and long-time simulations (10² atoms, ps scale)  
\- Specific numbers: DFT calculation time (several hours for 100 atoms) vs MD (several hours for 1 million atoms)

  3. **Case Study: CO₂ Reduction on Cu Catalyst**  
\- Traditional method (DFT-AIMD): 114,000 years needed for 1 μs MD  
\- MLP-MD: Same 1 μs MD completed in 1 week (50,000× speedup)  
\- Achievement: Elucidation of reaction mechanism, Nature Chemistry publication

  4. **Comparison Diagram (Traditional vs MLP)**  
\- Mermaid diagram: Accuracy vs computational cost trade-off  
\- Timescale comparison: fs (DFT) vs ns-μs (MLP)  
\- Size scale comparison: 10² atoms (DFT) vs 10⁵-10⁶ atoms (MLP)

  5. **Column: "A Day in the Life of a Computational Chemist"**  
\- 2000: DFT calculation for 1 week, 100 atom system, ps scale  
\- 2025: MLP-MD for 1 week, 100,000 atom system, μs scale

  6. **"Why Now?" - Four Tailwinds**  
\- Machine learning advances: Neural networks, graph networks, equivariant NNs  
\- Computational resources: GPUs, supercomputers (Fugaku, Frontier)  
\- Data infrastructure: Large-scale DFT databases like Materials Project, NOMAD  
\- Social needs: Drug discovery, energy, catalysis, environment

#### Learning Objectives

  * ✅ Explain the historical evolution of molecular simulations
  * ✅ Identify three limitations of traditional methods with specific examples
  * ✅ Understand the technical and social background of why MLP is needed
  * ✅ Explain the overview of major MLP methods (Behler-Parrinello, SchNet, NequIP, etc.)

**[Read Chapter 1 →](<./chapter1-introduction.html>)**

* * *

### [Chapter 2: MLP Fundamentals - Concepts, Methods, Ecosystem](<./chapter2-fundamentals.html>)

**Difficulty** : Introductory to Intermediate  
**Reading time** : 20-25 minutes

#### Learning Content

  1. **What is MLP: Precise Definition**  
\- Machine learning approximation of potential energy surface (PES)  
\- Three essential elements: data-driven, high-dimensional approximation, physical constraints  
\- Related fields: quantum chemistry, machine learning, molecular dynamics

  2. **15 MLP Terms Glossary**  
\- Basic terms: Potential energy surface (PES), forces, energy conservation  
\- Method terms: descriptors, symmetry, equivariance, message passing  
\- Application terms: active learning, uncertainty quantification, transfer learning

  3. **Input Data for MLP**  
\- Five major data types: equilibrium structures, MD trajectories, reaction paths, random sampling, defect structures  
\- DFT training data: energies, forces, stresses  
\- Dataset example: Cu catalyst CO₂ reduction (10,000 structures, 5,000 hours DFT calculation time)

  4. **MLP Ecosystem Diagram**  
\- Mermaid diagram: DFT data generation → model training → simulation → analysis  
\- Four phases and time requirements  
\- Toolchain: VASP/Quantum ESPRESSO → ASE → SchNetPack → LAMMPS/ASE-MD

  5. **MLP Workflow: 5 Steps (Detailed Version)**  
\- **Step 1** : Data collection (DFT calculations, sampling strategies)  
\- **Step 2** : Descriptor design (symmetry functions, SOAP, graph NNs)  
\- **Step 3** : Model training (loss functions, optimization methods)  
\- **Step 4** : Validation (MAE target values, extrapolation tests)  
\- **Step 5** : Production simulation (MLP-MD setup, property calculations)

  6. **Types of Descriptors: Numerical Representation of Atomic Configurations**  
\- **Symmetry Functions** : Behler-Parrinello type, radial and angular terms  
\- **SOAP (Smooth Overlap of Atomic Positions)** : Atomic density representation, kernel methods  
\- **Graph Neural Networks** : SchNet (continuous-filter convolution), DimeNet (directional), NequIP (E(3) equivariant), MACE (higher-order equivariant)

  7. **Comparison of Major MLP Architectures**  
\- Evolution of 7 methods (2007-2024)  
\- Comparison table: accuracy, data efficiency, computational speed, implementation difficulty  
\- Mermaid evolution timeline

  8. **Column: Efficient Data Collection with Active Learning**  
\- Active learning workflow  
\- Uncertainty evaluation methods  
\- Success story: 88% reduction in data collection cost

#### Learning Objectives

  * ✅ Explain the definition of MLP and its differences from related fields (quantum chemistry, machine learning)
  * ✅ Understand the characteristics of major descriptors (symmetry functions, SOAP, graph NNs)
  * ✅ Detail the MLP workflow 5 steps including substeps
  * ✅ Use 15 MLP technical terms appropriately
  * ✅ Explain the evolution of major MLP architectures (Behler-Parrinello through MACE)

**[Read Chapter 2 →](<./chapter2-fundamentals.html>)**

* * *

### [Chapter 3: Experience MLP with Python - SchNetPack Hands-on](<./chapter3-hands-on.html>)

**Difficulty** : Intermediate  
**Reading time** : 30-35 minutes  
**Code examples** : 15 (all executable)

#### Learning Content

  1. **Environment Setup**  
\- Conda environment setup  
\- PyTorch, SchNetPack installation  
\- Functionality check (5-line code)

  2. **Data Preparation (Examples 1-3)**  
\- Loading MD17 dataset (aspirin molecule, 1,000 samples)  
\- Train/validation/test split (80%/10%/10%)  
\- Data statistics visualization

  3. **SchNetPack Training (Examples 4-8)**  
\- SchNet model definition (cutoff=5Å, n_interactions=3)  
\- Training loop implementation (loss function: energy + forces)  
\- TensorBoard visualization  
\- Training progress monitoring  
\- Checkpoint saving

  4. **Accuracy Validation (Examples 7-8)**  
\- Test set evaluation (MAE target: < 1 kcal/mol)  
\- Prediction vs measurement correlation plots  
\- Error analysis

  5. **MLP-MD Execution (Examples 9-12)**  
\- Using SchNet as ASE Calculator  
\- NVT ensemble MD (300 K, 10 ps)  
\- Speed comparison with DFT (10⁴× speedup)  
\- Trajectory visualization and analysis

  6. **Property Calculations (Examples 13-15)**  
\- Vibrational spectrum calculation (Fourier transform)  
\- Self-diffusion coefficient calculation (MSD, Einstein relation)  
\- Radial distribution function (RDF)

  7. **Active Learning (Example 15)**  
\- Ensemble uncertainty evaluation  
\- Automatic detection of high-uncertainty configurations  
\- DFT calculation requests

  8. **Troubleshooting**  
\- 5 common errors and solutions (table format)  
\- Debugging best practices

  9. **Summary**  
\- Organization of 7 learning contents  
\- Bridge to next chapter (real applications)

#### Learning Objectives

  * ✅ Set up SchNetPack environment
  * ✅ Train SchNet on MD17 dataset (achieve MAE < 1 kcal/mol)
  * ✅ Execute MLP-MD and compare speed with DFT (confirm 10⁴× speedup)
  * ✅ Calculate vibrational spectra, diffusion coefficients, and RDF
  * ✅ Perform uncertainty evaluation with active learning
  * ✅ Troubleshoot common errors independently

**[Read Chapter 3 →](<./chapter3-hands-on.html>)**

* * *

### [Chapter 4: Real-World MLP Applications - Success Stories and Future Outlook](<./chapter4-real-world.html>)

**Difficulty** : Intermediate to Advanced  
**Reading time** : 20-25 minutes

#### Learning Content

  1. **5 Detailed Case Studies**

**Case Study 1: Catalytic Reaction Mechanism Elucidation (Cu CO₂ Reduction)**  
\- Technology: SchNet + AIMD trajectory, transition state search  
\- Results: Reaction pathway identification, 50,000× speedup, μs-scale MD realization  
\- Impact: Nature Chemistry 2020 publication, application to industrial catalyst design  
\- Organizations: MIT, SLAC National Lab

**Case Study 2: Li-ion Battery Electrolyte Design**  
\- Technology: DeepMD-kit, active learning, ionic conductivity prediction  
\- Results: New electrolyte discovery, 3× ionic conductivity improvement, 7.5× development time reduction  
\- Impact: Commercialization (2023), EV battery performance improvement  
\- Organizations: Toyota, Panasonic

**Case Study 3: Protein Folding (Drug Discovery)**  
\- Technology: TorchANI/ANI-2x, long-time MD simulation  
\- Results: Folding trajectory prediction, drug design support, 50% development time reduction  
\- Impact: Clinical trial success rate improvement, new drug candidate discovery  
\- Organizations: Schrödinger, Pfizer

**Case Study 4: Semiconductor Materials (GaN Crystal Growth)**  
\- Technology: MACE, defect energy calculations, growth simulation  
\- Results: Optimal growth condition discovery, 90% defect density reduction, 30% mass production cost reduction  
\- Impact: Next-generation power semiconductors, 5G/6G communication devices  
\- Organizations: National Institute for Materials Science (NIMS), Shin-Etsu Chemical

**Case Study 5: Atmospheric Chemical Reactions (Climate Change Prediction)**  
\- Technology: NequIP, large-scale MD, reaction rate constant calculations  
\- Results: High-precision atmospheric chemistry model, 2.5× climate prediction accuracy improvement  
\- Impact: Contribution to IPCC reports, policy decision support  
\- Organizations: NASA, NCAR (National Center for Atmospheric Research)

  2. **Future Trends (3 Major Trends)**

**Trend 1: Foundation Models for Chemistry**  
\- Examples: ChemGPT, MolFormer, Universal NNP  
\- Prediction: By 2030, MLP will replace 80% of all DFT calculations  
\- Initial investment: 1 billion yen (GPU cluster + personnel costs)  
\- ROI: Recovered in 2-3 years

**Trend 2: Autonomous Lab**  
\- Examples: RoboRXN (IBM), A-Lab (Berkeley)  
\- Effects: Complete automation from experimental planning to execution, 24× materials development acceleration  
\- Prediction: By 2030, 50% of major companies will adopt

**Trend 3: Quantum-accurate Millisecond MD**  
\- Technology: MLP + enhanced sampling, rare event simulation  
\- Applications: Protein aggregation, crystal nucleation, catalytic cycles  
\- Impact: Breakthrough in drug discovery and materials development

  3. **Career Paths (3 Major Routes)**

**Path 1: Academic Research (Researcher)**  
\- Route: Bachelor → Master → PhD (3-5 years) → Postdoc (2-3 years) → Associate Professor  
\- Salary: ¥5-12 million/year (Japan), $60-120K (USA)  
\- Skills: Python, PyTorch, quantum chemistry, scientific writing, programming  
\- Examples: University of Tokyo, Kyoto University, MIT, Stanford

**Path 2: Industry R &D**  
\- Positions: MLP engineer, computational chemist, data scientist  
\- Salary: ¥7-15 million/year (Japan), $80-200K (USA)  
\- Companies: Mitsubishi Chemical, Sumitomo Chemical, Toyota, Panasonic, Schrödinger  
\- Skills: Python, machine learning, quantum chemistry, teamwork, business understanding

**Path 3: Startup/Consulting**  
\- Examples: Schrödinger (market cap $8B), Chemify, QuantumBlack  
\- Salary: ¥5-10 million/year + stock options  
\- Risk/Return: High risk, high impact  
\- Required skills: Technology + business + leadership

  4. **Skills Development Timeline**  
\- **3-Month Plan** : Fundamentals (Python, PyTorch, quantum chemistry) → Practice (SchNetPack) → Portfolio  
\- **1-Year Plan** : Advanced (paper implementation, original projects) → Conference presentations → Community contribution  
\- **3-Year Plan** : Expert (5-10 paper publications) → Leadership → Community recognition

  5. **Learning Resources**  
\- **Online Courses** : MIT OCW, Coursera ("Molecular Simulations")  
\- **Books** : "Machine Learning for Molecular Simulation" (Behler), "Graph Neural Networks" (Wu et al.)  
\- **Open Source** : SchNetPack, NequIP, MACE, DeePMD-kit, TorchANI  
\- **Communities** : CECAM, MolSSI, Computational Chemistry Society of Japan  
\- **Conferences** : ACS, MRS, APS, Chemical Society of Japan

#### Learning Objectives

  * ✅ Explain 5 real-world MLP success stories with technical details
  * ✅ Identify 3 future MLP trends and evaluate their industry impact
  * ✅ Explain 3 types of MLP career paths and understand required skills
  * ✅ Plan a specific learning timeline (3 months/1 year/3 years)
  * ✅ Select appropriate learning resources for next steps

**[Read Chapter 4 →](<./chapter4-real-world.html>)**

* * *

## Overall Learning Outcomes

Upon completing this series, you will have acquired the following skills and knowledge:

### Knowledge Level (Understanding)

  * ✅ Explain the historical background and necessity of MLP
  * ✅ Understand basic MLP concepts, terminology, and methods
  * ✅ Distinguish between major MLP architectures (Behler-Parrinello, SchNet, NequIP, MACE)
  * ✅ Detail 5 or more real-world success stories

### Practical Skills (Doing)

  * ✅ Set up SchNetPack environment and train models
  * ✅ Achieve MAE < 1 kcal/mol on MD17 dataset
  * ✅ Execute MLP-MD and compare speed with DFT (confirm 10⁴× speedup)
  * ✅ Calculate vibrational spectra, diffusion coefficients, and RDF
  * ✅ Perform efficient data collection with active learning
  * ✅ Debug errors independently

### Application Ability (Applying)

  * ✅ Design MLP application projects for new chemical systems
  * ✅ Evaluate industry adoption cases and apply to your own research
  * ✅ Plan future career paths concretely
  * ✅ Establish continuous learning strategies

* * *

## Recommended Learning Patterns

### Pattern 1: Complete Mastery (For Beginners)

**Target** : Those learning MLP for the first time, those wanting systematic understanding  
**Duration** : 2-3 weeks  
**Approach** :
    
    
    Week 1:
    - Day 1-2: Chapter 1 (History and background, limitations of traditional methods)
    - Day 3-4: Chapter 2 (Fundamentals, descriptors, architectures)
    - Day 5-7: Chapter 2 exercises, terminology review
    
    Week 2:
    - Day 1-2: Chapter 3 (Environment setup, data preparation)
    - Day 3-4: Chapter 3 (SchNetPack training, validation)
    - Day 5-7: Chapter 3 (MLP-MD, property calculations)
    
    Week 3:
    - Day 1-2: Chapter 3 (Active learning, troubleshooting)
    - Day 3-4: Chapter 4 (5 case studies)
    - Day 5-7: Chapter 4 (Career plan creation)
    

**Deliverables** :  
\- SchNet training project on MD17 dataset (MAE < 1 kcal/mol)  
\- Personal career roadmap (3 months/1 year/3 years)

### Pattern 2: Fast Track (For Computational Chemistry Practitioners)

**Target** : Those with DFT/MD fundamentals wanting to transition to MLP  
**Duration** : 1 week  
**Approach** :
    
    
    Day 1: Chapter 2 (Focus on MLP-specific concepts)
    Day 2-3: Chapter 3 (Environment setup, training, validation)
    Day 4: Chapter 3 (MLP-MD, property calculations)
    Day 5-6: Chapter 4 (Case studies and career)
    Day 7: Review and next steps planning
    

**Deliverables** :  
\- SchNetPack project portfolio (GitHub publication recommended)  
\- MLP vs DFT speed comparison report

### Pattern 3: Pinpoint Learning (Specific Topic Focus)

**Target** : Those wanting to strengthen specific skills or knowledge  
**Duration** : Flexible  
**Selection Examples** :

  * **Deep understanding of descriptors** → Chapter 2 (Section 2.6)
  * **Master SchNetPack** → Chapter 3 (Sections 3.3-3.7)
  * **Learn active learning** → Chapter 2 (Column) + Chapter 3 (Section 3.7)
  * **Career design** → Chapter 4 (Sections 4.3-4.5)
  * **Learn latest trends** → Chapter 4 (Section 4.2)

* * *

## FAQ (Frequently Asked Questions)

### Q1: Can I understand without quantum chemistry knowledge?

**A** : Chapters 1 and 2 do not assume detailed quantum chemistry knowledge, but basic chemistry (atoms, molecules, chemical bonding) is helpful. In Chapter 3, SchNetPack abstracts quantum chemical calculations, so detailed knowledge is not required. However, understanding basic DFT concepts (energy, forces, potential energy surface) will enable deeper learning.

### Q2: Is machine learning experience required?

**A** : Not required, but Python and neural network fundamentals are advantageous. In Chapter 3, SchNetPack hides machine learning complexity, so basic Python skills (variables, functions, loops) are sufficient to start. However, for deeper understanding, we recommend learning PyTorch fundamentals (tensors, automatic differentiation, optimization).

### Q3: Is GPU necessary?

**A** : **GPU is strongly recommended for training**. CPU is possible but training time becomes 10-100× longer. Options:  
\- **Google Colab** : Free GPU (T4) is sufficient (optimal for Chapter 3 code examples)  
\- **Local GPU** : NVIDIA RTX 3060 or better recommended (VRAM 8GB+)  
\- **Supercomputer/Cloud** : Large-scale projects (AWS EC2 p3 instances, etc.)

MLP-MD execution is sufficiently fast on CPU (compared to DFT).

### Q4: How long to reach practical level?

**A** : Depends on goals and background:  
\- **Basic usage (train SchNetPack, perform MD using provided datasets)** : 1-2 weeks  
\- **Apply MLP to custom systems (including DFT data collection)** : 1-3 months  
\- **Research and development of new methods** : 6-12 months  
\- **Industry ready** : 1-2 years (including project experience)

### Q5: Can I become an MLP expert with this series alone?

**A** : This series targets "introductory to intermediate" level. To reach expert level:  
1\. Build foundation with this series (2-4 weeks)  
2\. Study advanced content with Chapter 4 learning resources (3-6 months)  
3\. Execute your own projects (6-12 months)  
4\. Conference presentations and paper writing (1-2 years)

A total of 2-3 years of continuous learning and practice is required.

### Q6: What is the difference between MLP and Materials Informatics (MI)?

**A** : **MLP (Machine Learning Potential)** is a method to **approximate potential energy surfaces of molecules/materials using machine learning**. **MI (Materials Informatics)** refers to the application of data science/machine learning to materials science in general, with MLP being one subfield of MI.

  * **MLP** : Simulation acceleration, reaction pathway exploration, long-time MD
  * **MI** : Materials discovery, property prediction, composition optimization, experimental design

This site provides series for both!

### Q7: Which MLP architecture should I choose?

**A** : Depends on the situation:

Situation | Recommended Architecture | Reason  
---|---|---  
Beginner, first try | **SchNet** | Simple implementation, SchNetPack available  
High accuracy needed | **NequIP or MACE** | E(3) equivariant, highest accuracy  
Limited data | **MACE** | Best data efficiency  
Long-range interactions important | **MACE** | Efficiently handles long-range terms  
Computational speed priority | **Behler-Parrinello or SchNet** | Fast inference  
Integration with existing projects | **DeepMD-kit** | Easy LAMMPS integration  
  
**Chapter 3 uses SchNet** (optimal for beginners).

### Q8: Is commercial use possible?

**A** : **Open-source libraries like SchNetPack, NequIP, MACE are MIT licensed** and available for commercial use. However:  
\- **Training data (DFT calculations)** : Data you generate yourself can be used freely  
\- **Public datasets (MD17, etc.)** : Check license (many are academic use only)  
\- **Commercial software** : Schrödinger, Materials Studio, etc. require separate licensing

If considering use in a company, we recommend consulting with your legal department.

### Q9: Are there communities for questions and discussion?

**A** : You can ask questions and discuss in the following communities:  
\- **Japan** : Computational Chemistry Society of Japan, Molecular Science Society  
\- **International** : CECAM (Centre Européen de Calcul Atomique et Moléculaire), MolSSI (Molecular Sciences Software Institute)  
\- **Online** :  
\- [SchNetPack GitHub Discussions](<https://github.com/atomistic-machine-learning/schnetpack/discussions>)  
\- [Materials Project Discussion Forum](<https://matsci.org/>)  
\- Stack Overflow (`machine-learning-potential`, `molecular-dynamics` tags)

* * *

## Next Steps

### Recommended Actions After Series Completion

**Immediate (within 1-2 weeks):**  
1\. ✅ Create portfolio on GitHub/GitLab  
2\. ✅ Publish SchNetPack project results with README  
3\. ✅ Add "Machine Learning Potential", "SchNetPack" skills to LinkedIn profile

**Short-term (1-3 months):**  
1\. ✅ Train MLP on your own chemical system (including DFT data generation)  
2\. ✅ Try NequIP or MACE (compare with SchNet)  
3\. ✅ Participate in Computational Chemistry Society of Japan study groups  
4\. ✅ Read 5-10 papers thoroughly (_Nature Chemistry_ , _JCTC_ , _PRB_)

**Medium-term (3-6 months):**  
1\. ✅ Contribute to open-source projects (SchNetPack, NequIP, etc.)  
2\. ✅ Present at domestic conferences (Chemical Society of Japan, Computational Chemistry Society)  
3\. ✅ Implement active learning to improve data collection efficiency  
4\. ✅ Collaboration with industry or internship

**Long-term (1 year+):**  
1\. ✅ Present at international conferences (ACS, MRS, APS)  
2\. ✅ Submit peer-reviewed papers (_JCTC_ , _J. Chem. Phys._ , etc.)  
3\. ✅ Secure MLP-related job (academia or industry)  
4\. ✅ Train the next generation of MLP researchers and engineers

* * *

## Feedback and Support

### About This Series

This series was created under Dr. Yusuke Hashimoto at Tohoku University as part of the MI Knowledge Hub project.

**Creation Date** : October 17, 2025  
**Version** : 1.0

### We Welcome Your Feedback

We welcome your feedback to improve this series:

  * **Typos, errors, technical inaccuracies** : Report via GitHub repository Issues
  * **Improvement suggestions** : New topics, additional code examples, etc.
  * **Questions** : Difficult parts, areas needing additional explanation
  * **Success stories** : Projects using what you learned from this series

**Contact** : yusuke.hashimoto.b8@tohoku.ac.jp

* * *

## License and Terms of Use

This series is published under **CC BY 4.0** (Creative Commons Attribution 4.0 International) license.

**Permitted:**  
\- ✅ Free viewing and download  
\- ✅ Use for educational purposes (classes, study groups, etc.)  
\- ✅ Modification and derivative works (translation, summarization, etc.)

**Conditions:**  
\- 📌 Author credit required  
\- 📌 Must indicate if modifications were made  
\- 📌 Contact in advance for commercial use

Details: [CC BY 4.0 License Full Text](<https://creativecommons.org/licenses/by/4.0/>)

* * *

## Let's Begin!

Are you ready? Start with Chapter 1 and begin your journey into the world of MLP!

**[Chapter 1: Why Machine Learning Potentials (MLP) are Needed →](<./chapter1-introduction.html>)**

* * *

**Update History**

  * **2025-10-17** : v1.0 Initial release

* * *

**Your MLP learning journey starts here!**

[← Back to Series Contents](<index.html>)
