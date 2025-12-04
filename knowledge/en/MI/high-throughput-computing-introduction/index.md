---
title: High-Throughput Computing Introduction Series
chapter_title: High-Throughput Computing Introduction Series
subtitle: Accelerate materials discovery through automation and parallelization
difficulty: Intermediate to Advanced
version: 1.0
created_at: 2025-10-17
---

# High-Throughput Computing Introduction Series

**Complete guide to accelerating materials discovery 1000x through automation and parallelization**

## Series Overview

This series is a comprehensive 5-chapter educational resource for researchers and engineers who want to learn High-Throughput Computational Materials Science (HTCMS). It covers practical skills from DFT calculation automation to workflow management, parallel computing, and cloud HPC utilization, with step-by-step progression.

**Features:**

  * ✅ **Practice-oriented** : Implementation examples using ASE, pymatgen, FireWorks, and AiiDA
  * ✅ **Comprehensive coverage** : From automation design to large-scale parallel computing
  * ✅ **Industrial applications** : Success stories from Materials Project, AFLOW, and more
  * ✅ **Reproducibility** : Fully reproducible with Docker and environment setup scripts

**Total study time** : 110-140 minutes (including code execution and exercises) 

\---

## How to Study

### Recommended Learning Sequence
    
    
    ```mermaid
    flowchart TD
        A[Chapter 1: Need for HTC] --> B[Chapter 2: DFT Automation]
        B --> C[Chapter 3: Job Scheduling]
        C --> D[Chapter 4: Data Management & Workflows]
        D --> E[Chapter 5: Cloud HPC Utilization]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
    ```

**For DFT-experienced users (recommended):**

  * Chapter 1 → Chapter 2 → Chapter 3 → Chapter 4 → Chapter 5
  * Time required: 110-140 minutes

**Automation focus only:**

  * Chapter 2 → Chapter 4
  * Time required: 50-60 minutes

**Cloud utilization only:**

  * Chapter 1 → Chapter 5
  * Time required: 35-45 minutes

\---

## Chapter Details

### [Chapter 1: The Need for High-Throughput Computing and Workflow Design](<chapter-1.html>)

**Difficulty** : Intermediate **Reading time** : 20-30 minutes 

#### Learning Content

  1. **Challenges in materials discovery**

\- Vast search space: 10^60 combinations

\- Limitations of traditional methods: from 1 material/week → 1000 materials/week needed

\- Materials Genome Initiative (MGI) goals

  1. **Definition of High-Throughput Computing**

\- Automation

\- Parallelization

\- Standardization

\- Data Management

  1. **Success stories**

\- Materials Project: DFT calculations of 140,000 materials

\- AFLOW: Automated analysis of 3,500,000 crystal structures

\- OQMD: Thermodynamic data of 815,000 materials

\- JARVIS: Diverse property calculations for 40,000 materials

  1. **Workflow design principles**

\- Modularity

\- Error Handling

\- Reproducibility

\- Scalability

  1. **Cost and benefits**

\- Development time: 15-20 years → 3-5 years (67% reduction)

\- Experimental cost: 95% reduction

\- Computational cost: Initial investment vs long-term ROI

#### Learning Objectives

  * ✅ Explain the four elements of High-Throughput Computing
  * ✅ Analyze the success factors of Materials Project
  * ✅ Understand workflow design principles
  * ✅ Quantitatively evaluate cost reduction effects

**[Read Chapter 1 →](<chapter-1.html>)**

\---

### [Chapter 2: DFT Calculation Automation (VASP, Quantum ESPRESSO)](<chapter-2.html>)

**Difficulty** : Intermediate to Advanced **Reading time** : 20-25 minutes **Code examples** : 6 

#### Learning Content

  1. **ASE (Atomic Simulation Environment) basics**

\- Installation and environment setup

\- Structure generation and manipulation

\- Calculator interface

\- Automated result analysis

  1. **VASP automation**

\- Automatic INCAR file generation

\- Automatic K-point settings

\- POTCAR management

\- Automated convergence checking

  1. **Quantum ESPRESSO automation**

\- Input file templates

\- Pseudopotential management

\- SCF/NSCF/DOS calculation chains

\- Automatic band structure plotting

  1. **Advanced automation with pymatgen**

\- InputSet: Standardized input generation

\- Taskflow: Calculation flow definition

\- Error detection and restart

\- Result database integration

  1. **Structure optimization automation**

\- Relaxation calculation convergence checking

\- Lattice constant optimization

\- Internal coordinate optimization

\- Automatic energy cutoff determination

  1. **Troubleshooting**

\- Common errors and solutions

\- Diagnosing non-convergent calculations

\- Handling memory shortages

#### Learning Objectives

  * ✅ Execute basic DFT calculations automatically using ASE
  * ✅ Auto-generate VASP and Quantum ESPRESSO input files
  * ✅ Master pymatgen InputSet usage
  * ✅ Detect errors and auto-restart calculations

**[Read Chapter 2 →](<chapter-2.html>)**

\---

### [Chapter 3: Job Scheduling and Parallelization (SLURM, PBS)](<chapter-3.html>)

**Difficulty** : Advanced **Reading time** : 25-30 minutes **Code examples** : 5-6 

#### Learning Content

  1. **Job scheduler fundamentals**

\- SLURM vs PBS vs Torque

\- Queue system mechanisms

\- Resource request optimization

\- Priority and fairness

  1. **SLURM script creation**

\- Header: `#SBATCH` directives

\- Node count, core count, memory requests

\- Time limit settings

\- Array jobs for parallel execution

  1. **MPI parallel computing**

\- MPI parallelization principles

\- Running VASP/QE with MPI

\- Inter-node communication optimization

\- Scaling efficiency evaluation

  1. **Job management with Python**

\- Job submission via subprocess

\- Job status monitoring

\- Wait for completion and auto-resubmit

\- Dependent job chains

  1. **Large-scale parallel computing**

\- Simultaneous calculation of 1000 materials

\- Avoiding resource contention

\- Fail-safe design

\- Computational cost optimization

  1. **Benchmarking and tuning**

\- Measuring parallel efficiency

\- Bottleneck analysis

\- I/O optimization

\- Memory bandwidth considerations

#### Learning Objectives

  * ✅ Create and submit SLURM scripts
  * ✅ Evaluate MPI parallel computing efficiency
  * ✅ Write Python job management scripts
  * ✅ Design parallel computations at 1000-material scale

**[Read Chapter 3 →](<chapter-3.html>)**

\---

### [Chapter 4: Data Management and Post-processing (FireWorks, AiiDA)](<chapter-4.html>)

**Difficulty** : Advanced **Reading time** : 20-25 minutes **Code examples** : 6 

#### Learning Content

  1. **Workflow management with FireWorks**

\- FireWorks architecture

\- Firework (single task) definition

\- Workflow (task chain) construction

\- LaunchPad (database) setup

  1. **Atomate workflows adopted by Materials Project**

\- Standard workflows: Structure optimization → Static calculation → Band structure

\- Custom workflow creation

\- Error handling and restart

\- JSON output of results

  1. **Provenance management with AiiDA**

\- Importance of data provenance tracking

\- AiiDA data model

\- WorkChain definition

\- Querying and data search

  1. **Structuring computational data**

\- JSON schema design

\- MongoDB/SQLite selection

\- Index optimization

\- Version control

  1. **Post-processing automation**

\- Automatic DOS/band structure plotting

\- Phonon dispersion analysis

\- Thermodynamic quantity calculation

\- Automatic report generation

  1. **Data sharing and archiving**

\- Uploading to NOMAD Repository

\- Publishing on Materials Cloud

\- DOI acquisition

\- FAIR data principles

#### Learning Objectives

  * ✅ Build complex workflows with FireWorks
  * ✅ Master Atomate standard workflows
  * ✅ Record data provenance with AiiDA
  * ✅ Publish calculation results to NOMAD

**[Read Chapter 4 →](<chapter-4.html>)**

\---

### [Chapter 5: Cloud HPC Utilization and Optimization](<chapter-5.html>)

**Difficulty** : Intermediate to Advanced **Reading time** : 15-20 minutes **Code examples** : 5 

#### Learning Content

  1. **Cloud HPC options**

\- AWS Parallel Cluster

\- Google Cloud HPC Toolkit

\- Azure CycleCloud

\- Dedicated HPC: TSUBAME, Fugaku

  1. **AWS Parallel Cluster setup**

\- VPC/subnet configuration

\- Cluster configuration file

\- SLURM integration

\- Storage (EFS/FSx)

  1. **Cost optimization**

\- Spot instance utilization

\- Auto-scaling

\- Idle timeout

\- Storage tiering

  1. **Containerization with Docker/Singularity**

\- Complete environment reproduction

\- Dockerfile best practices

\- Singularity on HPC

\- Image registry management

  1. **Security and compliance**

\- Access control (IAM)

\- Data encryption

\- Log auditing

\- Academic license compliance

  1. **Case study: 10,000 materials screening**

\- Requirements definition

\- Architecture design

\- Implementation and execution

\- Cost analysis (total $500-1000)

#### Learning Objectives

  * ✅ Build AWS Parallel Cluster
  * ✅ Reduce costs by 50% using spot instances
  * ✅ Fully reproduce computational environment with Docker
  * ✅ Design and execute 10,000-material scale projects

**[Read Chapter 5 →](<chapter-5.html>)**

\---

## Overall Learning Outcomes

Upon completing this series, you will acquire the following skills and knowledge:

### Knowledge Level (Understanding)

  * ✅ Explain the principles and necessity of High-Throughput Computing
  * ✅ Understand the Materials Project technology stack
  * ✅ Compare workflow management tools
  * ✅ Know cloud HPC options and their characteristics

### Practical Skills (Doing)

  * ✅ Automate DFT calculations with ASE/pymatgen
  * ✅ Write SLURM scripts and submit parallel jobs
  * ✅ Build complex workflows with FireWorks/AiiDA
  * ✅ Execute large-scale calculations on AWS Parallel Cluster
  * ✅ Publish calculation results to NOMAD

### Application (Applying)

  * ✅ Design 1000-material scale screening projects
  * ✅ Optimize computational costs to stay within budget
  * ✅ Build reproducible research workflows
  * ✅ Propose HT computing implementation in industry

\---

## Recommended Study Patterns

### Pattern 1: Complete Mastery (for HTC beginners)

**Target audience** : Those with DFT calculation experience but new to automation **Duration** : 2 weeks **Approach** : 
    
    
    Week 1:
    
    
    
    
        * Day 1-2: Chapter 1 (conceptual understanding)
    
    
        * Day 3-4: Chapter 2 (ASE/pymatgen implementation)
    
    
        * Day 5-7: Chapter 3 (SLURM practice)
    
    
    
    
    
    
    Week 2:
    
    
    
    
    
    
        * Day 1-3: Chapter 4 (FireWorks/AiiDA)
    
    
        * Day 4-5: Chapter 5 (Cloud HPC)
    
    
        * Day 6-7: Integration project (100-material screening)
    
    
    
    

**Deliverables** : 

  * 100-material band gap prediction project
  * GitHub-published workflow code

### Pattern 2: Fast Track (for experienced users)

**Target audience** : Those with DFT automation experience who want to learn workflow management **Duration** : 3-5 days **Approach** : 
    
    
    Day 1: Chapter 1 (can skip) + Chapter 2 (review)
    
    
    Day 2-3: Chapter 4 (intensive FireWorks learning)
    
    
    
    
    Day 4: Chapter 5 (cloud practice)
    
    
    
    
    Day 5: Project integration
    
    
    

### Pattern 3: Cloud Specialization

**Target audience** : Those considering migration from on-premise HPC to cloud **Duration** : 1 week **Approach** : 
    
    
    Day 1-2: Chapter 1 + Chapter 5 (understand cloud options)
    
    
    Day 3-4: AWS Parallel Cluster construction
    
    
    
    
    Day 5-6: Porting existing workflows
    
    
    
    
    Day 7: Cost optimization and benchmarking
    
    
    

\---

## Prerequisites

### Required

  * ✅ **DFT calculation fundamentals** : Experience with VASP, Quantum ESPRESSO, or CP2K
  * ✅ **Linux/UNIX commands** : bash, ssh, file operations
  * ✅ **Python basics** : Functions, classes, modules

### Recommended

  * ✅ **Materials science basics** : Crystal structures, band theory, density of states
  * ✅ **Machine learning basics** : Completed MI introduction series
  * ✅ **Git/GitHub** : Version control basics

### Nice to have

  * ✅ **Cluster computing experience** : Job scheduler usage history
  * ✅ **Database basics** : SQL, JSON, MongoDB
  * ✅ **Docker basics** : Container concepts

\---

## Tools and Software

### Required Tools

| Tool | Purpose | License | Installation |

|--------|------|-----------|-------------|

| Python 3.10+ | Script execution | Open | conda, pip |

| ASE | Structure manipulation & calculation | LGPL | `pip install ase` |

| pymatgen | Materials science computing | MIT | `pip install pymatgen` |

### DFT Codes (at least one)

| Code | License | Features |

|--------|-----------|------|

| VASP | Commercial (academic license) | High accuracy, widely used |

| Quantum ESPRESSO | GPL | Open source, plane-wave basis |

| CP2K | GPL | Open source, hybrid basis |

### Workflow Tools

| Tool | Adopting Project | Learning Difficulty |

|--------|----------------|-----------|

| FireWorks | Materials Project | Medium |

| AiiDA | MARVEL (Europe) | High |

| Atomate | Materials Project | Medium |

### Cloud HPC

| Service | Recommended use | Initial cost |

|---------|---------|---------|

| AWS Parallel Cluster | Large-scale computing | $0 (pay-as-you-go) |

| Google Cloud HPC | ML integration | $0 (pay-as-you-go) |

| Azure CycleCloud | Windows integration | $0 (pay-as-you-go) |

\---

## FAQ (Frequently Asked Questions)

### Q1: Can I take this course without DFT calculation experience?

**A** : Chapters 2 and beyond assume basic experience with VASP or Quantum ESPRESSO. If you are new to DFT, we recommend first learning the basics through a "Computational Materials Science Fundamentals" series. 

### Q2: Do I need a commercial VASP license?

**A** : Most code examples can be executed with open-source Quantum ESPRESSO. VASP is included in examples because of its widespread use in industry, but if you don't have an academic license, you can use alternative codes. 

### Q3: What if I don't have access to an HPC cluster?

**A** : Chapter 5 teaches how to use cloud HPC (AWS, Google Cloud). You can start without initial investment—a budget of $100-200 is sufficient for initial learning. 

### Q4: Should I learn FireWorks or AiiDA?

**A** : **FireWorks is recommended if you want the same environment as Materials Project**. AiiDA is popular in Europe and has strengths in data provenance tracking. Chapter 4 covers both, but we recommend starting with FireWorks. 

### Q5: How much computational resources do I need?

**A** : For the learning phase: 

  * **Local PC** : Chapters 1-2 (small test calculations)
  * **University cluster** : Chapters 3-4 (parallel computing)
  * **Cloud** : Chapter 5 (budget of $50-100)

For real projects, 1000 materials costs approximately $500-1000.

### Q6: Is the Materials Project code publicly available?

**A** : Yes, the core technology of Materials Project is open-sourced at: 

  * **pymatgen** : https://github.com/materialsproject/pymatgen
  * **FireWorks** : https://github.com/materialsproject/fireworks
  * **Atomate** : https://github.com/hackingmaterials/atomate

This series uses these tools.

### Q7: Are the skills learned useful in industry?

**A** : Extremely useful. High-Throughput Computing is used in the following companies/research institutes: 

  * **Japan** : Toyota, Panasonic, Mitsubishi Chemical, NIMS
  * **Overseas** : Tesla, IBM Research, BASF, DuPont

MI engineer positions offer salaries of 7-15M JPY (Japan), $80-200K (US).

### Q8: What should I be careful about when using calculation results in papers?

**A** : Please confirm the following: 

  1. **Reproducibility** : Specify calculation conditions (k-points, cutoff, etc.)
  2. **License** : Cite software used in the paper
  3. **Data publication** : Publish raw data on NOMAD etc. (recommended)
  4. **Validation** : Compare at least some results with experimental data

\---

## Next Steps

### Recommended Actions After Series Completion

**Immediate (within 1-2 weeks):**

  1. ✅ Execute a small-scale project of 100-1000 materials
  2. ✅ Publish calculation results to NOMAD
  3. ✅ Publish workflow code to GitHub

**Short-term (1-3 months):**

  1. ✅ Contribute to Materials Project codebase
  2. ✅ Develop custom FireWorks workflows
  3. ✅ Submit paper (including computational dataset publication)

**Medium-term (3-6 months):**

  1. ✅ Execute 10,000-material scale project
  2. ✅ Accumulate cloud HPC cost optimization know-how
  3. ✅ Present at international conferences (MRS, E-MRS)

**Long-term (1 year+):**

  1. ✅ Build HT computing system in laboratory/company
  2. ✅ Publish proprietary database
  3. ✅ Be recognized as an MI field expert

\---

## Related Series

  * ****: Materials property prediction using machine learning
  * **[MLP Introduction Series](<../mlp-introduction/>)** : Machine learning potentials
  * **[Materials Database Utilization Introduction](<../materials-databases-introduction/>)** : Complete Materials Project guide
  * **[GNN Introduction Series](<../../ML/gnn-introduction/>)** : Graph neural networks

\---

## Feedback and Support

### Author

This series was created under Dr. Yusuke Hashimoto at Tohoku University as part of the Materials Informatics Dojo project.

**Creation date** : October 17, 2025 **Version** : 1.0 

### Feedback

  * **Typos/technical errors** : Report via GitHub repository Issues
  * **Improvement suggestions** : New topics, additional code examples, etc.
  * **Questions** : Difficult sections, areas needing additional explanation

**Contact** : yusuke.hashimoto.b8@tohoku.ac.jp 

\---

## License

This series is published under the **CC BY 4.0** (Creative Commons Attribution 4.0 International) license.

**Permitted:**

  * ✅ Free viewing and downloading
  * ✅ Educational use (classes, study groups, etc.)
  * ✅ Modification and derivative works (translation, summarization, etc.)

**Conditions:**

  * 📌 Author credit required
  * 📌 Modifications must be clearly indicated

\---

## Let's Get Started!

Are you ready? Start with Chapter 1 and begin your journey into the world of High-Throughput Computing!

**[Chapter 1: The Need for High-Throughput Computing and Workflow Design →](<chapter-1.html>)**

\---

**Update History**

  * **2025-10-17** : v1.0 initial release

\---

**Materials discovery acceleration starts here!**
