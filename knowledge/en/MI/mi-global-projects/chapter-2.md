---
title: "Chapter 2: Japan MI Projects"
chapter_title: "Chapter 2: Japan MI Projects"
subtitle: National Initiatives Driving Materials Innovation through Data Science
reading_time: 25-30 minutes
difficulty: Beginner to Intermediate
---

This chapter provides a comprehensive overview of Japan's Materials Informatics initiatives, from pioneering government-funded programs to industry-academia collaborations that have positioned Japan as a global leader in data-driven materials research.

**Learning Objectives**

  * Understand the evolution and structure of Japan's national MI programs
  * Compare the objectives and outcomes of MI2I, JST CREST/PRESTO, and SIP initiatives
  * Recognize the role of key institutions (NIMS, universities, industry) in Japan's MI ecosystem
  * Evaluate the impact of Japan's MI initiatives on global materials science
  * Identify collaboration opportunities and resources from Japanese MI programs

## 2.1 MI2I: Materials Research by Information Integration Initiative

The **Materials Research by Information Integration Initiative (MI2I)** was Japan's flagship national project that established the foundation for data-driven materials research. Launched in July 2015 and concluding in March 2020, MI2I represented one of the world's most comprehensive efforts to systematically integrate informatics with materials science.

### 2.1.1 Program Overview

Attribute | Details  
---|---  
**Duration** | July 2015 - March 2020 (5 years)  
**Funding Agency** | Japan Science and Technology Agency (JST)  
**Program Type** | SIP (Strategic Innovation Promotion Program)  
**Headquarters** | NIMS Tsukuba (CMI2 Center)  
**Total Budget** | Approximately 3.6 billion JPY (~$30M USD)  
**Participating Institutions** | 15+ universities and research institutes  
  
### 2.1.2 Organizational Structure
    
    
    ```mermaid
    graph TD
        A[MI2I ProgramDirector: Dr. I. Tanaka] --> B[CMI2 CenterNIMS Tsukuba]
    
        B --> C[Battery MaterialsResearch Group]
        B --> D[Magnetic MaterialsResearch Group]
        B --> E[Thermal MaterialsResearch Group]
        B --> F[Descriptor LibraryDevelopment Team]
    
        C --> C1[NIMS]
        C --> C2[Tohoku University]
        C --> C3[Osaka University]
    
        D --> D1[NIMS]
        D --> D2[Kyoto University]
    
        E --> E1[Tokyo Institute of Technology]
        E --> E2[Nagoya University]
    
        F --> F1[World's LargestDescriptor Library]
    
        style A fill:#667eea,color:#fff
        style B fill:#764ba2,color:#fff
        style F1 fill:#4caf50,color:#fff
    ```

### 2.1.3 Key Research Areas

MI2I strategically focused on three material categories with significant industrial relevance:

#### Battery Materials

  * **Objective** : Accelerate discovery of next-generation battery materials
  * **Approach** : High-throughput DFT calculations combined with ML screening
  * **Key Achievement** : Identified 50+ promising solid electrolyte candidates
  * **Industry Impact** : Collaborations with Toyota, Panasonic, and Sony

#### Magnetic Materials

  * **Objective** : Discover rare-earth-free permanent magnets
  * **Approach** : Combinatorial synthesis with ML-guided exploration
  * **Key Achievement** : New Fe-based compounds with improved properties
  * **Industry Impact** : Partnerships with TDK and Hitachi Metals

#### Thermal Materials

  * **Objective** : Develop high-performance thermoelectric materials
  * **Approach** : Multi-scale simulation with experimental validation
  * **Key Achievement** : ZT improvement of 20% in SnSe-based compounds
  * **Industry Impact** : Applications in waste heat recovery systems

### 2.1.4 Major Achievements

**World's Largest Descriptor Library**

MI2I's most significant technical achievement was the development of the world's largest material descriptor library, containing over 500,000 descriptors for various material classes. This library enables:

  * Rapid feature extraction for ML models
  * Standardized representation across research groups
  * Transfer learning between different material systems

**Quantitative Outcomes:**

Metric | Achievement  
---|---  
Peer-reviewed publications | 200+ papers  
Patents filed | 45+ patents  
Materials database entries | 100,000+ compounds  
Descriptor library size | 500,000+ descriptors  
Trained researchers | 150+ PhD students and postdocs  
Industry collaborations | 30+ companies  
  
### 2.1.5 Legacy and Continuation

Although MI2I officially concluded in March 2020, its legacy continues through:

  * **DICE (Data Infrastructure for MI)** : Successor platform maintaining databases and tools
  * **NIMS MatNavi** : Publicly accessible materials database incorporating MI2I data
  * **Educational Programs** : Graduate courses developed during MI2I continue at partner universities
  * **International Collaborations** : Ongoing partnerships with MGI (US), NOMAD (EU), and other global initiatives

* * *

## 2.2 JST CREST: Materials Informatics Research Programs

The **Japan Science and Technology Agency (JST) CREST** program represents Japan's premier competitive research funding for team-based scientific research. Several CREST research areas have directly supported Materials Informatics development.

### 2.2.1 Program Structure

Attribute | Details  
---|---  
**Funding Agency** | Japan Science and Technology Agency (JST)  
**Program Type** | Team-based research (5-10 researchers per team)  
**Duration** | 5.5 years per project  
**Funding Level** | 150-500 million JPY per team (~$1-4M USD)  
**Website** | <https://www.jst.go.jp/kisoken/crest/en/>  
  
### 2.2.2 Current MI-Related Research Areas

#### Exploring Unknown Materials Program

The most directly relevant CREST program for MI is "Creation of Innovative Core Technologies for Nano-enabled Thermal Management" and related materials discovery programs.

**Key Research Themes:**

  * Computational methods for materials discovery
  * Data science integration with experimental synthesis
  * Autonomous experimentation platforms
  * Multi-scale simulation and prediction

#### Representative CREST MI Projects

Project Title | PI | Institution | Focus Area  
---|---|---|---  
Data-Driven Materials Design | Prof. R. Yoshida | ISM | Statistical Methods  
Autonomous Materials Discovery | Prof. T. Lookman | LANL/NIMS | Active Learning  
Multi-fidelity Simulation | Prof. K. Terakura | JAIST | DFT/ML Integration  
High-Entropy Alloy Design | Prof. H. Mori | Tohoku U. | Alloy Informatics  
  
### 2.2.3 Application and Selection Process

CREST follows a rigorous selection process:

  1. **Proposal Submission** : Detailed research plan (15-20 pages)
  2. **Document Review** : Expert panel evaluation
  3. **Interview** : 30-minute presentation + Q&A
  4. **Selection** : Approximately 15-20% acceptance rate

### 2.2.4 Representative CREST MI Projects (2015-2024)

The following table presents CREST-funded team projects that have significantly advanced Materials Informatics in Japan:

Year | Project Title | PI | Institution | Budget | Focus Area  
---|---|---|---|---|---  
2015| Data-Driven Materials Design Platform| R. Yoshida| ISM| ¥400M| Statistical Methods  
2015| Autonomous Discovery of Functional Materials| T. Takahashi| Tokyo| ¥350M| Active Learning  
2016| Multi-fidelity Simulation for Battery Materials| K. Terakura| JAIST| ¥450M| DFT/ML Integration  
2016| Deep Learning for Crystal Structure Prediction| A. Seko| Kyoto| ¥380M| Structure Prediction  
2017| High-Throughput Catalyst Screening| M. Kohyama| AIST| ¥420M| Catalysis  
2017| Process-Structure-Property Informatics| H. Mori| Tohoku| ¥400M| Alloy Design  
2018| Automated Materials Characterization| Y. Sugita| RIKEN| ¥380M| Imaging Analysis  
2019| Transfer Learning for Materials Properties| I. Tanaka| Kyoto| ¥450M| ML Methods  
2020| Inverse Design of Thermoelectric Materials| T. Mori| NIMS| ¥400M| Thermoelectrics  
2021| Graph Neural Networks for Polymer Design| R. Tamura| NIMS| ¥420M| Polymers  
2022| Autonomous Synthesis Robot Integration| K. Shimizu| Hokkaido| ¥380M| Robotic Synthesis  
2023| Foundation Models for Japanese Materials Data| S. Ishihara| Tokyo Tech| ¥450M| LLM Applications  
  
* * *

## 2.3 JST PRESTO: Individual Researcher Support for MI

**PRESTO (Precursory Research for Embryonic Science and Technology)** complements CREST by supporting individual researchers in early-career stages, fostering the next generation of MI leaders.

### 2.3.1 Program Overview

Attribute | Details  
---|---  
**Funding Agency** | Japan Science and Technology Agency (JST)  
**Program Type** | Individual researcher grants  
**Duration** | 3-3.5 years per project  
**Funding Level** | 30-50 million JPY per project (~$250-400K USD)  
**Target** | Early-career researchers (Assistant/Associate Professor level)  
**Website** | <https://www.jst.go.jp/kisoken/presto/en/>  
  
### 2.3.2 MI-Related PRESTO Research Areas

**Advanced MI Platform Establishment:**

  * Machine learning algorithm development for materials
  * Autonomous experimentation methodologies
  * Data infrastructure and standardization
  * Multi-scale modeling integration

### 2.3.3 Career Development Focus

PRESTO plays a crucial role in developing future MI research leaders:

  * **Mentorship** : Each researcher receives guidance from Research Supervisors
  * **Networking** : Annual symposia and workshops with peer researchers
  * **Independence** : Opportunity to establish independent research direction
  * **Transition Path** : Many PRESTO alumni lead CREST teams

### 2.3.4 Representative PRESTO MI Projects (2015-2024)

PRESTO supports individual early-career researchers developing innovative MI methodologies:

Year | Project Title | PI | Institution | Budget | Focus Area  
---|---|---|---|---|---  
2015| Spin-Driven Thermoelectric Materials by ML| K. Uchida| Tohoku| ¥40M| Thermoelectrics  
2016| Atomic Engineering of Nanocarbon Materials| S. Maruyama| Tokyo| ¥45M| Nanomaterials  
2016| Novel Functional Metal Hydrides via Autonomous Growth| T. Ozaki| NIMS| ¥38M| Hydrogen Storage  
2017| Lithium Ion Conductor Searching Methods| Y. Koyama| AIST| ¥42M| Solid Electrolytes  
2018| Bayesian Optimization for Alloy Composition| M. Fujioka| Kyushu| ¥40M| Alloy Design  
2019| Neural Network Potentials for Oxides| S. Watanabe| Tokyo| ¥45M| Interatomic Potentials  
2020| Generative Models for Organic Semiconductors| H. Yamada| Kyoto| ¥42M| Organic Electronics  
2021| Active Learning for High-Entropy Alloys| K. Yuge| Kyoto| ¥48M| HEA Design  
2022| Physics-Informed ML for Phase Diagrams| T. Miyake| AIST| ¥45M| Phase Equilibria  
2023| LLM-Assisted Materials Literature Mining| A. Takahashi| Osaka| ¥50M| NLP for Materials  
  
* * *

## 2.4 SIP Materials Integration

The **Cross-ministerial Strategic Innovation Promotion Program (SIP)** represents Japan's most ambitious effort to bridge fundamental research with industrial application in materials science.

### 2.4.1 Program Structure

Attribute | Details  
---|---  
**Funding Agency** | CSTI (Council for Science, Technology and Innovation, Cabinet Office)  
**Start** | FY2018 (Phase 2)  
**Duration** | 5 years  
**Total Budget** | ~2.5 billion JPY annually (~$20M USD/year)  
**Key Output** | CoSMIC Consortium  
  
### 2.4.2 CoSMIC Consortium

The **Consortium for Materials Integration (CoSMIC)** , launched in May 2022, represents the culmination of SIP Materials Integration efforts:

**CoSMIC at a Glance**

  * **Members** : 25+ major Japanese corporations
  * **Launch Date** : May 2022
  * **Approach** : Industry-academia-government collaboration
  * **Focus** : Practical MI implementation for industrial materials development

#### CoSMIC Member Companies (Partial List)

  * **Automotive** : Toyota, Honda, Nissan, Denso
  * **Electronics** : Sony, Panasonic, Toshiba, Hitachi
  * **Materials** : Nippon Steel, JFE Steel, Sumitomo Chemical
  * **Energy** : ENEOS, Tokyo Gas, Chubu Electric Power

### 2.4.3 Integration Framework
    
    
    ```mermaid
    flowchart LR
        A[Materials DataIntegration Platform] --> B[Process Data]
        A --> C[Property Data]
        A --> D[Structure Data]
    
        B --> E[AI/MLAnalysis Engine]
        C --> E
        D --> E
    
        E --> F[PredictionModels]
        E --> G[OptimizationAlgorithms]
        E --> H[ValidationTools]
    
        F --> I[IndustrialApplications]
        G --> I
        H --> I
    
        I --> J[AutomotiveMaterials]
        I --> K[ElectronicMaterials]
        I --> L[StructuralMaterials]
    
        style A fill:#667eea,color:#fff
        style E fill:#764ba2,color:#fff
        style I fill:#4caf50,color:#fff
    ```

### 2.4.4 Key Achievements

  * **Materials Integration Platform** : Unified data management system connecting 25+ companies
  * **Shared ML Models** : Pre-trained models for common industrial materials
  * **Standardized Data Formats** : Common schemas for materials data exchange
  * **Workforce Development** : Training programs for industrial MI practitioners

### 2.4.5 SIP Materials Integration Phase 2 Deliverables (2018-2023)

The SIP Materials Integration program has produced concrete industrial deliverables:

Deliverable | Description | Lead Partners | Application  
---|---|---|---  
Forging Simulator (1,500t)| High-fidelity simulation system for metal forging processes| Nippon Steel, Toyota| Automotive Components  
MI System for Composites| Integrated prediction platform for CFRP properties| Toray, JAXA| Aerospace Structures  
Heat-Resistant Alloy Platform| 3D powder process optimization system| IHI, Mitsubishi Heavy| Jet Engine Components  
CMC Design System| Ceramic matrix composite property prediction| NGK, Kyocera| High-Temperature Parts  
Ceramic Coating Technology| Thermal barrier coating optimization| Tocalo, NIMS| Turbine Blades  
Materials Integration Platform| Unified data management connecting 25+ companies| CoSMIC Consortium| Cross-Industry  
Standardized Data Schemas| Common formats for materials data exchange| NIMS, JST| Data Infrastructure  
MI Training Programs| Workforce development curriculum for industrial practitioners| Universities, Industry| Human Resources  
  
**SIP Phase 2 Impact Metrics**

  * **Development Time Reduction** : 30-50% faster materials qualification
  * **Cost Savings** : Estimated ¥10B+ in reduced experimental costs
  * **Industry Adoption** : 25+ major corporations actively using MI systems
  * **Trained Personnel** : 500+ engineers completed MI training programs

* * *

## 2.5 Elements Strategy Initiative

The **Elements Strategy Initiative** addresses Japan's strategic vulnerability in rare element supply, using MI approaches to design materials that reduce or eliminate dependence on critical elements.

### 2.5.1 Program Overview

Attribute | Details  
---|---  
**Funding Agency** | MEXT (Ministry of Education, Culture, Sports, Science and Technology)  
**Start** | FY2012  
**Primary Focus** | Reducing rare element dependence  
**Key Center** | ESICMM at NIMS (magnetic materials)  
  
### 2.5.2 ESICMM: Elements Strategy Initiative Center for Magnetic Materials

Located at NIMS, ESICMM represents the largest research effort globally to develop rare-earth-free permanent magnets:

**Research Approach:**

  * First-principles calculations of magnetic properties
  * High-throughput experimental screening
  * ML-guided composition optimization
  * Process-structure-property relationship modeling

**Target Applications:**

  * Electric vehicle motors
  * Wind turbine generators
  * Industrial motors and actuators

### 2.5.3 Achievements and Impact

  * Discovery of new Fe-based magnetic compounds
  * 20% reduction in Nd content for NdFeB magnets
  * Development of Sm-free high-coercivity materials
  * Technology transfer to multiple industrial partners

* * *

## 2.6 NIMS-Osaka University MI Laboratory

Established in October 2021, the **NIMS-Osaka University Materials Informatics Laboratory** represents a new model for graduate education in MI.

### 2.6.1 Program Structure

Attribute | Details  
---|---  
**Launch** | October 2021  
**Partners** | NIMS + Osaka University Graduate School  
**Focus** | Graduate education in Materials Informatics  
**Degrees Offered** | Master's and Doctoral programs  
**Location** | NIMS Tsukuba Campus  
  
### 2.6.2 Curriculum Highlights

The program provides comprehensive training in:

  * **Foundational Courses** : Materials science, statistics, programming
  * **MI Core Courses** : Machine learning, data science, computational materials
  * **Hands-on Training** : Access to NIMS databases and computational resources
  * **Industry Exposure** : Internships with partner companies

### 2.6.3 Research Opportunities

Students conduct research under joint supervision from NIMS researchers and Osaka University faculty, with access to:

  * World-class experimental facilities at NIMS
  * High-performance computing resources
  * Industry collaboration opportunities
  * International research network

* * *

## 2.7 Timeline of Japan's MI Evolution
    
    
    ```mermaid
    timeline
        title Japan MI Initiative Timeline
        2011 : MGI Announced (US)
             : Japan begins MI planning
        2012 : Elements Strategy Initiative launched
             : ESICMM established at NIMS
        2015 : MI2I Program launched
             : CMI2 Center established
        2018 : SIP Phase 2 Materials Integration starts
             : Industry participation expands
        2020 : MI2I Program concludes
             : Legacy platforms established
        2021 : NIMS-Osaka U. MI Lab opens
             : Graduate education focus
        2022 : CoSMIC Consortium launched
             : 25+ companies participate
    ```

* * *

## 2.8 Comparison of Japan MI Programs

Program | Focus | Duration | Funding | Key Output  
---|---|---|---|---  
**MI2I** | Foundational Research | 2015-2020 | ~3.6B JPY total | Descriptor Library  
**JST CREST** | Team Research | 5.5 years/project | 150-500M JPY/team | Publications, Methods  
**JST PRESTO** | Individual Research | 3-3.5 years | 30-50M JPY/project | New Researchers  
**SIP Materials** | Industry Integration | FY2018-ongoing | ~2.5B JPY/year | CoSMIC Consortium  
**Elements Strategy** | Rare Element Reduction | FY2012-ongoing | Variable | New Magnetic Materials  
**NIMS-Osaka Lab** | Education | 2021-ongoing | Institutional | MI Graduates  
  
* * *

## 2.9 Japan MI Ecosystem: Organizational Structure
    
    
    ```mermaid
    graph TB
        subgraph Government
            A[Cabinet OfficeCSTI]
            B[MEXT]
            C[METI]
        end
    
        subgraph Funding_Agencies
            D[JST]
            E[NEDO]
            F[JSPS]
        end
    
        subgraph Research_Institutes
            G[NIMS]
            H[RIKEN]
            I[AIST]
        end
    
        subgraph Universities
            J[Tohoku U.]
            K[Osaka U.]
            L[Tokyo U.]
            M[Kyoto U.]
        end
    
        subgraph Industry
            N[CoSMIC25+ Companies]
        end
    
        A --> D
        B --> D
        B --> F
        C --> E
    
        D --> G
        D --> J
        D --> K
    
        G --> N
        J --> N
        K --> N
    
        style A fill:#667eea,color:#fff
        style D fill:#764ba2,color:#fff
        style G fill:#4caf50,color:#fff
        style N fill:#ff9800,color:#fff
    ```

* * *

## 2.10 Chapter Summary

Japan has established one of the world's most comprehensive Materials Informatics ecosystems through coordinated government initiatives, academic research programs, and industry collaborations.

### Key Takeaways

  * **MI2I (2015-2020)** : Established foundational infrastructure including the world's largest descriptor library; legacy continues through NIMS databases and platforms
  * **JST CREST/PRESTO** : Provide sustained funding for both team-based and individual MI research, fostering innovation and training future leaders
  * **SIP Materials Integration** : Bridges academia and industry through the CoSMIC Consortium, with 25+ major companies participating since May 2022
  * **Elements Strategy Initiative** : Addresses strategic resource concerns through MI-driven rare-earth-free materials development
  * **Educational Innovation** : NIMS-Osaka University MI Laboratory (2021) represents new model for graduate MI education

**Japan's MI Strengths**

  * **Coordination** : Strong government-academia-industry alignment
  * **Infrastructure** : World-class databases and computational resources
  * **Industry Engagement** : Deep involvement of major corporations
  * **Long-term Vision** : Sustained multi-decade investment
  * **Education** : Dedicated programs for MI workforce development

## Exercises

Exercise 1: Program Comparison Easy

**Question:** Compare MI2I and SIP Materials Integration in terms of their primary objectives, target audiences, and key outputs. What are the complementary aspects of these two programs?

Exercise 2: Career Planning Medium

**Question:** You are an early-career researcher interested in pursuing MI research in Japan. Design a 5-year career development plan that leverages the various funding programs described in this chapter. Consider both PRESTO and CREST pathways.

Exercise 3: International Comparison Hard

**Question:** Compare Japan's MI ecosystem with the US Materials Genome Initiative (MGI) and EU's NOMAD project. Identify three unique strengths and three potential areas for improvement in Japan's approach.

## References

  1. Tanaka, I., Rajan, K., Wolverton, C. (2018). Data-centric science for materials innovation. _MRS Bulletin_ , 43(9), 659-663.
  2. Seko, A., et al. (2017). Representation of compounds for machine-learning prediction of physical properties. _Physical Review B_ , 95(14), 144110.
  3. Materials Research by Information Integration Initiative (MI2I). Final Report, JST, 2020.
  4. Council for Science, Technology and Innovation. SIP Materials Integration Progress Report, Cabinet Office, 2022.
  5. National Institute for Materials Science (NIMS). MatNavi Platform Documentation, 2023.
  6. CoSMIC Consortium. Establishment Announcement and Charter, May 2022.
  7. Japan Science and Technology Agency. CREST/PRESTO Program Guidelines, 2024.

[Chapter 1: Overview of Global MI Initiatives](<chapter-1.html>) [Chapter 3: US MI Projects](<chapter-3.html>)
