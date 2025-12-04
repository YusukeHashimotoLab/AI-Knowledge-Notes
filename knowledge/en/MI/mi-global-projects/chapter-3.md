---
title: "Chapter 3: European MI Projects"
chapter_title: "Chapter 3: European MI Projects"
subtitle: Continental Collaboration for Data-Driven Materials Science
reading_time: 25-30 minutes
difficulty: Beginner to Intermediate
---

This chapter provides a comprehensive overview of European Materials Informatics initiatives, highlighting the unique strengths of pan-European collaboration, open science infrastructure, and strategic investments in data-driven materials research across the continent.

**Learning Objectives**

  * Understand the structure and impact of NOMAD as the largest computational materials science repository
  * Learn about FAIRmat and Germany's National Research Data Infrastructure approach
  * Explore the Battery 2030+ initiative and European battery innovation ecosystem
  * Recognize the role of the Henry Royce Institute in UK materials research
  * Identify collaboration opportunities within European MI programs

## 3.1 NOMAD (Novel Materials Discovery)

The **NOMAD (Novel Materials Discovery)** project represents Europe's flagship initiative in computational materials science data infrastructure. Originally launched under Horizon 2020 and continuing through Horizon Europe, NOMAD has built the world's largest repository for computational materials science data.

### 3.1.1 Project Overview

Attribute | Details  
---|---  
**Region** | European Union (Germany-led)  
**Funding Programs** | Horizon 2020 (2015-2019), Horizon Europe (2021-present)  
**Founded** | 2015  
**Primary Goal** | Largest computational materials science data repository  
**Lead Institution** | Fritz Haber Institute of the Max Planck Society  
**Status** | Active  
**Official Website** | <https://nomad-lab.eu/>  
  
### 3.1.2 Key Features and Capabilities

NOMAD provides an integrated ecosystem for computational materials science:

#### Big-Data Services

  * **Data Repository** : Over 12 million calculations from major DFT codes
  * **Standardized Formats** : Common metadata schema across different simulation codes
  * **Open Access** : Free access to all archived calculations and workflows
  * **DOI Assignment** : Permanent identifiers for reproducibility and citation

#### AI Workflow Engines

  * **NOMAD Oasis** : Local installation for institutional data management
  * **AI Toolkit** : Pre-built machine learning models for property prediction
  * **Active Learning** : Automated experimental design for high-throughput screening
  * **Transfer Learning** : Pre-trained models for rapid adaptation to new materials systems

#### Exascale Computing Integration

  * **EuroHPC Partnership** : Integration with European supercomputing resources
  * **Scalable Workflows** : Designed for petascale and exascale computing environments
  * **Cloud Infrastructure** : Elastic computing resources for varying workloads

### 3.1.3 Organizational Structure
    
    
    ```mermaid
    graph TD
        A[NOMAD ProjectCoordinator: Matthias Scheffler] --> B[Data RepositoryFritz Haber Institute]
    
        B --> C[NOMAD CoECenter of Excellence]
        B --> D[NOMAD OasisLocal Installations]
        B --> E[AI/ML ServicesPrediction Models]
    
        C --> C1[8 Research Groups]
        C --> C2[4 HPC Centers]
    
        D --> D1[University Nodes]
        D --> D2[Industry Partners]
    
        E --> E1[Property Prediction]
        E --> E2[Materials Discovery]
    
        style A fill:#667eea,color:#fff
        style B fill:#764ba2,color:#fff
        style C fill:#4caf50,color:#fff
    ```

### 3.1.4 Participating Organizations

**Research Groups (8 core partners):**

  * Fritz Haber Institute, Max Planck Society (Germany) - Coordination
  * Humboldt-Universitat zu Berlin (Germany)
  * King's College London (UK)
  * Barcelona Supercomputing Center (Spain)
  * Aalto University (Finland)
  * EPFL (Switzerland)
  * Technical University of Denmark
  * University of Cambridge (UK)

**HPC Centers (4 partners):**

  * SURF (Netherlands)
  * Max Planck Computing and Data Facility (Germany)
  * CSC - IT Center for Science (Finland)
  * CINECA (Italy)

### 3.1.5 Use Cases and Applications

**NOMAD Discovery Examples**

  * **Catalytic Water Splitting** : ML-guided screening identified 50+ novel photocatalyst candidates for hydrogen production
  * **Thermoelectrics** : High-throughput calculations discovered compounds with ZT > 2.0 at room temperature
  * **Perovskites** : Stability predictions for 10,000+ perovskite compositions for solar cells
  * **2D Materials** : Database of 2,000+ exfoliable 2D materials with predicted properties

### 3.1.6 Data Statistics

Metric | Value  
---|---  
Total Calculations | 12+ million  
Unique Materials | 3+ million  
Supported DFT Codes | 50+  
Registered Users | 100,000+  
Data Volume | 100+ TB  
Annual Downloads | 10+ million files  
  
* * *

## 3.2 FAIRmat (Germany)

**FAIRmat** is Germany's flagship initiative for establishing FAIR (Findable, Accessible, Interoperable, Reusable) data infrastructure for condensed-matter physics and materials science, operating as part of the National Research Data Infrastructure (NFDI).

### 3.2.1 Program Overview

Attribute | Details  
---|---  
**Full Name** | FAIR Data Infrastructure for Condensed-Matter Physics and the Chemical Physics of Solids  
**Parent Program** | NFDI (National Research Data Infrastructure)  
**NFDI Total Budget** | Up to 90 million EUR/year (all consortia)  
**Duration** | 2019-2028 (10 years)  
**Lead Institution** | Humboldt-Universitat zu Berlin  
**Director** | Prof. Claudia Draxl  
**Official Website** | <https://fairmat-nfdi.eu/>  
  
### 3.2.2 NFDI Context

FAIRmat operates within Germany's broader National Research Data Infrastructure initiative:

  * **NFDI Mission** : Systematic management of scientific data across all disciplines
  * **Consortium Model** : 30+ discipline-specific consortia share common infrastructure
  * **Funding Structure** : Joint federal-state funding (DFG coordination)
  * **Integration** : Cross-consortium data sharing and tool development

### 3.2.3 Participating Institutions

FAIRmat brings together a broad coalition of German materials science institutions:

**FAIRmat Network**

  * **Principal Investigators** : 60 PIs from 34 German institutions
  * **Major Universities** : Humboldt, TU Berlin, RWTH Aachen, TU Munich, FAU Erlangen
  * **Max Planck Institutes** : Fritz Haber Institute, MPI for Iron Research, MPI for Polymer Research
  * **Helmholtz Centers** : Forschungszentrum Julich, DESY, Helmholtz-Zentrum Berlin
  * **Fraunhofer Institutes** : Multiple materials-focused Fraunhofer institutes

### 3.2.4 International Collaborations

FAIRmat has established formal partnerships with leading international institutions:

Partner | Country | Focus Area  
---|---|---  
NIST (MGI) | USA | Data standards, interoperability  
Shanghai University | China | Computational materials database exchange  
NIMS | Japan | Materials database integration  
Materials Project | USA | DFT data harmonization  
  
### 3.2.5 Key Infrastructure Components

  * **NOMAD Integration** : FAIRmat extends and enhances NOMAD repository capabilities
  * **Experimental Data** : Standardized schemas for X-ray, neutron, and electron microscopy data
  * **Electronic Lab Notebooks** : Integration with ELN systems for seamless data capture
  * **Ontologies** : Domain-specific vocabularies for materials science
  * **Training Programs** : Workshops and courses on FAIR data practices

* * *

## 3.3 Battery 2030+ / BATT4EU

The **Battery 2030+** initiative represents Europe's long-term research strategy for next-generation batteries, while **BATT4EU** serves as the institutionalized public-private partnership driving European battery innovation.

### 3.3.1 Initiative Overview

Attribute | Details  
---|---  
**Total Budget** | 925 million EUR (public-private partnership)  
**Horizon Europe Funding** | 518 million EUR through 34 calls (2021-2024)  
**Recent Investment** | 1 billion EUR call for battery manufacturing (December 2024)  
**Focus** | Next-generation battery technologies, digital passport  
**Partnership** | BATT4EU (Batteries European Partnership)  
**Official Website** | <https://battery2030.eu/>  
  
### 3.3.2 Key Research Projects

#### BIG-MAP (Battery Interface Genome - Materials Acceleration Platform)

  * **Budget** : 20 million EUR
  * **Duration** : 2020-2024
  * **Focus** : AI-driven discovery of battery materials and interfaces
  * **Key Output** : Autonomous experimentation platform for battery research
  * **Partners** : 34 institutions from 15 countries

#### INSTABAT (Innovative Sensors for Battery Applications)

  * **Focus** : In-situ sensing technologies for battery health monitoring
  * **MI Component** : Machine learning for state-of-health prediction
  * **Application** : Electric vehicle battery management systems

#### SENSIBAT (Smart Sensors for Battery Management)

  * **Focus** : Advanced sensing solutions for cell-level monitoring
  * **Technology** : Embedded sensors with edge computing
  * **Data Integration** : Real-time data feeding into digital twin models

#### SPARTACUS (Spatially Resolved Acoustic, Thermal and Electrochemical Operando Study)

  * **Focus** : Multi-modal operando characterization methods
  * **Innovation** : Combined acoustic, thermal, and electrochemical monitoring
  * **Data Science** : Multi-physics data fusion and analysis

### 3.3.3 European Battery Funding Timeline
    
    
    ```mermaid
    timeline
        title European Battery Initiative Timeline
        section 2018-2019
            2018 : EU Battery Alliance launched
                 : Strategic Action Plan adopted
            2019 : Battery 2030+ roadmap published
        section 2020-2021
            2020 : BIG-MAP project starts
                 : Horizon 2020 battery calls
            2021 : BATT4EU partnership established
                 : Horizon Europe funding begins
        section 2022-2024
            2022 : European Battery Passport initiative
                 : Manufacturing scale-up funding
            2023 : 518M EUR through 34 Horizon calls
            2024 : 1B EUR manufacturing call announced
                 : BIG-MAP Phase 2 planning
    ```

### 3.3.4 Digital Battery Passport

A key innovation from Battery 2030+ is the Digital Battery Passport:

**Battery Passport Features**

  * **Lifecycle Tracking** : Complete history from manufacturing to recycling
  * **Material Composition** : Detailed material and chemistry information
  * **Performance Data** : Capacity, efficiency, and degradation metrics
  * **Carbon Footprint** : Environmental impact assessment
  * **Circular Economy** : Second-life and recycling information

* * *

## 3.4 Henry Royce Institute (UK)

The **Henry Royce Institute** is the UK's national institute for advanced materials research and innovation, named after the pioneering engineer Sir Henry Royce. It represents the UK's largest investment in materials research infrastructure.

### 3.4.1 Institute Overview

Attribute | Details  
---|---  
**Founded** | 2014 (announced), 2019 (Hub opened)  
**Total Budget** | 250+ million GBP  
**Hub Location** | University of Manchester  
**Partner Universities** | 9 universities  
**Research Focus** | Advanced materials for clean growth  
**Official Website** | <https://www.royce.ac.uk/>  
  
### 3.4.2 Partner Network

The Henry Royce Institute operates as a distributed network across nine UK universities:

  * **University of Manchester** \- National Hub and Coordination
  * **University of Cambridge** \- Semiconductors, Energy Materials
  * **University of Oxford** \- Functional Materials, Devices
  * **Imperial College London** \- Structural Materials, Manufacturing
  * **University of Leeds** \- Polymers, Soft Materials
  * **University of Liverpool** \- Materials Chemistry
  * **University of Sheffield** \- Nuclear Materials, Metallurgy
  * **National Nuclear Laboratory** \- Nuclear Materials
  * **UKAEA** \- Fusion Materials

### 3.4.3 Research Themes

The Royce Institute organizes research around key societal challenges:

#### Low Carbon Power

  * Battery materials and manufacturing
  * Hydrogen storage and fuel cells
  * Photovoltaic materials
  * Nuclear materials (fission and fusion)

#### Circular Economy

  * Materials recycling and recovery
  * Sustainable materials design
  * Lifecycle assessment integration
  * Bio-based and biodegradable materials

#### Digital Materials

  * Materials Informatics platforms
  * Digital twins for materials processing
  * AI-guided materials discovery
  * Automated characterization and analysis

### 3.4.4 Materials 4.0 Centre for Doctoral Training

A flagship educational program within the Royce ecosystem:

**Materials 4.0 CDT**

  * **Focus** : Integration of digital technologies with materials science
  * **Duration** : 4-year PhD program
  * **Training** : Data science, machine learning, computational materials
  * **Industry Engagement** : Mandatory industry placements
  * **Cohort Model** : Collaborative learning across partner universities

### 3.4.5 Equipment and Capabilities

The Royce Institute provides world-class research infrastructure:

  * **Advanced Electron Microscopy** : Aberration-corrected TEM/STEM facilities
  * **Atom Probe Tomography** : 3D atomic-scale characterization
  * **Synchrotron Access** : Diamond Light Source partnership
  * **Processing Equipment** : Pilot-scale manufacturing facilities
  * **Computing Resources** : HPC clusters for simulation and MI

* * *

## 3.5 Other European Initiatives

Beyond the major programs described above, several other European initiatives contribute significantly to the MI landscape.

### 3.5.1 Empa (Switzerland)

**Empa - Swiss Federal Laboratories for Materials Science and Technology** operates as Switzerland's premier materials research institution:

Attribute | Details  
---|---  
**Location** | Dubendorf, St. Gallen, Thun (Switzerland)  
**Status** | ETH Domain research institute  
**Staff** | 1,000+ researchers  
**Focus** | Sustainable materials, nanotechnology, energy  
**MI Activities** | Digital labs, autonomous experimentation  
  
**Key MI Programs at Empa:**

  * Materials Data Science and Informatics group
  * Self-driving laboratories for materials synthesis
  * Digital twin development for manufacturing processes
  * Integration with Swiss Data Science Center (SDSC)

### 3.5.2 Paul Scherrer Institute (PSI, Switzerland)

**PSI** is Switzerland's largest research institute, hosting unique large-scale research facilities:

  * **Swiss Light Source (SLS)** : Synchrotron for materials characterization
  * **SINQ** : Continuous neutron source for structural studies
  * **SwissFEL** : X-ray free-electron laser for ultrafast dynamics
  * **Data Science Focus** : Automated analysis of large-scale facility data

### 3.5.3 EOSC-Pillar (European Open Science Cloud)

The **European Open Science Cloud** provides overarching data infrastructure supporting materials science:

  * **Mission** : Federated research data infrastructure for Europe
  * **Materials Connection** : Integration of NOMAD, FAIRmat with broader EOSC ecosystem
  * **Services** : Data storage, computing resources, analysis platforms
  * **Standards** : FAIR principles implementation across disciplines

### 3.5.4 Max Planck Institutes (Germany)

Several Max Planck Institutes contribute significantly to European MI:

  * **Fritz Haber Institute** : NOMAD coordination, computational catalysis
  * **MPI for Iron Research** : Computational metallurgy, alloy design
  * **MPI for Polymer Research** : Soft matter informatics
  * **MPI for Solid State Research** : Electronic structure calculations

* * *

## 3.6 Comparison of European MI Projects

The following table summarizes key characteristics of major European Materials Informatics initiatives:

Project | Country/Region | Budget | Focus | Key Output | Status  
---|---|---|---|---|---  
**NOMAD** | EU (Germany-led) | Horizon funding | Data repository | 12M+ calculations | Active  
**FAIRmat** | Germany | ~90M EUR/year (NFDI) | FAIR infrastructure | Standards, ontologies | Active  
**Battery 2030+** | EU | 925M EUR | Battery innovation | Digital passport | Active  
**Henry Royce** | UK | 250M+ GBP | Advanced materials | Infrastructure, CDT | Active  
**Empa** | Switzerland | Institutional | Sustainable materials | Digital labs | Active  
**PSI** | Switzerland | Institutional | Large facilities | Automated analysis | Active  
  
* * *

## 3.7 European Funding Landscape

Understanding the European funding ecosystem is essential for engaging with EU MI initiatives:

### 3.7.1 Horizon Europe Framework

**Horizon Europe Materials Funding**

  * **Total Budget** : 95.5 billion EUR (2021-2027)
  * **Cluster 4** : Digital, Industry and Space - includes materials science
  * **Key Instruments** : Research and Innovation Actions (RIA), Innovation Actions (IA)
  * **Partnerships** : BATT4EU, Clean Hydrogen, Clean Steel

### 3.7.2 NOMAD and FAIRmat Funded Projects (2015-2024)

The NOMAD Laboratory and FAIRmat consortium have received significant funding from European and German sources:

Year | Project/Grant | Funding Source | Lead Institution | Budget | Focus  
---|---|---|---|---|---  
2015| NOMAD Repository Development| Einstein Foundation| FHI Berlin| EUR 2.0M| Data Infrastructure  
2017| TEC1P (ERC Advanced Grant)| ERC| FHI/HU Berlin| EUR 2.5M| Theory-Experiment Coupling  
2018| NOMAD CoE Phase 1| H2020| MPCDF Garching| EUR 5.0M| Center of Excellence  
2020| NOMAD CoE Phase 2| H2020 (951786)| MPCDF/FHI| EUR 8.0M| Expanded Repository  
2021| FAIRmat NFDI| DFG (460197019)| HU Berlin| EUR 15M (5yr)| FAIR Data Ecosystem  
2022| FAIRmat Extension| DFG| 34 Institutions| EUR 8.0M| Experimental Data  
2023| NOMAD Oasis Network| BMBF| FHI Berlin| EUR 3.0M| Distributed Infrastructure  
  
### 3.7.3 Horizon 2020 / Horizon Europe Materials Projects (2015-2024)

The European Union's framework programs have funded numerous large-scale materials informatics initiatives:

Year | Project Name | Coordinator | Countries | Budget | Focus  
---|---|---|---|---|---  
2015| MARVEL NCCR| EPFL (Switzerland)| CH| CHF 50M| Computational Materials  
2017| INTERSECT| CNR (Italy)| IT, DE, CH| EUR 8.0M| Interoperable Simulations  
2018| EMMC-CSA| TU Vienna| AT, DE, FR, UK| EUR 4.0M| Modeling Standards  
2019| OpenModel| Fraunhofer IWM| DE, NL, IT| EUR 6.0M| Ontology Development  
2020| BIG-MAP| DTU (Denmark)| 15 countries| EUR 67M| Battery Materials  
2020| Battery 2030+| Uppsala (Sweden)| SE, DE, FR| EUR 32M| Smart Batteries  
2021| DOME 4.0| Fraunhofer| DE, FR, ES| EUR 8.0M| Digital Materials  
2022| DigiPass| VTT (Finland)| FI, DE, NL| EUR 5.0M| Materials Passport  
2022| AI4MAT| KIT (Germany)| DE, FR, IT| EUR 4.5M| AI for Manufacturing  
2023| MatCHMaker| SINTEF (Norway)| NO, DE, UK| EUR 6.0M| Sustainable Materials  
2023| DigiBatt| CEA (France)| FR, DE, BE| EUR 8.0M| Digital Battery Twin  
2024| SUNRISE| TNO (Netherlands)| NL, DE, IT| EUR 7.0M| Solar Materials  
  
### 3.7.4 UK EPSRC Materials Informatics Funding (2015-2024)

The UK Engineering and Physical Sciences Research Council has made substantial investments in materials informatics:

Year | Program/Grant | Lead Institution | Budget | Focus  
---|---|---|---|---  
2015| Henry Royce Institute Establishment| Manchester| GBP 235M| National Facility  
2017| Royce Hub Buildings| 9 Universities| GBP 150M| Infrastructure  
2019| Digital Manufacturing Programme| Sheffield| GBP 12M| Industry 4.0  
2020| AI for Materials Discovery| Cambridge| GBP 8.0M| ML Methods  
2021| Autonomous Materials Discovery| Liverpool| GBP 6.5M| Robotic Labs  
2022| Materials Data Hub| Imperial| GBP 4.0M| Data Infrastructure  
2024| Materials 4.0 CDT| Strathclyde (lead)| GBP 16.5M| Doctoral Training  
2024| Royce Phase 2| Manchester| GBP 95M| Expansion  
  
**Materials 4.0 CDT Highlights**

  * **Partner Universities** : Strathclyde, Leeds, Manchester, Sheffield, Cambridge, Oxford, Imperial
  * **Training Output** : 70+ PhD students over 4 years
  * **Focus Areas** : AI/ML, Manufacturing Informatics, Lifecycle Simulation
  * **Industry Partners** : 50+ companies engaged

### 3.7.5 National Funding Streams

  * **Germany** : DFG (German Research Foundation), BMBF (Federal Ministry of Education and Research)
  * **UK** : UKRI (UK Research and Innovation), EPSRC (Engineering and Physical Sciences Research Council)
  * **France** : ANR (French National Research Agency), CNRS
  * **Switzerland** : SNF (Swiss National Science Foundation), Innosuisse

### 3.7.6 Funding Timeline Overview
    
    
    ```mermaid
    graph LR
        subgraph EU_Level
            A[Horizon Europe95.5B EUR] --> B[Cluster 4Digital/Industry]
            A --> C[PartnershipsBATT4EU, etc.]
            A --> D[ERC GrantsFrontier Research]
        end
    
        subgraph National_Level
            E[GermanyNFDI, DFG] --> F[FAIRmatNOMAD]
            G[UKUKRI, EPSRC] --> H[Henry Royce]
            I[SwitzerlandSNF, ETH] --> J[Empa, PSI]
        end
    
        B --> F
        B --> H
        C --> K[Battery Projects]
    
        style A fill:#667eea,color:#fff
        style E fill:#764ba2,color:#fff
        style G fill:#764ba2,color:#fff
        style I fill:#764ba2,color:#fff
    ```

* * *

## 3.8 Timeline of European MI Development
    
    
    ```mermaid
    timeline
        title European MI Initiative Timeline
        section 2014-2016
            2014 : Henry Royce announced
                 : NOMAD planning begins
            2015 : NOMAD project launches
                 : Horizon 2020 materials calls
        section 2017-2019
            2017 : FAIRmat planning starts
            2018 : EU Battery Alliance formed
                 : Battery 2030+ roadmap
            2019 : Royce Hub opens Manchester
                 : NFDI program begins Germany
        section 2020-2022
            2020 : BIG-MAP project launches
                 : COVID accelerates digital methods
            2021 : BATT4EU partnership established
                 : Horizon Europe starts
            2022 : Battery Passport initiative
                 : NOMAD 12M+ calculations
        section 2023-Present
            2023 : FAIRmat Phase 2
                 : Royce Materials 4.0 CDT
            2024 : 1B EUR battery manufacturing call
                 : Exascale MI computing
    ```

* * *

## 3.9 European MI Ecosystem Structure
    
    
    ```mermaid
    graph TB
        subgraph EU_Institutions
            A[European Commission]
            B[ERC]
            C[EuroHPC]
        end
    
        subgraph Funding_Programs
            D[Horizon Europe]
            E[NFDI Germany]
            F[UKRI UK]
        end
    
        subgraph Research_Networks
            G[NOMAD]
            H[FAIRmat]
            I[Battery 2030+]
        end
    
        subgraph National_Institutes
            J[Henry Royce UK]
            K[Max Planck DE]
            L[Empa/PSI CH]
        end
    
        subgraph Industry
            M[BATT4EUIndustry Partners]
        end
    
        A --> D
        B --> D
        C --> G
    
        D --> G
        D --> I
        E --> H
        F --> J
    
        G --> K
        H --> K
        I --> M
        J --> M
    
        style A fill:#667eea,color:#fff
        style D fill:#764ba2,color:#fff
        style G fill:#4caf50,color:#fff
        style M fill:#ff9800,color:#fff
    ```

* * *

## 3.10 Chapter Summary

Europe has developed a distinctive approach to Materials Informatics characterized by large-scale collaboration, open science principles, and strong integration between academic and industrial partners.

### Key Takeaways

  * **NOMAD (2015-present)** : World's largest computational materials science repository with 12+ million calculations; exemplifies European commitment to open data
  * **FAIRmat (2019-2028)** : Germany's 10-year investment in FAIR data infrastructure connects 60 PIs from 34 institutions with international MoUs
  * **Battery 2030+ / BATT4EU** : 925 million EUR public-private partnership driving next-generation battery innovation with Digital Battery Passport
  * **Henry Royce Institute** : UK's 250+ million GBP investment in advanced materials infrastructure across 9 partner universities
  * **Swiss Excellence** : Empa and PSI contribute world-class facilities and digital lab capabilities

**European MI Strengths**

  * **Open Science Culture** : Strong commitment to FAIR principles and open data repositories
  * **Pan-European Collaboration** : Cross-border partnerships leverage diverse expertise
  * **Industry Integration** : Public-private partnerships (e.g., BATT4EU) bridge academia and industry
  * **Infrastructure Investment** : World-class facilities for characterization and computation
  * **Standardization Focus** : Leadership in data formats, ontologies, and interoperability

### Practical Recommendations

  * **For Researchers** : Deposit data in NOMAD, apply for Horizon Europe funding, engage with EOSC infrastructure
  * **For Industry** : Join BATT4EU or sector-specific partnerships, access Royce facilities for collaborative research
  * **For Students** : Consider Materials 4.0 CDT or similar programs, build skills in both computation and experiment
  * **For International Collaborators** : Establish MoUs with FAIRmat, contribute to NOMAD, participate in Horizon Europe calls

## Exercises

Exercise 1: Data Repository Comparison Easy

**Question:** Compare NOMAD (Europe) with the Materials Project (US) in terms of data types, access policies, and primary use cases. What are the complementary strengths of each repository?

Exercise 2: Funding Strategy Medium

**Question:** You are leading a research group in Germany focused on solid-state batteries. Design a funding strategy that leverages both German national programs (NFDI, DFG) and European programs (Horizon Europe, BATT4EU). Which specific instruments would you target and in what sequence?

Exercise 3: FAIR Data Implementation Medium

**Question:** Your laboratory generates X-ray diffraction and electron microscopy data for 100 new materials per year. Develop a data management plan that ensures compliance with FAIRmat standards. What metadata schemas would you use, and how would you integrate with NOMAD?

Exercise 4: International Collaboration Hard

**Question:** Compare the European (NOMAD/FAIRmat), US (MGI/Materials Project), and Japanese (MI2I/NIMS) approaches to materials data infrastructure. Identify three opportunities for improved international interoperability and propose concrete technical solutions.

## References

  1. Draxl, C., Scheffler, M. (2019). The NOMAD laboratory: from data sharing to artificial intelligence. _Journal of Physics: Materials_ , 2(3), 036001.
  2. Scheffler, M., et al. (2022). FAIR data enabling new horizons for materials research. _Nature_ , 604, 635-642.
  3. Battery 2030+ Roadmap. (2020). Inventing the sustainable batteries of the future. European Commission.
  4. FAIRmat Consortium. (2024). FAIR Data Infrastructure for Condensed-Matter Physics - Annual Report 2023.
  5. Henry Royce Institute. (2023). Strategic Plan 2023-2028: Advanced Materials for Clean Growth.
  6. BATT4EU. (2024). Batteries European Partnership - Strategic Research and Innovation Agenda.
  7. Himanen, L., et al. (2019). Data-driven materials science: status, challenges, and perspectives. _Advanced Science_ , 6(21), 1900808.

[Chapter 2: Japan MI Projects](<chapter-2.html>) [Chapter 4: Asia-Pacific MI Projects](<chapter-4.html>)
