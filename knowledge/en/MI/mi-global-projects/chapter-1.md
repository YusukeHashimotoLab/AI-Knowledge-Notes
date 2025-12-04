---
title: "Chapter 1: United States MI Projects"
chapter_title: "Chapter 1: United States MI Projects"
---

[AI Terakoya Top](<../index.html>)>[Materials Informatics](<../index.html>)>[MI Global Projects](<index.html>)>Chapter 1: United States

EN | [JP](<#> "Japanese version coming soon") | Last sync: 2025-11-21

# Chapter 1: United States MI Projects

The United States has been at the forefront of Materials Informatics (MI) development since the launch of the Materials Genome Initiative in 2011. This chapter provides a comprehensive overview of major US government-funded programs, national laboratory initiatives, and academic-industry collaborations that have shaped the global MI landscape.

**Learning Objectives**

  * Understand the structure and goals of the Materials Genome Initiative (MGI)
  * Learn about NSF's DMREF program and its funding mechanisms
  * Explore ARPA-E's high-risk energy materials programs
  * Understand the Materials Project database and its impact on the field
  * Learn about DOE national laboratory MI initiatives
  * Understand NIST's role in materials data infrastructure

## 1.1 Materials Genome Initiative (MGI)

The **Materials Genome Initiative (MGI)** is a multi-agency initiative launched by the White House Office of Science and Technology Policy (OSTP) in June 2011. It represents the most comprehensive government-led effort to accelerate materials discovery and deployment.

### 1.1.1 Initiative Overview

Attribute | Details  
---|---  
Launch Date | June 24, 2011  
Launched By | White House Office of Science and Technology Policy (OSTP)  
Cumulative Budget | Over $1 billion (2011-2024)  
Primary Goal | Reduce materials development time by 50% (from 20 years to 10 years)  
Status | Active  
Official Website | <https://www.mgi.gov/>  
  
### 1.1.2 Participating Agencies

MGI coordinates efforts across multiple federal agencies:

  * **National Science Foundation (NSF)** : Fundamental research and education
  * **Department of Energy (DOE)** : Energy materials and computational resources
  * **National Institute of Standards and Technology (NIST)** : Standards and data infrastructure
  * **Defense Advanced Research Projects Agency (DARPA)** : High-risk/high-reward projects
  * **Department of Defense (DoD)** : Defense-related materials
  * **National Aeronautics and Space Administration (NASA)** : Aerospace materials
  * **National Institutes of Health (NIH)** : Biomaterials

### 1.1.3 Key Outputs and Infrastructure

**MGI Key Deliverables**

  * **MGI Code Catalog** : Repository of computational tools and software for materials research
  * **Materials Data Repository** : Centralized storage for materials datasets with standardized formats
  * **Strategic Plans** : Triennial strategic plans guiding national MI priorities (2014, 2017, 2021)
  * **Interagency Coordination** : Regular coordination meetings and joint funding opportunities

### 1.1.4 MGI Strategic Framework
    
    
    ```mermaid
    graph TD
        A[MGI Vision] --> B[Computational Tools]
        A --> C[Experimental Tools]
        A --> D[Digital Data]
    
        B --> E[Open-Source Software]
        B --> F[High-Performance Computing]
    
        C --> G[High-Throughput Synthesis]
        C --> H[Automated Characterization]
    
        D --> I[Standardized Formats]
        D --> J[Interoperable Databases]
    
        E --> K[Accelerated Materials Discovery]
        F --> K
        G --> K
        H --> K
        I --> K
        J --> K
    
        K --> L[50% Reduction in Development Time]
    
        style A fill:#667eea,color:#fff
        style K fill:#764ba2,color:#fff
        style L fill:#4caf50,color:#fff
    ```

## 1.2 DMREF (NSF)

The **Designing Materials to Revolutionize and Engineer our Future (DMREF)** program is NSF's primary contribution to the Materials Genome Initiative, focusing on fundamental research that integrates theory, computation, and experiment.

### 1.2.1 Program Overview

Attribute | Details  
---|---  
Full Name | Designing Materials to Revolutionize and Engineer our Future  
Duration | 2012 - Present  
FY2024-2025 Budget | $72.5 million  
Typical Award Size | $1.5 - $2.0 million over 4 years  
Award Duration | 4 years (renewable)  
Official Website | <https://www.nsf.gov/funding/opportunities/dmref>  
  
### 1.2.2 Program Objectives

DMREF supports research that:

  * Develops new theoretical and computational approaches for materials design
  * Creates feedback loops between computation, synthesis, and characterization
  * Establishes materials design principles that can be broadly applied
  * Trains the next generation of materials researchers in integrated approaches

### 1.2.3 Research Focus Areas

**DMREF Priority Areas**

  * **Quantum Materials** : Superconductors, topological insulators, quantum sensors
  * **Energy Materials** : Batteries, photovoltaics, thermoelectrics
  * **Structural Materials** : High-entropy alloys, composites, ceramics
  * **Functional Materials** : Catalysts, membranes, sensors
  * **Soft Materials** : Polymers, gels, biological materials

### 1.2.4 Representative DMREF Funded Projects (2015-2024)

The following table presents representative DMREF-funded projects that exemplify the program's integrated computational-experimental approach to materials discovery:

Year | Project Title | Lead PI | Institution | Budget | Focus Area  
---|---|---|---|---|---  
2015| Integrated Framework for Design of Alloy-Oxide Thin Films| C. Wolverton| Northwestern| $1.6M| Thin Films  
2015| Accelerated Discovery of Sodium Ion Conductors| G. Ceder| MIT/LBNL| $1.8M| Battery Materials  
2016| Design of Multifunctional Perovskite Oxides| L.Q. Chen| Penn State| $1.7M| Functional Oxides  
2016| Predictive Design of High-Entropy Ceramics| K. Page| Oak Ridge NL| $1.5M| Ceramics  
2017| Machine Learning for Soft Materials Design| F. Escobedo| Cornell/UCSB| $1.9M| Polymers  
2017| Data-Driven Discovery of Thermoelectric Materials| E. Toberer| Colorado School of Mines| $1.6M| Thermoelectrics  
2018| Deep Learning for Metallic Glass Discovery| J. Schroers| Yale| $1.8M| Metallic Glasses  
2018| AI-Accelerated Design of Photocatalysts| J. Gregoire| Caltech| $2.0M| Catalysis  
2019| Active Learning for Autonomous Materials Discovery| R. Gomez-Bombarelli| MIT| $1.9M| ML Methods  
2019| Inverse Design of Quantum Materials| P. Narang| Harvard| $1.7M| Quantum Materials  
2020| High-Entropy Alloy Design via Multi-fidelity ML| A. Mishra| Georgia Tech| $1.8M| Alloys  
2020| Autonomous Synthesis of 2D Materials| E. Pop| Stanford| $2.0M| 2D Materials  
2021| Self-Driving Labs for Battery Electrolyte Discovery| J. Nanda| Michigan/ORNL| $1.9M| Batteries  
2021| Foundation Models for Materials Property Prediction| S. Ermon| Stanford| $2.0M| ML Foundation  
2022| Graph Neural Networks for Heterogeneous Catalysis| Z. Ulissi| CMU/Meta| $1.8M| Catalysis  
2022| Generative AI for Organic Semiconductor Design| A. Aspuru-Guzik| Toronto/Harvard| $1.7M| Organic Electronics  
2023| Large Language Models for Materials Synthesis| E. Olivetti| MIT| $2.0M| Synthesis Planning  
2023| Multi-Agent Systems for Autonomous Labs| C. Coley| MIT| $1.9M| Autonomous Labs  
2024| Physics-Informed Neural Networks for Alloy Design| W. Curtin| EPFL/Brown| $1.8M| Structural Alloys  
2024| Closed-Loop Polymer Discovery Platform| R. Ramprasad| Georgia Tech| $2.0M| Polymers  
  
**DMREF Program Statistics (2012-2024)**

  * **Total Awards** : 258+ grants to 80+ academic institutions in 30+ US states
  * **Funding Model** : Biennial competitions (odd-numbered years)
  * **Federal Partners** : AFRL, DOE EERE, ONR, NIST, Army GVSC, ARL
  * **Success Rate** : Approximately 15-20% of proposals funded

## 1.3 ARPA-E Programs

The **Advanced Research Projects Agency-Energy (ARPA-E)** funds high-risk, high-reward energy technology projects, including several materials-focused programs that leverage MI approaches.

### 1.3.1 Agency Overview

Attribute | Details  
---|---  
Established | 2009  
Total Investment | $4.21 billion (2009-2024)  
Projects Funded | 1,700+ projects  
Follow-on Investment | $15+ billion in private follow-on funding  
Official Website | <https://arpa-e.energy.gov/>  
  
### 1.3.2 Materials-Focused Programs

#### HESTIA (High-Energy Storage Through Advanced Materials)

  * **Budget** : $39 million
  * **Focus** : Next-generation thermal energy storage materials
  * **MI Component** : Computational screening of phase-change materials

#### MAGNITO (Magnetic Materials for Grid Technologies)

  * **Focus** : Advanced magnetic materials for grid-scale applications
  * **MI Component** : Machine learning for rare-earth-free magnet discovery

#### ULTIMATE (Ultra-Lightweight Technologies for Innovative Materials and Advanced Transportation Efficiency)

  * **Focus** : Lightweight structural materials for transportation
  * **MI Component** : High-throughput computational screening of alloys

### 1.3.3 Impact Metrics

**ARPA-E Success Indicators**

  * 129 new companies formed from ARPA-E projects
  * $15 billion in follow-on private investment
  * 200+ patents filed
  * Multiple technologies transitioned to commercial deployment

### 1.3.4 ARPA-E DIFFERENTIATE Program Projects (2019-2024)

The DIFFERENTIATE (Design Intelligence Fostering Formidable Energy Reduction Through Integrated AI and Experimentation) program specifically funds AI/ML approaches for energy materials discovery:

Year | Project Title | Lead Organization | Budget | Focus  
---|---|---|---|---  
2019| ML-Accelerated Battery Materials Discovery| CMU/Citrine/MIT| $600K| Battery Cathodes  
2019| Neural Network Potentials for Alloy Design| Northwestern/MIT| $750K| Metal-Insulator Transition  
2020| Autonomous Discovery of Solid Ion Conductors| Citrine Informatics| $800K| Solid Electrolytes  
2020| ML-Enhanced Battery Materials Screening| NREL| $650K| Li-ion Batteries  
2021| AI-Driven Fusion Materials Discovery| Savannah River NL| $900K| Radiation-Resistant Alloys  
2021| Generative Models for Solar Cell Design| Stanford/SLAC| $850K| Perovskite PV  
2022| Robotic Synthesis with Active Learning| Argonne NL| $1.0M| Catalyst Discovery  
2023| Foundation Models for Energy Materials| Microsoft Research/PNNL| $1.2M| Cross-domain Transfer  
  
**DIFFERENTIATE Program Impact**

  * **Total Funding** : Up to $20 million across 23 projects
  * **Acceleration Factor** : Target 10-100x faster materials discovery
  * **Industry Transition** : Multiple projects commercialized through startups

## 1.4 Materials Project (LBNL)

The **Materials Project** is one of the most influential open-access databases in materials science, hosted at Lawrence Berkeley National Laboratory (LBNL). It provides computed materials properties using density functional theory (DFT) calculations.

### 1.4.1 Database Overview

Attribute | Details  
---|---  
Launch Year | 2011  
Host Institution | Lawrence Berkeley National Laboratory (LBNL)  
Compounds in Database | 80,000+ inorganic compounds  
Registered Users | 300,000+ worldwide  
Founders | Kristin Persson, Gerbrand Ceder  
Official Website | <https://materialsproject.org/>  
  
### 1.4.2 Available Data and Properties

The Materials Project provides the following computed properties:

  * **Thermodynamic Properties** : Formation energy, stability (energy above hull), decomposition products
  * **Electronic Properties** : Band gap, band structure, density of states
  * **Mechanical Properties** : Elastic constants, bulk modulus, shear modulus
  * **Magnetic Properties** : Magnetic ordering, magnetic moments
  * **Structural Properties** : Crystal structure, space group, lattice parameters
  * **Phonon Properties** : Phonon band structure, thermal properties

### 1.4.3 API and Tools

**Code Example: Accessing Materials Project API**
    
    
    # Requirements:
    # - Python 3.9+
    # - mp-api>=0.30.0
    
    """
    Example: Querying Materials Project for battery materials
    
    Purpose: Demonstrate Materials Project API usage
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: mp-api
    """
    
    from mp_api.client import MPRester
    
    # Initialize with your API key (get from materialsproject.org)
    with MPRester("YOUR_API_KEY") as mpr:
        # Search for lithium-containing oxides with band gap > 2 eV
        results = mpr.summary.search(
            elements=["Li", "O"],
            band_gap=(2, None),  # Band gap greater than 2 eV
            fields=["material_id", "formula_pretty", "band_gap", "energy_above_hull"]
        )
    
        print(f"Found {len(results)} materials")
    
        # Display top 5 results
        for doc in results[:5]:
            print(f"ID: {doc.material_id}")
            print(f"  Formula: {doc.formula_pretty}")
            print(f"  Band Gap: {doc.band_gap:.2f} eV")
            print(f"  Energy Above Hull: {doc.energy_above_hull:.4f} eV/atom")
            print()
    
    # Example output:
    # Found 2847 materials
    # ID: mp-1234
    #   Formula: Li2O
    #   Band Gap: 5.02 eV
    #   Energy Above Hull: 0.0000 eV/atom
    

### 1.4.4 Impact on the Field

  * **Citation Impact** : Original paper cited 10,000+ times
  * **Discovery Examples** : Used to discover new battery cathode materials, thermoelectrics, and photocatalysts
  * **Industry Adoption** : Used by major companies including Toyota, Samsung, and BASF
  * **Educational Use** : Integrated into university curricula worldwide

## 1.5 MICCoM (DOE)

The **Midwest Integrated Center for Computational Materials (MICCoM)** is a DOE-funded center focused on developing and applying computational methods for materials discovery.

### 1.5.1 Center Overview

Attribute | Details  
---|---  
Full Name | Midwest Integrated Center for Computational Materials  
Annual Budget | $3 million/year  
Headquarters | Argonne National Laboratory  
Partner Institutions | University of Chicago, Northwestern University, University of Notre Dame  
  
### 1.5.2 Research Focus Areas

  * **Solar Energy Materials** : Perovskites, organic photovoltaics, tandem cells
  * **Battery Materials** : Solid-state electrolytes, high-capacity cathodes
  * **Thermoelectric Materials** : High-efficiency thermoelectrics for waste heat recovery
  * **Method Development** : Beyond-DFT methods, machine learning potentials

### 1.5.3 Computational Capabilities

MICCoM leverages DOE supercomputing resources including:

  * **Aurora** : Exascale supercomputer at Argonne (2+ exaflops)
  * **Polaris** : GPU-accelerated system for AI/ML workloads
  * **ALCF** : Argonne Leadership Computing Facility resources

## 1.6 ORNL Autonomous Chemistry Lab

Oak Ridge National Laboratory (ORNL) operates an **Autonomous Chemistry Laboratory** that combines AI-driven planning with robotic synthesis for accelerated materials discovery.

### 1.6.1 Facility Overview

Attribute | Details  
---|---  
Location | Oak Ridge National Laboratory, Tennessee  
Focus | AI-driven autonomous materials discovery  
Capabilities | Robotics + AI synthesis planning + automated characterization  
  
### 1.6.2 Key Capabilities

  * **Robotic Synthesis** : Automated preparation and processing of materials
  * **AI Planning** : Machine learning algorithms for experiment design
  * **Real-time Characterization** : In-situ monitoring of synthesis outcomes
  * **Closed-loop Optimization** : Iterative refinement based on experimental feedback

### 1.6.3 Notable Achievements

**ORNL Autonomous Lab Accomplishments**

  * Demonstrated 10x acceleration in materials synthesis optimization
  * Integrated neutron scattering characterization for real-time structure analysis
  * Published methodology for autonomous catalyst discovery

## 1.7 NIST Materials Data Infrastructure

The **National Institute of Standards and Technology (NIST)** plays a crucial role in MGI by developing standards, reference data, and infrastructure for materials data.

### 1.7.1 NIST MGI Role

Attribute | Details  
---|---  
Role | MGI leadership and coordination  
Focus Areas | Data standards, reference databases, measurement science  
Key Outputs | Materials Resource Registry, CALPHAD databases  
  
### 1.7.2 Key Infrastructure Projects

#### Materials Resource Registry (MRR)

  * Catalog of materials data resources
  * Standardized metadata for data discovery
  * Links to distributed data repositories

#### CALPHAD Databases

  * Thermodynamic data for phase diagram calculations
  * Critically assessed experimental data
  * Integration with computational tools (Thermo-Calc, PANDAT)

#### Materials Data Curation System (MDCS)

  * Tools for standardized data entry and validation
  * XML schema for materials data
  * Support for FAIR data principles

## 1.8 Comparison of US MI Projects

The following table provides a comprehensive comparison of major US Materials Informatics initiatives:

Project | Lead Agency | Budget | Focus | Primary Output | Status  
---|---|---|---|---|---  
MGI | OSTP (Multi-agency) | $1B+ cumulative | Overall coordination | Policy, infrastructure | Active  
DMREF | NSF | $72.5M/year | Fundamental research | Academic publications | Active  
ARPA-E | DOE | $4.21B total | High-risk energy tech | Startups, patents | Active  
Materials Project | DOE (LBNL) | ~$5M/year | Open database | 80,000+ compounds | Active  
MICCoM | DOE (ANL) | $3M/year | Computational methods | Software, methods | Active  
ORNL Auto Lab | DOE (ORNL) | ~$10M/year | Autonomous discovery | Automated synthesis | Active  
NIST MRR | NIST | ~$5M/year | Data infrastructure | Standards, registries | Active  
  
## 1.9 Timeline of US MI Development

The following diagram illustrates the chronological development of Materials Informatics in the United States:
    
    
    ```mermaid
    timeline
        title US Materials Informatics Timeline
        section 2009-2011
            2009 : ARPA-E Established
            2011 : MGI Launched by White House
                 : Materials Project Goes Public
        section 2012-2015
            2012 : DMREF Program Begins
            2014 : First MGI Strategic Plan
            2015 : MICCoM Established
        section 2016-2020
            2017 : Second MGI Strategic Plan
            2019 : Materials Project 100K users
            2020 : COVID-19 accelerates digital methods
        section 2021-Present
            2021 : Third MGI Strategic Plan
                 : ORNL Autonomous Lab Operational
            2022 : Materials Project 150K+ compounds
            2023 : AI/ML integration accelerates
            2024 : Exascale computing for materials
    ```

## 1.10 Chapter Summary

The United States has established a comprehensive ecosystem for Materials Informatics through coordinated government initiatives, national laboratory programs, and academic research. Key takeaways include:

**Key Findings**

  * **MGI Leadership** : The Materials Genome Initiative has provided strategic direction and over $1 billion in cumulative funding since 2011
  * **Multi-Agency Coordination** : NSF, DOE, NIST, and other agencies coordinate complementary programs
  * **Open Data Culture** : The Materials Project exemplifies the US commitment to open science with 80,000+ compounds freely available
  * **Emerging Capabilities** : Autonomous laboratories and exascale computing are accelerating discovery
  * **Industry Impact** : ARPA-E's $15 billion in follow-on investment demonstrates commercial viability

### Practical Recommendations

  * **For Researchers** : Leverage Materials Project and DMREF funding opportunities for data-driven research
  * **For Industry** : Engage with ARPA-E programs for high-risk technology development
  * **For Students** : Build skills in both computation and experiment to align with MGI goals
  * **For International Collaborators** : Partner with US institutions through established exchange programs

## Exercises

Exercise 1: MGI Agency Roles Easy

**Problem:** Match each federal agency with its primary role in the Materials Genome Initiative:

  1. NSF
  2. DOE
  3. NIST
  4. DARPA

Roles: (A) Standards and data infrastructure, (B) Fundamental research and education, (C) Energy materials and computing, (D) High-risk defense research

**Solution:**

  1. NSF - (B) Fundamental research and education
  2. DOE - (C) Energy materials and computing
  3. NIST - (A) Standards and data infrastructure
  4. DARPA - (D) High-risk defense research

Exercise 2: Materials Project Query Medium

**Problem:** Write a Materials Project API query to find all stable (energy above hull = 0) compounds containing both Fe and O with a band gap between 1.5 and 3.0 eV. How many materials match these criteria?

**Solution:**
    
    
    from mp_api.client import MPRester
    
    with MPRester("YOUR_API_KEY") as mpr:
        results = mpr.summary.search(
            elements=["Fe", "O"],
            band_gap=(1.5, 3.0),
            energy_above_hull=(0, 0.001),  # Essentially stable
            fields=["material_id", "formula_pretty", "band_gap"]
        )
    
        print(f"Found {len(results)} stable Fe-O compounds with band gap 1.5-3.0 eV")
        for doc in results[:10]:
            print(f"  {doc.formula_pretty}: {doc.band_gap:.2f} eV")
    
    # Typical result: ~50-100 materials depending on database version
    

Exercise 3: Funding Analysis Medium

**Problem:** Calculate the approximate annual funding rate of the MGI since its launch in 2011. Compare this to DMREF's annual budget. What does this comparison tell you about the relative priorities of different funding mechanisms?

**Solution:**

**Calculation:**

  * MGI cumulative: $1 billion over 13 years (2011-2024)
  * MGI annual average: $1B / 13 = ~$77 million/year
  * DMREF FY2024-25: $72.5 million/year

**Analysis:**

  * DMREF alone accounts for approximately 94% of MGI's annual average investment
  * This indicates that fundamental academic research (NSF) is the primary funding mechanism
  * Other agencies contribute through complementary programs (DOE computing, NIST standards)
  * The comparable scale suggests strong prioritization of bottom-up research over top-down coordination

Exercise 4: Program Selection Hard

**Problem:** You are a materials scientist at a startup developing new solid-state battery electrolytes. Which US funding program(s) would be most appropriate for your research, and why? Consider: (1) Technology readiness level, (2) Funding size, (3) Timeline, and (4) IP considerations.

**Solution:**

**Recommended Programs:**

  1. **Primary: ARPA-E**
     * Best fit for startup with transformational energy technology
     * Funding size: $1-10M typical, suitable for R&D scale-up
     * Flexible IP terms (startup retains ownership)
     * 3-year timeline matches startup development cycles
  2. **Secondary: DOE SBIR/STTR**
     * Phase I: $200K for proof-of-concept (6 months)
     * Phase II: $1.5M for development (2 years)
     * Lower barrier to entry than ARPA-E
  3. **Complementary: MICCoM Collaboration**
     * Access to computational resources for electrolyte screening
     * Partnership rather than direct funding
     * Leverage existing DFT databases for solid electrolytes

**Not Recommended:**

  * DMREF: Targets academic institutions, 4-year timeline too slow for startup
  * NIST: Standards focus, not commercialization

## References

  1. National Science and Technology Council. (2021). Materials Genome Initiative Strategic Plan 2021. White House Office of Science and Technology Policy.
  2. Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., Cholia, S., Gunter, D., Skinner, D., Ceder, G., Persson, K. A. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. _APL Materials_ , 1(1), 011002.
  3. National Science Foundation. (2024). DMREF: Designing Materials to Revolutionize and Engineer our Future. NSF Program Solicitation 24-529.
  4. ARPA-E. (2024). ARPA-E Impact Report 2024. U.S. Department of Energy.
  5. Curtarolo, S., Hart, G. L., Nardelli, M. B., Mingo, N., Sanvito, S., Levy, O. (2013). The high-throughput highway to computational materials design. _Nature Materials_ , 12(3), 191-201.
  6. de Pablo, J. J., Jackson, N. E., Webb, M. A., Chen, L. Q., Moore, J. E., Morgan, D., Jacobs, R., Pollock, T., Schlom, D. G., Toberer, E. S., Analytis, J., Dabo, I., DeLongchamp, D. M., Fiez, G. A., Grason, G. M., Hautier, G., Mo, Y., Rajan, K., Reed, E. J., ... Ward, L. (2019). New frontiers for the materials genome initiative. _npj Computational Materials_ , 5(1), 41.

[Back to Series Contents](<index.html>) [Chapter 2: European Union MI Projects](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
